//! Qwen3.5 language model, used as MOSS-Image's text encoder. Ported from
//! transformers' `modeling_qwen3_5` (`Qwen3_5TextModel`).
//!
//! Three of every four layers are Gated DeltaNet linear attention (see
//! [`gated_delta`]); every fourth is full attention with an output gate.
//! Returns `last_hidden_state`, after the final norm, which is what the
//! MOSS-Image transformer reads.

use super::gated_delta;
use crate::nn::{linear, rotate_half, silu, Quantize, WeightCache, Weights};
use anyhow::{Context, Result};
use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::nn::softplus;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{broadcast_to, concatenate, pad, sigmoid, split_at_indices};
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::Path;

const NUM_LAYERS: usize = 32;
/// Layers `3, 7, ..., 31` are full attention, the rest linear attention.
const FULL_ATTENTION_EVERY: usize = 4;
const HEADS: i32 = 16;
const KV_HEADS: i32 = 4;
const HEAD_DIM: i32 = 256;
/// Only the first quarter of each attention head is rotated.
const ROTARY_DIM: i32 = 64;
const ROPE_THETA: f64 = 10_000_000.0;
const LINEAR_K_HEADS: i32 = 16;
const LINEAR_V_HEADS: i32 = 32;
/// Key and value head size in the linear-attention layers.
const LINEAR_HEAD_DIM: i32 = 128;
const LINEAR_KEY_DIM: i32 = LINEAR_K_HEADS * LINEAR_HEAD_DIM;
const CONV_KERNEL: i32 = 4;
const EPS: f32 = 1e-6;

fn is_full_attention(layer: usize) -> bool {
    layer % FULL_ATTENTION_EVERY == FULL_ATTENTION_EVERY - 1
}

pub struct TextEncoder {
    w: Weights,
    dtype: Dtype,
    /// `1 + weight` in f32 for each zero-centered RMSNorm, by prefix.
    norms: HashMap<String, Array>,
    /// Each linear-attention layer's conv taps, (4, 8192) in f32; the last
    /// tap weighs the current token.
    conv_taps: HashMap<usize, Array>,
}

impl TextEncoder {
    /// Loads the language model half of the Qwen3.5 checkpoint. With
    /// `quantize`, the attention and MLP projections are quantized, except the
    /// two tiny ones that set each linear-attention head's decay and write
    /// strength (`in_proj_a`, `in_proj_b`).
    pub fn load(
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        quantize: Option<Quantize>,
        cache: Option<&WeightCache>,
    ) -> Result<Self> {
        let w = Weights::load_maybe_cached(
            files,
            dtype,
            |n| n.starts_with("language_model."),
            quantize,
            |layer| {
                layer.starts_with("language_model.layers.")
                    && !layer.ends_with(".in_proj_a")
                    && !layer.ends_with(".in_proj_b")
            },
            cache,
            "moss-text-encoder",
        )?;
        let mut norms = HashMap::new();
        let mut conv_taps = HashMap::new();
        let mut zero_centered = |prefix: String| -> Result<()> {
            let one_plus = w
                .get(&format!("{prefix}.weight"))?
                .as_dtype(Dtype::Float32)?
                .add(Array::from_f32(1.0))?;
            one_plus.eval()?;
            norms.insert(prefix, one_plus);
            Ok(())
        };
        for i in 0..NUM_LAYERS {
            let p = format!("language_model.layers.{i}");
            zero_centered(format!("{p}.input_layernorm"))?;
            zero_centered(format!("{p}.post_attention_layernorm"))?;
            if is_full_attention(i) {
                zero_centered(format!("{p}.self_attn.q_norm"))?;
                zero_centered(format!("{p}.self_attn.k_norm"))?;
            } else {
                // PyTorch's depthwise layout (C, 1, K) -> taps (K, C).
                let name = format!("{p}.linear_attn.conv1d.weight");
                let weight = w.get(&name)?;
                anyhow::ensure!(
                    weight.shape()
                        == [
                            2 * LINEAR_KEY_DIM + LINEAR_V_HEADS * LINEAR_HEAD_DIM,
                            1,
                            CONV_KERNEL
                        ],
                    "{name} has shape {:?}",
                    weight.shape()
                );
                let taps = weight
                    .reshape(&[weight.shape()[0], CONV_KERNEL])?
                    .t()
                    .as_dtype(Dtype::Float32)?;
                taps.eval()?;
                conv_taps.insert(i, taps);
            }
        }
        zero_centered("language_model.norm".into())?;
        Ok(Self {
            w,
            dtype,
            norms,
            conv_taps,
        })
    }

    /// Runs the model over `ids`. Returns (1, L, 4096).
    pub fn forward(&self, ids: &[u32]) -> Result<Array> {
        let last = self.run(ids, &mut |_, _| {})?;
        self.norm(&last, "language_model.norm")
    }

    /// Each decoder layer's output (before the final norm), for the parity
    /// check.
    pub fn layer_outputs(&self, ids: &[u32]) -> Result<Vec<Array>> {
        let mut out = Vec::with_capacity(NUM_LAYERS);
        self.run(ids, &mut |_, h| out.push(h.clone()))?;
        Ok(out)
    }

    fn run(&self, ids: &[u32], on_layer: &mut dyn FnMut(usize, &Array)) -> Result<Array> {
        anyhow::ensure!(!ids.is_empty(), "no prompt tokens");
        let len = ids.len() as i32;
        let idx: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let mut h = self
            .w
            .get("language_model.embed_tokens.weight")?
            .take_axis(Array::from_slice(&idx, &[len]), 0)?
            .expand_dims(0)?; // (1, L, 4096)
        let (cos, sin) = rope_tables(len, self.dtype)?;
        for i in 0..NUM_LAYERS {
            h = self.layer(i, &h, &cos, &sin)?;
            on_layer(i, &h);
        }
        Ok(h)
    }

    fn layer(&self, i: usize, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let p = format!("language_model.layers.{i}");
        let h = self.norm(x, &format!("{p}.input_layernorm"))?;
        let mixed = if is_full_attention(i) {
            self.attention(&p, &h, cos, sin)?
        } else {
            self.linear_attention(i, &p, &h)?
        };
        let x = x.add(mixed)?;
        let h = self.norm(&x, &format!("{p}.post_attention_layernorm"))?;
        let gate = silu(&linear(&h, &self.w, &format!("{p}.mlp.gate_proj"))?)?;
        let up = linear(&h, &self.w, &format!("{p}.mlp.up_proj"))?;
        Ok(x.add(linear(
            &gate.multiply(up)?,
            &self.w,
            &format!("{p}.mlp.down_proj"),
        )?)?)
    }

    /// Zero-centered RMSNorm, `x * rsqrt(mean(x^2) + eps) * (1 + w)`, all in
    /// f32 and rounded once, like transformers.
    fn norm(&self, x: &Array, prefix: &str) -> Result<Array> {
        let w = self
            .norms
            .get(prefix)
            .with_context(|| format!("no norm {prefix}"))?;
        Ok(
            mlx_rs::fast::rms_norm(x.as_dtype(Dtype::Float32)?, Some(w), EPS)?
                .as_dtype(x.dtype())?,
        )
    }

    /// Full attention with a sigmoid output gate. `q_proj` holds each head's
    /// query and gate side by side.
    fn attention(&self, p: &str, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let p = format!("{p}.self_attn");
        let proj = |name: &str| linear(x, &self.w, &format!("{p}.{name}"));
        let [q, gate]: [Array; 2] = proj("q_proj")?
            .reshape(&[b, l, HEADS, 2 * HEAD_DIM])?
            .split_equal(2, -1)?
            .try_into()
            .expect("2 chunks");
        let gate = gate.reshape(&[b, l, HEADS * HEAD_DIM])?;
        let to_bhld = |a: Array| a.transpose_axes(&[0, 2, 1, 3]);
        let q = to_bhld(self.norm(&q, &format!("{p}.q_norm"))?)?;
        let k = proj("k_proj")?.reshape(&[b, l, KV_HEADS, HEAD_DIM])?;
        let k = to_bhld(self.norm(&k, &format!("{p}.k_norm"))?)?;
        let v = to_bhld(proj("v_proj")?.reshape(&[b, l, KV_HEADS, HEAD_DIM])?)?;
        let q = apply_partial_rope(&q, cos, sin)?;
        let k = apply_partial_rope(&k, cos, sin)?;
        let o = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0 / (HEAD_DIM as f32).sqrt(),
            ScaledDotProductAttentionMask::Causal,
            None,
        )?;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, HEADS * HEAD_DIM])?
            .multiply(sigmoid(&gate)?)?;
        linear(&o, &self.w, &format!("{p}.o_proj"))
    }

    /// Gated DeltaNet: a causal conv over q|k|v, the gated delta rule, then a
    /// per-head RMSNorm gated by `z`.
    fn linear_attention(&self, i: usize, p: &str, x: &Array) -> Result<Array> {
        let l = x.shape()[1];
        let p = format!("{p}.linear_attn");
        let proj = |name: &str| linear(x, &self.w, &format!("{p}.{name}"));
        let qkv = self.causal_conv(i, &proj("in_proj_qkv")?)?;
        let [q, k, v]: [Array; 3] =
            split_at_indices(&qkv, &[LINEAR_KEY_DIM, 2 * LINEAR_KEY_DIM], -1)?
                .try_into()
                .expect("3 parts");
        let heads = |a: &Array, n: i32| a.reshape(&[l, n, LINEAR_HEAD_DIM]);
        // Value head h reads key head h / 2 (`repeat_interleave`).
        let repeat = |a: Array| -> Result<Array> {
            let rep = LINEAR_V_HEADS / LINEAR_K_HEADS;
            Ok(broadcast_to(
                a.expand_dims(2)?,
                &[l, LINEAR_K_HEADS, rep, LINEAR_HEAD_DIM],
            )?
            .reshape(&[l, LINEAR_V_HEADS, LINEAR_HEAD_DIM])?)
        };
        let q = repeat(heads(&q, LINEAR_K_HEADS)?)?;
        let k = repeat(heads(&k, LINEAR_K_HEADS)?)?;
        let v = heads(&v, LINEAR_V_HEADS)?;

        // Write strength in the model dtype, log decay in f32.
        let beta = sigmoid(&proj("in_proj_b")?)?.reshape(&[l, LINEAR_V_HEADS])?;
        let a = proj("in_proj_a")?
            .reshape(&[l, LINEAR_V_HEADS])?
            .as_dtype(Dtype::Float32)?;
        let f32_param = |name: &str| -> Result<Array> {
            Ok(self
                .w
                .get(&format!("{p}.{name}"))?
                .as_dtype(Dtype::Float32)?)
        };
        let g = f32_param("A_log")?
            .exp()?
            .negative()?
            .multiply(softplus(a.add(f32_param("dt_bias")?)?)?)?;

        let o = gated_delta::chunked(&q, &k, &v, &g, &beta)?.as_dtype(self.dtype)?;

        // Norm, weight, then the gate, with the reference's roundings.
        let z = proj("in_proj_z")?.reshape(&[l, LINEAR_V_HEADS, LINEAR_HEAD_DIM])?;
        let o = mlx_rs::fast::rms_norm(o.as_dtype(Dtype::Float32)?, None, EPS)?
            .as_dtype(self.dtype)?
            .multiply(self.w.get(&format!("{p}.norm.weight"))?)?
            .as_dtype(Dtype::Float32)?
            .multiply(silu(&z.as_dtype(Dtype::Float32)?)?)?
            .as_dtype(self.dtype)?;
        linear(
            &o.reshape(&[1, l, LINEAR_V_HEADS * LINEAR_HEAD_DIM])?,
            &self.w,
            &format!("{p}.out_proj"),
        )
    }

    /// Depthwise causal conv (kernel 4, no bias) along the sequence, then
    /// silu. Summed in f32 and rounded once, like a bf16 convolution.
    fn causal_conv(&self, i: usize, x: &Array) -> Result<Array> {
        let taps = &self.conv_taps[&i];
        let l = x.shape()[1];
        let padded = pad(
            &x.as_dtype(Dtype::Float32)?,
            &[(0, 0), (CONV_KERNEL - 1, 0), (0, 0)],
            None,
            None,
        )?;
        let mut sum = padded.index((.., 0..l, ..)).multiply(taps.index(0))?;
        for t in 1..CONV_KERNEL {
            sum = sum.add(padded.index((.., t..t + l, ..)).multiply(taps.index(t))?)?;
        }
        silu(&sum.as_dtype(self.dtype)?)
    }
}

/// cos/sin tables (1, 1, L, 64) for positions `0..len`, computed in f32 and
/// cast to `dtype` like transformers. A text-only prompt has the same
/// position on all three M-RoPE axes, so it is plain RoPE.
fn rope_tables(len: i32, dtype: Dtype) -> Result<(Array, Array)> {
    let half = (ROTARY_DIM / 2) as usize;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f64 / ROTARY_DIM as f64) as f32)
        .collect();
    let mut cos = Vec::with_capacity(len as usize * 2 * half);
    let mut sin = Vec::with_capacity(len as usize * 2 * half);
    for pos in 0..len {
        for _ in 0..2 {
            for f in &inv_freq {
                let (s, c) = (pos as f32 * f).sin_cos();
                cos.push(c);
                sin.push(s);
            }
        }
    }
    let shape = [1, 1, len, ROTARY_DIM];
    Ok((
        Array::from_slice(&cos, &shape).as_dtype(dtype)?,
        Array::from_slice(&sin, &shape).as_dtype(dtype)?,
    ))
}

/// RoPE on the first [`ROTARY_DIM`] dims of (B, H, L, D); the rest pass.
fn apply_partial_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let [rot, pass]: [Array; 2] = split_at_indices(x, &[ROTARY_DIM], -1)?
        .try_into()
        .expect("2 parts");
    let rot = rot.multiply(cos)?.add(rotate_half(&rot)?.multiply(sin)?)?;
    Ok(concatenate(&[rot, pass], -1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fourth_layer_is_full_attention() {
        let full: Vec<usize> = (0..NUM_LAYERS).filter(|&i| is_full_attention(i)).collect();
        assert_eq!(full, [3, 7, 11, 15, 19, 23, 27, 31]);
    }

    #[test]
    fn partial_rope_rotates_only_the_first_quarter() {
        let _mlx = crate::nn::test_device();
        let x = mlx_rs::random::normal::<f32>(
            &[1, 2, 3, HEAD_DIM][..],
            None,
            None,
            &mlx_rs::random::key(0).unwrap(),
        )
        .unwrap();
        let (cos, sin) = rope_tables(3, Dtype::Float32).unwrap();
        let y = apply_partial_rope(&x, &cos, &sin).unwrap();
        // Position 0 is not rotated; dims past the rotary ones never are.
        let diff = |a: Array, b: Array| {
            a.subtract(b)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item_exact::<f32>()
        };
        assert_eq!(
            diff(y.index((.., .., 0, ..)), x.index((.., .., 0, ..))),
            0.0
        );
        assert_eq!(
            diff(
                y.index((.., .., .., ROTARY_DIM..)),
                x.index((.., .., .., ROTARY_DIM..))
            ),
            0.0
        );
        assert!(
            diff(
                y.index((.., .., 1, ..ROTARY_DIM)),
                x.index((.., .., 1, ..ROTARY_DIM))
            ) > 0.01
        );
    }
}

//! MOSS-Image DiT (`MOSSImageTransformer`), ported from the
//! `pipeline_moss_image.py` bundled with the weights.
//!
//! A single-stream transformer: the image tokens and the prompt's (refined by
//! two text blocks) attend jointly through 42 blocks that share one
//! modulation, computed from the timestep alone. Latents are tokens
//! (1, h * w, 128), row-major, as the reference flattens them.

use super::fast::Adapter;
use crate::nn::{linear, scalar, silu, Quantize, WeightCache, Weights};
use anyhow::{Context, Result};
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{concatenate, sigmoid, split_at_indices, tanh};
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::Path;

const DIM: i32 = 3072;
const HEADS: i32 = 24;
const HEAD_DIM: i32 = DIM / HEADS;
pub const NUM_LAYERS: usize = 42;
const NUM_TEXT_BLOCKS: usize = 2;
const EPS: f32 = 1e-6;
const FREQ_DIM: usize = 256;
const ROPE_BASE: f64 = 10000.0;
/// RoPE frequencies per position axis: 4 axes (unused, text index, row,
/// column) of `HEAD_DIM / 8` bands, sin and cos.
const ROPE_BANDS: usize = HEAD_DIM as usize / 8;

pub struct Transformer {
    w: Weights,
    dtype: Dtype,
    /// With the 4-step adapter: the time embedding and both modulations'
    /// weights and biases in f32 (see [`super::fast`]).
    time_f32: Option<HashMap<String, Array>>,
}

/// A prompt prepared for [`Transformer::forward`] at one image size: its
/// tokens after the text blocks, and the joint sequence's RoPE tables.
pub struct Conditioning {
    text: Array,
    /// (1, h * w + text, 1, 128) each, in the model dtype.
    cos: Array,
    sin: Array,
}

impl Transformer {
    /// With `quantize`, the projections of every block and text block are
    /// quantized (~95% of the weights). The embedders, the modulations and
    /// the output layer are small and stay unquantized.
    pub fn load(
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        quantize: Option<Quantize>,
        cache: Option<&WeightCache>,
    ) -> Result<Self> {
        let w = Weights::load_maybe_cached(
            files,
            dtype,
            // The RoPE bands are recomputed in f32, as the reference does.
            |n| n != "rope.bands",
            quantize,
            |layer| layer.starts_with("layers.") || layer.starts_with("text_embedder.blocks."),
            cache,
            "moss-transformer",
        )?;
        Ok(Self {
            w,
            dtype,
            time_f32: None,
        })
    }

    /// Applies the 4-step adapter: its low-rank updates, and its f32 time
    /// path (the base modulation weights, upcast, with the adapter's biases).
    pub fn apply_adapter(&mut self, adapter: &Adapter) -> Result<()> {
        for u in adapter.lora.updates(self.dtype)? {
            self.w
                .add_lora(&u.layer, u.down, u.up)
                .with_context(|| format!("adapter update to {}", u.layer))?;
        }
        let mut time = adapter.time.clone();
        for name in ["shared_block_modulation", "video_head_modulation"] {
            let weight = format!("{name}.weight");
            let w = self.w.get(&weight)?.as_dtype(Dtype::Float32)?;
            w.eval()?;
            time.insert(weight, w);
        }
        self.time_f32 = Some(time);
        Ok(())
    }

    /// Adds a low-rank update to the linear layer `layer` (see
    /// [`Weights::add_lora`]).
    pub fn add_lora(&mut self, layer: &str, down: Array, up: Array) -> Result<()> {
        self.w.add_lora(layer, down, up)
    }

    /// Runs the text blocks over prompt embeddings `text` (1, S, 4096) and
    /// builds the RoPE tables for an `h` x `w` latent grid. The result serves
    /// every step for that prompt and size.
    pub fn condition(&self, text: &Array, h: i32, w: i32) -> Result<Conditioning> {
        let s = text.shape()[1];
        let (cos, sin) = rope_tables(h, w, s, self.dtype)?;
        let n_img = h * w;
        let text_rope = |t: &Array| -> Result<Array> {
            Ok(split_at_indices(t, &[n_img], 1)?.pop().expect("text part"))
        };
        let (text_cos, text_sin) = (text_rope(&cos)?, text_rope(&sin)?);
        let mut x = linear(&text.as_dtype(self.dtype)?, &self.w, "text_embedder.proj")?;
        for i in 0..NUM_TEXT_BLOCKS {
            x = self.text_block(
                &format!("text_embedder.blocks.{i}"),
                &x,
                &text_cos,
                &text_sin,
            )?;
        }
        Ok(Conditioning { text: x, cos, sin })
    }

    /// Predicts the flow velocity for latent tokens `x` (1, h * w, 128) at
    /// time `t` (sigma * 1000), leaving out block `skip` if given (the
    /// guidance's negative branch). Returns (1, h * w, 128) in the model dtype.
    pub fn forward(
        &self,
        x: &Array,
        t: f32,
        cond: &Conditioning,
        skip: Option<usize>,
    ) -> Result<Array> {
        let n_img = x.shape()[1];
        let (modulation, head) = self.modulation(t)?;
        let video = linear(&x.as_dtype(self.dtype)?, &self.w, "video_embedder")?;
        let mut u = concatenate(&[&video, &cond.text], 1)?;
        for i in (0..NUM_LAYERS).filter(|&i| Some(i) != skip) {
            u = self.block(
                &format!("layers.{i}"),
                &u,
                &cond.cos,
                &cond.sin,
                &modulation,
            )?;
        }
        // The image tokens, in the model dtype (the adapter's f32 residual
        // stream is rounded here, like the reference).
        let x = split_at_indices(&u, &[n_img], 1)?
            .swap_remove(0)
            .as_dtype(self.dtype)?;
        let [shift, scale]: [Array; 2] = head.try_into().expect("2 chunks");
        let x = self
            .norm(&x, "final_norm_video")?
            .multiply(scale.add(scalar(1.0, scale.dtype())?)?)?
            .add(&shift)?;
        linear(&x.as_dtype(self.dtype)?, &self.w, "final_linear_video")
    }

    /// The shared block modulation (6 vectors of (1, 1, DIM): attention shift,
    /// scale, gate, then the same for the feed-forward) and the output layer's
    /// (shift, scale), from the timestep `t`.
    fn modulation(&self, t: f32) -> Result<(Vec<Array>, Vec<Array>)> {
        // Like the reference: the sinusoid in f64, the embedding in f32, the
        // layers in the model dtype, or all in f32 with the adapter.
        let half = FREQ_DIM / 2;
        let t = t as f64;
        let mut emb = vec![0f32; FREQ_DIM];
        for i in 0..half {
            let phase = t * 10000f64.powf(-(i as f64) / half as f64);
            emb[i] = phase.cos() as f32;
            emb[half + i] = phase.sin() as f32;
        }
        let emb = Array::from_slice(&emb, &[1, FREQ_DIM as i32]);
        let layer = |x: &Array, name: &str| -> Result<Array> {
            match &self.time_f32 {
                Some(f) => {
                    let get = |n: &str| f.get(&format!("{name}.{n}")).context("adapter time path");
                    Ok(x.matmul(get("weight")?.t())?.add(get("bias")?)?)
                }
                None => linear(x, &self.w, name),
            }
        };
        let emb = match self.time_f32 {
            Some(_) => emb,
            None => emb.as_dtype(self.dtype)?,
        };
        let time = layer(
            &silu(&layer(&emb, "time_embedding.0")?)?,
            "time_embedding.2",
        )?;
        let chunks = |name: &str, n: i32| -> Result<Vec<Array>> {
            Ok(layer(&time, name)?
                .reshape(&[1, n, 1, DIM])?
                .split_equal(n, 1)?
                .into_iter()
                .map(|c| c.reshape(&[1, 1, DIM]))
                .collect::<std::result::Result<_, _>>()?)
        };
        Ok((
            chunks("shared_block_modulation", 6)?,
            chunks("video_head_modulation", 2)?,
        ))
    }

    fn block(&self, p: &str, x: &Array, cos: &Array, sin: &Array, m: &[Array]) -> Result<Array> {
        let (shift_a, scale_a, gate_a) = (&m[0], &m[1], &m[2]);
        let (shift_f, scale_f, gate_f) = (&m[3], &m[4], &m[5]);
        let modulate = |x: &Array, norm: &str, shift: &Array, scale: &Array| -> Result<Array> {
            let one = scalar(1.0, scale.dtype())?;
            Ok(self
                .norm(x, &format!("{p}.{norm}"))?
                .multiply(scale.add(one)?)?
                .add(shift)?
                .as_dtype(self.dtype)?)
        };
        let y = modulate(x, "norm1", shift_a, scale_a)?;
        let proj = |name: &str, y: &Array| linear(y, &self.w, &format!("{p}.proj.{name}"));
        let q = self.head_norm(&proj("q", &y)?, &format!("{p}.q_norm"))?;
        let k = self.head_norm(&proj("k", &y)?, &format!("{p}.k_norm"))?;
        let a = attention(&q, &k, &proj("v", &y)?, cos, sin)?;
        let x = x.add(proj("o", &a)?.multiply(gate_a)?)?;
        let y = modulate(&x, "norm2", shift_f, scale_f)?;
        let [hidden, gate]: [Array; 2] = proj("ffn_up_gate", &y)?
            .split_equal(2, -1)?
            .try_into()
            .expect("2 chunks");
        let f = proj("ffn_down", &situ(&hidden, &gate)?)?;
        Ok(x.add(f.multiply(gate_f)?)?)
    }

    fn text_block(&self, p: &str, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let y = self.norm(x, &format!("{p}.norm1"))?;
        let [q, k, v]: [Array; 3] = linear(&y, &self.w, &format!("{p}.qkv"))?
            .split_equal(3, -1)?
            .try_into()
            .expect("3 chunks");
        let q = self.head_norm(&q, &format!("{p}.q_norm"))?;
        let k = self.head_norm(&k, &format!("{p}.k_norm"))?;
        let a = attention(&q, &k, &v, cos, sin)?;
        let x = x.add(linear(&a, &self.w, &format!("{p}.proj"))?)?;
        let y = self.norm(&x, &format!("{p}.norm2"))?;
        let gate = linear(&y, &self.w, &format!("{p}.ffn_gate"))?;
        let up = linear(&y, &self.w, &format!("{p}.ffn_up"))?;
        Ok(x.add(linear(
            &situ(&up, &gate)?,
            &self.w,
            &format!("{p}.ffn_down"),
        )?)?)
    }

    /// The reference's RMSNorm: normalized in f32, rounded to the input's
    /// dtype, then scaled by the (model dtype) weight.
    fn norm(&self, x: &Array, prefix: &str) -> Result<Array> {
        let normed = mlx_rs::fast::rms_norm(x, None, EPS)?;
        Ok(normed.multiply(self.w.get(&format!("{prefix}.weight"))?)?)
    }

    /// [`Self::norm`] over each head of (1, S, DIM).
    fn head_norm(&self, x: &Array, prefix: &str) -> Result<Array> {
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let heads = x.reshape(&[b, s, HEADS, HEAD_DIM])?;
        Ok(self.norm(&heads, prefix)?.reshape(&[b, s, DIM])?)
    }
}

/// Scaled dot-product attention of (1, S, DIM) q, k, v with RoPE on q and k.
fn attention(q: &Array, k: &Array, v: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let (b, s) = (q.shape()[0], q.shape()[1]);
    let heads = |x: &Array| x.reshape(&[b, s, HEADS, HEAD_DIM]);
    let to_bhsd = |x: Array| x.transpose_axes(&[0, 2, 1, 3]);
    let q = to_bhsd(apply_rope(&heads(q)?, cos, sin)?)?;
    let k = to_bhsd(apply_rope(&heads(k)?, cos, sin)?)?;
    let v = to_bhsd(heads(v)?)?;
    let o = scaled_dot_product_attention(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), None, None)?;
    Ok(o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, DIM])?)
}

/// `x * cos + rotate_half(x) * sin` on (B, S, H, D).
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    Ok(x.multiply(cos)?
        .add(crate::nn::rotate_half(x)?.multiply(sin)?)?)
}

/// The feed-forward activation: a soft-clamped SiLU-like gate times a
/// soft-clamped value, `(4 tanh(g/4) sigmoid(g)) * (25 tanh(h/25))`.
fn situ(hidden: &Array, gate: &Array) -> Result<Array> {
    let dtype = gate.dtype();
    let s = |v: f32| scalar(v, dtype);
    let g = s(4.0)?
        .multiply(tanh(&gate.divide(s(4.0)?)?)?)?
        .multiply(sigmoid(gate)?)?;
    let h = s(25.0)?.multiply(tanh(&hidden.divide(s(25.0)?)?)?)?;
    Ok(g.multiply(h)?)
}

/// The joint sequence's RoPE tables, (1, h * w + text_len, 1, 128) each:
/// image tokens first, at `(0, text_len, row, column)` with the grid
/// centered on 0, then text tokens at `(0, i, 0, 0)`. Each position's
/// phases (4 axes x 16 bands) are laid out sin-then-cos, and each half is
/// repeated for the rotate-half pairing, as the reference's `_rotate` does.
/// Computed in f32 and cast to `dtype`.
fn rope_tables(h: i32, w: i32, text_len: i32, dtype: Dtype) -> Result<(Array, Array)> {
    let bands: Vec<f32> = (0..ROPE_BANDS)
        .map(|i| 1.0 / ROPE_BASE.powf(i as f64 / ROPE_BANDS as f64) as f32)
        .collect();
    let (ch, cw) = ((h - 1) as f32 / 2.0, (w - 1) as f32 / 2.0);
    let coords = (0..h)
        .flat_map(|r| (0..w).map(move |c| [0.0, text_len as f32, r as f32 - ch, c as f32 - cw]))
        .chain((0..text_len).map(|i| [0.0, i as f32, 0.0, 0.0]));
    let n = (h * w + text_len) as usize;
    let half = 4 * ROPE_BANDS;
    let mut cos = Vec::with_capacity(n * 2 * half);
    let mut sin = Vec::with_capacity(n * 2 * half);
    for pos in coords {
        let phases: Vec<f32> = pos
            .iter()
            .flat_map(|&p| bands.iter().map(move |&b| p * b))
            .collect();
        for _ in 0..2 {
            sin.extend(phases.iter().map(|p| p.sin()));
            cos.extend(phases.iter().map(|p| p.cos()));
        }
    }
    let shape = [1, n as i32, 1, (2 * half) as i32];
    Ok((
        Array::from_slice(&cos, &shape).as_dtype(dtype)?,
        Array::from_slice(&sin, &shape).as_dtype(dtype)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_positions_center_the_grid_and_follow_the_text() {
        crate::nn::test_device();
        let (cos, sin) = rope_tables(2, 2, 3, Dtype::Float32).unwrap();
        assert_eq!(cos.shape(), &[1, 7, 1, 128]);
        sin.eval().unwrap();
        let sin: Vec<f32> = sin.as_slice::<f32>().to_vec();
        let at = |token: usize, axis: usize, band: usize| sin[token * 128 + axis * 16 + band];
        // Band 0 has frequency 1, so its phase is the coordinate itself.
        // Image token (row 1, column 0): text index 3, centered (0.5, -0.5).
        assert_eq!(at(2, 1, 0), 3f32.sin());
        assert_eq!(at(2, 2, 0), 0.5f32.sin());
        assert_eq!(at(2, 3, 0), (-0.5f32).sin());
        // Text token 2: index 2 on axis 1, nothing else.
        assert_eq!(at(6, 1, 0), 2f32.sin());
        assert_eq!(at(6, 2, 0), 0.0);
        // The second half repeats the first.
        assert_eq!(sin[2 * 128 + 64 + 16], at(2, 1, 0));
    }

    #[test]
    fn situ_matches_its_formula() {
        crate::nn::test_device();
        let h = Array::from_slice(&[-60.0f32, -1.0, 0.0, 2.0, 80.0], &[5]);
        let g = Array::from_slice(&[-3.0f32, 0.5, 1.0, 10.0, -0.25], &[5]);
        let out = situ(&h, &g).unwrap();
        out.eval().unwrap();
        let expect: Vec<f32> = [
            (-60.0f32, -3.0f32),
            (-1.0, 0.5),
            (0.0, 1.0),
            (2.0, 10.0),
            (80.0, -0.25),
        ]
        .iter()
        .map(|&(h, g)| (4.0 * (g / 4.0).tanh() / (1.0 + (-g).exp())) * (25.0 * (h / 25.0).tanh()))
        .collect();
        for (a, b) in out.as_slice::<f32>().iter().zip(&expect) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}

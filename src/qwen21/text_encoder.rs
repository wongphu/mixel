//! Qwen3-VL language model, used as Qwen-Image-2.1's text encoder. Ported from
//! transformers' `modeling_qwen3_vl` (`Qwen3VLTextModel`).
//!
//! Returns the last decoder layer's hidden states *before* the final RMSNorm,
//! which is what the Qwen-Image-2.1 transformer was trained on.

use crate::nn::{linear, rms_norm, rotate_half, silu, split_seq, Weights};
use anyhow::Result;
use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::concatenate;
use mlx_rs::{Array, Dtype};
use std::path::Path;

const HIDDEN: i32 = 4096;
const HEADS: i32 = 32;
const KV_HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;
const NUM_LAYERS: usize = 36;
const EPS: f32 = 1e-6;
const ROPE_THETA: f64 = 5_000_000.0;
/// Interleaved M-RoPE: frequency `i` uses the T position unless `i % 3 == 1`
/// (H) or `i % 3 == 2` (W) within the first `3 * section` frequencies.
const MROPE_SECTION: [usize; 3] = [24, 20, 20];

/// One image's vision features, replacing a run of `<|image_pad|>` tokens.
pub struct ImageEmbeds {
    /// Index of the first `<|image_pad|>` token of this image.
    pub start: usize,
    /// (n_tokens, 4096) merged vision features.
    pub merged: Array,
    /// Deepstack features added after the first decoder layers, each (n_tokens, 4096).
    pub deepstack: Vec<Array>,
}

pub struct TextEncoder {
    w: Weights,
    dtype: Dtype,
}

impl TextEncoder {
    /// Loads the language model half of the Qwen3-VL checkpoint.
    pub fn load(files: &[impl AsRef<Path>], dtype: Dtype) -> Result<Self> {
        let w = Weights::load(files, dtype, |n| n.starts_with("model.language_model."))?;
        Ok(Self { w, dtype })
    }

    /// Runs the decoder over `ids` with 3-axis `positions` (T, H, W per token),
    /// substituting `images` at their `<|image_pad|>` runs. Returns (1, L, 4096).
    pub fn forward(
        &self,
        ids: &[u32],
        positions: &[[i64; 3]],
        images: &[ImageEmbeds],
    ) -> Result<Array> {
        anyhow::ensure!(
            ids.len() == positions.len(),
            "ids/positions length mismatch"
        );
        let len = ids.len() as i32;
        let idx: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let mut h = self
            .w
            .get("model.language_model.embed_tokens.weight")?
            .take_axis(&Array::from_slice(&idx, &[len]), 0)?; // (L, 4096)

        // Image runs, in order: replace the placeholder embeddings.
        let runs: Vec<(i32, i32)> = images
            .iter()
            .map(|im| (im.start as i32, im.start as i32 + im.merged.shape()[0]))
            .collect();
        if !images.is_empty() {
            let merged: Vec<&Array> = images.iter().map(|im| &im.merged).collect();
            h = replace_runs(&h, &runs, &merged, self.dtype)?;
        }
        let mut h = h.expand_dims(0)?; // (1, L, 4096)

        let (cos, sin) = mrope_tables(positions, self.dtype)?;
        let num_deepstack = images.first().map_or(0, |im| im.deepstack.len());
        for layer in 0..NUM_LAYERS {
            h = self.layer(layer, &h, &cos, &sin)?;
            if layer < num_deepstack {
                let feats: Vec<&Array> = images.iter().map(|im| &im.deepstack[layer]).collect();
                let add = replace_runs(
                    &Array::zeros::<f32>(&[len, HIDDEN])?.as_dtype(self.dtype)?,
                    &runs,
                    &feats,
                    self.dtype,
                )?;
                h = h.add(add.expand_dims(0)?)?;
            }
        }
        Ok(h)
    }

    fn layer(&self, i: usize, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let p = format!("model.language_model.layers.{i}");
        let h = rms_norm(x, &self.w, &format!("{p}.input_layernorm"), EPS)?;
        let x = x.add(self.attention(&p, &h, cos, sin)?)?;
        let h = rms_norm(&x, &self.w, &format!("{p}.post_attention_layernorm"), EPS)?;
        let gate = silu(&linear(&h, &self.w, &format!("{p}.mlp.gate_proj"))?)?;
        let up = linear(&h, &self.w, &format!("{p}.mlp.up_proj"))?;
        Ok(x.add(linear(
            &gate.multiply(up)?,
            &self.w,
            &format!("{p}.mlp.down_proj"),
        )?)?)
    }

    fn attention(&self, p: &str, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let proj = |name: &str, heads: i32| -> Result<Array> {
            Ok(linear(x, &self.w, &format!("{p}.self_attn.{name}"))?
                .reshape(&[b, l, heads, HEAD_DIM])?)
        };
        let q = rms_norm(
            &proj("q_proj", HEADS)?,
            &self.w,
            &format!("{p}.self_attn.q_norm"),
            EPS,
        )?
        .transpose_axes(&[0, 2, 1, 3])?;
        let k = rms_norm(
            &proj("k_proj", KV_HEADS)?,
            &self.w,
            &format!("{p}.self_attn.k_norm"),
            EPS,
        )?
        .transpose_axes(&[0, 2, 1, 3])?;
        let v = proj("v_proj", KV_HEADS)?.transpose_axes(&[0, 2, 1, 3])?;
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
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
            .reshape(&[b, l, HEADS * HEAD_DIM])?;
        linear(&o, &self.w, &format!("{p}.self_attn.o_proj"))
    }
}

/// cos/sin tables (1, 1, L, 128) for interleaved M-RoPE, computed in f32 and
/// cast to `dtype` like transformers.
fn mrope_tables(positions: &[[i64; 3]], dtype: Dtype) -> Result<(Array, Array)> {
    let half = (HEAD_DIM / 2) as usize;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1.0 / ROPE_THETA.powf((2 * i) as f64 / HEAD_DIM as f64)) as f32)
        .collect();
    let axis_of = |i: usize| -> usize {
        match i % 3 {
            1 if i < 3 * MROPE_SECTION[1] => 1,
            2 if i < 3 * MROPE_SECTION[2] => 2,
            _ => 0,
        }
    };
    let l = positions.len();
    let mut cos = vec![0f32; l * 2 * half];
    let mut sin = vec![0f32; l * 2 * half];
    for (t, pos) in positions.iter().enumerate() {
        for i in 0..half {
            let angle = pos[axis_of(i)] as f32 * inv_freq[i];
            let (s, c) = angle.sin_cos();
            for j in [i, i + half] {
                cos[t * 2 * half + j] = c;
                sin[t * 2 * half + j] = s;
            }
        }
    }
    let shape = [1, 1, l as i32, 2 * half as i32];
    Ok((
        Array::from_slice(&cos, &shape).as_dtype(dtype)?,
        Array::from_slice(&sin, &shape).as_dtype(dtype)?,
    ))
}

/// `x * cos + rotate_half(x) * sin` on (B, H, L, D).
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    Ok(x.multiply(cos)?.add(rotate_half(x)?.multiply(sin)?)?)
}

/// Replaces rows `[start, end)` of `base` (L, D) with `values[i]` for each run.
fn replace_runs(
    base: &Array,
    runs: &[(i32, i32)],
    values: &[&Array],
    dtype: Dtype,
) -> Result<Array> {
    let mut bounds = Vec::new();
    for &(s, e) in runs {
        bounds.push(s);
        bounds.push(e);
    }
    let len = base.shape()[0];
    bounds.retain(|&b| b > 0 && b < len);
    bounds.dedup();
    let pieces = split_seq(base, &bounds, 0)?;
    // Walk the pieces, swapping in values where a run starts.
    let mut out = Vec::with_capacity(pieces.len());
    let mut pos = 0;
    let mut next_run = 0;
    for piece in pieces {
        let n = piece.shape()[0];
        if next_run < runs.len() && runs[next_run].0 == pos {
            anyhow::ensure!(
                values[next_run].shape()[0] == n,
                "image features ({}) do not match placeholder run ({n})",
                values[next_run].shape()[0]
            );
            out.push(values[next_run].as_dtype(dtype)?);
            next_run += 1;
        } else {
            out.push(piece);
        }
        pos += n;
    }
    anyhow::ensure!(next_run == runs.len(), "unmatched image runs");
    Ok(concatenate(&out, 0)?)
}

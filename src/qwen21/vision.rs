//! Qwen3-VL vision encoder and its image preprocessing, ported from
//! transformers' `Qwen3VLVisionModel` and `Qwen2VLImageProcessor`.
//!
//! Each image is encoded on its own (attention never crosses images), giving
//! merged tokens for the text encoder plus deepstack features.

use crate::nn::{gelu, gelu_tanh, layer_norm_affine, linear, rotate_half, Weights};
use anyhow::Result;
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::{Array, Dtype};
use std::path::Path;

const HIDDEN: i32 = 1152;
const HEADS: i32 = 16;
const HEAD_DIM: i32 = HIDDEN / HEADS; // 72
const DEPTH: usize = 27;
const PATCH: usize = 16;
const TEMPORAL_PATCH: usize = 2;
pub const MERGE: usize = 2;
const POS_GRID: usize = 48; // sqrt(num_position_embeddings = 2304)
const DEEPSTACK_LAYERS: [usize; 3] = [8, 16, 24];
const ROPE_THETA: f32 = 10000.0;
const EPS: f32 = 1e-6;

/// Output of the vision encoder for one image.
pub struct VisionFeatures {
    /// Merged grid size in tokens (each covers 32x32 px).
    pub grid: (usize, usize),
    /// (tokens, 4096)
    pub merged: Array,
    /// Three (tokens, 4096) deepstack feature maps.
    pub deepstack: Vec<Array>,
}

pub struct VisionEncoder {
    w: Weights,
    dtype: Dtype,
}

impl VisionEncoder {
    pub fn load(files: &[impl AsRef<Path>], dtype: Dtype) -> Result<Self> {
        let w = Weights::load(files, dtype, |n| n.starts_with("model.visual."))?;
        Ok(Self { w, dtype })
    }

    /// Encodes an RGB image whose sides are multiples of 32.
    pub fn encode(&self, img: &image::RgbImage) -> Result<VisionFeatures> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        anyhow::ensure!(
            w % (PATCH * MERGE) == 0 && h % (PATCH * MERGE) == 0,
            "vision input must be a multiple of 32 px, got {w}x{h}"
        );
        let (gh, gw) = (h / PATCH, w / PATCH);
        let n = (gh * gw) as i32;

        let patches = patchify(img)?.as_dtype(self.dtype)?; // (n, 1536)
        let proj = self
            .w
            .get("model.visual.patch_embed.proj.weight")?
            .reshape(&[HIDDEN, -1])?;
        let mut x = patches
            .matmul(proj.t())?
            .add(self.w.get("model.visual.patch_embed.proj.bias")?)?;
        x = x.add(self.position_embeddings(gh, gw)?)?;

        let (cos, sin) = rope_tables(gh, gw)?;
        let mut deepstack = Vec::new();
        for layer in 0..DEPTH {
            x = self.block(layer, &x, &cos, &sin, n)?;
            if let Some(k) = DEEPSTACK_LAYERS.iter().position(|&l| l == layer) {
                deepstack.push(self.merger(
                    &x,
                    &format!("model.visual.deepstack_merger_list.{k}"),
                    true,
                )?);
            }
        }
        let merged = self.merger(&x, "model.visual.merger", false)?;
        Ok(VisionFeatures {
            grid: (gh / MERGE, gw / MERGE),
            merged,
            deepstack,
        })
    }

    /// Learned 48x48 position table, bilinearly resampled (align_corners) to
    /// the patch grid, in merge-block patch order. Summed in f32.
    fn position_embeddings(&self, gh: usize, gw: usize) -> Result<Array> {
        let taps = |i: usize, size: usize| -> [(usize, f32); 2] {
            let side = POS_GRID as f32;
            let src = i as f32 * (side - 1.0) / (size.max(2) - 1) as f32;
            let src = if size == 1 { 0.0 } else { src };
            let f = src.floor();
            let lo = f as usize;
            let hi = (lo + 1).min(POS_GRID - 1);
            let d = src - f;
            [(lo, 1.0 - d), (hi, d)]
        };
        let mut idx = Vec::new();
        let mut wts = Vec::new();
        for (row, col) in block_order(gh, gw) {
            for (hr, hw) in taps(row, gh) {
                for (wc, ww) in taps(col, gw) {
                    idx.push((hr * POS_GRID + wc) as i32);
                    wts.push(hw * ww);
                }
            }
        }
        let n = (gh * gw) as i32;
        let table = self.w.get("model.visual.pos_embed.weight")?;
        let rows = table
            .take_axis(&Array::from_slice(&idx, &[n * 4]), 0)?
            .as_dtype(Dtype::Float32)?
            .reshape(&[n, 4, HIDDEN])?;
        let weights = Array::from_slice(&wts, &[n, 4, 1]);
        Ok(rows
            .multiply(weights)?
            .sum_axis(1, None)?
            .as_dtype(self.dtype)?)
    }

    fn block(&self, i: usize, x: &Array, cos: &Array, sin: &Array, n: i32) -> Result<Array> {
        let p = format!("model.visual.blocks.{i}");
        let h = layer_norm_affine(x, &self.w, &format!("{p}.norm1"), EPS)?;
        let qkv =
            linear(&h, &self.w, &format!("{p}.attn.qkv"))?.reshape(&[n, 3, HEADS, HEAD_DIM])?;
        let [q, k, v]: [Array; 3] = qkv.split_equal(3, 1)?.try_into().expect("3 parts");
        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[n, HEADS, HEAD_DIM])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?)
        };
        // RoPE in f32, cast back (like `apply_rotary_pos_emb_vision`).
        let rope = |a: Array| -> Result<Array> {
            let a32 = a.as_dtype(Dtype::Float32)?.reshape(&[n, HEADS, HEAD_DIM])?;
            let out = a32.multiply(cos)?.add(rotate_half(&a32)?.multiply(sin)?)?;
            Ok(out.as_dtype(self.dtype)?)
        };
        let q = to_heads(rope(q)?)?;
        let k = to_heads(rope(k)?)?;
        let v = to_heads(v)?;
        let o =
            scaled_dot_product_attention(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), None, None)?;
        let o = o
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[n, HIDDEN])?;
        let x = x.add(linear(&o, &self.w, &format!("{p}.attn.proj"))?)?;
        let h = layer_norm_affine(&x, &self.w, &format!("{p}.norm2"), EPS)?;
        let h = gelu_tanh(&linear(&h, &self.w, &format!("{p}.mlp.linear_fc1"))?)?;
        Ok(x.add(linear(&h, &self.w, &format!("{p}.mlp.linear_fc2"))?)?)
    }

    /// Merges 2x2 patch blocks into one 4096-dim token. The deepstack mergers
    /// normalize after merging (`use_postshuffle_norm`), the final one before.
    fn merger(&self, x: &Array, p: &str, postshuffle_norm: bool) -> Result<Array> {
        let groups = x.shape()[0] / (MERGE * MERGE) as i32;
        let merged_dim = HIDDEN * (MERGE * MERGE) as i32;
        let x = if postshuffle_norm {
            layer_norm_affine(
                &x.reshape(&[groups, merged_dim])?,
                &self.w,
                &format!("{p}.norm"),
                EPS,
            )?
        } else {
            layer_norm_affine(x, &self.w, &format!("{p}.norm"), EPS)?
                .reshape(&[groups, merged_dim])?
        };
        let h = gelu(&linear(&x, &self.w, &format!("{p}.linear_fc1"))?)?;
        linear(&h, &self.w, &format!("{p}.linear_fc2"))
    }
}

/// Patch (row, col) coordinates in merge-block order: 2x2 blocks in raster
/// order, and raster order inside each block.
fn block_order(gh: usize, gw: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..gh / MERGE).flat_map(move |br| {
        (0..gw / MERGE).flat_map(move |bc| {
            (0..MERGE)
                .flat_map(move |ir| (0..MERGE).map(move |ic| (br * MERGE + ir, bc * MERGE + ic)))
        })
    })
}

/// Normalized pixels as (patches, C * T * 16 * 16) in merge-block order, each
/// patch repeated over the 2 temporal slots (`Qwen2VLImageProcessor.patchify`).
pub fn patchify(img: &image::RgbImage) -> Result<Array> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let (gh, gw) = (h / PATCH, w / PATCH);
    let px = img.as_raw();
    let per_patch = 3 * TEMPORAL_PATCH * PATCH * PATCH;
    let mut out = Vec::with_capacity(gh * gw * per_patch);
    for (row, col) in block_order(gh, gw) {
        for c in 0..3 {
            for _t in 0..TEMPORAL_PATCH {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let (yy, xx) = (row * PATCH + y, col * PATCH + x);
                        let v = px[(yy * w + xx) * 3 + c] as f32 / 255.0;
                        out.push((v - 0.5) / 0.5);
                    }
                }
            }
        }
    }
    Ok(Array::from_slice(
        &out,
        &[(gh * gw) as i32, per_patch as i32],
    ))
}

/// 2D rotary tables (n, 1, 72) for the patch grid: 18 frequencies for the row
/// and 18 for the column, repeated over both halves. f32.
fn rope_tables(gh: usize, gw: usize) -> Result<(Array, Array)> {
    let spatial = (HEAD_DIM / 2) as usize; // 36
    let inv_freq: Vec<f32> = (0..spatial / 2)
        .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / spatial as f32))
        .collect();
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for (row, col) in block_order(gh, gw) {
        let angles: Vec<f32> = inv_freq
            .iter()
            .map(|f| row as f32 * f)
            .chain(inv_freq.iter().map(|f| col as f32 * f))
            .collect();
        for _ in 0..2 {
            cos.extend(angles.iter().map(|a| a.cos()));
            sin.extend(angles.iter().map(|a| a.sin()));
        }
    }
    let shape = [(gh * gw) as i32, 1, HEAD_DIM];
    Ok((
        Array::from_slice(&cos, &shape),
        Array::from_slice(&sin, &shape),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_order_groups_2x2_blocks() {
        let order: Vec<_> = block_order(2, 4).collect();
        assert_eq!(
            order,
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3)
            ]
        );
    }
}

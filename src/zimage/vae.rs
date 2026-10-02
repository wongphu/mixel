//! Z-Image VAE (diffusers AutoencoderKL layout), ported from candle's
//! `z_image::vae`. Runs in NHWC, MLX's native convolution layout.

use super::{linear, scalar, silu, Weights};
use crate::nn::conv_in_bands;
use anyhow::Result;
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{broadcast_to, conv2d, pad, split_at_indices};
use mlx_rs::{Array, Dtype};
use std::path::Path;

const SCALING_FACTOR: f32 = 0.3611;
const SHIFT_FACTOR: f32 = 0.1159;
const GROUPS: i32 = 32;
const EPS: f32 = 1e-6;
const NUM_BLOCKS: usize = 4;
const RESNETS_PER_DOWN_BLOCK: usize = 2;
const RESNETS_PER_UP_BLOCK: usize = 3;
const LATENT_CHANNELS: i32 = 16;
/// Bytes of f32 per band when summing GroupNorm statistics.
const STATS_BAND_BYTES: usize = 128 << 20;

pub struct Vae {
    w: Weights,
    dtype: Dtype,
}

impl Vae {
    pub fn load(file: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let w = Weights::load(&[file], dtype, |name| {
            name.starts_with("decoder.") || name.starts_with("encoder.")
        })?;
        Ok(Self { w, dtype })
    }

    /// Image (B, H, W, 3) in [-1, 1] -> latents (B, H/8, W/8, 16).
    ///
    /// Uses the mean of the latent distribution (no sampling), so encoding is
    /// deterministic; candle's `AutoEncoderKL::encode` samples instead.
    pub fn encode(&self, x: &Array) -> Result<Array> {
        let moments = self.encoder_moments(x)?;
        let mean = split_at_indices(&moments, &[LATENT_CHANNELS], -1)?.swap_remove(0);
        Ok(mean
            .subtract(scalar(SHIFT_FACTOR, self.dtype)?)?
            .multiply(scalar(SCALING_FACTOR, self.dtype)?)?)
    }

    /// Raw encoder output (B, H/8, W/8, 32): latent mean and log-variance.
    pub fn encoder_moments(&self, x: &Array) -> Result<Array> {
        let mut h = self.conv(x, "encoder.conv_in", 1)?;
        for i in 0..NUM_BLOCKS {
            for j in 0..RESNETS_PER_DOWN_BLOCK {
                h = self.resnet(&format!("encoder.down_blocks.{i}.resnets.{j}"), &h)?;
            }
            let down = format!("encoder.down_blocks.{i}.downsamplers.0.conv");
            if self.w.has(&format!("{down}.weight")) {
                // Pad right/bottom by 1, then a stride-2 conv without padding.
                let padded = pad(&h, &[(0, 0), (0, 1), (0, 1), (0, 0)], None, None)?;
                h = self.conv_strided(&padded, &down, 0, 2)?;
            }
        }
        h = self.resnet("encoder.mid_block.resnets.0", &h)?;
        h = self.attention("encoder.mid_block.attentions.0", &h)?;
        h = self.resnet("encoder.mid_block.resnets.1", &h)?;
        self.norm_silu_conv(&h, "encoder.conv_norm_out", "encoder.conv_out")
    }

    /// Latents (B, H, W, 16) -> image (B, 8H, 8W, 3) in [-1, 1].
    pub fn decode(&self, z: &Array) -> Result<Array> {
        let z = z
            .divide(scalar(SCALING_FACTOR, self.dtype)?)?
            .add(scalar(SHIFT_FACTOR, self.dtype)?)?;
        let mut h = self.conv(&z, "decoder.conv_in", 1)?;
        h = self.resnet("decoder.mid_block.resnets.0", &h)?;
        h = self.attention("decoder.mid_block.attentions.0", &h)?;
        h = self.resnet("decoder.mid_block.resnets.1", &h)?;
        for i in 0..NUM_BLOCKS {
            for j in 0..RESNETS_PER_UP_BLOCK {
                h = self.resnet(&format!("decoder.up_blocks.{i}.resnets.{j}"), &h)?;
            }
            let up = format!("decoder.up_blocks.{i}.upsamplers.0.conv");
            if self.w.has(&format!("{up}.weight")) {
                h = self.conv(&upsample_nearest_2x(&h)?, &up, 1)?;
            }
        }
        self.norm_silu_conv(&h, "decoder.conv_norm_out", "decoder.conv_out")
    }

    fn conv(&self, x: &Array, p: &str, padding: i32) -> Result<Array> {
        self.conv_strided(x, p, padding, 1)
    }

    fn conv_strided(&self, x: &Array, p: &str, padding: i32, stride: i32) -> Result<Array> {
        let weight = self.w.get(&format!("{p}.weight"))?;
        let bias = self.w.get(&format!("{p}.bias"))?;
        if stride == 1 {
            return conv_in_bands(x, weight, Some(bias), padding, |x| Ok(x.clone()));
        }
        let y = conv2d(x, weight, (stride, stride), (padding, padding), None, None)?;
        Ok(y.add(bias)?)
    }

    /// `conv(silu(group_norm(x)))` with a 3x3 `conv`, in bands of rows (see
    /// [`conv_in_bands`]): only the norm's statistics need all of `x`.
    fn norm_silu_conv(&self, x: &Array, norm: &str, conv: &str) -> Result<Array> {
        let (mean, rstd) = group_stats(x)?;
        let (gamma, beta) = (
            self.w.get(&format!("{norm}.weight"))?,
            self.w.get(&format!("{norm}.bias"))?,
        );
        conv_in_bands(
            x,
            self.w.get(&format!("{conv}.weight"))?,
            Some(self.w.get(&format!("{conv}.bias"))?),
            1,
            |x| {
                silu(
                    &normalize(x, &mean, &rstd, self.dtype)?
                        .multiply(gamma)?
                        .add(beta)?,
                )
            },
        )
    }

    /// GroupNorm over NHWC. Like candle, normalizes in f32, then applies the
    /// affine parameters in the model dtype.
    fn group_norm(&self, x: &Array, p: &str) -> Result<Array> {
        let (mean, rstd) = group_stats(x)?;
        Ok(normalize(x, &mean, &rstd, self.dtype)?
            .multiply(self.w.get(&format!("{p}.weight"))?)?
            .add(self.w.get(&format!("{p}.bias"))?)?)
    }

    fn resnet(&self, p: &str, x: &Array) -> Result<Array> {
        let h = self.norm_silu_conv(x, &format!("{p}.norm1"), &format!("{p}.conv1"))?;
        let h = self.norm_silu_conv(&h, &format!("{p}.norm2"), &format!("{p}.conv2"))?;
        let shortcut = format!("{p}.conv_shortcut");
        let x = if self.w.has(&format!("{shortcut}.weight")) {
            self.conv(x, &shortcut, 0)?
        } else {
            x.clone()
        };
        Ok(x.add(h)?)
    }

    /// Single-head self-attention over all pixels.
    fn attention(&self, p: &str, x: &Array) -> Result<Array> {
        let sh = x.shape().to_vec();
        let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
        let n = self
            .group_norm(x, &format!("{p}.group_norm"))?
            .reshape(&[b, h * w, c])?;
        let qkv = |name: &str| -> Result<Array> {
            Ok(linear(&n, &self.w, &format!("{p}.{name}"))?.reshape(&[b, 1, h * w, c])?)
        };
        let o = scaled_dot_product_attention(
            qkv("to_q")?,
            qkv("to_k")?,
            qkv("to_v")?,
            1.0 / (c as f32).sqrt(),
            None,
            None,
        )?;
        let o = linear(
            &o.reshape(&[b, h * w, c])?,
            &self.w,
            &format!("{p}.to_out.0"),
        )?;
        Ok(o.reshape(&[b, h, w, c])?.add(x)?)
    }
}

/// GroupNorm statistics of NHWC `x`: per-group mean and 1/std in f32, each
/// (B, 1, GROUPS, 1). Summed over bands of rows, so `x` is never all in f32.
fn group_stats(x: &Array) -> Result<(Array, Array)> {
    let (b, h, w, c) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let rows = (STATS_BAND_BYTES / (w * c * 4).max(1) as usize).clamp(1, h as usize) as i32;
    let groups = |r: i32| -> Result<Array> {
        let n = rows.min(h - r);
        Ok(x.index((.., r..r + n, .., ..))
            .as_dtype(Dtype::Float32)?
            .reshape(&[b, n * w, GROUPS, c / GROUPS])?)
    };
    let count = Array::from_f32((h * w * (c / GROUPS)) as f32);
    let sum_bands = |f: &dyn Fn(Array) -> Result<Array>| -> Result<Array> {
        let mut total: Option<Array> = None;
        for r in (0..h).step_by(rows as usize) {
            let s = f(groups(r)?)?.sum_axes(&[1, 3], true)?;
            total = Some(match total {
                Some(t) => t.add(s)?,
                None => s,
            });
            total.as_ref().unwrap().eval()?;
        }
        Ok(total.expect("at least one row"))
    };
    let mean = sum_bands(&|g| Ok(g))?.divide(&count)?;
    let var = sum_bands(&|g| Ok(g.subtract(&mean)?.square()?))?.divide(&count)?;
    Ok((mean, var.add(Array::from_f32(EPS))?.rsqrt()?))
}

/// `(x - mean) * rstd` per group in f32, cast to `dtype`.
fn normalize(x: &Array, mean: &Array, rstd: &Array, dtype: Dtype) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    Ok(x.as_dtype(Dtype::Float32)?
        .reshape(&[b, h * w, GROUPS, c / GROUPS])?
        .subtract(mean)?
        .multiply(rstd)?
        .reshape(&[b, h, w, c])?
        .as_dtype(dtype)?)
}

/// Nearest-neighbor 2x upsampling of NHWC.
fn upsample_nearest_2x(x: &Array) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.reshape(&[b, h, 1, w, 1, c])?;
    Ok(broadcast_to(&x, &[b, h, 2, w, 2, c])?.reshape(&[b, 2 * h, 2 * w, c])?)
}

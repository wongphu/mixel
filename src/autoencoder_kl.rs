//! diffusers' AutoencoderKL encoder and decoder, shared by Z-Image's VAE
//! (FLUX.1's) and MOSS-Image's (FLUX.2's), ported from candle's
//! `z_image::vae`. Runs in NHWC, MLX's native convolution layout. Channel
//! counts come from the weights, so the decoder may be narrower than the
//! encoder; the 1x1 `quant_conv` / `post_quant_conv` run when present.

use crate::nn::{conv_in_bands, linear, silu, Weights};
use anyhow::Result;
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{broadcast_to, conv2d, pad};
use mlx_rs::{Array, Dtype};
use std::path::Path;

const GROUPS: i32 = 32;
const EPS: f32 = 1e-6;
const NUM_BLOCKS: usize = 4;
const RESNETS_PER_DOWN_BLOCK: usize = 2;
const RESNETS_PER_UP_BLOCK: usize = 3;
/// Bytes of f32 per band when summing GroupNorm statistics.
const STATS_BAND_BYTES: usize = 128 << 20;

pub(crate) struct AutoencoderKl {
    pub(crate) w: Weights,
    dtype: Dtype,
}

impl AutoencoderKl {
    /// Loads the encoder, decoder and quant convs, plus any other tensors
    /// whose names pass `extra`.
    pub(crate) fn load(
        file: impl AsRef<Path>,
        dtype: Dtype,
        extra: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        let w = Weights::load(&[file], dtype, |name| {
            name.starts_with("decoder.")
                || name.starts_with("encoder.")
                || name.starts_with("quant_conv.")
                || name.starts_with("post_quant_conv.")
                || extra(name)
        })?;
        Ok(Self { w, dtype })
    }

    /// Raw encoder output (B, H/8, W/8, 2 * latent channels): latent mean and
    /// log-variance, after `quant_conv` if the model has one.
    pub(crate) fn encoder_moments(&self, x: &Array) -> Result<Array> {
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
        let h = self.norm_silu_conv(&h, "encoder.conv_norm_out", "encoder.conv_out")?;
        if self.w.has("quant_conv.weight") {
            self.conv(&h, "quant_conv", 0)
        } else {
            Ok(h)
        }
    }

    /// Latents as the decoder takes them (B, H, W, latent channels), before
    /// `post_quant_conv` if the model has one -> image (B, 8H, 8W, 3).
    pub(crate) fn decode(&self, z: &Array) -> Result<Array> {
        let z = if self.w.has("post_quant_conv.weight") {
            self.conv(z, "post_quant_conv", 0)?
        } else {
            z.clone()
        };
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

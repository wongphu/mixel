//! Z-Image VAE decoder (diffusers AutoencoderKL layout), ported from candle's
//! `z_image::vae`. Runs in NHWC, MLX's native convolution layout.

use super::{linear, scalar, silu, Weights};
use anyhow::Result;
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{broadcast_to, conv2d};
use mlx_rs::{Array, Dtype};
use std::path::Path;

const SCALING_FACTOR: f32 = 0.3611;
const SHIFT_FACTOR: f32 = 0.1159;
const GROUPS: i32 = 32;
const EPS: f32 = 1e-6;
const NUM_UP_BLOCKS: usize = 4;
const RESNETS_PER_UP_BLOCK: usize = 3;

pub struct Decoder {
    w: Weights,
    dtype: Dtype,
}

impl Decoder {
    pub fn load(file: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let w = Weights::load(&[file], dtype, |name| name.starts_with("decoder."))?;
        Ok(Self { w, dtype })
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
        for i in 0..NUM_UP_BLOCKS {
            for j in 0..RESNETS_PER_UP_BLOCK {
                h = self.resnet(&format!("decoder.up_blocks.{i}.resnets.{j}"), &h)?;
            }
            let up = format!("decoder.up_blocks.{i}.upsamplers.0.conv");
            if self.w.has(&format!("{up}.weight")) {
                h = self.conv(&upsample_nearest_2x(&h)?, &up, 1)?;
            }
        }
        let h = silu(&self.group_norm(&h, "decoder.conv_norm_out")?)?;
        self.conv(&h, "decoder.conv_out", 1)
    }

    fn conv(&self, x: &Array, p: &str, padding: i32) -> Result<Array> {
        let y = conv2d(
            x,
            self.w.get(&format!("{p}.weight"))?,
            None,
            (padding, padding),
            None,
            None,
        )?;
        Ok(y.add(self.w.get(&format!("{p}.bias"))?)?)
    }

    /// GroupNorm over NHWC. Like candle, normalizes in f32, then applies the
    /// affine parameters in the model dtype.
    fn group_norm(&self, x: &Array, p: &str) -> Result<Array> {
        let sh = x.shape().to_vec();
        let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
        let g = x
            .as_dtype(Dtype::Float32)?
            .reshape(&[b, h * w, GROUPS, c / GROUPS])?;
        let mean = g.mean_axes(&[1, 3], true)?;
        let var = g.var_axes(&[1, 3], true, None)?;
        let n = g
            .subtract(&mean)?
            .multiply(var.add(Array::from_f32(EPS))?.rsqrt()?)?
            .reshape(&[b, h, w, c])?
            .as_dtype(self.dtype)?;
        Ok(n.multiply(self.w.get(&format!("{p}.weight"))?)?
            .add(self.w.get(&format!("{p}.bias"))?)?)
    }

    fn resnet(&self, p: &str, x: &Array) -> Result<Array> {
        let h = silu(&self.group_norm(x, &format!("{p}.norm1"))?)?;
        let h = self.conv(&h, &format!("{p}.conv1"), 1)?;
        let h = silu(&self.group_norm(&h, &format!("{p}.norm2"))?)?;
        let h = self.conv(&h, &format!("{p}.conv2"), 1)?;
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

/// Nearest-neighbor 2x upsampling of NHWC.
fn upsample_nearest_2x(x: &Array) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.reshape(&[b, h, 1, w, 1, c])?;
    Ok(broadcast_to(&x, &[b, h, 2, w, 2, c])?.reshape(&[b, 2 * h, 2 * w, c])?)
}

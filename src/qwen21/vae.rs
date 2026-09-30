//! Qwen-Image-2.1 VAE (`AutoencoderKLQwenImage21`), ported from diffusers for
//! single images. Runs in NHWC.
//!
//! The diffusers module is written for video, but for one frame its causal 3D
//! convolutions are plain 2D ones and its temporal convolutions never run.
//! What is left of the temporal axis lives in the resampling shortcuts: the
//! down-shortcuts average over a zero frame padded in front, and the
//! up-shortcuts keep only the second of their two temporal copies.

use crate::nn::{scalar, silu, Weights};
use anyhow::Result;
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{broadcast_to, concatenate, conv2d, pad, split_at_indices};
use mlx_rs::{Array, Dtype};
use std::path::Path;

pub const Z_DIM: i32 = 64;
const ENC_DIMS: [i32; 6] = [96, 96, 192, 384, 768, 768];
const DEC_DIMS: [i32; 6] = [1152, 1152, 1152, 576, 288, 144];
/// Temporal down/upsampling per block (encoder order; the decoder reverses it).
const TEMPORAL: [bool; 4] = [false, true, true, true];
const RESNETS: usize = 2;

// Checkpoint values; -0.5236 is not an approximation of pi/6.
#[allow(clippy::approx_constant)]
#[rustfmt::skip]
const LATENTS_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651,
    0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391,
    -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342,
    -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368,
    0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162,
    1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
#[rustfmt::skip]
const LATENTS_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904,
    2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075,
    3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149,
    3.7662, 3.6746, 3.4563, 3.8161,
];

pub struct Vae {
    w: Weights,
    dtype: Dtype,
}

impl Vae {
    pub fn load(file: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let w = Weights::load(&[file], dtype, |n| !n.contains("time_conv"))?;
        Ok(Self { w, dtype })
    }

    /// Normalized latents (1, h, w, 64) -> RGBA image (1, 16h, 16w, 4) in [-1, 1].
    pub fn decode(&self, z: &Array) -> Result<Array> {
        let z = z
            .multiply(self.stats(&LATENTS_STD)?)?
            .add(self.stats(&LATENTS_MEAN)?)?;
        self.decode_denormalized(&z)
    }

    /// Like [`decode`](Self::decode) for latents already mapped back with the
    /// latent mean and std (what diffusers' `vae.decode` receives).
    pub fn decode_denormalized(&self, z: &Array) -> Result<Array> {
        let z = self.conv(z, "post_quant_conv", 0, 1)?;
        let mut x = self.conv(&z, "decoder.conv_in", 1, 1)?;
        x = self.mid_block("decoder.mid_block", &x)?;
        let temporal_up: Vec<bool> = TEMPORAL.iter().rev().copied().collect();
        for i in 0..5 {
            let p = format!("decoder.up_blocks.{i}");
            let (cin, cout) = (DEC_DIMS[i], DEC_DIMS[i + 1]);
            let up = i < 4;
            let copy = x.clone();
            for j in 0..=RESNETS {
                x = self.resnet(&format!("{p}.resnets.{j}"), &x)?;
            }
            if up {
                x = self.conv(
                    &upsample_nearest_2x(&x)?,
                    &format!("{p}.upsampler.resample.1"),
                    1,
                    1,
                )?;
                let factor_t = if temporal_up[i] { 2 } else { 1 };
                x = x.add(dup_up(&copy, cin, cout, factor_t)?)?;
            }
        }
        let x = silu(&self.rms_norm(&x, "decoder.norm_out")?)?;
        let x = self.conv(&x, "decoder.conv_out", 1, 1)?;
        Ok(mlx_rs::ops::clip(&x, (-1.0f32, 1.0f32))?)
    }

    /// RGBA image (1, H, W, 4) in [-1, 1] -> normalized latents (1, H/16, W/16, 64),
    /// using the mean of the latent distribution.
    pub fn encode(&self, x: &Array) -> Result<Array> {
        self.encode_traced(x, &mut |_, _| {})
    }

    /// [`encode`](Self::encode), calling `trace` with each stage's output
    /// (for checking the port against the reference).
    #[doc(hidden)]
    pub fn encode_traced(&self, x: &Array, trace: &mut dyn FnMut(&str, &Array)) -> Result<Array> {
        let mut x = self.conv(x, "encoder.conv_in", 1, 1)?;
        trace("enc_conv_in", &x);
        for i in 0..5 {
            let p = format!("encoder.down_blocks.{i}");
            let (cin, cout) = (ENC_DIMS[i], ENC_DIMS[i + 1]);
            let down = i < 4;
            let copy = x.clone();
            for j in 0..RESNETS {
                x = self.resnet(&format!("{p}.resnets.{j}"), &x)?;
            }
            trace(&format!("enc_down_{i}_resnets"), &x);
            if down {
                let padded = pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)], None, None)?;
                x = self.conv(&padded, &format!("{p}.downsampler.resample.1"), 0, 2)?;
            }
            let factor_t = if down && TEMPORAL[i] { 2 } else { 1 };
            let factor_s = if down { 2 } else { 1 };
            let shortcut = avg_down(&copy, cin, cout, factor_t, factor_s)?;
            trace(&format!("enc_down_{i}_shortcut"), &shortcut);
            x = x.add(shortcut)?;
            trace(&format!("enc_down_{i}"), &x);
        }
        x = self.mid_block("encoder.mid_block", &x)?;
        trace("enc_mid", &x);
        let x = silu(&self.rms_norm(&x, "encoder.norm_out")?)?;
        let x = self.conv(&x, "encoder.conv_out", 1, 1)?;
        let moments = self.conv(&x, "quant_conv", 0, 1)?;
        trace("enc_moments", &moments);
        let mean = split_at_indices(&moments, &[Z_DIM], -1)?.swap_remove(0);
        Ok(mean
            .subtract(self.stats(&LATENTS_MEAN)?)?
            .divide(self.stats(&LATENTS_STD)?)?)
    }

    fn stats(&self, v: &[f32; 64]) -> Result<Array> {
        Ok(Array::from_slice(v, &[64]).as_dtype(self.dtype)?)
    }

    fn conv(&self, x: &Array, p: &str, padding: i32, stride: i32) -> Result<Array> {
        let y = conv2d(
            x,
            self.w.get(&format!("{p}.weight"))?,
            (stride, stride),
            (padding, padding),
            None,
            None,
        )?;
        Ok(y.add(self.w.get(&format!("{p}.bias"))?)?)
    }

    /// L2-normalizes channels (in f32), times sqrt(C) * gamma.
    fn rms_norm(&self, x: &Array, p: &str) -> Result<Array> {
        let c = *x.shape().last().unwrap();
        let x32 = x.as_dtype(Dtype::Float32)?;
        let norm = mlx_rs::ops::maximum(
            x32.square()?.sum_axis(-1, true)?.sqrt()?,
            Array::from_f32(1e-12),
        )?;
        let normed = x32.divide(norm)?.as_dtype(self.dtype)?;
        let gamma = self.w.get(&format!("{p}.gamma"))?.reshape(&[c])?;
        Ok(normed
            .multiply(scalar((c as f32).sqrt(), self.dtype)?)?
            .multiply(gamma)?)
    }

    fn resnet(&self, p: &str, x: &Array) -> Result<Array> {
        let shortcut = format!("{p}.conv_shortcut");
        let h = if self.w.has(&format!("{shortcut}.weight")) {
            self.conv(x, &shortcut, 0, 1)?
        } else {
            x.clone()
        };
        let y = silu(&self.rms_norm(x, &format!("{p}.norm1"))?)?;
        let y = self.conv(&y, &format!("{p}.conv1"), 1, 1)?;
        let y = silu(&self.rms_norm(&y, &format!("{p}.norm2"))?)?;
        let y = self.conv(&y, &format!("{p}.conv2"), 1, 1)?;
        Ok(y.add(h)?)
    }

    fn mid_block(&self, p: &str, x: &Array) -> Result<Array> {
        let x = self.resnet(&format!("{p}.resnets.0"), x)?;
        let x = self.attention(&format!("{p}.attentions.0"), &x)?;
        self.resnet(&format!("{p}.resnets.1"), &x)
    }

    /// Single-head self-attention over all pixels.
    fn attention(&self, p: &str, x: &Array) -> Result<Array> {
        let sh = x.shape().to_vec();
        let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
        let n = self.rms_norm(x, &format!("{p}.norm"))?;
        let qkv = self
            .conv(&n, &format!("{p}.to_qkv"), 0, 1)?
            .reshape(&[b, 1, h * w, 3 * c])?;
        let [q, k, v]: [Array; 3] = qkv.split_equal(3, -1)?.try_into().expect("3 parts");
        let o = scaled_dot_product_attention(&q, &k, &v, 1.0 / (c as f32).sqrt(), None, None)?;
        let o = self.conv(&o.reshape(&[b, h, w, c])?, &format!("{p}.proj"), 0, 1)?;
        Ok(o.add(x)?)
    }
}

/// `AvgDown3D` for one frame: space-to-depth (with a zero frame in front when
/// `factor_t = 2`), then averages consecutive channel groups down to `cout`.
fn avg_down(x: &Array, cin: i32, cout: i32, factor_t: i32, factor_s: i32) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, h, w) = (sh[0], sh[1], sh[2]);
    let (hs, ws) = (h / factor_s, w / factor_s);
    // (B, H', fs, W', fs, C) -> (B, H', W', C, fs, fs)
    let x = x
        .reshape(&[b, hs, factor_s, ws, factor_s, cin])?
        .transpose_axes(&[0, 1, 3, 5, 2, 4])?
        .reshape(&[b, hs, ws, cin, 1, factor_s * factor_s])?;
    let x = if factor_t == 2 {
        concatenate(
            &[&Array::zeros::<f32>(x.shape())?.as_dtype(x.dtype())?, &x],
            4,
        )?
    } else {
        x
    };
    let factor = factor_t * factor_s * factor_s;
    let group = cin * factor / cout;
    Ok(x.reshape(&[b, hs, ws, cout, group])?.mean_axis(-1, None)?)
}

/// `DupUp3D` for the first (only) chunk: repeat channels, depth-to-space, and
/// keep the last temporal copy.
fn dup_up(x: &Array, cin: i32, cout: i32, factor_t: i32) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, h, w) = (sh[0], sh[1], sh[2]);
    let factor = factor_t * 4;
    let repeats = cout * factor / cin;
    let x = broadcast_to(&x.reshape(&[b, h, w, cin, 1])?, &[b, h, w, cin, repeats])?
        .reshape(&[b, h, w, cout, factor_t, 2, 2])?;
    let x = split_at_indices(&x, &[factor_t - 1], 4)?
        .pop()
        .expect("last copy");
    Ok(x.reshape(&[b, h, w, cout, 2, 2])?
        .transpose_axes(&[0, 1, 4, 2, 5, 3])?
        .reshape(&[b, 2 * h, 2 * w, cout])?)
}

fn upsample_nearest_2x(x: &Array) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.reshape(&[b, h, 1, w, 1, c])?;
    Ok(broadcast_to(&x, &[b, h, 2, w, 2, c])?.reshape(&[b, 2 * h, 2 * w, c])?)
}

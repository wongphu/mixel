//! Z-Image VAE: FLUX.1's AutoencoderKL (see [`crate::autoencoder_kl`]) with
//! its latent scale and shift. Runs in NHWC, MLX's native convolution layout.

use super::scalar;
use crate::autoencoder_kl::AutoencoderKl;
use anyhow::Result;
use mlx_rs::ops::split_at_indices;
use mlx_rs::{Array, Dtype};
use std::path::Path;

const SCALING_FACTOR: f32 = 0.3611;
const SHIFT_FACTOR: f32 = 0.1159;
const LATENT_CHANNELS: i32 = 16;

pub struct Vae {
    kl: AutoencoderKl,
    dtype: Dtype,
}

impl Vae {
    pub fn load(file: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            kl: AutoencoderKl::load(file, dtype, |_| false)?,
            dtype,
        })
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
        self.kl.encoder_moments(x)
    }

    /// Latents (B, H, W, 16) -> image (B, 8H, 8W, 3) in [-1, 1].
    pub fn decode(&self, z: &Array) -> Result<Array> {
        let z = z
            .divide(scalar(SCALING_FACTOR, self.dtype)?)?
            .add(scalar(SHIFT_FACTOR, self.dtype)?)?;
        self.kl.decode(&z)
    }
}

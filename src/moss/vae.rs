//! MOSS-Image VAE: FLUX.2's AutoencoderKL (`AutoencoderKLFlux2`, see
//! [`crate::autoencoder_kl`]) with 32 latent channels. The transformer works
//! on its latents patchified 2x2 to 128 channels at 1/16 of the image size,
//! normalized by the VAE's BatchNorm statistics. Runs in f32, like the
//! reference, and in NHWC.

use crate::autoencoder_kl::AutoencoderKl;
use anyhow::Result;
use mlx_rs::ops::split_at_indices;
use mlx_rs::{Array, Dtype};
use std::path::Path;

const LATENT_CHANNELS: i32 = 32;
const PATCHED_CHANNELS: i32 = 4 * LATENT_CHANNELS;
/// `batch_norm_eps` in the VAE's config.
const BN_EPS: f32 = 1e-4;

pub struct Vae {
    kl: AutoencoderKl,
    /// BatchNorm running mean and `sqrt(var + eps)` per patched channel.
    bn_mean: Array,
    bn_std: Array,
}

impl Vae {
    pub fn load(file: impl AsRef<Path>) -> Result<Self> {
        let kl = AutoencoderKl::load(file, Dtype::Float32, |name| {
            name == "bn.running_mean" || name == "bn.running_var"
        })?;
        let bn_mean = kl.w.get("bn.running_mean")?.clone();
        let bn_std =
            kl.w.get("bn.running_var")?
                .add(Array::from_f32(BN_EPS))?
                .sqrt()?;
        anyhow::ensure!(
            bn_mean.shape() == [PATCHED_CHANNELS],
            "bn.running_mean has shape {:?}",
            bn_mean.shape()
        );
        Ok(Self {
            kl,
            bn_mean,
            bn_std,
        })
    }

    /// Image (1, H, W, 3) in [-1, 1] -> latent tokens (1, H/16 * W/16, 128),
    /// normalized. Uses the latent distribution's mean, so it is deterministic.
    pub fn encode(&self, x: &Array) -> Result<Array> {
        let moments = self.kl.encoder_moments(&x.as_dtype(Dtype::Float32)?)?;
        let mean = split_at_indices(&moments, &[LATENT_CHANNELS], -1)?.swap_remove(0);
        let (h, w) = (mean.shape()[1] / 2, mean.shape()[2] / 2);
        // (1, 2h, 2w, 32) -> (1, h, w, 128), channel c * 4 + dy * 2 + dx.
        let patched = mean
            .reshape(&[1, h, 2, w, 2, LATENT_CHANNELS])?
            .transpose_axes(&[0, 1, 3, 5, 2, 4])?
            .reshape(&[1, h * w, PATCHED_CHANNELS])?;
        Ok(patched.subtract(&self.bn_mean)?.divide(&self.bn_std)?)
    }

    /// Latent tokens (1, h * w, 128) -> image (1, 16h, 16w, 3) in about [-1, 1].
    pub fn decode(&self, z: &Array, h: i32, w: i32) -> Result<Array> {
        let z = z
            .as_dtype(Dtype::Float32)?
            .multiply(&self.bn_std)?
            .add(&self.bn_mean)?
            .reshape(&[1, h, w, LATENT_CHANNELS, 2, 2])?
            .transpose_axes(&[0, 1, 4, 2, 5, 3])?
            .reshape(&[1, 2 * h, 2 * w, LATENT_CHANNELS])?;
        self.kl.decode(&z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_and_unpatchify_are_inverses_in_the_reference_order() {
        let _mlx = crate::nn::test_device();
        let (h, w) = (2, 3);
        let x = Array::from_iter(
            0..(4 * h * w * LATENT_CHANNELS),
            &[1, 2 * h, 2 * w, LATENT_CHANNELS],
        );
        let patched = x
            .reshape(&[1, h, 2, w, 2, LATENT_CHANNELS])
            .unwrap()
            .transpose_axes(&[0, 1, 3, 5, 2, 4])
            .unwrap()
            .reshape(&[1, h, w, PATCHED_CHANNELS])
            .unwrap();
        // Patched channel c * 4 + dy * 2 + dx at (i, j) holds pixel
        // (2i + dy, 2j + dx) of channel c.
        patched.eval().unwrap();
        let p: Vec<i32> = patched.as_slice::<i32>().to_vec();
        let (i, j, c, dy, dx) = (1, 2, 5, 1, 0);
        let pixel = ((2 * i + dy) * 2 * w + (2 * j + dx)) * LATENT_CHANNELS + c;
        assert_eq!(
            p[((i * w + j) * PATCHED_CHANNELS + c * 4 + dy * 2 + dx) as usize],
            pixel
        );
        let back = patched
            .reshape(&[1, h, w, LATENT_CHANNELS, 2, 2])
            .unwrap()
            .transpose_axes(&[0, 1, 4, 2, 5, 3])
            .unwrap()
            .reshape(&[1, 2 * h, 2 * w, LATENT_CHANNELS])
            .unwrap();
        back.eval().unwrap();
        assert_eq!(back.as_slice::<i32>(), x.as_slice::<i32>());
    }
}

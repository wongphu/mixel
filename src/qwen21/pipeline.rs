//! Qwen-Image-2.1 generation: text-to-image, img2img, and editing with
//! reference images (ports `QwenImage21Pipeline.__call__`).

use super::prompt::{self, EncodedPrompt};
use super::text_encoder::{ImageEmbeds, TextEncoder};
use super::transformer::{Segment, Transformer, TOKENS_PER_SLOT};
use super::vae::Vae;
use super::vision::VisionEncoder;
use super::{scheduler, OUTPUT_RESOLUTION, VAE_SCALE};
use crate::nn::ModelFiles;
use crate::pipeline::{
    resize_to_fill, seeded_noise, GenerateOptions, Generated, Progress, Timings,
};
use anyhow::{Error as E, Result};
use mlx_rs::{Array, Dtype};
use std::time::Instant;
use tokenizers::Tokenizer;

const TEXT_ENCODER_SHARDS: usize = 4;
const TRANSFORMER_SHARDS: usize = 2;

/// Qwen-Image-2.1's models.
pub struct QwenPipeline {
    dtype: Dtype,
    tokenizer: Tokenizer,
    image_pad_id: u32,
    text_encoder: TextEncoder,
    vision: VisionEncoder,
    transformer: Transformer,
    vae: Vae,
}

/// A reference image prepared once for both encoders.
struct Reference {
    /// Resized RGB image for the vision encoder.
    rgb: image::RgbImage,
    /// Latent grid (height, width) in 16 px tokens.
    grid: (usize, usize),
}

impl QwenPipeline {
    pub fn load(files: &ModelFiles) -> Result<Self> {
        let dtype = Dtype::Bfloat16;
        let tokenizer =
            Tokenizer::from_file(files.get("processor/tokenizer.json")?).map_err(E::msg)?;
        let image_pad_id = prompt::image_pad_id(&tokenizer)?;
        let te_files = (1..=TEXT_ENCODER_SHARDS)
            .map(|i| {
                files.get(&format!(
                    "text_encoder/model-{i:05}-of-{TEXT_ENCODER_SHARDS:05}.safetensors"
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let text_encoder = TextEncoder::load(&te_files, dtype)?;
        // The vision encoder is small (0.4B) and runs in f32: in bf16 its output
        // drifts ~6% from f32, which the text encoder amplifies ~4x.
        let vision = VisionEncoder::load(&te_files, Dtype::Float32)?;
        let tr_files = (1..=TRANSFORMER_SHARDS)
            .map(|i| {
                files.get(&format!(
                    "transformer/diffusion_pytorch_model-{i:05}-of-{TRANSFORMER_SHARDS:05}.safetensors"
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let transformer = Transformer::load(&tr_files, dtype)?;
        let vae = Vae::load(files.get("vae/diffusion_pytorch_model.safetensors")?, dtype)?;
        mlx_rs::memory::clear_cache()?;
        Ok(Self {
            dtype,
            tokenizer,
            image_pad_id,
            text_encoder,
            vision,
            transformer,
            vae,
        })
    }

    pub fn generate_with(
        &self,
        opts: &GenerateOptions,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Generated> {
        let started = Instant::now();
        let (w, h) = (opts.width, opts.height);
        let (lh, lw) = (h / VAE_SCALE, w / VAE_SCALE);
        let n_tokens = lh * lw;

        // Reference images: each resized to ~1024^2 px at its own aspect ratio,
        // like the reference pipeline, for both the vision encoder and the VAE.
        let refs: Vec<Reference> = opts
            .reference_images
            .iter()
            .map(|img| {
                let (rw, rh) = calculate_dimensions(img.width(), img.height());
                let rgb = image::imageops::resize(
                    img,
                    rw as u32,
                    rh as u32,
                    image::imageops::FilterType::Lanczos3,
                );
                Reference {
                    rgb,
                    grid: (rh / VAE_SCALE, rw / VAE_SCALE),
                }
            })
            .collect();

        // Text (and vision) encoding, then the step-independent prefix.
        let vision: Vec<_> = refs
            .iter()
            .map(|r| self.vision.encode(&r.rgb))
            .collect::<Result<_>>()?;
        let grids: Vec<(usize, usize)> = vision.iter().map(|f| f.grid).collect();
        let pos = prompt::encode(&self.tokenizer, &opts.prompt, &grids)?;
        on_progress(Progress::Encoded {
            tokens: pos.ids.len(),
        });
        let embeds = |p: &EncodedPrompt| -> Result<Array> {
            let images: Vec<ImageEmbeds> = vision
                .iter()
                .zip(&p.image_starts)
                .map(|(f, &start)| ImageEmbeds {
                    start,
                    merged: f.merged.as_dtype(self.dtype).expect("cast"),
                    deepstack: f
                        .deepstack
                        .iter()
                        .map(|d| d.as_dtype(self.dtype).expect("cast"))
                        .collect(),
                })
                .collect();
            let hs = self.text_encoder.forward(&p.ids, &p.positions, &images)?;
            let kept = split_off_front(&hs, p.drop as i32)?;
            kept.eval()?;
            Ok(kept)
        };
        let txt = embeds(&pos)?;
        let neg = if !opts.negative_prompt.is_empty() && opts.guidance_scale > 1.0 {
            let pneg = prompt::encode(&self.tokenizer, &opts.negative_prompt, &grids)?;
            Some((embeds(&pneg)?, pneg))
        } else {
            None
        };
        let encoded = Instant::now();

        // VAE: reference latents, and the init image for img2img.
        let ref_latents: Vec<Array> = refs
            .iter()
            .map(|r| self.encode_image(&r.rgb, r.grid))
            .collect::<Result<_>>()?;
        let segments = |p: &EncodedPrompt| segments(p, self.image_pad_id, &refs);
        let cache = self
            .transformer
            .prefill(&txt, &ref_latents, &segments(&pos)?)?;
        let neg_cache = match &neg {
            Some((t, p)) => Some(self.transformer.prefill(t, &ref_latents, &segments(p)?)?),
            None => None,
        };

        let sigmas = scheduler::sigmas(opts.num_steps, n_tokens);
        let shape = [1, n_tokens, super::LATENT_CHANNELS];
        let noise = Array::from_slice(
            &seeded_noise(opts.seed, [1, lh, lw, 64]),
            &shape.map(|d| d as i32),
        );
        let (mut x, start) = match &opts.init_image {
            None => (noise.as_dtype(self.dtype)?, 0),
            Some(img) => {
                let start = scheduler::start_index(opts.num_steps, opts.strength);
                let init = self.encode_image(&resize_to_fill(img, w as u32, h as u32), (lh, lw))?;
                let sigma = sigmas[start];
                let x = noise.multiply(Array::from_f32(sigma))?.add(
                    init.as_dtype(Dtype::Float32)?
                        .multiply(Array::from_f32(1.0 - sigma))?,
                )?;
                (x.as_dtype(self.dtype)?, start)
            }
        };
        let image_encoded = Instant::now();

        let steps = opts.num_steps - start;
        for (i, step) in (start..opts.num_steps).enumerate() {
            let t = timestep(sigmas[step]);
            let mut v = self.transformer.forward(&x, t, lh, lw, &cache)?;
            if let Some(nc) = &neg_cache {
                // True CFG: v = neg + scale * (pos - neg)
                let nv = self.transformer.forward(&x, t, lh, lw, nc)?;
                v = nv.add(
                    v.subtract(&nv)?
                        .multiply(crate::nn::scalar(opts.guidance_scale as f32, self.dtype)?)?,
                )?;
            }
            // Euler step in f32, like the scheduler.
            let dt = sigmas[step + 1] - sigmas[step];
            x = x
                .as_dtype(Dtype::Float32)?
                .add(v.as_dtype(Dtype::Float32)?.multiply(Array::from_f32(dt))?)?
                .as_dtype(self.dtype)?;
            x.eval()?;
            on_progress(Progress::Step {
                step: i + 1,
                total: steps,
                t: t as f64,
                sigma: sigmas[step + 1] as f64,
            });
        }
        drop((cache, neg_cache));

        let denoised = Instant::now();
        on_progress(Progress::Decoding);
        let rgba = self
            .vae
            .decode(&x.reshape(&[1, lh as i32, lw as i32, 64])?)?;
        let image = to_rgb(&rgba)?;
        mlx_rs::memory::clear_cache()?;

        Ok(Generated {
            image,
            timings: Timings {
                text: encoded - started,
                init_image: image_encoded - encoded,
                denoise: denoised - image_encoded,
                vae: denoised.elapsed(),
            },
        })
    }

    /// RGB image (already at the latent grid's pixel size) -> packed
    /// normalized latents (1, h*w, 64). The VAE reads RGBA; alpha is opaque.
    fn encode_image(&self, img: &image::RgbImage, grid: (usize, usize)) -> Result<Array> {
        let (w, h) = (img.width() as i32, img.height() as i32);
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for p in img.pixels() {
            px.extend(p.0.iter().map(|&c| c as f32 / 127.5 - 1.0));
            px.push(1.0);
        }
        let x = Array::from_slice(&px, &[1, h, w, 4]).as_dtype(self.dtype)?;
        let z = self
            .vae
            .encode(&x)?
            .reshape(&[1, (grid.0 * grid.1) as i32, 64])?;
        z.eval()?;
        Ok(z)
    }
}

/// The transformer's prefix layout: text runs, and each reference image's
/// latent block where the text encoder had its image slots.
fn segments(p: &EncodedPrompt, pad_id: u32, refs: &[Reference]) -> Result<Vec<Segment>> {
    let mask = p.image_pad_mask(pad_id);
    let mut out = Vec::new();
    let mut refs = refs.iter();
    let mut i = 0;
    while i < mask.len() {
        let mut j = i;
        while j < mask.len() && mask[j] == mask[i] {
            j += 1;
        }
        if mask[i] {
            // Adjacent images have no text between them; split by slot count.
            let mut slots = j - i;
            while slots > 0 {
                let r = refs
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("image slots without an image"))?;
                let (gh, gw) = r.grid;
                let n = gh * gw / TOKENS_PER_SLOT;
                anyhow::ensure!(n <= slots, "image slots do not match the reference image");
                out.push(Segment::Image { h: gh, w: gw });
                slots -= n;
            }
        } else {
            out.push(Segment::Text(j - i));
        }
        i = j;
    }
    Ok(out)
}

/// The transformer's input time for a sigma, rounded like the reference:
/// `t = bf16(sigma * 1000)`, then `bf16(t / 1000)`.
fn timestep(sigma: f32) -> f32 {
    to_bf16(to_bf16(sigma * 1000.0) / 1000.0)
}

/// Rounds to the nearest bf16 value (ties to even).
fn to_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    let rounded = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
    f32::from_bits(rounded & 0xFFFF_0000)
}

/// Target size for a reference image: area ~1024^2 at its aspect ratio,
/// sides rounded to multiples of 32 (`calculate_dimensions`).
pub fn calculate_dimensions(width: u32, height: u32) -> (usize, usize) {
    let ratio = width as f64 / height as f64;
    let area = (OUTPUT_RESOLUTION * OUTPUT_RESOLUTION) as f64;
    let w = (area * ratio).sqrt();
    let h = w / ratio;
    let round32 = |v: f64| ((v / 32.0).round() as usize * 32).max(32);
    (round32(w), round32(h))
}

/// Drops the first `n` tokens along axis 1.
fn split_off_front(x: &Array, n: i32) -> Result<Array> {
    Ok(mlx_rs::ops::split_at_indices(x, &[n], 1)?
        .pop()
        .expect("tail"))
}

/// RGBA in [-1, 1] (1, H, W, 4) -> RGB image, rounding like diffusers'
/// `postprocess`. The alpha channel is dropped (it is ~opaque for photos).
fn to_rgb(rgba: &Array) -> Result<image::RgbImage> {
    let sh = rgba.shape().to_vec();
    let (h, w) = (sh[1] as u32, sh[2] as u32);
    let x = rgba
        .as_dtype(Dtype::Float32)?
        .multiply(Array::from_f32(0.5))?
        .add(Array::from_f32(0.5))?;
    let x = mlx_rs::ops::clip(&x, (0.0f32, 1.0f32))?
        .multiply(Array::from_f32(255.0))?
        .round(None)?;
    let rgb = mlx_rs::ops::split_at_indices(&x, &[3], -1)?
        .swap_remove(0)
        .as_dtype(Dtype::Uint8)?
        .contiguous()?;
    rgb.eval()?;
    image::RgbImage::from_raw(w, h, rgb.as_slice::<u8>().to_vec())
        .ok_or_else(|| anyhow::anyhow!("image buffer size mismatch"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_matches_reference_timesteps() {
        // Values recorded from the reference pipeline (8 steps, 1024 tokens).
        let s = scheduler::sigmas(8, 1024);
        let t: Vec<f32> = s[..8].iter().map(|&x| timestep(x)).collect();
        assert_eq!(
            t,
            [
                1.0,
                0.90625,
                0.80078125,
                0.68359375,
                0.55078125,
                0.3984375,
                0.2236328125,
                0.02001953125
            ]
        );
    }

    #[test]
    fn calculate_dimensions_keeps_area_and_aspect() {
        assert_eq!(calculate_dimensions(512, 512), (1024, 1024));
        assert_eq!(calculate_dimensions(1920, 1080), (1376, 768));
        let (w, h) = calculate_dimensions(600, 400);
        assert_eq!((w % 32, h % 32), (0, 0));
    }
}

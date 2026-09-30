//! The end-to-end text-to-image pipeline.

use crate::zimage::{
    scalar, scheduler::Scheduler, text_encoder::TextEncoder, transformer::Transformer,
    vae::Decoder, ModelFiles,
};
use anyhow::{Error as E, Result};
use mlx_rs::{Array, Dtype};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// Hugging Face repo of the weights.
pub const DEFAULT_REPO: &str = "Tongyi-MAI/Z-Image-Turbo";
/// Denoising steps Z-Image-Turbo is tuned for.
pub const DEFAULT_STEPS: usize = 9;
/// Width and height must be multiples of this (VAE 8x * patch size 2).
pub const SIZE_ALIGN: usize = 16;

/// Where to load the weights from, and on which device.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Hugging Face repo id, used when `model_path` is `None`.
    pub repo: String,
    /// A local copy of the repo (with `tokenizer/`, `text_encoder/`, ...).
    pub model_path: Option<PathBuf>,
    /// Run on the CPU instead of the GPU.
    pub cpu: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            repo: DEFAULT_REPO.into(),
            model_path: None,
            cpu: false,
        }
    }
}

/// What to generate.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOptions {
    pub prompt: String,
    /// Used for classifier-free guidance when non-empty and `guidance_scale > 1`.
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub num_steps: usize,
    pub guidance_scale: f64,
    /// Seeds the initial noise; the same seed and options give the same image.
    pub seed: u64,
}

impl GenerateOptions {
    /// 1024x1024, 9 steps, seed 0.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: String::new(),
            width: 1024,
            height: 1024,
            num_steps: DEFAULT_STEPS,
            guidance_scale: 5.0,
            seed: 0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let a = SIZE_ALIGN;
        anyhow::ensure!(!self.prompt.trim().is_empty(), "prompt is empty");
        anyhow::ensure!(self.num_steps > 0, "num_steps must be at least 1");
        if !self.height.is_multiple_of(a) || !self.width.is_multiple_of(a) {
            anyhow::bail!(
                "Image dimensions must be divisible by {a}. Got {}x{}. Try {}x{} or {}x{} instead.",
                self.width,
                self.height,
                (self.width / a) * a,
                (self.height / a) * a,
                ((self.width / a) + 1) * a,
                ((self.height / a) + 1) * a
            );
        }
        Ok(())
    }
}

/// Progress reported by [`Pipeline::generate_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Progress {
    /// The prompt was encoded into this many tokens.
    Encoded { tokens: usize },
    /// A denoising step finished (1-based `step`).
    Step {
        step: usize,
        total: usize,
        t: f64,
        sigma: f64,
    },
    /// Denoising is done and the VAE is decoding the image.
    Decoding,
}

/// Wall-clock time spent in each phase.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub text: Duration,
    pub denoise: Duration,
    pub vae: Duration,
}

/// A generated image and how long it took.
pub struct Generated {
    pub image: image::RgbImage,
    pub timings: Timings,
}

/// The loaded models, reusable across any number of images.
///
/// ```no_run
/// let pipeline = mixel::Pipeline::load(&Default::default())?;
/// let mut opts = mixel::GenerateOptions::new("a red fox in fresh snow");
/// opts.seed = 42;
/// pipeline.generate(&opts)?.image.save("fox.png")?;
/// # anyhow::Ok(())
/// ```
pub struct Pipeline {
    dtype: Dtype,
    tokenizer: Tokenizer,
    text_encoder: TextEncoder,
    transformer: Transformer,
    vae: Decoder,
}

impl Pipeline {
    /// Loads the tokenizer, text encoder, transformer and VAE (downloading the
    /// ~33 GB of weights on first use).
    pub fn load(opts: &LoadOptions) -> Result<Self> {
        if opts.cpu {
            mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        }
        let dtype = Dtype::Bfloat16;
        let files = ModelFiles::new(&opts.repo, opts.model_path.as_deref())?;

        let tokenizer =
            Tokenizer::from_file(files.get("tokenizer/tokenizer.json")?).map_err(E::msg)?;
        let te_files = (1..=3)
            .map(|i| files.get(&format!("text_encoder/model-{i:05}-of-00003.safetensors")))
            .collect::<Result<Vec<_>>>()?;
        let text_encoder = TextEncoder::load(&te_files, dtype)?;
        let tr_files = (1..=3)
            .map(|i| {
                files.get(&format!(
                    "transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors"
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let transformer = Transformer::load(&tr_files, dtype)?;
        let vae = Decoder::load(files.get("vae/diffusion_pytorch_model.safetensors")?, dtype)?;
        // Drop the buffers left over from converting f32 weights to bf16.
        mlx_rs::memory::clear_cache()?;

        Ok(Self {
            dtype,
            tokenizer,
            text_encoder,
            transformer,
            vae,
        })
    }

    pub fn generate(&self, opts: &GenerateOptions) -> Result<Generated> {
        self.generate_with(opts, |_| {})
    }

    /// Like [`generate`](Self::generate), calling `on_progress` as it goes.
    pub fn generate_with(
        &self,
        opts: &GenerateOptions,
        mut on_progress: impl FnMut(Progress),
    ) -> Result<Generated> {
        opts.validate()?;
        let dtype = self.dtype;
        let scalar = |v: f64| scalar(v as f32, dtype);

        let started = Instant::now();
        let cap_feats = self.encode_prompt(&opts.prompt, &mut on_progress)?;
        let neg_cap_feats = if !opts.negative_prompt.is_empty() && opts.guidance_scale > 1.0 {
            Some(self.encode_prompt(&opts.negative_prompt, &mut |_| {})?)
        } else {
            None
        };

        // latent = 2 * (image_size // 16): divisible by the patch size, and 8x VAE upsampling.
        let shape = [1, 16, 2 * (opts.height / 16), 2 * (opts.width / 16)];
        let noise = seeded_noise(opts.seed, shape);
        let mut latents = Array::from_slice(&noise, &shape.map(|d| d as i32)).as_dtype(dtype)?;

        let encoded = Instant::now();
        let mut scheduler = Scheduler::new(opts.num_steps);
        for step in 0..opts.num_steps {
            let t = scheduler.current_timestep_normalized();
            let mut pred = self.transformer.forward(&latents, t as f32, &cap_feats)?;
            if let Some(neg) = &neg_cap_feats {
                // CFG: pred = neg + scale * (pos - neg)
                let neg_pred = self.transformer.forward(&latents, t as f32, neg)?;
                pred = neg_pred.add(
                    pred.subtract(&neg_pred)?
                        .multiply(scalar(opts.guidance_scale)?)?,
                )?;
            }
            // Z-Image predicts the negated velocity; Euler step: x + dt * v.
            let dt = scheduler.step_dt();
            latents = latents.add(pred.negative()?.multiply(scalar(dt)?)?)?;
            latents.eval()?;
            on_progress(Progress::Step {
                step: step + 1,
                total: opts.num_steps,
                t,
                sigma: scheduler.current_sigma(),
            });
        }

        let denoised = Instant::now();
        on_progress(Progress::Decoding);
        let image = self.vae.decode(&latents.transpose_axes(&[0, 2, 3, 1])?)?;
        // [-1, 1] -> [0, 255], computed in the model dtype like candle.
        let image = mlx_rs::ops::clip(&image, (-1.0f32, 1.0f32))?
            .add(scalar(1.0)?)?
            .multiply(scalar(127.5)?)?
            .as_dtype(Dtype::Uint8)?;
        let image = to_rgb_image(&image)?;
        // MLX keeps freed buffers for reuse. Between images (often of different
        // sizes) they only pile up: in a mixed-size batch the cache grew to
        // ~75 GB and steps slowed ~1.5x under memory pressure.
        mlx_rs::memory::clear_cache()?;

        Ok(Generated {
            image,
            timings: Timings {
                text: encoded - started,
                denoise: denoised - encoded,
                vae: denoised.elapsed(),
            },
        })
    }

    fn encode_prompt(&self, prompt: &str, on_progress: &mut impl FnMut(Progress)) -> Result<Array> {
        let tokens = self
            .tokenizer
            .encode(format_prompt_for_qwen3(prompt).as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        on_progress(Progress::Encoded {
            tokens: tokens.len(),
        });
        let feats = self.text_encoder.forward(&tokens)?;
        feats.eval()?;
        Ok(feats)
    }
}

/// Standard normal noise from a seeded CPU RNG. Identical to candy's, so the
/// same seed starts both tools from the same latents.
pub fn seeded_noise(seed: u64, shape: [usize; 4]) -> Vec<f32> {
    use rand::SeedableRng;
    use rand_distr::Distribution;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    rand_distr::StandardNormal
        .sample_iter(&mut rng)
        .take(shape.iter().product())
        .collect()
}

/// Qwen3 chat template (add_generation_prompt=True, enable_thinking=True):
/// `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// (1, H, W, 3) u8 array -> RgbImage.
fn to_rgb_image(img: &Array) -> Result<image::RgbImage> {
    let sh = img.shape();
    anyhow::ensure!(
        sh.len() == 4 && sh[0] == 1 && sh[3] == 3,
        "expected a (1, H, W, 3) image, got {sh:?}"
    );
    let (height, width) = (sh[1] as u32, sh[2] as u32);
    let img = img.contiguous()?;
    img.eval()?;
    image::ImageBuffer::from_raw(width, height, img.as_slice::<u8>().to_vec())
        .ok_or_else(|| anyhow::anyhow!("image buffer size mismatch"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(width: usize, height: usize) -> GenerateOptions {
        GenerateOptions {
            width,
            height,
            ..GenerateOptions::new("a cat")
        }
    }

    #[test]
    fn validate_accepts_multiples_of_16() {
        for (w, h) in [(1024, 1024), (512, 512), (384, 512), (640, 368), (16, 16)] {
            opts(w, h).validate().unwrap();
        }
    }

    #[test]
    fn validate_rejects_bad_dimensions_with_suggestion() {
        let err = opts(1000, 1024).validate().unwrap_err().to_string();
        assert!(err.contains("divisible by 16"), "{err}");
        assert!(err.contains("992x1024"), "{err}");
    }

    #[test]
    fn validate_rejects_empty_prompt_and_zero_steps() {
        let mut o = opts(512, 512);
        o.prompt = "   ".into();
        assert!(o
            .validate()
            .unwrap_err()
            .to_string()
            .contains("prompt is empty"));

        let mut o = opts(512, 512);
        o.num_steps = 0;
        assert!(o.validate().unwrap_err().to_string().contains("num_steps"));
    }

    #[test]
    fn new_uses_turbo_defaults() {
        let o = GenerateOptions::new("x");
        assert_eq!(
            (o.width, o.height, o.num_steps, o.seed),
            (1024, 1024, DEFAULT_STEPS, 0)
        );
        assert!(o.negative_prompt.is_empty());
    }

    #[test]
    fn seeded_noise_is_reproducible() {
        let shape = [1, 16, 8, 8];
        assert_eq!(seeded_noise(1, shape), seeded_noise(1, shape));
        assert_ne!(seeded_noise(1, shape), seeded_noise(2, shape));
        assert_eq!(seeded_noise(1, shape).len(), 16 * 64);
    }

    #[test]
    fn seeded_noise_is_standard_normal() {
        let v = seeded_noise(3, [1, 16, 64, 64]);
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((std - 1.0).abs() < 0.02, "std {std}");
    }

    #[test]
    fn prompt_uses_qwen3_chat_template() {
        assert_eq!(
            format_prompt_for_qwen3("hi"),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn to_rgb_image_checks_shape_and_copies_pixels() {
        let px: Vec<u8> = (0..2 * 3 * 3).map(|i| i as u8).collect();
        let a = Array::from_slice(&px, &[1, 2, 3, 3]);
        let img = to_rgb_image(&a).unwrap();
        assert_eq!(img.dimensions(), (3, 2));
        assert_eq!(img.get_pixel(1, 0).0, [3, 4, 5]);
        assert!(to_rgb_image(&Array::from_slice(&px, &[2, 3, 3])).is_err());
    }
}

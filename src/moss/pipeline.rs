//! MOSS-Image generation: text-to-image and img2img (ports
//! `MOSSImagePipeline.__call__`), and text-to-image in 4 steps with the
//! rapid-moss adapter (see [`super::fast`]).

use super::fast::Adapter;
use super::text_encoder::TextEncoder;
use super::transformer::Transformer;
use super::vae::Vae;
use super::{prompt, scheduler, LATENT_CHANNELS, SIZE_ALIGN, UNCOND_SKIP_LAYER};
use crate::lora::Lora;
use crate::nn::{ModelFiles, Quantize, WeightCache};
use crate::pipeline::{
    image_to_array, resize_to_fill, seeded_noise, to_pixels, to_rgb_image, GenerateOptions,
    Generated, Parts, Progress, Timings,
};
use anyhow::{Context, Error as E, Result};
use mlx_rs::{Array, Dtype};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const TEXT_ENCODER_SHARDS: usize = 4;
const TRANSFORMER_SHARDS: usize = 3;

/// MOSS-Image's models, or the [`Parts`] of them that were loaded.
pub struct MossPipeline {
    tokenizer: Tokenizer,
    text_encoder: Option<TextEncoder>,
    transformer: Option<Transformer>,
    vae: Option<Vae>,
    /// With the 4-step adapter: its latent mix and step codes.
    fast: Option<(Array, Vec<f32>)>,
}

/// One image's prompt embeddings, from [`MossPipeline::encode`].
pub struct Encoded {
    txt: Array,
    /// With guidance: the negative prompt's embeddings (the empty prompt's
    /// if there is none, like the reference).
    neg: Option<Array>,
    pub tokens: usize,
    /// Whether the prompt was cut to [`super::MAX_PROMPT_TOKENS`].
    pub truncated: bool,
    pub took: Duration,
}

impl Encoded {
    /// Memory the embeddings take.
    pub fn nbytes(&self) -> usize {
        self.txt.nbytes() + self.neg.as_ref().map_or(0, Array::nbytes)
    }
}

/// Whether `guidance_scale` turns on guidance: anything but 1, like the
/// reference.
pub fn guided(guidance_scale: f64) -> bool {
    guidance_scale != 1.0
}

impl MossPipeline {
    /// Loads the tokenizer and the given `parts`: the text encoder, and the
    /// transformer (with the 4-step adapter applied if given) and VAE, with
    /// the text encoder and transformer quantized if `quantize` is set
    /// (through `cache` when given), and `loras` added to the transformer.
    pub fn load(
        files: &ModelFiles,
        adapter: Option<&ModelFiles>,
        quantize: Option<Quantize>,
        parts: Parts,
        cache: Option<&WeightCache>,
        loras: &[Lora],
    ) -> Result<Self> {
        let dtype = Dtype::Bfloat16;
        let tokenizer =
            Tokenizer::from_file(files.get("tokenizer/tokenizer.json")?).map_err(E::msg)?;
        let text_encoder = if parts.encoders() {
            let shards = (1..=TEXT_ENCODER_SHARDS)
                .map(|i| {
                    files.get(&format!(
                        "text_encoder/model-{i:05}-of-{TEXT_ENCODER_SHARDS:05}.safetensors"
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Some(TextEncoder::load(&shards, dtype, quantize, cache)?)
        } else {
            None
        };
        let (transformer, vae, fast) = if parts.generator() {
            let shards = (1..=TRANSFORMER_SHARDS)
                .map(|i| {
                    files.get(&format!(
                        "transformer/diffusion_pytorch_model-{i:05}-of-{TRANSFORMER_SHARDS:05}.safetensors"
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut transformer = Transformer::load(&shards, dtype, quantize, cache)?;
            let fast = match adapter {
                Some(files) => {
                    let adapter = Adapter::load(files)?;
                    transformer.apply_adapter(&adapter)?;
                    Some((adapter.q, adapter.step_codes))
                }
                None => None,
            };
            for lora in loras {
                super::fast::check_user_lora(lora)?;
                for u in lora.updates(dtype)? {
                    transformer
                        .add_lora(&u.layer, u.down, u.up)
                        .with_context(|| {
                            format!(
                                "{}: can't apply its update to layer {}",
                                lora.path.display(),
                                u.layer
                            )
                        })?;
                }
            }
            // The VAE runs in f32, like the reference.
            let vae = Vae::load(files.get("vae/diffusion_pytorch_model.safetensors")?)?;
            (Some(transformer), Some(vae), fast)
        } else {
            (None, None, None)
        };
        mlx_rs::memory::clear_cache()?;
        Ok(Self {
            tokenizer,
            text_encoder,
            transformer,
            vae,
            fast,
        })
    }

    pub fn generate_with(
        &self,
        opts: &GenerateOptions,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Generated> {
        let encoded = self.encode(opts)?;
        self.generate_encoded(opts, &encoded, on_progress)
    }

    /// The prompt's embeddings, and with guidance the negative prompt's.
    /// Needs the text encoder.
    pub fn encode(&self, opts: &GenerateOptions) -> Result<Encoded> {
        let text_encoder = self
            .text_encoder
            .as_ref()
            .context("this pipeline was loaded without its text encoder")?;
        let started = Instant::now();
        let embed = |text: &str| -> Result<(Array, prompt::EncodedPrompt)> {
            let p = prompt::encode(&self.tokenizer, text)?;
            let hs = text_encoder.forward(&p.ids)?;
            hs.eval()?;
            Ok((hs, p))
        };
        let (txt, p) = embed(&opts.prompt)?;
        let neg = if guided(opts.guidance_scale) {
            Some(embed(&opts.negative_prompt)?.0)
        } else {
            None
        };
        Ok(Encoded {
            txt,
            neg,
            tokens: p.ids.len(),
            truncated: p.truncated,
            took: started.elapsed(),
        })
    }

    /// Generates from [`encode`](Self::encode)'s output for the same
    /// options. Needs the transformer and VAE.
    pub fn generate_encoded(
        &self,
        opts: &GenerateOptions,
        encoded: &Encoded,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Generated> {
        let (transformer, vae) = match (&self.transformer, &self.vae) {
            (Some(t), Some(v)) => (t, v),
            _ => anyhow::bail!("this pipeline was loaded without its transformer and VAE"),
        };
        on_progress(Progress::Encoded {
            tokens: encoded.tokens,
        });
        let started = Instant::now();
        let (lh, lw) = (opts.height / SIZE_ALIGN, opts.width / SIZE_ALIGN);
        let noise = Array::from_slice(
            &seeded_noise(opts.seed, [1, lh, lw, LATENT_CHANNELS]),
            &[1, (lh * lw) as i32, LATENT_CHANNELS as i32],
        );
        let grid = (lh as i32, lw as i32);
        // Latents, and when the setup before the first step finished.
        let (x, set_up) = match &self.fast {
            Some((q, codes)) => sample_fast(
                transformer,
                opts,
                encoded,
                noise,
                grid,
                q,
                codes,
                on_progress,
            )?,
            None => sample(transformer, vae, opts, encoded, noise, grid, on_progress)?,
        };
        let denoised = Instant::now();
        on_progress(Progress::Decoding);
        let image = to_rgb(&vae.decode(&x, grid.0, grid.1)?)?;
        Ok(Generated {
            image: image.into(),
            timings: Timings {
                text: encoded.took,
                init_image: set_up - started,
                denoise: denoised - set_up,
                vae: denoised.elapsed(),
            },
        })
    }
}

/// The base model's sampler: flow-matching Euler from noise (or from the init
/// image at `strength`), guided with the negative branch skipping a block.
/// The latents stay in f32; the model reads bf16.
#[allow(clippy::too_many_arguments)]
fn sample(
    transformer: &Transformer,
    vae: &Vae,
    opts: &GenerateOptions,
    encoded: &Encoded,
    noise: Array,
    (lh, lw): (i32, i32),
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(Array, Instant)> {
    anyhow::ensure!(
        guided(opts.guidance_scale) == encoded.neg.is_some(),
        "the prompt was encoded with guidance {}, but generation has it {}",
        if encoded.neg.is_some() { "on" } else { "off" },
        if guided(opts.guidance_scale) {
            "on"
        } else {
            "off"
        }
    );
    let sigmas = scheduler::sigmas(opts.num_steps);
    let n = sigmas.len() - 1;
    let (mut x, start) = match &opts.init_image {
        None => (noise, 0),
        Some(img) => {
            let start = scheduler::start_index(n, opts.strength);
            let init = encode_image(vae, img, opts.width, opts.height)?;
            // Flow matching: x_sigma = sigma * noise + (1 - sigma) * x_0.
            let sigma = sigmas[start];
            let x = noise
                .multiply(Array::from_f32(sigma))?
                .add(init.multiply(Array::from_f32(1.0 - sigma))?)?;
            (x, start)
        }
    };
    let cond = transformer.condition(&encoded.txt, lh, lw)?;
    let uncond = match &encoded.neg {
        Some(neg) => Some(transformer.condition(neg, lh, lw)?),
        None => None,
    };
    let set_up = Instant::now();

    let scale = Array::from_f32(opts.guidance_scale as f32);
    for (i, step) in (start..n).enumerate() {
        let m = transformer.modulation(scheduler::timestep(sigmas[step]))?;
        let mut v = transformer
            .forward_with(&x, &m, &cond, None)?
            .as_dtype(Dtype::Float32)?;
        if let Some(uncond) = &uncond {
            // The negative branch skips a block, as the reference's default
            // guidance does: v = neg + scale * (pos - neg).
            let nv = transformer
                .forward_with(&x, &m, uncond, Some(UNCOND_SKIP_LAYER))?
                .as_dtype(Dtype::Float32)?;
            v = nv.add(scale.multiply(v.subtract(&nv)?)?)?;
        }
        let dt = sigmas[step + 1] - sigmas[step];
        x = x.add(v.multiply(Array::from_f32(dt))?)?;
        x.eval()?;
        on_progress(Progress::Step {
            step: i + 1,
            total: n - start,
            t: sigmas[step] as f64,
            sigma: sigmas[step + 1] as f64,
        });
    }
    Ok((x, set_up))
}

/// The adapter's sampler: from one noise `z`, each step predicts the image
/// `x0 = z - v(z + Q x0)` at the next step code, without guidance.
#[allow(clippy::too_many_arguments)]
fn sample_fast(
    transformer: &Transformer,
    opts: &GenerateOptions,
    encoded: &Encoded,
    z: Array,
    (lh, lw): (i32, i32),
    q: &Array,
    codes: &[f32],
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(Array, Instant)> {
    anyhow::ensure!(
        (1..=codes.len()).contains(&opts.num_steps),
        "the 4-step adapter runs 1 to {} steps, got {}",
        codes.len(),
        opts.num_steps
    );
    let cond = transformer.condition(&encoded.txt, lh, lw)?;
    let set_up = Instant::now();
    let mut x0 = Array::zeros::<f32>(z.shape())?;
    for r in 0..opts.num_steps {
        let x = z.add(x0.matmul(q.t())?)?;
        let v = transformer
            .forward(&x, codes[r], &cond, None)?
            .as_dtype(Dtype::Float32)?;
        x0 = z.subtract(v)?;
        x0.eval()?;
        let next = codes.get(r + 1).filter(|_| r + 1 < opts.num_steps);
        on_progress(Progress::Step {
            step: r + 1,
            total: opts.num_steps,
            t: codes[r] as f64 / 1000.0,
            sigma: next.map_or(0.0, |&c| c as f64 / 1000.0),
        });
    }
    Ok((x0, set_up))
}

/// RGB image -> normalized latent tokens (1, h * w, 128) in f32, resized to
/// `width` x `height`.
fn encode_image(vae: &Vae, img: &image::RgbImage, width: usize, height: usize) -> Result<Array> {
    let img = resize_to_fill(img, width as u32, height as u32);
    let z = vae.encode(&image_to_array(&img))?;
    z.eval()?;
    Ok(z)
}

/// (1, H, W, 3) in [-1, 1] -> RGB, rounding like diffusers' `postprocess`.
fn to_rgb(x: &Array) -> Result<image::RgbImage> {
    to_rgb_image(&to_pixels(x)?)
}

//! Z-Image-Turbo generation (text-to-image and img2img).

use super::{
    scalar, scheduler::Scheduler, text_encoder::TextEncoder, transformer::Transformer, vae::Vae,
    ModelFiles, Quantize, WeightCache,
};
use crate::lora::Lora;
use crate::pipeline::{
    resize_to_fill, seeded_noise, to_rgb_image, GenerateOptions, Generated, Parts, Progress,
    Timings,
};
use anyhow::{Context, Error as E, Result};
use mlx_rs::{Array, Dtype};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// Prompts are cut to this many tokens, like diffusers' `max_sequence_length`
/// (the transformer's RoPE tables can't go much further).
pub const MAX_PROMPT_TOKENS: usize = 512;

/// Z-Image-Turbo's models, or the [`Parts`] of them that were loaded.
pub struct ZImagePipeline {
    dtype: Dtype,
    tokenizer: Tokenizer,
    text_encoder: Option<TextEncoder>,
    transformer: Option<Transformer>,
    vae: Option<Vae>,
}

/// One image's caption features, from [`ZImagePipeline::encode`].
pub struct Encoded {
    cap: Array,
    /// With guidance: the negative (or empty) prompt's features.
    neg: Option<Array>,
    /// Prompt tokens (at most [`MAX_PROMPT_TOKENS`]).
    pub tokens: usize,
    pub took: Duration,
}

impl Encoded {
    /// Memory the features take.
    pub fn nbytes(&self) -> usize {
        self.cap.nbytes() + self.neg.as_ref().map_or(0, |n| n.nbytes())
    }
}

impl ZImagePipeline {
    /// Loads the tokenizer and the given `parts`: the text encoder, and the
    /// transformer and VAE, with the text encoder and transformer quantized
    /// if `quantize` is set.
    /// Quantized weights go through `cache` when given; `loras` are added to
    /// the transformer.
    pub fn load(
        files: &ModelFiles,
        quantize: Option<Quantize>,
        parts: Parts,
        cache: Option<&WeightCache>,
        loras: &[Lora],
    ) -> Result<Self> {
        let dtype = Dtype::Bfloat16;

        let tokenizer =
            Tokenizer::from_file(files.get("tokenizer/tokenizer.json")?).map_err(E::msg)?;
        let text_encoder = if parts.encoders() {
            let te_files = (1..=3)
                .map(|i| files.get(&format!("text_encoder/model-{i:05}-of-00003.safetensors")))
                .collect::<Result<Vec<_>>>()?;
            Some(TextEncoder::load(&te_files, dtype, quantize, cache)?)
        } else {
            None
        };
        let (transformer, vae) = if parts.generator() {
            let tr_files = (1..=3)
                .map(|i| {
                    files.get(&format!(
                        "transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors"
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut transformer = Transformer::load(&tr_files, dtype, quantize, cache)?;
            for lora in loras {
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
            let vae = Vae::load(files.get("vae/diffusion_pytorch_model.safetensors")?, dtype)?;
            (Some(transformer), Some(vae))
        } else {
            (None, None)
        };
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

    pub fn generate_with(
        &self,
        opts: &GenerateOptions,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Generated> {
        let encoded = self.encode(opts)?;
        self.generate_encoded(opts, &encoded, on_progress)
    }

    /// The prompt's caption features, and with guidance the negative
    /// prompt's. Needs the text encoder.
    pub fn encode(&self, opts: &GenerateOptions) -> Result<Encoded> {
        let started = Instant::now();
        let (cap, tokens) = self.encode_prompt(&opts.prompt)?;
        // Like diffusers, guidance runs whenever the scale is positive, against
        // the negative prompt or, without one, the empty prompt.
        let neg = if opts.guidance_scale > 0.0 {
            Some(self.encode_prompt(&opts.negative_prompt)?.0)
        } else {
            None
        };
        Ok(Encoded {
            cap,
            neg,
            tokens,
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
        let dtype = self.dtype;
        let (transformer, vae) = match (&self.transformer, &self.vae) {
            (Some(t), Some(v)) => (t, v),
            _ => anyhow::bail!("this pipeline was loaded without its transformer and VAE"),
        };
        on_progress(Progress::Encoded {
            tokens: encoded.tokens,
        });

        let started = Instant::now();
        // latent = 2 * (image_size // 16): divisible by the patch size, and 8x VAE upsampling.
        let shape = [1, 16, 2 * (opts.height / 16), 2 * (opts.width / 16)];
        let noise = Array::from_slice(&seeded_noise(opts.seed, shape), &shape.map(|d| d as i32));

        // Like diffusers, the latents stay in f32 and the model reads bf16.
        let mut scheduler = Scheduler::new(opts.num_steps);
        let (mut latents, steps) = match &opts.init_image {
            None => (noise, opts.num_steps),
            Some(img) => {
                let steps = scheduler.skip_for_strength(opts.strength);
                let init = self.encode_image(vae, img, opts.width, opts.height)?;
                // Flow matching: x_sigma = sigma * noise + (1 - sigma) * x_0.
                // At strength 1, sigma is exactly 1 and this is plain noise.
                let sigma = scheduler.current_sigma();
                let x = noise.multiply(Array::from_f32(sigma))?.add(
                    init.as_dtype(Dtype::Float32)?
                        .multiply(Array::from_f32(1.0 - sigma))?,
                )?;
                (x, steps)
            }
        };
        let image_encoded = Instant::now();

        for step in 0..steps {
            let t = scheduler.current_timestep_normalized();
            let x = latents.as_dtype(dtype)?;
            let mut pred = transformer
                .forward(&x, t, &encoded.cap)?
                .as_dtype(Dtype::Float32)?;
            if let Some(neg) = &encoded.neg {
                let neg_pred = transformer.forward(&x, t, neg)?.as_dtype(Dtype::Float32)?;
                pred = guide(&pred, &neg_pred, opts.guidance_scale as f32)?;
            }
            // Z-Image predicts the negated velocity; Euler step: x + dt * v.
            let dt = scheduler.step_dt();
            latents = latents.add(pred.negative()?.multiply(Array::from_f32(dt))?)?;
            latents.eval()?;
            on_progress(Progress::Step {
                step: step + 1,
                total: steps,
                t: t as f64,
                sigma: scheduler.current_sigma() as f64,
            });
        }

        let denoised = Instant::now();
        on_progress(Progress::Decoding);
        let image = vae.decode(&latents.as_dtype(dtype)?.transpose_axes(&[0, 2, 3, 1])?)?;
        // [-1, 1] -> [0, 255], computed in the model dtype like candle.
        let image = mlx_rs::ops::clip(&image, (-1.0f32, 1.0f32))?
            .add(scalar(1.0, dtype)?)?
            .multiply(scalar(127.5, dtype)?)?
            .as_dtype(Dtype::Uint8)?;
        let image = to_rgb_image(&image)?;

        Ok(Generated {
            image: image.into(),
            timings: Timings {
                text: encoded.took,
                init_image: image_encoded - started,
                denoise: denoised - image_encoded,
                vae: denoised.elapsed(),
            },
        })
    }

    /// RGB image -> (1, 16, H/8, W/8) latents, resized to `width` x `height`.
    fn encode_image(
        &self,
        vae: &Vae,
        img: &image::RgbImage,
        width: usize,
        height: usize,
    ) -> Result<Array> {
        let (w, h) = (width as u32, height as u32);
        let img = resize_to_fill(img, w, h);
        // [0, 255] -> [-1, 1], NHWC.
        let pixels: Vec<f32> = img
            .as_raw()
            .iter()
            .map(|&p| p as f32 / 127.5 - 1.0)
            .collect();
        let x = Array::from_slice(&pixels, &[1, h as i32, w as i32, 3]).as_dtype(self.dtype)?;
        let z = vae.encode(&x)?.transpose_axes(&[0, 3, 1, 2])?;
        z.eval()?;
        Ok(z)
    }

    /// Caption features (1, tokens, 2560) and the token count.
    fn encode_prompt(&self, prompt: &str) -> Result<(Array, usize)> {
        let text_encoder = self
            .text_encoder
            .as_ref()
            .context("this pipeline was loaded without its text encoder")?;
        let tokens = self
            .tokenizer
            .encode(format_prompt_for_qwen3(prompt).as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .iter()
            .copied()
            .take(MAX_PROMPT_TOKENS)
            .collect::<Vec<_>>();
        let feats = text_encoder.forward(&tokens)?;
        feats.eval()?;
        Ok((feats, tokens.len()))
    }
}

/// Classifier-free guidance as diffusers' `ZImagePipeline` does it:
/// `pos + scale * (pos - neg)`, so 0 is no guidance.
pub fn guide(pos: &Array, neg: &Array, scale: f32) -> Result<Array> {
    Ok(pos.add(pos.subtract(neg)?.multiply(Array::from_f32(scale))?)?)
}

/// Qwen3 chat template (add_generation_prompt=True, enable_thinking=True):
/// `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
pub fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_uses_qwen3_chat_template() {
        assert_eq!(
            format_prompt_for_qwen3("hi"),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn guidance_pushes_away_from_the_negative_prediction() {
        crate::nn::test_device();
        let pos = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let neg = Array::from_slice(&[0.5f32, 3.0], &[2]);
        let at = |scale: f32| {
            let g = guide(&pos, &neg, scale).unwrap();
            g.eval().unwrap();
            g.as_slice::<f32>().to_vec()
        };
        assert_eq!(at(0.0), [1.0, 2.0]); // 0: the prompt's prediction alone
        assert_eq!(at(2.0), [2.0, 0.0]); // pos + 2 * (pos - neg)
    }
}

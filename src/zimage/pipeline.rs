//! Z-Image-Turbo generation (text-to-image and img2img).

use super::{
    scalar, scheduler::Scheduler, text_encoder::TextEncoder, transformer::Transformer, vae::Vae,
    ModelFiles,
};
use crate::pipeline::{
    resize_to_fill, seeded_noise, to_rgb_image, GenerateOptions, Generated, Progress, Timings,
};
use anyhow::{Error as E, Result};
use mlx_rs::{Array, Dtype};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Prompts are cut to this many tokens, like diffusers' `max_sequence_length`
/// (the transformer's RoPE tables can't go much further).
pub const MAX_PROMPT_TOKENS: usize = 512;

/// Z-Image-Turbo's models.
pub struct ZImagePipeline {
    dtype: Dtype,
    tokenizer: Tokenizer,
    text_encoder: TextEncoder,
    transformer: Transformer,
    vae: Vae,
}

impl ZImagePipeline {
    /// Loads the tokenizer, text encoder, transformer and VAE.
    pub fn load(files: &ModelFiles) -> Result<Self> {
        let dtype = Dtype::Bfloat16;

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
        let vae = Vae::load(files.get("vae/diffusion_pytorch_model.safetensors")?, dtype)?;
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
        let dtype = self.dtype;

        let started = Instant::now();
        let cap_feats = self.encode_prompt(&opts.prompt, on_progress)?;
        let neg_cap_feats = if !opts.negative_prompt.is_empty() && opts.guidance_scale > 1.0 {
            Some(self.encode_prompt(&opts.negative_prompt, &mut |_| {})?)
        } else {
            None
        };

        // latent = 2 * (image_size // 16): divisible by the patch size, and 8x VAE upsampling.
        let shape = [1, 16, 2 * (opts.height / 16), 2 * (opts.width / 16)];
        let noise = Array::from_slice(&seeded_noise(opts.seed, shape), &shape.map(|d| d as i32));
        let encoded = Instant::now();

        // Like diffusers, the latents stay in f32 and the model reads bf16.
        let mut scheduler = Scheduler::new(opts.num_steps);
        let (mut latents, steps) = match &opts.init_image {
            None => (noise, opts.num_steps),
            Some(img) => {
                let steps = scheduler.skip_for_strength(opts.strength);
                let init = self.encode_image(img, opts.width, opts.height)?;
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
            let mut pred = self
                .transformer
                .forward(&x, t, &cap_feats)?
                .as_dtype(Dtype::Float32)?;
            if let Some(neg) = &neg_cap_feats {
                // CFG: pred = neg + scale * (pos - neg)
                let neg_pred = self
                    .transformer
                    .forward(&x, t, neg)?
                    .as_dtype(Dtype::Float32)?;
                pred = neg_pred.add(
                    pred.subtract(&neg_pred)?
                        .multiply(Array::from_f32(opts.guidance_scale as f32))?,
                )?;
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
        let image = self
            .vae
            .decode(&latents.as_dtype(dtype)?.transpose_axes(&[0, 2, 3, 1])?)?;
        // [-1, 1] -> [0, 255], computed in the model dtype like candle.
        let image = mlx_rs::ops::clip(&image, (-1.0f32, 1.0f32))?
            .add(scalar(1.0, dtype)?)?
            .multiply(scalar(127.5, dtype)?)?
            .as_dtype(Dtype::Uint8)?;
        let image = to_rgb_image(&image)?;

        Ok(Generated {
            image: image.into(),
            timings: Timings {
                text: encoded - started,
                init_image: image_encoded - encoded,
                denoise: denoised - image_encoded,
                vae: denoised.elapsed(),
            },
        })
    }

    /// RGB image -> (1, 16, H/8, W/8) latents, resized to `width` x `height`.
    fn encode_image(&self, img: &image::RgbImage, width: usize, height: usize) -> Result<Array> {
        let (w, h) = (width as u32, height as u32);
        let img = resize_to_fill(img, w, h);
        // [0, 255] -> [-1, 1], NHWC.
        let pixels: Vec<f32> = img
            .as_raw()
            .iter()
            .map(|&p| p as f32 / 127.5 - 1.0)
            .collect();
        let x = Array::from_slice(&pixels, &[1, h as i32, w as i32, 3]).as_dtype(self.dtype)?;
        let z = self.vae.encode(&x)?.transpose_axes(&[0, 3, 1, 2])?;
        z.eval()?;
        Ok(z)
    }

    fn encode_prompt(&self, prompt: &str, on_progress: &mut dyn FnMut(Progress)) -> Result<Array> {
        let tokens = self
            .tokenizer
            .encode(format_prompt_for_qwen3(prompt).as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .iter()
            .copied()
            .take(MAX_PROMPT_TOKENS)
            .collect::<Vec<_>>();
        on_progress(Progress::Encoded {
            tokens: tokens.len(),
        });
        let feats = self.text_encoder.forward(&tokens)?;
        feats.eval()?;
        Ok(feats)
    }
}

/// Qwen3 chat template (add_generation_prompt=True, enable_thinking=True):
/// `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
fn format_prompt_for_qwen3(prompt: &str) -> String {
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
}

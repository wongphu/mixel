//! Text-to-image, img2img and image editing on Apple Silicon with mlx-rs:
//! Z-Image-Turbo, Qwen-Image-2.1 and MOSS-Image-SFT.
//!
//! ```no_run
//! use mixel::{GenerateOptions, LoadOptions, Pipeline};
//!
//! let pipeline = Pipeline::load(&LoadOptions::default())?;
//! let opts = GenerateOptions { seed: 42, width: 768, ..GenerateOptions::new("a red fox in fresh snow") };
//! let out = pipeline.generate(&opts)?;
//! out.image.save("fox.png")?;
//! println!("denoising took {:?}", out.timings.denoise);
//! # anyhow::Ok(())
//! ```
//!
//! [`zimage`], [`qwen21`] and [`moss`] hold the models themselves for
//! lower-level use.

mod autoencoder_kl;
pub mod lora;
pub mod moss;
pub mod nn;
mod pipeline;
pub mod qwen21;
mod schedule;
pub mod zimage;

pub use lora::Lora;
pub use nn::Quantize;
pub use pipeline::{
    composite_over_white, seeded_noise, Encoded, GenerateOptions, Generated, LoadOptions, Model,
    Parts, Pipeline, Progress, Timings, DEFAULT_REPO, DEFAULT_STEPS, SIZE_ALIGN,
};

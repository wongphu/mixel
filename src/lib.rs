//! Text-to-image, img2img and image editing on Apple Silicon with mlx-rs:
//! Z-Image-Turbo and Qwen-Image-2.1.
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
//! [`zimage`] and [`qwen21`] hold the models themselves for lower-level use.

pub mod nn;
mod pipeline;
pub mod qwen21;
pub mod zimage;

pub use pipeline::{
    composite_over_white, seeded_noise, GenerateOptions, Generated, LoadOptions, Model, Pipeline,
    Progress, Timings, DEFAULT_REPO, DEFAULT_STEPS, SIZE_ALIGN,
};

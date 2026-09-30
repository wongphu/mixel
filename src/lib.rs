//! Z-Image-Turbo text-to-image on Apple Silicon with mlx-rs.
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
//! [`zimage`] holds the model itself (a port of candle-transformers' `z_image`)
//! for lower-level use.

mod pipeline;
pub mod zimage;

pub use pipeline::{
    seeded_noise, GenerateOptions, Generated, LoadOptions, Pipeline, Progress, Timings,
    DEFAULT_REPO, DEFAULT_STEPS, SIZE_ALIGN,
};

//! Z-Image-Turbo ported to mlx-rs.
//!
//! A port of candle-transformers 0.11.0's `z_image` module. Layer structure,
//! weight names and numerics (bf16 compute, bf16 timestep, unshifted
//! scheduler sigmas) follow candle's implementation so outputs match `candy`.
//! Differences: images are NHWC inside the VAE (MLX's native conv layout), and
//! the all-ones attention masks candle builds are dropped since they are no-ops.

pub mod pipeline;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use crate::nn::{linear, rms_norm, scalar, silu, ModelFiles, Weights};

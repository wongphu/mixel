//! Z-Image-Turbo ported to mlx-rs.
//!
//! The layers are a port of candle-transformers 0.11.0's `z_image` module
//! (layer structure, weight names, bf16 compute). Sampling follows diffusers'
//! `ZImagePipeline`, the reference implementation, where candle differs: the
//! shifted sigma schedule, the time input and the latents in f32, and the
//! caption and image padded to a multiple of 32 tokens with the model's
//! learned pad tokens. `examples/zimage_diffusers_parity.rs` checks it against
//! diffusers. Images are NHWC inside the VAE (MLX's native conv layout), and
//! the all-ones attention masks candle builds are dropped since they are no-ops.

pub mod pipeline;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use crate::nn::{linear, rms_norm, scalar, silu, ModelFiles, Quantize, WeightCache, Weights};

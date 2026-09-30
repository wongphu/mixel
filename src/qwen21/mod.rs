//! Qwen-Image-2.1 ported to mlx-rs.
//!
//! A port of the diffusers `QwenImage21Pipeline` (transformer, VAE, scheduler)
//! and transformers' Qwen3-VL (text encoder and vision encoder). Every stage is
//! checked against the PyTorch reference in `examples/qwen21_parity.rs`.

pub mod pipeline;
pub mod prompt;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
pub mod vision;

/// Hugging Face repo of the weights.
pub const REPO: &str = "Qwen/Qwen-Image-2.1";
/// The reference pipeline's default number of denoising steps.
pub const DEFAULT_STEPS: usize = 40;
/// Width and height must be multiples of this (VAE 16x * 2x2 vision slots).
pub const SIZE_ALIGN: usize = 32;
/// Target side length used to size edits from a reference image.
pub const OUTPUT_RESOLUTION: usize = 1024;
/// Latent channels.
pub const LATENT_CHANNELS: usize = 64;
/// Pixels per latent token along each axis.
pub const VAE_SCALE: usize = 16;

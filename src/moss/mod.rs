//! MOSS-Image-SFT ported to mlx-rs.
//!
//! A port of the `MOSSImagePipeline` bundled with the weights
//! (`pipeline_moss_image.py`) and of transformers' Qwen3.5 text model, its
//! text encoder. Every stage is checked against the PyTorch reference in
//! `examples/moss_parity.rs`.

pub mod gated_delta;
pub mod prompt;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

/// Hugging Face repo of the weights.
pub const REPO: &str = "OpenMOSS-Team/MOSS-Image-SFT";
/// The reference pipeline's default number of denoising steps.
pub const DEFAULT_STEPS: usize = 50;
/// The reference pipeline's default guidance scale.
pub const DEFAULT_GUIDANCE: f32 = 3.0;
/// Width and height must be multiples of this (VAE 8x * 2x2 patches).
pub const SIZE_ALIGN: usize = 16;
/// The transformer block the guidance's negative branch skips.
pub const UNCOND_SKIP_LAYER: usize = 22;
/// Latent channels after the VAE's 2x2 patchify.
pub const LATENT_CHANNELS: usize = 128;
/// Prompt tokens the text encoder reads; the reference cuts the rest.
pub const MAX_PROMPT_TOKENS: usize = 4096;

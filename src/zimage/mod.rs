//! Z-Image-Turbo ported to mlx-rs.
//!
//! A port of candle-transformers 0.11.0's `z_image` module. Layer structure,
//! weight names and numerics (bf16 compute, bf16 timestep, unshifted
//! scheduler sigmas) follow candle's implementation so outputs match `candy`.
//! Differences: images are NHWC inside the VAE (MLX's native conv layout), and
//! the all-ones attention masks candle builds are dropped since they are no-ops.

pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::Path;

/// Named weights loaded from safetensors, converted to one dtype.
pub struct Weights {
    map: HashMap<String, Array>,
}

impl Weights {
    /// Loads the tensors whose names pass `keep`, one file at a time, casting to
    /// `dtype`. Conv weights (4D, PyTorch OIHW) are transposed to MLX's OHWI.
    pub fn load(
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        keep: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        let mut map = HashMap::new();
        for file in files {
            let file = file.as_ref();
            let arrays = Array::load_safetensors(file)
                .with_context(|| format!("loading {}", file.display()))?;
            let mut converted = Vec::with_capacity(arrays.len());
            for (name, a) in arrays {
                if !keep(&name) {
                    continue;
                }
                let a = if a.dtype() == dtype {
                    a
                } else {
                    a.as_dtype(dtype)?
                };
                let a = if a.ndim() == 4 {
                    a.transpose_axes(&[0, 2, 3, 1])?.contiguous()?
                } else {
                    a
                };
                converted.push((name, a));
            }
            // Materialize this file's tensors before reading the next, so the
            // original (e.g. f32) buffers are freed as we go.
            mlx_rs::transforms::eval(converted.iter().map(|(_, a)| a))?;
            map.extend(converted);
        }
        anyhow::ensure!(!map.is_empty(), "no weights loaded");
        Ok(Self { map })
    }

    pub fn get(&self, name: &str) -> Result<&Array> {
        self.map
            .get(name)
            .with_context(|| format!("missing weight {name}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
}

/// A 0-d array of `dtype`, so arithmetic with it keeps the other operand's dtype.
pub fn scalar(v: f32, dtype: Dtype) -> Result<Array> {
    Ok(Array::from_f32(v).as_dtype(dtype)?)
}

/// `x @ W^T (+ b)` for PyTorch-style `prefix.weight` / optional `prefix.bias`.
pub fn linear(x: &Array, w: &Weights, prefix: &str) -> Result<Array> {
    let y = x.matmul(w.get(&format!("{prefix}.weight"))?.t())?;
    let bias = format!("{prefix}.bias");
    if w.has(&bias) {
        Ok(y.add(w.get(&bias)?)?)
    } else {
        Ok(y)
    }
}

/// RMSNorm over the last axis with `prefix.weight`.
pub fn rms_norm(x: &Array, w: &Weights, prefix: &str, eps: f32) -> Result<Array> {
    Ok(mlx_rs::fast::rms_norm(
        x,
        Some(w.get(&format!("{prefix}.weight"))?),
        eps,
    )?)
}

pub fn silu(x: &Array) -> Result<Array> {
    Ok(mlx_rs::nn::silu(x)?)
}

/// Resolves `name` inside the model directory or the Hugging Face cache/hub.
pub struct ModelFiles {
    local: Option<std::path::PathBuf>,
    repo: Option<hf_hub::api::sync::ApiRepo>,
}

impl ModelFiles {
    pub fn new(repo_id: &str, local: Option<&str>) -> Result<Self> {
        Ok(match local {
            Some(dir) => Self {
                local: Some(dir.into()),
                repo: None,
            },
            None => Self {
                local: None,
                repo: Some(hf_hub::api::sync::Api::new()?.model(repo_id.to_string())),
            },
        })
    }

    pub fn get(&self, name: &str) -> Result<std::path::PathBuf> {
        match (&self.local, &self.repo) {
            (Some(dir), _) => {
                let p = dir.join(name);
                anyhow::ensure!(p.exists(), "{} not found", p.display());
                Ok(p)
            }
            (None, Some(repo)) => Ok(repo.get(name)?),
            _ => unreachable!(),
        }
    }
}

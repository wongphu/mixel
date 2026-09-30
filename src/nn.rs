//! Building blocks shared by the model ports: weight loading, common layers,
//! and locating model files.

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
    pub fn new(repo_id: &str, local: Option<&Path>) -> Result<Self> {
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

/// LayerNorm over the last axis without learnable parameters.
pub fn layer_norm(x: &Array, eps: f32) -> Result<Array> {
    Ok(mlx_rs::fast::layer_norm(x, None, None, eps)?)
}

/// LayerNorm over the last axis with `prefix.weight` and `prefix.bias`.
pub fn layer_norm_affine(x: &Array, w: &Weights, prefix: &str, eps: f32) -> Result<Array> {
    Ok(mlx_rs::fast::layer_norm(
        x,
        w.get(&format!("{prefix}.weight"))?,
        w.get(&format!("{prefix}.bias"))?,
        eps,
    )?)
}

/// GELU with the tanh approximation (PyTorch `approximate="tanh"`).
/// Computed in f32 and cast back, like PyTorch does for bf16 inputs (MLX's
/// built-in uses f32 constants, which would promote the result to f32).
pub fn gelu_tanh(x: &Array) -> Result<Array> {
    let x32 = x.as_dtype(Dtype::Float32)?;
    let inner = x32
        .add(
            x32.power(Array::from_f32(3.0))?
                .multiply(Array::from_f32(0.044715))?,
        )?
        .multiply(Array::from_f32((2.0 / std::f32::consts::PI).sqrt()))?;
    let y = x32
        .multiply(Array::from_f32(0.5))?
        .multiply(mlx_rs::ops::tanh(&inner)?.add(Array::from_f32(1.0))?)?;
    Ok(y.as_dtype(x.dtype())?)
}

/// Exact (erf) GELU, PyTorch's default `nn.GELU()`, computed in f32.
pub fn gelu(x: &Array) -> Result<Array> {
    let x32 = x.as_dtype(Dtype::Float32)?;
    let erf = mlx_rs::ops::erf(x32.multiply(Array::from_f32(std::f32::consts::FRAC_1_SQRT_2))?)?;
    let y = x32
        .multiply(Array::from_f32(0.5))?
        .multiply(erf.add(Array::from_f32(1.0))?)?;
    Ok(y.as_dtype(x.dtype())?)
}

/// `[-x2, x1]` for halves `x1, x2` of the last axis (non-interleaved RoPE).
pub fn rotate_half(x: &Array) -> Result<Array> {
    let d = *x.shape().last().expect("rank >= 1");
    let [x1, x2]: [Array; 2] = mlx_rs::ops::split_at_indices(x, &[d / 2], -1)?
        .try_into()
        .expect("2 halves");
    Ok(mlx_rs::ops::concatenate(&[x2.negative()?, x1], -1)?)
}

/// Splits a sequence axis at `bounds` (sorted, exclusive of 0 and len).
pub fn split_seq(x: &Array, bounds: &[i32], axis: i32) -> Result<Vec<Array>> {
    if bounds.is_empty() {
        return Ok(vec![x.clone()]);
    }
    Ok(mlx_rs::ops::split_at_indices(x, bounds, axis)?)
}

//! MOSS-Image in 4 steps: darask0's rapid-moss adapter
//! (`darask0/MOSS-Image_4-step_adapter_lora`, data-space DMD continued with
//! DMAD), ported from its `inference.py`.
//!
//! On top of the base weights it adds rank-128 updates to the projections of
//! every block, run as separate low-rank matmuls (merged into bf16 weights,
//! most of each would round away), and replaces the time embedding and the
//! two modulation biases with f32 ones. Those three modules then run in f32,
//! which keeps the residual stream in f32 too. The updates run in bf16, like
//! the authors' training sampler: `inference.py` computes them in f32, which
//! took 25% longer per step here for the same result within bf16 noise.
//! Sampling is the adapter's own: from one noise `z`, each step predicts the
//! image from `z + Q x0` at a fixed step code, without guidance.

use crate::lora::Lora;
use crate::nn::ModelFiles;
use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;

/// Hugging Face repo of the adapter, and the revision mixel was checked
/// against.
pub const REPO: &str = "darask0/MOSS-Image_4-step_adapter_lora";
pub const REVISION: &str = "c88f33bcb3251eb39943d0635e74b650732977c8";
/// Steps the adapter is distilled for; each shorter prefix was trained too.
pub const STEPS: usize = 4;
/// Blocks x projections the LoRA updates.
const LORA_LAYERS: usize = 42 * 6;

/// The f32 tensors the adapter replaces the time path with.
pub(crate) const TIME_TENSORS: [&str; 6] = [
    "time_embedding.0.weight",
    "time_embedding.0.bias",
    "time_embedding.2.weight",
    "time_embedding.2.bias",
    "shared_block_modulation.bias",
    "video_head_modulation.bias",
];

/// Fails if `lora` is a data-space DMD adapter's LoRA (this one, or the
/// 8-step one it continues), which as a plain `--lora` would run without the
/// adapter's time path and sampler. Both are rank 128 on all the block
/// projections, with metadata holding only the rank and alpha (and the
/// training steps); regular LoRAs carry their trainer's metadata or none.
pub fn check_user_lora(lora: &Lora) -> Result<()> {
    let (_, metadata) = Array::load_safetensors_with_metadata(&lora.path)
        .with_context(|| format!("reading LoRA {}", lora.path.display()))?;
    let adapter_like = metadata.get("rank").map(String::as_str) == Some("128")
        && metadata.contains_key("alpha")
        && metadata
            .keys()
            .all(|k| matches!(k.as_str(), "rank" | "alpha" | "iteration" | "steps"));
    anyhow::ensure!(
        !(adapter_like && lora.layers()?.len() == LORA_LAYERS),
        "{} looks like a few-step adapter for MOSS-Image (data-space DMD), not a LoRA: \
         it needs its own time embedding and sampler. Use --model moss-image-fast instead.",
        lora.path.display()
    );
    Ok(())
}

/// The loaded adapter, ready to apply to the transformer.
pub struct Adapter {
    pub(crate) lora: Lora,
    /// [`TIME_TENSORS`] in f32.
    pub(crate) time: HashMap<String, Array>,
    /// (128, 128) latent mix, applied as `x = z + x0 @ q^T`.
    pub q: Array,
    /// The transformer's time input at each step.
    pub step_codes: Vec<f32>,
}

impl Adapter {
    /// Downloads (on first use) and checks the adapter.
    pub fn load(files: &ModelFiles) -> Result<Self> {
        let lora = Lora {
            path: files.get("lora.safetensors")?,
            scale: 1.0,
        };
        let layers = lora.layers()?;
        anyhow::ensure!(
            layers.len() == LORA_LAYERS && layers.iter().all(|l| l.starts_with("layers.")),
            "{}: expected updates to the {LORA_LAYERS} block projections, got {} layers",
            lora.path.display(),
            layers.len()
        );
        let load = |name: &str| -> Result<HashMap<String, Array>> {
            let path = files.get(name)?;
            Array::load_safetensors(&path).with_context(|| format!("loading {}", path.display()))
        };
        let mut time = load("time_embedding.safetensors")?;
        anyhow::ensure!(
            time.len() == TIME_TENSORS.len() && TIME_TENSORS.iter().all(|n| time.contains_key(*n)),
            "time_embedding.safetensors holds {:?}",
            time.keys().collect::<Vec<_>>()
        );
        for a in time.values_mut() {
            *a = a.as_dtype(Dtype::Float32)?;
        }
        let mut dataspace = load("dataspace.safetensors")?;
        let q = dataspace
            .remove("latent_mix.Q")
            .context("dataspace.safetensors has no latent_mix.Q")?
            .as_dtype(Dtype::Float32)?;
        anyhow::ensure!(
            q.shape() == [128, 128],
            "latent_mix.Q has shape {:?}",
            q.shape()
        );
        let codes = dataspace
            .remove("step_codes")
            .context("dataspace.safetensors has no step_codes")?
            .as_dtype(Dtype::Float32)?;
        mlx_rs::transforms::eval(time.values().chain([&q, &codes]))?;
        let step_codes = codes.as_slice::<f32>().to_vec();
        anyhow::ensure!(
            step_codes.len() == STEPS && step_codes.windows(2).all(|w| w[0] > w[1]),
            "unexpected step codes {step_codes:?}"
        );
        Ok(Self {
            lora,
            time,
            q,
            step_codes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A LoRA file on all 252 block projections (tiny tensors), with `metadata`.
    fn write_lora(dir: &std::path::Path, name: &str, metadata: &[(&str, &str)]) -> Lora {
        let mut arrays = HashMap::new();
        for i in 0..42 {
            for p in ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"] {
                let l = format!("diffusion_model.layers.{i}.proj.{p}");
                arrays.insert(
                    format!("{l}.lora_down.weight"),
                    Array::zeros::<f32>(&[1, 4]).unwrap(),
                );
                arrays.insert(
                    format!("{l}.lora_up.weight"),
                    Array::zeros::<f32>(&[4, 1]).unwrap(),
                );
                arrays.insert(format!("{l}.alpha"), Array::from_f32(1.0));
            }
        }
        let metadata: HashMap<String, String> = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let path = dir.join(name);
        Array::save_safetensors(arrays, &metadata, &path).unwrap();
        Lora { path, scale: 1.0 }
    }

    #[test]
    fn refuses_the_adapters_lora_but_not_a_regular_one() {
        crate::nn::test_device();
        let dir = tempfile::tempdir().unwrap();
        let rapid = [
            ("rank", "128"),
            ("alpha", "128.0"),
            ("iteration", "500"),
            ("steps", "4"),
        ];
        let err = check_user_lora(&write_lora(dir.path(), "a.safetensors", &rapid)).unwrap_err();
        assert!(err.to_string().contains("moss-image-fast"), "{err}");
        let eight_step = [("rank", "128"), ("alpha", "128.0")];
        assert!(check_user_lora(&write_lora(dir.path(), "b.safetensors", &eight_step)).is_err());
        // A regular LoRA: no metadata, or its trainer's.
        check_user_lora(&write_lora(dir.path(), "c.safetensors", &[])).unwrap();
        let kohya = [
            ("ss_network_dim", "128"),
            ("rank", "128"),
            ("alpha", "128.0"),
        ];
        check_user_lora(&write_lora(dir.path(), "d.safetensors", &kohya)).unwrap();
    }
}

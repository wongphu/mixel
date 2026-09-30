//! Qwen-Image-2.1 in 4 steps: Alibaba PAI's Fun-Acc LoRA
//! (`alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs`, Parallel Decoding
//! Distillation), ported from its `qwenimage21_pdd.py`.
//!
//! On top of the base weights it adds rank-64 updates to the transformer's
//! linear layers, replaces the QK-norm and text-norm weights, and gives
//! `proj_out` one head per step of a fixed 4-step schedule. The updates run
//! as separate low-rank matmuls, like the reference, adding ~3% to a step:
//! they are ~0.2% of the weights, so merging them into bf16 weights would
//! round away about half of each. Sampling runs without guidance and keeps the
//! latents in f32 between steps (the reference's `native_time_fp32_state`).

use crate::nn::ModelFiles;
use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

/// Hugging Face repo of the adapter.
pub const REPO: &str = "alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs";
const WEIGHTS: &str = "models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors";
const CONFIG: &str = "models/pdd_config.json";
/// Denoising steps the adapter is distilled for.
pub const STEPS: usize = 4;
/// The export format this port reads: merged single-step heads.
const EXPORT_FORMAT: &str = "qwenimage21_extracted_prefused_v1";
const PRECISION: &str = "native_time_fp32_state";

/// `pdd_config.json`.
#[derive(Debug, Deserialize)]
struct Config {
    pdd_num_steps: usize,
    pdd_block_size: usize,
    lora_rank: usize,
    lora_alpha: f32,
    lora_targets: String,
    pdd_sigmas: Vec<f32>,
    pdd_full_parameters: Vec<String>,
    pdd_export_format: String,
    pdd_sampling_precision: String,
}

impl Config {
    fn parse(json: &str) -> Result<Self> {
        let c: Config = serde_json::from_str(json)?;
        anyhow::ensure!(
            c.pdd_export_format == EXPORT_FORMAT && c.pdd_sampling_precision == PRECISION,
            "unsupported adapter format {:?} / {:?}",
            c.pdd_export_format,
            c.pdd_sampling_precision
        );
        anyhow::ensure!(
            c.pdd_num_steps == STEPS && c.pdd_block_size == 1,
            "expected a {STEPS}-step adapter with block size 1, got {} steps, block size {}",
            c.pdd_num_steps,
            c.pdd_block_size
        );
        let s = &c.pdd_sigmas;
        anyhow::ensure!(
            s.len() == STEPS + 1
                && s[0] == 1.0
                && s[STEPS] == 0.0
                && s.windows(2).all(|w| w[0] > w[1]),
            "bad sigma schedule {s:?}"
        );
        anyhow::ensure!(c.lora_rank > 0, "lora_rank is 0");
        Ok(c)
    }

    fn targets(&self) -> Vec<&str> {
        self.lora_targets
            .split(',')
            .filter(|t| !t.is_empty())
            .collect()
    }

    /// Every tensor the checkpoint must hold, as the reference checks.
    fn expected_tensors(&self) -> HashSet<String> {
        let mut names: HashSet<String> = self
            .targets()
            .iter()
            .flat_map(|t| [format!("{t}.lora_down"), format!("{t}.lora_up")])
            .collect();
        names.extend(self.pdd_full_parameters.iter().cloned());
        names.insert("proj_out.weight".into());
        names
    }
}

/// The loaded adapter, ready to apply to the transformer's weights.
pub struct Adapter {
    /// The fixed schedule, `STEPS + 1` sigmas ending in 0.
    pub sigmas: Vec<f32>,
    /// `alpha / rank`.
    pub scaling: f32,
    /// Linear layers (weight name prefixes) with a low-rank update.
    pub targets: Vec<String>,
    /// Weights replaced outright (QK norms, text norm).
    pub full: Vec<String>,
    /// `proj_out` weight of each step, (64, 4096).
    pub heads: Vec<Array>,
    tensors: HashMap<String, Array>,
}

impl Adapter {
    /// Downloads (on first use) and checks the adapter.
    pub fn load(files: &ModelFiles, dtype: Dtype) -> Result<Self> {
        let config_path = files.get(CONFIG)?;
        let config = Config::parse(&std::fs::read_to_string(&config_path)?)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let path = files.get(WEIGHTS)?;
        let mut tensors = Array::load_safetensors(&path)
            .with_context(|| format!("loading {}", path.display()))?;
        let expected = config.expected_tensors();
        let found: HashSet<String> = tensors.keys().cloned().collect();
        anyhow::ensure!(
            found == expected,
            "{} does not match its config: {} missing, {} unexpected",
            path.display(),
            expected.difference(&found).count(),
            found.difference(&expected).count()
        );
        for a in tensors.values_mut() {
            if a.dtype() != dtype {
                *a = a.as_dtype(dtype)?;
            }
        }
        let proj = tensors.remove("proj_out.weight").expect("checked above");
        anyhow::ensure!(
            proj.ndim() == 3 && proj.shape()[0] as usize == STEPS,
            "proj_out heads have shape {:?}",
            proj.shape()
        );
        let heads = (0..STEPS as i32)
            .map(|i| {
                let head = proj.take_axis(Array::from_int(i), 0)?;
                head.eval()?;
                Ok(head)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            scaling: config.lora_alpha / config.lora_rank as f32,
            targets: config.targets().iter().map(|t| t.to_string()).collect(),
            full: config.pdd_full_parameters.clone(),
            sigmas: config.pdd_sigmas,
            heads,
            tensors,
        })
    }

    /// `(down, up)` of a target layer, with the scaling folded into `up`.
    pub fn lora(&self, target: &str) -> Result<(Array, Array)> {
        let get = |suffix: &str| {
            self.tensors
                .get(&format!("{target}.{suffix}"))
                .with_context(|| format!("adapter has no {target}.{suffix}"))
        };
        let (down, up) = (get("lora_down")?.clone(), get("lora_up")?);
        let up = if self.scaling == 1.0 {
            up.clone()
        } else {
            up.as_dtype(Dtype::Float32)?
                .multiply(Array::from_f32(self.scaling))?
                .as_dtype(up.dtype())?
        };
        Ok((down, up))
    }

    /// A weight the adapter replaces.
    pub fn full(&self, name: &str) -> Result<&Array> {
        self.tensors
            .get(name)
            .with_context(|| format!("adapter has no {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_json(steps: usize, block: usize, sigmas: &str) -> String {
        format!(
            r#"{{"pdd_num_steps": {steps}, "pdd_block_size": {block}, "lora_rank": 64,
                "lora_alpha": 64.0, "lora_targets": "img_in,transformer_blocks.0.attn.to_q",
                "pdd_sigmas": {sigmas}, "pdd_full_parameters": ["txt_in.text_norm.weight"],
                "pdd_sampling_precision": "native_time_fp32_state",
                "pdd_export_format": "qwenimage21_extracted_prefused_v1",
                "pdd_inference_only": true, "pdd_sample_size": [2048, 2048]}}"#
        )
    }

    const SIGMAS: &str = "[1.0, 0.9169867038726807, 0.7861579060554504, 0.5494909882545471, 0.0]";

    #[test]
    fn parses_the_released_config() {
        let c = Config::parse(&config_json(4, 1, SIGMAS)).unwrap();
        assert_eq!(c.pdd_sigmas.len(), 5);
        assert_eq!(c.targets(), ["img_in", "transformer_blocks.0.attn.to_q"]);
        let expected = c.expected_tensors();
        for name in [
            "img_in.lora_down",
            "img_in.lora_up",
            "transformer_blocks.0.attn.to_q.lora_up",
            "txt_in.text_norm.weight",
            "proj_out.weight",
        ] {
            assert!(expected.contains(name), "{name}");
        }
        assert_eq!(expected.len(), 6);
    }

    #[test]
    fn rejects_other_step_counts_and_schedules() {
        let err = |json: String| Config::parse(&json).unwrap_err().to_string();
        assert!(err(config_json(8, 1, SIGMAS)).contains("4-step"));
        assert!(err(config_json(4, 2, SIGMAS)).contains("block size 2"));
        assert!(err(config_json(4, 1, "[1.0, 0.5, 0.0]")).contains("bad sigma"));
        assert!(err(config_json(4, 1, "[1.0, 0.9, 0.95, 0.5, 0.0]")).contains("bad sigma"));
    }
}

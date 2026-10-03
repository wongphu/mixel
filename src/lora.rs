//! LoRA files: low-rank updates to a model's linear layers, applied on top of
//! its weights (quantized or not) rather than merged into them.
//!
//! Reads the two layouts Z-Image-Turbo LoRAs come in, under a
//! `diffusion_model.` or `transformer.` prefix or none:
//! `<layer>.lora_A.weight` / `.lora_B.weight` (PEFT, ai-toolkit), and
//! `<layer>.lora_down.weight` / `.lora_up.weight` with an optional `.alpha`
//! (kohya, ComfyUI). The update is `scale * alpha / rank * up @ down`, alpha
//! defaulting to the rank.

use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A LoRA file and how strongly to apply it (1.0: as trained).
#[derive(Debug, Clone, PartialEq)]
pub struct Lora {
    pub path: PathBuf,
    pub scale: f32,
}

/// One layer's update, `x @ down^T @ up^T`, with the scale folded into `up`.
pub(crate) struct Update {
    pub layer: String,
    pub down: Array,
    pub up: Array,
}

const PREFIXES: [&str; 3] = ["diffusion_model.", "transformer.", "base_model.model."];
/// (down, up) suffixes.
const PAIRS: [(&str, &str); 3] = [
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_A.default.weight", ".lora_B.default.weight"),
];

/// A layer's tensor names in the file.
#[derive(Default)]
struct Names {
    down: Option<String>,
    up: Option<String>,
    alpha: Option<String>,
}

impl Lora {
    /// Parses `path` or `path:scale`.
    pub fn parse(s: &str) -> Result<Self> {
        let (path, scale) = match s.rsplit_once(':') {
            Some((path, scale)) if !path.is_empty() && scale.parse::<f32>().is_ok() => {
                (path, scale.parse::<f32>()?)
            }
            _ => (s, 1.0),
        };
        anyhow::ensure!(
            scale.is_finite(),
            "LoRA scale must be a number, got {scale}"
        );
        Ok(Self {
            path: PathBuf::from(path),
            scale,
        })
    }

    /// The layers it changes, from the file's header (no tensor data is
    /// read): fails on a file it can't use.
    pub fn layers(&self) -> Result<Vec<String>> {
        Ok(self.names()?.into_keys().collect())
    }

    fn names(&self) -> Result<BTreeMap<String, Names>> {
        let tensors = Array::load_safetensors(&self.path)
            .with_context(|| format!("reading LoRA {}", self.path.display()))?;
        group(tensors.keys().map(String::as_str), &self.path)
    }

    /// Reads the updates in `dtype`.
    pub(crate) fn updates(&self, dtype: Dtype) -> Result<Vec<Update>> {
        let mut tensors = Array::load_safetensors(&self.path)
            .with_context(|| format!("reading LoRA {}", self.path.display()))?;
        let names = group(tensors.keys().map(String::as_str), &self.path)?;
        let mut take = |name: &Option<String>| -> Result<Array> {
            let name = name.as_ref().expect("checked by group");
            tensors
                .remove(name)
                .with_context(|| format!("missing {name}"))
        };
        let mut updates = Vec::new();
        for (layer, n) in &names {
            let (down, up) = (take(&n.down)?, take(&n.up)?);
            anyhow::ensure!(
                down.ndim() == 2 && up.ndim() == 2 && down.shape()[0] == up.shape()[1],
                "{}: {layer}: down {:?} and up {:?} don't pair up",
                self.path.display(),
                down.shape(),
                up.shape()
            );
            let rank = down.shape()[0] as f32;
            let alpha = match &n.alpha {
                Some(_) => {
                    let a = take(&n.alpha)?.as_dtype(Dtype::Float32)?;
                    a.eval()?;
                    a.as_slice::<f32>()[0]
                }
                None => rank,
            };
            let up = up
                .as_dtype(Dtype::Float32)?
                .multiply(Array::from_f32(self.scale * alpha / rank))?
                .as_dtype(dtype)?;
            let down = down.as_dtype(dtype)?;
            mlx_rs::transforms::eval([&down, &up])?;
            updates.push(Update {
                layer: layer.clone(),
                down,
                up,
            });
        }
        Ok(updates)
    }
}

/// Groups a LoRA file's tensor names by the layer they update.
fn group<'a>(names: impl Iterator<Item = &'a str>, path: &Path) -> Result<BTreeMap<String, Names>> {
    let file = path.display();
    let mut layers: BTreeMap<String, Names> = BTreeMap::new();
    let mut unknown = Vec::new();
    for full in names {
        let name = PREFIXES
            .iter()
            .find_map(|p| full.strip_prefix(p))
            .unwrap_or(full);
        anyhow::ensure!(
            !(name.starts_with("lora_te") || name.starts_with("text_encoder")),
            "{file}: text-encoder LoRAs aren't supported ({full})"
        );
        anyhow::ensure!(
            !name.ends_with(".dora_scale"),
            "{file}: DoRA isn't supported ({full})"
        );
        anyhow::ensure!(
            !name.starts_with("lora_unet_"),
            "{file}: kohya's underscore names (lora_unet_...) aren't supported yet ({full})"
        );
        if let Some(layer) = name.strip_suffix(".alpha") {
            layers.entry(layer.to_string()).or_default().alpha = Some(full.to_string());
        } else if let Some((layer, is_up)) = PAIRS.iter().find_map(|(d, u)| {
            name.strip_suffix(d)
                .map(|l| (l, false))
                .or_else(|| name.strip_suffix(u).map(|l| (l, true)))
        }) {
            let entry = layers.entry(layer.to_string()).or_default();
            if is_up {
                entry.up = Some(full.to_string());
            } else {
                entry.down = Some(full.to_string());
            }
        } else {
            unknown.push(full);
        }
    }
    if !unknown.is_empty() {
        anyhow::bail!(
            "{file}: {} tensor(s) that aren't LoRA weights, e.g. {}",
            unknown.len(),
            unknown
                .iter()
                .take(3)
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    anyhow::ensure!(!layers.is_empty(), "{file}: no LoRA weights");
    for (layer, n) in &layers {
        anyhow::ensure!(
            n.down.is_some() && n.up.is_some(),
            "{file}: {layer} has only half of its update"
        );
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, tensors: &[(&str, Array)]) -> PathBuf {
        let path = dir.join(name);
        Array::save_safetensors(tensors.iter().map(|(n, a)| (*n, a)), None, &path).unwrap();
        path
    }

    #[test]
    fn parses_path_and_scale() {
        assert_eq!(Lora::parse("a.safetensors").unwrap().scale, 1.0);
        let l = Lora::parse("dir/a.safetensors:0.75").unwrap();
        assert_eq!(
            (l.path, l.scale),
            (PathBuf::from("dir/a.safetensors"), 0.75)
        );
        assert_eq!(Lora::parse("c:/x.safetensors").unwrap().scale, 1.0);
        assert!(Lora::parse("a.safetensors:inf").is_err());
    }

    #[test]
    fn reads_both_layouts_and_folds_in_alpha_and_scale() {
        crate::nn::test_device();
        let dir = tempfile::tempdir().unwrap();
        let down = Array::from_slice(&[1.0f32; 2 * 4], &[2, 4]);
        let up = Array::from_slice(&[1.0f32; 3 * 2], &[3, 2]);
        let peft = write(
            dir.path(),
            "peft.safetensors",
            &[
                (
                    "diffusion_model.layers.0.attention.to_q.lora_A.weight",
                    down.clone(),
                ),
                (
                    "diffusion_model.layers.0.attention.to_q.lora_B.weight",
                    up.clone(),
                ),
            ],
        );
        let kohya = write(
            dir.path(),
            "kohya.safetensors",
            &[
                ("layers.0.feed_forward.w1.lora_down.weight", down.clone()),
                ("layers.0.feed_forward.w1.lora_up.weight", up.clone()),
                ("layers.0.feed_forward.w1.alpha", Array::from_f32(1.0)),
            ],
        );
        let lora = |path: &Path, scale| Lora {
            path: path.to_path_buf(),
            scale,
        };
        assert_eq!(
            lora(&peft, 1.0).layers().unwrap(),
            ["layers.0.attention.to_q"]
        );
        assert_eq!(
            lora(&kohya, 1.0).layers().unwrap(),
            ["layers.0.feed_forward.w1"]
        );

        // Without alpha the scale is the user's; with it, times alpha / rank.
        let first_up = |path: &Path, scale| {
            let u = lora(path, scale).updates(Dtype::Float32).unwrap().remove(0);
            u.up.eval().unwrap();
            u.up.as_slice::<f32>()[0]
        };
        assert_eq!(first_up(&peft, 0.5), 0.5);
        assert_eq!(first_up(&kohya, 1.0), 0.5); // alpha 1, rank 2
    }

    #[test]
    fn rejects_files_it_cannot_use() {
        let dir = tempfile::tempdir().unwrap();
        let t = || Array::from_slice(&[0.0f32; 4], &[2, 2]);
        let err = |name: &str, tensors: &[(&str, Array)]| {
            let path = write(dir.path(), name, tensors);
            Lora { path, scale: 1.0 }.layers().unwrap_err().to_string()
        };
        assert!(err(
            "te.safetensors",
            &[("lora_te1_text_model.lora_down.weight", t())]
        )
        .contains("text-encoder"));
        assert!(
            err("half.safetensors", &[("layers.0.x.lora_A.weight", t())]).contains("only half")
        );
        assert!(
            err("odd.safetensors", &[("layers.0.x.weight", t())]).contains("aren't LoRA weights")
        );
        assert!(err(
            "kohya.safetensors",
            &[("lora_unet_layers_0_x.lora_down.weight", t())]
        )
        .contains("lora_unet_"));
    }
}

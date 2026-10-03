//! Building blocks shared by the model ports: weight loading, common layers,
//! and locating model files.

use anyhow::{Context, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Weight-only quantization of linear layers: MLX's affine quantization,
/// with a scale and bias per group of [`GROUP_SIZE`] weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantize {
    /// 8 bits per weight (8.5 with the group scales and biases).
    Q8,
    /// 4 bits per weight (4.5 with the group scales and biases).
    Q4,
}

impl Quantize {
    pub fn bits(self) -> i32 {
        match self {
            Quantize::Q8 => 8,
            Quantize::Q4 => 4,
        }
    }
}

/// Weights per scale and bias in a quantized layer. Groups of 32 bring a 4-bit
/// Z-Image step only from 16% to 15% off bf16, for 10% more memory.
pub const GROUP_SIZE: i32 = 64;

/// A linear layer's weight, quantized: packed `w`, and a scale and bias per group.
struct QuantizedWeight {
    w: Array,
    scales: Array,
    biases: Array,
    bits: i32,
    /// The weight's shape before quantization, (out, in).
    shape: Vec<i32>,
}

/// Named weights loaded from safetensors, converted to one dtype.
pub struct Weights {
    map: HashMap<String, Array>,
    /// Quantized linear layers, by layer prefix (see [`Weights::load_quantized`]).
    quantized: HashMap<String, QuantizedWeight>,
    /// Low-rank updates `(down, up)` that [`linear`] adds to a layer's output.
    lora: HashMap<String, (Array, Array)>,
}

impl Weights {
    /// Loads the tensors whose names pass `keep`, one file at a time, casting to
    /// `dtype`. Conv weights (4D, PyTorch OIHW) are transposed to MLX's OHWI.
    pub fn load(
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        keep: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        Self::load_quantized(files, dtype, keep, None, |_| false)
    }

    /// Like [`Weights::load`], and with `quantize`, also quantizes the
    /// `{prefix}.weight` of each linear layer whose prefix passes `layers`,
    /// file by file, so the unquantized weights are never all in memory.
    /// [`linear`] then runs those layers quantized.
    pub fn load_quantized(
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        keep: impl Fn(&str) -> bool,
        quantize: Option<Quantize>,
        layers: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        let mut map = HashMap::new();
        let mut quantized = HashMap::new();
        for file in files {
            let file = file.as_ref();
            let arrays = Array::load_safetensors(file)
                .with_context(|| format!("loading {}", file.display()))?;
            let mut converted = Vec::with_capacity(arrays.len());
            let mut file_quantized = Vec::new();
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
                let layer = name.strip_suffix(".weight");
                match (quantize, layer) {
                    (Some(q), Some(layer)) if a.ndim() == 2 && layers(layer) => {
                        let shape = a.shape().to_vec();
                        anyhow::ensure!(
                            shape[0] % 32 == 0 && shape[1] % GROUP_SIZE == 0,
                            "{name}: can't quantize shape {shape:?}"
                        );
                        let (w, scales, biases) = mlx_rs::ops::quantize(&a, GROUP_SIZE, q.bits())?;
                        // One at a time: a batch would hold many unquantized
                        // weights at once.
                        mlx_rs::transforms::eval([&w, &scales, &biases])?;
                        let qw = QuantizedWeight {
                            w,
                            scales,
                            biases,
                            bits: q.bits(),
                            shape,
                        };
                        file_quantized.push((layer.to_string(), qw));
                    }
                    _ => converted.push((name, a)),
                }
            }
            // Materialize this file's tensors before reading the next, so the
            // original (e.g. f32) buffers are freed as we go.
            mlx_rs::transforms::eval(converted.iter().map(|(_, a)| a))?;
            map.extend(converted);
            quantized.extend(file_quantized);
            // MLX keeps freed buffers for reuse; the source tensors' would
            // otherwise pile up across files.
            mlx_rs::memory::clear_cache()?;
        }
        anyhow::ensure!(
            !map.is_empty() || !quantized.is_empty(),
            "no weights loaded"
        );
        Ok(Self {
            map,
            quantized,
            lora: HashMap::new(),
        })
    }

    /// [`Weights::load_quantized`], through `cache` (as `name`) when there is
    /// one and `quantize` is set.
    #[allow(clippy::too_many_arguments)]
    pub fn load_maybe_cached(
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        keep: impl Fn(&str) -> bool,
        quantize: Option<Quantize>,
        layers: impl Fn(&str) -> bool,
        cache: Option<&WeightCache>,
        name: &str,
    ) -> Result<Self> {
        match (quantize, cache) {
            (Some(q), Some(c)) => {
                Self::load_quantized_cached(files, dtype, keep, q, layers, c, name)
            }
            _ => Self::load_quantized(files, dtype, keep, quantize, layers),
        }
    }

    /// [`Weights::load_quantized`] through `cache`: loads the quantized
    /// weights from it if it has them for these files, else loads and
    /// quantizes the files and saves the result there as `name`.
    pub fn load_quantized_cached(
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        keep: impl Fn(&str) -> bool,
        quantize: Quantize,
        layers: impl Fn(&str) -> bool,
        cache: &WeightCache,
        name: &str,
    ) -> Result<Self> {
        let (path, prefix) = cache.path(name, files, dtype, quantize)?;
        if path.exists() {
            match Self::load_cache_file(&path, quantize) {
                Ok(w) => return Ok(w),
                Err(e) => cache.note(format!(
                    "rebuilding {}, which couldn't be read: {e:#}",
                    path.display()
                )),
            }
        }
        let w = Self::load_quantized(files, dtype, keep, Some(quantize), layers)?;
        match w.save_cache_file(&path, quantize) {
            Ok(bytes) => {
                let removed = cache.remove_stale(&prefix, &path);
                cache.note(format!(
                    "saved {:.1} GB of {}-bit {name} weights to {}{}",
                    bytes as f64 / (1u64 << 30) as f64,
                    quantize.bits(),
                    path.display(),
                    if removed > 0 {
                        " (replacing an older copy)"
                    } else {
                        ""
                    }
                ));
            }
            Err(e) => cache.note(format!("couldn't save {}: {e:#}", path.display())),
        }
        Ok(w)
    }

    /// Writes every tensor, quantized ones as `{layer}.weight` (packed) with
    /// `.scales` and `.biases`, to a temporary file renamed to `path`.
    /// Returns its size.
    fn save_cache_file(&self, path: &Path, quantize: Quantize) -> Result<u64> {
        let dir = path.parent().context("cache file without a directory")?;
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        // MLX only writes files named *.safetensors; the leading dot keeps
        // another run's cleanup (WeightCache::remove_stale) away from it.
        let file = path.file_name().context("cache file without a name")?;
        let tmp = dir.join(format!(
            ".tmp-{}-{}",
            std::process::id(),
            file.to_string_lossy()
        ));
        let mut arrays: Vec<(String, &Array)> =
            self.map.iter().map(|(k, v)| (k.clone(), v)).collect();
        for (layer, q) in &self.quantized {
            arrays.push((format!("{layer}.weight"), &q.w));
            arrays.push((format!("{layer}.weight.scales"), &q.scales));
            arrays.push((format!("{layer}.weight.biases"), &q.biases));
        }
        let metadata = HashMap::from([
            ("mixel_cache".to_string(), CACHE_FORMAT.to_string()),
            ("bits".to_string(), quantize.bits().to_string()),
            ("group_size".to_string(), GROUP_SIZE.to_string()),
        ]);
        let saved = Array::save_safetensors(arrays, &metadata, &tmp)
            .map_err(anyhow::Error::from)
            .and_then(|()| Ok(std::fs::rename(&tmp, path)?));
        if saved.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        saved?;
        Ok(std::fs::metadata(path)?.len())
    }

    /// Reads a file written by [`save_cache_file`](Self::save_cache_file).
    fn load_cache_file(path: &Path, quantize: Quantize) -> Result<Self> {
        let (mut arrays, metadata) = Array::load_safetensors_with_metadata(path)?;
        let meta = |k: &str| metadata.get(k).map(String::as_str);
        anyhow::ensure!(
            meta("mixel_cache") == Some(CACHE_FORMAT)
                && meta("bits") == Some(quantize.bits().to_string().as_str())
                && meta("group_size") == Some(GROUP_SIZE.to_string().as_str()),
            "not a {}-bit mixel cache file (format {CACHE_FORMAT})",
            quantize.bits()
        );
        let layers: Vec<String> = arrays
            .keys()
            .filter_map(|k| k.strip_suffix(".weight.scales").map(str::to_string))
            .collect();
        let mut quantized = HashMap::new();
        for layer in layers {
            let mut take = |suffix: &str| {
                arrays
                    .remove(&format!("{layer}.{suffix}"))
                    .with_context(|| format!("missing {layer}.{suffix}"))
            };
            let (w, scales, biases) = (
                take("weight")?,
                take("weight.scales")?,
                take("weight.biases")?,
            );
            let bits = quantize.bits();
            let shape = vec![w.shape()[0], w.shape()[1] * 32 / bits];
            quantized.insert(
                layer,
                QuantizedWeight {
                    w,
                    scales,
                    biases,
                    bits,
                    shape,
                },
            );
        }
        mlx_rs::transforms::eval(
            arrays.values().chain(
                quantized
                    .values()
                    .flat_map(|q| [&q.w, &q.scales, &q.biases]),
            ),
        )?;
        Ok(Self {
            map: arrays,
            quantized,
            lora: HashMap::new(),
        })
    }

    pub fn get(&self, name: &str) -> Result<&Array> {
        if let Some(layer) = name.strip_suffix(".weight") {
            anyhow::ensure!(
                !self.quantized.contains_key(layer),
                "weight {name} is quantized; only linear() can use it"
            );
        }
        self.map
            .get(name)
            .with_context(|| format!("missing weight {name}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Shape of the weight `name`, also when it is quantized.
    pub fn shape(&self, name: &str) -> Result<Vec<i32>> {
        if let Some(q) = name
            .strip_suffix(".weight")
            .and_then(|layer| self.quantized.get(layer))
        {
            return Ok(q.shape.clone());
        }
        Ok(self.get(name)?.shape().to_vec())
    }

    /// Replaces an existing weight with one of the same shape and dtype.
    pub fn replace(&mut self, name: &str, a: Array) -> Result<()> {
        let old = self
            .map
            .get_mut(name)
            .with_context(|| format!("missing weight {name}"))?;
        anyhow::ensure!(
            old.shape() == a.shape() && old.dtype() == a.dtype(),
            "{name}: replacement {:?} {:?} does not match {:?} {:?}",
            a.shape(),
            a.dtype(),
            old.shape(),
            old.dtype()
        );
        *old = a;
        Ok(())
    }

    /// Adds a low-rank update to the linear layer `prefix`: [`linear`] then
    /// returns `x W^T + b + (x down^T) up^T`, like a PEFT LoRA with its
    /// scaling folded into `up`. Kept separate rather than merged into `W`:
    /// updates much smaller than the weights mostly round away in bf16.
    pub fn add_lora(&mut self, prefix: &str, down: Array, up: Array) -> Result<()> {
        let w = self.shape(&format!("{prefix}.weight"))?;
        anyhow::ensure!(
            down.ndim() == 2
                && up.ndim() == 2
                && down.shape()[0] == up.shape()[1]
                && [up.shape()[0], down.shape()[1]] == w[..],
            "{prefix}: low-rank update {:?} x {:?} does not fit weight {w:?}",
            up.shape(),
            down.shape()
        );
        self.lora.insert(prefix.to_string(), (down, up));
        Ok(())
    }
}

/// Bumped whenever what a cache file holds changes for the same files.
const CACHE_FORMAT: &str = "1";

/// Where quantized weights are saved after their first load, so later loads
/// read them instead of the full-precision files (Z-Image-Turbo: 33 GB of
/// files, 6 GB at 4 bits) and skip quantizing. Entries are keyed by the
/// source files (path, size, modification time), the mixel version, dtype
/// and bits; saving one removes older entries for the same weights.
pub struct WeightCache {
    dir: PathBuf,
    notes: RefCell<Vec<String>>,
}

impl WeightCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            notes: RefCell::new(Vec::new()),
        }
    }

    /// What happened since the last call: entries saved, or failures to
    /// save or read them (which don't fail the load).
    pub fn take_notes(&self) -> Vec<String> {
        std::mem::take(&mut self.notes.borrow_mut())
    }

    fn note(&self, note: String) {
        self.notes.borrow_mut().push(note);
    }

    /// The cache file for `name` loaded from `files`, and the prefix it
    /// shares with older entries for the same weights.
    fn path(
        &self,
        name: &str,
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        quantize: Quantize,
    ) -> Result<(PathBuf, String)> {
        let mut key = Fnv::default();
        let mut source = Fnv::default();
        for part in [
            CACHE_FORMAT,
            env!("CARGO_PKG_VERSION"),
            name,
            &format!("{dtype:?}"),
        ] {
            key.write(part.as_bytes());
        }
        key.write(&[quantize.bits() as u8]);
        key.write(&GROUP_SIZE.to_le_bytes());
        for (i, file) in files.iter().enumerate() {
            // Hugging Face snapshot files are links to content-addressed blobs.
            let real = std::fs::canonicalize(file.as_ref())
                .with_context(|| format!("reading {}", file.as_ref().display()))?;
            let meta = std::fs::metadata(&real)?;
            let modified = meta
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            key.write(real.as_os_str().as_encoded_bytes());
            key.write(&meta.len().to_le_bytes());
            key.write(&modified.as_nanos().to_le_bytes());
            if i == 0 {
                if let Some(dir) = real.parent() {
                    source.write(dir.as_os_str().as_encoded_bytes());
                }
            }
        }
        let prefix = format!("{name}-q{}-{:08x}-", quantize.bits(), source.0 as u32);
        let path = self.dir.join(format!("{prefix}{:016x}.safetensors", key.0));
        Ok((path, prefix))
    }

    /// Removes entries with `prefix` other than `keep`; returns how many.
    fn remove_stale(&self, prefix: &str, keep: &Path) -> usize {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return 0;
        };
        let mut removed = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            let stale = path != keep
                && entry
                    .file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".safetensors"));
            if stale && std::fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        removed
    }
}

/// 64-bit FNV-1a, for cache keys.
struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self(0xcbf29ce484222325)
    }
}

impl Fnv {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
}

/// A 0-d array of `dtype`, so arithmetic with it keeps the other operand's dtype.
pub fn scalar(v: f32, dtype: Dtype) -> Result<Array> {
    Ok(Array::from_f32(v).as_dtype(dtype)?)
}

/// `x @ W^T (+ b)` for PyTorch-style `prefix.weight` / optional `prefix.bias`,
/// plus the layer's low-rank update if it has one ([`Weights::add_lora`]).
/// Quantized layers ([`Weights::load_quantized`]) run as quantized matmuls.
pub fn linear(x: &Array, w: &Weights, prefix: &str) -> Result<Array> {
    let mut y = match w.quantized.get(prefix) {
        Some(q) => {
            mlx_rs::ops::quantized_matmul(x, &q.w, &q.scales, &q.biases, true, GROUP_SIZE, q.bits)?
        }
        None => x.matmul(w.get(&format!("{prefix}.weight"))?.t())?,
    };
    let bias = format!("{prefix}.bias");
    if w.has(&bias) {
        y = y.add(w.get(&bias)?)?;
    }
    if let Some((down, up)) = w.lora.get(prefix) {
        y = y.add(x.matmul(down.t())?.matmul(up.t())?)?;
    }
    Ok(y)
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

/// Bytes of convolution input per band in [`conv_in_bands`]. At 1024x1024,
/// smaller bands barely lower the VAE's peak and decode more slowly (Z-Image:
/// 1.4 GiB above the weights in 2.3 s, against 6.8 GiB in 1.5 s unbanded).
const BAND_BYTES: usize = 512 << 20;

/// A stride-1 convolution of `pre(x)` (NHWC, `padding` on each side, plus
/// `bias`), computed in bands of rows when `x` is large.
///
/// MLX's convolutions allocate a workspace several times their input (a 3x3
/// at 1024x1024 with 256 channels: ~4.5 GiB), and `pre` (a norm and an
/// activation) makes full-size temporaries of its own. In bands, each with
/// the rows its kernel overlaps, both shrink to a band's worth. `pre` must
/// work pixel by pixel, so applying it to a band equals cropping its output.
/// The result matches one convolution exactly in f32; in bf16, MLX may pick
/// another algorithm for the smaller shape, which rounds differently.
pub fn conv_in_bands(
    x: &Array,
    weight: &Array,
    bias: Option<&Array>,
    padding: i32,
    pre: impl Fn(&Array) -> Result<Array>,
) -> Result<Array> {
    conv_in_bands_of(x, weight, bias, padding, pre, BAND_BYTES)
}

fn conv_in_bands_of(
    x: &Array,
    weight: &Array,
    bias: Option<&Array>,
    padding: i32,
    pre: impl Fn(&Array) -> Result<Array>,
    band_bytes: usize,
) -> Result<Array> {
    let (h, w, c) = (x.shape()[1], x.shape()[2], x.shape()[3]);
    let k = weight.shape()[1];
    let out_h = h + 2 * padding - k + 1;
    let row_bytes = (w * c * k * k) as usize * x.item_size();
    let rows = (band_bytes / row_bytes.max(1)).clamp(1, out_h.max(1) as usize) as i32;
    let conv = |x: &Array, padding: i32| -> Result<Array> {
        let y = mlx_rs::ops::conv2d(x, weight, (1, 1), (padding, padding), None, None)?;
        Ok(match bias {
            Some(b) => y.add(b)?,
            None => y,
        })
    };
    if rows >= out_h {
        return conv(&pre(x)?, padding);
    }
    let mut bands = Vec::new();
    for r in (0..out_h).step_by(rows as usize) {
        // Rows [r - padding, r + n + k - 1 - padding) of x, zero outside it.
        let n = rows.min(out_h - r);
        let (want_lo, want_hi) = (r - padding, r + n + k - 1 - padding);
        let (lo, hi) = (want_lo.max(0), want_hi.min(h));
        let slab = pre(&x.index((.., lo..hi, .., ..)))?;
        let slab = mlx_rs::ops::pad(
            &slab,
            &[
                (0, 0),
                (lo - want_lo, want_hi - hi),
                (padding, padding),
                (0, 0),
            ],
            None,
            None,
        )?;
        let y = conv(&slab, 0)?;
        // Evaluate now, so each band's workspace is freed before the next.
        y.eval()?;
        bands.push(y);
    }
    Ok(mlx_rs::ops::concatenate(&bands, 1)?)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_adds_the_low_rank_update() {
        // W = I (2x2), b = [1, 1], update = up (2x1) @ down (1x2)
        let mut w = Weights {
            map: HashMap::from([
                (
                    "l.weight".to_string(),
                    Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]),
                ),
                (
                    "l.bias".to_string(),
                    Array::from_slice(&[1.0f32, 1.0], &[2]),
                ),
            ]),
            quantized: HashMap::new(),
            lora: HashMap::new(),
        };
        let x = Array::from_slice(&[2.0f32, 3.0], &[1, 2]);
        let y = linear(&x, &w, "l").unwrap();
        y.eval().unwrap();
        assert_eq!(y.as_slice::<f32>(), &[3.0, 4.0]);

        let down = Array::from_slice(&[1.0f32, 1.0], &[1, 2]); // x . [1, 1] = 5
        let up = Array::from_slice(&[10.0f32, -1.0], &[2, 1]);
        w.add_lora("l", down, up).unwrap();
        let y = linear(&x, &w, "l").unwrap();
        y.eval().unwrap();
        assert_eq!(y.as_slice::<f32>(), &[53.0, -1.0]);

        // Shapes must fit the layer.
        let bad = Array::from_slice(&[1.0f32; 3], &[1, 3]);
        let up = Array::from_slice(&[1.0f32; 2], &[2, 1]);
        assert!(w.add_lora("l", bad, up.clone()).is_err());
        let down = Array::from_slice(&[1.0f32; 2], &[1, 2]);
        assert!(w.add_lora("missing", down, up).is_err());
    }

    /// Relative L2 distance between two arrays.
    fn rel_err(a: &Array, b: &Array) -> f32 {
        let a = a.as_dtype(Dtype::Float32).unwrap();
        let b = b.as_dtype(Dtype::Float32).unwrap();
        let d = a
            .subtract(&b)
            .unwrap()
            .square()
            .unwrap()
            .sum(None)
            .unwrap()
            .sqrt()
            .unwrap();
        let n = b.square().unwrap().sum(None).unwrap().sqrt().unwrap();
        let r = d.divide(&n).unwrap();
        r.eval().unwrap();
        r.as_slice::<f32>()[0]
    }

    #[test]
    fn quantized_layers_match_the_unquantized_ones() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("w.safetensors");
        let key = mlx_rs::random::key(0).unwrap();
        let weight = mlx_rs::random::normal::<f32>(&[128, 256][..], None, None, &key).unwrap();
        let bias = Array::from_slice(&[0.5f32; 128], &[128]);
        let embed = Array::from_slice(&[1.0f32; 64 * 64], &[64, 64]);
        Array::save_safetensors(
            [
                ("l.weight", &weight),
                ("l.bias", &bias),
                ("embed.weight", &embed),
            ],
            None,
            &file,
        )
        .unwrap();
        let x = mlx_rs::random::normal::<f32>(
            &[3, 256][..],
            None,
            None,
            &mlx_rs::random::key(1).unwrap(),
        )
        .unwrap();

        let dense = Weights::load(&[&file], Dtype::Float32, |_| true).unwrap();
        let expected = linear(&x, &dense, "l").unwrap();
        for (q, max_err) in [(Quantize::Q8, 0.01), (Quantize::Q4, 0.15)] {
            let mut w =
                Weights::load_quantized(&[&file], Dtype::Float32, |_| true, Some(q), |l| l == "l")
                    .unwrap();
            let err = rel_err(&linear(&x, &w, "l").unwrap(), &expected);
            assert!(err > 0.0 && err < max_err, "{q:?}: {err}");
            // Only the selected layers are quantized; the others are as loaded.
            assert!(w
                .get("l.weight")
                .unwrap_err()
                .to_string()
                .contains("quantized"));
            assert_eq!(w.shape("l.weight").unwrap(), [128, 256]);
            assert!(w.get("embed.weight").is_ok() && w.get("l.bias").is_ok());
            // A low-rank update still applies on top.
            let down = Array::from_slice(&[0.0f32; 256], &[1, 256]);
            let up = Array::from_slice(&[0.0f32; 128], &[128, 1]);
            w.add_lora("l", down, up).unwrap();
            assert_eq!(rel_err(&linear(&x, &w, "l").unwrap(), &expected), err);
        }
    }

    #[test]
    fn convolution_in_bands_matches_one_convolution() {
        let key = |i| mlx_rs::random::key(i).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 100, 16, 8][..], None, None, &key(0)).unwrap();
        let w3 = mlx_rs::random::normal::<f32>(&[4, 3, 3, 8][..], None, None, &key(1)).unwrap();
        let w1 = mlx_rs::random::normal::<f32>(&[4, 1, 1, 8][..], None, None, &key(2)).unwrap();
        let b = Array::from_slice(&[0.5f32; 4], &[4]);
        let one = |w: &Array, padding: i32| {
            mlx_rs::ops::conv2d(silu(&x).unwrap(), w, (1, 1), (padding, padding), None, None)
                .unwrap()
                .add(&b)
                .unwrap()
        };
        // A 3x3 input row is 16 px * 8 channels * 9 * 4 bytes: bands of 7 rows
        // (the last one 2), of 1 row, and a single band.
        let row = 16 * 8 * 9 * 4;
        for (w, padding) in [(&w3, 1), (&w1, 0)] {
            let expected = one(w, padding);
            for band_bytes in [7 * row, 1, usize::MAX] {
                let banded = conv_in_bands_of(&x, w, Some(&b), padding, silu, band_bytes).unwrap();
                assert_eq!(banded.shape(), expected.shape());
                assert!(rel_err(&banded, &expected) < 1e-6, "{band_bytes}");
            }
        }
    }

    #[test]
    fn quantized_weights_round_trip_through_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("w.safetensors");
        let key = mlx_rs::random::key(0).unwrap();
        let weight = mlx_rs::random::normal::<f32>(&[64, 128][..], None, None, &key).unwrap();
        let norm = Array::from_slice(&[1.0f32; 128], &[128]);
        let save = |w: &Array| {
            Array::save_safetensors([("l.weight", w), ("n.weight", &norm)], None, &src).unwrap()
        };
        save(&weight);
        let cache = WeightCache::new(dir.path().join("cache"));
        let load = || {
            Weights::load_quantized_cached(
                &[&src],
                Dtype::Float32,
                |_| true,
                Quantize::Q4,
                |l| l == "l",
                &cache,
                "test",
            )
            .unwrap()
        };
        let entries = || std::fs::read_dir(dir.path().join("cache")).map_or(0, |d| d.count());
        let x = mlx_rs::random::normal::<f32>(&[2, 128][..], None, None, &key).unwrap();

        // The first load quantizes and saves; the second reads the same back.
        let first = linear(&x, &load(), "l").unwrap();
        let notes = cache.take_notes();
        assert!(notes.len() == 1 && notes[0].contains("saved"), "{notes:?}");
        let second = load();
        assert!(cache.take_notes().is_empty());
        assert_eq!(rel_err(&linear(&x, &second, "l").unwrap(), &first), 0.0);
        assert_eq!(second.shape("l.weight").unwrap(), [64, 128]);
        assert!(second.get("n.weight").is_ok() && second.get("l.weight").is_err());

        // Changed source files get a new entry, which replaces the old one.
        std::thread::sleep(std::time::Duration::from_millis(10));
        save(&weight.multiply(Array::from_f32(2.0)).unwrap());
        let doubled = linear(&x, &load(), "l").unwrap();
        assert!(cache.take_notes()[0].contains("replacing an older copy"));
        assert_eq!(entries(), 1);
        assert!((rel_err(&doubled, &first) - 1.0).abs() < 1e-3);

        // An unreadable entry is rebuilt.
        let entry = std::fs::read_dir(dir.path().join("cache"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        std::fs::write(entry.path(), b"not safetensors").unwrap();
        let rebuilt = linear(&x, &load(), "l").unwrap();
        let notes = cache.take_notes();
        assert!(
            notes[0].contains("couldn't be read") && notes[1].contains("saved"),
            "{notes:?}"
        );
        assert_eq!(rel_err(&rebuilt, &doubled), 0.0);
    }
}

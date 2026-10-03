//! The `mixel` command: single images or JSONL batches on top of the
//! [`mixel`] library. The CLI and batch behavior mirror `candy`.
//!
//! ```bash
//! mixel --prompt "A beautiful landscape with mountains" --seed 42
//! mixel --input prompts.jsonl --output-dir out
//! mixel --init-image photo.jpg --strength 0.6 --prompt "a watercolor painting"
//! mixel --model qwen-image-2.1 --ref-image photo.jpg --prompt "Make it night"
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use image::ImageDecoder;
use mixel::{
    Encoded, GenerateOptions, LoadOptions, Lora, Model, Parts, Pipeline, Progress, Quantize,
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum ModelArg {
    /// Z-Image-Turbo: fast (9 steps); text-to-image and img2img
    #[value(name = "z-image-turbo", alias = "turbo")]
    ZImageTurbo,
    /// Qwen-Image-2.1: slower (40 steps); also edits with --ref-image
    #[value(name = "qwen-image-2.1", alias = "qwen")]
    QwenImage21,
    /// Qwen-Image-2.1 in 4 steps (Fun-Acc adapter): 7-10x faster, slightly
    /// softer fine detail; also edits with --ref-image
    #[value(name = "qwen-image-2.1-fast", alias = "qwen-fast")]
    QwenImage21Fast,
}

impl ModelArg {
    fn model(self) -> Model {
        match self {
            Self::ZImageTurbo => Model::ZImageTurbo,
            Self::QwenImage21 => Model::QwenImage21,
            Self::QwenImage21Fast => Model::QwenImage21Fast,
        }
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum QuantizeArg {
    /// 8 bits: about half the memory, practically the same images
    #[value(name = "8")]
    Q8,
    /// 4 bits: about a third of the memory, as good images but not the
    /// same ones for a given seed
    #[value(name = "4")]
    Q4,
}

impl QuantizeArg {
    fn quantize(self) -> Quantize {
        match self {
            Self::Q8 => Quantize::Q8,
            Self::Q4 => Quantize::Q4,
        }
    }
}

/// How much memory the encoded prompts (and Qwen's resized reference images)
/// may take before the batch generates them: ~0.3 MB per Z-Image-Turbo prompt,
/// ~10 MB per Qwen-Image-2.1 edit. Beyond it, the encoders load again.
const ENCODED_BUDGET: usize = 512 << 20;

/// Shown after the options by `mixel --help` (not `-h`): a usage guide for
/// scripts and AI agents.
const GUIDE: &str = "\
USAGE GUIDE (for scripts and AI agents)

Choosing a model:
  z-image-turbo (default)  ~59 s for 1024x1024 on an M3 Max (30-core GPU).
                           Use for text-to-image and img2img.
  qwen-image-2.1-fast      ~38 s for 1024x1024, ~58 s for an edit (4 steps).
  (alias qwen-fast)        Use to edit existing images (--ref-image). Fine
                           detail and small text are a little softer than
                           with qwen-image-2.1.
  qwen-image-2.1           ~6 min for 1024x1024, ~8 min for an edit (40
                           steps). Use when the fast variant's detail is
                           not enough.
  Both Qwen variants' weights are for research, not commercial use.

Recipes:
  Text to image:
    mixel --prompt \"a red fox in snow\" --seed 1 --output fox.png
  Edit a photo (keeps its content, changes what the prompt says):
    mixel --model qwen-image-2.1-fast --ref-image in.png \\
          --prompt \"make it night\" --seed 1 --output night.png
  Variation of an image (img2img; lower --strength = closer to it):
    mixel --init-image in.png --strength 0.7 --prompt \"...\" --seed 1 --output v.png
  Add a LoRA trained for the model (repeatable; FILE[:SCALE]):
    mixel --lora style.safetensors:0.8 --prompt \"<its trigger> ...\" --seed 1 --output s.png
  Many images (loads the model once; much faster than a loop):
    mixel --input jobs.jsonl --output-dir out --seed 1

Batch file (--input): one JSON object per line.
  {\"id\": \"fox\", \"prompt\": \"a red fox in snow\"}
  {\"id\": \"cat\", \"prompt\": \"a cat\", \"width\": 768, \"height\": 512, \"seed\": 7}
  {\"id\": \"night\", \"prompt\": \"make it night\", \"reference_images\": [\"in.png\"]}
  Only prompt is required. Output file: output if given, else <id>.png,
  else the line number (0003.png), under --output-dir. Other fields
  (negative_prompt, width, height, num_steps, guidance_scale, seed,
  init_image, strength, reference_images) default to the CLI flags. Image
  paths in the file are relative to the file. ids/outputs must be unique
  (ignoring case) and ids can't contain / or \\. --strength only applies
  to lines with an init image.

Where the image goes:
  - Always pass --prompt: without it a default landscape prompt is used.
  - Pass --seed for a predictable filename. Without it, a random seed is
    appended: --output a.png writes a-<seed>.png.
  - Parent directories are created. Single-image mode overwrites; batch mode
    skips outputs that already exist (--overwrite to regenerate).
  - Each saved image prints a line: \"Done! Image saved to <path> (<secs>s)\".
  - Without --output: z_image_output.png or qwen_image_output.png.
  - Output names must end in .png or .jpg (checked before loading).

Sizes: width and height must be multiples of 16 (z-image-turbo) or 32
(qwen models); default 1024x1024, or follows --init-image / --ref-image.
Smaller is faster: 512x512 takes ~14 s with z-image-turbo.

Prompts: z-image-turbo reads at most 512 tokens (a few hundred words) and
ignores the rest; the \"Token count\" line says when a prompt was cut.

Guidance (--guidance-scale, --negative-prompt) is off by default, as all
three models are meant to run; it doubles the time per step. z-image-turbo
turns it on above 0, steering away from --negative-prompt (or from the
empty prompt without one); a negative prompt alone is rejected.
qwen-image-2.1 turns it on above 1 and needs a negative prompt.

Running it:
  - Per 1024x1024 image: ~1 min (z-image-turbo), ~38-58 s
    (qwen-image-2.1-fast) or 6-8 min (qwen-image-2.1), plus a few seconds
    of model loading once the weights are cached. The first run downloads
    ~33 GB (z-image-turbo) or ~31 GB (qwen models; the fast variant adds
    0.35 GB) to ~/.cache/huggingface. Use long timeouts or run it in the
    background.
  - qwen-image-2.1-fast always runs 4 steps without guidance: --num-steps
    other than 4, or --guidance-scale above 1 with a negative prompt, is
    rejected.
  - Needs Apple Silicon and memory: at 1024x1024, ~14 GB peak for
    z-image-turbo and ~17-18 GB for the qwen models (GB = 2^30 bytes). Run
    one mixel at a time.
  - Less memory: --quantize 8 (~8 GB z-image-turbo, ~11-12 GB qwen) gives
    practically the same images; --quantize 4 (~5.4 GB, ~7.5-9 GB) gives
    images as good but not the same ones for a seed. Both are ~10-25%
    slower per step on an M3 Max (on an M4, 8 and 4 bits run at the same
    speed). The GPU can use only part of the memory (~74% on a 16 GB Mac);
    past that, steps run 2-3x slower. On a 16 GB Mac use z-image-turbo
    --quantize 8 (~155 s per 1024x1024 image on an M4 Mac mini) and the qwen
    models with --quantize 4 (~90 s, 4 steps); 32 GB runs everything in bf16.
  - With --quantize, the first run saves the quantized weights to
    ~/.cache/mixel/weights (6 GB for z-image-turbo at 4 bits, 8.7 GB for
    qwen) and later runs load them in seconds (16 GB Mac mini: 1-5 s instead
    of ~17 s); --no-cache skips it.
    Delete ~/.cache/mixel to free the space.
  - The text encoders run first for all images (a batch in chunks), then
    are freed before the transformer loads, so the two phases print two
    \"Loaded in\" lines.
  - Arguments and every batch line are validated before the model loads,
    so mistakes fail within a second.
  - Exit status 0 on success; non-zero on invalid input, a failed image,
    or any failed batch line (the others are still generated).
  - Progress goes to stdout, errors to stderr.";

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Text-to-image, img2img and image editing on Apple Silicon (MLX).",
    after_long_help = GUIDE
)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A beautiful landscape with mountains and a lake",
        conflicts_with = "input"
    )]
    prompt: String,

    /// What guidance steers away from; needs --guidance-scale (see there).
    #[arg(long, default_value = "")]
    negative_prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The height in pixels of the generated image [default: 1024, or from --ref-image / --init-image].
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image [default: 1024, or from --ref-image / --init-image].
    #[arg(long)]
    width: Option<usize>,

    /// Start from this image (img2img). Without --width/--height, the output
    /// keeps its aspect ratio, rounded to the model's size multiple, longest
    /// side <= 1024.
    #[arg(long)]
    init_image: Option<PathBuf>,

    /// Base the image on this reference image and edit it as the prompt says
    /// (qwen-image-2.1 only). Repeat for several images. Without
    /// --width/--height, the output takes the last image's aspect ratio at
    /// about 1024x1024 px.
    #[arg(long = "ref-image")]
    ref_images: Vec<PathBuf>,

    /// With --init-image: how much to change it, in (0, 1]. Low keeps the
    /// image close to the original; 1.0 ignores its content. [default: 0.6]
    #[arg(long)]
    strength: Option<f64>,

    /// Number of inference steps [default: 9 for z-image-turbo, 40 for
    /// qwen-image-2.1; qwen-image-2.1-fast always runs 4].
    #[arg(long)]
    num_steps: Option<usize>,

    /// Classifier-free guidance, as each model's reference pipeline defines
    /// it. z-image-turbo: on above 0 (default 0), against --negative-prompt or
    /// the empty prompt. qwen-image-2.1: on above 1 (default 1) with
    /// --negative-prompt. qwen-image-2.1-fast: none. Guidance runs the model
    /// twice per step.
    #[arg(long)]
    guidance_scale: Option<f64>,

    /// The seed to use when generating random samples. If omitted, a random
    /// seed is used and appended to the output filename, e.g. out-1234.png.
    #[arg(long)]
    seed: Option<u64>,

    /// Which model to use.
    #[arg(long, value_enum, default_value = "z-image-turbo")]
    model: ModelArg,

    /// Quantize the model's large layers to 8 or 4 bits as it loads, to fit
    /// in less memory. The first run saves them to ~/.cache/mixel/weights
    /// (~6 GB for z-image-turbo at 4 bits), so later runs load much faster.
    #[arg(long, value_enum)]
    quantize: Option<QuantizeArg>,

    /// With --quantize: don't save or use the quantized weights in
    /// ~/.cache/mixel/weights.
    #[arg(long, requires = "quantize")]
    no_cache: bool,

    /// Add a LoRA trained for the model (z-image-turbo or Qwen-Image-2.1): a
    /// .safetensors file, with an optional strength, e.g. --lora
    /// style.safetensors:0.8 [default: 1]. Repeat for several; their effects
    /// add up. Applies to every image of a batch.
    #[arg(long = "lora", value_name = "FILE[:SCALE]")]
    loras: Vec<String>,

    /// Override path to the model weights directory (uses HuggingFace by default).
    #[arg(long)]
    model_path: Option<String>,

    /// Output image filename [default: z_image_output.png or qwen_image_output.png].
    #[arg(long, conflicts_with = "input")]
    output: Option<String>,

    /// JSONL file with one image per line, e.g.
    /// {"id": "cat", "prompt": "a cat", "seed": 1, "width": 768}.
    /// Fields: prompt (required), id, negative_prompt, width, height, num_steps,
    /// guidance_scale, seed, output, init_image, strength, reference_images
    /// (a list). Omitted fields use the CLI values; omitted output defaults to
    /// the id (cat.png), else the line number (0003.png). Relative image paths
    /// are relative to the JSONL file.
    #[arg(long, short)]
    input: Option<PathBuf>,

    /// Directory for images generated from --input; relative outputs are placed here.
    #[arg(long, default_value = ".", requires = "input")]
    output_dir: PathBuf,

    /// With --input, regenerate images whose output file already exists.
    #[arg(long, requires = "input")]
    overwrite: bool,
}

impl Args {
    fn model(&self) -> Model {
        self.model.model()
    }
}

/// One image to generate, with all defaults resolved.
#[derive(Debug)]
struct Job {
    /// 1-based line number in the input file (0 for a single CLI job).
    line: usize,
    prompt: String,
    negative_prompt: String,
    width: usize,
    height: usize,
    num_steps: usize,
    guidance_scale: f64,
    seed: u64,
    init_image: Option<PathBuf>,
    strength: f64,
    reference_images: Vec<PathBuf>,
    model: Model,
    /// True when the seed was picked at random; it is then part of `output`.
    random_seed: bool,
    /// The output path before any random seed is appended.
    base_output: PathBuf,
    output: PathBuf,
}

/// One line of the JSONL input. Missing fields fall back to the CLI values.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JobSpec {
    prompt: String,
    /// Names the output file when `output` is absent; a string or a number.
    id: Option<serde_json::Value>,
    negative_prompt: Option<String>,
    width: Option<usize>,
    height: Option<usize>,
    num_steps: Option<usize>,
    guidance_scale: Option<f64>,
    seed: Option<u64>,
    output: Option<String>,
    init_image: Option<String>,
    strength: Option<f64>,
    reference_images: Option<Vec<String>>,
}

/// Default img2img strength: keeps the layout and colors, changes the rest.
const DEFAULT_STRENGTH: f64 = 0.6;
/// Longest side for sizes derived from an init image.
const MAX_AUTO_SIDE: f64 = 1024.0;

/// Rounds to the nearest multiple of `align`, at least `align`.
fn round_to_align(x: f64, align: usize) -> usize {
    ((x / align as f64).round() as usize).max(1) * align
}

/// A decoder for `path` and its EXIF orientation (phone photos are often
/// stored sideways with a tag saying how to turn them upright).
fn decoder(path: &Path) -> Result<(impl image::ImageDecoder, image::metadata::Orientation)> {
    let mut decoder = image::ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    Ok((decoder, orientation))
}

/// Size of the image at `path` as displayed, i.e. after its EXIF orientation.
fn dimensions(path: &Path, what: &str) -> Result<(u32, u32)> {
    use image::metadata::Orientation::*;
    let read = || -> Result<(u32, u32)> {
        let (decoder, orientation) = decoder(path)?;
        let (w, h) = decoder.dimensions();
        Ok(match orientation {
            Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH => (h, w),
            _ => (w, h),
        })
    };
    read().with_context(|| format!("reading {what} {}", path.display()))
}

/// Output size: explicit values win. Missing sides come from the last
/// reference image (at ~1024x1024 px, like Qwen-Image's own pipeline), else
/// from the init image's aspect ratio (longest side capped at 1024), else 1024.
fn resolve_size(
    width: Option<usize>,
    height: Option<usize>,
    init_image: Option<&Path>,
    reference_images: &[PathBuf],
    model: Model,
) -> Result<(usize, usize)> {
    // Check every reference image; the last one sets the size.
    let mut last = None;
    for r in reference_images {
        last = Some(dimensions(r, "reference image")?);
    }
    if let Some((rw, rh)) = last {
        let (cw, ch) = mixel::qwen21::pipeline::calculate_dimensions(rw, rh);
        return Ok((width.unwrap_or(cw), height.unwrap_or(ch)));
    }
    let Some(path) = init_image else {
        return Ok((width.unwrap_or(1024), height.unwrap_or(1024)));
    };
    let (iw, ih) = dimensions(path, "init image")?;
    let (iw, ih) = (iw as f64, ih as f64);
    let align = model.size_align();
    Ok(match (width, height) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => (w, round_to_align(w as f64 * ih / iw, align)),
        (None, Some(h)) => (round_to_align(h as f64 * iw / ih, align), h),
        (None, None) => {
            let scale = (MAX_AUTO_SIDE / iw.max(ih)).min(1.0);
            (
                round_to_align(iw * scale, align),
                round_to_align(ih * scale, align),
            )
        }
    })
}

/// Opens an image, upright per its EXIF orientation.
fn load_image(path: &Path, what: &str) -> Result<image::DynamicImage> {
    let open = || -> Result<image::DynamicImage> {
        let (decoder, orientation) = decoder(path)?;
        let mut img = image::DynamicImage::from_decoder(decoder)?;
        img.apply_orientation(orientation);
        Ok(img)
    };
    open().with_context(|| format!("reading {what} {}", path.display()))
}

/// Opens an image as RGB (see [`load_image`]), compositing any transparency
/// over white.
fn load_rgb(path: &Path, what: &str) -> Result<image::RgbImage> {
    let img = load_image(path, what)?;
    Ok(if img.color().has_alpha() {
        mixel::composite_over_white(&img.to_rgba8())
    } else {
        img.to_rgb8()
    })
}

impl Job {
    /// Library options with 1x1 stand-ins for the images: enough to validate
    /// and count steps without decoding them.
    fn settings(&self) -> GenerateOptions {
        GenerateOptions {
            prompt: self.prompt.clone(),
            negative_prompt: self.negative_prompt.clone(),
            width: self.width,
            height: self.height,
            num_steps: self.num_steps,
            guidance_scale: self.guidance_scale,
            seed: self.seed,
            init_image: self.init_image.as_ref().map(|_| image::RgbImage::new(1, 1)),
            strength: self.strength,
            reference_images: vec![image::RgbaImage::new(1, 1); self.reference_images.len()],
        }
    }

    /// Library options for this job, loading its images.
    fn options(&self) -> Result<GenerateOptions> {
        let init_image = match &self.init_image {
            Some(p) => Some(load_rgb(p, "init image")?),
            None => None,
        };
        let reference_images = self
            .reference_images
            .iter()
            .map(|p| Ok(load_image(p, "reference image")?.to_rgba8()))
            .collect::<Result<_>>()?;
        Ok(GenerateOptions {
            init_image,
            reference_images,
            ..self.settings()
        })
    }

    /// Checks the job before the model loads. Images are decoded to catch
    /// corrupt files, each path once across all jobs (tracked in `decoded`).
    fn validate(&self, decoded: &mut HashSet<PathBuf>) -> Result<()> {
        let ext = self.base_output.extension().and_then(|e| e.to_str());
        anyhow::ensure!(
            ext.is_some_and(|e| ["png", "jpg", "jpeg"].contains(&e.to_ascii_lowercase().as_str())),
            "output {} must end in .png or .jpg",
            self.base_output.display()
        );
        self.settings().validate(self.model)?;
        let init = self.init_image.iter().map(|p| (p, "init image"));
        let refs = self.reference_images.iter().map(|p| (p, "reference image"));
        for (path, what) in init.chain(refs) {
            if !decoded.contains(path) {
                load_image(path, what)?;
                decoded.insert(path.clone());
            }
        }
        Ok(())
    }

    /// Denoising steps that run (fewer than `num_steps` for img2img).
    fn steps_to_run(&self) -> usize {
        self.settings().steps_to_run()
    }

    /// Fills in the seed, picking a random one and appending it to the
    /// output filename when none was given.
    fn with_seed(mut self, seed: Option<u64>) -> Self {
        match seed {
            Some(seed) => self.seed = seed,
            None => {
                self.seed = random_seed();
                self.random_seed = true;
                self.output = seeded_path(&self.base_output, self.seed);
            }
        }
        self
    }

    /// An existing image for this job, if any. For random seeds this matches
    /// any `<stem>-<seed>.<ext>` file, since the seed differs on every run.
    fn existing_output(&self) -> Option<PathBuf> {
        if !self.random_seed {
            return self.output.exists().then(|| self.output.clone());
        }
        let dir = match self.base_output.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let stem = self.base_output.file_stem()?.to_str()?;
        let ext = self.base_output.extension().and_then(|e| e.to_str());
        std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| {
                let seed_part = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_prefix(stem))
                    .and_then(|s| s.strip_prefix('-'));
                let same_ext = p.extension().and_then(|e| e.to_str()) == ext;
                same_ext
                    && seed_part
                        .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            })
    }
}

fn random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    // RandomState is seeded from OS randomness; keep seeds short for filenames.
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    );
    hasher.finish() % u32::MAX as u64
}

/// `dir/name.png` -> `dir/name-<seed>.png`.
fn seeded_path(path: &Path, seed: u64) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let name = match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{stem}-{seed}.{ext}"),
        None => format!("{stem}-{seed}"),
    };
    path.with_file_name(name)
}

/// Strength for a job: explicit, else the img2img default, else 1.0.
fn resolve_strength(strength: Option<f64>, has_init_image: bool) -> f64 {
    strength.unwrap_or(if has_init_image {
        DEFAULT_STRENGTH
    } else {
        1.0
    })
}

/// The single-image output file: `--output`, else a per-model default.
fn default_output(args: &Args) -> PathBuf {
    match (&args.output, args.model()) {
        (Some(o), _) => PathBuf::from(o),
        (None, Model::ZImageTurbo) => PathBuf::from("z_image_output.png"),
        (None, Model::QwenImage21 | Model::QwenImage21Fast) => {
            PathBuf::from("qwen_image_output.png")
        }
    }
}

fn single_job(args: &Args) -> Result<Job> {
    let model = args.model();
    let init_image = args.init_image.clone();
    let reference_images = args.ref_images.clone();
    let (width, height) = resolve_size(
        args.width,
        args.height,
        init_image.as_deref(),
        &reference_images,
        model,
    )?;
    let output = default_output(args);
    Ok(Job {
        line: 0,
        prompt: args.prompt.clone(),
        negative_prompt: args.negative_prompt.clone(),
        width,
        height,
        num_steps: args.num_steps.unwrap_or_else(|| model.default_steps()),
        guidance_scale: args
            .guidance_scale
            .unwrap_or_else(|| model.default_guidance()),
        seed: 0,
        strength: resolve_strength(args.strength, init_image.is_some()),
        init_image,
        reference_images,
        model,
        random_seed: false,
        base_output: output.clone(),
        output,
    }
    .with_seed(args.seed))
}

/// The output filename for a line's `id`: `<id>.png`.
fn id_file_name(id: &serde_json::Value) -> Result<String> {
    let id = match id {
        serde_json::Value::String(s) => s.trim().to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        other => anyhow::bail!("id must be a string or a number, got {other}"),
    };
    anyhow::ensure!(!id.is_empty(), "id is empty");
    anyhow::ensure!(
        !id.contains(['/', '\\']),
        "id {id:?} contains a path separator; use output for subdirectories"
    );
    Ok(format!("{id}.png"))
}

/// Parses and validates every line of a JSONL file, reporting all problems at once.
fn read_jobs(path: &Path, args: &Args) -> Result<Vec<Job>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut jobs: Vec<Job> = Vec::new();
    let mut errors = Vec::new();
    let mut seen_outputs = std::collections::HashMap::new();
    let mut decoded = HashSet::new();

    for (idx, line) in content.lines().enumerate() {
        let line_no = idx + 1;
        if line.trim().is_empty() {
            continue;
        }
        let spec: JobSpec = match serde_json::from_str(line) {
            Ok(spec) => spec,
            Err(e) => {
                errors.push(format!("line {line_no}: {e}"));
                continue;
            }
        };
        let output = match (spec.output, spec.id) {
            (Some(output), _) => output,
            (None, Some(id)) => match id_file_name(&id) {
                Ok(name) => name,
                Err(e) => {
                    errors.push(format!("line {line_no}: {e:#}"));
                    continue;
                }
            },
            (None, None) => format!("{line_no:04}.png"),
        };
        let seed = spec.seed.or(args.seed);
        // Line paths are relative to the JSONL file; the CLI defaults to the cwd.
        let base = path.parent().unwrap_or(Path::new("."));
        let init_image = match spec.init_image {
            Some(p) => Some(base.join(p)),
            None => args.init_image.clone(),
        };
        let reference_images = match spec.reference_images {
            Some(list) => list.into_iter().map(|p| base.join(p)).collect(),
            None => args.ref_images.clone(),
        };
        let model = args.model();
        let size = resolve_size(
            spec.width.or(args.width),
            spec.height.or(args.height),
            init_image.as_deref(),
            &reference_images,
            model,
        );
        let (width, height) = match size {
            Ok(size) => size,
            Err(e) => {
                errors.push(format!("line {line_no}: {e:#}"));
                continue;
            }
        };
        let job = Job {
            line: line_no,
            prompt: spec.prompt,
            negative_prompt: spec
                .negative_prompt
                .unwrap_or_else(|| args.negative_prompt.clone()),
            width,
            height,
            num_steps: spec
                .num_steps
                .or(args.num_steps)
                .unwrap_or_else(|| model.default_steps()),
            guidance_scale: spec
                .guidance_scale
                .or(args.guidance_scale)
                .unwrap_or_else(|| model.default_guidance()),
            seed: 0,
            // --strength only applies to lines that have an init image.
            strength: resolve_strength(
                spec.strength
                    .or(args.strength.filter(|_| init_image.is_some())),
                init_image.is_some(),
            ),
            init_image,
            reference_images,
            model,
            random_seed: false,
            base_output: args.output_dir.join(&output),
            output: args.output_dir.join(output),
        }
        .with_seed(seed);
        if let Err(e) = job.validate(&mut decoded) {
            errors.push(format!("line {line_no}: {e:#}"));
            continue;
        }
        // macOS file systems ignore case: Fox.png and fox.png are one file.
        let key = job.base_output.to_string_lossy().to_lowercase();
        if let Some(prev) = seen_outputs.insert(key, line_no) {
            errors.push(format!(
                "line {line_no}: output {} is also used by line {prev}",
                job.base_output.display()
            ));
            continue;
        }
        jobs.push(job);
    }

    if !errors.is_empty() {
        anyhow::bail!(
            "{} invalid line(s) in {}:\n  {}",
            errors.len(),
            path.display(),
            errors.join("\n  ")
        );
    }
    anyhow::ensure!(!jobs.is_empty(), "no prompts found in {}", path.display());
    Ok(jobs)
}

fn run(args: Args) -> Result<()> {
    // LoRAs: parsed and checked (from their headers) before anything loads.
    let loras = args
        .loras
        .iter()
        .map(|s| Lora::parse(s))
        .collect::<Result<Vec<_>>>()?;
    let mut lora_lines = Vec::new();
    for lora in &loras {
        let layers = lora.layers()?;
        lora_lines.push(format!(
            "LoRA: {} (scale {}, {} layers)",
            lora.path.display(),
            lora.scale,
            layers.len()
        ));
    }

    let jobs = match &args.input {
        Some(path) => read_jobs(path, &args)?,
        None => {
            let job = single_job(&args)?;
            job.validate(&mut HashSet::new())?;
            vec![job]
        }
    };
    let batch = args.input.is_some();

    // In batch mode, skip images that already exist unless --overwrite is set.
    let mut todo = Vec::new();
    let mut skipped = Vec::new();
    for job in jobs {
        match job.existing_output() {
            Some(existing) if batch && !args.overwrite => {
                println!("Skipping line {}: {} exists", job.line, existing.display());
                skipped.push(job);
            }
            _ => todo.push(job),
        }
    }
    if todo.is_empty() {
        println!("Nothing to do (use --overwrite to regenerate).");
        return Ok(());
    }

    let model = args.model();
    println!("mixel: {model}");
    for line in &lora_lines {
        println!("{line}");
    }
    if let Some(path) = &args.input {
        println!(
            "Input: {} ({} to generate, {} skipped)",
            path.display(),
            todo.len(),
            skipped.len()
        );
    }

    let quantized = match args.quantize {
        Some(q) => format!(", quantized to {} bits", q.quantize().bits()),
        None => String::new(),
    };
    let source = match &args.model_path {
        Some(p) => format!("from {p}"),
        None => model.repo().to_string(),
    };
    let weight_cache = match std::env::var_os("HOME") {
        Some(home) if !args.no_cache => Some(PathBuf::from(home).join(".cache/mixel/weights")),
        _ => None,
    };
    let load = |parts: Parts| {
        let pipeline = Pipeline::load(&LoadOptions {
            model,
            repo: None,
            model_path: args.model_path.as_ref().map(PathBuf::from),
            cpu: args.cpu,
            quantize: args.quantize.map(QuantizeArg::quantize),
            parts,
            weight_cache: weight_cache.clone(),
            loras: if parts == Parts::Encoders {
                Vec::new()
            } else {
                loras.clone()
            },
        })?;
        for note in pipeline.notes() {
            println!("Cache: {note}");
        }
        anyhow::Ok(pipeline)
    };
    let encoders = if model.supports_reference_images() {
        "text and vision encoders"
    } else {
        "text encoder"
    };

    // The encoders and the transformer are never in memory together: encode
    // the prompts (up to a memory budget), free the encoders, then generate.
    let mut failed = Vec::new();
    let mut next = 0;
    while next < todo.len() {
        println!("\nLoading the {encoders} of {source}{quantized}...");
        let start = std::time::Instant::now();
        let encoder = load(Parts::Encoders)?;
        println!("Loaded in {:.1}s", start.elapsed().as_secs_f64());
        let start = std::time::Instant::now();
        let first = next;
        let mut encoded = Vec::new();
        let mut bytes = 0;
        while next < todo.len() && bytes < ENCODED_BUDGET {
            let result = todo[next].options().and_then(|opts| encoder.encode(&opts));
            if let Ok(e) = &result {
                bytes += e.nbytes();
            }
            encoded.push(result);
            next += 1;
        }
        drop(encoder);
        println!(
            "Encoded {} prompt(s) in {:.1}s",
            encoded.len(),
            start.elapsed().as_secs_f64()
        );

        println!("\nLoading model {source}{quantized}...");
        let start = std::time::Instant::now();
        let pipeline = load(Parts::Generator)?;
        println!("Loaded in {:.1}s", start.elapsed().as_secs_f64());

        for (i, (job, encoded)) in todo[first..next].iter().zip(encoded).enumerate() {
            println!(
                "\n[{}/{}] {}",
                first + i + 1,
                todo.len(),
                job.output.display()
            );
            println!("Prompt: {}", job.prompt);
            println!("Size: {}x{}", job.width, job.height);
            println!("Steps: {}", job.num_steps);
            println!("Guidance scale: {}", job.guidance_scale);
            let kind = if job.random_seed { " (random)" } else { "" };
            println!("Seed: {}{kind}", job.seed);
            if let Some(init) = &job.init_image {
                println!(
                    "Init image: {} (strength {}, {} of {} steps)",
                    init.display(),
                    job.strength,
                    job.steps_to_run(),
                    job.num_steps
                );
            }
            for r in &job.reference_images {
                println!("Reference image: {}", r.display());
            }
            let start = std::time::Instant::now();
            let result = match job.output.parent() {
                Some(parent) => std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display())),
                None => Ok(()),
            }
            .and(encoded)
            .and_then(|encoded| generate(&pipeline, job, &encoded));
            match result {
                Ok(()) => println!(
                    "Done! Image saved to {} ({:.1}s)",
                    job.output.display(),
                    start.elapsed().as_secs_f64()
                ),
                Err(e) if !batch => return Err(e),
                Err(e) => {
                    eprintln!("Line {} failed: {e:#}", job.line);
                    failed.push(job.line);
                }
            }
        }
    }

    if batch {
        println!(
            "\nBatch finished: {} generated, {} skipped, {} failed",
            todo.len() - failed.len(),
            skipped.len(),
            failed.len()
        );
        if !failed.is_empty() {
            anyhow::bail!("failed lines: {:?}", failed);
        }
    }
    Ok(())
}

/// Generates one job's image with progress output and saves it.
fn generate(pipeline: &Pipeline, job: &Job, encoded: &Encoded) -> Result<()> {
    let out = pipeline.generate_encoded_with(&job.options()?, encoded, |p| match p {
        Progress::Encoded { tokens } => {
            let cut = job.model == Model::ZImageTurbo
                && tokens >= mixel::zimage::pipeline::MAX_PROMPT_TOKENS;
            let note = if cut {
                " (the limit; the rest of the prompt is ignored)"
            } else {
                ""
            };
            println!("Token count: {tokens}{note}");
        }
        Progress::Step {
            step,
            total,
            t,
            sigma,
        } => println!("Step {step}/{total}: t = {t:.4}, sigma = {sigma:.4}"),
        Progress::Decoding => println!("Decoding latents with VAE..."),
    })?;
    let t = out.timings;
    println!(
        "Timings: text {:.1}s, init image {:.1}s, denoise {:.1}s ({:.1}s/step), VAE {:.1}s",
        t.text.as_secs_f64(),
        t.init_image.as_secs_f64(),
        t.denoise.as_secs_f64(),
        t.denoise.as_secs_f64() / job.steps_to_run().max(1) as f64,
        t.vae.as_secs_f64()
    );
    // JPEG has no alpha: composite transparent (Qwen-Image) output over white.
    let png = job
        .output
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("png"));
    let image = match out.image {
        img if img.color().has_alpha() && !png => {
            mixel::composite_over_white(&img.to_rgba8()).into()
        }
        img => img,
    };
    image
        .save(&job.output)
        .with_context(|| format!("saving {}", job.output.display()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Args {
        let mut argv = vec!["candy"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).unwrap()
    }

    fn batch_args(input: &Path, extra: &[&str]) -> Args {
        let mut argv = vec!["--input", input.to_str().unwrap()];
        argv.extend_from_slice(extra);
        args(&argv)
    }

    fn write_jsonl(dir: &Path, content: &str) -> PathBuf {
        let path = dir.join("prompts.jsonl");
        std::fs::write(&path, content).unwrap();
        path
    }

    fn job(width: usize, height: usize) -> Job {
        Job {
            line: 1,
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            width,
            height,
            num_steps: 9,
            guidance_scale: 5.0,
            seed: 0,
            init_image: None,
            strength: 1.0,
            reference_images: Vec::new(),
            model: Model::ZImageTurbo,
            random_seed: false,
            base_output: "out.png".into(),
            output: "out.png".into(),
        }
    }

    // ---------- seeds and filenames ----------

    #[test]
    fn seeded_path_inserts_seed_before_extension() {
        assert_eq!(
            seeded_path(Path::new("out/a.png"), 42),
            PathBuf::from("out/a-42.png")
        );
        assert_eq!(seeded_path(Path::new("a"), 7), PathBuf::from("a-7"));
        assert_eq!(
            seeded_path(Path::new("x.y.png"), 1),
            PathBuf::from("x.y-1.png")
        );
    }

    #[test]
    fn random_seed_fits_in_u32_and_varies() {
        let seeds: std::collections::HashSet<u64> = (0..20).map(|_| random_seed()).collect();
        assert!(seeds.iter().all(|&s| s < u32::MAX as u64));
        assert!(seeds.len() > 1, "20 random seeds were all equal");
    }

    #[test]
    fn single_job_with_explicit_seed_keeps_output_name() {
        let job = single_job(&args(&["--seed", "5", "--output", "x.png"])).unwrap();
        assert_eq!(job.seed, 5);
        assert!(!job.random_seed);
        assert_eq!(job.output, PathBuf::from("x.png"));
    }

    #[test]
    fn single_job_without_seed_appends_random_seed() {
        let job = single_job(&args(&["--output", "x.png"])).unwrap();
        assert!(job.random_seed);
        assert_eq!(job.output, PathBuf::from(format!("x-{}.png", job.seed)));
        assert_eq!(job.base_output, PathBuf::from("x.png"));
    }

    #[test]
    fn single_job_uses_model_default_steps() {
        assert_eq!(single_job(&args(&[])).unwrap().num_steps, 9);
        assert_eq!(
            single_job(&args(&["--num-steps", "4"])).unwrap().num_steps,
            4
        );
    }

    // ---------- JSONL parsing ----------

    #[test]
    fn read_jobs_applies_cli_defaults_and_line_overrides() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_jsonl(
            dir.path(),
            r#"{"prompt": "a", "seed": 1}
{"prompt": "b", "seed": 2, "width": 768, "height": 512, "num_steps": 4, "guidance_scale": 2.5, "negative_prompt": "blur"}
"#,
        );
        let a = batch_args(
            &input,
            &[
                "--width",
                "512",
                "--height",
                "512",
                "--negative-prompt",
                "ugly",
                "--guidance-scale",
                "1.5",
            ],
        );
        let jobs = read_jobs(&input, &a).unwrap();

        assert_eq!(jobs.len(), 2);
        let (j1, j2) = (&jobs[0], &jobs[1]);
        assert_eq!((j1.width, j1.height, j1.num_steps), (512, 512, 9));
        assert_eq!(j1.negative_prompt, "ugly");
        assert_eq!(j1.guidance_scale, 1.5);
        assert_eq!((j2.width, j2.height, j2.num_steps), (768, 512, 4));
        assert_eq!(j2.negative_prompt, "blur");
        assert_eq!(j2.guidance_scale, 2.5);
    }

    #[test]
    fn read_jobs_names_outputs_by_file_line_number() {
        let dir = tempfile::tempdir().unwrap();
        // Blank lines are skipped but still count, so names match the file's line numbers.
        let input = write_jsonl(
            dir.path(),
            "{\"prompt\": \"a\", \"seed\": 1}\n\n{\"prompt\": \"b\", \"seed\": 2, \"output\": \"sub/b.png\"}\n{\"prompt\": \"c\", \"seed\": 3}\n",
        );
        let jobs = read_jobs(&input, &batch_args(&input, &["--output-dir", "out"])).unwrap();
        let outputs: Vec<_> = jobs.iter().map(|j| j.output.clone()).collect();
        assert_eq!(
            outputs,
            vec![
                PathBuf::from("out/0001.png"),
                PathBuf::from("out/sub/b.png"),
                PathBuf::from("out/0004.png")
            ]
        );
        assert_eq!(
            jobs.iter().map(|j| j.line).collect::<Vec<_>>(),
            vec![1, 3, 4]
        );
    }

    #[test]
    fn read_jobs_names_outputs_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_jsonl(
            dir.path(),
            concat!(
                "{\"id\": \"fox\", \"prompt\": \"a\", \"seed\": 1}\n",
                "{\"id\": 42, \"prompt\": \"b\", \"seed\": 2}\n",
                "{\"id\": \"x\", \"prompt\": \"c\", \"seed\": 3, \"output\": \"c.png\"}\n",
                "{\"prompt\": \"d\", \"seed\": 4}\n",
            ),
        );
        let jobs = read_jobs(&input, &batch_args(&input, &["--output-dir", "out"])).unwrap();
        let outputs: Vec<_> = jobs.iter().map(|j| j.output.clone()).collect();
        assert_eq!(
            outputs,
            vec![
                PathBuf::from("out/fox.png"),
                PathBuf::from("out/42.png"),
                PathBuf::from("out/c.png"),
                PathBuf::from("out/0004.png")
            ]
        );
    }

    #[test]
    fn read_jobs_rejects_bad_and_duplicate_ids() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_jsonl(
            dir.path(),
            concat!(
                "{\"id\": \"a\", \"prompt\": \"a\"}\n",
                "{\"id\": \"a\", \"prompt\": \"b\"}\n",
                "{\"id\": \" \", \"prompt\": \"c\"}\n",
                "{\"id\": [1], \"prompt\": \"d\"}\n",
            ),
        );
        let err = read_jobs(&input, &batch_args(&input, &[]))
            .unwrap_err()
            .to_string();
        for expected in [
            "line 2: output ./a.png is also used by line 1",
            "line 3: id is empty",
            "line 4: id must be a string or a number",
        ] {
            assert!(err.contains(expected), "missing {expected:?} in:\n{err}");
        }
    }

    #[test]
    fn read_jobs_rejects_ids_with_path_separators() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_jsonl(
            dir.path(),
            concat!(
                "{\"id\": \"AC/DC\", \"prompt\": \"a\"}\n",
                "{\"id\": \"..\\\\x\", \"prompt\": \"b\"}\n",
            ),
        );
        let err = read_jobs(&input, &batch_args(&input, &[]))
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("2 invalid line(s)"), "{err}");
        assert!(
            err.contains("line 1: id \"AC/DC\" contains a path separator"),
            "{err}"
        );
    }

    #[test]
    fn read_jobs_rejects_outputs_differing_only_in_case() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_jsonl(
            dir.path(),
            "{\"id\": \"Fox\", \"prompt\": \"a\"}\n{\"id\": \"fox\", \"prompt\": \"b\"}\n",
        );
        let err = read_jobs(&input, &batch_args(&input, &[]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("line 2: output ./fox.png is also used by line 1"),
            "{err}"
        );
    }

    #[test]
    fn read_jobs_checks_output_format_before_loading() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_jsonl(
            dir.path(),
            concat!(
                "{\"prompt\": \"a\", \"output\": \"a.webp\"}\n",
                "{\"prompt\": \"b\", \"output\": \"noext\"}\n",
                "{\"prompt\": \"c\", \"output\": \"\"}\n",
                "{\"prompt\": \"d\", \"output\": \"D.JPG\"}\n",
            ),
        );
        let err = read_jobs(&input, &batch_args(&input, &[]))
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("3 invalid line(s)"), "{err}");
        for line in 1..=3 {
            assert!(err.contains(&format!("line {line}: output")), "{err}");
        }
        assert!(err.contains("must end in .png or .jpg"), "{err}");
    }

    #[test]
    fn cli_strength_only_applies_to_lines_with_an_init_image() {
        let dir = tempfile::tempdir().unwrap();
        write_png(dir.path(), "in.png", 64, 64);
        let input = write_jsonl(
            dir.path(),
            "{\"prompt\": \"a\"}\n{\"prompt\": \"b\", \"init_image\": \"in.png\"}\n",
        );
        let jobs = read_jobs(&input, &batch_args(&input, &["--strength", "0.5"])).unwrap();
        assert_eq!((jobs[0].strength, jobs[1].strength), (1.0, 0.5));
    }

    #[test]
    fn read_jobs_seed_precedence_and_random_suffix() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_jsonl(
            dir.path(),
            "{\"prompt\": \"a\", \"seed\": 9}\n{\"prompt\": \"b\"}\n",
        );

        // Line seed wins over CLI seed; CLI seed fills in missing ones.
        let jobs = read_jobs(&input, &batch_args(&input, &["--seed", "100"])).unwrap();
        assert_eq!((jobs[0].seed, jobs[1].seed), (9, 100));
        assert!(!jobs[1].random_seed);
        assert_eq!(jobs[1].output, PathBuf::from("./0002.png"));

        // No seed anywhere: random, and appended to the filename.
        let jobs = read_jobs(&input, &batch_args(&input, &[])).unwrap();
        assert!(jobs[1].random_seed);
        assert_eq!(
            jobs[1].output,
            PathBuf::from(format!("./0002-{}.png", jobs[1].seed))
        );
    }

    #[test]
    fn read_jobs_reports_all_errors_with_line_numbers() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_jsonl(
            dir.path(),
            r#"{"prompt": "ok"}
{"prompt": "bad size", "width": 1000}
{"promt": "typo"}
not json
{"prompt": "dup", "output": "x.png"}
{"prompt": "dup", "output": "x.png"}
{"prompt": ""}
"#,
        );
        let err = read_jobs(&input, &batch_args(&input, &[]))
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("5 invalid line(s)"), "{err}");
        for expected in [
            "line 2: Image dimensions",
            "line 3: unknown field `promt`",
            "line 4:",
            "line 6: output ./x.png is also used by line 5",
            "line 7: prompt is empty",
        ] {
            assert!(err.contains(expected), "missing {expected:?} in:\n{err}");
        }
        assert!(!err.contains("line 1:"), "{err}");
    }

    #[test]
    fn read_jobs_rejects_duplicate_outputs_even_with_random_seeds() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_jsonl(
            dir.path(),
            "{\"prompt\": \"a\", \"output\": \"x.png\"}\n{\"prompt\": \"b\", \"output\": \"x.png\"}\n",
        );
        let err = read_jobs(&input, &batch_args(&input, &[]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("also used by line 1"), "{err}");
    }

    #[test]
    fn read_jobs_rejects_empty_and_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_jsonl(dir.path(), "\n  \n");
        let err = read_jobs(&input, &batch_args(&input, &[]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no prompts found"), "{err}");

        let missing = dir.path().join("nope.jsonl");
        let err = read_jobs(&missing, &batch_args(&missing, &[]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("reading"), "{err}");
    }

    // ---------- resume (existing outputs) ----------

    #[test]
    fn existing_output_with_fixed_seed_checks_exact_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.png");
        let mut j = job(512, 512);
        j.base_output = path.clone();
        j.output = path.clone();
        assert_eq!(j.existing_output(), None);
        std::fs::write(&path, b"").unwrap();
        assert_eq!(j.existing_output(), Some(path));
    }

    #[test]
    fn existing_output_with_random_seed_matches_any_seed_suffix() {
        let dir = tempfile::tempdir().unwrap();
        let mut j = job(512, 512);
        j.base_output = dir.path().join("0001.png");
        let j = j.with_seed(None);
        assert_eq!(j.existing_output(), None);

        // Files that must not count as an existing output for 0001.png.
        for name in [
            "0001.png",
            "0001-.png",
            "0001-abc.png",
            "0001-12.jpg",
            "00012-5.png",
            "0002-5.png",
        ] {
            std::fs::write(dir.path().join(name), b"").unwrap();
        }
        assert_eq!(j.existing_output(), None);

        let hit = dir.path().join("0001-123456.png");
        std::fs::write(&hit, b"").unwrap();
        assert_eq!(j.existing_output(), Some(hit));
    }

    #[test]
    fn existing_output_handles_missing_directory() {
        let mut j = job(512, 512);
        j.base_output = PathBuf::from("/definitely/not/here/0001.png");
        assert_eq!(j.with_seed(None).existing_output(), None);
    }

    // ---------- img2img ----------

    fn write_png(dir: &Path, name: &str, w: u32, h: u32) -> PathBuf {
        let path = dir.join(name);
        image::RgbImage::from_pixel(w, h, image::Rgb([200, 100, 50]))
            .save(&path)
            .unwrap();
        path
    }

    #[test]
    fn round_to_align_rounds_to_nearest_16() {
        assert_eq!(round_to_align(1000.0, 16), 1008);
        assert_eq!(round_to_align(1024.0, 16), 1024);
        assert_eq!(round_to_align(575.9, 16), 576);
        assert_eq!(round_to_align(3.0, 16), 16);
    }

    #[test]
    fn resolve_size_without_image_defaults_to_1024() {
        assert_eq!(
            resolve_size(None, None, None, &[], Model::ZImageTurbo).unwrap(),
            (1024, 1024)
        );
        assert_eq!(
            resolve_size(Some(512), None, None, &[], Model::ZImageTurbo).unwrap(),
            (512, 1024)
        );
    }

    #[test]
    fn resolve_size_follows_image_aspect_ratio() {
        let dir = tempfile::tempdir().unwrap();
        let img = write_png(dir.path(), "wide.png", 400, 300);
        // Small images keep their size (rounded), large ones are capped at 1024.
        assert_eq!(
            resolve_size(None, None, Some(&img), &[], Model::ZImageTurbo).unwrap(),
            (400, 304)
        );
        let big = write_png(dir.path(), "big.png", 3000, 2000);
        assert_eq!(
            resolve_size(None, None, Some(&big), &[], Model::ZImageTurbo).unwrap(),
            (1024, 688)
        );
        // One explicit side: the other follows the aspect ratio.
        assert_eq!(
            resolve_size(Some(800), None, Some(&img), &[], Model::ZImageTurbo).unwrap(),
            (800, 608)
        );
        assert_eq!(
            resolve_size(None, Some(600), Some(&img), &[], Model::ZImageTurbo).unwrap(),
            (800, 600)
        );
        // Both explicit: used as-is (the image is cropped to fill).
        assert_eq!(
            resolve_size(Some(512), Some(512), Some(&img), &[], Model::ZImageTurbo).unwrap(),
            (512, 512)
        );
    }

    #[test]
    fn resolve_size_reports_missing_image() {
        let err = resolve_size(
            None,
            None,
            Some(Path::new("/no/such.png")),
            &[],
            Model::ZImageTurbo,
        )
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("reading init image /no/such.png"),
            "{err:#}"
        );
    }

    #[test]
    fn single_job_with_init_image_uses_default_strength_and_image_size() {
        let dir = tempfile::tempdir().unwrap();
        let img = write_png(dir.path(), "in.png", 640, 480);
        let job = single_job(&args(&[
            "--init-image",
            img.to_str().unwrap(),
            "--seed",
            "1",
        ]))
        .unwrap();
        assert_eq!((job.width, job.height), (640, 480));
        assert_eq!(job.strength, DEFAULT_STRENGTH);
        assert_eq!(job.steps_to_run(), 6); // int(9 - 5.4) = 3 skipped
        job.validate(&mut HashSet::new()).unwrap();

        let opts = job.options().unwrap();
        assert_eq!(opts.init_image.unwrap().dimensions(), (640, 480));
    }

    #[test]
    fn strength_without_init_image_is_rejected() {
        let job = single_job(&args(&["--strength", "0.5", "--seed", "1"])).unwrap();
        let err = job.validate(&mut HashSet::new()).unwrap_err().to_string();
        assert!(err.contains("needs an init image"), "{err}");
    }

    #[test]
    fn read_jobs_resolves_init_images_relative_to_jsonl_and_checks_them() {
        let dir = tempfile::tempdir().unwrap();
        write_png(dir.path(), "a.png", 320, 320);
        let input = write_jsonl(
            dir.path(),
            r#"{"prompt": "a", "seed": 1, "init_image": "a.png", "strength": 0.3}
{"prompt": "b", "seed": 2, "init_image": "missing.png"}
{"prompt": "c", "seed": 3, "init_image": "a.png", "strength": 1.5}
"#,
        );
        // Run from another directory: paths must resolve against the JSONL file.
        let err = read_jobs(&input, &batch_args(&input, &[]))
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("2 invalid line(s)"), "{err}");
        assert!(err.contains("line 2: reading init image"), "{err}");
        assert!(err.contains("line 3: strength must be in (0, 1]"), "{err}");

        // A file whose header reads but whose pixels don't is caught up front.
        let png = std::fs::read(dir.path().join("a.png")).unwrap();
        std::fs::write(dir.path().join("cut.png"), &png[..png.len() / 2]).unwrap();
        let cut = write_jsonl(dir.path(), r#"{"prompt": "a", "init_image": "cut.png"}"#);
        let err = read_jobs(&cut, &batch_args(&cut, &[]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("line 1: reading init image"), "{err}");

        let ok = write_jsonl(
            dir.path(),
            r#"{"prompt": "a", "seed": 1, "init_image": "a.png", "strength": 0.3}"#,
        );
        let jobs = read_jobs(&ok, &batch_args(&ok, &[])).unwrap();
        assert_eq!(
            jobs[0].init_image.as_deref(),
            Some(dir.path().join("a.png").as_path())
        );
        assert_eq!(
            (jobs[0].width, jobs[0].height, jobs[0].strength),
            (320, 320, 0.3)
        );
        assert_eq!(jobs[0].steps_to_run(), 3); // int(9 - 2.7) = 6 skipped
    }

    // ---------- models ----------

    #[test]
    fn model_flag_accepts_names_and_aliases() {
        assert_eq!(args(&[]).model(), Model::ZImageTurbo);
        assert_eq!(args(&["--model", "turbo"]).model(), Model::ZImageTurbo);
        assert_eq!(
            args(&["--model", "qwen-image-2.1"]).model(),
            Model::QwenImage21
        );
        assert_eq!(args(&["--model", "qwen"]).model(), Model::QwenImage21);
        assert_eq!(
            args(&["--model", "qwen-image-2.1-fast"]).model(),
            Model::QwenImage21Fast
        );
        assert_eq!(
            args(&["--model", "qwen-fast"]).model(),
            Model::QwenImage21Fast
        );
        assert!(Args::try_parse_from(["candy", "--model", "sdxl"]).is_err());
    }

    #[test]
    fn qwen_uses_its_own_defaults() {
        let job = single_job(&args(&["--model", "qwen-image-2.1", "--seed", "1"])).unwrap();
        assert_eq!((job.num_steps, job.guidance_scale), (40, 1.0));
        assert_eq!(job.output, PathBuf::from("qwen_image_output.png"));
        let job = single_job(&args(&["--seed", "1"])).unwrap();
        assert_eq!((job.num_steps, job.guidance_scale), (9, 0.0));
        assert_eq!(job.output, PathBuf::from("z_image_output.png"));
        let job = single_job(&args(&["--model", "qwen-fast", "--seed", "1"])).unwrap();
        assert_eq!((job.num_steps, job.guidance_scale), (4, 1.0));
        assert_eq!(job.output, PathBuf::from("qwen_image_output.png"));
    }

    #[test]
    fn fast_qwen_rejects_other_step_counts_before_loading() {
        let job = single_job(&args(&[
            "--model",
            "qwen-fast",
            "--num-steps",
            "8",
            "--seed",
            "1",
        ]))
        .unwrap();
        let err = job.validate(&mut HashSet::new()).unwrap_err().to_string();
        assert!(err.contains("always runs 4 steps"), "{err}");
    }

    #[test]
    fn qwen_sizes_must_be_multiples_of_32() {
        let job = single_job(&args(&[
            "--model", "qwen", "--width", "1008", "--seed", "1",
        ]))
        .unwrap();
        let err = job.validate(&mut HashSet::new()).unwrap_err().to_string();
        assert!(err.contains("divisible by 32"), "{err}");
        let job = single_job(&args(&["--width", "1008", "--seed", "1"])).unwrap();
        job.validate(&mut HashSet::new()).unwrap();
    }

    #[test]
    fn reference_images_set_the_size_and_need_qwen() {
        let dir = tempfile::tempdir().unwrap();
        let wide = write_png(dir.path(), "wide.png", 1920, 1080);
        let sq = write_png(dir.path(), "sq.png", 300, 300);
        let (w, s) = (wide.to_str().unwrap(), sq.to_str().unwrap());
        // The last reference image decides the aspect ratio, at ~1024^2 px.
        let job = single_job(&args(&[
            "--model",
            "qwen",
            "--ref-image",
            s,
            "--ref-image",
            w,
            "--seed",
            "1",
        ]))
        .unwrap();
        assert_eq!((job.width, job.height), (1376, 768));
        assert_eq!(job.reference_images.len(), 2);
        job.validate(&mut HashSet::new()).unwrap();
        // An explicit side wins.
        let job = single_job(&args(&[
            "--model",
            "qwen",
            "--ref-image",
            w,
            "--width",
            "512",
            "--seed",
            "1",
        ]))
        .unwrap();
        assert_eq!((job.width, job.height), (512, 768));

        let job = single_job(&args(&["--ref-image", s, "--seed", "1"])).unwrap();
        let err = job.validate(&mut HashSet::new()).unwrap_err().to_string();
        assert!(err.contains("does not take reference images"), "{err}");
    }

    #[test]
    fn read_jobs_resolves_reference_images_relative_to_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        write_png(dir.path(), "a.png", 640, 640);
        let input = write_jsonl(
            dir.path(),
            r#"{"prompt": "edit", "seed": 1, "reference_images": ["a.png"]}
{"prompt": "bad", "seed": 2, "reference_images": ["nope.png"]}
"#,
        );
        let err = read_jobs(&input, &batch_args(&input, &["--model", "qwen"]))
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("1 invalid line(s)"), "{err}");
        assert!(err.contains("line 2: reading reference image"), "{err}");

        let ok = write_jsonl(
            dir.path(),
            r#"{"prompt": "edit", "seed": 1, "reference_images": ["a.png"]}"#,
        );
        let jobs = read_jobs(&ok, &batch_args(&ok, &["--model", "qwen"])).unwrap();
        assert_eq!(jobs[0].reference_images, vec![dir.path().join("a.png")]);
        assert_eq!((jobs[0].width, jobs[0].height), (1024, 1024));
    }

    #[test]
    fn images_are_turned_upright_by_exif_orientation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("phone.jpg");
        // Stored 64x32 (landscape) with Orientation = 6: rotate 90 to display.
        #[rustfmt::skip]
        let exif = vec![
            b'I', b'I', 42, 0, 8, 0, 0, 0, // TIFF header, first IFD at 8
            1, 0,                          // one entry
            0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, // Orientation, SHORT, 1, 6
            0, 0, 0, 0,                    // no next IFD
        ];
        let mut img = image::RgbImage::from_pixel(64, 32, image::Rgb([0, 0, 0]));
        img.put_pixel(0, 0, image::Rgb([255, 255, 255]));
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
            std::fs::File::create(&path).unwrap(),
            100,
        );
        image::ImageEncoder::set_exif_metadata(&mut encoder, exif).unwrap();
        encoder.encode_image(&img).unwrap();

        assert_eq!(dimensions(&path, "image").unwrap(), (32, 64));
        let rgb = load_rgb(&path, "image").unwrap();
        assert_eq!(rgb.dimensions(), (32, 64));
        // Rotating 90 clockwise moves the top-left pixel to the top-right.
        assert!(
            rgb.get_pixel(31, 0).0[0] > 200,
            "{:?}",
            rgb.get_pixel(31, 0)
        );
        assert!(rgb.get_pixel(0, 0).0[0] < 50, "{:?}", rgb.get_pixel(0, 0));
    }

    #[test]
    fn load_rgb_composites_transparency_over_white() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.png");
        let mut img = image::RgbaImage::from_pixel(2, 1, image::Rgba([0, 0, 0, 0]));
        img.put_pixel(1, 0, image::Rgba([10, 20, 30, 255]));
        img.save(&path).unwrap();
        let rgb = load_rgb(&path, "image").unwrap();
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(rgb.get_pixel(1, 0).0, [10, 20, 30]);
    }

    // ---------- CLI ----------

    #[test]
    fn cli_rejects_conflicting_and_batch_only_flags() {
        let parse = |a: &[&str]| Args::try_parse_from([&["candy"], a].concat());
        assert!(parse(&["--input", "p.jsonl", "--prompt", "x"]).is_err());
        assert!(parse(&["--input", "p.jsonl", "--output", "x.png"]).is_err());
        assert!(parse(&["--overwrite"]).is_err());
        assert!(parse(&["--output-dir", "out"]).is_err());
        assert!(parse(&["-i", "p.jsonl", "--output-dir", "out", "--overwrite"]).is_ok());
    }
}

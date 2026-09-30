//! mixel: Z-Image-Turbo text-to-image on Apple Silicon with mlx-rs.
//!
//! The model is a port of candle-transformers' `z_image` (see `src/zimage`);
//! the CLI and JSONL batch mode mirror `candy` (candle-diffusion).
//!
//! ```bash
//! cargo run --release -- --prompt "A beautiful landscape with mountains" \
//!     --height 1024 --width 1024
//! ```

mod zimage;

use anyhow::{Context, Error as E, Result};
use clap::Parser;
use mlx_rs::{Array, Dtype};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use zimage::{
    scheduler::Scheduler, text_encoder::TextEncoder, transformer::Transformer, vae::Decoder,
    ModelFiles,
};

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum Model {
    /// Z-Image-Turbo: optimized for fast inference (8-9 steps)
    Turbo,
}

impl Model {
    fn repo(&self) -> &'static str {
        match self {
            Self::Turbo => "Tongyi-MAI/Z-Image-Turbo",
        }
    }

    fn default_steps(&self) -> usize {
        match self {
            Self::Turbo => 9,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A beautiful landscape with mountains and a lake",
        conflicts_with = "input"
    )]
    prompt: String,

    /// The negative prompt (for CFG).
    #[arg(long, default_value = "")]
    negative_prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The height in pixels of the generated image.
    #[arg(long, default_value_t = 1024)]
    height: usize,

    /// The width in pixels of the generated image.
    #[arg(long, default_value_t = 1024)]
    width: usize,

    /// Number of inference steps.
    #[arg(long)]
    num_steps: Option<usize>,

    /// Guidance scale for CFG.
    #[arg(long, default_value_t = 5.0)]
    guidance_scale: f64,

    /// The seed to use when generating random samples. If omitted, a random
    /// seed is used and appended to the output filename, e.g. out-1234.png.
    #[arg(long)]
    seed: Option<u64>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value = "turbo")]
    model: Model,

    /// Override path to the model weights directory (uses HuggingFace by default).
    #[arg(long)]
    model_path: Option<String>,

    /// Output image filename.
    #[arg(long, default_value = "z_image_output.png", conflicts_with = "input")]
    output: String,

    /// JSONL file with one image per line, e.g.
    /// {"prompt": "a cat", "seed": 1, "width": 768, "output": "cat.png"}.
    /// Fields: prompt (required), negative_prompt, width, height, num_steps,
    /// guidance_scale, seed, output. Omitted fields use the CLI values;
    /// omitted output defaults to the line number, e.g. 0003.png.
    #[arg(long, short)]
    input: Option<PathBuf>,

    /// Directory for images generated from --input; relative outputs are placed here.
    #[arg(long, default_value = ".", requires = "input")]
    output_dir: PathBuf,

    /// With --input, regenerate images whose output file already exists.
    #[arg(long, requires = "input")]
    overwrite: bool,
}

/// Format user prompt for Qwen3 chat template
/// Corresponds to add_generation_prompt=True, enable_thinking=True
///
/// Format:
/// <|im_start|>user
/// {prompt}<|im_end|>
/// <|im_start|>assistant
fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
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
    negative_prompt: Option<String>,
    width: Option<usize>,
    height: Option<usize>,
    num_steps: Option<usize>,
    guidance_scale: Option<f64>,
    seed: Option<u64>,
    output: Option<String>,
}

impl Job {
    fn validate(&self) -> Result<()> {
        let vae_align = 16; // vae_scale_factor * 2 = 8 * 2 = 16
        anyhow::ensure!(!self.prompt.trim().is_empty(), "prompt is empty");
        anyhow::ensure!(self.num_steps > 0, "num_steps must be at least 1");
        if !self.height.is_multiple_of(vae_align) || !self.width.is_multiple_of(vae_align) {
            anyhow::bail!(
                "Image dimensions must be divisible by {}. Got {}x{}. \
                 Try {}x{} or {}x{} instead.",
                vae_align,
                self.width,
                self.height,
                (self.width / vae_align) * vae_align,
                (self.height / vae_align) * vae_align,
                ((self.width / vae_align) + 1) * vae_align,
                ((self.height / vae_align) + 1) * vae_align
            );
        }
        Ok(())
    }
}

impl Job {
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

/// Standard normal noise from a seeded CPU RNG. Identical to candy's, so the
/// same seed starts both tools from the same latents.
fn seeded_noise(seed: u64, shape: [usize; 4]) -> Vec<f32> {
    use rand::SeedableRng;
    use rand_distr::Distribution;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    rand_distr::StandardNormal
        .sample_iter(&mut rng)
        .take(shape.iter().product())
        .collect()
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

fn single_job(args: &Args) -> Job {
    Job {
        line: 0,
        prompt: args.prompt.clone(),
        negative_prompt: args.negative_prompt.clone(),
        width: args.width,
        height: args.height,
        num_steps: args.num_steps.unwrap_or_else(|| args.model.default_steps()),
        guidance_scale: args.guidance_scale,
        seed: 0,
        random_seed: false,
        base_output: PathBuf::from(&args.output),
        output: PathBuf::from(&args.output),
    }
    .with_seed(args.seed)
}

/// Parses and validates every line of a JSONL file, reporting all problems at once.
fn read_jobs(path: &Path, args: &Args) -> Result<Vec<Job>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut jobs: Vec<Job> = Vec::new();
    let mut errors = Vec::new();
    let mut seen_outputs = std::collections::HashMap::new();

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
        let output = spec.output.unwrap_or_else(|| format!("{line_no:04}.png"));
        let seed = spec.seed.or(args.seed);
        let job = Job {
            line: line_no,
            prompt: spec.prompt,
            negative_prompt: spec
                .negative_prompt
                .unwrap_or_else(|| args.negative_prompt.clone()),
            width: spec.width.unwrap_or(args.width),
            height: spec.height.unwrap_or(args.height),
            num_steps: spec
                .num_steps
                .or(args.num_steps)
                .unwrap_or_else(|| args.model.default_steps()),
            guidance_scale: spec.guidance_scale.unwrap_or(args.guidance_scale),
            seed: 0,
            random_seed: false,
            base_output: args.output_dir.join(&output),
            output: args.output_dir.join(output),
        }
        .with_seed(seed);
        if let Err(e) = job.validate() {
            errors.push(format!("line {line_no}: {e}"));
            continue;
        }
        if let Some(prev) = seen_outputs.insert(job.base_output.clone(), line_no) {
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

/// The loaded models, reused across every image in a run.
struct Pipeline {
    dtype: Dtype,
    tokenizer: Tokenizer,
    text_encoder: TextEncoder,
    transformer: Transformer,
    vae: Decoder,
}

impl Pipeline {
    fn load(args: &Args) -> Result<Self> {
        if args.cpu {
            mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        }
        let dtype = Dtype::Bfloat16;
        let files = ModelFiles::new(args.model.repo(), args.model_path.as_deref())?;
        match &args.model_path {
            Some(p) => println!("\nLoading models from local path: {p}"),
            None => println!("\nLoading model from HuggingFace: {}", args.model.repo()),
        }

        println!("Loading tokenizer...");
        let tokenizer =
            Tokenizer::from_file(files.get("tokenizer/tokenizer.json")?).map_err(E::msg)?;

        println!("Loading text encoder...");
        let te_files = (1..=3)
            .map(|i| files.get(&format!("text_encoder/model-{i:05}-of-00003.safetensors")))
            .collect::<Result<Vec<_>>>()?;
        let text_encoder = TextEncoder::load(&te_files, dtype)?;

        println!("Loading transformer...");
        let tr_files = (1..=3)
            .map(|i| {
                files.get(&format!(
                    "transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors"
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let transformer = Transformer::load(&tr_files, dtype)?;

        println!("Loading VAE...");
        let vae = Decoder::load(files.get("vae/diffusion_pytorch_model.safetensors")?, dtype)?;

        Ok(Self {
            dtype,
            tokenizer,
            text_encoder,
            transformer,
            vae,
        })
    }

    fn encode_prompt(&self, prompt: &str) -> Result<Array> {
        let tokens = self
            .tokenizer
            .encode(format_prompt_for_qwen3(prompt).as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        println!("Token count: {}", tokens.len());
        let feats = self.text_encoder.forward(&tokens)?;
        feats.eval()?;
        Ok(feats)
    }

    fn generate(&self, job: &Job) -> Result<()> {
        let dtype = self.dtype;
        let scalar = |v: f64| zimage::scalar(v as f32, dtype);

        let started = std::time::Instant::now();
        println!("\nEncoding prompt...");
        let cap_feats = self.encode_prompt(&job.prompt)?;
        let neg_cap_feats = if !job.negative_prompt.is_empty() && job.guidance_scale > 1.0 {
            Some(self.encode_prompt(&job.negative_prompt)?)
        } else {
            None
        };

        // latent = 2 * (image_size // 16): divisible by the patch size, and 8x VAE upsampling.
        let latent_h = 2 * (job.height / 16);
        let latent_w = 2 * (job.width / 16);
        println!("Latent size: {latent_w}x{latent_h}");

        let shape = [1, 16, latent_h, latent_w];
        let noise = seeded_noise(job.seed, shape);
        let mut latents = Array::from_slice(&noise, &shape.map(|d| d as i32)).as_dtype(dtype)?;

        let encoded = std::time::Instant::now();
        let mut scheduler = Scheduler::new(job.num_steps);
        println!("\nStarting denoising loop ({} steps)...", job.num_steps);
        for step in 0..job.num_steps {
            let t = scheduler.current_timestep_normalized();
            let mut pred = self.transformer.forward(&latents, t as f32, &cap_feats)?;
            if let Some(neg) = &neg_cap_feats {
                // CFG: pred = neg + scale * (pos - neg)
                let neg_pred = self.transformer.forward(&latents, t as f32, neg)?;
                pred = neg_pred.add(
                    pred.subtract(&neg_pred)?
                        .multiply(scalar(job.guidance_scale)?)?,
                )?;
            }
            // Z-Image predicts the negated velocity; Euler step: x + dt * v.
            let dt = scheduler.step_dt();
            latents = latents.add(pred.negative()?.multiply(scalar(dt)?)?)?;
            latents.eval()?;
            println!(
                "Step {}/{}: t = {:.4}, sigma = {:.4}",
                step + 1,
                job.num_steps,
                t,
                scheduler.current_sigma()
            );
        }

        let denoised = std::time::Instant::now();
        println!("\nDecoding latents with VAE...");
        let image = self.vae.decode(&latents.transpose_axes(&[0, 2, 3, 1])?)?;
        // [-1, 1] -> [0, 255], computed in the model dtype like candle.
        let image = mlx_rs::ops::clip(&image, (-1.0f32, 1.0f32))?
            .add(scalar(1.0)?)?
            .multiply(scalar(127.5)?)?
            .as_dtype(Dtype::Uint8)?;
        image.eval()?;
        println!(
            "Timings: text {:.1}s, denoise {:.1}s ({:.1}s/step), VAE {:.1}s",
            (encoded - started).as_secs_f64(),
            (denoised - encoded).as_secs_f64(),
            (denoised - encoded).as_secs_f64() / job.num_steps as f64,
            denoised.elapsed().as_secs_f64()
        );

        println!("Saving image to {}...", job.output.display());
        save_image(&image, &job.output)?;
        Ok(())
    }
}

fn run(args: Args) -> Result<()> {
    let jobs = match &args.input {
        Some(path) => read_jobs(path, &args)?,
        None => {
            let job = single_job(&args);
            job.validate()?;
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

    println!("Z-Image Text-to-Image Generation");
    println!("================================");
    println!("Model: {:?}", args.model);
    if let Some(path) = &args.input {
        println!(
            "Input: {} ({} to generate, {} skipped)",
            path.display(),
            todo.len(),
            skipped.len()
        );
    }

    let pipeline = Pipeline::load(&args)?;

    let mut failed = Vec::new();
    for (i, job) in todo.iter().enumerate() {
        println!("\n[{}/{}] {}", i + 1, todo.len(), job.output.display());
        println!("Prompt: {}", job.prompt);
        println!("Size: {}x{}", job.width, job.height);
        println!("Steps: {}", job.num_steps);
        println!("Guidance scale: {}", job.guidance_scale);
        let kind = if job.random_seed { " (random)" } else { "" };
        println!("Seed: {}{kind}", job.seed);
        if let Some(parent) = job.output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let start = std::time::Instant::now();
        match pipeline.generate(job) {
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

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

/// Saves a (1, H, W, 3) u8 array as an image.
fn save_image<P: AsRef<std::path::Path>>(img: &Array, p: P) -> Result<()> {
    let sh = img.shape();
    anyhow::ensure!(
        sh.len() == 4 && sh[3] == 3,
        "save_image expects (1, H, W, 3), got {sh:?}"
    );
    let (height, width) = (sh[1] as u32, sh[2] as u32);
    let img = img.contiguous()?;
    img.eval()?;
    let pixels = img.as_slice::<u8>().to_vec();
    let image: image::RgbImage = image::ImageBuffer::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow::anyhow!("error saving image {:?}", p.as_ref()))?;
    image.save(p)?;
    Ok(())
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
            random_seed: false,
            base_output: "out.png".into(),
            output: "out.png".into(),
        }
    }

    // ---------- validation ----------

    #[test]
    fn validate_accepts_multiples_of_16() {
        for (w, h) in [(1024, 1024), (512, 512), (384, 512), (640, 368), (16, 16)] {
            job(w, h).validate().unwrap();
        }
    }

    #[test]
    fn validate_rejects_bad_dimensions_with_suggestion() {
        let err = job(1000, 1024).validate().unwrap_err().to_string();
        assert!(err.contains("divisible by 16"), "{err}");
        assert!(err.contains("992x1024"), "{err}");
    }

    #[test]
    fn validate_rejects_empty_prompt_and_zero_steps() {
        let mut j = job(512, 512);
        j.prompt = "   ".into();
        assert!(j
            .validate()
            .unwrap_err()
            .to_string()
            .contains("prompt is empty"));

        let mut j = job(512, 512);
        j.num_steps = 0;
        assert!(j.validate().unwrap_err().to_string().contains("num_steps"));
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
        let job = single_job(&args(&["--seed", "5", "--output", "x.png"]));
        assert_eq!(job.seed, 5);
        assert!(!job.random_seed);
        assert_eq!(job.output, PathBuf::from("x.png"));
    }

    #[test]
    fn single_job_without_seed_appends_random_seed() {
        let job = single_job(&args(&["--output", "x.png"]));
        assert!(job.random_seed);
        assert_eq!(job.output, PathBuf::from(format!("x-{}.png", job.seed)));
        assert_eq!(job.base_output, PathBuf::from("x.png"));
    }

    #[test]
    fn single_job_uses_model_default_steps() {
        assert_eq!(single_job(&args(&[])).num_steps, 9);
        assert_eq!(single_job(&args(&["--num-steps", "4"])).num_steps, 4);
    }

    // ---------- noise ----------

    #[test]
    fn seeded_noise_is_reproducible() {
        let shape = [1, 16, 8, 8];
        assert_eq!(seeded_noise(1, shape), seeded_noise(1, shape));
        assert_ne!(seeded_noise(1, shape), seeded_noise(2, shape));
        assert_eq!(seeded_noise(1, shape).len(), 16 * 64);
    }

    #[test]
    fn seeded_noise_is_standard_normal() {
        let v = seeded_noise(3, [1, 16, 64, 64]);
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((std - 1.0).abs() < 0.02, "std {std}");
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
            ],
        );
        let jobs = read_jobs(&input, &a).unwrap();

        assert_eq!(jobs.len(), 2);
        let (j1, j2) = (&jobs[0], &jobs[1]);
        assert_eq!((j1.width, j1.height, j1.num_steps), (512, 512, 9));
        assert_eq!(j1.negative_prompt, "ugly");
        assert_eq!(j1.guidance_scale, 5.0);
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

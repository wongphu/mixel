//! The end-to-end text-to-image pipeline.

use crate::nn::{ModelFiles, Quantize, WeightCache};
use crate::qwen21::pipeline::QwenPipeline;
use crate::zimage::pipeline::ZImagePipeline;
use anyhow::Result;
use mlx_rs::Array;
use std::path::PathBuf;
use std::time::Duration;

/// Hugging Face repo of Z-Image-Turbo, the default model.
pub const DEFAULT_REPO: &str = "Tongyi-MAI/Z-Image-Turbo";
/// Denoising steps Z-Image-Turbo is tuned for.
pub const DEFAULT_STEPS: usize = 9;
/// Z-Image-Turbo's size multiple (VAE 8x * patch size 2).
pub const SIZE_ALIGN: usize = 16;

/// A supported text-to-image model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Model {
    /// Z-Image-Turbo: fast (9 steps), text-to-image and img2img.
    #[default]
    ZImageTurbo,
    /// Qwen-Image-2.1: slower (40 steps), text-to-image, img2img, and editing
    /// with reference images.
    QwenImage21,
    /// Qwen-Image-2.1 with the 4-step Fun-Acc adapter ([`crate::qwen21::fast`]):
    /// the same tasks 7-10x faster, with slightly softer fine detail.
    QwenImage21Fast,
}

impl Model {
    pub const ALL: [Model; 3] = [
        Model::ZImageTurbo,
        Model::QwenImage21,
        Model::QwenImage21Fast,
    ];

    /// Short name, as used on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Model::ZImageTurbo => "z-image-turbo",
            Model::QwenImage21 => "qwen-image-2.1",
            Model::QwenImage21Fast => "qwen-image-2.1-fast",
        }
    }

    /// Hugging Face repo of the weights (the fast variant also downloads
    /// its adapter from [`crate::qwen21::fast::REPO`]).
    pub fn repo(self) -> &'static str {
        match self {
            Model::ZImageTurbo => DEFAULT_REPO,
            Model::QwenImage21 | Model::QwenImage21Fast => crate::qwen21::REPO,
        }
    }

    pub fn default_steps(self) -> usize {
        match self {
            Model::ZImageTurbo => DEFAULT_STEPS,
            Model::QwenImage21 => crate::qwen21::DEFAULT_STEPS,
            Model::QwenImage21Fast => crate::qwen21::fast::STEPS,
        }
    }

    /// Default guidance scale: off for every model. Each follows its
    /// reference pipeline's convention (see [`GenerateOptions::guidance_scale`]),
    /// so off is 0 for Z-Image-Turbo and 1 for Qwen-Image-2.1.
    pub fn default_guidance(self) -> f64 {
        match self {
            Model::ZImageTurbo => 0.0,
            Model::QwenImage21 | Model::QwenImage21Fast => 1.0,
        }
    }

    /// Width and height must be multiples of this.
    pub fn size_align(self) -> usize {
        match self {
            Model::ZImageTurbo => SIZE_ALIGN,
            Model::QwenImage21 | Model::QwenImage21Fast => crate::qwen21::SIZE_ALIGN,
        }
    }

    /// Whether the model can be conditioned on reference images (editing).
    pub fn supports_reference_images(self) -> bool {
        matches!(self, Model::QwenImage21 | Model::QwenImage21Fast)
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Model {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Model::ALL
            .into_iter()
            .find(|m| m.name() == s)
            .ok_or_else(|| {
                let names: Vec<_> = Model::ALL.iter().map(|m| m.name()).collect();
                anyhow::anyhow!("unknown model {s:?}; expected one of {}", names.join(", "))
            })
    }
}

/// Which parts of a model [`Pipeline::load`] loads.
///
/// The text encoders only run at the start of each image, yet are a third
/// (Z-Image-Turbo) to half (Qwen-Image-2.1, with its vision encoder) of the
/// weights. To keep them out of memory while the transformer runs, load the
/// [`Encoders`](Parts::Encoders), [`Pipeline::encode`] each image's options,
/// drop that pipeline, then load the [`Generator`](Parts::Generator) and
/// [`Pipeline::generate_encoded`] each one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Parts {
    /// Everything: [`Pipeline::generate`] works.
    #[default]
    All,
    /// The tokenizer and text encoder (and Qwen's vision encoder):
    /// [`Pipeline::encode`] works.
    Encoders,
    /// The transformer and VAE: [`Pipeline::generate_encoded`] works.
    Generator,
}

impl Parts {
    pub(crate) fn encoders(self) -> bool {
        self != Parts::Generator
    }

    pub(crate) fn generator(self) -> bool {
        self != Parts::Encoders
    }
}

/// Which model to load, from where, and on which device.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    pub model: Model,
    /// Hugging Face repo id to use instead of the model's own.
    pub repo: Option<String>,
    /// A local copy of the repo, instead of the Hugging Face cache.
    pub model_path: Option<PathBuf>,
    /// Run on the CPU instead of the GPU.
    pub cpu: bool,
    /// Quantize the text encoder and transformer as they load, to use less
    /// memory (see [`Quantize`]).
    pub quantize: Option<Quantize>,
    /// Which parts to load (all by default).
    pub parts: Parts,
    /// With `quantize`: a directory to save the quantized weights in after
    /// their first load, and load them from afterwards, instead of reading
    /// and quantizing the full-precision files each time (Z-Image-Turbo: 33
    /// GB of files, 6 GB at 4 bits). Off by default: the library writes no
    /// files unless asked. [`Pipeline::notes`] says what it saved.
    pub weight_cache: Option<PathBuf>,
}

/// What to generate.
#[derive(Clone, PartialEq)]
pub struct GenerateOptions {
    pub prompt: String,
    /// What guidance steers away from (see `guidance_scale`).
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub num_steps: usize,
    /// Classifier-free guidance, in each model's own convention:
    /// - Z-Image-Turbo (diffusers' `ZImagePipeline`): `pos + s * (pos - neg)`,
    ///   on when `s > 0`, against the empty prompt if `negative_prompt` is
    ///   empty. Turbo is distilled to run without it (0).
    /// - Qwen-Image-2.1 (true CFG): `neg + s * (pos - neg)`, on when `s > 1`
    ///   and `negative_prompt` is set. The 4-step variant runs without it.
    ///
    /// Guidance runs the model twice per step.
    pub guidance_scale: f64,
    /// Seeds the initial noise; the same seed and options give the same image.
    pub seed: u64,
    /// Start from this image instead of pure noise (img2img). It is resized
    /// (center-cropped to fill) to `width` x `height` if needed.
    pub init_image: Option<image::RgbImage>,
    /// How much of `init_image` to replace, in (0, 1]: the fraction of the
    /// denoising steps that run. 1.0 ignores the image's content entirely.
    /// Must be 1.0 without an `init_image`.
    pub strength: f64,
    /// Images the result should be based on (editing), for models that
    /// support it (Qwen-Image-2.1). Each is resized to about 1024x1024 px at
    /// its own aspect ratio. Like the reference pipeline, the VAE sees the
    /// alpha channel and the vision encoder a copy composited over white.
    pub reference_images: Vec<image::RgbaImage>,
}

impl std::fmt::Debug for GenerateOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerateOptions")
            .field("prompt", &self.prompt)
            .field("negative_prompt", &self.negative_prompt)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("num_steps", &self.num_steps)
            .field("guidance_scale", &self.guidance_scale)
            .field("seed", &self.seed)
            .field(
                "init_image",
                &self.init_image.as_ref().map(|i| i.dimensions()),
            )
            .field("strength", &self.strength)
            .field(
                "reference_images",
                &self
                    .reference_images
                    .iter()
                    .map(|i| i.dimensions())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl GenerateOptions {
    /// Z-Image-Turbo defaults: 1024x1024, 9 steps, seed 0.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self::for_model(Model::ZImageTurbo, prompt)
    }

    /// `model`'s defaults at 1024x1024, seed 0.
    pub fn for_model(model: Model, prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: String::new(),
            width: 1024,
            height: 1024,
            num_steps: model.default_steps(),
            guidance_scale: model.default_guidance(),
            seed: 0,
            init_image: None,
            strength: 1.0,
            reference_images: Vec::new(),
        }
    }

    /// Denoising steps that actually run: all of them for text-to-image,
    /// fewer for img2img with `strength < 1`.
    pub fn steps_to_run(&self) -> usize {
        match self.init_image {
            Some(_) => self
                .num_steps
                .saturating_sub(crate::zimage::scheduler::start_index(
                    self.num_steps,
                    self.strength,
                )),
            None => self.num_steps,
        }
    }

    /// Checks the options for `model`.
    pub fn validate(&self, model: Model) -> Result<()> {
        let a = model.size_align();
        anyhow::ensure!(
            self.reference_images.is_empty() || model.supports_reference_images(),
            "{model} does not take reference images; use {}",
            Model::QwenImage21
        );
        anyhow::ensure!(!self.prompt.trim().is_empty(), "prompt is empty");
        anyhow::ensure!(self.num_steps > 0, "num_steps must be at least 1");
        if model == Model::ZImageTurbo {
            anyhow::ensure!(
                self.guidance_scale >= 0.0,
                "guidance_scale must be at least 0, got {}",
                self.guidance_scale
            );
            anyhow::ensure!(
                self.negative_prompt.is_empty() || self.guidance_scale > 0.0,
                "a negative prompt needs guidance_scale above 0 ({model} runs without guidance by default)"
            );
        }
        if model == Model::QwenImage21Fast {
            let steps = crate::qwen21::fast::STEPS;
            anyhow::ensure!(
                self.num_steps == steps,
                "{model} always runs {steps} steps (its distilled schedule), got {}",
                self.num_steps
            );
            anyhow::ensure!(
                self.negative_prompt.is_empty() || self.guidance_scale <= 1.0,
                "{model} runs without guidance: drop the negative prompt or set guidance_scale to 1"
            );
        }
        anyhow::ensure!(
            self.width > 0 && self.height > 0,
            "width and height must be positive, got {}x{}",
            self.width,
            self.height
        );
        match &self.init_image {
            Some(img) => {
                anyhow::ensure!(
                    self.strength > 0.0 && self.strength <= 1.0,
                    "strength must be in (0, 1], got {}",
                    self.strength
                );
                anyhow::ensure!(img.width() > 0 && img.height() > 0, "init image is empty");
            }
            None => anyhow::ensure!(
                self.strength == 1.0,
                "strength {} needs an init image",
                self.strength
            ),
        }
        if !self.height.is_multiple_of(a) || !self.width.is_multiple_of(a) {
            // Round only the sides that need it, and never down to 0.
            let down = |x: usize| (x / a * a).max(a);
            let up = |x: usize| x.div_ceil(a) * a;
            let (lo, hi) = (
                (down(self.width), down(self.height)),
                (up(self.width), up(self.height)),
            );
            let suggestion = if lo == hi {
                format!("{}x{}", lo.0, lo.1)
            } else {
                format!("{}x{} or {}x{}", lo.0, lo.1, hi.0, hi.1)
            };
            anyhow::bail!(
                "Image dimensions must be divisible by {a}. Got {}x{}. Try {suggestion} instead.",
                self.width,
                self.height,
            );
        }
        if model == Model::ZImageTurbo {
            let max = crate::zimage::transformer::MAX_SIDE;
            anyhow::ensure!(
                self.width <= max && self.height <= max,
                "{model} supports at most {max}x{max}, got {}x{}",
                self.width,
                self.height
            );
        }
        Ok(())
    }
}

/// Progress reported by [`Pipeline::generate_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Progress {
    /// The prompt was encoded into this many tokens.
    Encoded { tokens: usize },
    /// A denoising step finished (1-based `step`).
    Step {
        step: usize,
        total: usize,
        t: f64,
        sigma: f64,
    },
    /// Denoising is done and the VAE is decoding the image.
    Decoding,
}

/// Wall-clock time spent in each phase.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub text: Duration,
    /// Encoding the init image with the VAE (zero for text-to-image).
    pub init_image: Duration,
    pub denoise: Duration,
    pub vae: Duration,
}

/// A generated image and how long it took.
pub struct Generated {
    /// RGB, or RGBA when Qwen-Image-2.1's output has any transparency.
    pub image: image::DynamicImage,
    pub timings: Timings,
}

/// The loaded models, reusable across any number of images.
///
/// ```no_run
/// let pipeline = mixel::Pipeline::load(&Default::default())?;
/// let mut opts = mixel::GenerateOptions::new("a red fox in fresh snow");
/// opts.seed = 42;
/// pipeline.generate(&opts)?.image.save("fox.png")?;
/// # anyhow::Ok(())
/// ```
pub struct Pipeline {
    model: Model,
    inner: Inner,
    notes: Vec<String>,
}

enum Inner {
    ZImage(Box<ZImagePipeline>),
    Qwen(Box<QwenPipeline>),
}

impl Pipeline {
    /// Loads the model's weights (downloading them on first use).
    ///
    /// This turns off MLX's buffer cache for the process
    /// (`mlx_rs::memory::set_cache_limit(0)`): MLX otherwise keeps freed
    /// buffers for reuse, which added up to 10 GiB to the peak (most of it
    /// while decoding) for no measurable speedup.
    pub fn load(opts: &LoadOptions) -> Result<Self> {
        if opts.cpu {
            mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        }
        mlx_rs::memory::set_cache_limit(0)?;
        let repo = opts.repo.as_deref().unwrap_or(opts.model.repo());
        let files = ModelFiles::new(repo, opts.model_path.as_deref())?;
        let cache = opts.weight_cache.as_ref().map(WeightCache::new);
        let cache = cache.as_ref();
        let inner = match opts.model {
            Model::ZImageTurbo => Inner::ZImage(Box::new(ZImagePipeline::load(
                &files,
                opts.quantize,
                opts.parts,
                cache,
            )?)),
            Model::QwenImage21 => Inner::Qwen(Box::new(QwenPipeline::load(
                &files,
                None,
                opts.quantize,
                opts.parts,
                cache,
            )?)),
            Model::QwenImage21Fast => {
                let adapter = ModelFiles::new(crate::qwen21::fast::REPO, None)?;
                Inner::Qwen(Box::new(QwenPipeline::load(
                    &files,
                    Some(&adapter),
                    opts.quantize,
                    opts.parts,
                    cache,
                )?))
            }
        };
        Ok(Self {
            model: opts.model,
            inner,
            notes: cache.map(WeightCache::take_notes).unwrap_or_default(),
        })
    }

    /// What loading did with [`LoadOptions::weight_cache`]: entries saved,
    /// or failures to save or read them (which don't fail the load).
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    pub fn model(&self) -> Model {
        self.model
    }

    pub fn generate(&self, opts: &GenerateOptions) -> Result<Generated> {
        self.generate_with(opts, |_| {})
    }

    /// Like [`generate`](Self::generate), calling `on_progress` as it goes.
    pub fn generate_with(
        &self,
        opts: &GenerateOptions,
        mut on_progress: impl FnMut(Progress),
    ) -> Result<Generated> {
        opts.validate(self.model)?;
        let result = match &self.inner {
            Inner::ZImage(p) => p.generate_with(opts, &mut on_progress),
            Inner::Qwen(p) => p.generate_with(opts, &mut on_progress),
        };
        // MLX keeps freed buffers for reuse. Between images (often of different
        // sizes) they only pile up: in a mixed-size batch the cache grew to
        // ~75 GB and steps slowed ~1.5x under memory pressure. Clear it after
        // failures too, so they don't slow down the next image.
        mlx_rs::memory::clear_cache()?;
        result
    }

    /// Runs the text side for one image: its prompt and negative prompt
    /// through the text encoder, and Qwen's reference images through the
    /// vision encoder. Needs a pipeline loaded with [`Parts::All`] or
    /// [`Parts::Encoders`]; see [`Parts`] for why.
    pub fn encode(&self, opts: &GenerateOptions) -> Result<Encoded> {
        opts.validate(self.model)?;
        let inner = match &self.inner {
            Inner::ZImage(p) => EncodedInner::ZImage(p.encode(opts)?),
            Inner::Qwen(p) => EncodedInner::Qwen(p.encode(opts)?),
        };
        Ok(Encoded {
            model: self.model,
            inner,
        })
    }

    /// Generates an image from [`encode`](Self::encode)'s output for the
    /// same options. Needs a pipeline loaded with [`Parts::All`] or
    /// [`Parts::Generator`].
    pub fn generate_encoded(&self, opts: &GenerateOptions, encoded: &Encoded) -> Result<Generated> {
        self.generate_encoded_with(opts, encoded, |_| {})
    }

    /// Like [`generate_encoded`](Self::generate_encoded), calling
    /// `on_progress` as it goes.
    pub fn generate_encoded_with(
        &self,
        opts: &GenerateOptions,
        encoded: &Encoded,
        mut on_progress: impl FnMut(Progress),
    ) -> Result<Generated> {
        opts.validate(self.model)?;
        anyhow::ensure!(
            encoded.model == self.model,
            "encoded for {}, not {}",
            encoded.model,
            self.model
        );
        let result = match (&self.inner, &encoded.inner) {
            (Inner::ZImage(p), EncodedInner::ZImage(e)) => {
                p.generate_encoded(opts, e, &mut on_progress)
            }
            (Inner::Qwen(p), EncodedInner::Qwen(e)) => {
                p.generate_encoded(opts, e, &mut on_progress)
            }
            _ => unreachable!("same model"),
        };
        mlx_rs::memory::clear_cache()?;
        result
    }
}

/// One image's prompt embeddings (and Qwen's reference images), from
/// [`Pipeline::encode`]: small next to the models (a few MB for an edit).
pub struct Encoded {
    model: Model,
    inner: EncodedInner,
}

enum EncodedInner {
    ZImage(crate::zimage::pipeline::Encoded),
    Qwen(crate::qwen21::pipeline::Encoded),
}

impl Encoded {
    /// Prompt tokens (for Qwen, image slots included).
    pub fn tokens(&self) -> usize {
        match &self.inner {
            EncodedInner::ZImage(e) => e.tokens,
            EncodedInner::Qwen(e) => e.tokens,
        }
    }

    /// Memory it takes.
    pub fn nbytes(&self) -> usize {
        match &self.inner {
            EncodedInner::ZImage(e) => e.nbytes(),
            EncodedInner::Qwen(e) => e.nbytes(),
        }
    }
}

/// Standard normal noise from a seeded CPU RNG. Identical to candy's, so the
/// same seed starts both tools from the same latents.
pub fn seeded_noise(seed: u64, shape: [usize; 4]) -> Vec<f32> {
    use rand::SeedableRng;
    use rand_distr::Distribution;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    rand_distr::StandardNormal
        .sample_iter(&mut rng)
        .take(shape.iter().product())
        .collect()
}

/// Resizes to exactly `w` x `h`, center-cropping to keep the aspect ratio.
pub(crate) fn resize_to_fill(img: &image::RgbImage, w: u32, h: u32) -> image::RgbImage {
    if img.dimensions() == (w, h) {
        return img.clone();
    }
    image::DynamicImage::ImageRgb8(img.clone())
        .resize_to_fill(w, h, image::imageops::FilterType::Lanczos3)
        .to_rgb8()
}

/// Composites an RGBA image over white, as the vision encoder was trained on.
pub fn composite_over_white(img: &image::RgbaImage) -> image::RgbImage {
    image::RgbImage::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y).0;
        let a = p[3] as f32 / 255.0;
        let mix = |c: u8| (c as f32 * a + 255.0 * (1.0 - a)).round() as u8;
        image::Rgb([mix(p[0]), mix(p[1]), mix(p[2])])
    })
}

/// (1, H, W, 3) u8 array -> RgbImage.
pub(crate) fn to_rgb_image(img: &Array) -> Result<image::RgbImage> {
    let sh = img.shape();
    anyhow::ensure!(
        sh.len() == 4 && sh[0] == 1 && sh[3] == 3,
        "expected a (1, H, W, 3) image, got {sh:?}"
    );
    let (height, width) = (sh[1] as u32, sh[2] as u32);
    let img = img.contiguous()?;
    img.eval()?;
    image::ImageBuffer::from_raw(width, height, img.as_slice::<u8>().to_vec())
        .ok_or_else(|| anyhow::anyhow!("image buffer size mismatch"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(width: usize, height: usize) -> GenerateOptions {
        GenerateOptions {
            width,
            height,
            ..GenerateOptions::new("a cat")
        }
    }

    #[test]
    fn validate_accepts_multiples_of_16() {
        for (w, h) in [(1024, 1024), (512, 512), (384, 512), (640, 368), (16, 16)] {
            opts(w, h).validate(Model::ZImageTurbo).unwrap();
        }
    }

    #[test]
    fn validate_rejects_bad_dimensions_with_suggestion() {
        let err = opts(1000, 1024)
            .validate(Model::ZImageTurbo)
            .unwrap_err()
            .to_string();
        assert!(err.contains("divisible by 16"), "{err}");
        // Only the side that needs it is rounded.
        assert!(err.contains("Try 992x1024 or 1008x1024 instead"), "{err}");

        // Never suggests 0.
        let err = opts(8, 1024)
            .validate(Model::ZImageTurbo)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Try 16x1024 instead"), "{err}");
    }

    #[test]
    fn validate_limits_z_image_size_to_its_rope_tables() {
        opts(8192, 16).validate(Model::ZImageTurbo).unwrap();
        let err = opts(8208, 16)
            .validate(Model::ZImageTurbo)
            .unwrap_err()
            .to_string();
        assert!(err.contains("at most 8192x8192"), "{err}");
        opts(8224, 32).validate(Model::QwenImage21).unwrap();
    }

    #[test]
    fn steps_to_run_never_underflows() {
        let mut o = opts(512, 512);
        o.init_image = Some(image::RgbImage::new(1, 1));
        o.strength = -0.5;
        assert_eq!(o.steps_to_run(), 0);
        o.strength = 0.6;
        assert_eq!(o.steps_to_run(), 6);
    }

    #[test]
    fn composite_over_white_blends_by_alpha() {
        let mut img = image::RgbaImage::from_pixel(3, 1, image::Rgba([0, 0, 0, 0]));
        img.put_pixel(1, 0, image::Rgba([10, 20, 30, 255]));
        img.put_pixel(2, 0, image::Rgba([0, 0, 0, 128]));
        let rgb = composite_over_white(&img);
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(rgb.get_pixel(1, 0).0, [10, 20, 30]);
        assert_eq!(rgb.get_pixel(2, 0).0, [127, 127, 127]);
    }

    #[test]
    fn validate_rejects_empty_prompt_and_zero_steps() {
        let mut o = opts(512, 512);
        o.prompt = "   ".into();
        assert!(o
            .validate(Model::ZImageTurbo)
            .unwrap_err()
            .to_string()
            .contains("prompt is empty"));

        let mut o = opts(512, 512);
        o.num_steps = 0;
        assert!(o
            .validate(Model::ZImageTurbo)
            .unwrap_err()
            .to_string()
            .contains("num_steps"));

        assert!(opts(0, 512)
            .validate(Model::ZImageTurbo)
            .unwrap_err()
            .to_string()
            .contains("must be positive"));
    }

    #[test]
    fn new_uses_turbo_defaults() {
        let o = GenerateOptions::new("x");
        assert_eq!(
            (o.width, o.height, o.num_steps, o.seed),
            (1024, 1024, DEFAULT_STEPS, 0)
        );
        assert!(o.negative_prompt.is_empty());
        assert_eq!(o.guidance_scale, 0.0); // Turbo runs without guidance
    }

    #[test]
    fn z_image_guidance_follows_diffusers_convention() {
        let mut o = opts(512, 512);
        o.guidance_scale = 2.0; // against the empty prompt
        o.validate(Model::ZImageTurbo).unwrap();
        o.negative_prompt = "blurry".into();
        o.validate(Model::ZImageTurbo).unwrap();

        // A negative prompt without guidance would be ignored: an error instead.
        o.guidance_scale = 0.0;
        let err = o.validate(Model::ZImageTurbo).unwrap_err().to_string();
        assert!(err.contains("needs guidance_scale above 0"), "{err}");

        o.guidance_scale = -1.0;
        let err = o.validate(Model::ZImageTurbo).unwrap_err().to_string();
        assert!(err.contains("at least 0"), "{err}");
    }

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

    #[test]
    fn to_rgb_image_checks_shape_and_copies_pixels() {
        let px: Vec<u8> = (0..2 * 3 * 3).map(|i| i as u8).collect();
        let a = Array::from_slice(&px, &[1, 2, 3, 3]);
        let img = to_rgb_image(&a).unwrap();
        assert_eq!(img.dimensions(), (3, 2));
        assert_eq!(img.get_pixel(1, 0).0, [3, 4, 5]);
        assert!(to_rgb_image(&Array::from_slice(&px, &[2, 3, 3])).is_err());
    }

    #[test]
    fn models_have_their_own_defaults_and_names() {
        let q = GenerateOptions::for_model(Model::QwenImage21, "x");
        assert_eq!((q.num_steps, q.guidance_scale), (40, 1.0));
        assert_eq!(GenerateOptions::new("x").num_steps, 9);
        for m in Model::ALL {
            assert_eq!(m.name().parse::<Model>().unwrap(), m);
        }
        assert!("sdxl".parse::<Model>().is_err());
        assert_eq!(Model::default(), Model::ZImageTurbo);
    }

    #[test]
    fn fast_qwen_runs_its_four_steps_without_guidance() {
        let fast = Model::QwenImage21Fast;
        let o = GenerateOptions::for_model(fast, "x");
        assert_eq!((o.num_steps, o.guidance_scale), (4, 1.0));
        o.validate(fast).unwrap();
        assert_eq!(fast.repo(), Model::QwenImage21.repo());
        assert!(fast.supports_reference_images());

        let mut o = GenerateOptions::for_model(fast, "x");
        o.num_steps = 8;
        let err = o.validate(fast).unwrap_err().to_string();
        assert!(err.contains("always runs 4 steps"), "{err}");

        let mut o = GenerateOptions::for_model(fast, "x");
        o.negative_prompt = "blurry".into();
        o.validate(fast).unwrap(); // guidance 1: the negative prompt is unused
        o.guidance_scale = 4.0;
        let err = o.validate(fast).unwrap_err().to_string();
        assert!(err.contains("without guidance"), "{err}");

        // img2img skips steps of the same 4-step schedule.
        let mut o = GenerateOptions::for_model(fast, "x");
        o.init_image = Some(image::RgbImage::new(64, 64));
        o.strength = 0.5;
        o.validate(fast).unwrap();
        assert_eq!(o.steps_to_run(), 2);
    }

    #[test]
    fn validate_checks_model_specific_rules() {
        // Qwen-Image-2.1 needs multiples of 32; Z-Image-Turbo only 16.
        let o = opts(1008, 1024);
        o.validate(Model::ZImageTurbo).unwrap();
        assert!(o
            .validate(Model::QwenImage21)
            .unwrap_err()
            .to_string()
            .contains("divisible by 32"));
        // Reference images only for models that support them.
        let mut o = opts(1024, 1024);
        o.reference_images.push(image::RgbaImage::new(64, 64));
        o.validate(Model::QwenImage21).unwrap();
        let err = o.validate(Model::ZImageTurbo).unwrap_err().to_string();
        assert!(err.contains("does not take reference images"), "{err}");
    }
}

# mixel

Text-to-image and image editing on Apple Silicon with
[mlx-rs](https://github.com/oxiglade/mlx-rs) (Rust bindings to MLX), for two models, one of
them also in a 4-step variant:

| `--model` | Model | Default steps | 1024×1024 image | Can do |
|---|---|---:|---:|---|
| `z-image-turbo` (default) | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 9 | ~59 s | text-to-image, img2img |
| `qwen-image-2.1` | [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 40 | ~6 min | text-to-image, img2img, **editing with reference images** |
| `qwen-image-2.1-fast` | Qwen-Image-2.1 + [4-step Fun-Acc LoRA](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs) | 4 (fixed) | ~38 s | the same, slightly softer fine detail |

Times are medians from the [reference benchmark](benchmarks/) on an M3 Max (30-core GPU,
96 GB), with the weights already in memory; an edit with a ~1024×1024 reference image takes
~8 min (40 steps) or ~58 s (4 steps). Memory figures count GB as Apple does (2^30 bytes).
Both are ports: Z-Image of
candle-transformers' `z_image` (`src/zimage/`), Qwen-Image-2.1 of the diffusers pipeline and
transformers' Qwen3-VL (`src/qwen21/`). The CLI, JSONL batch mode and seeding match
[`candy`](https://github.com/wongphu/candle-diffusion), the candle version, but Z-Image
sampling follows diffusers, so the same seed gives a slightly different image
([details](#z-image-sampling-follows-diffusers)).

```bash
cargo install --path .
mixel --prompt "A cute robot holding a candle" --seed 42
mixel --model qwen-image-2.1 --prompt "A capybara wearing a wizard hat, oil painting" --seed 1
mixel --model qwen-fast --ref-image fox.png --prompt "Turn the fox into a gray wolf" --seed 1
```

- Or download a ready-made `mixel` for macOS 14 or later from
  [Releases](https://github.com/wongphu/mixel/releases), and keep `mlx.metallib` next to it.
- The first run downloads the weights to `~/.cache/huggingface`: ~33 GB for Z-Image-Turbo
  (shared with `candy`), ~31 GB for Qwen-Image-2.1, plus 0.35 GB for the 4-step adapter.
- Without `--seed`, a random seed is used and added to the filename (`z_image_output-1234567.png`).
- Width/height must be multiples of 16 (Z-Image) or 32 (Qwen-Image). `--model-path <dir>`
  uses local weights (the 4-step adapter still comes from the Hugging Face cache).
- Each run prints a timing breakdown: text encoding, init image, denoising, VAE.
- A 1024×1024 image needs ~14 GB of memory with Z-Image-Turbo and ~17 GB with Qwen-Image-2.1;
  [`--quantize 4`](#less-memory-8--and-4-bit-weights) brings that down to 5.4 and 7.5 GB.
- Please [benchmark your Mac](#benchmark-your-mac) and send us the report.

## For AI agents and scripts

`mixel --help` ends with a usage guide written for agents: which model to pick, copy-paste
recipes, the JSONL format, where output files go (and how to get predictable names),
timings, memory, and exit codes. Rules of thumb:

- Always pass `--prompt`, `--seed` and `--output` (or `id`s in JSONL) so the output path is known up front.
- Generate several images with one `--input` JSONL run; the model loads once.
- Allow minutes per image (more on the first run, which downloads the weights) and run one
  `mixel` at a time: it needs 13–18 GB of memory, or 5–9 GB with `--quantize 4`.

## Editing with reference images (Qwen-Image-2.1)

```bash
mixel --model qwen-image-2.1 --ref-image fox.png --prompt "Turn the fox into a gray wolf"
mixel --model qwen-image-2.1 --ref-image a.png --ref-image b.png --prompt "Put the cat from image 1 on the sofa from image 2"
```

The reference images are read by the Qwen3-VL text encoder *and* passed to the transformer
as latents, so the result keeps their content where the prompt doesn't change it.
Each image is resized to about 1024×1024 px at its own aspect ratio; without
`--width`/`--height` the output takes the last image's aspect ratio at the same size.
In JSONL use `"reference_images": ["a.png", "b.png"]` (relative to the JSONL file).
Editing is slower than text-to-image, since each step also attends to every
reference-image token.

## Qwen-Image-2.1 in 4 steps

`--model qwen-image-2.1-fast` (or `qwen-fast`) adds Alibaba PAI's
[Fun-Acc LoRA](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs), distilled
with Parallel Decoding Distillation, to the same base weights: text-to-image, img2img and
editing run in 4 steps instead of 40: a 1024×1024 image takes ~38 s instead of ~6 min, an
edit ~58 s instead of ~8 min. A step costs about the same as the base model's, so the gain
is the step count.

- It always runs its fixed 4-step schedule (`1.0, 0.917, 0.786, 0.549, 0`), without guidance:
  other `--num-steps`, or a negative prompt with `--guidance-scale` above 1, are rejected.
  With `--init-image`, `--strength` skips steps of that schedule (0.5 runs the last 2).
- Its authors note that small dense text can lose legibility and some edits come out
  slightly blurrier and darker than with 40 steps. Use `qwen-image-2.1` when that matters.
- Like the reference, the rank-64 updates run as separate low-rank matmuls. They are only
  ~0.2% of the weights, so merging them into bf16 weights would round away 40–60% of each.

## Image to image

Start from an existing image instead of pure noise:

```bash
mixel --init-image photo.jpg --prompt "A gray wolf sitting in fresh snow" --strength 0.75 --seed 7
```

- Works with both models. `--strength` in (0, 1] is the fraction of denoising steps that
  run (like diffusers): with 9 steps, 0.6 runs 6 and 0.75 runs 7. Low values stay close to the image, 1.0
  ignores its content and equals plain text-to-image with the same seed. Default 0.6.
- Without `--width`/`--height`, the output keeps the image's aspect ratio (multiples of 16
  or 32, longest side at most 1024). With explicit sizes, the image is center-cropped to fill.
- JSONL lines take `init_image` (relative to the JSONL file) and `strength`.
- **Content vs style.** Changing *what* is in the picture (fox to wolf) works at 0.6 to 0.8
  and keeps pose and framing. Changing a photo's *style* (to watercolor) needs a start
  closer to pure noise; with 9 steps the finest option skips a whole step, so use more:
  `--num-steps 20 --strength 0.95` gives a watercolor with the original composition.

## Less memory: 8- and 4-bit weights

```bash
mixel --quantize 8 --prompt "A cute robot holding a candle" --seed 42
mixel --quantize 4 --model qwen-fast --ref-image fox.png --prompt "Turn the fox into a gray wolf"
```

`--quantize 8` or `--quantize 4` stores the large layers of the text encoder and the
transformer (their attention and MLP projections) in 8 or 4 bits as they load, with MLX's
affine quantization (a scale and bias per 64 weights). The embeddings, norms, modulation
layers and the VAE stay in bf16, Qwen's vision encoder in f32. The files on disk don't
change. Peak memory, measured as the process footprint on the M3 Max:

| | bf16 | `--quantize 8` | `--quantize 4` |
|---|---:|---:|---:|
| Z-Image-Turbo, 512×512 | 12.6 GB | 7.2 GB | **4.6 GB** |
| Z-Image-Turbo, 1024×1024 | 13.5 GB | 8.3 GB | **5.4 GB** |
| Qwen-Image-2.1 (40 or 4 steps), 1024×1024 | 17.0 GB | 10.9 GB | **7.5 GB** |
| Qwen-Image-2.1, edit with a ~1024×1024 image | 18.3 GB | 12.2 GB | **9.0 GB** |

**What runs well where.** The GPU may only use part of a Mac's memory (its working set:
81% on this 96 GB Mac, about 2/3 on a 16 GB one), and a model that needs more runs 2–3×
slower. Measured on a [16 GB Mac mini](benchmarks/) (M4, 10-core GPU), 1024×1024:

| | Peak | Per image | Per step |
|---|---:|---:|---:|
| Z-Image-Turbo, `--quantize 8` / `4` | 8.2 / 5.4 GB | 166 s | 18.0 s |
| Qwen-Image-2.1 in 4 steps, `--quantize 4` | 8.0 GB | 96 s | 22.6 s |
| Qwen-Image-2.1 in 4 steps, edit, `--quantize 4` | 8.3 GB | 132 s | 25.7 s |
| Z-Image-Turbo, bf16 (over the working set) | 13.4 GB | 182–374 s | ~40 s |

So on 16 GB, use `--quantize 8` for Z-Image-Turbo (as fast as 4 bits there, and practically
the bf16 images) and `--quantize 4` for Qwen-Image-2.1. An 8 GB Mac (~5.3 GB working set)
is borderline even at 4 bits; 32 GB and up runs everything in bf16.

- **8 bits gives practically the same images.** Each stage stays within 1–2% of bf16, about
  the noise between two bf16 implementations (Z-Image: text encoder 0.6–1.1%, one
  transformer step 1.7–1.9%).
- **4 bits gives images as good, but not the same ones**: a seed no longer reproduces the
  bf16 image. A Z-Image transformer step is ~16% off bf16 (the text encoder ~5%), enough to
  change pose, layout and fine detail. No single kind of layer is to blame, and groups of
  32 weights instead of 64 only bring it to 15%.
- **Both are slower per step**, as MLX's quantized matmuls unpack the weights as they go: in
  the [reference benchmark](benchmarks/), a 1024×1024 Z-Image-Turbo step takes 7.7 s at 8
  bits and 7.9 s at 4 against 6.3 s in bf16, and a 4-step Qwen-Image-2.1 step 10.3 s at 4
  bits against 8.7 s (an edit 11.7 against 10.5). Loading takes ~2 s longer.
- `cargo run --release --example quantize_parity -- <model>` compares the stages and whole
  images against bf16; `cargo run --release --example memory -- <model> <size> <bf16|8|4>`
  shows the memory in each phase.

With or without `--quantize`, mixel keeps memory down three more ways. A 1024×1024
Z-Image-Turbo image peaked at 36 GB in mixel 0.3, and a Qwen-Image-2.1 edit at 65 GB.

- **The text encoders are never in memory with the transformer.** They run once per image,
  at the start, yet are a third (Z-Image-Turbo) to half (Qwen-Image-2.1, with its vision
  encoder) of the weights. mixel loads them, encodes every prompt of the run (a JSONL batch
  in chunks of up to 512 MB of encodings, ~0.3 MB per prompt or ~10 MB per edit), frees
  them, then loads the transformer and VAE. The images are the same bit for bit, at the
  same speed.
- The VAE decodes in bands of rows: MLX's convolutions allocate a workspace several times
  their input (~4.5 GB for one 3×3 convolution at 1024×1024). Decoding takes ~1 s longer.
- MLX's buffer cache is off: it kept up to 10 GB of freed buffers.

## As a library

```toml
[dependencies]
mixel = { git = "https://github.com/wongphu/mixel" }
```

```rust
use mixel::{GenerateOptions, LoadOptions, Model, Parts, Pipeline, Progress, Quantize};

let pipeline = Pipeline::load(&LoadOptions::default())?; // Z-Image-Turbo; load once, reuse
let opts = GenerateOptions { seed: 42, width: 768, ..GenerateOptions::new("a red fox in fresh snow") };
let out = pipeline.generate_with(&opts, |p| {
    if let Progress::Step { step, total, .. } = p { eprintln!("step {step}/{total}") }
})?;
out.image.save("fox.png")?;          // image::DynamicImage (RGBA if Qwen output has transparency)

println!("{:?}", out.timings);       // text / init image / denoise / vae durations

// img2img: set init_image and strength
let photo = image::open("photo.jpg")?.to_rgb8();
let variation = GenerateOptions { init_image: Some(photo.clone()), strength: 0.75, ..opts };

// Qwen-Image-2.1 with a reference image (Model::QwenImage21Fast for 4 steps)
let qwen = Pipeline::load(&LoadOptions { model: Model::QwenImage21, ..Default::default() })?;
let edit = GenerateOptions {
    reference_images: vec![image::open("photo.png")?.to_rgba8()], // alpha is kept
    ..GenerateOptions::for_model(Model::QwenImage21, "make it night")
};
qwen.generate(&edit)?.image.save("night.png")?;

// Least memory, like the mixel command: 4 bits, and the encoders and the
// transformer never loaded together (encode many options before switching)
let load = |parts| Pipeline::load(&LoadOptions { quantize: Some(Quantize::Q4), parts, ..Default::default() });
let encoded = load(Parts::Encoders)?.encode(&opts)?; // the encoders are freed here
load(Parts::Generator)?.generate_encoded(&opts, &encoded)?.image.save("fox-4bit.png")?;
```

The library prints nothing and never writes files; the `mixel` command adds the CLI,
JSONL batching, seed-in-filename naming and saving. The models themselves are in
`mixel::zimage` and `mixel::qwen21` for lower-level use.

## Batch mode (JSONL)

```bash
mixel --input prompts.jsonl --output-dir out
```

```json
{"id": "fox", "prompt": "A red fox in fresh snow", "seed": 3}
{"prompt": "A bowl of ramen, top-down", "output": "food/ramen.png"}
{"prompt": "A watercolor sailboat at dawn", "width": 768, "height": 512, "num_steps": 6}
```

Fields: `prompt` (required), `id`, `negative_prompt`, `width`, `height`, `num_steps`,
`guidance_scale`, `seed`, `output`, `init_image`, `strength`, `reference_images`. Without
`output`, the file is named after `id` (`fox.png`), else the line number (`0003.png`). Omitted
fields fall back to the CLI flags (the model is chosen with `--model` for the whole batch). Lines are
validated before the model loads, existing outputs are skipped (`--overwrite` to redo),
and failed lines are reported at the end. Same behavior as `candy`.

## Benchmark your Mac

We're collecting mixel timings across Macs. If you can spare the time, run:

```bash
scripts/benchmark.sh --dry-run   # what would run, and download, on this Mac
scripts/benchmark.sh             # run it: ~65 min on an M3 Max
```

It runs each test that fits in your GPU's working set: Z-Image-Turbo at 512×512 and
1024×1024 (also in 8 and 4 bits), and Qwen-Image-2.1 in 4 and 40 steps, text-to-image and an
edit (the 4-step ones also in 4 bits). A 16 GB Mac runs the quantized tests, a 32 GB one
all of them. It asks before downloading weights. The result is one file, `mixel-benchmark-<chip>-<gpu>-<memory>-<date>.md`, labelled with
your Mac, chip, CPU and GPU cores, memory, macOS version and power source, with a table to
read and a JSON block for us to compile.

**Please email it to [mixelate@proton.me](mailto:mixelate@proton.me).** It holds only the
hardware summary and the timings: no serial numbers, hostnames, user names or images. For
steadier numbers, plug in, close other apps and leave the Mac alone while it runs, with the
lid open: the script keeps it from sleeping, but closing the lid still does.
[`benchmarks/`](benchmarks/) has the reports so far: the reference from an M3 Max (96 GB), and
an M4 Mac mini (16 GB).

## mixel vs candy

Apple M3 Max (30-core GPU, 96 GB), same prompt and seed, weights already cached. candy's column is from the
original comparison; mixel's is from the [reference benchmark](benchmarks/):

| | candy (candle 0.11) | mixel (mlx-rs 0.32) |
|---|---:|---:|
| 512×512 image | 22.2 s | **13.6 s** |
| 1024×1024 image | ~127 s | **59 s** |
| 1024×1024: denoising | ~10.3 s/step | **6.3 s/step** |
| 1024×1024: VAE decode | ~35 s | **2.4 s** |
| Peak memory, 1024×1024 | 81 GB | **13.5 GB** (5.4 GB with `--quantize 4`) |

candy's phase split is measured from its log timestamps (candle queues GPU work
asynchronously, so treat it as approximate). The VAE gap matches MLX's much faster 3×3
convolutions (see [mlx-vs-candle](https://github.com/wongphu/mlx-vs-candle)).

**Output parity.** `cargo run --release --example parity -- 128` runs each stage of both
implementations on identical inputs. Relative L2 error at 1024×1024: text encoder 0.9%,
transformer (one step, 32-token caption) 1.4%, VAE decode 2.5%, VAE encode 1.5% (at
512×512), which is bf16 rounding noise.

### Z-Image sampling follows diffusers

mixel first sampled like candle's port and `candy`; it now follows the reference
implementation, diffusers' `ZImagePipeline`, in three ways:

- **The schedule.** The sigmas are `linspace(1, 1/n, n)` shifted by `3s / (1 + 2s)`
  (1.0, 0.96, 0.91 … 0.27 for 9 steps), so more steps are spent at high noise; candle
  doesn't shift them (1.0, 0.89, 0.78 … 0.11).
- **Pad tokens.** The caption and the image are padded to a multiple of 32 tokens with the
  model's learned `cap_pad_token` / `x_pad_token`, which the image attends to, and the
  image's RoPE position comes after the padded caption.
- **Guidance.** `--guidance-scale s` gives `pos + s·(pos − neg)`, on for any `s > 0`, against
  `--negative-prompt` or the empty prompt (mixel used `neg + s·(pos − neg)`, only with a
  negative prompt). The default is 0, as Turbo is meant to run; a negative prompt without
  guidance is rejected. An old scale `s` with a negative prompt is `s − 1` now.

The time input and the latents also stay in f32 between steps, as in diffusers.
`scripts/make_zimage_reference.py` records diffusers' sampling at 512×512, and
`cargo run --release --example zimage_diffusers_parity -- <dir>` checks mixel against it:
schedule and time inputs identical, each step within 1–3%, and the final image 6.5% off
(bf16 noise compounding over 9 steps; the images look the same), or 7.4% with guidance 3.
Without the pad tokens each step was 3–21% off and the final image 34%; with the old
guidance formula the guided image was 27% off. The candle check above uses a 32-token
caption, where neither implementation pads.

## Qwen-Image-2.1: accuracy and speed

**Checked against the reference.** `scripts/make_qwen21_reference.py` runs the diffusers
pipeline and records every stage; `cargo run --release --example qwen21_parity -- <dir>
<image>` feeds each stage of this port the same inputs. Relative L2 error (bf16):
scheduler exact, token ids identical, text encoder 2.5%, transformer step 1.0% (1.2% with a
reference image), VAE decode 0.9%, encode 3.1%, and a full 8-step text-to-image 3.3%.
With the text and vision encoders in f32 the prompt embeddings agree with an f32 reference
to 1e-5, so the rest is rounding.

mixel runs the vision encoder in f32 (it is small): in bf16, the reference's own vision
output is ~6% off f32 and the text encoder amplifies that to ~28% in edit prompts; mixel's
edit prompts are 6% off f32.

**A PyTorch bug on Macs.** On PyTorch's MPS backend, the VAE encoder's `AvgDown3D`
returns zeros at 256 px and up (an 8-D reshape/permute), so diffusers' Qwen-Image-2.1
image encoding, and with it editing, is wrong on Apple GPUs. mixel isn't affected; the
reference script works around it by running that module on the CPU.

**Speed.** 1024×1024, 40 steps: 9.1 s/step (366 s per image, 17 GB peak) against 11.9
s/step for diffusers on the same M3 Max (measured in an earlier comparison). An edit with one
~1024×1024 reference image: 11.7 s/step (484 s, 18 GB peak). These are medians of 3 runs back
to back; the GPU slows as it heats under sustained load, so later runs take longer (450–489 s
for the edit, the benchmark's last test, after an hour of load). The
text and reference-image tokens are computed once per image and cached (as in the
reference), so each step only runs the target tokens.

**The 4-step variant.** `scripts/make_qwen21_fast_reference.py` runs the adapter through its
authors' own code (`qwenimage21_pdd.py`) at 512×512 and records every step;
`cargo run --release --example qwen21_fast_parity -- <dir>` checks this port against it.
Each step, given the reference's input, is within 0.5–1% (bf16 noise), and a full edit ends
1.3% off. A full text-to-image run ends ~11% off with the same composition: its four large
steps amplify any input difference 10–16× per step even within one implementation (the
example measures this), so the ~1% gap between two bf16 implementations grows. On MPS,
diffusers rounds the latents back to bf16 after each step (a workaround for an old PyTorch
bug); the adapter expects them in f32, as on CUDA, so the script turns that off.

## Tests

```bash
cargo test --release                  # unit + CLI tests, no model needed
cargo test --release -- --ignored     # end-to-end generation with the real models

# Parity with candle-transformers (no Python needed)
cargo run --release --example parity -- 128

# Parity with the PyTorch reference (each script writes into the current directory)
python scripts/make_zimage_reference.py
cargo run --release --example zimage_diffusers_parity -- .
python scripts/make_qwen21_reference.py <qwen-snapshot-dir> fox.png
cargo run --release --example qwen21_parity -- . fox.png
python scripts/make_qwen21_fast_reference.py <qwen-snapshot-dir> fox.png
cargo run --release --example qwen21_fast_parity -- .
```

The PyTorch scripts need torch, torchvision, transformers >= 5 and diffusers >= 0.41; until
0.41 is on PyPI, `pip install git+https://github.com/huggingface/diffusers`. The Qwen ones
need the whole Qwen-Image-2.1 snapshot (`hf download Qwen/Qwen-Image-2.1`; mixel itself only
fetches the files it uses, without `model_index.json` and the configs), and `fox.png` must be
512×512, the size the scripts run the edit at. Last run 2026-10-02 (torch 2.14.1, diffusers
main at 578c9b2): every stage within the numbers above.

## Building

mlx-rs compiles MLX from source, which needs:

- `cmake` (e.g. `brew install cmake`)
- Xcode's Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`

The candle crates are dev-dependencies, used only by `examples/parity.rs`.

A local build targets your own macOS version: MLX builds for the build machine's macOS
unless `MACOSX_DEPLOYMENT_TARGET` says otherwise, and the binary and its GPU kernels
(`mlx.metallib`) then need that version or later. `scripts/release.sh <version>` builds the
released binaries for macOS 14, MLX's minimum, tests and packages them, and commits and
tags the release. On macOS 14, MLX leaves out its NAX kernels for M5 GPUs (they need 26.2),
so on an M5 a local build may be faster.

## License

MIT, see [LICENSE](LICENSE). The Z-Image code is ported from candle-transformers
(MIT/Apache-2.0). The Qwen-Image-2.1 code is ported from diffusers and transformers
(Apache-2.0, see [LICENSE-APACHE](LICENSE-APACHE)).

The model weights have their own licenses, which you accept when downloading them.
In particular, **Qwen-Image-2.1's weights, and the 4-step adapter's, are under the Qwen
Research License: research and evaluation only, not commercial use** without a separate
license from Qwen.

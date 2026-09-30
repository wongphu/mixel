# mixel

Text-to-image and image editing on Apple Silicon with
[mlx-rs](https://github.com/oxiglade/mlx-rs) (Rust bindings to MLX), for two models:

| `--model` | Model | Default steps | 1024×1024 image | Can do |
|---|---|---:|---:|---|
| `z-image-turbo` (default) | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 9 | ~68 s | text-to-image, img2img |
| `qwen-image-2.1` | [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 40 | ~7 min | text-to-image, img2img, **editing with reference images** |

Times are for an M3 Max. Both are ports: Z-Image of candle-transformers' `z_image`
(`src/zimage/`), Qwen-Image-2.1 of the diffusers pipeline and transformers' Qwen3-VL
(`src/qwen21/`). The CLI, JSONL batch mode and seeding match
[`candy`](https://github.com/wongphu/candle-diffusion), the candle version.

```bash
cargo install --path .
mixel --prompt "A cute robot holding a candle" --seed 42
mixel --model qwen-image-2.1 --prompt "A capybara wearing a wizard hat, oil painting" --seed 1
```

- The first run downloads the weights to `~/.cache/huggingface`: ~33 GB for Z-Image-Turbo
  (shared with `candy`), ~31 GB for Qwen-Image-2.1.
- Without `--seed`, a random seed is used and added to the filename (`z_image_output-1234567.png`).
- Width/height must be multiples of 16 (Z-Image) or 32 (Qwen-Image). `--model-path <dir>`
  uses local weights.
- Each run prints a timing breakdown: text encoding, init image, denoising, VAE.

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

## As a library

```toml
[dependencies]
mixel = { git = "https://github.com/wongphu/mixel" }
```

```rust
use mixel::{GenerateOptions, LoadOptions, Model, Pipeline, Progress};

let pipeline = Pipeline::load(&LoadOptions::default())?; // Z-Image-Turbo; load once, reuse
let opts = GenerateOptions { seed: 42, width: 768, ..GenerateOptions::new("a red fox in fresh snow") };
let out = pipeline.generate_with(&opts, |p| {
    if let Progress::Step { step, total, .. } = p { eprintln!("step {step}/{total}") }
})?;
out.image.save("fox.png")?;          // image::RgbImage

println!("{:?}", out.timings);       // text / init image / denoise / vae durations

// img2img: set init_image and strength
let photo = image::open("photo.jpg")?.to_rgb8();
let variation = GenerateOptions { init_image: Some(photo.clone()), strength: 0.75, ..opts };

// Qwen-Image-2.1 with a reference image
let qwen = Pipeline::load(&LoadOptions { model: Model::QwenImage21, ..Default::default() })?;
let edit = GenerateOptions {
    reference_images: vec![photo],
    ..GenerateOptions::for_model(Model::QwenImage21, "make it night")
};
qwen.generate(&edit)?.image.save("night.png")?;
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

## mixel vs candy

Apple M3 Max, same prompt and seed, weights already cached:

| | candy (candle 0.11) | mixel (mlx-rs 0.32) |
|---|---:|---:|
| 512×512 image | 22.2 s | **14.2 s** |
| 1024×1024 image | ~127 s | **68 s** |
| 1024×1024: denoising | ~10.3 s/step | **7.4 s/step** |
| 1024×1024: VAE decode | ~35 s | **1.8 s** |
| Peak memory, 1024×1024 | 81 GB | **39 GB** |

candy's phase split is measured from its log timestamps (candle queues GPU work
asynchronously, so treat it as approximate). The VAE gap matches MLX's much faster 3×3
convolutions (see [mlx-vs-candle](https://github.com/wongphu/mlx-vs-candle)).

**Output parity.** `cargo run --release --example parity -- 128` runs each stage of both
implementations on identical inputs. Relative L2 error at 1024×1024: text encoder 0.9%,
transformer (one step) 1.7%, VAE decode 2.5%, VAE encode 1.5% (at 512×512), which is bf16
rounding noise. Final images use the
same seeded noise and match closely at 512×512 (PSNR 32.5 dB); at 1024×1024 the small
per-step differences compound, so the composition matches but fine details can differ.

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

**Speed.** 1024×1024, 40 steps: 10.4 s/step (418 s per image, 55 GB peak) against 11.9
s/step for diffusers on the same M3 Max. An edit with one ~1024×1024 reference image:
11.8 s/step (485 s, 69 GB peak). The text and reference-image tokens are computed
once per image and cached (as in the reference), so each step only runs the target tokens.

## Tests

```bash
cargo test --release                  # unit + CLI tests, no model needed
cargo test --release -- --ignored     # end-to-end generation with the real models
```

## Building

mlx-rs compiles MLX from source, which needs:

- `cmake` (e.g. `brew install cmake`)
- Xcode's Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`

The candle crates are dev-dependencies, used only by `examples/parity.rs`.

## License

MIT, see [LICENSE](LICENSE). The Z-Image code is ported from candle-transformers
(MIT/Apache-2.0). The Qwen-Image-2.1 code is ported from diffusers and transformers
(Apache-2.0, see [LICENSE-APACHE](LICENSE-APACHE)).

The model weights have their own licenses, which you accept when downloading them.
In particular, **Qwen-Image-2.1's weights are under the Qwen Research License: research
and evaluation only, not commercial use** without a separate license from Qwen.

# mixel

Text-to-image with [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) on
Apple Silicon using [mlx-rs](https://github.com/oxiglade/mlx-rs) (Rust bindings to MLX).

The model is a port of candle-transformers' `z_image` to mlx-rs (`src/zimage/`). The CLI,
JSONL batch mode and seeding match [`candy`](https://github.com/wongphu/candle-diffusion),
the candle version, so the same command and seed work with either.

```bash
cargo install --path .
mixel --prompt "A cute robot holding a candle" --width 1024 --height 1024 --seed 42
```

- The first run downloads ~33 GB of weights to `~/.cache/huggingface` (shared with `candy`).
- Without `--seed`, a random seed is used and added to the filename (`z_image_output-1234567.png`).
- Width/height must be divisible by 16. Default steps: 9. `--model-path <dir>` uses local weights.
- Each run prints a timing breakdown: text encoding, denoising, VAE.

## As a library

```toml
[dependencies]
mixel = { git = "https://github.com/wongphu/mixel" }
```

```rust
use mixel::{GenerateOptions, LoadOptions, Pipeline, Progress};

let pipeline = Pipeline::load(&LoadOptions::default())?; // load once, reuse
let opts = GenerateOptions { seed: 42, width: 768, ..GenerateOptions::new("a red fox in fresh snow") };
let out = pipeline.generate_with(&opts, |p| {
    if let Progress::Step { step, total, .. } = p { eprintln!("step {step}/{total}") }
})?;
out.image.save("fox.png")?;          // image::RgbImage
println!("{:?}", out.timings);       // text / denoise / vae durations
```

The library prints nothing and never writes files; the `mixel` command adds the CLI,
JSONL batching, seed-in-filename naming and saving. The model itself is in `mixel::zimage`
for lower-level use.

## Batch mode (JSONL)

```bash
mixel --input prompts.jsonl --output-dir out
```

```json
{"prompt": "A red fox in fresh snow", "seed": 3}
{"prompt": "A bowl of ramen, top-down", "output": "food/ramen.png"}
{"prompt": "A watercolor sailboat at dawn", "width": 768, "height": 512, "num_steps": 6}
```

Fields: `prompt` (required), `negative_prompt`, `width`, `height`, `num_steps`,
`guidance_scale`, `seed`, `output`; omitted fields fall back to the CLI flags. Lines are
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
transformer (one step) 1.7%, VAE 2.5%, which is bf16 rounding noise. Final images use the
same seeded noise and match closely at 512×512 (PSNR 32.5 dB); at 1024×1024 the small
per-step differences compound, so the composition matches but fine details can differ.

## Tests

```bash
cargo test --release                  # unit + CLI tests, no model needed
cargo test --release -- --ignored     # end-to-end generation with the real model
```

## Building

mlx-rs compiles MLX from source, which needs:

- `cmake` (e.g. `brew install cmake`)
- Xcode's Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`

The candle crates are dev-dependencies, used only by the parity example.

## License

MIT, see [LICENSE](LICENSE). The model code is ported from candle-transformers (MIT/Apache-2.0).

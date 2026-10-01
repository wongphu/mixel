# mixel benchmark: Apple M3 Max, 30-core GPU, 96 GB

> **Please email this file to mixelate@proton.me** so we can compile results across Macs.
> It holds the hardware summary and timings below, nothing else: no serial numbers,
> hostnames, user names or images.

## Hardware

| | |
|---|---|
| Mac | MacBook Pro (Mac15,10) |
| Chip | Apple M3 Max |
| CPU | 14 cores (10 performance, 4 efficiency) |
| GPU | 30 cores |
| Memory | 96 GB |
| macOS | 26.6.2 (25G83) |
| Power | AC power, energy mode Automatic |

## Software

mixel 0.2.0 (git 91d6ae4 (modified)), mlx-rs 0.32.0. Run 2026-10-01T03:35:24Z, took 49.9 min.

## Results

| Test | Time per image (median) | Range | Per step | Peak memory | Reference* |
|---|---:|---:|---:|---:|---:|
| Z-Image-Turbo, 512x512 | **13.2 s** | 13.2-13.2 s | 1.4 s | 26.7 GB | 13.2 s |
| Z-Image-Turbo, 1024x1024 | **59.0 s** | 58.5-59.8 s | 6.4 s | 36.4 GB | 59.0 s |
| Qwen-Image-2.1 fast, 1024x1024 | **36.8 s** | 33.8-39.8 s | 8.7 s | 52.1 GB | 36.8 s |
| Qwen-Image-2.1 fast, edit | **63.6 s** | 58.5-65.3 s | 11.6 s | 64.8 GB | 63.6 s |
| Qwen-Image-2.1, 1024x1024 | **373.0 s** | 349.0-374.6 s | 9.3 s | 51.5 GB | 373.0 s |
| Qwen-Image-2.1, edit | **444.5 s** | 442.9-445.8 s | 10.7 s | 64.2 GB | 444.5 s |

\* The same test on an Apple M3 Max, 30-core GPU, 96 GB, for comparison.

## How to read this

- **Time per image**: generating one image with the model already loaded (text encoding,
  denoising, VAE decoding): the median of 3 runs back to back, with no pause, as in a
  batch. **Range** is the fastest and slowest of them: a Mac slows down as it heats up,
  so a wide range mostly shows how much this one throttles under sustained load. Loading
  the model adds a few seconds per run of mixel; the first run of a model also downloads it.
- **Per step**: denoising time per step, the part that grows with the step count: Z-Image-Turbo
  runs 9 steps, Qwen-Image-2.1 fast 4 and Qwen-Image-2.1 40.
- **Peak memory**: the most memory mixel used (1 GB = 2^30 bytes, as Apple counts RAM).
  Tests that need more than 85% of this Mac's memory are skipped. "Swapped" means macOS
  moved memory to disk during the test, which makes it slower than the chip can do.
- **Settings**: prompt "A red fox sitting in fresh snow, wildlife photography" (edits: "Turn the fox into a gray wolf",
  on the fast test's image), seed 1. The tests run in the order above, so the later
  ones start on an already warm Mac.

## Raw data

```json
{
  "schema": 1,
  "hardware": {"mac": "MacBook Pro", "model_id": "Mac15,10", "chip": "Apple M3 Max", "cpu_performance_cores": 10, "cpu_efficiency_cores": 4, "gpu_cores": 30, "memory_gb": 96, "macos": "26.6.2", "macos_build": "25G83", "power": "AC power", "energy_mode": "Automatic"},
  "software": {"mixel": "mixel 0.2.0", "git": "91d6ae4 (modified)", "mlx_rs": "0.32.0"},
  "run": {"started": "2026-10-01T03:35:24Z", "minutes": 49.9, "prompt": "A red fox sitting in fresh snow, wildlife photography", "edit_prompt": "Turn the fox into a gray wolf", "seed": 1},
  "results": [
    {"test": "z512", "label": "Z-Image-Turbo, 512x512", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 13.2, "runs": [{"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 12.8, "per_step_s": 1.4, "vae_s": 0.3, "image_s": 13.2, "peak_gb": 26.7, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 12.8, "per_step_s": 1.4, "vae_s": 0.3, "image_s": 13.2, "peak_gb": 26.7, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 12.8, "per_step_s": 1.4, "vae_s": 0.3, "image_s": 13.2, "peak_gb": 26.7, "swap_gb": 0.0}]},
    {"test": "z1024", "label": "Z-Image-Turbo, 1024x1024", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 59.0, "runs": [{"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 57.5, "per_step_s": 6.4, "vae_s": 1.4, "image_s": 59.0, "peak_gb": 36.4, "swap_gb": 0.0}, {"load_s": 1.3, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 57.0, "per_step_s": 6.3, "vae_s": 1.4, "image_s": 58.5, "peak_gb": 36.4, "swap_gb": 0.0}, {"load_s": 1.3, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 58.3, "per_step_s": 6.5, "vae_s": 1.4, "image_s": 59.8, "peak_gb": 36.4, "swap_gb": 0.0}]},
    {"test": "fast", "label": "Qwen-Image-2.1 fast, 1024x1024", "model": "qwen-image-2.1-fast", "steps": 4, "status": "ok", "reference_image_s": 36.8, "runs": [{"load_s": 3.7, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 31.8, "per_step_s": 8.0, "vae_s": 1.6, "image_s": 33.8, "peak_gb": 52.1, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 34.7, "per_step_s": 8.7, "vae_s": 1.8, "image_s": 36.8, "peak_gb": 52.1, "swap_gb": 0.0}, {"load_s": 1.3, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 37.6, "per_step_s": 9.4, "vae_s": 1.9, "image_s": 39.8, "peak_gb": 52.1, "swap_gb": 0.0}]},
    {"test": "fastedit", "label": "Qwen-Image-2.1 fast, edit", "model": "qwen-image-2.1-fast", "steps": 4, "status": "ok", "reference_image_s": 63.6, "runs": [{"load_s": 1.3, "text_s": 3.6, "init_image_s": 10.9, "denoise_s": 48.6, "per_step_s": 12.2, "vae_s": 2.1, "image_s": 65.3, "peak_gb": 64.8, "swap_gb": 0.0}, {"load_s": 2.8, "text_s": 3.9, "init_image_s": 11.5, "denoise_s": 46.3, "per_step_s": 11.6, "vae_s": 1.9, "image_s": 63.6, "peak_gb": 64.8, "swap_gb": 0.0}, {"load_s": 2.8, "text_s": 3.5, "init_image_s": 10.2, "denoise_s": 42.8, "per_step_s": 10.7, "vae_s": 1.8, "image_s": 58.5, "peak_gb": 64.8, "swap_gb": 0.0}]},
    {"test": "qwen", "label": "Qwen-Image-2.1, 1024x1024", "model": "qwen-image-2.1", "steps": 40, "status": "ok", "reference_image_s": 373.0, "runs": [{"load_s": 1.1, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 346.7, "per_step_s": 8.7, "vae_s": 1.9, "image_s": 349.0, "peak_gb": 51.4, "swap_gb": 0.0}, {"load_s": 1.1, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 372.4, "per_step_s": 9.3, "vae_s": 1.8, "image_s": 374.6, "peak_gb": 51.4, "swap_gb": 0.0}, {"load_s": 1.1, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 370.7, "per_step_s": 9.3, "vae_s": 1.9, "image_s": 373.0, "peak_gb": 51.5, "swap_gb": 0.0}]},
    {"test": "qwenedit", "label": "Qwen-Image-2.1, edit", "model": "qwen-image-2.1", "steps": 40, "status": "ok", "reference_image_s": 444.5, "runs": [{"load_s": 1.2, "text_s": 3.4, "init_image_s": 9.9, "denoise_s": 427.6, "per_step_s": 10.7, "vae_s": 1.9, "image_s": 442.9, "peak_gb": 64.1, "swap_gb": 0.0}, {"load_s": 2.7, "text_s": 3.5, "init_image_s": 9.9, "denoise_s": 429.1, "per_step_s": 10.7, "vae_s": 1.9, "image_s": 444.5, "peak_gb": 64.2, "swap_gb": 0.0}, {"load_s": 2.7, "text_s": 3.5, "init_image_s": 9.9, "denoise_s": 430.3, "per_step_s": 10.8, "vae_s": 1.9, "image_s": 445.8, "peak_gb": 64.2, "swap_gb": 0.0}]}
  ]
}
```

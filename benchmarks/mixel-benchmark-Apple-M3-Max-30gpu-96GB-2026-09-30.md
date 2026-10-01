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

mixel 0.2.0 (git 4a613f1 (modified)), mlx-rs 0.32.0. Run 2026-10-01T00:09:48Z, took 24.3 min.

## Results

| Test | Time per image | Per step | Peak memory | Reference* |
|---|---:|---:|---:|---:|
| Z-Image-Turbo, 512x512 | **13.2 s** | 1.4 s | 26.7 GB | 13.2 s |
| Z-Image-Turbo, 1024x1024 | **60.6 s** | 6.6 s | 36.4 GB | 60.6 s |
| Qwen-Image-2.1 fast, 1024x1024 | **42.7 s** (runs varied: up to 58.6 s) | 10.0 s | 52.1 GB | 42.7 s |
| Qwen-Image-2.1 fast, edit | **58.5 s** (runs varied: up to 70.7 s) | 10.7 s | 64.8 GB | 58.5 s |
| Qwen-Image-2.1, 1024x1024 | **390.2 s** | 9.7 s | 51.4 GB | 390.2 s |
| Qwen-Image-2.1, edit | **458.3 s** | 11.0 s | 64.2 GB | 458.3 s |

\* The same test on an Apple M3 Max, 30-core GPU, 96 GB, for comparison.

## How to read this

- **Time per image**: generating one image with the model already loaded (text encoding,
  denoising, VAE decoding), best of the runs (3 for the short tests, 1 for the 40-step
  ones). Loading the model adds a few seconds per run of mixel; the first run of a model
  also downloads it. "Runs varied" marks tests whose slowest run took over 20% longer
  than the best, usually because another app was using the GPU; treat those as noisy.
- **Per step**: denoising time per step, the part that grows with the step count: Z-Image-Turbo
  runs 9 steps, Qwen-Image-2.1 fast 4 and Qwen-Image-2.1 40.
- **Peak memory**: the most memory mixel used (1 GB = 2^30 bytes, as Apple counts RAM).
  Tests that need more than 85% of this Mac's memory are skipped. "Swapped" means macOS
  moved memory to disk during the test, which makes it slower than the chip can do.
- **Settings**: prompt "A red fox sitting in fresh snow, wildlife photography" (edits: "Turn the fox into a gray wolf",
  on the fast test's image), seed 1. Long runs heat up the chip, and a warm Mac runs
  a little slower, so the 40-step tests are slower per step than the short ones.

## Raw data

```json
{
  "schema": 1,
  "hardware": {"mac": "MacBook Pro", "model_id": "Mac15,10", "chip": "Apple M3 Max", "cpu_performance_cores": 10, "cpu_efficiency_cores": 4, "gpu_cores": 30, "memory_gb": 96, "macos": "26.6.2", "macos_build": "25G83", "power": "AC power", "energy_mode": "Automatic"},
  "software": {"mixel": "mixel 0.2.0", "git": "4a613f1 (modified)", "mlx_rs": "0.32.0"},
  "run": {"started": "2026-10-01T00:09:48Z", "minutes": 24.3, "prompt": "A red fox sitting in fresh snow, wildlife photography", "edit_prompt": "Turn the fox into a gray wolf", "seed": 1},
  "results": [
    {"test": "z512", "label": "Z-Image-Turbo, 512x512", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 13.2, "runs": [{"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 12.8, "per_step_s": 1.4, "vae_s": 0.3, "image_s": 13.2, "peak_gb": 26.7, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 12.8, "per_step_s": 1.4, "vae_s": 0.3, "image_s": 13.2, "peak_gb": 26.7, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 12.8, "per_step_s": 1.4, "vae_s": 0.4, "image_s": 13.3, "peak_gb": 26.7, "swap_gb": 0.0}]},
    {"test": "z1024", "label": "Z-Image-Turbo, 1024x1024", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 60.6, "runs": [{"load_s": 1.3, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 59.6, "per_step_s": 6.6, "vae_s": 1.4, "image_s": 61.2, "peak_gb": 36.4, "swap_gb": 0.0}, {"load_s": 1.3, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 59.1, "per_step_s": 6.6, "vae_s": 1.4, "image_s": 60.6, "peak_gb": 36.4, "swap_gb": 0.0}, {"load_s": 1.3, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 61.0, "per_step_s": 6.8, "vae_s": 1.5, "image_s": 62.7, "peak_gb": 36.4, "swap_gb": 0.0}]},
    {"test": "fast", "label": "Qwen-Image-2.1 fast, 1024x1024", "model": "qwen-image-2.1-fast", "steps": 4, "status": "ok", "reference_image_s": 42.7, "runs": [{"load_s": 3.7, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 40.1, "per_step_s": 10.0, "vae_s": 2.2, "image_s": 42.7, "peak_gb": 52.1, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 49.2, "per_step_s": 12.3, "vae_s": 2.8, "image_s": 52.5, "peak_gb": 52.1, "swap_gb": 0.0}, {"load_s": 1.6, "text_s": 0.2, "init_image_s": 0.4, "denoise_s": 55.4, "per_step_s": 13.9, "vae_s": 2.6, "image_s": 58.6, "peak_gb": 52.1, "swap_gb": 0.0}]},
    {"test": "fastedit", "label": "Qwen-Image-2.1 fast, edit", "model": "qwen-image-2.1-fast", "steps": 4, "status": "ok", "reference_image_s": 58.5, "runs": [{"load_s": 1.5, "text_s": 4.6, "init_image_s": 13.7, "denoise_s": 50.2, "per_step_s": 12.6, "vae_s": 2.1, "image_s": 70.7, "peak_gb": 64.8, "swap_gb": 0.0}, {"load_s": 2.6, "text_s": 3.8, "init_image_s": 11.0, "denoise_s": 45.0, "per_step_s": 11.3, "vae_s": 1.9, "image_s": 61.8, "peak_gb": 64.8, "swap_gb": 0.0}, {"load_s": 2.6, "text_s": 3.5, "init_image_s": 10.2, "denoise_s": 42.8, "per_step_s": 10.7, "vae_s": 1.8, "image_s": 58.5, "peak_gb": 64.8, "swap_gb": 0.0}]},
    {"test": "qwen", "label": "Qwen-Image-2.1, 1024x1024", "model": "qwen-image-2.1", "steps": 40, "status": "ok", "reference_image_s": 390.2, "runs": [{"load_s": 1.1, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 387.6, "per_step_s": 9.7, "vae_s": 2.2, "image_s": 390.2, "peak_gb": 51.4, "swap_gb": 0.0}]},
    {"test": "qwenedit", "label": "Qwen-Image-2.1, edit", "model": "qwen-image-2.1", "steps": 40, "status": "ok", "reference_image_s": 458.3, "runs": [{"load_s": 1.2, "text_s": 4.2, "init_image_s": 11.4, "denoise_s": 440.4, "per_step_s": 11.0, "vae_s": 2.2, "image_s": 458.3, "peak_gb": 64.2, "swap_gb": 0.0}]}
  ]
}
```

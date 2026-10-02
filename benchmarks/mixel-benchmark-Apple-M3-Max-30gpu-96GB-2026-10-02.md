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

mixel 0.3.0 (git 8585685), mlx-rs 0.32.0. Run 2026-10-02T10:53:49Z, took 64.6 min.

## Results

| Test | Time per image (median) | Range | Per step | Peak memory | Reference* |
|---|---:|---:|---:|---:|---:|
| Z-Image-Turbo, 512x512 | **13.6 s** | 13.5-13.7 s | 1.4 s | 19.9 GB | 13.2 s |
| Z-Image-Turbo, 1024x1024 | **59.3 s** | 58.6-59.4 s | 6.3 s | 20.8 GB | 59.0 s |
| Z-Image-Turbo 8-bit, 1024x1024 | **71.6 s** | 67.4-79.5 s | 7.7 s | 12.5 GB | 70.0 s |
| Z-Image-Turbo 4-bit, 1024x1024 | **73.5 s** | 72.6-76.5 s | 7.9 s | 8.1 GB | 70.0 s |
| Qwen-Image-2.1 fast, 1024x1024 | **37.8 s** | 37.7-37.9 s | 8.7 s | 33 GB | 36.8 s |
| Qwen-Image-2.1 fast 4-bit, 1024x1024 | **43.8 s** | 43.5-44.4 s | 10.3 s | 14.6 GB | 42.0 s |
| Qwen-Image-2.1 fast, edit | **57.8 s** | 57.8-58.3 s | 10.5 s | 34.1 GB | 63.6 s |
| Qwen-Image-2.1 fast 4-bit, edit | **64.6 s** | 64.3-65.0 s | 11.7 s | 15.5 GB | 70.0 s |
| Qwen-Image-2.1, 1024x1024 | **365.6 s** | 363.0-365.8 s | 9.1 s | 32.9 GB | 373.0 s |
| Qwen-Image-2.1, edit | **484.1 s** | 450.1-488.9 s | 11.7 s | 34 GB | 444.5 s |

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
  "software": {"mixel": "mixel 0.3.0", "git": "8585685", "mlx_rs": "0.32.0"},
  "run": {"started": "2026-10-02T10:53:49Z", "minutes": 64.6, "prompt": "A red fox sitting in fresh snow, wildlife photography", "edit_prompt": "Turn the fox into a gray wolf", "seed": 1},
  "results": [
    {"test": "z512", "label": "Z-Image-Turbo, 512x512", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 13.2, "runs": [{"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 12.9, "per_step_s": 1.4, "vae_s": 0.6, "image_s": 13.6, "peak_gb": 19.9, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 12.8, "per_step_s": 1.4, "vae_s": 0.6, "image_s": 13.5, "peak_gb": 19.8, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 12.8, "per_step_s": 1.4, "vae_s": 0.7, "image_s": 13.7, "peak_gb": 19.9, "swap_gb": 0.0}]},
    {"test": "z1024", "label": "Z-Image-Turbo, 1024x1024", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 59.0, "runs": [{"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 56.8, "per_step_s": 6.3, "vae_s": 2.4, "image_s": 59.3, "peak_gb": 20.8, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 56.1, "per_step_s": 6.2, "vae_s": 2.4, "image_s": 58.6, "peak_gb": 20.8, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 56.9, "per_step_s": 6.3, "vae_s": 2.4, "image_s": 59.4, "peak_gb": 20.8, "swap_gb": 0.0}]},
    {"test": "z1024q8", "label": "Z-Image-Turbo 8-bit, 1024x1024", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 70.0, "runs": [{"load_s": 3.4, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 64.9, "per_step_s": 7.2, "vae_s": 2.4, "image_s": 67.4, "peak_gb": 12.5, "swap_gb": 0.0}, {"load_s": 3.4, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 69.1, "per_step_s": 7.7, "vae_s": 2.4, "image_s": 71.6, "peak_gb": 12.5, "swap_gb": 0.0}, {"load_s": 3.4, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 77.0, "per_step_s": 8.6, "vae_s": 2.4, "image_s": 79.5, "peak_gb": 12.5, "swap_gb": 0.0}]},
    {"test": "z1024q4", "label": "Z-Image-Turbo 4-bit, 1024x1024", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 70.0, "runs": [{"load_s": 3.3, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 74.0, "per_step_s": 8.2, "vae_s": 2.4, "image_s": 76.5, "peak_gb": 8.1, "swap_gb": 0.0}, {"load_s": 3.2, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 71.1, "per_step_s": 7.9, "vae_s": 2.4, "image_s": 73.5, "peak_gb": 8.0, "swap_gb": 0.0}, {"load_s": 3.3, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 70.1, "per_step_s": 7.8, "vae_s": 2.4, "image_s": 72.6, "peak_gb": 8.1, "swap_gb": 0.0}]},
    {"test": "fast", "label": "Qwen-Image-2.1 fast, 1024x1024", "model": "qwen-image-2.1-fast", "steps": 4, "status": "ok", "reference_image_s": 36.8, "runs": [{"load_s": 2.8, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 34.9, "per_step_s": 8.7, "vae_s": 2.6, "image_s": 37.9, "peak_gb": 32.9, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 34.8, "per_step_s": 8.7, "vae_s": 2.6, "image_s": 37.7, "peak_gb": 32.9, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 34.9, "per_step_s": 8.7, "vae_s": 2.6, "image_s": 37.8, "peak_gb": 33.0, "swap_gb": 0.0}]},
    {"test": "fastq4", "label": "Qwen-Image-2.1 fast 4-bit, 1024x1024", "model": "qwen-image-2.1-fast", "steps": 4, "status": "ok", "reference_image_s": 42.0, "runs": [{"load_s": 3.1, "text_s": 0.1, "init_image_s": 0.1, "denoise_s": 40.7, "per_step_s": 10.2, "vae_s": 2.6, "image_s": 43.5, "peak_gb": 14.6, "swap_gb": 0.0}, {"load_s": 3.0, "text_s": 0.1, "init_image_s": 0.1, "denoise_s": 41.0, "per_step_s": 10.3, "vae_s": 2.6, "image_s": 43.8, "peak_gb": 14.6, "swap_gb": 0.0}, {"load_s": 3.0, "text_s": 0.1, "init_image_s": 0.1, "denoise_s": 41.6, "per_step_s": 10.4, "vae_s": 2.6, "image_s": 44.4, "peak_gb": 14.6, "swap_gb": 0.0}]},
    {"test": "fastedit", "label": "Qwen-Image-2.1 fast, edit", "model": "qwen-image-2.1-fast", "steps": 4, "status": "ok", "reference_image_s": 63.6, "runs": [{"load_s": 1.2, "text_s": 3.2, "init_image_s": 10.1, "denoise_s": 41.9, "per_step_s": 10.5, "vae_s": 2.7, "image_s": 57.8, "peak_gb": 34.1, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 3.2, "init_image_s": 10.0, "denoise_s": 41.9, "per_step_s": 10.5, "vae_s": 2.6, "image_s": 57.8, "peak_gb": 34.1, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 3.2, "init_image_s": 10.1, "denoise_s": 42.4, "per_step_s": 10.6, "vae_s": 2.7, "image_s": 58.3, "peak_gb": 34.1, "swap_gb": 0.0}]},
    {"test": "fasteditq4", "label": "Qwen-Image-2.1 fast 4-bit, edit", "model": "qwen-image-2.1-fast", "steps": 4, "status": "ok", "reference_image_s": 70.0, "runs": [{"load_s": 3.0, "text_s": 3.6, "init_image_s": 11.3, "denoise_s": 47.4, "per_step_s": 11.8, "vae_s": 2.6, "image_s": 65.0, "peak_gb": 15.5, "swap_gb": 0.0}, {"load_s": 3.0, "text_s": 3.6, "init_image_s": 11.4, "denoise_s": 47.0, "per_step_s": 11.7, "vae_s": 2.7, "image_s": 64.6, "peak_gb": 15.3, "swap_gb": 0.0}, {"load_s": 3.0, "text_s": 3.6, "init_image_s": 11.2, "denoise_s": 46.8, "per_step_s": 11.7, "vae_s": 2.6, "image_s": 64.3, "peak_gb": 15.3, "swap_gb": 0.0}]},
    {"test": "qwen", "label": "Qwen-Image-2.1, 1024x1024", "model": "qwen-image-2.1", "steps": 40, "status": "ok", "reference_image_s": 373.0, "runs": [{"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 360.0, "per_step_s": 9.0, "vae_s": 2.7, "image_s": 363.0, "peak_gb": 32.9, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 362.7, "per_step_s": 9.1, "vae_s": 2.8, "image_s": 365.8, "peak_gb": 32.9, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 0.1, "init_image_s": 0.2, "denoise_s": 362.6, "per_step_s": 9.1, "vae_s": 2.7, "image_s": 365.6, "peak_gb": 32.9, "swap_gb": 0.0}]},
    {"test": "qwenedit", "label": "Qwen-Image-2.1, edit", "model": "qwen-image-2.1", "steps": 40, "status": "ok", "reference_image_s": 444.5, "runs": [{"load_s": 1.2, "text_s": 3.3, "init_image_s": 10.0, "denoise_s": 433.9, "per_step_s": 10.8, "vae_s": 2.9, "image_s": 450.1, "peak_gb": 33.8, "swap_gb": 0.0}, {"load_s": 1.2, "text_s": 3.3, "init_image_s": 9.9, "denoise_s": 472.8, "per_step_s": 11.8, "vae_s": 2.9, "image_s": 488.9, "peak_gb": 33.8, "swap_gb": 0.0}, {"load_s": 1.3, "text_s": 3.6, "init_image_s": 11.2, "denoise_s": 466.3, "per_step_s": 11.7, "vae_s": 2.9, "image_s": 484.1, "peak_gb": 34.0, "swap_gb": 0.0}]}
  ]
}
```

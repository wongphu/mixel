# mixel benchmark: Apple M4, 10-core GPU, 16 GB

> **Please email this file to mixelate@proton.me** so we can compile results across Macs.
> It holds the hardware summary and timings below, nothing else: no serial numbers,
> hostnames, user names or images.

## Hardware

| | |
|---|---|
| Mac | Mac mini (Mac16,10) |
| Chip | Apple M4 |
| CPU | 10 cores (4 performance, 6 efficiency) |
| GPU | 10 cores |
| Memory | 16 GB |
| macOS | 27.0 (26A428) |
| Power | AC power, energy mode n/a |

## Software

mixel 0.5.0 (git 8545983), mlx-rs 0.32.0. Run 2026-10-03T04:15:27Z, took 52.1 min.

## Results

| Test | Time per image (median) | Range | Per step | Peak memory | Reference* |
|---|---:|---:|---:|---:|---:|
| Z-Image-Turbo, 512x512 | **48.2 s** (swapped 0.9 GB) | 46.2-56.9 s | 5.1 s | 12.4 GB | 13.6 s |
| Z-Image-Turbo, 1024x1024 | **329.6 s** (swapped 0.7 GB) | 196.9-530.2 s | 36.1 s | 13.4 GB | 59.3 s |
| Z-Image-Turbo 8-bit, 1024x1024 | **155.1 s** | 152.6-156.5 s | 16.7 s | 8.2 GB | 71.6 s |
| Z-Image-Turbo 4-bit, 1024x1024 | **157.7 s** | 156.9-162.8 s | 17.0 s | 5.4 GB | 73.5 s |
| Qwen-Image-2.1 fast, 1024x1024 | skipped: needs ~17.0 GB | | | | 37.8 s |
| Qwen-Image-2.1 fast 4-bit, 1024x1024 | **90.4 s** | 90.3-91.2 s | 21.1 s | 8 GB | 43.8 s |
| Qwen-Image-2.1 fast, edit | skipped: needs ~18.3 GB | | | | 57.8 s |
| Qwen-Image-2.1 fast 4-bit, edit | **125.1 s** | 124.9-125.2 s | 24.3 s | 8.2 GB | 64.6 s |
| Qwen-Image-2.1, 1024x1024 | skipped: needs ~16.7 GB | | | | 365.6 s |
| Qwen-Image-2.1, edit | skipped: needs ~18.3 GB | | | | 484.1 s |

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
  "hardware": {"mac": "Mac mini", "model_id": "Mac16,10", "chip": "Apple M4", "cpu_performance_cores": 4, "cpu_efficiency_cores": 6, "gpu_cores": 10, "memory_gb": 16, "macos": "27.0", "macos_build": "26A428", "power": "AC power", "energy_mode": "n/a"},
  "software": {"mixel": "mixel 0.5.0", "git": "8545983", "mlx_rs": "0.32.0"},
  "run": {"started": "2026-10-03T04:15:27Z", "minutes": 52.1, "prompt": "A red fox sitting in fresh snow, wildlife photography", "edit_prompt": "Turn the fox into a gray wolf", "seed": 1},
  "results": [
    {"test": "z512", "label": "Z-Image-Turbo, 512x512", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 13.6, "runs": [{"load_s": 17.0, "text_s": 0.6, "init_image_s": 0.0, "denoise_s": 46.1, "per_step_s": 5.1, "vae_s": 2.1, "image_s": 48.2, "peak_gb": 12.4, "swap_gb": 0.0}, {"load_s": 16.7, "text_s": 0.5, "init_image_s": 0.0, "denoise_s": 54.4, "per_step_s": 6.0, "vae_s": 2.6, "image_s": 56.9, "peak_gb": 12.4, "swap_gb": 0.9}, {"load_s": 16.8, "text_s": 0.6, "init_image_s": 0.0, "denoise_s": 43.7, "per_step_s": 4.9, "vae_s": 2.5, "image_s": 46.2, "peak_gb": 12.4, "swap_gb": 0.0}]},
    {"test": "z1024", "label": "Z-Image-Turbo, 1024x1024", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 59.3, "runs": [{"load_s": 16.9, "text_s": 0.5, "init_image_s": 0.0, "denoise_s": 324.7, "per_step_s": 36.1, "vae_s": 4.8, "image_s": 329.6, "peak_gb": 13.4, "swap_gb": 0.0}, {"load_s": 16.8, "text_s": 0.6, "init_image_s": 0.0, "denoise_s": 191.8, "per_step_s": 21.3, "vae_s": 5.1, "image_s": 196.9, "peak_gb": 13.4, "swap_gb": 0.2}, {"load_s": 17.0, "text_s": 0.7, "init_image_s": 0.0, "denoise_s": 523.2, "per_step_s": 58.1, "vae_s": 6.9, "image_s": 530.2, "peak_gb": 13.4, "swap_gb": 0.7}]},
    {"test": "z1024q8", "label": "Z-Image-Turbo 8-bit, 1024x1024", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 71.6, "runs": [{"load_s": 4.9, "text_s": 0.2, "init_image_s": 0.0, "denoise_s": 148.2, "per_step_s": 16.5, "vae_s": 4.4, "image_s": 152.6, "peak_gb": 8.2, "swap_gb": 0.0}, {"load_s": 5.1, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 150.7, "per_step_s": 16.7, "vae_s": 4.4, "image_s": 155.1, "peak_gb": 8.2, "swap_gb": 0.0}, {"load_s": 5.3, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 152.1, "per_step_s": 16.9, "vae_s": 4.4, "image_s": 156.5, "peak_gb": 8.2, "swap_gb": 0.0}]},
    {"test": "z1024q4", "label": "Z-Image-Turbo 4-bit, 1024x1024", "model": "z-image-turbo", "steps": 9, "status": "ok", "reference_image_s": 73.5, "runs": [{"load_s": 0.9, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 152.5, "per_step_s": 16.9, "vae_s": 4.4, "image_s": 156.9, "peak_gb": 5.4, "swap_gb": 0.0}, {"load_s": 1.5, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 153.1, "per_step_s": 17.0, "vae_s": 4.6, "image_s": 157.7, "peak_gb": 5.4, "swap_gb": 0.0}, {"load_s": 2.4, "text_s": 0.1, "init_image_s": 0.0, "denoise_s": 158.2, "per_step_s": 17.6, "vae_s": 4.6, "image_s": 162.8, "peak_gb": 5.4, "swap_gb": 0.0}]},
    {"test": "fast", "label": "Qwen-Image-2.1 fast, 1024x1024", "model": "qwen-image-2.1-fast", "status": "skipped", "needs_gb": 17.0},
    {"test": "fastq4", "label": "Qwen-Image-2.1 fast 4-bit, 1024x1024", "model": "qwen-image-2.1-fast", "steps": 4, "status": "ok", "reference_image_s": 43.8, "runs": [{"load_s": 22.7, "text_s": 0.6, "init_image_s": 0.7, "denoise_s": 84.8, "per_step_s": 21.2, "vae_s": 5.6, "image_s": 91.2, "peak_gb": 7.9, "swap_gb": 0.0}, {"load_s": 4.7, "text_s": 0.3, "init_image_s": 0.5, "denoise_s": 84.4, "per_step_s": 21.1, "vae_s": 5.5, "image_s": 90.4, "peak_gb": 8.0, "swap_gb": 0.0}, {"load_s": 4.7, "text_s": 0.3, "init_image_s": 0.4, "denoise_s": 84.4, "per_step_s": 21.1, "vae_s": 5.4, "image_s": 90.3, "peak_gb": 8.0, "swap_gb": 0.0}]},
    {"test": "fastedit", "label": "Qwen-Image-2.1 fast, edit", "model": "qwen-image-2.1-fast", "status": "skipped", "needs_gb": 18.3},
    {"test": "fasteditq4", "label": "Qwen-Image-2.1 fast 4-bit, edit", "model": "qwen-image-2.1-fast", "steps": 4, "status": "ok", "reference_image_s": 64.6, "runs": [{"load_s": 4.7, "text_s": 7.8, "init_image_s": 22.5, "denoise_s": 97.2, "per_step_s": 24.3, "vae_s": 5.4, "image_s": 125.1, "peak_gb": 8.2, "swap_gb": 0.0}, {"load_s": 4.9, "text_s": 7.7, "init_image_s": 22.6, "denoise_s": 97.2, "per_step_s": 24.3, "vae_s": 5.4, "image_s": 125.2, "peak_gb": 8.2, "swap_gb": 0.0}, {"load_s": 5.0, "text_s": 7.7, "init_image_s": 22.6, "denoise_s": 96.9, "per_step_s": 24.2, "vae_s": 5.4, "image_s": 124.9, "peak_gb": 8.2, "swap_gb": 0.0}]},
    {"test": "qwen", "label": "Qwen-Image-2.1, 1024x1024", "model": "qwen-image-2.1", "status": "skipped", "needs_gb": 16.7},
    {"test": "qwenedit", "label": "Qwen-Image-2.1, edit", "model": "qwen-image-2.1", "status": "skipped", "needs_gb": 18.3}
  ]
}
```

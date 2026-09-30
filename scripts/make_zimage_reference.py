"""Reference tensors for checking mixel's Z-Image-Turbo sampling against diffusers.

Runs diffusers' ZImagePipeline (bf16, MPS) at 512x512 with 9 steps from fixed
noise and records every transformer call, the VAE input and output, and the
image, for `cargo run --release --example zimage_diffusers_parity -- <out-dir>`.

    python make_zimage_reference.py

Needs torch and diffusers >= 0.41 (with ZImagePipeline). Writes into the
current directory.
"""
import json, time

import numpy as np, torch
from safetensors.torch import save_file
from diffusers import ZImagePipeline

dev = "mps"
pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", dtype=torch.bfloat16).to(dev)
pipe.set_progress_bar_config(disable=True)


def f32(t):
    return t.detach().to("cpu", torch.float32).contiguous()


rec, calls = {}, []
tr_fwd = pipe.transformer.forward


def tr_hook(x, t, cap_feats, *a, **kw):
    out = tr_fwd(x, t, cap_feats, *a, **kw)
    i = len(calls)
    calls.append(t.float().item())
    rec[f"tr_x_{i}"] = f32(x[0])  # (16, 1, H, W), bf16 model input
    rec[f"tr_t_{i}"] = f32(t)
    rec[f"tr_out_{i}"] = f32(out[0][0])
    if i == 0:
        rec["tr_cap"] = f32(cap_feats[0])
    return out


vae_dec = pipe.vae.decode


def dec_hook(z, *a, **kw):
    out = vae_dec(z, *a, **kw)
    rec["vae_dec_in"] = f32(z)
    rec["vae_dec_out"] = f32(out[0] if isinstance(out, tuple) else out.sample)
    return out


pipe.transformer.forward = tr_hook
pipe.vae.decode = dec_hook

rng = np.random.default_rng(5)
latents = torch.from_numpy(rng.standard_normal((1, 16, 64, 64)).astype(np.float32))
save_file({"latents": latents}, "zimage_noise.safetensors")

t0 = time.time()
with torch.inference_mode():
    img = pipe(
        prompt="A red fox sitting in fresh snow, wildlife photography",
        height=512,
        width=512,
        num_inference_steps=9,
        guidance_scale=0.0,
        latents=latents.to(dev),
        output_type="pil",
    ).images[0]
print(f"{time.time() - t0:.1f}s, {len(calls)} transformer calls, t = {calls}", flush=True)

rec["sched_sigmas"] = f32(pipe.scheduler.sigmas)
img.save("zimage.png")
save_file(rec, "zimage.safetensors", metadata={"meta": json.dumps({"timesteps": calls})})
print(sorted((k, tuple(v.shape)) for k, v in rec.items()), flush=True)

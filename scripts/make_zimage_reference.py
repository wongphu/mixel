"""Reference tensors for checking mixel's Z-Image-Turbo sampling against diffusers.

Runs diffusers' ZImagePipeline (bf16, MPS) at 512x512 with 9 steps from fixed
noise, without guidance and with guidance 3 against the default empty negative
prompt, and records every transformer call, the VAE input and output, and the
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


def record(name, **call):
    """Runs the pipeline and saves `<name>.safetensors` and `<name>.png`."""
    rec, calls = {}, []
    tr_fwd = pipe.transformer.forward

    # With guidance, each call is a batch of [prompt, negative prompt].
    def tr_hook(x, t, cap_feats, *a, **kw):
        out = tr_fwd(x, t, cap_feats, *a, **kw)
        i = len(calls)
        calls.append(t[0].float().item())
        rec[f"tr_x_{i}"] = f32(x[0])  # (16, 1, H, W), bf16 model input
        rec[f"tr_t_{i}"] = f32(t[:1])
        rec[f"tr_out_{i}"] = f32(out[0][0])
        if len(out[0]) > 1:
            rec[f"tr_out_neg_{i}"] = f32(out[0][1])
        if i == 0:
            rec["tr_cap"] = f32(cap_feats[0])
            if len(cap_feats) > 1:
                rec["tr_cap_neg"] = f32(cap_feats[1])
        return out

    vae_dec = pipe.vae.decode

    def dec_hook(z, *a, **kw):
        out = vae_dec(z, *a, **kw)
        rec["vae_dec_in"] = f32(z)
        rec["vae_dec_out"] = f32(out[0] if isinstance(out, tuple) else out.sample)
        return out

    pipe.transformer.forward = tr_hook
    pipe.vae.decode = dec_hook
    try:
        t0 = time.time()
        with torch.inference_mode():
            img = pipe(**call, output_type="pil").images[0]
        print(f"{name}: {time.time() - t0:.1f}s, {len(calls)} transformer calls, t = {calls}", flush=True)
    finally:
        pipe.transformer.forward = tr_fwd
        pipe.vae.decode = vae_dec

    rec["sched_sigmas"] = f32(pipe.scheduler.sigmas)
    img.save(f"{name}.png")
    save_file(rec, f"{name}.safetensors", metadata={"meta": json.dumps({"timesteps": calls})})
    print(sorted((k, tuple(v.shape)) for k, v in rec.items()), flush=True)


rng = np.random.default_rng(5)
latents = torch.from_numpy(rng.standard_normal((1, 16, 64, 64)).astype(np.float32))
save_file({"latents": latents}, "zimage_noise.safetensors")
prompt = "A red fox sitting in fresh snow, wildlife photography"
common = dict(prompt=prompt, height=512, width=512, num_inference_steps=9, latents=latents.to(dev))
record("zimage", guidance_scale=0.0, **common)
# Guidance against the default (empty) negative prompt.
record("zimage_cfg", guidance_scale=3.0, **common)

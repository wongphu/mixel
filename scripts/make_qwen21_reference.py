"""Reference tensors for checking mixel's Qwen-Image-2.1 port.

Runs the diffusers pipeline (bf16, MPS) with fixed initial noise and records
each stage's inputs/outputs to safetensors, for a text-to-image and an edit run,
for `cargo run --release --example qwen21_parity -- <out-dir> <image>`.

    python make_qwen21_reference.py <model-snapshot-dir> <reference-image.png>

Needs torch, torchvision, diffusers >= 0.41 (with QwenImage21Pipeline) and
transformers >= 5, and the whole snapshot (`hf download Qwen/Qwen-Image-2.1`).
The image should be 512x512, the size the edit runs at (the parity example
feeds it to the vision encoder as is). Writes into the current directory.
"""
import json, sys, time
import numpy as np, torch
from PIL import Image
from safetensors.torch import save_file
from diffusers import QwenImage21Pipeline
from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as vae_mod

# PyTorch's MPS backend returns zeros from AvgDown3D's 8-D reshape/permute at
# >= 256 px, which corrupts VAE encoding (and so every edit). Run just that
# module on the CPU.
_avg_down = vae_mod.QwenImage21AvgDown3D.forward
vae_mod.QwenImage21AvgDown3D.forward = lambda self, x: _avg_down(self, x.float().cpu()).to(x.device, x.dtype)

SNAP = sys.argv[1]
FOX = sys.argv[2]
dev = "mps"
pipe = QwenImage21Pipeline.from_pretrained(SNAP, torch_dtype=torch.bfloat16).to(dev)
pipe.set_progress_bar_config(disable=True)


def f32(t):
    return t.detach().to("cpu", torch.float32).contiguous()


def record(run_name, **call):
    rec, calls = {}, []
    tr_fwd = pipe.transformer.forward

    def tr_hook(*a, **kw):
        out = tr_fwd(*a, **kw)
        calls.append(kw["timestep"].float().item())
        if len(calls) == 1:
            rec["tr_hidden_in"] = f32(kw["hidden_states"])
            rec["tr_encoder_in"] = f32(kw["encoder_hidden_states"])
            rec["tr_img_mask"] = f32(kw["img_mask"].to(torch.float32))
            rec["tr_timestep"] = f32(kw["timestep"].to(torch.float32))
            rec["tr_out"] = f32(out[0])
            rec["_img_shapes"] = kw["img_shapes"][0]
        return out

    vae_dec = pipe.vae.decode

    def dec_hook(z, *a, **kw):
        out = vae_dec(z, *a, **kw)
        rec["vae_dec_in"] = f32(z)
        rec["vae_dec_out"] = f32(out[0] if isinstance(out, tuple) else out.sample)
        return out

    enc_img = pipe._encode_vae_image

    def enc_hook(image, generator):
        out = enc_img(image, generator)
        rec["vae_enc_in"] = f32(image)
        rec["vae_enc_out"] = f32(out)
        return out

    visual = pipe.text_encoder.model.visual
    vis_fwd = visual.forward

    def vis_hook(pixel_values, grid_thw=None, **kw):
        out = vis_fwd(pixel_values, grid_thw=grid_thw, **kw)
        rec["vis_pixel_values"] = f32(pixel_values)
        rec["vis_grid_thw"] = f32(grid_thw.to(torch.float32))
        rec["vis_merged"] = f32(out.pooler_output)
        for i, d in enumerate(out.deepstack_features):
            rec[f"vis_deepstack_{i}"] = f32(d)
        return out

    proc = pipe.processor

    class ProcWrap:
        def __getattr__(self, n):
            return getattr(proc, n)

        def __call__(self, *a, **kw):
            out = proc(*a, **kw)
            rec["proc_input_ids"] = f32(out.input_ids.to(torch.float32))
            return out

    pipe.transformer.forward = tr_hook
    pipe.vae.decode = dec_hook
    pipe._encode_vae_image = enc_hook
    visual.forward = vis_hook
    pipe.processor = ProcWrap()
    try:
        t0 = time.time()
        img = pipe(**call, output_type="pil").images[0]
        print(f"{run_name}: {time.time() - t0:.1f}s, {len(calls)} transformer calls", flush=True)
    finally:
        pipe.transformer.forward = tr_fwd
        pipe.vae.decode = vae_dec
        pipe._encode_vae_image = enc_img
        visual.forward = vis_fwd
        pipe.processor = proc

    rec["sched_sigmas"] = f32(pipe.scheduler.sigmas)
    meta = {"img_shapes": rec.pop("_img_shapes"), "timesteps": calls, "image_mode": img.mode, "image_size": img.size}
    img.save(f"{run_name}.png")
    save_file(rec, f"{run_name}.safetensors", metadata={"meta": json.dumps(meta)})
    print(run_name, json.dumps(meta), sorted((k, tuple(v.shape)) for k, v in rec.items()), flush=True)


def noise(seed, n_tokens):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, n_tokens, 64)).astype(np.float32)
    return torch.from_numpy(x).to(dev, torch.bfloat16)


with torch.no_grad():
    prompt = "A red fox sitting in fresh snow, wildlife photography"
    lat = noise(1, 32 * 32)
    save_file({"latents": f32(lat)}, "t2i_noise.safetensors")
    record("t2i", prompt=prompt, height=512, width=512, num_inference_steps=8, latents=lat)

    fox = Image.open(FOX).convert("RGB")
    lat = noise(2, 32 * 32)
    save_file({"latents": f32(lat)}, "edit_noise.safetensors")
    record(
        "edit",
        prompt="Turn the fox into a gray wolf",
        image=fox,
        output_resolution=512,
        num_inference_steps=4,
        latents=lat,
    )

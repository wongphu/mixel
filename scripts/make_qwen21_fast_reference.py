"""Reference tensors for checking mixel's 4-step Qwen-Image-2.1 (Fun-Acc adapter).

Runs diffusers' QwenImage21Pipeline (bf16, MPS) with the adapter loaded by its
authors' own code (`qwenimage21_pdd.py` from alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs)
from fixed noise, and records every transformer call, the final latents and the
image, for a text-to-image and an edit run, for
`cargo run --release --example qwen21_fast_parity -- <out-dir> <image>`.

    python make_qwen21_fast_reference.py <model-snapshot-dir> <reference-image.png>

Needs torch, torchvision, diffusers >= 0.41 (with QwenImage21Pipeline),
transformers >= 5 and huggingface_hub, and the whole snapshot
(`hf download Qwen/Qwen-Image-2.1`). Writes into the current directory.
"""
import json, sys, time
from contextlib import contextmanager
from unittest import mock

import numpy as np, torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import save_file
from diffusers import QwenImage21Pipeline
from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as vae_mod

ADAPTER = "alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs"
for f in ["qwenimage21_pdd.py", "lora_utils_pdd.py", "models/pdd_config.json"]:
    path = hf_hub_download(ADAPTER, f)
weights = hf_hub_download(ADAPTER, "models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors")
sys.path.insert(0, path.rsplit("/models/", 1)[0])
from qwenimage21_pdd import QwenImage21PDDScheduler, load_pdd_lora, pdd_step_callback  # noqa: E402

# PyTorch's MPS backend returns zeros from AvgDown3D's 8-D reshape/permute at
# >= 256 px, which corrupts VAE encoding (and so every edit). Run just that
# module on the CPU.
_avg_down = vae_mod.QwenImage21AvgDown3D.forward
vae_mod.QwenImage21AvgDown3D.forward = lambda self, x: _avg_down(self, x.float().cpu()).to(x.device, x.dtype)


@contextmanager
def cuda_latent_dtype():
    """The pipeline rounds the latents back to their bf16 dtype after each step
    only when MPS is available (a workaround for an old PyTorch bug). The adapter
    is meant to keep them in f32 (`native_time_fp32_state`), as it does on CUDA,
    so hide MPS from that one check."""
    with mock.patch("torch.backends.mps.is_available", return_value=False):
        yield


SNAP = sys.argv[1]
FOX = sys.argv[2]
dev = "mps"
pipe = QwenImage21Pipeline.from_pretrained(SNAP, torch_dtype=torch.bfloat16).to(dev)
config = load_pdd_lora(pipe.transformer, weights)
pipe.transformer.to(dev)  # the adapter's modules are created on the CPU
pipe.transformer.eval()
pipe.scheduler = QwenImage21PDDScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.register_to_config(**config)
sigmas = torch.tensor(config["pdd_sigmas"], dtype=torch.float32)
pipe.set_progress_bar_config(disable=True)


def f32(t):
    return t.detach().to("cpu", torch.float32).contiguous()


def record(run_name, **call):
    rec, calls = {}, []
    tr_fwd = pipe.transformer.forward

    # Runs inside the module call, after the adapter's hook has set the
    # timestep and cast the input to bf16.
    def tr_hook(*a, **kw):
        out = tr_fwd(*a, **kw)
        i = len(calls)
        calls.append(kw["timestep"].float().item())
        rec[f"tr_hidden_in_{i}"] = f32(kw["hidden_states"])
        rec[f"tr_out_{i}"] = f32(out[0])
        if i == 0:
            rec["tr_encoder_in"] = f32(kw["encoder_hidden_states"])
            rec["tr_img_mask"] = f32(kw["img_mask"].to(torch.float32))
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
        with torch.inference_mode(), cuda_latent_dtype():
            img = pipe(
                **call,
                true_cfg_scale=1.0,
                use_kv_cache=False,
                num_inference_steps=4,
                callback_on_step_end=pdd_step_callback(pipe.transformer, sigmas, config["pdd_block_size"]),
                output_type="pil",
            ).images[0]
        print(f"{run_name}: {time.time() - t0:.1f}s, {len(calls)} transformer calls", flush=True)
    finally:
        pipe.transformer.forward = tr_fwd
        pipe.vae.decode = vae_dec

    rec["sched_sigmas"] = f32(pipe.scheduler.sigmas)
    meta = {"timesteps": calls, "image_mode": img.mode, "image_size": img.size}
    img.save(f"fast_{run_name}.png")
    save_file(rec, f"fast_{run_name}.safetensors", metadata={"meta": json.dumps(meta)})
    print(run_name, json.dumps(meta), sorted((k, tuple(v.shape)) for k, v in rec.items()), flush=True)


def noise(seed, n_tokens):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, n_tokens, 64)).astype(np.float32)
    return torch.from_numpy(x).to(dev, torch.bfloat16)


lat = noise(3, 32 * 32)
save_file({"latents": f32(lat)}, "fast_t2i_noise.safetensors")
record("t2i", prompt="A red fox sitting in fresh snow, wildlife photography", height=512, width=512, latents=lat)

fox = Image.open(FOX).convert("RGB")
lat = noise(4, 32 * 32)
save_file({"latents": f32(lat)}, "fast_edit_noise.safetensors")
record("edit", prompt="Turn the fox into a gray wolf", image=fox, output_resolution=512, latents=lat)

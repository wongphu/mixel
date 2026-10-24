"""Reference tensors for checking mixel's 4-step MOSS-Image (the rapid-moss
adapter, darask0/MOSS-Image_4-step_adapter_lora).

Applies the adapter as its `inference.py` does (an unmerged f32 LoRA on every
block's projections; the time embedding and modulations in f32 with the
adapter's tensors) to the reference pipeline (see make_moss_reference.py),
runs its 4-step sampler at 512x512 from fixed noise, and records every
transformer call and the VAE decode (and the same steps with the transformer
in f32) to `moss_fast.safetensors`, for
`cargo run --release --example moss_fast_parity -- .`.

    python make_moss_fast_reference.py <model-snapshot-dir>

Needs what make_moss_reference.py needs, plus huggingface_hub (it downloads
the adapter at the revision mixel was checked against). Writes into the
current directory.
"""
import sys, time
import numpy as np, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from make_moss_reference import PROMPTS, dev, f32, load_pipeline

REPO, REVISION = "darask0/MOSS-Image_4-step_adapter_lora", "c88f33bcb3251eb39943d0635e74b650732977c8"


def adapter_file(name):
    return hf_hub_download(REPO, name, revision=REVISION)


def apply_adapter(tr):
    """As the adapter's inference.py `load()` does."""
    for name in ("time_embedding", "shared_block_modulation", "video_head_modulation"):
        getattr(tr, name).float()
    result = tr.load_state_dict(load_file(adapter_file("time_embedding.safetensors"), device=dev), strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    lora = load_file(adapter_file("lora.safetensors"), device=dev)
    for i, layer in enumerate(tr.layers):
        for name, lin in layer.proj.items():
            key = f"diffusion_model.layers.{i}.proj.{name}"
            down, up = lora[key + ".lora_down.weight"], lora[key + ".lora_up.weight"]
            scale = float(lora[key + ".alpha"]) / down.shape[0]

            def hook(module, inputs, output, down=down, up=up, scale=scale):
                return output + (scale * ((inputs[0].float() @ down.t()) @ up.t())).to(output.dtype)

            lin.register_forward_hook(hook)
    ds = load_file(adapter_file("dataspace.safetensors"), device=dev)
    return ds["latent_mix.Q"], ds["step_codes"]


if __name__ == "__main__":
    pipe = load_pipeline(sys.argv[1])
    Q, codes = apply_adapter(pipe.transformer)
    rec = {}
    with torch.no_grad():
        emb, mask = pipe.encode_prompt(PROMPTS["fox"])
        rng = np.random.default_rng(3)
        z = torch.from_numpy(rng.standard_normal((1, 128, 32, 32)).astype(np.float32)).to(dev)
        save_file({"latents": f32(z)}, "moss_fast_noise.safetensors")
        rec["txt"] = f32(emb[0])
        t0 = time.time()
        x0 = torch.zeros_like(z)
        for r in range(len(codes)):
            x = z + torch.einsum("oc,bchw->bohw", Q, x0)
            v = pipe.transformer(x, codes[r], emb, mask).sample.float()
            x0 = z - v
            rec[f"x_{r}"], rec[f"v_{r}"], rec[f"x0_{r}"] = f32(x), f32(v), f32(x0)
        vae = pipe.vae
        lat = x0.to(vae.dtype)
        bn, eps = vae.bn, vae.config.batch_norm_eps
        lat = lat * (bn.running_var + eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.view(1, -1, 1, 1)
        b, c, h, w = lat.shape
        lat = lat.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
        image = vae.decode(lat, return_dict=False)[0].clamp(-1, 1)
        rec["image"] = f32(image)
        print(f"moss_fast: {time.time() - t0:.1f}s, step codes {codes.tolist()}", flush=True)
        pipe.image_processor.postprocess(image, output_type="pil")[0].save("moss_fast.png")

        # The same 4 steps with the whole transformer in f32, where two correct
        # implementations should agree closely.
        pipe.transformer.to(torch.float32)
        x0 = torch.zeros_like(z)
        for r in range(len(codes)):
            x = z + torch.einsum("oc,bchw->bohw", Q, x0)
            v = pipe.transformer(x, codes[r], emb, mask).sample.float()
            rec[f"f32_v_{r}"] = f32(v)
            x0 = z - v
        rec["f32_x0"] = f32(x0)
    save_file(rec, "moss_fast.safetensors")

"""Reference tensors for checking mixel's MOSS-Image-SFT port.

Runs the pipeline bundled with the weights (`pipeline_moss_image.py`, bf16 on
MPS, the VAE in f32) with fixed initial noise and records each stage's inputs
and outputs to safetensors, for `cargo run --release --example moss_parity -- .`:

- `moss_te.safetensors`: token ids, every decoder layer's output and the final
  embeddings for three prompts, and one Gated DeltaNet kernel call's inputs and
  output (layer 0 of the long prompt, which spans three 64-token chunks);
- `moss_t2i.safetensors`: a 512x512, 8-step run with the default guidance
  (CFG 3, the negative branch skipping block 22): every transformer call, the
  scheduler, the VAE decode, and a VAE encode of the decoded image;
- `moss_t2i_f32.safetensors`: the same 8 steps with an f32 transformer;
- `moss_sched.safetensors`: the scheduler's sigmas for 4, 8 and 50 steps.

    python make_moss_reference.py <model-snapshot-dir>

Needs torch, diffusers >= 0.38, transformers >= 5.3 (Qwen3.5), accelerate and
the whole snapshot (`hf download OpenMOSS-Team/MOSS-Image-SFT`; last checked at
revision f972bd4). Writes into the current directory.
"""
import json, sys, time
import numpy as np, torch
from safetensors.torch import load_file, save_file
from diffusers import AutoencoderKLFlux2
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5

SNAP = sys.argv[1]
sys.path.insert(0, SNAP)
import pipeline_moss_image as moss  # noqa: E402  (the reference implementation)

dev = "mps"
bf16 = torch.bfloat16


# MPS has no float64, which the transformer's timestep sinusoid is computed
# in. This is the reference forward with that sinusoid on the CPU (still in
# f64, as on CUDA); everything else is unchanged.
def _moss_forward(self, hidden_states, timestep, encoder_hidden_states,
                  encoder_attention_mask=None, return_dict=True, skip_layers=None):
    skip_layers = moss._validate_skip_layers(skip_layers, len(self.layers))
    batch, _, height, width = hidden_states.shape
    mask = encoder_attention_mask
    if mask is None:
        mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
    times = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1).cpu().expand(batch)
    half = self.config.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float().to(hidden_states.device)
    time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
    mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
    head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
    outputs = []
    for index in range(batch):
        context = encoder_hidden_states[index:index + 1, mask[index].bool()]
        video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
        video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
        rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
        text = self.text_embedder(context, rope[:, height * width:])
        x = torch.cat((video, text), 1)
        for layer_index, layer in enumerate(self.layers):
            if layer_index not in skip_layers:
                x = layer(x, rope, mod[index:index + 1])
        x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
    sample = torch.cat(outputs)
    return moss.Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


moss.MOSSImageTransformer.forward = _moss_forward
pipe = moss.MOSSImagePipeline(
    transformer=moss.MOSSImageTransformer.from_pretrained(SNAP, subfolder="transformer", torch_dtype=bf16).eval(),
    scheduler=moss.MOSSImageScheduler.from_pretrained(SNAP, subfolder="scheduler"),
    text_encoder=AutoModel.from_pretrained(f"{SNAP}/text_encoder", dtype=bf16).eval(),
    tokenizer=AutoTokenizer.from_pretrained(f"{SNAP}/tokenizer"),
    vae=AutoencoderKLFlux2.from_pretrained(SNAP, subfolder="vae", torch_dtype=torch.float32).eval(),
).to(dev)
# As the model card does: rebuild the scheduler from its own config.
pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)


def f32(t):
    return t.detach().to("cpu", torch.float32).contiguous()


PROMPTS = {
    "fox": "A red fox sitting in fresh snow, wildlife photography",
    "long": (
        "A wide view of a cozy bookstore window on a rainy evening. Warm light spills onto the wet "
        "street, and a ginger cat sleeps on a stack of old hardcover books. A hand-painted sign in "
        'the window reads in Chinese: "今日新书到店，欢迎进来避雨。" Below it, a smaller chalkboard '
        'says in English: "Open late - hot tea inside." Reflections of passing umbrellas shimmer on '
        "the glass, a bicycle leans against the lamppost outside, and a few fallen maple leaves stick "
        "to the pavement. Soft film grain, shallow depth of field, cinematic color grading."
    ),
    "empty": "",
}


def record_text_encoder():
    rec = {}
    layers = pipe.text_encoder.language_model.layers
    kernel = modeling_qwen3_5.torch_chunk_gated_delta_rule
    for name, prompt in PROMPTS.items():
        outs = {}
        handles = [
            layer.register_forward_hook(lambda m, a, o, i=i: outs.__setitem__(i, o[0] if isinstance(o, tuple) else o))
            for i, layer in enumerate(layers)
        ]
        kernel_calls = []

        def kernel_hook(q, k, v, g, beta, **kw):
            out = kernel(q, k, v, g, beta, **kw)
            kernel_calls.append((q, k, v, g, beta, out[0], kw.get("use_qk_l2norm_in_kernel")))
            return out

        modeling_qwen3_5.torch_chunk_gated_delta_rule = kernel_hook
        formatted = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        ids = pipe.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).input_ids
        try:
            embeds, mask = pipe.encode_prompt(prompt)
        finally:
            modeling_qwen3_5.torch_chunk_gated_delta_rule = kernel
            for h in handles:
                h.remove()
        assert mask.all() and embeds.shape[1] == ids.shape[1]
        rec[f"te_{name}_ids"] = f32(ids[0].to(torch.float32))
        rec[f"te_{name}_out"] = f32(embeds[0])
        for i, o in sorted(outs.items()):
            rec[f"te_{name}_layer_{i}"] = f32(o[0])
        if name == "long":
            q, k, v, g, beta, out, l2 = kernel_calls[0]
            assert l2 is True and len(kernel_calls) == 24, len(kernel_calls)
            for key, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("out", out)):
                rec[f"gdn_{key}"] = f32(t[0])
        print(f"text encoder: {name!r}: {ids.shape[1]} tokens", flush=True)
    save_file(rec, "moss_te.safetensors")


def record_schedules():
    rec = {}
    for steps in (4, 8, 50):
        s = type(pipe.scheduler).from_config(pipe.scheduler.config)
        s.set_timesteps(steps)
        rec[f"sigmas_{steps}"] = f32(s.sigmas)
        rec[f"timesteps_{steps}"] = f32(s.timesteps)
        print(f"sigmas for {steps} steps ({len(s.timesteps)} kept), f32 bits:",
              s.sigmas.to(torch.float32).view(torch.int32).tolist(), flush=True)
    save_file(rec, "moss_sched.safetensors")


def record_t2i(run_name, latents, **call):
    rec, calls = {}, []
    tr_fwd = pipe.transformer.forward

    def tr_hook(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None, **kw):
        out = tr_fwd(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, **kw)
        step, branch = len(calls) // 2, ("cond", "uncond")[len(calls) % 2]
        calls.append(float(torch.as_tensor(timestep).float().item()))
        assert bool(kw.get("skip_layers")) == (branch == "uncond")
        rec[f"tr_{branch}_x_{step}"] = f32(hidden_states)
        rec[f"tr_{branch}_out_{step}"] = f32(out.sample)
        if step == 0:
            rec[f"tr_{branch}_txt"] = f32(encoder_hidden_states[0])
        return out

    sched_step = pipe.scheduler.step

    def step_hook(*a, **kw):
        out = sched_step(*a, **kw)
        rec["latents_final"] = f32(out[0])
        return out

    vae_dec = pipe.vae.decode

    def dec_hook(z, *a, **kw):
        out = vae_dec(z, *a, **kw)
        rec["vae_dec_in"] = f32(z)
        rec["vae_dec_out"] = f32(out[0] if isinstance(out, tuple) else out.sample)
        return out

    pipe.transformer.forward = tr_hook
    pipe.scheduler.step = step_hook
    pipe.vae.decode = dec_hook
    try:
        t0 = time.time()
        img = pipe(latents=latents, output_type="pil", **call).images[0]
        print(f"{run_name}: {time.time() - t0:.1f}s, {len(calls)} transformer calls", flush=True)
    finally:
        pipe.transformer.forward = tr_fwd
        pipe.scheduler.step = sched_step
        pipe.vae.decode = vae_dec

    # VAE encode of the decoded image, normalized like the transformer's latents:
    # the mean, patchified 2x2 (channel c*4 + dy*2 + dx), then BatchNorm stats.
    x = rec["vae_dec_out"].clamp(-1, 1).to(dev)
    mean = pipe.vae.encode(x).latent_dist.mean
    b, c, h, w = mean.shape
    patched = mean.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    bn, eps = pipe.vae.bn, pipe.vae.config.batch_norm_eps
    normed = (patched - bn.running_mean.view(1, -1, 1, 1)) / (bn.running_var.view(1, -1, 1, 1) + eps).sqrt()
    rec["vae_enc_in"] = f32(x)
    rec["vae_enc_mean"] = f32(mean)
    rec["vae_enc_latents"] = f32(normed)

    rec["sched_sigmas"] = f32(pipe.scheduler.sigmas)
    meta = {"timesteps": calls, "call": {k: v for k, v in call.items() if isinstance(v, (int, float, str, list))}}
    img.save(f"{run_name}.png")
    save_file(rec, f"{run_name}.safetensors", metadata={"meta": json.dumps(meta)})
    print(run_name, json.dumps(meta), flush=True)


def noise(seed, h, w):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 128, h // 16, w // 16)).astype(np.float32)
    return torch.from_numpy(x).to(dev)


def record_t2i_f32(run_name, latents):
    """The same 8 steps with the transformer in f32, from the bf16 run's
    prompt embeddings: bf16 rounding, amplified by guidance over large steps,
    moves the final latents of two correct bf16 runs apart; in f32 a port
    should agree closely."""
    bf16_run = load_file("moss_t2i.safetensors")
    pipe.transformer.to(torch.float32)
    try:
        out = pipe(
            prompt_embeds=bf16_run["tr_cond_txt"][None].to(dev),
            negative_prompt_embeds=bf16_run["tr_uncond_txt"][None].to(dev),
            latents=latents, height=512, width=512, num_inference_steps=8, output_type="latent",
        ).images
    finally:
        pipe.transformer.to(bf16)
    save_file({"latents_final": f32(out)}, f"{run_name}.safetensors")
    print(f"{run_name}: done", flush=True)


with torch.no_grad():
    record_text_encoder()
    record_schedules()
    lat = noise(1, 512, 512)
    save_file({"latents": f32(lat)}, "moss_t2i_noise.safetensors")
    record_t2i("moss_t2i", lat, prompt=PROMPTS["fox"], height=512, width=512, num_inference_steps=8)
    record_t2i_f32("moss_t2i_f32", lat)

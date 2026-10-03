---
name: "mixel"
description: "Generate and edit images with mixel, the user's local Z-Image-Turbo / Qwen-Image-2.1 generator on Apple Silicon: write and validate JSONL batch files, pick settings for the Mac's memory (`--quantize`), add LoRAs, run or hand off the command, fix flaws with fast reference-image edits, and keep characters consistent across illustrated stories."
---

# mixel

mixel is a command-line image generator the user wrote (`~/code/mixel`, installed as `mixel`). It runs on their Mac only. The usual job is: write a JSONL batch file with one image per line, validate it, and either run it (if you have a shell on their Mac) or hand them the file plus the exact command.

**mixel is under active development, and this skill can lag behind it.** The version number doesn't always change when the CLI does. The authoritative reference is `mixel --help` (it ends with a USAGE GUIDE written for agents) and `~/code/mixel/README.md`. If you have a shell on the user's Mac, run `mixel --help` before relying on anything below; if it disagrees with this skill, follow `--help` and tell the user which part of the skill is stale.

This version was written from mixel 0.5.0 (commit `bb7c156`, 2026-10-03). `mixel --version` tells you what's installed; builds before 0.5.0 lack some of what follows (`--quantize` came in 0.4.0, `--lora` for Z-Image-Turbo and the weight cache in 0.5.0, `--lora` for the Qwen models after 0.5.0).

## Models

| `--model` (alias) | Use for | Steps | 1024×1024 image | Edit with one ~1024² reference | Peak memory |
|---|---|---|---|---|---|
| `z-image-turbo` (`turbo`, default) | text-to-image, img2img, LoRAs | 9 (default) | ~59 s | — (can't take references) | ~13.5 GB |
| `qwen-image-2.1-fast` (`qwen-fast`) | **editing with reference images**; also t2i, img2img | 4 (fixed) | ~38 s | ~58 s | ~17–18 GB |
| `qwen-image-2.1` (`qwen`) | editing when the fast variant's detail isn't enough; also t2i, img2img | 40 (default) | ~6 min | ~8 min | ~17–18 GB |

Times are medians on an M3 Max (30-core GPU, 96 GB) with the weights cached; peak memory is at 1024×1024 in bf16, in GB of 2^30 bytes (less with `--quantize`, below). 512×512 with Z-Image takes about 14 s, which is good for drafts.

- The model is chosen once per run with `--model`; it can't vary per JSONL line. Put Z-Image lines and Qwen edit lines in separate files.
- **For edits, start with `qwen-image-2.1-fast`.** It's 7–10× faster than `qwen-image-2.1`. Its authors note small dense text can lose legibility and some edits come out slightly blurrier and darker; switch to `qwen-image-2.1` only when that matters.
- `qwen-image-2.1-fast` always runs 4 steps without guidance: a `num_steps` other than 4, or `guidance_scale` above 1 with a negative prompt, is rejected.
- Both Qwen variants' weights (and the 4-step adapter's) are research/evaluation only, not commercial use. Say so if the user's project sounds commercial.
- Z-Image-Turbo can't take `reference_images` (it fails before loading). It is text-only apart from `init_image`.
- Z-Image-Turbo reads at most 512 tokens (a few hundred words) and ignores the rest; the run prints a "Token count" line that says when a prompt was cut. Long character descriptions repeated in every scene can hit this.

## Guidance

Guidance is off by default for all three models, as they're meant to run; turning it on doubles the time per step. Usually leave `guidance_scale` and `negative_prompt` out.

- `z-image-turbo`: on for any `guidance_scale` above 0, steering away from `negative_prompt` (or from the empty prompt without one). **A negative prompt with guidance off is rejected.** Older batch files used e.g. `5` with a negative prompt; the equivalent now is `4`.
- `qwen-image-2.1`: on above 1, and needs a negative prompt.
- `qwen-image-2.1-fast`: no guidance.

## Memory: `--quantize` and smaller Macs

Peak memory at 1024×1024, by setting:

| | bf16 (default) | `--quantize 8` | `--quantize 4` |
|---|---:|---:|---:|
| `z-image-turbo` | 13.5 GB | 8.3 GB | 5.4 GB |
| Qwen models, text-to-image | 17 GB | 10.9 GB | 7.5 GB |
| Qwen models, edit with a ~1024² reference | 18.3 GB | 12.2 GB | 9 GB |

- The GPU may only use part of the Mac's memory (~74% on a 16 GB Mac, 81% on a 96 GB one); a run that needs more doesn't fail but runs 2–3× slower. Check the memory with `sysctl -n hw.memsize` and pick: **16 GB**: `--quantize 8` for z-image-turbo, `--quantize 4` for the Qwen models (on an M4 Mac mini: ~155 s per 1024² turbo image, ~90 s per 4-step Qwen image, ~125 s per edit). **8 GB** (untested): z-image-turbo with `--quantize 4`, borderline at 1024² (5.4 GB), safer at 512² (4.6 GB). **32 GB and up**: everything in bf16.
- **8 bits gives practically the same images as bf16. 4 bits gives images as good but not the same ones**: a seed reproduces a different picture. Keep one setting for a whole project, or a rerun or edit won't match the earlier images. Quantized steps are ~10–25% slower on an M3 Max; on an M4, 8 and 4 bits run at the same speed.
- The first `--quantize` run saves the quantized weights to `~/.cache/mixel/weights` (6 GB for z-image-turbo at 4 bits, 8.7 GB for the Qwen models; more at 8 bits), printing `Cache: saved …`, and is slower. Later runs load them in seconds (on a 16 GB Mac mini 1–5 s instead of ~17 s). Tell the user about the disk space; `rm -rf ~/.cache/mixel` frees it, `--no-cache` skips it.
- `--quantize` is per run, like `--model`: not a JSONL field.

## LoRAs

`--lora FILE[:SCALE]` adds a LoRA trained for the model: Z-Image-Turbo, or Qwen-Image-2.1 for both Qwen variants (a `.safetensors` file, e.g. from Hugging Face). It applies at its trained strength by default; `:0.7` weakens it, and `--lora` can repeat (effects add up). It works with `--quantize` too.

- It applies to every image of the run (not a JSONL field): images that need different LoRAs go in separate runs.
- Most LoRAs need their trigger phrase from the model card in the prompt (e.g. "Pixel art style."); without it the effect is often small.
- The file is checked before the model loads: text-encoder LoRAs, DoRA and kohya's `lora_unet_…` names are rejected with the reason.
- Qwen-Image-2.1 LoRAs made for the 40-step model also work with `qwen-image-2.1-fast` (edit LoRAs such as doodle-in, which turns a magenta scribble into a named object, or Natural-Exposure). Don't pass the 4-step adapter itself as `--lora` (mixel says to use `qwen-image-2.1-fast`), and treat speed-up LoRAs for other step counts (Turbo8, Viggle) as untested.
- Check the LoRA's license on its model card before commercial use.

## JSONL format

One JSON object per line. Unknown keys are rejected, so use exactly these:

| Field | Type | Notes |
|---|---|---|
| `prompt` | string | **Required**, non-empty. |
| `id` | string or number | Names the output `<id>.png` when `output` is absent. No `/` or `\`. |
| `output` | string | Path under `--output-dir`; must end in `.png` or `.jpg`. Subfolders allowed (`scenes/03.png`). |
| `negative_prompt` | string | Only with guidance on (see above). |
| `width`, `height` | int | Multiples of 16 (turbo) or 32 (qwen models). Default 1024, or derived from an input image. |
| `num_steps` | int | Not `steps`. Omit to get the model default; must be 4 (or omitted) for `qwen-image-2.1-fast`. |
| `guidance_scale` | float | Rarely needed; omit. |
| `seed` | int | Fixed seed ⇒ predictable filename and reproducible image. |
| `init_image` | string | img2img start image, path **relative to the JSONL file**. |
| `strength` | float in (0, 1] | Only with `init_image`; default 0.6. Fraction of steps that run. |
| `reference_images` | list of strings | Qwen models only. Paths relative to the JSONL file. |

Common mistakes seen before: `steps` instead of `num_steps`; `size` instead of `width`/`height`; a `strength` on a line without `init_image`; two lines resolving to the same output (case-insensitive: `Fox` and `fox` collide); a `negative_prompt` on a turbo line without guidance.

Output naming: `output` → else `<id>.png` → else the 4-digit line number (`0003.png`). Without a seed (line or `--seed`), a random one is appended: `fox-123456.png`. Always give either per-line seeds or a `--seed` so filenames are known and reruns are reproducible.

## Running

```bash
mixel --input story.jsonl --output-dir out/story
```

- Batch mode validates every line before loading the model and lists all errors at once with line numbers. Fix and rerun.
- A run loads in two phases: the text encoders (`Loaded in …`), which encode every prompt, then the transformer and VAE (a second `Loaded in …`); only then do the images start.
- Existing outputs are skipped, so an interrupted batch resumes by rerunning the same command. `--overwrite` regenerates. To redo one image, delete that file and rerun.
- CLI flags (`--seed`, `--width`, `--num-steps`, `--negative-prompt`, …) act as defaults for lines that omit them.
- Exit 0 on success; non-zero if any line failed (the rest still generate). Each saved image prints `Done! Image saved to <path> (<secs>s)`; the end prints `Batch finished: N generated, N skipped, N failed`.
- The first run downloads ~33 GB (turbo) or ~31 GB (qwen; the fast variant adds 0.35 GB) to `~/.cache/huggingface`.
- **Run one mixel at a time** (memory). Before starting one, check none is already running (`pgrep -fl '^mixel'`). An 18-image 896×1152 turbo batch takes about 20 minutes on an M3 Max (two to three times that on a 16 GB Mac mini), so if you run it yourself, run it in the background and wait for it to finish rather than polling.

Single images: `mixel --prompt "..." --seed 1 --output fox.png` (always pass `--prompt`; without it a default landscape prompt is used). `--input` can't be combined with `--prompt` or `--output`.

Single edit:

```bash
mixel --model qwen-fast --ref-image scene.png \
      --prompt "Remove the second axe. Keep everything else exactly the same. No text." \
      --seed 1 --output scene_fixed.png
```

Repeat `--ref-image` for several images ("the cat from image 1 on the sofa from image 2"). Without `--width`/`--height` the output takes the last image's aspect ratio at about 1024×1024 px.

## Where to run it

- **Shell on the user's Mac available** (device bash, Claude Code in the repo): write the JSONL next to where the images should go, validate, run it, then show the results.
- **No shell on the Mac** (a cloud sandbox, claude.ai chat): you can't run mixel. Write and validate the JSONL, deliver it as a file, and give the exact command with the output dir. Don't pretend to generate images.

Always validate before handing over or running. This checks what mixel checks up front, minus file existence and CLI-flag defaults:

```python
import json, sys
ALLOWED = {"prompt","id","negative_prompt","width","height","num_steps","guidance_scale",
           "seed","output","init_image","strength","reference_images"}
MODELS = {"turbo":"z-image-turbo","qwen":"qwen-image-2.1","qwen-fast":"qwen-image-2.1-fast"}
def validate(path, model="z-image-turbo"):
    model = MODELS.get(model, model)
    qwen = model.startswith("qwen"); fast = model == "qwen-image-2.1-fast"
    align = 32 if qwen else 16
    seen, errs = {}, []
    for n, line in enumerate(open(path), 1):
        if not line.strip(): continue
        try: d = json.loads(line)
        except Exception as e: errs.append(f"line {n}: bad JSON: {e}"); continue
        if bad := set(d) - ALLOWED: errs.append(f"line {n}: unknown fields {sorted(bad)}")
        if not str(d.get("prompt","")).strip(): errs.append(f"line {n}: empty prompt")
        for k in ("width","height"):
            if k in d and d[k] % align: errs.append(f"line {n}: {k} {d[k]} not a multiple of {align}")
        if "strength" in d and "init_image" not in d: errs.append(f"line {n}: strength without init_image")
        if "strength" in d and not 0 < d["strength"] <= 1: errs.append(f"line {n}: strength out of (0,1]")
        if "reference_images" in d and not qwen: errs.append(f"line {n}: reference_images need a qwen model")
        if fast and d.get("num_steps", 4) != 4: errs.append(f"line {n}: qwen-image-2.1-fast always runs 4 steps")
        g, neg = d.get("guidance_scale"), d.get("negative_prompt")
        if not qwen and neg and not (g or 0) > 0: errs.append(f"line {n}: negative_prompt needs guidance_scale > 0 on turbo (unless --guidance-scale is set)")
        if fast and neg and (g or 0) > 1: errs.append(f"line {n}: qwen-image-2.1-fast has no guidance")
        if "id" in d and any(c in str(d["id"]) for c in "/\\"): errs.append(f"line {n}: id has a path separator")
        out = d.get("output") or (f"{d['id']}.png" if "id" in d else f"{n:04d}.png")
        if not out.lower().endswith((".png",".jpg",".jpeg")): errs.append(f"line {n}: output must end in .png/.jpg")
        if (k := out.lower()) in seen: errs.append(f"line {n}: output {out} also used by line {seen[k]}")
        seen[out.lower()] = n
    print("\n".join(errs) or f"OK: {len(seen)} images"); return not errs
validate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "z-image-turbo")
```

## Illustrated stories and consistent characters

This is the most common request (e.g. a fairy tale as a watercolor picture book). What works:

1. **Agree a shot list first.** Before generating, write a short text list, one line per page: who's in it, where (indoors or outdoors), what's happening, which props. Have the user confirm it. A composition choice ("the wolf tucked under the covers, like most illustrations") is far cheaper to settle in text than after a round of images.
2. **Character bible.** One fixed, specific description per character (age, hair colour and style, face details, a distinctive outfit with colours), pasted identically into every scene where they appear. Build prompts in a small script from a dict of characters plus a shared style block, so the wording can't drift. Restate costume details that matter (e.g. "silver-white hair pinned in a bun, nothing on her head").
3. **Shared style block** appended to every prompt (medium, palette, lighting, era). Z-Image likes to add lettering and signatures, so end with "No text, no lettering, no signature."
4. **Prompt against Z-Image's known failure modes:**
   - Extra copies of a character: say "exactly one wolf", "only one goat and one troll in the picture".
   - Interiors turning into outdoors (grass, weeds, flowers on the floor): say "indoor scene … wooden plank floorboards … no grass, no plants, no sky".
   - Animals standing or gesturing like people: say "natural four-legged animal".
   - Giants drawn barely taller than the hero: state the scale in two ways ("as tall as a house", "the boy is no bigger than the giant's hand"), and still expect only 2–3×.
   - A character facing the wrong way for the action (e.g. escaping but facing into the room): say "seen from behind, his back to the room".
5. **Generate 2–3 seeds per scene on the first pass** (about a minute each) and pick the best. A single seed per scene usually means a second round.
6. **Reference sheets first**: one per main character, full-body front, side and back views plus a few facial expressions, on a plain background; landscape 1152×896 fits them. A size lineup sheet helps when characters differ in scale.
7. **Portrait book pages**: 896×1152 (multiples of 16 and 32). Landscape spreads: 1152×896.
8. Keep each scene to one or two main figures when possible; crowded scenes drift most.
9. Omit `num_steps` and `guidance_scale` unless there's a reason; Z-Image's defaults are tuned.
10. **Fix small flaws by editing, not regenerating.** A wrong prop, a stray figure, weeds on a floor, a character facing the wrong way: run a `qwen-image-2.1-fast` edit with the image as `--ref-image` and a prompt like "Remove X. Keep everything else, the characters, the room and the watercolor style exactly the same. No text." Regenerating tends to trade one flaw for another. Note that the JSONL then reproduces the pre-edit image.

Stronger consistency, at a cost: after the turbo batch, run a second Qwen JSONL where each scene has `"reference_images": ["out/story/00_character_sheet.png"]` and a prompt like "The boy from the image, now climbing a giant beanstalk…". With `qwen-image-2.1-fast` that's about a minute per image. Offer it; don't default to it.

img2img tips (any model): changing *what* is in the picture keeps pose and framing at strength 0.6–0.8. Changing a photo's *style* needs `num_steps` ~20 and strength ~0.95 (not on the fast variant, which is fixed at 4 steps).

## Reviewing the results

- Review every image at full size against the shot list: character count, setting, props and how many, costume continuity, no text or signature. Small contact sheets are fine for an overview but hide details like a second axe or a headscarf.
- **If an image fails to load when you look at it (for example "media removed"), that image has not been reviewed.** Say so and look again later; never describe or approve an image you haven't seen.
- Show the user the finished set before wiring images into a site, document or PDF.

## Deliverable

- The `.jsonl` file (named after the project, e.g. `jack_and_the_beanstalk.jsonl`), validated.
- The exact run command, with the `--output-dir` (and `--model`, `--quantize`, `--lora` if used: they're per run, and change the images).
- A short list of the scenes/ids, so the user can map files to the story.
- If you ran it: the output folder, how long it took, any failed lines, and which images were edited afterwards (so a rerun won't reproduce them).

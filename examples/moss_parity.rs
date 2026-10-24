//! Checks each stage of the MOSS-Image-SFT port against reference tensors
//! recorded from the pipeline bundled with the weights (bf16 on MPS).
//!
//! ```bash
//! cargo run --release --example moss_parity -- <ref-dir>
//! ```
//!
//! `<ref-dir>` holds the files written by `scripts/make_moss_reference.py`.
//! Each stage gets the reference's own inputs, so an error points at that
//! stage. bf16 noise is around 1e-2; a bug is ~1.

use anyhow::{Error as E, Result};
use mixel::moss::{
    gated_delta, prompt, scheduler, text_encoder, transformer, vae, DEFAULT_GUIDANCE, REPO,
    UNCOND_SKIP_LAYER,
};
use mixel::nn::ModelFiles;
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::Path;

/// The prompts `make_moss_reference.py` encodes, by name.
const PROMPTS: [(&str, &str); 3] = [
    ("fox", "A red fox sitting in fresh snow, wildlife photography"),
    (
        "long",
        "A wide view of a cozy bookstore window on a rainy evening. Warm light spills onto the wet \
         street, and a ginger cat sleeps on a stack of old hardcover books. A hand-painted sign in \
         the window reads in Chinese: \"今日新书到店，欢迎进来避雨。\" Below it, a smaller chalkboard \
         says in English: \"Open late - hot tea inside.\" Reflections of passing umbrellas shimmer on \
         the glass, a bicycle leans against the lamppost outside, and a few fallen maple leaves stick \
         to the pavement. Soft film grain, shallow depth of field, cinematic color grading.",
    ),
    ("empty", ""),
];

fn load(dir: &Path, name: &str) -> Result<HashMap<String, Array>> {
    Ok(Array::load_safetensors(dir.join(name))?)
}

fn get<'a>(m: &'a HashMap<String, Array>, name: &str) -> Result<&'a Array> {
    m.get(name)
        .ok_or_else(|| anyhow::anyhow!("reference has no {name}"))
}

fn vec(a: &Array) -> Result<Vec<f32>> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    a.eval()?;
    Ok(a.as_slice::<f32>().to_vec())
}

fn report(name: &str, ours: &Array, reference: &Array) -> Result<f64> {
    report_within(name, ours, reference, 0.05, "")
}

/// Like `report`, for checks where a larger gap is expected for `why`.
fn report_within(
    name: &str,
    ours: &Array,
    reference: &Array,
    expected: f64,
    why: &str,
) -> Result<f64> {
    let (m, c) = (vec(ours)?, vec(reference)?);
    anyhow::ensure!(
        m.len() == c.len(),
        "{name}: {:?} vs {:?}",
        ours.shape(),
        reference.shape()
    );
    let num: f64 = m
        .iter()
        .zip(&c)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let den: f64 = c
        .iter()
        .map(|b| (*b as f64).powi(2))
        .sum::<f64>()
        .max(1e-30);
    let err = (num / den).sqrt();
    let verdict = match err {
        e if e < 0.05 => "ok".to_string(),
        e if e < expected => format!("ok (expected: {why})"),
        _ => "MISMATCH".to_string(),
    };
    println!("  {name:<34} rel L2 {err:.2e}  {verdict}");
    Ok(err)
}

fn ids_of(a: &Array) -> Result<Vec<u32>> {
    Ok(vec(a)?.into_iter().map(|x| x as u32).collect())
}

fn text_encoder_stages(dir: &Path, files: &ModelFiles) -> Result<()> {
    let r = load(dir, "moss_te.safetensors")?;
    println!("Tokenizer");
    let tokenizer =
        tokenizers::Tokenizer::from_file(files.get("tokenizer/tokenizer.json")?).map_err(E::msg)?;
    for (name, text) in PROMPTS {
        let ours = prompt::encode(&tokenizer, text)?.ids;
        let theirs = ids_of(get(&r, &format!("te_{name}_ids"))?)?;
        let verdict = if ours == theirs {
            "identical"
        } else {
            "MISMATCH"
        };
        println!("  {name:<34} {} tokens  {verdict}", theirs.len());
    }

    println!("Gated DeltaNet kernel (layer 0, long prompt)");
    let g = |n: &str| get(&r, &format!("gdn_{n}"));
    let out = gated_delta::chunked(g("q")?, g("k")?, g("v")?, g("g")?, g("beta")?)?;
    report("kernel output", &out.as_dtype(Dtype::Bfloat16)?, g("out")?)?;

    println!("Text encoder (bf16)");
    let shards = (1..=4)
        .map(|i| files.get(&format!("text_encoder/model-{i:05}-of-00004.safetensors")))
        .collect::<Result<Vec<_>>>()?;
    let te = text_encoder::TextEncoder::load(&shards, Dtype::Bfloat16, None, None)?;
    for (name, _) in PROMPTS {
        // The reference's ids, so a tokenizer difference can't hide here.
        let ids = ids_of(get(&r, &format!("te_{name}_ids"))?)?;
        if name == "long" {
            for (i, h) in te.layer_outputs(&ids)?.iter().enumerate() {
                report(
                    &format!("{name}: layer {i}"),
                    &h.squeeze_axes(&[0])?,
                    get(&r, &format!("te_{name}_layer_{i}"))?,
                )?;
            }
        }
        let out = te.forward(&ids)?;
        report(
            &format!("{name}: embeddings"),
            &out.squeeze_axes(&[0])?,
            get(&r, &format!("te_{name}_out"))?,
        )?;
    }
    Ok(())
}

fn scheduler_stages(dir: &Path) -> Result<()> {
    let r = load(dir, "moss_sched.safetensors")?;
    println!("Scheduler");
    for steps in [4, 8, 50] {
        let theirs = vec(get(&r, &format!("sigmas_{steps}"))?)?;
        let ours = scheduler::sigmas(steps);
        let verdict = if ours == theirs {
            "identical"
        } else {
            "MISMATCH"
        };
        println!("  {:<34} {verdict}", format!("sigmas, {steps} steps"));
        let theirs = vec(get(&r, &format!("timesteps_{steps}"))?)?;
        let ours: Vec<f32> = ours[..ours.len() - 1]
            .iter()
            .map(|&s| scheduler::timestep(s))
            .collect();
        let verdict = if ours == theirs {
            "identical"
        } else {
            "MISMATCH"
        };
        println!("  {:<34} {verdict}", format!("timesteps, {steps} steps"));
    }
    Ok(())
}

/// The reference's (1, C, H, W) latents as tokens (1, H * W, C).
fn to_tokens(x: &Array) -> Result<Array> {
    let s = x.shape();
    Ok(x.reshape(&[s[0], s[1], s[2] * s[3]])?
        .transpose_axes(&[0, 2, 1])?)
}

/// Latent grid (H, W) of the 512x512 run.
const GRID: (i32, i32) = (32, 32);

fn load_transformer(files: &ModelFiles, dtype: Dtype) -> Result<transformer::Transformer> {
    let shards = (1..=3)
        .map(|i| {
            files.get(&format!(
                "transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors"
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    transformer::Transformer::load(&shards, dtype, None, None)
}

/// The 8-step guided loop from the reference's noise, like the pipeline: f32
/// latents and guidance, the negative branch skipping its block. Calls
/// `before_step` with each step's input.
fn guided_loop(
    dir: &Path,
    tr: &transformer::Transformer,
    r: &HashMap<String, Array>,
    mut before_step: impl FnMut(usize, &Array) -> Result<()>,
) -> Result<Array> {
    let (h, w) = GRID;
    let cond = tr.condition(&get(r, "tr_cond_txt")?.expand_dims(0)?, h, w)?;
    let uncond = tr.condition(&get(r, "tr_uncond_txt")?.expand_dims(0)?, h, w)?;
    let sigmas = scheduler::sigmas(8);
    let noise = load(dir, "moss_t2i_noise.safetensors")?;
    let mut x = to_tokens(get(&noise, "latents")?)?;
    let scale = Array::from_f32(DEFAULT_GUIDANCE);
    for step in 0..sigmas.len() - 1 {
        before_step(step, &x)?;
        let t = scheduler::timestep(sigmas[step]);
        let c = tr.forward(&x, t, &cond, None)?.as_dtype(Dtype::Float32)?;
        let u = tr
            .forward(&x, t, &uncond, Some(UNCOND_SKIP_LAYER))?
            .as_dtype(Dtype::Float32)?;
        let v = u.add(scale.multiply(c.subtract(&u)?)?)?;
        x = x.add(v.multiply(Array::from_f32(sigmas[step + 1] - sigmas[step]))?)?;
        x.eval()?;
    }
    Ok(x)
}

fn transformer_stages(dir: &Path, files: &ModelFiles) -> Result<()> {
    let r = load(dir, "moss_t2i.safetensors")?;
    println!("Transformer (bf16)");
    let tr = load_transformer(files, Dtype::Bfloat16)?;
    let (h, w) = GRID;
    let cond = tr.condition(&get(&r, "tr_cond_txt")?.expand_dims(0)?, h, w)?;
    let uncond = tr.condition(&get(&r, "tr_uncond_txt")?.expand_dims(0)?, h, w)?;
    let sigmas = scheduler::sigmas(8);
    let skip = Some(UNCOND_SKIP_LAYER);
    for step in [0, 4, 7] {
        let t = scheduler::timestep(sigmas[step]);
        for (branch, c, skip) in [("cond", &cond, None), ("uncond", &uncond, skip)] {
            let x = to_tokens(get(&r, &format!("tr_{branch}_x_{step}"))?)?;
            let out = tr.forward(&x, t, c, skip)?;
            report(
                &format!("step {step}, {branch}"),
                &out,
                &to_tokens(get(&r, &format!("tr_{branch}_out_{step}"))?)?,
            )?;
        }
    }
    let why = "guidance amplifies bf16 rounding";
    let x = guided_loop(dir, &tr, &r, |step, x| {
        let reference = to_tokens(get(&r, &format!("tr_cond_x_{step}"))?)?;
        report_within(
            &format!("loop: latents before step {step}"),
            x,
            &reference,
            0.5,
            why,
        )?;
        Ok(())
    })?;
    report_within(
        "loop: final latents",
        &x,
        &to_tokens(get(&r, "latents_final")?)?,
        0.5,
        why,
    )?;
    drop(tr);

    // In f32 the same loop has no rounding to amplify.
    println!("Transformer (f32), 8-step loop");
    let tr = load_transformer(files, Dtype::Float32)?;
    let x = guided_loop(dir, &tr, &r, |_, _| Ok(()))?;
    let f = load(dir, "moss_t2i_f32.safetensors")?;
    report("final latents", &x, &to_tokens(get(&f, "latents_final")?)?)?;
    Ok(())
}

/// The reference's NCHW images as NHWC.
fn to_nhwc(x: &Array) -> Result<Array> {
    Ok(x.transpose_axes(&[0, 2, 3, 1])?)
}

fn vae_stages(dir: &Path, files: &ModelFiles) -> Result<()> {
    let r = load(dir, "moss_t2i.safetensors")?;
    println!("VAE (f32)");
    let vae = vae::Vae::load(files.get("vae/diffusion_pytorch_model.safetensors")?)?;
    let (h, w) = GRID;
    let decoded = vae.decode(&to_tokens(get(&r, "latents_final")?)?, h, w)?;
    report("decode", &decoded, &to_nhwc(get(&r, "vae_dec_out")?)?)?;
    let encoded = vae.encode(&to_nhwc(get(&r, "vae_enc_in")?)?)?;
    report(
        "encode (normalized latents)",
        &encoded,
        &to_tokens(get(&r, "vae_enc_latents")?)?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: moss_parity <ref-dir>"))?;
    let dir = Path::new(&dir);
    let files = ModelFiles::new(REPO, None)?;
    text_encoder_stages(dir, &files)?;
    scheduler_stages(dir)?;
    transformer_stages(dir, &files)?;
    vae_stages(dir, &files)?;
    Ok(())
}

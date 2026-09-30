//! Checks Z-Image-Turbo sampling (schedule, time input, f32 latents) against
//! diffusers' `ZImagePipeline`.
//!
//! ```bash
//! cargo run --release --example zimage_diffusers_parity -- <ref-dir>
//! ```
//!
//! `<ref-dir>` holds `zimage.safetensors`, `zimage_cfg.safetensors` (guidance
//! 3 against the default empty negative prompt) and `zimage_noise.safetensors`,
//! written by `scripts/make_zimage_reference.py`. Each step first gets the
//! reference's own input, so an error points at that step; then the whole
//! 9-step loop runs from the reference noise, and the images are saved next to
//! the reference's as `ours_zimage*.png`. bf16 noise is around 1e-2 per step
//! and compounds to ~7e-2 over the loop; a bug is more: candle's schedule and
//! missing pad tokens left the image 34% off, the old guidance formula 27%.

use anyhow::{Error as E, Result};
use mixel::nn::ModelFiles;
use mixel::zimage::pipeline::{format_prompt_for_qwen3, guide};
use mixel::zimage::{
    scheduler::Scheduler, text_encoder::TextEncoder, transformer::Transformer, vae::Vae,
};
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::Path;

fn vec(a: &Array) -> Result<Vec<f32>> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    a.eval()?;
    Ok(a.as_slice::<f32>().to_vec())
}

fn rel_l2(ours: &Array, reference: &Array) -> Result<f64> {
    let (m, c) = (vec(ours)?, vec(reference)?);
    anyhow::ensure!(
        m.len() == c.len(),
        "{:?} vs {:?}",
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
    Ok((num / den).sqrt())
}

fn report(name: &str, ours: &Array, reference: &Array) -> Result<()> {
    report_within(name, ours, reference, 0.05)
}

/// For the full loop, where ~1% of bf16 noise per step compounds over 9 steps
/// (more with guidance, which scales the difference of two predictions).
fn report_loop(name: &str, ours: &Array, reference: &Array) -> Result<()> {
    report_within(name, ours, reference, 0.12)
}

fn report_within(name: &str, ours: &Array, reference: &Array, expected: f64) -> Result<()> {
    let err = rel_l2(ours, reference)?;
    let verdict = match err {
        e if e < 0.05 => "ok",
        e if e < expected => "ok (expected: compounded bf16 noise)",
        _ => "MISMATCH",
    };
    println!("  {name:<40} rel L2 {err:.2e}  {verdict}");
    Ok(())
}

/// (1, H, W, 3) in [-1, 1] -> PNG.
fn save_png(x: &Array, path: &Path) -> Result<()> {
    let (h, w) = (x.shape()[1] as u32, x.shape()[2] as u32);
    let px: Vec<u8> = vec(x)?
        .iter()
        .map(|v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    image::RgbImage::from_raw(w, h, px)
        .ok_or_else(|| anyhow::anyhow!("image size"))?
        .save(path)?;
    Ok(())
}

/// Each step from the reference's own input (both passes with guidance), then
/// the full loop from `noise`; returns the final latents (1, 16, 64, 64), f32.
fn run(
    name: &str,
    tr: &Transformer,
    rec: &HashMap<String, Array>,
    noise: &Array,
    guidance: Option<(&Array, f32)>,
) -> Result<Array> {
    let bf16 = Dtype::Bfloat16;
    let steps = 9;
    let cap = rec["tr_cap"].expand_dims(0)?.as_dtype(bf16)?;
    // Reference tensors are (16, 1, H, W); the model takes (1, 16, H, W).
    let nchw = |a: &Array| a.reshape(&[1, 16, 64, 64]);
    let mut s = Scheduler::new(steps);
    let ts: Vec<f32> = (0..steps)
        .map(|_| {
            let t = s.current_timestep_normalized();
            s.step_dt();
            t
        })
        .collect();

    println!("{name}: transformer, each step from the reference's input");
    let ref_neg_cap = match guidance {
        Some(_) => Some(rec["tr_cap_neg"].expand_dims(0)?.as_dtype(bf16)?),
        None => None,
    };
    for (i, &t) in ts.iter().enumerate() {
        let x = nchw(&rec[&format!("tr_x_{i}")])?.as_dtype(bf16)?;
        let out = tr.forward(&x, t, &cap)?;
        let label = format!("step {} (t = {t:.3})", i + 1);
        report(&label, &out, &nchw(&rec[&format!("tr_out_{i}")])?)?;
        if let Some(neg_cap) = &ref_neg_cap {
            let out = tr.forward(&x, t, neg_cap)?;
            let reference = nchw(&rec[&format!("tr_out_neg_{i}")])?;
            report(&format!("{label}, negative"), &out, &reference)?;
        }
    }

    println!("{name}: full 9-step loop from the reference noise (f32 latents)");
    let mut s = Scheduler::new(steps);
    let mut x = noise.clone();
    for i in 0..steps {
        if i > 0 {
            report_loop(
                &format!("latents before step {}", i + 1),
                &x,
                &nchw(&rec[&format!("tr_x_{i}")])?,
            )?;
        }
        let t = s.current_timestep_normalized();
        let xb = x.as_dtype(bf16)?;
        let mut v = tr.forward(&xb, t, &cap)?.as_dtype(Dtype::Float32)?;
        if let Some((neg_cap, scale)) = guidance {
            let neg = tr.forward(&xb, t, neg_cap)?.as_dtype(Dtype::Float32)?;
            v = guide(&v, &neg, scale)?;
        }
        x = x.add(v.negative()?.multiply(Array::from_f32(s.step_dt()))?)?;
        x.eval()?;
    }
    Ok(x)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let bf16 = Dtype::Bfloat16;
    let load = |name: &str| Array::load_safetensors(dir.join(name));
    let rec = load("zimage.safetensors")?;
    let cfg = load("zimage_cfg.safetensors")?;
    let noise = &load("zimage_noise.safetensors")?["latents"];
    let steps = 9;

    println!("Scheduler");
    let mut s = Scheduler::new(steps);
    let mut sigmas = Vec::new();
    let mut ts = Vec::new();
    for _ in 0..steps {
        sigmas.push(s.current_sigma());
        ts.push(s.current_timestep_normalized());
        s.step_dt();
    }
    sigmas.push(0.0);
    report(
        "sigmas (9 steps)",
        &Array::from_slice(&sigmas, &[steps as i32 + 1]),
        &rec["sched_sigmas"],
    )?;
    let ref_t: Vec<f32> = (0..steps)
        .map(|i| vec(&rec[&format!("tr_t_{i}")]).map(|v| v[0]))
        .collect::<Result<_>>()?;
    report(
        "model time inputs",
        &Array::from_slice(&ts, &[steps as i32]),
        &Array::from_slice(&ref_t, &[steps as i32]),
    )?;

    // The default negative prompt: the empty prompt, as mixel encodes it.
    println!("Text encoder");
    let files = ModelFiles::new(mixel::DEFAULT_REPO, None)?;
    let empty = {
        let tok = tokenizers::Tokenizer::from_file(files.get("tokenizer/tokenizer.json")?)
            .map_err(E::msg)?;
        let ids = tok
            .encode(format_prompt_for_qwen3("").as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let te = TextEncoder::load(
            &(1..=3)
                .map(|i| files.get(&format!("text_encoder/model-{i:05}-of-00003.safetensors")))
                .collect::<Result<Vec<_>>>()?,
            bf16,
        )?;
        let feats = te.forward(&ids)?;
        feats.eval()?;
        feats
    };
    report(
        "empty negative prompt embeds",
        &empty,
        &cfg["tr_cap_neg"].expand_dims(0)?,
    )?;

    let tr = Transformer::load(
        &(1..=3)
            .map(|i| {
                files.get(&format!(
                    "transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors"
                ))
            })
            .collect::<Result<Vec<_>>>()?,
        bf16,
    )?;
    let x_plain = run("No guidance", &tr, &rec, noise, None)?;
    let x_cfg = run("Guidance 3", &tr, &cfg, noise, Some((&empty, 3.0)))?;
    drop(tr);

    println!("Images");
    let vae = Vae::load(files.get("vae/diffusion_pytorch_model.safetensors")?, bf16)?;
    for (name, x, rec, file) in [
        ("no guidance", x_plain, &rec, "ours_zimage.png"),
        ("guidance 3", x_cfg, &cfg, "ours_zimage_cfg.png"),
    ] {
        let ours = vae.decode(&x.as_dtype(bf16)?.transpose_axes(&[0, 2, 3, 1])?)?;
        let reference = rec["vae_dec_out"].transpose_axes(&[0, 2, 3, 1])?;
        report_loop(&format!("{name}: image vs reference"), &ours, &reference)?;
        save_png(&ours, &dir.join(file))?;
    }
    Ok(())
}

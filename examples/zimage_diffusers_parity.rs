//! Checks Z-Image-Turbo sampling (schedule, time input, f32 latents) against
//! diffusers' `ZImagePipeline`.
//!
//! ```bash
//! cargo run --release --example zimage_diffusers_parity -- <ref-dir>
//! ```
//!
//! `<ref-dir>` holds `zimage.safetensors` and `zimage_noise.safetensors`,
//! written by `scripts/make_zimage_reference.py`. Each step first gets the
//! reference's own input, so an error points at that step; then the whole
//! 9-step loop runs from the reference noise, and the image is saved next to
//! the reference's as `ours_zimage.png`. bf16 noise is around 1e-2; a bug
//! (such as the wrong schedule) is ~1e-1 or more.

use anyhow::Result;
use mixel::nn::ModelFiles;
use mixel::zimage::{scheduler::Scheduler, transformer::Transformer, vae::Vae};
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
    let err = rel_l2(ours, reference)?;
    let verdict = if err < 0.05 { "ok" } else { "MISMATCH" };
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let bf16 = Dtype::Bfloat16;
    let rec: HashMap<String, Array> = Array::load_safetensors(dir.join("zimage.safetensors"))?;
    let noise = &Array::load_safetensors(dir.join("zimage_noise.safetensors"))?["latents"];
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

    println!("Transformer, each step from the reference's input");
    let files = ModelFiles::new(mixel::DEFAULT_REPO, None)?;
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
    let cap = rec["tr_cap"].expand_dims(0)?.as_dtype(bf16)?;
    // Reference tensors are (16, 1, H, W); the model takes (1, 16, H, W).
    let nchw = |a: &Array| a.reshape(&[1, 16, 64, 64]);
    for i in 0..steps {
        let x = nchw(&rec[&format!("tr_x_{i}")])?.as_dtype(bf16)?;
        let out = tr.forward(&x, ts[i], &cap)?;
        report(
            &format!("step {} (t = {:.3})", i + 1, ts[i]),
            &out,
            &nchw(&rec[&format!("tr_out_{i}")])?,
        )?;
    }

    println!("Full 9-step loop from the reference noise (f32 latents)");
    let mut s = Scheduler::new(steps);
    let mut x = noise.clone();
    for i in 0..steps {
        if i > 0 {
            report(
                &format!("latents before step {}", i + 1),
                &x,
                &nchw(&rec[&format!("tr_x_{i}")])?,
            )?;
        }
        let t = s.current_timestep_normalized();
        let v = tr
            .forward(&x.as_dtype(bf16)?, t, &cap)?
            .as_dtype(Dtype::Float32)?;
        x = x.add(v.negative()?.multiply(Array::from_f32(s.step_dt()))?)?;
        x.eval()?;
    }
    drop(tr);

    let vae = Vae::load(files.get("vae/diffusion_pytorch_model.safetensors")?, bf16)?;
    let ours = vae.decode(&x.as_dtype(bf16)?.transpose_axes(&[0, 2, 3, 1])?)?;
    let reference = rec["vae_dec_out"].transpose_axes(&[0, 2, 3, 1])?;
    report("image vs reference", &ours, &reference)?;
    save_png(&ours, &dir.join("ours_zimage.png"))?;
    Ok(())
}

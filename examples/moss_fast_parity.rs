//! Checks MOSS-Image in 4 steps (the rapid-moss adapter) against reference
//! tensors recorded with the adapter's own code on the reference pipeline
//! (bf16 on MPS).
//!
//! ```bash
//! cargo run --release --example moss_fast_parity -- <ref-dir>
//! ```
//!
//! `<ref-dir>` holds `moss_fast.safetensors` and `moss_fast_noise.safetensors`
//! from `scripts/make_moss_fast_reference.py`. Each step first gets the
//! reference's own input, then the whole sampler runs from the noise, in bf16
//! and then in f32.
//!
//! Expected: the f32 sampler "ok" (its first step within ~1e-4). In bf16 the
//! adapted model amplifies rounding: the reference's own bf16 run is ~11% off
//! its f32 run after one step and ~28% after four, so mixel's bf16 run is
//! "expected" to be that far from the reference's too. (mixel computes the
//! low-rank updates in the model dtype, the reference's `inference.py` in
//! f32; in f32 both agree.)

use anyhow::Result;
use mixel::moss::{fast, transformer, vae, REPO};
use mixel::nn::ModelFiles;
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::Path;

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
    report_within(name, ours, reference, 0.05)
}

/// Like `report`, for the bf16 stages (see the module docs).
fn report_within(name: &str, ours: &Array, reference: &Array, expected: f64) -> Result<f64> {
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
        e if e < 0.05 => "ok",
        e if e < expected => "ok (expected: bf16 rounding, amplified)",
        _ => "MISMATCH",
    };
    println!("  {name:<34} rel L2 {err:.2e}  {verdict}");
    Ok(err)
}

/// The reference's (1, C, H, W) latents as tokens (1, H * W, C).
fn to_tokens(x: &Array) -> Result<Array> {
    let s = x.shape();
    Ok(x.reshape(&[s[0], s[1], s[2] * s[3]])?
        .transpose_axes(&[0, 2, 1])?)
}

/// Latent grid (H, W) of the 512x512 run.
const GRID: (i32, i32) = (32, 32);

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: moss_fast_parity <ref-dir>"))?;
    let dir = Path::new(&dir);
    let r = Array::load_safetensors(dir.join("moss_fast.safetensors"))?;
    let noise = Array::load_safetensors(dir.join("moss_fast_noise.safetensors"))?;
    let files = ModelFiles::new(REPO, None)?;

    println!("Adapter");
    let adapter = fast::Adapter::load(&ModelFiles::at_revision(fast::REPO, fast::REVISION)?)?;
    println!("  step codes {:?}", adapter.step_codes);

    println!("Reference: its own bf16 run against its f32 run");
    report_within("step 0", get(&r, "v_0")?, get(&r, "f32_v_0")?, 0.35)?;
    report_within("final x0", get(&r, "x0_3")?, get(&r, "f32_x0")?, 0.35)?;

    println!("Transformer (bf16) with the adapter");
    let shards = (1..=3)
        .map(|i| {
            files.get(&format!(
                "transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors"
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut tr = transformer::Transformer::load(&shards, Dtype::Bfloat16, None, None)?;
    tr.apply_adapter(&adapter)?;
    let (h, w) = GRID;
    let cond = tr.condition(&get(&r, "txt")?.expand_dims(0)?, h, w)?;
    for (step, &code) in adapter.step_codes.iter().enumerate() {
        let x = to_tokens(get(&r, &format!("x_{step}"))?)?;
        let v = tr.forward(&x, code, &cond, None)?;
        report_within(
            &format!("step {step}, from its input"),
            &v,
            &to_tokens(get(&r, &format!("v_{step}"))?)?,
            0.15,
        )?;
    }

    // The whole sampler from the reference's noise.
    let z = to_tokens(get(&noise, "latents")?)?;
    let mut x0 = Array::zeros::<f32>(z.shape())?;
    for (step, &code) in adapter.step_codes.iter().enumerate() {
        let x = z.add(x0.matmul(adapter.q.t())?)?;
        report(
            &format!("sampler: input of step {step}"),
            &x,
            &to_tokens(get(&r, &format!("x_{step}"))?)?,
        )?;
        x0 = z.subtract(
            tr.forward(&x, code, &cond, None)?
                .as_dtype(Dtype::Float32)?,
        )?;
        x0.eval()?;
    }
    report_within(
        "sampler: final x0",
        &x0,
        &to_tokens(get(&r, &format!("x0_{}", fast::STEPS - 1))?)?,
        0.35,
    )?;
    drop(tr);

    // In f32 there is no rounding for the steps to amplify.
    println!("Transformer (f32) with the adapter");
    let mut tr = transformer::Transformer::load(&shards, Dtype::Float32, None, None)?;
    tr.apply_adapter(&adapter)?;
    let cond32 = tr.condition(&get(&r, "txt")?.expand_dims(0)?, h, w)?;
    let mut x0_32 = Array::zeros::<f32>(z.shape())?;
    for (step, &code) in adapter.step_codes.iter().enumerate() {
        let x = z.add(x0_32.matmul(adapter.q.t())?)?;
        let v = tr.forward(&x, code, &cond32, None)?;
        report(
            &format!("sampler: step {step}"),
            &v,
            &to_tokens(get(&r, &format!("f32_v_{step}"))?)?,
        )?;
        x0_32 = z.subtract(&v)?;
        x0_32.eval()?;
    }
    report("sampler: final x0", &x0_32, &to_tokens(get(&r, "f32_x0")?)?)?;
    drop(tr);

    println!("VAE (f32)");
    let vae = vae::Vae::load(files.get("vae/diffusion_pytorch_model.safetensors")?)?;
    let image = mlx_rs::ops::clip(&vae.decode(&x0, h, w)?, (-1.0f32, 1.0f32))?;
    report_within(
        "image (from the bf16 sampler)",
        &image,
        &get(&r, "image")?.transpose_axes(&[0, 2, 3, 1])?,
        0.35,
    )?;
    Ok(())
}

//! Checks the 4-step Qwen-Image-2.1 variant (Fun-Acc adapter) against
//! reference tensors from the adapter authors' own PyTorch code.
//!
//! ```bash
//! cargo run --release --example qwen21_fast_parity -- <ref-dir>
//! ```
//!
//! `<ref-dir>` holds `fast_t2i.safetensors`, `fast_edit.safetensors` and their
//! noise files, written by `scripts/make_qwen21_fast_reference.py`. Each step
//! first gets the reference's own input, so an error points at that step; then
//! the whole 4-step loop runs from the reference noise, and the images are
//! saved next to the reference's as `ours_fast_*.png`. bf16 noise is around
//! 1e-2; a bug is ~1.
//!
//! Expected: every step "ok"; the text-to-image loop drifts to ~1e-1 by the
//! end. Its four large steps amplify any input difference ~10x each (the
//! "same model" rows measure this with no implementation difference at all),
//! so the ~1% gap between two bf16 implementations grows. Edits, anchored by
//! the reference image, stay at ~1e-2.

use anyhow::Result;
use mixel::nn::ModelFiles;
use mixel::qwen21::{fast, transformer, vae, REPO};
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::Path;

fn load(dir: &Path, name: &str) -> Result<HashMap<String, Array>> {
    Ok(Array::load_safetensors(dir.join(name))?)
}

fn vec(a: &Array) -> Result<Vec<f32>> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    a.eval()?;
    Ok(a.as_slice::<f32>().to_vec())
}

fn rel_l2(name: &str, ours: &Array, reference: &Array) -> Result<f64> {
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
    Ok((num / den).sqrt())
}

fn report(name: &str, ours: &Array, reference: &Array) -> Result<()> {
    report_within(name, ours, reference, 0.05)
}

/// Like `report`, where the t2i loop's amplification makes a larger gap
/// expected (see the module docs).
fn report_within(name: &str, ours: &Array, reference: &Array, expected: f64) -> Result<()> {
    let err = rel_l2(name, ours, reference)?;
    let verdict = match err {
        e if e < 0.05 => "ok",
        e if e < expected => "ok (expected: amplified bf16 noise)",
        _ => "MISMATCH",
    };
    println!("  {name:<40} rel L2 {err:.2e}  {verdict}");
    Ok(())
}

/// Last `n` tokens along axis 1.
fn last_tokens(x: &Array, n: i32) -> Result<Array> {
    let len = x.shape()[1];
    Ok(mlx_rs::ops::split_at_indices(x, &[len - n], 1)?
        .pop()
        .unwrap())
}

/// Prefix segments from the reference `img_mask` (image slots stand for 2x2
/// latent tokens); `grids` are the reference images' latent grids.
fn segments(
    mask: &Array,
    target_slots: usize,
    grids: &[(usize, usize)],
) -> Result<Vec<transformer::Segment>> {
    let m = vec(mask)?;
    let prefix = &m[..m.len() - target_slots];
    let mut segs = Vec::new();
    let mut grids = grids.iter();
    let mut i = 0;
    while i < prefix.len() {
        let img = prefix[i] > 0.5;
        let mut j = i;
        while j < prefix.len() && (prefix[j] > 0.5) == img {
            j += 1;
        }
        segs.push(if img {
            let &(h, w) = grids.next().unwrap();
            transformer::Segment::Image { h, w }
        } else {
            transformer::Segment::Text(j - i)
        });
        i = j;
    }
    Ok(segs)
}

/// Like the pipeline: t = bf16(sigma * 1000), then / 1000 in bf16.
fn timestep(sigma: f32) -> Result<f32> {
    let bf16 = Dtype::Bfloat16;
    let t = Array::from_f32(sigma * 1000.0)
        .as_dtype(bf16)?
        .divide(Array::from_f32(1000.0).as_dtype(bf16)?)?;
    Ok(vec(&t)?[0])
}

/// Per-step checks from the reference inputs, then the full loop from
/// `noise`; returns the final latents (1, 1024, 64) in f32.
fn run(
    name: &str,
    tr: &transformer::Transformer,
    rec: &HashMap<String, Array>,
    noise: &Array,
    sigmas: &[f32],
    refs: &[(usize, usize)],
) -> Result<Array> {
    let bf16 = Dtype::Bfloat16;
    // The reference's input is [reference-image latents | target latents].
    let ref_latents: Vec<Array> = match refs {
        [] => vec![],
        _ => vec![
            mlx_rs::ops::split_at_indices(&rec["tr_hidden_in_0"], &[1024], 1)?[0].as_dtype(bf16)?,
        ],
    };
    let segs = segments(&rec["tr_img_mask"], 256, refs)?;
    let cache = tr.prefill(&rec["tr_encoder_in"].as_dtype(bf16)?, &ref_latents, &segs)?;
    let ref_input = |i: usize| -> Result<Array> {
        Ok(last_tokens(&rec[&format!("tr_hidden_in_{i}")], 1024)?.as_dtype(bf16)?)
    };
    let mut ref_outputs = Vec::new();
    for i in 0..fast::STEPS {
        let out = tr.forward(&ref_input(i)?, timestep(sigmas[i])?, 32, 32, &cache, i)?;
        report(
            &format!("{name} step {} (reference input)", i + 1),
            &out,
            &last_tokens(&rec[&format!("tr_out_{i}")], 1024)?,
        )?;
        ref_outputs.push(out);
    }
    // The full loop: latents stay in f32, the model reads bf16.
    let mut x = noise.as_dtype(bf16)?.as_dtype(Dtype::Float32)?;
    for i in 0..fast::STEPS {
        let v = tr.forward(&x.as_dtype(bf16)?, timestep(sigmas[i])?, 32, 32, &cache, i)?;
        if i > 0 {
            // The same model's output for the loop's latents vs the
            // reference's: how much this step amplifies the gap above.
            let gap_in = rel_l2(name, &x, &ref_input(i)?)?;
            let gap_out = rel_l2(name, &v, &ref_outputs[i])?;
            println!(
                "  {:<40} rel L2 {gap_in:.2e}  (same model: output gap {gap_out:.2e}, {:.0}x)",
                format!("{name} loop latents before step {}", i + 1),
                gap_out / gap_in
            );
        }
        x = x.add(
            v.as_dtype(Dtype::Float32)?
                .multiply(Array::from_f32(sigmas[i + 1] - sigmas[i]))?,
        )?;
        x.eval()?;
    }
    Ok(x)
}

/// (1, H, W, 4) in [-1, 1] -> PNG (RGB channels).
fn save_png(x: &Array, path: &Path) -> Result<()> {
    let (h, w) = (x.shape()[1] as u32, x.shape()[2] as u32);
    let px: Vec<u8> = vec(x)?
        .chunks(4)
        .flat_map(|p| p[..3].to_vec())
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
    let files = ModelFiles::new(REPO, None)?;
    let t2i = load(dir, "fast_t2i.safetensors")?;
    let edit = load(dir, "fast_edit.safetensors")?;

    println!("Adapter");
    let adapter = fast::Adapter::load(&ModelFiles::new(fast::REPO, None)?, bf16)?;
    let sigmas = adapter.sigmas.clone();
    report(
        "sigmas (fixed 4-step schedule)",
        &Array::from_slice(&sigmas, &[sigmas.len() as i32]),
        &t2i["sched_sigmas"],
    )?;

    println!("Transformer with the adapter");
    let mut tr = transformer::Transformer::load(
        &(1..=2)
            .map(|i| {
                files.get(&format!(
                    "transformer/diffusion_pytorch_model-{i:05}-of-00002.safetensors"
                ))
            })
            .collect::<Result<Vec<_>>>()?,
        bf16,
        None,
        None,
    )?;
    tr.apply_adapter(adapter)?;
    let noise = load(dir, "fast_t2i_noise.safetensors")?;
    let x_t2i = run("t2i", &tr, &t2i, &noise["latents"], &sigmas, &[])?;
    let noise = load(dir, "fast_edit_noise.safetensors")?;
    let x_edit = run("edit", &tr, &edit, &noise["latents"], &sigmas, &[(32, 32)])?;
    drop(tr);

    println!("Images (full 4-step loop from the reference noise)");
    let v = vae::Vae::load(files.get("vae/diffusion_pytorch_model.safetensors")?, bf16)?;
    for (name, x, rec, limit) in [("t2i", x_t2i, &t2i, 0.15), ("edit", x_edit, &edit, 0.05)] {
        let ours = v.decode(&x.as_dtype(bf16)?.reshape(&[1, 32, 32, 64])?)?;
        let reference = rec["vae_dec_out"]
            .reshape(&[1, 4, 512, 512])?
            .transpose_axes(&[0, 2, 3, 1])?;
        report_within(
            &format!("{name} image vs reference"),
            &ours,
            &reference,
            limit,
        )?;
        save_png(&ours, &dir.join(format!("ours_fast_{name}.png")))?;
    }
    Ok(())
}

//! Checks each stage of the Qwen-Image-2.1 port against reference tensors
//! recorded from the diffusers/transformers pipeline (bf16 on MPS).
//!
//! ```bash
//! cargo run --release --example qwen21_parity -- <ref-dir> <reference-image.png>
//! ```
//!
//! `<ref-dir>` holds `t2i.safetensors`, `edit.safetensors` and the noise files
//! written by `scripts/make_qwen21_reference.py` (run with the same image).
//! Each stage gets the reference's own inputs, so an error points at that
//! stage. bf16 noise is around 1e-2; a bug is ~1.
//!
//! Expected: everything "ok" except the vision encoder and the edit prompt
//! embeddings computed from it, ~6% and ~28%. mixel runs the vision encoder in
//! f32, while the reference's bf16 vision output is itself ~6% off f32 (and the
//! text encoder amplifies that). Against an f32 reference mixel's embeddings
//! agree to ~1e-5.

use anyhow::{Error as E, Result};
use mixel::nn::ModelFiles;
use mixel::qwen21::{prompt, scheduler, text_encoder, transformer, vae, vision, REPO};
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

fn report(name: &str, ours: &Array, reference: &Array) -> Result<f64> {
    report_within(name, ours, reference, 0.05)
}

/// Like `report`, for checks where the reference's own bf16 rounding makes a
/// larger gap expected (see the module docs).
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
        e if e < expected => "ok (expected: bf16 reference vision)",
        _ => "MISMATCH",
    };
    println!("  {name:<34} rel L2 {err:.2e}  {verdict}");
    Ok(err)
}

/// Last `n` tokens along axis 1.
fn last_tokens(x: &Array, n: i32) -> Result<Array> {
    let len = x.shape()[1];
    Ok(mlx_rs::ops::split_at_indices(x, &[len - n], 1)?
        .pop()
        .unwrap())
}

/// Segments of the prefix from the reference `img_mask` over the VLM sequence
/// (image slots stand for 2x2 latent tokens) and each image's latent grid.
fn segments_from_mask(
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
        if img {
            let &(h, w) = grids.next().unwrap();
            anyhow::ensure!((j - i) * 4 == h * w, "slot run {} vs grid {h}x{w}", j - i);
            segs.push(transformer::Segment::Image { h, w });
        } else {
            segs.push(transformer::Segment::Text(j - i));
        }
        i = j;
    }
    Ok(segs)
}

fn ids_of(a: &Array) -> Result<Vec<u32>> {
    Ok(vec(a)?.iter().map(|&v| v as u32).collect())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let ref_image = image::open(&args[2])?.to_rgb8();
    let files = ModelFiles::new(REPO, None)?;
    let bf16 = Dtype::Bfloat16;
    let t2i = load(dir, "t2i.safetensors")?;
    let edit = load(dir, "edit.safetensors")?;
    let tok =
        tokenizers::Tokenizer::from_file(files.get("processor/tokenizer.json")?).map_err(E::msg)?;
    let pad_id = prompt::image_pad_id(&tok)?;

    println!("Scheduler");
    let s = scheduler::sigmas(8, 1024);
    report(
        "sigmas (8 steps, 1024 tokens)",
        &Array::from_slice(&s, &[9]),
        &t2i["sched_sigmas"],
    )?;

    println!("Prompt / tokenizer");
    let p_t2i = prompt::encode(
        &tok,
        "A red fox sitting in fresh snow, wildlife photography",
        &[],
    )?;
    let ok = p_t2i.ids == ids_of(&t2i["proc_input_ids"])?;
    println!(
        "  t2i token ids                      {}",
        if ok { "identical" } else { "MISMATCH" }
    );
    let p_edit = prompt::encode(&tok, "Turn the fox into a gray wolf", &[(16, 16)])?;
    let ok = p_edit.ids == ids_of(&edit["proc_input_ids"])?;
    println!(
        "  edit token ids (256 image tokens)  {}",
        if ok { "identical" } else { "MISMATCH" }
    );
    println!("  drop = {} system tokens", p_t2i.drop);

    println!("Vision encoder");
    // f32, as mixel runs it: in bf16 both the reference and this port land ~6%
    // from an f32 run, which the text encoder amplifies.
    let vis = vision::VisionEncoder::load(
        &(1..=4)
            .map(|i| files.get(&format!("text_encoder/model-{i:05}-of-00004.safetensors")))
            .collect::<Result<Vec<_>>>()?,
        Dtype::Float32,
    )?;
    report(
        "patchify (pixel_values)",
        &vision::patchify(&ref_image)?,
        &edit["vis_pixel_values"],
    )?;
    let feats = vis.encode(&ref_image)?;
    report_within("merged tokens", &feats.merged, &edit["vis_merged"], 0.08)?;
    for (i, d) in feats.deepstack.iter().enumerate() {
        report_within(
            &format!("deepstack {i}"),
            d,
            &edit[&format!("vis_deepstack_{i}")],
            0.08,
        )?;
    }
    drop(vis);

    println!("Text encoder");
    let te = text_encoder::TextEncoder::load(
        &(1..=4)
            .map(|i| files.get(&format!("text_encoder/model-{i:05}-of-00004.safetensors")))
            .collect::<Result<Vec<_>>>()?,
        bf16,
    )?;
    let h = te.forward(&p_t2i.ids, &p_t2i.positions, &[])?;
    report(
        "t2i prompt embeds",
        &last_tokens(&h, (p_t2i.ids.len() - p_t2i.drop) as i32)?,
        &t2i["tr_encoder_in"],
    )?;
    let n_edit = (p_edit.ids.len() - p_edit.drop) as i32;
    let reference_vision = vec![text_encoder::ImageEmbeds {
        start: p_edit.image_starts[0],
        merged: edit["vis_merged"].as_dtype(bf16)?,
        deepstack: (0..3)
            .map(|i| edit[&format!("vis_deepstack_{i}")].as_dtype(bf16))
            .collect::<Result<_, _>>()?,
    }];
    let h = te.forward(&p_edit.ids, &p_edit.positions, &reference_vision)?;
    report_within(
        "edit embeds (reference vision in)",
        &last_tokens(&h, n_edit)?,
        &edit["tr_encoder_in"],
        0.08,
    )?;
    let images = vec![text_encoder::ImageEmbeds {
        start: p_edit.image_starts[0],
        merged: feats.merged.as_dtype(bf16)?,
        deepstack: feats
            .deepstack
            .iter()
            .map(|d| d.as_dtype(bf16))
            .collect::<Result<_, _>>()?,
    }];
    let h = te.forward(&p_edit.ids, &p_edit.positions, &images)?;
    report_within(
        "edit embeds (f32 vision, see note)",
        &last_tokens(&h, n_edit)?,
        &edit["tr_encoder_in"],
        0.35,
    )?;
    let mask: Vec<f32> = p_edit
        .image_pad_mask(pad_id)
        .iter()
        .map(|&b| b as u8 as f32)
        .collect();
    let ref_mask = vec(&edit["tr_img_mask"])?;
    println!(
        "  edit image-pad mask                {}",
        if mask[..] == ref_mask[..mask.len()] {
            "identical"
        } else {
            "MISMATCH"
        }
    );
    drop(te);

    println!("Transformer");
    let tr = transformer::Transformer::load(
        &(1..=2)
            .map(|i| {
                files.get(&format!(
                    "transformer/diffusion_pytorch_model-{i:05}-of-00002.safetensors"
                ))
            })
            .collect::<Result<Vec<_>>>()?,
        bf16,
    )?;
    let t0 = vec(&t2i["tr_timestep"])?[0];
    let segs = segments_from_mask(&t2i["tr_img_mask"], 256, &[])?;
    let cache = tr.prefill(&t2i["tr_encoder_in"].as_dtype(bf16)?, &[], &segs)?;
    let out = tr.forward(&t2i["tr_hidden_in"].as_dtype(bf16)?, t0, 32, 32, &cache, 0)?;
    report(
        "t2i step 1 (t=1.0)",
        &out,
        &last_tokens(&t2i["tr_out"], 1024)?,
    )?;

    let e_in = edit["tr_hidden_in"].as_dtype(bf16)?;
    let [cond, target]: [Array; 2] = mlx_rs::ops::split_at_indices(&e_in, &[1024], 1)?
        .try_into()
        .unwrap();
    let segs = segments_from_mask(&edit["tr_img_mask"], 256, &[(32, 32)])?;
    println!("  edit segments: {segs:?}");
    let cache = tr.prefill(&edit["tr_encoder_in"].as_dtype(bf16)?, &[cond], &segs)?;
    let out = tr.forward(&target, vec(&edit["tr_timestep"])?[0], 32, 32, &cache, 0)?;
    report(
        "edit step 1 (with reference image)",
        &out,
        &last_tokens(&edit["tr_out"], 1024)?,
    )?;

    // Full 8-step t2i from the reference noise and embeddings.
    let noise = load(dir, "t2i_noise.safetensors")?;
    let mut x = noise["latents"].as_dtype(bf16)?;
    let cache = tr.prefill(
        &t2i["tr_encoder_in"].as_dtype(bf16)?,
        &[],
        &segments_from_mask(&t2i["tr_img_mask"], 256, &[])?,
    )?;
    let sig = scheduler::sigmas(8, 1024);
    for i in 0..8 {
        // Like the pipeline: t = bf16(sigma * 1000), then / 1000 in bf16.
        let t = Array::from_f32(sig[i] * 1000.0)
            .as_dtype(bf16)?
            .divide(Array::from_f32(1000.0).as_dtype(bf16)?)?;
        let t = vec(&t)?[0];
        let v = tr.forward(&x, t, 32, 32, &cache, i)?;
        let dt = sig[i + 1] - sig[i];
        x = x
            .as_dtype(Dtype::Float32)?
            .add(v.as_dtype(Dtype::Float32)?.multiply(Array::from_f32(dt))?)?
            .as_dtype(bf16)?;
        x.eval()?;
    }
    // Reference decoder input is de-normalized, NCHW with a frame axis.
    let ref_z = t2i["vae_dec_in"]
        .reshape(&[1, 64, 32, 32])?
        .transpose_axes(&[0, 2, 3, 1])?;
    drop(tr);

    println!("VAE");
    let v = vae::Vae::load(files.get("vae/diffusion_pytorch_model.safetensors")?, bf16)?;
    let ours_z = x.reshape(&[1, 32, 32, 64])?; // packed (1, h*w, 64) -> NHWC
    let dec_ref = v.decode_denormalized(&ref_z.as_dtype(bf16)?)?;
    report(
        "decode(reference latents)",
        &dec_ref,
        &t2i["vae_dec_out"]
            .reshape(&[1, 4, 512, 512])?
            .transpose_axes(&[0, 2, 3, 1])?,
    )?;
    let dec_ours = v.decode(&ours_z)?;
    report(
        "8-step t2i image vs reference",
        &dec_ours,
        &t2i["vae_dec_out"]
            .reshape(&[1, 4, 512, 512])?
            .transpose_axes(&[0, 2, 3, 1])?,
    )?;
    let enc_in = edit["vae_enc_in"]
        .reshape(&[1, 4, 512, 512])?
        .transpose_axes(&[0, 2, 3, 1])?
        .as_dtype(bf16)?;
    let enc = v.encode(&enc_in)?;
    report(
        "encode(reference image)",
        &enc,
        &edit["vae_enc_out"]
            .reshape(&[1, 64, 32, 32])?
            .transpose_axes(&[0, 2, 3, 1])?,
    )?;
    Ok(())
}

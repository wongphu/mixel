//! How far 8- and 4-bit quantization moves each model's output from bf16.
//!
//! ```bash
//! cargo run --release --example quantize_parity -- [model] [size] [out-dir]
//! ```
//!
//! For Z-Image-Turbo, first each quantized stage on the same input as bf16:
//! the text encoder's caption features and one transformer step. Then, for any
//! model, whole images from the same prompts and seeds, saved to `out-dir` as
//! `<model>-<bf16|q8|q4>-<n>.png`. The relative L2 errors are in the pixels.

use anyhow::{Error as E, Result};
use mixel::nn::{ModelFiles, Quantize};
use mixel::zimage::pipeline::format_prompt_for_qwen3;
use mixel::zimage::{text_encoder::TextEncoder, transformer::Transformer};
use mixel::{GenerateOptions, LoadOptions, Model, Pipeline};
use mlx_rs::{Array, Dtype};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const PROMPTS: [&str; 3] = [
    "a red fox in fresh snow",
    "A neon sign that reads \"OPEN 24 HOURS\" above a rainy street at night",
    "portrait of an old fisherman with a gray beard, natural light",
];

const LEVELS: [(Option<Quantize>, &str); 3] = [
    (None, "bf16"),
    (Some(Quantize::Q8), "q8"),
    (Some(Quantize::Q4), "q4"),
];

fn rel_l2(ours: &Array, reference: &Array) -> Result<f64> {
    let f = |a: &Array| -> Result<Vec<f32>> {
        let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
        a.eval()?;
        Ok(a.as_slice::<f32>().to_vec())
    };
    let (m, c) = (f(ours)?, f(reference)?);
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

/// Each quantized Z-Image stage against bf16, on bf16's input.
fn zimage_stages() -> Result<()> {
    let files = ModelFiles::new(Model::ZImageTurbo.repo(), None)?;
    let tokenizer = Tokenizer::from_file(files.get("tokenizer/tokenizer.json")?).map_err(E::msg)?;
    let te_files = (1..=3)
        .map(|i| files.get(&format!("text_encoder/model-{i:05}-of-00003.safetensors")))
        .collect::<Result<Vec<_>>>()?;
    let tr_files = (1..=3)
        .map(|i| {
            files.get(&format!(
                "transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors"
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let bf16 = Dtype::Bfloat16;

    println!("Z-Image-Turbo stages (relative L2 against bf16, same input):");
    let mut caps = Vec::new();
    for prompt in PROMPTS {
        let ids = tokenizer
            .encode(format_prompt_for_qwen3(prompt).as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        caps.push(ids);
    }
    let mut reference = Vec::new();
    for (q, name) in LEVELS {
        let te = TextEncoder::load(&te_files, bf16, q)?;
        let feats = caps
            .iter()
            .map(|ids| {
                let f = te.forward(ids)?;
                f.eval()?;
                Ok(f)
            })
            .collect::<Result<Vec<_>>>()?;
        if q.is_none() {
            reference = feats;
            continue;
        }
        let errs = feats
            .iter()
            .zip(&reference)
            .map(|(f, r)| rel_l2(f, r))
            .collect::<Result<Vec<_>>>()?;
        println!("  text encoder {name:<5} {}", fmt_errs(&errs));
    }

    // One step at 1024x1024, mid-schedule, on bf16's caption features.
    let key = mlx_rs::random::key(0)?;
    let x =
        mlx_rs::random::normal::<f32>(&[1, 16, 128, 128][..], None, None, &key)?.as_dtype(bf16)?;
    let mut out_ref = Vec::new();
    for (q, name) in LEVELS {
        let tr = Transformer::load(&tr_files, bf16, q)?;
        let outs = reference
            .iter()
            .map(|cap| {
                let o = tr.forward(&x, 0.5, cap)?;
                o.eval()?;
                Ok(o)
            })
            .collect::<Result<Vec<_>>>()?;
        if q.is_none() {
            out_ref = outs;
            continue;
        }
        let errs = outs
            .iter()
            .zip(&out_ref)
            .map(|(o, r)| rel_l2(o, r))
            .collect::<Result<Vec<_>>>()?;
        println!("  transformer  {name:<5} {}", fmt_errs(&errs));
    }
    Ok(())
}

fn fmt_errs(errs: &[f64]) -> String {
    errs.iter()
        .map(|e| format!("{:5.1}%", e * 100.0))
        .collect::<Vec<_>>()
        .join("  ")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = match args.get(1).map(String::as_str) {
        None | Some("z-image-turbo") => Model::ZImageTurbo,
        Some("qwen-image-2.1") => Model::QwenImage21,
        Some("qwen-image-2.1-fast") => Model::QwenImage21Fast,
        Some(other) => anyhow::bail!("unknown model {other}"),
    };
    let size: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1024);
    let out_dir = PathBuf::from(args.get(3).map(String::as_str).unwrap_or("."));
    std::fs::create_dir_all(&out_dir)?;
    // Freed buffers would otherwise stay cached between the loads.
    mlx_rs::memory::set_cache_limit(0)?;

    if model == Model::ZImageTurbo {
        zimage_stages()?;
    }

    println!("{model} images at {size}x{size} (relative L2 of the pixels against bf16):");
    let mut reference = Vec::new();
    for (q, name) in LEVELS {
        let pipeline = Pipeline::load(&LoadOptions {
            model,
            quantize: q,
            ..Default::default()
        })?;
        let mut pixels = Vec::new();
        for (i, prompt) in PROMPTS.iter().enumerate() {
            let opts = GenerateOptions {
                seed: i as u64 + 1,
                width: size,
                height: size,
                ..GenerateOptions::for_model(model, *prompt)
            };
            let image = pipeline.generate(&opts)?.image.to_rgb8();
            image.save(out_dir.join(format!("{model}-{name}-{}.png", i + 1)))?;
            let (w, h) = image.dimensions();
            let data: Vec<f32> = image.as_raw().iter().map(|&p| p as f32).collect();
            pixels.push(Array::from_slice(&data, &[h as i32, w as i32, 3]));
        }
        if q.is_none() {
            reference = pixels;
            continue;
        }
        let errs = pixels
            .iter()
            .zip(&reference)
            .map(|(p, r)| rel_l2(p, r))
            .collect::<Result<Vec<_>>>()?;
        println!("  image {name:<5} {}", fmt_errs(&errs));
    }
    Ok(())
}

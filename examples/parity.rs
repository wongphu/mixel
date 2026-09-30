//! Compares each stage of the mlx-rs port against candle-transformers on
//! identical inputs and reports the relative L2 error ||mlx - candle|| / ||candle||.
//! Both run in bf16, so errors around 1e-2 are rounding noise; a porting bug
//! shows up as errors near 1.
//!
//! ```bash
//! cargo run --release --example parity -- [latent_size]   # default 128 (1024x1024)
//! ```

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image as cz;
use mixel::zimage;
use mlx_rs::{Array, Dtype};

fn c_to_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn m_to_vec(a: &Array) -> Result<Vec<f32>> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    a.eval()?;
    Ok(a.as_slice::<f32>().to_vec())
}

fn rel_err(name: &str, m: &[f32], c: &[f32]) {
    assert_eq!(m.len(), c.len(), "{name}: length mismatch");
    let num: f64 = m.iter().zip(c).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = c.iter().map(|b| (*b as f64).powi(2)).sum();
    let err = (num / den).sqrt();
    let verdict = if err < 0.05 {
        "ok (bf16 noise)"
    } else {
        "MISMATCH"
    };
    println!("  {name:<28} rel L2 error {err:.2e}  {verdict}");
}

fn main() -> Result<()> {
    let latent: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(128);
    let files = zimage::ModelFiles::new(mixel::DEFAULT_REPO, None)?;
    let dev = Device::new_metal(0)?;
    let prompt = "<|im_start|>user\nA cute robot holding a candle in a cozy workshop, digital art<|im_end|>\n<|im_start|>assistant\n";
    let tokenizer =
        tokenizers::Tokenizer::from_file(files.get("tokenizer/tokenizer.json")?).map_err(E::msg)?;
    let ids = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!("latent {latent}x{latent}, {} tokens", ids.len());

    // ---- text encoder
    let te_files: Vec<_> = (1..=3)
        .map(|i| files.get(&format!("text_encoder/model-{i:05}-of-00003.safetensors")))
        .collect::<Result<_>>()?;
    let m_cap =
        zimage::text_encoder::TextEncoder::load(&te_files, Dtype::Bfloat16)?.forward(&ids)?;
    m_cap.eval()?;
    let c_cap = {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&te_files, DType::BF16, &dev)? };
        let te = cz::ZImageTextEncoder::new(&cz::TextEncoderConfig::z_image(), vb)?;
        te.forward(&Tensor::new(ids.as_slice(), &dev)?.unsqueeze(0)?)?
    };
    rel_err("text encoder", &m_to_vec(&m_cap)?, &c_to_vec(&c_cap)?);
    // Feed the same caption features (mlx's) into both transformers.
    let cap_v = m_to_vec(&m_cap)?;
    let c_cap = Tensor::from_vec(cap_v, c_cap.dims(), &dev)?.to_dtype(DType::BF16)?;

    // ---- transformer, one forward pass at t = 0.5 on seeded noise
    let shape = [1usize, 16, latent, latent];
    let noise: Vec<f32> = {
        use rand::SeedableRng;
        use rand_distr::Distribution;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        rand_distr::StandardNormal
            .sample_iter(&mut rng)
            .take(shape.iter().product())
            .collect()
    };
    let tr_files: Vec<_> = (1..=3)
        .map(|i| {
            files.get(&format!(
                "transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors"
            ))
        })
        .collect::<Result<_>>()?;
    let m_out = {
        let tr = zimage::transformer::Transformer::load(&tr_files, Dtype::Bfloat16)?;
        let x = Array::from_slice(&noise, &shape.map(|d| d as i32)).as_dtype(Dtype::Bfloat16)?;
        let out = tr.forward(&x, 0.5, &m_cap)?;
        m_to_vec(&out)?
    };
    let c_out = {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&tr_files, DType::BF16, &dev)? };
        let tr = cz::ZImageTransformer2DModel::new(&cz::Config::z_image_turbo(), vb)?;
        let x = Tensor::from_vec(noise.clone(), &shape[..], &dev)?
            .to_dtype(DType::BF16)?
            .unsqueeze(2)?;
        let t = Tensor::new(&[0.5f32], &dev)?.to_dtype(DType::BF16)?;
        let mask = Tensor::ones((1, ids.len()), DType::U8, &dev)?;
        c_to_vec(&tr.forward(&x, &t, &c_cap, &mask)?.squeeze(2)?)?
    };
    rel_err("transformer (1 step)", &m_out, &c_out);

    // ---- VAE decode of the same latents (mlx NHWC vs candle NCHW)
    let vae_file = files.get("vae/diffusion_pytorch_model.safetensors")?;
    let m_img = {
        let vae = zimage::vae::Vae::load(&vae_file, Dtype::Bfloat16)?;
        let z = Array::from_slice(&noise, &shape.map(|d| d as i32))
            .as_dtype(Dtype::Bfloat16)?
            .transpose_axes(&[0, 2, 3, 1])?;
        m_to_vec(&vae.decode(&z)?)?
    };
    let c_img = {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&vae_file], DType::BF16, &dev)? };
        let vae = cz::AutoEncoderKL::new(&cz::VaeConfig::z_image(), vb)?;
        let z = Tensor::from_vec(noise, &shape[..], &dev)?.to_dtype(DType::BF16)?;
        c_to_vec(&vae.decode(&z)?.permute((0, 2, 3, 1))?.contiguous()?)?
    };
    rel_err("VAE decode", &m_img, &c_img);

    // ---- VAE encoder (img2img) on the decoded image, clamped to [-1, 1]
    let img: Vec<f32> = c_img.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    let (h, w) = (latent as i32 * 8, latent as i32 * 8);
    let m_moments = {
        let vae = zimage::vae::Vae::load(&vae_file, Dtype::Bfloat16)?;
        let x = Array::from_slice(&img, &[1, h, w, 3]).as_dtype(Dtype::Bfloat16)?;
        m_to_vec(&vae.encoder_moments(&x)?)?
    };
    let c_moments = {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&vae_file], DType::BF16, &dev)? };
        let enc = cz::vae::Encoder::new(&cz::VaeConfig::z_image(), vb.pp("encoder"))?;
        let x = Tensor::from_vec(img, (1, h as usize, w as usize, 3), &dev)?
            .permute((0, 3, 1, 2))?
            .contiguous()?
            .to_dtype(DType::BF16)?;
        c_to_vec(&enc.forward(&x)?.permute((0, 2, 3, 1))?.contiguous()?)?
    };
    rel_err("VAE encode", &m_moments, &c_moments);
    Ok(())
}

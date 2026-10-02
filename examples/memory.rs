//! Where the memory goes: MLX's active, cached and peak memory after loading
//! and in each phase of one image. `mixel::Pipeline` turns MLX's buffer cache
//! off; a cache limit in GiB turns it back on after loading.
//!
//! ```bash
//! cargo run --release --example memory -- [model] [size] [bf16|8|4] [cache limit GiB] [output.png]
//! ```

use anyhow::Result;
use mixel::{GenerateOptions, LoadOptions, Model, Pipeline, Progress, Quantize};
use mlx_rs::memory::{active_memory, cache_memory, peak_memory, reset_peak_memory};

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Prints active, cached and peak memory since the last report, then resets
/// the peak.
fn report(phase: &str) {
    println!(
        "{phase:<12} active {:6.2} GiB   cache {:6.2} GiB   peak {:6.2} GiB",
        gib(active_memory().unwrap()),
        gib(cache_memory().unwrap()),
        gib(peak_memory().unwrap())
    );
    reset_peak_memory().unwrap();
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
    let quantize = match args.get(3).map(String::as_str) {
        None | Some("bf16") => None,
        Some("8") => Some(Quantize::Q8),
        Some("4") => Some(Quantize::Q4),
        Some(other) => anyhow::bail!("unknown quantization {other}"),
    };

    let pipeline = Pipeline::load(&LoadOptions {
        model,
        quantize,
        ..Default::default()
    })?;
    report("load");
    // After loading, which turns the cache off.
    if let Some(limit) = args.get(4) {
        let gib: f64 = limit.parse()?;
        mlx_rs::memory::set_cache_limit((gib * (1u64 << 30) as f64) as usize)?;
    }

    let opts = GenerateOptions {
        seed: 1,
        width: size,
        height: size,
        ..GenerateOptions::for_model(model, "a red fox in fresh snow")
    };
    let mut steps_reported = false;
    let out = pipeline.generate_with(&opts, |p| match p {
        Progress::Encoded { .. } => report("tokenized"),
        Progress::Step { step, .. } if step == 1 && !steps_reported => {
            report("text+step 1");
            steps_reported = true;
        }
        Progress::Decoding => report("steps 2..n"),
        _ => {}
    })?;
    report("vae");
    println!("{:?}", out.timings);
    if let Some(path) = args.get(5) {
        out.image.save(path)?;
    }
    Ok(())
}

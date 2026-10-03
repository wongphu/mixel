//! Qwen-Image-2.1 single-stream transformer (`QwenImage21Transformer2DModel`),
//! ported from diffusers.
//!
//! The joint sequence is `[text | reference image | text | ... | target image]`.
//! Attention is block-causal: text is causal, each image block attends to
//! everything before it and to itself. Text and reference-image tokens are
//! modulated with `t = 0` ("causal condition"), so they never depend on the
//! denoising step: [`Transformer::prefill`] runs them once and caches each
//! layer's keys and values, and [`Transformer::forward`] then runs only the
//! target image's tokens per step (the reference pipeline's KV cache).

use super::fast::Adapter;
use crate::nn::{
    gelu_tanh, layer_norm, linear, rms_norm, silu, split_seq, Quantize, WeightCache, Weights,
};
use anyhow::{Context, Result};
use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{concatenate, split_at_indices, tanh};
use mlx_rs::{Array, Dtype};
use std::path::Path;

const DIM: i32 = 4096;
const HEADS: i32 = 32;
const HEAD_DIM: i32 = 128;
const NUM_LAYERS: usize = 32;
const EPS: f32 = 1e-6;
const AXES_DIMS: [usize; 3] = [16, 56, 56];
const ROPE_THETA: f64 = 10000.0;
const FREQ_EMBED: usize = 256;
/// Each vision-language image slot stands for a 2x2 group of latent tokens.
pub const TOKENS_PER_SLOT: usize = 4;

/// One piece of the prefix (everything before the target image).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    /// `n` text tokens (causal).
    Text(usize),
    /// A reference image of `h` x `w` latent tokens (bidirectional).
    Image { h: usize, w: usize },
}

/// Prefix keys and values of every layer, computed once per image.
pub struct KvCache {
    layers: Vec<(Array, Array)>,
    /// Number of prefix tokens (text plus reference-image latents).
    pub prefix_len: usize,
    /// Rotary position the target block starts at (frame axis).
    target_frame: i64,
}

pub struct Transformer {
    w: Weights,
    dtype: Dtype,
    /// With the 4-step adapter: the `proj_out` weight of each step.
    heads: Vec<Array>,
}

impl Transformer {
    /// With `quantize`, the attention and MLP layers of every block are
    /// quantized. The input, time and modulation layers and `proj_out` stay
    /// unquantized.
    pub fn load(
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        quantize: Option<Quantize>,
        cache: Option<&WeightCache>,
    ) -> Result<Self> {
        let blocks = |layer: &str| {
            layer.starts_with("transformer_blocks.")
                && (layer.contains(".attn.") || layer.contains(".img_mlp."))
        };
        Ok(Self {
            w: Weights::load_maybe_cached(
                files,
                dtype,
                |_| true,
                quantize,
                blocks,
                cache,
                "qwen21-transformer",
            )?,
            dtype,
            heads: Vec::new(),
        })
    }

    /// Adds a LoRA's update to the layer it names (see [`lora_targets`]).
    /// With the 4-step adapter, `proj_out` is replaced by per-step heads, so
    /// a LoRA can't change it.
    pub fn add_lora(&mut self, layer: &str, down: Array, up: Array) -> Result<()> {
        anyhow::ensure!(
            !(layer == "proj_out" && !self.heads.is_empty()),
            "a LoRA on proj_out can't be combined with the 4-step adapter, which replaces it"
        );
        for (layer, down, up) in lora_targets(layer, down, up)? {
            self.w.add_lora(&layer, down, up)?;
        }
        Ok(())
    }

    /// Applies the 4-step adapter: adds its low-rank updates, swaps in its
    /// norm weights, and switches `proj_out` to its per-step heads.
    pub fn apply_adapter(&mut self, adapter: Adapter) -> Result<()> {
        for target in &adapter.targets {
            let (down, up) = adapter.lora(target)?;
            self.w.add_lora(target, down, up)?;
        }
        for name in &adapter.full {
            self.w.replace(name, adapter.full(name)?.clone())?;
        }
        let proj = self.w.shape("proj_out.weight")?;
        for head in &adapter.heads {
            anyhow::ensure!(
                head.shape() == proj,
                "adapter head {:?} does not match proj_out {proj:?}",
                head.shape()
            );
        }
        self.heads = adapter.heads;
        Ok(())
    }

    /// Runs the prefix: text embeddings `txt` (1, L_text, 4096) from the text
    /// encoder, with reference-image latents `ref_latents` (each (1, h*w, 64))
    /// substituted into their image slots, laid out as `segments`.
    pub fn prefill(
        &self,
        txt: &Array,
        ref_latents: &[Array],
        segments: &[Segment],
    ) -> Result<KvCache> {
        // Text projection over the vision-language sequence, then expand the
        // image slots to their latent tokens.
        let txt = self.txt_in(txt)?;
        let mut pieces = Vec::new();
        let mut cursor = 0i32;
        let mut refs = ref_latents.iter();
        let mut bounds = Vec::new();
        for seg in segments {
            match *seg {
                Segment::Text(n) => {
                    bounds.push((cursor, cursor + n as i32));
                    cursor += n as i32;
                }
                Segment::Image { h, w } => {
                    // The slots in `txt` are dropped; their tokens come from the VAE latents.
                    cursor += (h * w / TOKENS_PER_SLOT) as i32;
                    bounds.push((-1, -1));
                    let lat = refs
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("missing reference latents"))?;
                    anyhow::ensure!(
                        lat.shape()[1] as usize == h * w,
                        "reference latents do not match {h}x{w}"
                    );
                }
            }
        }
        anyhow::ensure!(
            cursor == txt.shape()[1],
            "segments cover {cursor} of {} text positions",
            txt.shape()[1]
        );
        let mut refs = ref_latents.iter();
        for (seg, &(s, e)) in segments.iter().zip(&bounds) {
            match seg {
                Segment::Text(_) => pieces.push(slice_seq(&txt, s, e)?),
                Segment::Image { .. } => {
                    pieces.push(linear(refs.next().unwrap(), &self.w, "img_in")?)
                }
            }
        }
        let mut x = concatenate(&pieces, 1)?;
        let prefix_len = x.shape()[1] as usize;

        let (positions, target_frame) = prefix_positions(segments);
        let rope = rope_tables(&positions)?;
        // Prefix tokens modulate from t = 0.
        let temb = self.time_embed(0.0)?;
        let modulation = linear(&silu(&temb)?, &self.w, "modulation.1")?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let (out, kv) = self.block_prefix(i, &x, &modulation, &rope, segments)?;
            mlx_rs::transforms::eval([&out, &kv.0, &kv.1])?;
            layers.push(kv);
            x = out;
        }
        Ok(KvCache {
            layers,
            prefix_len,
            target_frame,
        })
    }

    /// Predicts the flow velocity for the target latents `x` (1, h*w, 64) at
    /// normalized time `t`, attending to the cached prefix. `step` is the
    /// index in the sampling schedule; with the 4-step adapter it picks the
    /// output head.
    pub fn forward(
        &self,
        x: &Array,
        t: f32,
        h: usize,
        w: usize,
        cache: &KvCache,
        step: usize,
    ) -> Result<Array> {
        let mut x = linear(x, &self.w, "img_in")?;
        let positions = image_positions(cache.target_frame, h, w);
        let rope = rope_tables(&positions)?;
        let temb = self.time_embed(t)?;
        let modulation = linear(&silu(&temb)?, &self.w, "modulation.1")?;
        for (i, kv) in cache.layers.iter().enumerate() {
            x = self.block_target(i, &x, &modulation, &rope, kv)?;
        }
        let scale = linear(&silu(&temb)?, &self.w, "norm_out.linear")?
            .add(scalar(1.0, self.dtype)?)?
            .expand_dims(1)?;
        let x = layer_norm(&x, EPS)?.multiply(&scale)?;
        if self.heads.is_empty() {
            return linear(&x, &self.w, "proj_out");
        }
        let head = self.heads.get(step).with_context(|| {
            format!(
                "step {step} is past the adapter's {} steps",
                self.heads.len()
            )
        })?;
        Ok(x.matmul(head.t())?)
    }

    /// Zero-centered RMSNorm (scale = weight + 1, in f32), then a 2-layer MLP.
    fn txt_in(&self, x: &Array) -> Result<Array> {
        let x32 = x.as_dtype(Dtype::Float32)?;
        let rrms = x32
            .square()?
            .mean_axis(-1, true)?
            .add(Array::from_f32(EPS))?
            .rsqrt()?;
        let weight = self
            .w
            .get("txt_in.text_norm.weight")?
            .as_dtype(Dtype::Float32)?
            .add(Array::from_f32(1.0))?;
        let normed = x32.multiply(rrms)?.multiply(weight)?.as_dtype(self.dtype)?;
        let h = gelu_tanh(&linear(&normed, &self.w, "txt_in.in_layer")?)?;
        linear(&h, &self.w, "txt_in.out_layer")
    }

    /// Timestep embedding (1, 4096). Like the reference, `t` is rounded to the
    /// model dtype before the sinusoidal projection (computed in f32).
    fn time_embed(&self, t: f32) -> Result<Array> {
        let t = Array::from_f32(t)
            .as_dtype(self.dtype)?
            .as_dtype(Dtype::Float32)?
            .multiply(Array::from_f32(1000.0))?;
        let half = FREQ_EMBED / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32)
            .collect();
        let args = Array::from_slice(&freqs, &[1, half as i32]).multiply(&t)?;
        let emb = concatenate(&[args.cos()?, args.sin()?], -1)?.as_dtype(self.dtype)?;
        let h = silu(&linear(
            &emb,
            &self.w,
            "time_text_embed.timestep_embedder.linear_1",
        )?)?;
        linear(&h, &self.w, "time_text_embed.timestep_embedder.linear_2")
    }

    /// (scale1, gate1, scale2, gate2), each (1, 1, 4096).
    fn mod_params(modulation: &Array) -> Result<[Array; 4]> {
        let parts: [Array; 4] = modulation.split_equal(4, -1)?.try_into().expect("4 parts");
        Ok(parts.map(|p| p.expand_dims(1).expect("expand")))
    }

    fn qkv(&self, p: &str, x: &Array, rope: &(Array, Array)) -> Result<(Array, Array, Array)> {
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let proj = |name: &str| -> Result<Array> {
            Ok(linear(x, &self.w, &format!("{p}.attn.{name}"))?
                .reshape(&[b, s, HEADS, HEAD_DIM])?)
        };
        let q = rms_norm(&proj("to_q")?, &self.w, &format!("{p}.attn.norm_q"), EPS)?;
        let k = rms_norm(&proj("to_k")?, &self.w, &format!("{p}.attn.norm_k"), EPS)?;
        let v = proj("to_v")?;
        let to_bhsd = |a: Array| a.transpose_axes(&[0, 2, 1, 3]);
        Ok((
            to_bhsd(apply_rope(&q, rope, self.dtype)?)?,
            to_bhsd(apply_rope(&k, rope, self.dtype)?)?,
            to_bhsd(v)?,
        ))
    }

    fn attn_out(&self, p: &str, o: Array) -> Result<Array> {
        let (b, s) = (o.shape()[0], o.shape()[2]);
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, DIM])?;
        linear(&o, &self.w, &format!("{p}.attn.to_out.0"))
    }

    fn mlp(&self, p: &str, x: &Array) -> Result<Array> {
        let g = silu(&linear(x, &self.w, &format!("{p}.img_mlp.gate_layer"))?)?;
        let u = linear(x, &self.w, &format!("{p}.img_mlp.proj"))?;
        linear(&g.multiply(u)?, &self.w, &format!("{p}.img_mlp.out"))
    }

    /// One block over the prefix with the block-causal mask; returns the
    /// output and this layer's (k, v).
    fn block_prefix(
        &self,
        i: usize,
        x: &Array,
        modulation: &Array,
        rope: &(Array, Array),
        segments: &[Segment],
    ) -> Result<(Array, (Array, Array))> {
        let p = format!("transformer_blocks.{i}");
        let [s1, g1, s2, g2] = Self::mod_params(modulation)?;
        let one = scalar(1.0, self.dtype)?;
        let h = layer_norm(x, EPS)?.multiply(s1.add(&one)?)?;
        let (q, k, v) = self.qkv(&p, &h, rope)?;

        // Block-causal attention, one call per segment: queries of a segment
        // see all keys up to its end; text segments are causal inside.
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut outs = Vec::new();
        let mut start = 0usize;
        for seg in segments {
            let len = match *seg {
                Segment::Text(n) => n,
                Segment::Image { h, w } => h * w,
            };
            let end = start + len;
            let qs = slice_seq_axis(&q, start as i32, end as i32, 2)?;
            let ks = slice_seq_axis(&k, 0, end as i32, 2)?;
            let vs = slice_seq_axis(&v, 0, end as i32, 2)?;
            let o = match seg {
                Segment::Text(_) if start == 0 => scaled_dot_product_attention(
                    &qs,
                    &ks,
                    &vs,
                    scale,
                    ScaledDotProductAttentionMask::Causal,
                    None,
                )?,
                Segment::Text(_) => {
                    let mask = text_mask(start, len)?;
                    scaled_dot_product_attention(
                        &qs,
                        &ks,
                        &vs,
                        scale,
                        ScaledDotProductAttentionMask::Array(&mask),
                        None,
                    )?
                }
                Segment::Image { .. } => {
                    scaled_dot_product_attention(&qs, &ks, &vs, scale, None, None)?
                }
            };
            outs.push(o);
            start = end;
        }
        let o = concatenate(&outs, 2)?;
        let x = x.add(tanh(&g1)?.multiply(self.attn_out(&p, o)?)?)?;
        let h = layer_norm(&x, EPS)?.multiply(s2.add(&one)?)?;
        let x = x.add(tanh(&g2)?.multiply(self.mlp(&p, &h)?)?)?;
        Ok((x, (k, v)))
    }

    /// One block over the target tokens, attending to the cached prefix.
    fn block_target(
        &self,
        i: usize,
        x: &Array,
        modulation: &Array,
        rope: &(Array, Array),
        kv: &(Array, Array),
    ) -> Result<Array> {
        let p = format!("transformer_blocks.{i}");
        let [s1, g1, s2, g2] = Self::mod_params(modulation)?;
        let one = scalar(1.0, self.dtype)?;
        let h = layer_norm(x, EPS)?.multiply(s1.add(&one)?)?;
        let (q, k, v) = self.qkv(&p, &h, rope)?;
        let k = concatenate(&[&kv.0, &k], 2)?;
        let v = concatenate(&[&kv.1, &v], 2)?;
        let o =
            scaled_dot_product_attention(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), None, None)?;
        let x = x.add(tanh(&g1)?.multiply(self.attn_out(&p, o)?)?)?;
        let h = layer_norm(&x, EPS)?.multiply(s2.add(&one)?)?;
        Ok(x.add(tanh(&g2)?.multiply(self.mlp(&p, &h)?)?)?)
    }
}

fn scalar(v: f32, dtype: Dtype) -> Result<Array> {
    crate::nn::scalar(v, dtype)
}

/// Rows `[s, e)` along axis 1.
fn slice_seq(x: &Array, s: i32, e: i32) -> Result<Array> {
    slice_seq_axis(x, s, e, 1)
}

fn slice_seq_axis(x: &Array, s: i32, e: i32, axis: i32) -> Result<Array> {
    let len = x.shape()[axis as usize];
    let bounds: Vec<i32> = [s, e].into_iter().filter(|&b| b > 0 && b < len).collect();
    let parts = split_seq(x, &bounds, axis)?;
    Ok(parts[if s > 0 { 1 } else { 0 }].clone())
}

/// Keys `[0, start + len)` for a text segment starting at `start`: everything
/// before it, then causal within it. Bool (len, start + len).
fn text_mask(start: usize, len: usize) -> Result<Array> {
    let mut m = Vec::with_capacity(len * (start + len));
    for q in 0..len {
        for k in 0..start + len {
            m.push(k < start || k - start <= q);
        }
    }
    Ok(Array::from_slice(&m, &[len as i32, (start + len) as i32]))
}

/// (frame, height, width) rotary positions of the prefix, and the frame
/// position where the target block starts (`QwenImage21Rope`).
fn prefix_positions(segments: &[Segment]) -> (Vec<[i64; 3]>, i64) {
    let mut out = Vec::new();
    let mut pos = 0i64;
    for seg in segments {
        match *seg {
            Segment::Text(n) => {
                for _ in 0..n {
                    out.push([pos; 3]);
                    pos += 1;
                }
            }
            Segment::Image { h, w } => {
                out.extend(image_positions(pos, h, w));
                pos += h.max(w) as i64;
            }
        }
    }
    (out, pos)
}

/// An image block's positions: the frame axis is fixed at `frame`, height and
/// width form a grid centered on zero.
fn image_positions(frame: i64, h: usize, w: usize) -> Vec<[i64; 3]> {
    let (h, w) = (h as i64, w as i64);
    let mut out = Vec::with_capacity((h * w) as usize);
    for r in -(h - h / 2)..h / 2 {
        for c in -(w - w / 2)..w / 2 {
            out.push([frame, r, c]);
        }
    }
    out
}

/// Per-token rotation angles for the 64 complex pairs: 8 frame, 28 height,
/// 28 width frequencies. Returns (cos, sin), each (1, S, 1, 64) f32.
fn rope_tables(positions: &[[i64; 3]]) -> Result<(Array, Array)> {
    let mut freqs = Vec::new();
    for (axis, &d) in AXES_DIMS.iter().enumerate() {
        for i in (0..d).step_by(2) {
            freqs.push((axis, (1.0 / ROPE_THETA.powf(i as f64 / d as f64)) as f32));
        }
    }
    let s = positions.len();
    let mut cos = Vec::with_capacity(s * freqs.len());
    let mut sin = Vec::with_capacity(s * freqs.len());
    for pos in positions {
        for &(axis, f) in &freqs {
            let (sn, cs) = (pos[axis] as f32 * f).sin_cos();
            cos.push(cs);
            sin.push(sn);
        }
    }
    let shape = [1, s as i32, 1, freqs.len() as i32];
    Ok((
        Array::from_slice(&cos, &shape),
        Array::from_slice(&sin, &shape),
    ))
}

/// Complex-pair (interleaved) rotation of (B, S, H, D) in f32, cast to `dtype`.
fn apply_rope(x: &Array, rope: &(Array, Array), dtype: Dtype) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, s, h, d) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.as_dtype(Dtype::Float32)?.reshape(&[b, s, h, d / 2, 2])?;
    let [re, im]: [Array; 2] = split_at_indices(&x, &[1], -1)?.try_into().expect("2 parts");
    let (re, im) = (re.squeeze_axes(&[-1])?, im.squeeze_axes(&[-1])?);
    let (cos, sin) = rope;
    let out_re = re.multiply(cos)?.subtract(im.multiply(sin)?)?;
    let out_im = re.multiply(sin)?.add(im.multiply(cos)?)?;
    let out = mlx_rs::ops::stack(&[out_re, out_im], -1)?.reshape(&[b, s, h, d])?;
    Ok(out.as_dtype(dtype)?)
}

/// The layers a LoRA update to `layer` goes to. ComfyUI fuses each block's
/// `img_mlp.gate_layer` and `img_mlp.proj` into one `img_mlp.gate_up`, gate
/// first, and LoRAs trained there address it: its update splits into the
/// two, the first half of its output rows for the gate (as ComfyUI's own
/// LoRA loader maps them), both sharing `down`.
pub fn lora_targets(layer: &str, down: Array, up: Array) -> Result<Vec<(String, Array, Array)>> {
    let Some(block) = layer.strip_suffix(".img_mlp.gate_up") else {
        return Ok(vec![(layer.to_string(), down, up)]);
    };
    let rows = up.shape()[0];
    anyhow::ensure!(
        up.ndim() == 2 && rows % 2 == 0,
        "{layer}: update {:?} doesn't split into gate and proj halves",
        up.shape()
    );
    let [gate, proj]: [Array; 2] = split_at_indices(&up, &[rows / 2], 0)?
        .try_into()
        .expect("2 halves");
    Ok(vec![
        (format!("{block}.img_mlp.gate_layer"), down.clone(), gate),
        (format!("{block}.img_mlp.proj"), down, proj),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_gate_up_updates_split_into_gate_and_proj() {
        crate::nn::test_device();
        let down = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        // Rows 0-1 for the gate, 2-3 for proj.
        let up = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4, 1]);
        let t = lora_targets("transformer_blocks.3.img_mlp.gate_up", down, up).unwrap();
        let names: Vec<_> = t.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "transformer_blocks.3.img_mlp.gate_layer",
                "transformer_blocks.3.img_mlp.proj"
            ]
        );
        let rows = |a: &Array| {
            let a = a.contiguous().unwrap();
            a.eval().unwrap();
            a.as_slice::<f32>().to_vec()
        };
        assert_eq!(
            (rows(&t[0].2), rows(&t[1].2)),
            (vec![1.0, 2.0], vec![3.0, 4.0])
        );
        assert_eq!(rows(&t[1].1), [1.0, 2.0]);
        // Other layers go through unchanged.
        let other = Array::from_slice(&[0.0f32; 2], &[2, 1]);
        let t = lora_targets("img_in", Array::from_slice(&[0.0f32; 2], &[1, 2]), other).unwrap();
        assert_eq!(t[0].0, "img_in");
    }

    #[test]
    fn image_positions_are_centered() {
        let p = image_positions(7, 2, 3);
        let hw: Vec<(i64, i64)> = p.iter().map(|q| (q[1], q[2])).collect();
        assert_eq!(
            hw,
            vec![(-1, -2), (-1, -1), (-1, 0), (0, -2), (0, -1), (0, 0)]
        );
        assert!(p.iter().all(|q| q[0] == 7));
    }

    #[test]
    fn prefix_positions_advance_past_images() {
        let segs = [
            Segment::Text(2),
            Segment::Image { h: 2, w: 4 },
            Segment::Text(1),
        ];
        let (p, target) = prefix_positions(&segs);
        assert_eq!(p.len(), 2 + 8 + 1);
        assert_eq!(p[0], [0, 0, 0]);
        assert_eq!(p[1], [1, 1, 1]);
        assert_eq!(p[2][0], 2); // image frame = position after the text
        assert_eq!(p[10], [6, 6, 6]); // 2 + max(2, 4)
        assert_eq!(target, 7);
    }

    #[test]
    fn text_mask_sees_prefix_and_is_causal_within() {
        crate::nn::test_device();
        let m = text_mask(2, 2).unwrap();
        m.eval().unwrap();
        assert_eq!(
            m.as_slice::<bool>(),
            &[true, true, true, false, true, true, true, true]
        );
    }
}

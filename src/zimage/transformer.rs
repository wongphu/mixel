//! Z-Image DiT (ZImageTransformer2DModel), ported from candle's
//! `z_image::transformer`.

use super::{linear, rms_norm, scalar, silu, Quantize, WeightCache, Weights};
use anyhow::Result;
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{concatenate, split_at_indices, tanh};
use mlx_rs::{Array, Dtype};
use std::path::Path;

const DIM: i32 = 3840;
const HEADS: i32 = 30;
const HEAD_DIM: i32 = 128;
const N_LAYERS: usize = 30;
const N_REFINER_LAYERS: usize = 2;
const NORM_EPS: f32 = 1e-5;
const QK_NORM_EPS: f32 = 1e-5;
const FINAL_NORM_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 256.0;
const T_SCALE: f32 = 1000.0;
const AXES_DIMS: [usize; 3] = [32, 48, 48];
const AXES_LENS: [usize; 3] = [1536, 512, 512];
/// Largest width or height: the RoPE tables cover 512 patches of 16 px per side.
pub const MAX_SIDE: usize = AXES_LENS[1] * 16;
const FREQ_EMBED_SIZE: usize = 256;
const MAX_PERIOD: f64 = 10000.0;
const PATCH: i32 = 2;
/// Like diffusers, the image and caption sequences are each padded to a
/// multiple of this with a learned pad token, which attention sees.
const SEQ_MULTI_OF: usize = 32;

pub struct Transformer {
    w: Weights,
    dtype: Dtype,
    /// Per-axis RoPE tables, (axis_len, axis_dim / 2).
    rope_cos: Vec<Array>,
    rope_sin: Vec<Array>,
}

/// (cos, sin), each (seq_len, HEAD_DIM / 2).
struct Rope {
    cos: Array,
    sin: Array,
}

impl Transformer {
    /// With `quantize`, the attention and feed-forward layers of every block
    /// are quantized (~98% of the weights). The embedders, the adaLN
    /// modulations and the final layer are small and stay unquantized.
    pub fn load(
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        quantize: Option<Quantize>,
        cache: Option<&WeightCache>,
    ) -> Result<Self> {
        let w = Weights::load_maybe_cached(
            files,
            dtype,
            |_| true,
            quantize,
            |layer| layer.contains(".attention.") || layer.contains(".feed_forward."),
            cache,
            "zimage-transformer",
        )?;
        let mut rope_cos = Vec::new();
        let mut rope_sin = Vec::new();
        for (&d, &len) in AXES_DIMS.iter().zip(&AXES_LENS) {
            let half = d / 2;
            let mut cos = Vec::with_capacity(len * half);
            let mut sin = Vec::with_capacity(len * half);
            for pos in 0..len {
                for i in 0..half {
                    let inv_freq = 1.0 / ROPE_THETA.powf((2 * i) as f32 / d as f32);
                    let f = pos as f32 * inv_freq;
                    cos.push(f.cos());
                    sin.push(f.sin());
                }
            }
            let shape = [len as i32, half as i32];
            rope_cos.push(Array::from_slice(&cos, &shape).as_dtype(dtype)?);
            rope_sin.push(Array::from_slice(&sin, &shape).as_dtype(dtype)?);
        }
        Ok(Self {
            w,
            dtype,
            rope_cos,
            rope_sin,
        })
    }

    /// Predicts the flow velocity.
    ///
    /// * `x` - latents (B, 16, H, W) in the model dtype
    /// * `t` - normalized time in [0, 1]
    /// * `cap` - caption features (B, text_len, 2560)
    pub fn forward(&self, x: &Array, t: f32, cap: &Array) -> Result<Array> {
        let (b, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let (ht, wt) = (h / PATCH, w / PATCH);
        let pad_to = |n: usize| n.div_ceil(SEQ_MULTI_OF) * SEQ_MULTI_OF;
        let img_len = (ht * wt) as usize;
        let cap_len = pad_to(cap.shape()[1] as usize);

        let adaln = self.t_embed(t)?; // (B, 256)

        // Patchify: (B, C, H, W) -> (B, H_t*W_t, pH*pW*C)
        let x = x
            .reshape(&[b, c, ht, PATCH, wt, PATCH])?
            .transpose_axes(&[0, 2, 4, 3, 5, 1])?
            .reshape(&[b, ht * wt, PATCH * PATCH * c])?;
        let x = linear(&x, &self.w, "all_x_embedder.2-1")?;
        let mut x = self.pad(&x, "x_pad_token", pad_to(img_len) - img_len)?;

        // Position ids: caption tokens (pad included) count up from 1; image
        // tokens sit after the padded caption on the frame axis; image pad
        // tokens are at 0.
        let x_ids: Vec<[usize; 3]> = (0..ht as usize)
            .flat_map(|hi| (0..wt as usize).map(move |wi| [cap_len + 1, hi, wi]))
            .chain(std::iter::repeat_n([0; 3], pad_to(img_len) - img_len))
            .collect();
        let cap_ids: Vec<[usize; 3]> = (0..cap_len).map(|i| [1 + i, 0, 0]).collect();
        let x_rope = self.rope(&x_ids)?;
        let cap_rope = self.rope(&cap_ids)?;

        let cap = linear(
            &rms_norm(cap, &self.w, "cap_embedder.0", NORM_EPS)?,
            &self.w,
            "cap_embedder.1",
        )?;
        let mut cap = self.pad(&cap, "cap_pad_token", cap_len - cap.shape()[1] as usize)?;

        for i in 0..N_REFINER_LAYERS {
            x = self.block(&format!("noise_refiner.{i}"), &x, &x_rope, Some(&adaln))?;
        }
        for i in 0..N_REFINER_LAYERS {
            cap = self.block(&format!("context_refiner.{i}"), &cap, &cap_rope, None)?;
        }

        let mut u = concatenate(&[&x, &cap], 1)?;
        let u_rope = Rope {
            cos: concatenate(&[&x_rope.cos, &cap_rope.cos], 0)?,
            sin: concatenate(&[&x_rope.sin, &cap_rope.sin], 0)?,
        };
        for i in 0..N_LAYERS {
            u = self.block(&format!("layers.{i}"), &u, &u_rope, Some(&adaln))?;
        }

        // Final layer on the image tokens only (not their padding).
        let x = split_at_indices(&u, &[img_len as i32], 1)?.swap_remove(0);
        let scale = linear(
            &silu(&adaln)?,
            &self.w,
            "all_final_layer.2-1.adaLN_modulation.1",
        )?
        .add(scalar(1.0, self.dtype)?)?
        .expand_dims(1)?;
        let x = mlx_rs::fast::layer_norm(x.as_dtype(Dtype::Float32)?, None, None, FINAL_NORM_EPS)?
            .as_dtype(self.dtype)?
            .multiply(&scale)?;
        let x = linear(&x, &self.w, "all_final_layer.2-1.linear")?;

        // Unpatchify: (B, H_t*W_t, pH*pW*C) -> (B, C, H, W)
        Ok(x.reshape(&[b, ht, wt, PATCH, PATCH, c])?
            .transpose_axes(&[0, 5, 1, 3, 2, 4])?
            .reshape(&[b, c, h, w])?)
    }

    /// Appends `n` copies of the learned pad token `name` (1, DIM) to the
    /// sequence (B, S, DIM).
    fn pad(&self, x: &Array, name: &str, n: usize) -> Result<Array> {
        if n == 0 {
            return Ok(x.clone());
        }
        let token = self.w.get(name)?.reshape(&[1, 1, DIM])?;
        let token = mlx_rs::ops::broadcast_to(&token, &[x.shape()[0], n as i32, DIM])?;
        Ok(concatenate(&[x, &token], 1)?)
    }

    /// Sinusoidal timestep embedding + MLP -> (1, 256).
    fn t_embed(&self, t: f32) -> Result<Array> {
        // Like diffusers, t * 1000 and the sinusoids are computed in f32, then
        // the embedding is cast to the model dtype. (candle rounds t and
        // t * 1000 to bf16: off by up to ~4, a few radians for the fastest
        // sinusoids.)
        let t = Array::from_f32(t).multiply(Array::from_f32(T_SCALE))?;
        let half = FREQ_EMBED_SIZE / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (i as f64 * (-MAX_PERIOD.ln() / half as f64)).exp() as f32)
            .collect();
        let args = Array::from_slice(&freqs, &[1, half as i32]).multiply(&t)?;
        let emb = concatenate(&[args.cos()?, args.sin()?], -1)?.as_dtype(self.dtype)?;
        let h = silu(&linear(&emb, &self.w, "t_embedder.mlp.0")?)?;
        linear(&h, &self.w, "t_embedder.mlp.2")
    }

    fn rope(&self, ids: &[[usize; 3]]) -> Result<Rope> {
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for axis in 0..3 {
            let idx: Vec<i32> = ids.iter().map(|id| id[axis] as i32).collect();
            let idx = Array::from_slice(&idx, &[idx.len() as i32]);
            cos.push(self.rope_cos[axis].take_axis(&idx, 0)?);
            sin.push(self.rope_sin[axis].take_axis(&idx, 0)?);
        }
        Ok(Rope {
            cos: concatenate(&cos, -1)?,
            sin: concatenate(&sin, -1)?,
        })
    }

    fn block(&self, p: &str, x: &Array, rope: &Rope, adaln: Option<&Array>) -> Result<Array> {
        let norm = |x: &Array, name: &str| rms_norm(x, &self.w, &format!("{p}.{name}"), NORM_EPS);
        match adaln {
            Some(c) => {
                let m = linear(c, &self.w, &format!("{p}.adaLN_modulation.0"))?.expand_dims(1)?;
                let [scale_msa, gate_msa, scale_mlp, gate_mlp]: [Array; 4] =
                    m.split_equal(4, -1)?.try_into().expect("4 chunks");
                let one = scalar(1.0, self.dtype)?;
                let (scale_msa, scale_mlp) = (scale_msa.add(&one)?, scale_mlp.add(&one)?);
                let (gate_msa, gate_mlp) = (tanh(&gate_msa)?, tanh(&gate_mlp)?);

                let a =
                    self.attention(p, &norm(x, "attention_norm1")?.multiply(&scale_msa)?, rope)?;
                let x = x.add(gate_msa.multiply(norm(&a, "attention_norm2")?)?)?;
                let f = self.feed_forward(p, &norm(&x, "ffn_norm1")?.multiply(&scale_mlp)?)?;
                Ok(x.add(gate_mlp.multiply(norm(&f, "ffn_norm2")?)?)?)
            }
            None => {
                let a = self.attention(p, &norm(x, "attention_norm1")?, rope)?;
                let x = x.add(norm(&a, "attention_norm2")?)?;
                let f = self.feed_forward(p, &norm(&x, "ffn_norm1")?)?;
                Ok(x.add(norm(&f, "ffn_norm2")?)?)
            }
        }
    }

    fn feed_forward(&self, p: &str, x: &Array) -> Result<Array> {
        let w1 = silu(&linear(x, &self.w, &format!("{p}.feed_forward.w1"))?)?;
        let w3 = linear(x, &self.w, &format!("{p}.feed_forward.w3"))?;
        linear(&w1.multiply(w3)?, &self.w, &format!("{p}.feed_forward.w2"))
    }

    fn attention(&self, p: &str, x: &Array, rope: &Rope) -> Result<Array> {
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let proj = |name: &str| -> Result<Array> {
            Ok(linear(x, &self.w, &format!("{p}.attention.{name}"))?
                .reshape(&[b, s, HEADS, HEAD_DIM])?)
        };
        let q = rms_norm(
            &proj("to_q")?,
            &self.w,
            &format!("{p}.attention.norm_q"),
            QK_NORM_EPS,
        )?;
        let k = rms_norm(
            &proj("to_k")?,
            &self.w,
            &format!("{p}.attention.norm_k"),
            QK_NORM_EPS,
        )?;
        let v = proj("to_v")?;
        let to_bhsd = |a: Array| a.transpose_axes(&[0, 2, 1, 3]);
        let q = to_bhsd(apply_rope(&q, rope)?)?;
        let k = to_bhsd(apply_rope(&k, rope)?)?;
        let v = to_bhsd(v)?;
        let o =
            scaled_dot_product_attention(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), None, None)?;
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, DIM])?;
        linear(&o, &self.w, &format!("{p}.attention.to_out.0"))
    }
}

/// Interleaved (complex-pair) RoPE on (B, S, H, D) with (S, D/2) tables.
fn apply_rope(x: &Array, rope: &Rope) -> Result<Array> {
    let sh = x.shape().to_vec();
    let (b, s, h, d) = (sh[0], sh[1], sh[2], sh[3]);
    let [re, im]: [Array; 2] = x
        .reshape(&[b, s, h, d / 2, 2])?
        .split_equal(2, -1)?
        .try_into()
        .expect("2 chunks");
    let cos = rope.cos.reshape(&[1, s, 1, d / 2, 1])?;
    let sin = rope.sin.reshape(&[1, s, 1, d / 2, 1])?;
    let out_re = re.multiply(&cos)?.subtract(im.multiply(&sin)?)?;
    let out_im = re.multiply(&sin)?.add(im.multiply(&cos)?)?;
    Ok(concatenate(&[out_re, out_im], -1)?.reshape(&[b, s, h, d])?)
}

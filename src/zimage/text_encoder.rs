//! Qwen3 text encoder, returning the second-to-last layer's hidden states
//! (no final norm), ported from candle's `z_image::text_encoder`.

use super::{linear, rms_norm, silu, Quantize, Weights};
use anyhow::Result;
use mlx_rs::fast::{rope, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::{Array, Dtype};
use std::path::Path;

const HEADS: i32 = 32;
const KV_HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;
const NUM_LAYERS: usize = 36;
/// Output is taken after this layer (hidden_states[-2]); later layers aren't loaded.
const LAST_LAYER: usize = NUM_LAYERS - 2;
const EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;

pub struct TextEncoder {
    w: Weights,
}

impl TextEncoder {
    /// With `quantize`, every attention and MLP projection is quantized (not
    /// the token embeddings).
    pub fn load(
        files: &[impl AsRef<Path>],
        dtype: Dtype,
        quantize: Option<Quantize>,
    ) -> Result<Self> {
        let keep = |name: &str| {
            if name == "model.embed_tokens.weight" {
                return true;
            }
            name.strip_prefix("model.layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| i <= LAST_LAYER)
        };
        let w = Weights::load_quantized(files, dtype, keep, quantize, |layer| {
            layer.starts_with("model.layers.")
        })?;
        Ok(Self { w })
    }

    /// Token ids -> (1, len, 2560) caption features.
    pub fn forward(&self, ids: &[u32]) -> Result<Array> {
        let ids: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let ids = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let mut h = self
            .w
            .get("model.embed_tokens.weight")?
            .take_axis(&ids, 0)?;
        for i in 0..=LAST_LAYER {
            h = self.layer(i, &h)?;
        }
        Ok(h)
    }

    fn layer(&self, i: usize, x: &Array) -> Result<Array> {
        let p = format!("model.layers.{i}");
        let h = rms_norm(x, &self.w, &format!("{p}.input_layernorm"), EPS)?;
        let x = x.add(self.attention(&p, &h)?)?;
        let h = rms_norm(&x, &self.w, &format!("{p}.post_attention_layernorm"), EPS)?;
        let gate = silu(&linear(&h, &self.w, &format!("{p}.mlp.gate_proj"))?)?;
        let up = linear(&h, &self.w, &format!("{p}.mlp.up_proj"))?;
        let mlp = linear(&gate.multiply(up)?, &self.w, &format!("{p}.mlp.down_proj"))?;
        Ok(x.add(mlp)?)
    }

    fn attention(&self, p: &str, x: &Array) -> Result<Array> {
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let proj = |name: &str, heads: i32| -> Result<Array> {
            Ok(linear(x, &self.w, &format!("{p}.self_attn.{name}"))?
                .reshape(&[b, l, heads, HEAD_DIM])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = rms_norm(
            &proj("q_proj", HEADS)?,
            &self.w,
            &format!("{p}.self_attn.q_norm"),
            EPS,
        )?;
        let k = rms_norm(
            &proj("k_proj", KV_HEADS)?,
            &self.w,
            &format!("{p}.self_attn.k_norm"),
            EPS,
        )?;
        let v = proj("v_proj", KV_HEADS)?;
        let q = rope(&q, HEAD_DIM, false, ROPE_THETA, 1.0, 0, None)?;
        let k = rope(&k, HEAD_DIM, false, ROPE_THETA, 1.0, 0, None)?;
        // GQA is handled by the fused kernel without tiling k/v.
        let o = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0 / (HEAD_DIM as f32).sqrt(),
            ScaledDotProductAttentionMask::Causal,
            None,
        )?;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, HEADS * HEAD_DIM])?;
        linear(&o, &self.w, &format!("{p}.self_attn.o_proj"))
    }
}

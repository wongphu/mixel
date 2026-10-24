//! The gated delta rule of Qwen3.5's linear-attention layers (Gated
//! DeltaNet), ported from transformers' `torch_chunk_gated_delta_rule` with
//! `use_qk_l2norm_in_kernel`: everything in f32, the sequence in chunks of 64
//! tokens, and a sequential scan over the chunks.

use anyhow::Result;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate, pad, select, tri};
use mlx_rs::{Array, Dtype, Stream};

/// Tokens per chunk, as in transformers.
const CHUNK: i32 = 64;
/// `l2norm`'s epsilon (matches the FLA library).
const L2_EPS: f32 = 1e-6;

/// `x * rsqrt(sum(x^2) + eps)` over the last axis.
fn l2norm(x: &Array) -> Result<Array> {
    let inv = x
        .square()?
        .sum_axis(-1, true)?
        .add(Array::from_f32(L2_EPS))?
        .rsqrt()?;
    Ok(x.multiply(inv)?)
}

/// Casts to f32 and moves heads first: q, k (H, L, Dk), v (H, L, Dv), and
/// g, beta (H, L). q and k are L2-normalized and q scaled by `Dk^-0.5`.
fn prepare(q: &Array, k: &Array, v: &Array, g: &Array, beta: &Array) -> Result<[Array; 5]> {
    let heads_first = |x: &Array| -> Result<Array> {
        let x = x.as_dtype(Dtype::Float32)?;
        Ok(if x.ndim() == 3 {
            x.transpose_axes(&[1, 0, 2])?
        } else {
            x.transpose_axes(&[1, 0])?
        })
    };
    let q = l2norm(&heads_first(q)?)?;
    let k = l2norm(&heads_first(k)?)?;
    let scale = (q.shape()[2] as f32).powf(-0.5);
    let q = q.multiply(Array::from_f32(scale))?;
    Ok([q, k, heads_first(v)?, heads_first(g)?, heads_first(beta)?])
}

/// The gated delta rule over a whole sequence (no initial state).
///
/// * `q`, `k` - (L, H, Dk), one key head per value head
/// * `v` - (L, H, Dv)
/// * `g` - (L, H) log decay (<= 0), `beta` - (L, H) write strength
///
/// Returns (L, H, Dv) in f32.
pub fn chunked(q: &Array, k: &Array, v: &Array, g: &Array, beta: &Array) -> Result<Array> {
    let [q, k, v, g, beta] = prepare(q, k, v, g, beta)?;
    let (h, l, dk) = (k.shape()[0], k.shape()[1], k.shape()[2]);
    let dv = v.shape()[2];
    let pad_len = (CHUNK - l % CHUNK) % CHUNK;
    let n = (l + pad_len) / CHUNK;
    let pad_seq = |x: &Array| -> Result<Array> {
        let widths: Vec<(i32, i32)> = (0..x.ndim())
            .map(|axis| if axis == 1 { (0, pad_len) } else { (0, 0) })
            .collect();
        Ok(pad(x, widths.as_slice(), None, None)?)
    };
    let chunks = |x: &Array, d: i32| x.reshape(&[h, n, CHUNK, d]);

    let beta = pad_seq(&beta)?.expand_dims(-1)?;
    let v = pad_seq(&v)?;
    let k = pad_seq(&k)?;
    let v_beta = chunks(&v.multiply(&beta)?, dv)?;
    let k_beta = chunks(&k.multiply(&beta)?, dk)?;
    let q = chunks(&pad_seq(&q)?, dk)?;
    let k = chunks(&k, dk)?;
    // Log decay accumulated from the start of each chunk, (H, N, C).
    let cum = pad_seq(&g)?
        .reshape(&[h, n, CHUNK])?
        .cumsum(-1, None, None)?;

    // decay[i, j] = exp(cum_i - cum_j) for j <= i, else 0.
    let causal = tri::<bool>(CHUNK, None, Some(0))?;
    let diff = cum.expand_dims(-1)?.subtract(cum.expand_dims(-2)?)?;
    let decay = select(&causal, &diff, Array::from_f32(f32::NEG_INFINITY))?.exp()?;

    let kt = k.transpose_axes(&[0, 1, 3, 2])?;
    let system = k_beta.matmul(&kt)?.multiply(&decay)?;
    let intra = q.matmul(&kt)?.multiply(&decay)?;
    let exp_cum = cum.exp()?.expand_dims(-1)?;

    // Solve (I + strictly_lower(system)) X = B for both right-hand sides.
    let inv = unit_lower_inverse(&system)?;
    let new_values = inv.matmul(&v_beta)?;
    let k_cumdecay = inv.matmul(&k_beta.multiply(&exp_cum)?)?;

    let q = q.multiply(&exp_cum)?;
    let last = cum.index((.., .., CHUNK - 1..CHUNK)); // (H, N, 1)
    let k = k.multiply(last.subtract(&cum)?.exp()?.expand_dims(-1)?)?;
    let chunk_decay = last.exp()?.expand_dims(-1)?; // (H, N, 1, 1)

    let mut state = Array::zeros::<f32>(&[h, dk, dv])?;
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let at = |x: &Array| x.index((.., i));
        let v_new = at(&new_values).subtract(at(&k_cumdecay).matmul(&state)?)?;
        out.push(at(&q).matmul(&state)?.add(at(&intra).matmul(&v_new)?)?);
        state = state
            .multiply(at(&chunk_decay))?
            .add(at(&k).transpose_axes(&[0, 2, 1])?.matmul(&v_new)?)?;
    }
    let out = concatenate(&out, 1)?.index((.., ..l, ..));
    Ok(out.transpose_axes(&[1, 0, 2])?)
}

/// `(I + strictly_lower(a))^-1` for a batch of (C, C) matrices: what
/// transformers' `solve_triangular(.., unitriangular=True)` solves with.
/// MLX's triangular inverse only runs on the CPU; the matrices are small.
fn unit_lower_inverse(a: &Array) -> Result<Array> {
    let strict = tri::<f32>(CHUNK, None, Some(-1))?;
    let eye = Array::eye::<f32>(CHUNK, None, None)?;
    let t = a.multiply(&strict)?.add(&eye)?;
    let cpu = Stream::cpu();
    Ok(mlx_rs::with_stream(&cpu, || {
        mlx_rs::linalg::tri_inv(&t, Some(false))
    })?)
}

/// The same rule one token at a time (transformers'
/// `torch_recurrent_gated_delta_rule`), as a reference for [`chunked`].
#[cfg(test)]
pub fn recurrent(q: &Array, k: &Array, v: &Array, g: &Array, beta: &Array) -> Result<Array> {
    let [q, k, v, g, beta] = prepare(q, k, v, g, beta)?;
    let (h, l, dk) = (k.shape()[0], k.shape()[1], k.shape()[2]);
    let dv = v.shape()[2];
    let mut state = Array::zeros::<f32>(&[h, dk, dv])?;
    let mut out = Vec::with_capacity(l as usize);
    for t in 0..l {
        let (q_t, k_t, v_t) = (q.index((.., t)), k.index((.., t)), v.index((.., t)));
        let decay = g.index((.., t..t + 1)).exp()?.expand_dims(-1)?; // (H, 1, 1)
        state = state.multiply(&decay)?;
        let k_col = k_t.expand_dims(-1)?; // (H, Dk, 1)
        let kv_mem = state.multiply(&k_col)?.sum_axis(-2, None)?; // (H, Dv)
        let delta = v_t
            .subtract(&kv_mem)?
            .multiply(beta.index((.., t..t + 1)))?;
        state = state.add(k_col.multiply(delta.expand_dims(-2)?)?)?;
        out.push(state.multiply(q_t.expand_dims(-1)?)?.sum_axis(-2, None)?);
    }
    let out = mlx_rs::ops::stack(&out, 1)?; // (H, L, Dv)
    Ok(out.transpose_axes(&[1, 0, 2])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(len: i32, seed: u64) -> [Array; 5] {
        let key = |i: u64| mlx_rs::random::key(seed * 16 + i).unwrap();
        let (h, dk, dv) = (3, 8, 5);
        let normal =
            |shape: &[i32], i| mlx_rs::random::normal::<f32>(shape, None, None, &key(i)).unwrap();
        let uniform = |lo: f32, hi: f32, i| {
            mlx_rs::random::uniform::<_, f32>(lo, hi, &[len, h][..], &key(i)).unwrap()
        };
        [
            normal(&[len, h, dk], 0),
            normal(&[len, h, dk], 1),
            normal(&[len, h, dv], 2),
            uniform(-1.5, -0.01, 3),
            uniform(0.0, 1.0, 4),
        ]
    }

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let d = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
        d.item_exact::<f32>()
    }

    #[test]
    fn chunked_matches_the_token_by_token_recurrence() {
        let _mlx = crate::nn::test_device();
        // Within a chunk, at its edges, and across several chunks.
        for (i, len) in [1, 5, 63, 64, 65, 130].into_iter().enumerate() {
            let [q, k, v, g, beta] = inputs(len, i as u64);
            let fast = chunked(&q, &k, &v, &g, &beta).unwrap();
            let slow = recurrent(&q, &k, &v, &g, &beta).unwrap();
            assert_eq!(fast.shape(), &[len, 3, 5]);
            let err = max_abs_diff(&fast, &slow);
            assert!(err < 1e-4, "length {len}: max difference {err}");
        }
    }

    #[test]
    fn the_inverse_inverts() {
        let _mlx = crate::nn::test_device();
        let a = mlx_rs::random::normal::<f32>(
            &[2, CHUNK, CHUNK][..],
            None,
            None,
            &mlx_rs::random::key(9).unwrap(),
        )
        .unwrap()
        .multiply(Array::from_f32(0.1))
        .unwrap();
        let t = a
            .multiply(tri::<f32>(CHUNK, None, Some(-1)).unwrap())
            .unwrap()
            .add(Array::eye::<f32>(CHUNK, None, None).unwrap())
            .unwrap();
        let product = t.matmul(unit_lower_inverse(&a).unwrap()).unwrap();
        let eye = Array::eye::<f32>(CHUNK, None, None).unwrap();
        assert!(max_abs_diff(&product, &eye) < 1e-4);
    }
}

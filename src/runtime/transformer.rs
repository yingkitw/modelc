//! Runtime transformer forward passes for `modelc run`.
//!
//! These execute GPT-2 / LLaMA artifacts directly from the in-memory `Runtime` so that
//! `modelc run` produces real logits instead of echoing the input. The numeric helpers and
//! the block structure mirror the code emitted by [`crate::codegen::native::forward`] exactly,
//! so `modelc run` and `modelc compile` yield identical outputs for the same model. Structure
//! inspection (layer counts, hidden dim, head count) is shared via [`crate::arch`].

use crate::arch;
use crate::model::{DataType, Model, TensorData};
use crate::runtime::serve::Runtime;

/// Run a single-vector (single-token) GPT-2 forward. `input.len()` must equal the model's
/// hidden size. Returns `None` when the model lacks any output projection (`lm_head.weight`
/// or tied `transformer.wte.weight`), so the caller can fall back.
pub fn forward_gpt2(runtime: &Runtime, input: &[f32], profile: bool) -> Option<Vec<f32>> {
    forward_gpt2_cached(runtime, input, profile, &mut None)
}

pub fn forward_gpt2_cached(
    runtime: &Runtime,
    input: &[f32],
    profile: bool,
    kv_cache: &mut Option<KvCache>,
) -> Option<Vec<f32>> {
    let view = runtime_model_view(runtime);
    let layers = arch::detect_layers(&view, "transformer.h.");
    let hidden = arch::gpt2_hidden_dim(&view, &layers);
    let n_heads = arch::gpt2_head_count(hidden);

    let mut h = input.to_vec();

    for (idx, &layer) in layers.iter().enumerate() {
        let p = format!("transformer.h.{layer}.");
        let start = std::time::Instant::now();
        let cache = kv_cache.as_mut().and_then(|c| c.layers.get_mut(idx));
        let changed = gpt2_step(runtime, &p, hidden, n_heads, &mut h, cache);
        if profile && changed {
            eprintln!(
                "    gpt2 block {layer}: {:.3} ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    if let (Some(w), Some(b)) = (
        runtime.get("transformer.ln_f.weight"),
        runtime.get("transformer.ln_f.bias"),
    ) {
        h = layer_norm(&h, &w.data, &b.data, 1e-5);
    }

    linear_named(runtime, "lm_head.weight", "lm_head.bias", &h)
        .or_else(|| linear_named(runtime, "transformer.wte.weight", "", &h))
}

/// Run a single-vector (single-token) LLaMA forward. `input.len()` must equal the model's
/// hidden size. Returns `None` when the model lacks any output projection (`lm_head.weight`
/// or tied `model.embed_tokens.weight`).
pub fn forward_llama(runtime: &Runtime, input: &[f32], profile: bool) -> Option<Vec<f32>> {
    forward_llama_cached(runtime, input, profile, &mut None)
}

pub fn forward_llama_cached(
    runtime: &Runtime,
    input: &[f32],
    profile: bool,
    kv_cache: &mut Option<KvCache>,
) -> Option<Vec<f32>> {
    let view = runtime_model_view(runtime);
    let layers = arch::llama_layers(&view);
    let hidden = arch::llama_hidden_dim(&view, &layers);
    let n_heads = arch::llama_head_count(&view, hidden);

    let mut h = input.to_vec();

    for (idx, &layer) in layers.iter().enumerate() {
        let p = format!("model.layers.{layer}.");
        let start = std::time::Instant::now();
        let cache = kv_cache.as_mut().and_then(|c| c.layers.get_mut(idx));
        let changed = llama_step(runtime, &p, n_heads, &mut h, cache);
        if profile && changed {
            eprintln!(
                "    llama block {layer}: {:.3} ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    if let Some(w) = runtime.get("model.norm.weight") {
        h = rms_norm(&h, &w.data, 1e-6);
    }

    linear_bias_free(runtime, "lm_head.weight", &h)
        .or_else(|| linear_bias_free(runtime, "model.embed_tokens.weight", &h))
}

/// Extract the final hidden-state vector (after layer norm, before the output head) for
/// GPT-2 artifacts. Returns `None` when the model lacks the necessary tensors.
pub fn embed_gpt2(runtime: &Runtime, input: &[f32], profile: bool) -> Option<Vec<f32>> {
    let view = runtime_model_view(runtime);
    let layers = arch::detect_layers(&view, "transformer.h.");
    let hidden = arch::gpt2_hidden_dim(&view, &layers);
    let n_heads = arch::gpt2_head_count(hidden);

    let mut h = input.to_vec();

    for &layer in &layers {
        let p = format!("transformer.h.{layer}.");
        let start = std::time::Instant::now();
        let changed = gpt2_step(runtime, &p, hidden, n_heads, &mut h, None);
        if profile && changed {
            eprintln!("    gpt2 block {layer}: {:.3} ms", start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    if let (Some(w), Some(b)) = (
        runtime.get("transformer.ln_f.weight"),
        runtime.get("transformer.ln_f.bias"),
    ) {
        h = layer_norm(&h, &w.data, &b.data, 1e-5);
    }

    Some(h)
}

/// Extract the final hidden-state vector (after RMS norm, before the output head) for
/// LLaMA artifacts.
pub fn embed_llama(runtime: &Runtime, input: &[f32], profile: bool) -> Option<Vec<f32>> {
    let view = runtime_model_view(runtime);
    let layers = arch::llama_layers(&view);
    let hidden = arch::llama_hidden_dim(&view, &layers);
    let n_heads = arch::llama_head_count(&view, hidden);

    let mut h = input.to_vec();

    for &layer in &layers {
        let p = format!("model.layers.{layer}.");
        let start = std::time::Instant::now();
        let changed = llama_step(runtime, &p, n_heads, &mut h, None);
        if profile && changed {
            eprintln!("    llama block {layer}: {:.3} ms", start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    if let Some(w) = runtime.get("model.norm.weight") {
        h = rms_norm(&h, &w.data, 1e-6);
    }

    Some(h)
}

// ---------------------------------------------------------------------------
// KV cache — stores per-layer Key and Value vectors for autoregressive generation.
// ---------------------------------------------------------------------------

/// A cache of Key/Value vectors for each transformer layer.
/// `layers[i]` is `Some((k, v))` when the i-th layer has cached K/V vectors.
pub struct KvCache {
    layers: Vec<Option<(Vec<f32>, Vec<f32>)>>,
}

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: vec![None; n_layers],
        }
    }
}

// ---------------------------------------------------------------------------
// Per-layer steps. Each mutates the residual stream `h` in place and returns whether the
// block had any effect (so callers can skip profiling when the block was a no-op).
// ---------------------------------------------------------------------------

fn gpt2_step(
    runtime: &Runtime,
    prefix: &str,
    hidden: usize,
    n_heads: usize,
    h: &mut Vec<f32>,
    kv_cache: Option<&mut Option<(Vec<f32>, Vec<f32>)>>,
) -> bool {
    let mut touched = false;

    let mut binding = None;
    let cache = kv_cache.unwrap_or(&mut binding);
    if let Some(next) = gpt2_attention(runtime, prefix, hidden, n_heads, h, cache) {
        *h = next;
        touched = true;
    }
    if let Some(next) = gpt2_mlp(runtime, prefix, h) {
        *h = next;
        touched = true;
    }

    touched
}

fn gpt2_attention(
    runtime: &Runtime,
    prefix: &str,
    hidden: usize,
    n_heads: usize,
    h: &[f32],
    kv_cache: &mut Option<(Vec<f32>, Vec<f32>)>,
) -> Option<Vec<f32>> {
    let c_attn_w = runtime.get(&format!("{prefix}attn.c_attn.weight"))?;
    let c_attn_b = runtime.get(&format!("{prefix}attn.c_attn.bias"))?;

    let normed = match (
        runtime.get(&format!("{prefix}ln_1.weight")),
        runtime.get(&format!("{prefix}ln_1.bias")),
    ) {
        (Some(w), Some(b)) => layer_norm(h, &w.data, &b.data, 1e-5),
        _ => h.to_vec(),
    };

    let qkv = linear_with(c_attn_w, Some(c_attn_b), &normed);
    let (q, rest) = qkv.split_at(hidden);
    let (k, v) = rest.split_at(hidden);

    let (k_all, v_all) = if let Some((cached_k, cached_v)) = kv_cache {
        let mut k_all = cached_k.clone();
        let mut v_all = cached_v.clone();
        k_all.extend_from_slice(k);
        v_all.extend_from_slice(v);
        *cached_k = k_all.clone();
        *cached_v = v_all.clone();
        (k_all, v_all)
    } else {
        (k.to_vec(), v.to_vec())
    };

    let attn = attention_kv(q, &k_all, &v_all, n_heads);

    let proj = match (
        runtime.get(&format!("{prefix}attn.c_proj.weight")),
        runtime.get(&format!("{prefix}attn.c_proj.bias")),
    ) {
        (Some(w), Some(b)) => linear_with(w, Some(b), &attn),
        _ => attn,
    };

    Some(add(h, &proj))
}

fn gpt2_mlp(runtime: &Runtime, prefix: &str, h: &[f32]) -> Option<Vec<f32>> {
    let c_fc_w = runtime.get(&format!("{prefix}mlp.c_fc.weight"))?;
    let c_fc_b = runtime.get(&format!("{prefix}mlp.c_fc.bias"))?;
    let c_proj_w = runtime.get(&format!("{prefix}mlp.c_proj.weight"))?;
    let c_proj_b = runtime.get(&format!("{prefix}mlp.c_proj.bias"))?;

    let normed = match (
        runtime.get(&format!("{prefix}ln_2.weight")),
        runtime.get(&format!("{prefix}ln_2.bias")),
    ) {
        (Some(w), Some(b)) => layer_norm(h, &w.data, &b.data, 1e-5),
        _ => h.to_vec(),
    };

    let fc = linear_with(c_fc_w, Some(c_fc_b), &normed);
    let act = gelu(&fc);
    let proj = linear_with(c_proj_w, Some(c_proj_b), &act);
    Some(add(h, &proj))
}

fn llama_step(
    runtime: &Runtime,
    prefix: &str,
    n_heads: usize,
    h: &mut Vec<f32>,
    kv_cache: Option<&mut Option<(Vec<f32>, Vec<f32>)>>,
) -> bool {
    let mut touched = false;

    let mut binding = None;
    let cache = kv_cache.unwrap_or(&mut binding);
    if let Some(next) = llama_attention(runtime, prefix, n_heads, h, cache) {
        *h = next;
        touched = true;
    }
    if let Some(next) = llama_mlp(runtime, prefix, h) {
        *h = next;
        touched = true;
    }

    touched
}

fn llama_attention(
    runtime: &Runtime,
    prefix: &str,
    n_heads: usize,
    h: &[f32],
    kv_cache: &mut Option<(Vec<f32>, Vec<f32>)>,
) -> Option<Vec<f32>> {
    let q_w = runtime.get(&format!("{prefix}self_attn.q_proj.weight"))?;
    let k_w = runtime.get(&format!("{prefix}self_attn.k_proj.weight"))?;
    let v_w = runtime.get(&format!("{prefix}self_attn.v_proj.weight"))?;
    let o_w = runtime.get(&format!("{prefix}self_attn.o_proj.weight"))?;

    let normed = match runtime.get(&format!("{prefix}input_layernorm.weight")) {
        Some(w) => rms_norm(h, &w.data, 1e-6),
        None => h.to_vec(),
    };

    let q = linear_with(q_w, None, &normed);
    let k = linear_with(k_w, None, &normed);
    let v = linear_with(v_w, None, &normed);

    let pos = kv_cache.as_ref().map(|(k, _)| k.len() / q.len()).unwrap_or(0);
    let q = rope(&q, pos, n_heads);
    let k = rope(&k, pos, n_heads);

    let (k_all, v_all) = if let Some((cached_k, cached_v)) = kv_cache {
        let mut k_all = cached_k.clone();
        let mut v_all = cached_v.clone();
        k_all.extend_from_slice(&k);
        v_all.extend_from_slice(&v);
        *cached_k = k_all.clone();
        *cached_v = v_all.clone();
        (k_all, v_all)
    } else {
        (k, v)
    };

    let attn = attention_kv(&q, &k_all, &v_all, n_heads);
    let proj = linear_with(o_w, None, &attn);

    Some(add(h, &proj))
}

fn llama_mlp(runtime: &Runtime, prefix: &str, h: &[f32]) -> Option<Vec<f32>> {
    let gate_w = runtime.get(&format!("{prefix}mlp.gate_proj.weight"))?;
    let up_w = runtime.get(&format!("{prefix}mlp.up_proj.weight"))?;
    let down_w = runtime.get(&format!("{prefix}mlp.down_proj.weight"))?;

    let normed = match runtime.get(&format!("{prefix}post_attention_layernorm.weight")) {
        Some(w) => rms_norm(h, &w.data, 1e-6),
        None => h.to_vec(),
    };

    // SwiGLU: down(silu(gate(x)) * up(x))
    let gate = linear_with(gate_w, None, &normed);
    let up = linear_with(up_w, None, &normed);
    let act = silu(&gate);
    let prod: Vec<f32> = act.iter().zip(up.iter()).map(|(a, b)| a * b).collect();
    let proj = linear_with(down_w, None, &prod);
    Some(add(h, &proj))
}

// ---------------------------------------------------------------------------
// Numeric helpers — identical math to the emitted codegen helpers
// (src/codegen/native/helpers.rs) so `run` and `compile` outputs match.
// ---------------------------------------------------------------------------

/// `y = W x + b` for a dense `[out, in]` weight. `bias` may be empty/None → treated as zero.
fn linear_with(
    weight: &crate::runtime::tensor::Tensor,
    bias: Option<&crate::runtime::tensor::Tensor>,
    x: &[f32],
) -> Vec<f32> {
    let (rows, cols) = matrix_dims(weight);
    assert_eq!(cols, x.len(), "linear: cols {cols} != input {}", x.len());

    let mut out = vec![0.0f32; rows];
    for (r, out_v) in out.iter_mut().enumerate() {
        let mut acc = bias.and_then(|b| b.data.get(r).copied()).unwrap_or(0.0);
        let row = &weight.data[r * cols..(r + 1) * cols];
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += *wv * *xv;
        }
        *out_v = acc;
    }
    out
}

/// Look up `weight_key` (+ optional `bias_key`) in the runtime and run [`linear_with`].
/// Returns `None` if the weight is absent.
fn linear_named(
    runtime: &Runtime,
    weight_key: &str,
    bias_key: &str,
    x: &[f32],
) -> Option<Vec<f32>> {
    let w = runtime.get(weight_key)?;
    let b = if bias_key.is_empty() {
        None
    } else {
        runtime.get(bias_key)
    };
    Some(linear_with(w, b, x))
}

/// Bias-free linear (`y = W x`) for LLaMA projections.
fn linear_bias_free(runtime: &Runtime, weight_key: &str, x: &[f32]) -> Option<Vec<f32>> {
    let w = runtime.get(weight_key)?;
    Some(linear_with(w, None, x))
}

fn matrix_dims(t: &crate::runtime::tensor::Tensor) -> (usize, usize) {
    assert_eq!(t.shape.len(), 2, "expected dense matrix, got {:?}", t.shape);
    (t.shape[0], t.shape[1])
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n.max(1) as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n.max(1) as f32;
    let std = (var + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let wi = w.get(i).copied().unwrap_or(1.0);
            let bi = b.get(i).copied().unwrap_or(0.0);
            (v - mean) / std * wi + bi
        })
        .collect()
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len().max(1) as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let rms = (ms + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| v / rms * w.get(i).copied().unwrap_or(1.0))
        .collect()
}

/// Exact (tanh-approx) GeLU as used by GPT-2.
fn gelu(x: &[f32]) -> Vec<f32> {
    let c = f32::sqrt(2.0 / std::f32::consts::PI);
    x.iter()
        .map(|&v| 0.5 * v * (1.0 + (c * (v + 0.044715 * v.powi(3))).tanh()))
        .collect()
}

/// SiLU / swish: `x / (1 + e^-x)`.
fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0))
        .collect()
}

/// Rotary position embedding for `pos` (head_dim = x.len() / n_heads). For `pos = 0` this is
/// the identity; the rotation is still applied so the structure matches a full-sequence
/// forward and matches the codegen output.
fn rope(x: &[f32], pos: usize, n_heads: usize) -> Vec<f32> {
    let head_dim = (x.len() / n_heads.max(1)).max(1);
    let half = head_dim / 2;
    let mut out = x.to_vec();
    let base = 10000.0f32;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let theta = pos as f32 * base.powf(-(i as f32) / half as f32);
            let (s, c) = theta.sin_cos();
            let x0 = x[off + i];
            let x1 = x[off + i + half];
            out[off + i] = x0 * c - x1 * s;
            out[off + i + half] = x0 * s + x1 * c;
        }
    }
    out
}

/// Multi-token attention with optional KV cache.
///
/// `q` is [n_heads * head_dim] for the current token.  
/// `k_all` and `v_all` are [seq_len * n_heads * head_dim] including cached + current.
fn attention_kv(q: &[f32], k_all: &[f32], v_all: &[f32], n_heads: usize) -> Vec<f32> {
    let head_dim = (q.len() / n_heads.max(1)).max(1);
    let seq_len = k_all.len() / (n_heads * head_dim).max(1);
    let mut out = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let q_off = h * head_dim;
        let q_head = &q[q_off..q_off + head_dim];

        let mut scores = vec![0.0f32; seq_len];
        for (t, score) in scores.iter_mut().enumerate() {
            let k_off = t * n_heads * head_dim + h * head_dim;
            let k_head = &k_all[k_off..k_off + head_dim];
            let dot: f32 = q_head.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum();
            *score = dot / (head_dim as f32).sqrt();
        }

        let weights = softmax(&scores);

        for (t, w) in weights.iter().enumerate() {
            let v_off = t * n_heads * head_dim + h * head_dim;
            let v_head = &v_all[v_off..v_off + head_dim];
            for i in 0..head_dim {
                out[h * head_dim + i] += w * v_head[i];
            }
        }
    }

    out
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Bridge: present the runtime's tensor names/shapes as a synthetic `Model` so the
// shared `arch` inspection helpers can read structure without a real `Model` instance.
// ---------------------------------------------------------------------------

fn runtime_model_view(runtime: &Runtime) -> Model {
    let mut tensors = std::collections::HashMap::new();
    for name in runtime.tensor_names() {
        if let Some(t) = runtime.get(name) {
            tensors.insert(
                name.clone(),
                TensorData {
                    shape: t.shape.clone(),
                    dtype: DataType::F32,
                    data: Vec::new(),
                },
            );
        }
    }
    Model {
        name: String::new(),
        architecture: String::new(),
        tensors,
        metadata: std::collections::HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_zero_and_sign() {
        let g = gelu(&[0.0, 1.0, -1.0]);
        assert!(g[0].abs() < 1e-6);
        assert!(g[1] > 0.0 && g[1] < 1.0);
        assert!(g[2] < 0.0 && g[2] > -0.16);
    }

    #[test]
    fn silu_known_values() {
        let s = silu(&[0.0, 1.0]);
        assert!(s[0].abs() < 1e-6);
        assert!((s[1] - (1.0 / (1.0 + (-1.0_f32).exp()))).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_centers_and_scales() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let b = vec![0.0; 4];
        let y = layer_norm(&x, &w, &b, 1e-5);
        let mean = y.iter().sum::<f32>() / y.len() as f32;
        assert!(mean.abs() < 1e-5, "normalized mean ~ 0, got {mean}");
    }

    #[test]
    fn rms_norm_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let y = rms_norm(&x, &w, 1e-6);
        assert!(y.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn rope_at_zero_is_identity() {
        let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let y = rope(&x, 0, 2);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-5, "rope(pos=0) must be identity");
        }
    }

    #[test]
    fn add_is_elementwise() {
        assert_eq!(
            add(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]),
            vec![11.0, 22.0, 33.0]
        );
    }

    #[test]
    fn attention_kv_single_token_is_identity() {
        // With seq_len=1, attention_kv should return v (same as old single_token_attention).
        let q = vec![1.0f32, 2.0];
        let k = vec![1.0f32, 2.0];
        let v = vec![0.5f32, 0.25];
        let out = attention_kv(&q, &k, &v, 1);
        assert_eq!(out, v);
    }

    #[test]
    fn attention_kv_with_cache_two_tokens() {
        // Two tokens, single head, head_dim=2.
        // Token 0: q=[1,0], k=[1,0], v=[1,0]
        // Token 1: q=[0,1], k=[0,1], v=[0,1]
        // Querying with token 1 against both keys.
        let q = vec![0.0f32, 1.0]; // current query
        let k_all = vec![
            1.0f32, 0.0, // token 0 key
            0.0f32, 1.0, // token 1 key
        ];
        let v_all = vec![
            1.0f32, 0.0, // token 0 value
            0.0f32, 1.0, // token 1 value
        ];
        let out = attention_kv(&q, &k_all, &v_all, 1);
        // q·k0 = 0, q·k1 = 1. After softmax, weights ≈ [0.27, 0.73].
        // out = 0.27*[1,0] + 0.73*[0,1] = [0.27, 0.73]
        assert_eq!(out.len(), 2);
        assert!(out[0] > 0.2 && out[0] < 0.4);
        assert!(out[1] > 0.6 && out[1] < 0.8);
    }

    #[test]
    fn kv_cache_appends_and_reuses() {
        let mut cache = KvCache::new(1);
        cache.layers[0] = Some((vec![1.0f32, 2.0], vec![3.0f32, 4.0]));

        // Simulate attention reading from cache and appending.
        let k = vec![5.0f32, 6.0];
        let v = vec![7.0f32, 8.0];
        if let Some((ref mut ck, ref mut cv)) = cache.layers[0] {
            let mut k_all = ck.clone();
            let mut v_all = cv.clone();
            k_all.extend_from_slice(&k);
            v_all.extend_from_slice(&v);
            *ck = k_all;
            *cv = v_all;
        }

        let (ck, cv) = cache.layers[0].as_ref().unwrap();
        assert_eq!(ck, &vec![1.0f32, 2.0, 5.0, 6.0]);
        assert_eq!(cv, &vec![3.0f32, 4.0, 7.0, 8.0]);
    }
}

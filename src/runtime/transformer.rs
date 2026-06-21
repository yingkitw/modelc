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
            eprintln!(
                "    llama block {layer}: {:.3} ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
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

/// A single layer's cached Key/Value vectors.
/// Can store either FP32, INT8-quantized data (4x memory reduction), or a
/// mixed-precision mode where recent tokens stay in FP32 and older tokens are
/// quantized to INT8 (combines accuracy for active context with memory savings).
#[derive(Clone)]
pub enum KvLayer {
    Fp32 {
        k: Vec<f32>,
        v: Vec<f32>,
        hidden: usize,
    },
    Int8 {
        k_q: Vec<i8>,
        v_q: Vec<i8>,
        k_scales: Vec<f32>,
        v_scales: Vec<f32>,
        hidden: usize,
    },
    /// Recent `window` tokens in FP32 ("hot"), older tokens in INT8 ("cold").
    Mixed {
        k_hot: Vec<f32>,
        v_hot: Vec<f32>,
        k_cold_q: Vec<i8>,
        v_cold_q: Vec<i8>,
        k_cold_scales: Vec<f32>,
        v_cold_scales: Vec<f32>,
        hidden: usize,
        window: usize,
    },
}

impl KvLayer {
    fn quantize_scale(xs: &[f32]) -> f32 {
        let max = xs.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max > 0.0 { max / 127.0 } else { 1.0 }
    }

    fn quantize(xs: &[f32], scale: f32) -> Vec<i8> {
        xs.iter()
            .map(|x| (*x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect()
    }

    pub fn new_fp32() -> Self {
        KvLayer::Fp32 {
            k: Vec::new(),
            v: Vec::new(),
            hidden: 0,
        }
    }

    pub fn new_int8(hidden: usize) -> Self {
        KvLayer::Int8 {
            k_q: Vec::new(),
            v_q: Vec::new(),
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            hidden,
        }
    }

    pub fn new_mixed(hidden: usize, window: usize) -> Self {
        KvLayer::Mixed {
            k_hot: Vec::new(),
            v_hot: Vec::new(),
            k_cold_q: Vec::new(),
            v_cold_q: Vec::new(),
            k_cold_scales: Vec::new(),
            v_cold_scales: Vec::new(),
            hidden,
            window,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            KvLayer::Fp32 { k, hidden, .. } => {
                if *hidden > 0 {
                    k.len() / hidden
                } else {
                    0
                }
            }
            KvLayer::Int8 { k_q, hidden, .. } => {
                if *hidden > 0 {
                    k_q.len() / hidden
                } else {
                    0
                }
            }
            KvLayer::Mixed {
                k_hot,
                k_cold_q,
                hidden,
                ..
            } => {
                if *hidden > 0 {
                    k_hot.len() / hidden + k_cold_q.len() / hidden
                } else {
                    0
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn k_all(&self) -> Vec<f32> {
        match self {
            KvLayer::Fp32 { k, .. } => k.clone(),
            KvLayer::Int8 {
                k_q,
                k_scales,
                hidden,
                ..
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let n = k_q.len() / h;
                let mut result = Vec::with_capacity(k_q.len());
                for i in 0..n {
                    let scale = k_scales[i];
                    for j in 0..*hidden {
                        result.push(k_q[i * hidden + j] as f32 * scale);
                    }
                }
                result
            }
            KvLayer::Mixed {
                k_hot,
                k_cold_q,
                k_cold_scales,
                hidden,
                ..
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let n = k_cold_q.len() / h;
                let mut result = Vec::with_capacity(k_hot.len() + k_cold_q.len());
                for i in 0..n {
                    let scale = k_cold_scales[i];
                    for j in 0..*hidden {
                        result.push(k_cold_q[i * hidden + j] as f32 * scale);
                    }
                }
                result.extend_from_slice(k_hot);
                result
            }
        }
    }

    pub fn v_all(&self) -> Vec<f32> {
        match self {
            KvLayer::Fp32 { v, .. } => v.clone(),
            KvLayer::Int8 {
                v_q,
                v_scales,
                hidden,
                ..
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let n = v_q.len() / h;
                let mut result = Vec::with_capacity(v_q.len());
                for i in 0..n {
                    let scale = v_scales[i];
                    for j in 0..*hidden {
                        result.push(v_q[i * hidden + j] as f32 * scale);
                    }
                }
                result
            }
            KvLayer::Mixed {
                v_hot,
                v_cold_q,
                v_cold_scales,
                hidden,
                ..
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let n = v_cold_q.len() / h;
                let mut result = Vec::with_capacity(v_hot.len() + v_cold_q.len());
                for i in 0..n {
                    let scale = v_cold_scales[i];
                    for j in 0..*hidden {
                        result.push(v_cold_q[i * hidden + j] as f32 * scale);
                    }
                }
                result.extend_from_slice(v_hot);
                result
            }
        }
    }

    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        match self {
            KvLayer::Fp32 {
                k: ck,
                v: cv,
                hidden,
            } => {
                if *hidden == 0 {
                    *hidden = k.len();
                }
                ck.extend_from_slice(k);
                cv.extend_from_slice(v);
            }
            KvLayer::Int8 {
                k_q,
                v_q,
                k_scales,
                v_scales,
                hidden,
            } => {
                *hidden = k.len();
                let k_scale = Self::quantize_scale(k);
                let v_scale = Self::quantize_scale(v);
                k_scales.push(k_scale);
                v_scales.push(v_scale);
                k_q.extend(Self::quantize(k, k_scale));
                v_q.extend(Self::quantize(v, v_scale));
            }
            KvLayer::Mixed {
                k_hot,
                v_hot,
                k_cold_q,
                v_cold_q,
                k_cold_scales,
                v_cold_scales,
                hidden,
                window,
            } => {
                if *hidden == 0 {
                    *hidden = k.len();
                }
                let h = *hidden;
                k_hot.extend_from_slice(k);
                v_hot.extend_from_slice(v);
                // If hot buffer exceeds window, move oldest tokens to cold.
                let hot_tokens = k_hot.len() / h;
                if hot_tokens > *window {
                    let to_move = hot_tokens - *window;
                    let move_elements = to_move * h;
                    for i in 0..to_move {
                        let slice_k = &k_hot[i * h..(i + 1) * h];
                        let slice_v = &v_hot[i * h..(i + 1) * h];
                        let k_scale = Self::quantize_scale(slice_k);
                        let v_scale = Self::quantize_scale(slice_v);
                        k_cold_scales.push(k_scale);
                        v_cold_scales.push(v_scale);
                        k_cold_q.extend(Self::quantize(slice_k, k_scale));
                        v_cold_q.extend(Self::quantize(slice_v, v_scale));
                    }
                    *k_hot = k_hot.split_off(move_elements);
                    *v_hot = v_hot.split_off(move_elements);
                }
            }
        }
    }

    /// Remove the oldest `n` tokens from the cache, shifting remaining tokens left.
    /// If `n` >= current length, the cache is cleared.
    pub fn shift(&mut self, n: usize) {
        match self {
            KvLayer::Fp32 { k, v, hidden } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let keep = k.len() / h;
                let remove = n.min(keep);
                let remove_elements = remove * h;
                *k = k.split_off(remove_elements);
                *v = v.split_off(remove_elements);
            }
            KvLayer::Int8 {
                k_q,
                v_q,
                k_scales,
                v_scales,
                hidden,
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let keep = k_q.len() / h;
                let remove = n.min(keep);
                let remove_elements = remove * h;
                *k_q = k_q.split_off(remove_elements);
                *v_q = v_q.split_off(remove_elements);
                let scale_remove = k_scales.len().min(remove);
                *k_scales = k_scales.split_off(scale_remove);
                *v_scales = v_scales.split_off(scale_remove);
            }
            KvLayer::Mixed {
                k_hot,
                v_hot,
                k_cold_q,
                v_cold_q,
                k_cold_scales,
                v_cold_scales,
                hidden,
                ..
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let cold_tokens = k_cold_q.len() / h;
                let hot_tokens = k_hot.len() / h;
                let total = cold_tokens + hot_tokens;
                let remove = n.min(total);

                if remove <= cold_tokens {
                    // Remove only from cold.
                    let remove_elements = remove * h;
                    *k_cold_q = k_cold_q.split_off(remove_elements);
                    *v_cold_q = v_cold_q.split_off(remove_elements);
                    let scale_remove = k_cold_scales.len().min(remove);
                    *k_cold_scales = k_cold_scales.split_off(scale_remove);
                    *v_cold_scales = v_cold_scales.split_off(scale_remove);
                } else {
                    // Remove all cold, then some from hot.
                    k_cold_q.clear();
                    v_cold_q.clear();
                    k_cold_scales.clear();
                    v_cold_scales.clear();
                    let hot_remove = remove - cold_tokens;
                    let remove_elements = hot_remove * h;
                    *k_hot = k_hot.split_off(remove_elements);
                    *v_hot = v_hot.split_off(remove_elements);
                }
            }
        }
    }

    /// Remove `n` tokens from the middle of the cache, preserving the first `anchor`
    /// tokens at the beginning. This implements StreamingLLM-style eviction where
    /// "attention sink" anchor tokens (e.g., the first 4 tokens of a prompt) are
    /// retained while older non-anchor tokens are evicted to make room for new ones.
    ///
    /// If `anchor` is 0 or there are not enough non-anchor tokens to remove,
    /// falls back to regular `shift(n)`.
    pub fn shift_anchored(&mut self, n: usize, anchor: usize) {
        if anchor == 0 || n == 0 {
            self.shift(n);
            return;
        }
        let len = self.len();
        if n >= len {
            self.shift(n);
            return;
        }
        let keep = len - n;
        if keep <= anchor {
            self.shift(n);
            return;
        }

        let h = match self {
            KvLayer::Fp32 { hidden, .. } => *hidden,
            KvLayer::Int8 { hidden, .. } => *hidden,
            KvLayer::Mixed { hidden, .. } => *hidden,
        };

        let full_k = self.k_all();
        let full_v = self.v_all();

        let prefix_k = &full_k[..anchor * h];
        let prefix_v = &full_v[..anchor * h];
        let suffix_k = &full_k[(anchor + n) * h..];
        let suffix_v = &full_v[(anchor + n) * h..];

        // Clear current state while preserving variant and window.
        match self {
            KvLayer::Fp32 { k, v, hidden } => {
                k.clear();
                v.clear();
                *hidden = h;
            }
            KvLayer::Int8 {
                k_q,
                v_q,
                k_scales,
                v_scales,
                hidden,
            } => {
                k_q.clear();
                v_q.clear();
                k_scales.clear();
                v_scales.clear();
                *hidden = h;
            }
            KvLayer::Mixed {
                k_hot,
                v_hot,
                k_cold_q,
                v_cold_q,
                k_cold_scales,
                v_cold_scales,
                hidden,
                window,
            } => {
                let win = *window;
                k_hot.clear();
                v_hot.clear();
                k_cold_q.clear();
                v_cold_q.clear();
                k_cold_scales.clear();
                v_cold_scales.clear();
                *hidden = h;
                *window = win;
            }
        }

        // Re-append anchor tokens.
        for i in 0..anchor {
            let s = i * h;
            let e = s + h;
            self.append(&prefix_k[s..e], &prefix_v[s..e]);
        }

        // Re-append remaining suffix tokens.
        let suffix_tokens = keep - anchor;
        for i in 0..suffix_tokens {
            let s = i * h;
            let e = s + h;
            self.append(&suffix_k[s..e], &suffix_v[s..e]);
        }
    }
}

/// A cache of Key/Value vectors for each transformer layer.
#[derive(Clone)]
pub struct KvCache {
    pub layers: Vec<Option<KvLayer>>,
}

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: vec![None; n_layers],
        }
    }

    pub fn new_quantized(n_layers: usize, hidden: usize, use_int8: bool) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(Some(if use_int8 {
                KvLayer::new_int8(hidden)
            } else {
                KvLayer::new_fp32()
            }));
        }
        Self { layers }
    }

    pub fn new_mixed(n_layers: usize, hidden: usize, window: usize) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(Some(KvLayer::new_mixed(hidden, window)));
        }
        Self { layers }
    }

    /// Remove the oldest `n` tokens from every layer in the cache.
    pub fn shift(&mut self, n: usize) {
        for l in self.layers.iter_mut().flatten() {
            l.shift(n);
        }
    }

    /// Remove `n` tokens from the middle of every layer, preserving the first `anchor` tokens.
    /// See [`KvLayer::shift_anchored`] for details.
    pub fn shift_anchored(&mut self, n: usize, anchor: usize) {
        for l in self.layers.iter_mut().flatten() {
            l.shift_anchored(n, anchor);
        }
    }

    /// Total number of cached tokens (assumes all layers are in sync).
    pub fn len(&self) -> usize {
        self.layers
            .first()
            .and_then(|l| l.as_ref().map(|layer| layer.len()))
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    kv_cache: Option<&mut Option<KvLayer>>,
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
    kv_cache: &mut Option<KvLayer>,
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

    let (k_all, v_all) = if let Some(layer) = kv_cache {
        layer.append(k, v);
        (layer.k_all(), layer.v_all())
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
    kv_cache: Option<&mut Option<KvLayer>>,
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
    kv_cache: &mut Option<KvLayer>,
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

    let pos = kv_cache.as_ref().map(|layer| layer.len()).unwrap_or(0);
    let q = rope(&q, pos, n_heads);
    let k = rope(&k, pos, n_heads);

    let (k_all, v_all) = if let Some(layer) = kv_cache {
        layer.append(&k, &v);
        (layer.k_all(), layer.v_all())
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
    // Pre-allocate one scratch buffer reused across heads — eliminates
    // per-head Vec allocations (the dominant allocation hotspot).
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..n_heads {
        let q_off = h * head_dim;
        let q_head = &q[q_off..q_off + head_dim];

        for (t, score) in scores.iter_mut().enumerate() {
            let k_off = t * n_heads * head_dim + h * head_dim;
            let k_head = &k_all[k_off..k_off + head_dim];
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q_head[i] * k_head[i];
            }
            *score = dot / (head_dim as f32).sqrt();
        }

        softmax_inplace(&mut scores);

        let out_off = h * head_dim;
        for i in 0..head_dim {
            out[out_off + i] = 0.0;
        }
        for (t, w) in scores.iter().enumerate() {
            let v_off = t * n_heads * head_dim + h * head_dim;
            let v_head = &v_all[v_off..v_off + head_dim];
            for i in 0..head_dim {
                out[out_off + i] += w * v_head[i];
            }
        }
    }

    out
}

/// In-place softmax: replaces `xs` with normalized probabilities. Zero allocations.
fn softmax_inplace(xs: &mut [f32]) {
    if xs.is_empty() {
        return;
    }
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in xs.iter_mut() {
            *x /= sum;
        }
    }
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
        cache.layers[0] = Some(KvLayer::new_fp32());
        cache.layers[0]
            .as_mut()
            .unwrap()
            .append(&[1.0f32, 2.0], &[3.0f32, 4.0]);

        // Simulate attention reading from cache and appending.
        cache.layers[0]
            .as_mut()
            .unwrap()
            .append(&[5.0f32, 6.0], &[7.0f32, 8.0]);

        let layer = cache.layers[0].as_ref().unwrap();
        assert_eq!(layer.k_all(), vec![1.0f32, 2.0, 5.0, 6.0]);
        assert_eq!(layer.v_all(), vec![3.0f32, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn kv_layer_int8_roundtrip() {
        let mut layer = KvLayer::new_int8(2);
        let k = vec![0.5f32, -0.3];
        let v = vec![1.2f32, -0.8];
        layer.append(&k, &v);

        let k_all = layer.k_all();
        let v_all = layer.v_all();

        // INT8 quantization is lossy; verify within ~2% relative error.
        for (expected, actual) in k.iter().zip(k_all.iter()) {
            assert!(
                (expected - actual).abs() < 0.02,
                "k mismatch: expected {expected}, got {actual}"
            );
        }
        for (expected, actual) in v.iter().zip(v_all.iter()) {
            assert!(
                (expected - actual).abs() < 0.02,
                "v mismatch: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn kv_layer_int8_multiple_tokens() {
        let mut layer = KvLayer::new_int8(2);
        layer.append(&[1.0f32, 0.0], &[0.0f32, 1.0]);
        layer.append(&[0.5f32, 0.5], &[0.5f32, 0.5]);

        assert_eq!(layer.len(), 2);
        let k_all = layer.k_all();
        let v_all = layer.v_all();
        assert_eq!(k_all.len(), 4);
        assert_eq!(v_all.len(), 4);
    }

    #[test]
    fn kv_layer_fp32_shift_discards_oldest() {
        let mut layer = KvLayer::new_fp32();
        layer.append(&[1.0f32, 2.0], &[3.0f32, 4.0]);
        layer.append(&[5.0f32, 6.0], &[7.0f32, 8.0]);
        layer.append(&[9.0f32, 10.0], &[11.0f32, 12.0]);

        layer.shift(1);
        assert_eq!(layer.len(), 2);
        assert_eq!(layer.k_all(), vec![5.0f32, 6.0, 9.0, 10.0]);
        assert_eq!(layer.v_all(), vec![7.0f32, 8.0, 11.0, 12.0]);
    }

    #[test]
    fn kv_layer_int8_shift_discards_oldest() {
        let mut layer = KvLayer::new_int8(2);
        layer.append(&[1.0f32, 0.0], &[0.0f32, 1.0]);
        layer.append(&[0.5f32, 0.5], &[0.5f32, 0.5]);
        layer.append(&[0.0f32, 1.0], &[1.0f32, 0.0]);

        layer.shift(2);
        assert_eq!(layer.len(), 1);
        let k_all = layer.k_all();
        let v_all = layer.v_all();
        // Last token only
        assert_eq!(k_all.len(), 2);
        assert_eq!(v_all.len(), 2);
    }

    #[test]
    fn kv_cache_shift_syncs_all_layers() {
        let mut cache = KvCache::new(2);
        cache.layers[0] = Some(KvLayer::new_fp32());
        cache.layers[1] = Some(KvLayer::new_fp32());
        cache.layers[0].as_mut().unwrap().append(&[1.0f32, 2.0], &[3.0f32, 4.0]);
        cache.layers[0].as_mut().unwrap().append(&[5.0f32, 6.0], &[7.0f32, 8.0]);
        cache.layers[1].as_mut().unwrap().append(&[9.0f32, 10.0], &[11.0f32, 12.0]);
        cache.layers[1].as_mut().unwrap().append(&[13.0f32, 14.0], &[15.0f32, 16.0]);

        cache.shift(1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.layers[0].as_ref().unwrap().k_all(), vec![5.0f32, 6.0]);
        assert_eq!(cache.layers[1].as_ref().unwrap().k_all(), vec![13.0f32, 14.0]);
    }

    #[test]
    fn kv_layer_mixed_keeps_recent_in_fp32() {
        let mut layer = KvLayer::new_mixed(2, 2);
        layer.append(&[1.0f32, 0.0], &[0.0f32, 1.0]);
        layer.append(&[0.5f32, 0.5], &[0.5f32, 0.5]);
        layer.append(&[0.0f32, 1.0], &[1.0f32, 0.0]);

        // Window=2, so 1 token should have been moved to cold (INT8).
        assert_eq!(layer.len(), 3);
        let k_all = layer.k_all();
        let v_all = layer.v_all();
        // Order should be preserved: oldest first, then newer.
        assert_eq!(k_all.len(), 6);
        assert_eq!(v_all.len(), 6);
        // First token (cold) within INT8 quantization error.
        assert!((k_all[0] - 1.0).abs() < 0.02);
        assert!((k_all[1] - 0.0).abs() < 0.02);
        // Recent tokens (hot) exact.
        assert_eq!(k_all[2..], vec![0.5f32, 0.5, 0.0, 1.0]);
    }

    #[test]
    fn kv_layer_mixed_matches_fp32_output() {
        let mut mixed = KvLayer::new_mixed(4, 3);
        let mut fp32 = KvLayer::new_fp32();
        for i in 0..8 {
            let k = vec![i as f32; 4];
            let v = vec![(i + 1) as f32; 4];
            mixed.append(&k, &v);
            fp32.append(&k, &v);
        }
        // After 8 appends with window=3, 5 tokens should be in cold, 3 in hot.
        assert_eq!(mixed.len(), 8);
        let mixed_k = mixed.k_all();
        let fp32_k = fp32.k_all();
        // Cold tokens have INT8 quantization error; verify roughly equal.
        assert_eq!(mixed_k.len(), fp32_k.len());
        for (a, b) in mixed_k.iter().zip(fp32_k.iter()) {
            assert!((a - b).abs() < 0.05, "mixed k mismatch: expected {b}, got {a}");
        }
    }

    #[test]
    fn kv_layer_mixed_shift_discards_oldest() {
        let mut layer = KvLayer::new_mixed(2, 2);
        layer.append(&[1.0f32, 0.0], &[0.0f32, 1.0]);
        layer.append(&[0.5f32, 0.5], &[0.5f32, 0.5]);
        layer.append(&[0.0f32, 1.0], &[1.0f32, 0.0]);
        layer.append(&[2.0f32, 2.0], &[3.0f32, 3.0]);

        layer.shift(2);
        assert_eq!(layer.len(), 2);
        let k_all = layer.k_all();
        // Remaining should be the two newest tokens.
        assert_eq!(k_all, vec![0.0f32, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn kv_layer_fp32_shift_anchored_preserves_first() {
        let mut layer = KvLayer::new_fp32();
        for i in 0..6 {
            layer.append(&[i as f32, i as f32 + 0.5], &[i as f32 + 10.0, i as f32 + 10.5]);
        }
        // 6 tokens, remove 2 with anchor=2: keep [0,1] + [4,5]
        layer.shift_anchored(2, 2);
        assert_eq!(layer.len(), 4);
        let k = layer.k_all();
        assert_eq!(k, vec![0.0f32, 0.5, 1.0, 1.5, 4.0, 4.5, 5.0, 5.5]);
    }

    #[test]
    fn kv_layer_int8_shift_anchored_preserves_first() {
        let mut layer = KvLayer::new_int8(2);
        for i in 0..6 {
            layer.append(&[i as f32, i as f32 + 0.5], &[i as f32 + 10.0, i as f32 + 10.5]);
        }
        layer.shift_anchored(2, 2);
        assert_eq!(layer.len(), 4);
        let k = layer.k_all();
        // Within INT8 quantization error.
        assert!((k[0] - 0.0).abs() < 0.02);
        assert!((k[2] - 1.0).abs() < 0.02);
        assert!((k[4] - 4.0).abs() < 0.02);
        assert!((k[6] - 5.0).abs() < 0.02);
    }

    #[test]
    fn kv_layer_mixed_shift_anchored_preserves_first() {
        let mut layer = KvLayer::new_mixed(2, 3);
        for i in 0..8 {
            layer.append(&[i as f32, i as f32 + 0.5], &[i as f32 + 10.0, i as f32 + 10.5]);
        }
        // 8 tokens, remove 3 with anchor=2: keep [0,1] + [5,6,7]
        layer.shift_anchored(3, 2);
        assert_eq!(layer.len(), 5);
        let k = layer.k_all();
        // First two are anchors.
        assert!((k[0] - 0.0).abs() < 0.02);
        assert!((k[2] - 1.0).abs() < 0.02);
        // Last three are the suffix.
        assert!((k[4] - 5.0).abs() < 0.02);
        assert!((k[6] - 6.0).abs() < 0.02);
        assert!((k[8] - 7.0).abs() < 0.02);
    }

    #[test]
    fn kv_layer_shift_anchored_falls_back_when_not_enough_tokens() {
        let mut layer = KvLayer::new_fp32();
        layer.append(&[1.0f32, 2.0], &[3.0f32, 4.0]);
        layer.append(&[5.0f32, 6.0], &[7.0f32, 8.0]);
        // 2 tokens, anchor=2, remove=1: not enough non-anchor tokens, falls back to shift.
        layer.shift_anchored(1, 2);
        assert_eq!(layer.len(), 1);
        assert_eq!(layer.k_all(), vec![5.0f32, 6.0]);
    }

    #[test]
    fn softmax_inplace_matches_softmax() {
        let xs = vec![1.0f32, 2.0, 3.0, 4.0];
        let expected = crate::generate::softmax(&xs);
        let mut actual = xs.clone();
        softmax_inplace(&mut actual);
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "softmax_inplace mismatch: expected {e}, got {a}");
        }
    }

    #[test]
    fn attention_kv_multi_head_no_alloc_bloat() {
        // Regression test: attention_kv should not allocate per-head buffers.
        // The optimization is structural (pre-allocated scratch), so we just
        // verify correctness for a multi-head, multi-token case.
        let n_heads = 4;
        let head_dim = 8;
        let seq_len = 16;
        let q: Vec<f32> = (0..(n_heads * head_dim)).map(|i| (i as f32) * 0.1).collect();
        let k_all: Vec<f32> = (0..(seq_len * n_heads * head_dim)).map(|i| (i as f32) * 0.05).collect();
        let v_all: Vec<f32> = (0..(seq_len * n_heads * head_dim)).map(|i| (i as f32) * 0.03).collect();

        let out = attention_kv(&q, &k_all, &v_all, n_heads);
        assert_eq!(out.len(), n_heads * head_dim);
        // Each head's output should be a weighted sum of its V vectors,
        // so values should be finite and non-zero.
        assert!(out.iter().all(|&v| v.is_finite()));
    }
}

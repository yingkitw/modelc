//! Shared model-structure inspection for transformer architectures.
//!
//! Both the codegen emitter ([`crate::codegen::native::forward`]) and the runtime
//! transformer forward ([`crate::runtime::transformer`]) use these helpers so the two
//! inference paths agree on layer counts, hidden dimensions, and head counts. Keeping
//! the inspection logic in one place prevents drift between `modelc compile` and
//! `modelc run`.

use crate::model::{DataType, Model, TensorData};

/// Sorted, deduplicated transformer-block indices discovered under `prefix`
/// (e.g. `"transformer.h."` → indices parsed from `transformer.h.{i}.*`).
pub fn detect_layers(model: &Model, prefix: &str) -> Vec<u32> {
    let mut ids: Vec<u32> = model
        .tensors
        .keys()
        .filter_map(|k| {
            let tail = k.strip_prefix(prefix)?;
            let (idx, _) = tail.split_once('.')?;
            idx.parse::<u32>().ok()
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// LLaMA-style layers can live under either `model.layers.` or `layers.`; merge both.
pub fn llama_layers(model: &Model) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    for prefix in ["model.layers.", "layers."] {
        for id in detect_layers(model, prefix) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// True when `weight_key` is an F32 matrix and `bias_key` is present (GPT-2 LN/attn pair check).
pub fn has_pair(model: &Model, weight_key: &str, bias_key: &str) -> bool {
    let w = match model.tensors.get(weight_key) {
        Some(t) => t,
        None => return false,
    };
    model.tensors.contains_key(bias_key) && w.dtype == DataType::F32
}

pub fn tensor_rows(tensor: &TensorData) -> Option<usize> {
    tensor.shape.first().copied()
}

pub fn tensor_cols(tensor: &TensorData) -> Option<usize> {
    tensor.shape.get(1).copied()
}

/// GPT-2 hidden size inferred from the first block's `c_attn`/`ln_1`, falling back to the
/// embedding / `lm_head` column count.
pub fn gpt2_hidden_dim(model: &Model, layers: &[u32]) -> usize {
    if let Some(&l0) = layers.first() {
        let p = format!("transformer.h.{l0}.");
        for (w, b) in [
            (
                format!("{p}attn.c_attn.weight"),
                format!("{p}attn.c_attn.bias"),
            ),
            (format!("{p}ln_1.weight"), format!("{p}ln_1.bias")),
        ] {
            if let Some(t) = model.tensors.get(&w).or_else(|| model.tensors.get(&b))
                && let Some(cols) = tensor_cols(t).or_else(|| tensor_rows(t))
            {
                // c_attn is [3H, H] → use cols; ln_1 is [H] → use rows.
                if w.ends_with("c_attn.weight")
                    && let Some(c) = tensor_cols(t)
                {
                    return c;
                }
                return cols;
            }
        }
    }
    model
        .tensors
        .get("transformer.wte.weight")
        .and_then(tensor_cols)
        .or_else(|| model.tensors.get("lm_head.weight").and_then(tensor_cols))
        .unwrap_or(0)
}

/// GPT-2 convention: `head_dim = 64` → `n_heads = hidden / 64`.
pub fn gpt2_head_count(hidden: usize) -> usize {
    if hidden == 0 { 1 } else { (hidden / 64).max(1) }
}

/// LLaMA hidden size inferred from the first block's `q_proj`/`input_layernorm`, falling
/// back to the embedding / `lm_head` column count.
pub fn llama_hidden_dim(model: &Model, layers: &[u32]) -> usize {
    if let Some(&l0) = layers.first() {
        let p = format!("model.layers.{l0}.");
        for key in [
            format!("{p}self_attn.q_proj.weight"),
            format!("{p}input_layernorm.weight"),
        ] {
            if let Some(t) = model.tensors.get(&key) {
                if key.ends_with("q_proj.weight")
                    && let Some(c) = tensor_cols(t)
                {
                    return c;
                }
                if let Some(r) = tensor_rows(t) {
                    return r;
                }
            }
        }
    }
    model
        .tensors
        .get("model.embed_tokens.weight")
        .and_then(tensor_cols)
        .or_else(|| model.tensors.get("lm_head.weight").and_then(tensor_cols))
        .unwrap_or(0)
}

/// Prefer an explicit `attention.head_count` metadata hint; otherwise default to hidden/64.
pub fn llama_head_count(model: &Model, hidden: usize) -> usize {
    if let Some(h) = model
        .metadata
        .get("attention.head_count")
        .and_then(|s| s.parse::<usize>().ok())
    {
        return h.max(1);
    }
    if hidden == 0 { 1 } else { (hidden / 64).max(1) }
}

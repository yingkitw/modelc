//! Shared model-structure inspection for transformer architectures.
//!
//! Both the codegen emitter ([`crate::codegen::native::NativeCodegen`]) and the runtime
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
            if let Some(t) = model.tensors.get(&w).or_else(|| model.tensors.get(&b)) {
                if let Some(cols) = tensor_cols(t).or_else(|| tensor_rows(t)) {
                    // c_attn is [3H, H] → use cols; ln_1 is [H] → use rows.
                    if w.ends_with("c_attn.weight") {
                        if let Some(c) = tensor_cols(t) {
                            return c;
                        }
                    }
                    return cols;
                }
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
                if key.ends_with("q_proj.weight") {
                    if let Some(c) = tensor_cols(t) {
                        return c;
                    }
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

/// Hidden size for GPT-2 / LLaMA artifacts, or `None` when the architecture is not a
/// transformer or the size cannot be inferred. Shared by `modelc run` (HTTP) and
/// `modelc prompt` (one-shot CLI) so both paths size the KV cache the same way.
pub fn transformer_hidden_dim(model: &Model) -> Option<usize> {
    let hidden = match model.architecture.as_str() {
        "gpt2" => {
            let layers = detect_layers(model, "transformer.h.");
            gpt2_hidden_dim(model, &layers)
        }
        "llama" => {
            let layers = llama_layers(model);
            llama_hidden_dim(model, &layers)
        }
        _ => return None,
    };
    (hidden > 0).then_some(hidden)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn td(shape: Vec<usize>, dtype: DataType) -> TensorData {
        let n: usize = shape.iter().product();
        TensorData {
            shape,
            dtype,
            data: vec![0u8; n * dtype.byte_size()],
        }
    }

    fn model(arch: &str, tensors: Vec<(&str, TensorData)>, metadata: Vec<(&str, &str)>) -> Model {
        Model {
            name: "t".into(),
            architecture: arch.to_string(),
            tensors: tensors
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            metadata: metadata
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    #[test]
    fn detect_layers_parses_indices_and_skips_invalid() {
        let m = model(
            "gpt2",
            vec![
                ("transformer.h.0.a", td(vec![1], DataType::F32)),
                ("transformer.h.2.a", td(vec![1], DataType::F32)),
                ("transformer.h.x.a", td(vec![1], DataType::F32)),
                ("transformer.h.", td(vec![1], DataType::F32)),
            ],
            vec![],
        );
        assert_eq!(detect_layers(&m, "transformer.h."), vec![0, 2]);
    }

    #[test]
    fn llama_layers_merges_and_dedups_prefixes() {
        let m = model(
            "llama",
            vec![
                ("model.layers.0.mlp", td(vec![1], DataType::F32)),
                ("layers.1.mlp", td(vec![1], DataType::F32)),
                ("layers.0.mlp", td(vec![1], DataType::F32)),
            ],
            vec![],
        );
        assert_eq!(llama_layers(&m), vec![0, 1]);
    }

    #[test]
    fn has_pair_requires_f32_weight_and_bias() {
        let m = model(
            "gpt2",
            vec![
                ("w", td(vec![2], DataType::F32)),
                ("b", td(vec![2], DataType::F32)),
                ("i8w", td(vec![2], DataType::I8)),
                ("i8b", td(vec![2], DataType::F32)),
                ("f32w", td(vec![2], DataType::F32)),
            ],
            vec![],
        );
        assert!(has_pair(&m, "w", "b"));
        assert!(!has_pair(&m, "missing", "b"), "absent weight → false");
        assert!(!has_pair(&m, "w", "missing"), "absent bias → false");
        assert!(!has_pair(&m, "i8w", "i8b"), "non-F32 weight → false");
        assert!(has_pair(&m, "f32w", "b"));
    }

    #[test]
    fn gpt2_hidden_from_c_attn_columns() {
        let m = model(
            "gpt2",
            vec![
                ("transformer.h.0.attn.c_attn.weight", td(vec![96, 32], DataType::F32)),
                ("transformer.h.0.attn.c_attn.bias", td(vec![96], DataType::F32)),
            ],
            vec![],
        );
        let layers = detect_layers(&m, "transformer.h.");
        assert_eq!(gpt2_hidden_dim(&m, &layers), 32);
    }

    #[test]
    fn gpt2_hidden_from_ln1_rows_when_c_attn_missing() {
        let m = model(
            "gpt2",
            vec![
                ("transformer.h.0.ln_1.weight", td(vec![16], DataType::F32)),
                ("transformer.h.0.ln_1.bias", td(vec![16], DataType::F32)),
            ],
            vec![],
        );
        let layers = detect_layers(&m, "transformer.h.");
        assert_eq!(gpt2_hidden_dim(&m, &layers), 16);
    }

    #[test]
    fn gpt2_hidden_falls_back_to_wte_then_lm_head() {
        let wte = model(
            "gpt2",
            vec![("transformer.wte.weight", td(vec![50257, 768], DataType::F32))],
            vec![],
        );
        assert_eq!(gpt2_hidden_dim(&wte, &[]), 768);
        let lm_head = model(
            "gpt2",
            vec![("lm_head.weight", td(vec![50257, 768], DataType::F32))],
            vec![],
        );
        assert_eq!(gpt2_hidden_dim(&lm_head, &[]), 768);
        let empty = model("gpt2", vec![], vec![]);
        assert_eq!(gpt2_hidden_dim(&empty, &[]), 0);
    }

    #[test]
    fn llama_hidden_from_q_proj_columns() {
        let m = model(
            "llama",
            vec![
                ("model.layers.0.self_attn.q_proj.weight", td(vec![64, 32], DataType::F32)),
                ("model.layers.0.input_layernorm.weight", td(vec![32], DataType::F32)),
            ],
            vec![],
        );
        let layers = llama_layers(&m);
        assert_eq!(llama_hidden_dim(&m, &layers), 32);
    }

    #[test]
    fn llama_hidden_from_layernorm_rows_when_q_proj_missing() {
        let m = model(
            "llama",
            vec![("model.layers.0.input_layernorm.weight", td(vec![48], DataType::F32))],
            vec![],
        );
        let layers = llama_layers(&m);
        assert_eq!(llama_hidden_dim(&m, &layers), 48);
    }

    #[test]
    fn llama_hidden_falls_back_to_embeddings_then_lm_head() {
        let embed = model(
            "llama",
            vec![("model.embed_tokens.weight", td(vec![32000, 4096], DataType::F32))],
            vec![],
        );
        assert_eq!(llama_hidden_dim(&embed, &[]), 4096);
        let lm_head = model(
            "llama",
            vec![("lm_head.weight", td(vec![32000, 4096], DataType::F32))],
            vec![],
        );
        assert_eq!(llama_hidden_dim(&lm_head, &[]), 4096);
        let empty = model("llama", vec![], vec![]);
        assert_eq!(llama_hidden_dim(&empty, &[]), 0);
    }

    #[test]
    fn transformer_hidden_dim_selects_by_architecture() {
        assert_eq!(transformer_hidden_dim(&model("bert", vec![], vec![])), None);
        assert_eq!(transformer_hidden_dim(&model("gpt2", vec![], vec![])), None);
        assert_eq!(transformer_hidden_dim(&model("llama", vec![], vec![])), None);
        let gpt2 = model(
            "gpt2",
            vec![("transformer.h.0.attn.c_attn.weight", td(vec![96, 32], DataType::F32))],
            vec![],
        );
        assert_eq!(transformer_hidden_dim(&gpt2), Some(32));
    }

    #[test]
    fn llama_head_count_prefers_metadata_hint() {
        let hinted = model("llama", vec![], vec![("attention.head_count", "8")]);
        assert_eq!(llama_head_count(&hinted, 4096), 8);
        let zero_hint = model("llama", vec![], vec![("attention.head_count", "0")]);
        assert_eq!(llama_head_count(&zero_hint, 4096), 1, "hint clamps to ≥ 1");
        let invalid = model("llama", vec![], vec![("attention.head_count", "abc")]);
        assert_eq!(llama_head_count(&invalid, 128), 2, "unparseable hint → fallback");
        let none = model("llama", vec![], vec![]);
        assert_eq!(llama_head_count(&none, 128), 2);
        assert_eq!(llama_head_count(&none, 0), 1);
    }

    #[test]
    fn gpt2_head_count_is_hidden_per_64() {
        assert_eq!(gpt2_head_count(0), 1);
        assert_eq!(gpt2_head_count(128), 2);
        assert_eq!(gpt2_head_count(10), 1);
    }
}

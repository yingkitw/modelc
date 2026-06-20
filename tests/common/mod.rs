//! Shared fixtures for integration tests; not every helper is referenced by each test binary.

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use modelc::model::{DataType, Model, TensorData};

pub fn create_test_model() -> Model {
    let mut tensors = HashMap::new();

    let weight_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    tensors.insert(
        "weight".to_string(),
        TensorData {
            shape: vec![2, 3],
            dtype: DataType::F32,
            data: weight_data,
        },
    );

    let bias_data: Vec<u8> = [0.1f32, 0.2].iter().flat_map(|f| f.to_le_bytes()).collect();
    tensors.insert(
        "bias".to_string(),
        TensorData {
            shape: vec![2],
            dtype: DataType::F32,
            data: bias_data,
        },
    );

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "test".to_string());

    Model {
        name: "test_model".to_string(),
        architecture: "mlp".to_string(),
        tensors,
        metadata,
    }
}

pub fn create_large_test_model() -> Model {
    let mut tensors = HashMap::new();

    let hidden_dim = 64usize;
    let weight_data: Vec<u8> = (0..hidden_dim * hidden_dim)
        .flat_map(|i| (i as f32 / 100.0).to_le_bytes())
        .collect();
    tensors.insert(
        "layer0.weight".to_string(),
        TensorData {
            shape: vec![hidden_dim, hidden_dim],
            dtype: DataType::F32,
            data: weight_data,
        },
    );

    let bias_data: Vec<u8> = (0..hidden_dim)
        .flat_map(|i| (i as f32 / 100.0).to_le_bytes())
        .collect();
    tensors.insert(
        "layer0.bias".to_string(),
        TensorData {
            shape: vec![hidden_dim],
            dtype: DataType::F32,
            data: bias_data,
        },
    );

    let ln_weight: Vec<u8> = (0..hidden_dim).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    tensors.insert(
        "layer0.ln_weight".to_string(),
        TensorData {
            shape: vec![hidden_dim],
            dtype: DataType::F32,
            data: ln_weight,
        },
    );

    let ln_bias: Vec<u8> = (0..hidden_dim).flat_map(|_| 0.0f32.to_le_bytes()).collect();
    tensors.insert(
        "layer0.ln_bias".to_string(),
        TensorData {
            shape: vec![hidden_dim],
            dtype: DataType::F32,
            data: ln_bias,
        },
    );

    tensors.insert(
        "layer1.weight".to_string(),
        TensorData {
            shape: vec![hidden_dim, hidden_dim],
            dtype: DataType::F32,
            data: vec![0u8; hidden_dim * hidden_dim * 4],
        },
    );

    tensors.insert(
        "layer1.bias".to_string(),
        TensorData {
            shape: vec![hidden_dim],
            dtype: DataType::F32,
            data: vec![0u8; hidden_dim * 4],
        },
    );

    Model {
        name: "large_test_model".to_string(),
        architecture: "mlp".to_string(),
        tensors,
        metadata: HashMap::new(),
    }
}

pub fn create_safetensors_file(path: &Path, tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)>) {
    let mut sorted: Vec<_> = tensors.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));

    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_string(),
        serde_json::Value::Object(serde_json::Map::new()),
    );

    let mut offset = 0usize;
    let mut entries = Vec::new();
    for (name, dtype, shape, data) in sorted {
        let end = offset + data.len();
        entries.push((
            name.to_string(),
            dtype.to_string(),
            shape,
            data,
            offset,
            end,
        ));
        offset = end;
    }

    for (name, dtype, shape, _, start, end) in &entries {
        let mut obj = serde_json::Map::new();
        obj.insert(
            "dtype".to_string(),
            serde_json::Value::String(dtype.clone()),
        );
        obj.insert("shape".to_string(), serde_json::json!(shape));
        obj.insert(
            "data_offsets".to_string(),
            serde_json::json!([*start, *end]),
        );
        header.insert(name.clone(), serde_json::Value::Object(obj));
    }

    let header_json = serde_json::to_string(&header).unwrap();
    let header_bytes = header_json.as_bytes();

    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header_bytes).unwrap();
    for (_, _, _, data, _, _) in &entries {
        file.write_all(data).unwrap();
    }
}

pub fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

const GPT2_HIDDEN: usize = 12;
const GPT2_FFN: usize = 48;
const GPT2_VOCAB: usize = 10;

fn fp32_matrix(rows: usize, cols: usize) -> Vec<u8> {
    (0..(rows * cols))
        .flat_map(|i| ((i as f32 % 7.0) - 3.0).to_le_bytes())
        .collect()
}

fn fp32_vector(len: usize, fill: f32) -> Vec<u8> {
    (0..len).flat_map(|_| fill.to_le_bytes()).collect()
}

/// Minimal but complete single-layer GPT-2 fixture with tied `wte`/`lm_head`.
pub fn create_gpt2_test_model() -> Model {
    let h = GPT2_HIDDEN;
    let mut tensors = HashMap::new();
    let mut mk = |name: &str, shape: Vec<usize>, data: Vec<u8>| {
        tensors.insert(
            name.to_string(),
            TensorData {
                shape,
                dtype: DataType::F32,
                data,
            },
        );
    };

    mk("transformer.h.0.ln_1.weight", vec![h], fp32_vector(h, 1.0));
    mk("transformer.h.0.ln_1.bias", vec![h], fp32_vector(h, 0.0));
    mk(
        "transformer.h.0.attn.c_attn.weight",
        vec![3 * h, h],
        fp32_matrix(3 * h, h),
    );
    mk(
        "transformer.h.0.attn.c_attn.bias",
        vec![3 * h],
        fp32_vector(3 * h, 0.0),
    );
    mk(
        "transformer.h.0.attn.c_proj.weight",
        vec![h, h],
        fp32_matrix(h, h),
    );
    mk(
        "transformer.h.0.attn.c_proj.bias",
        vec![h],
        fp32_vector(h, 0.0),
    );
    mk("transformer.h.0.ln_2.weight", vec![h], fp32_vector(h, 1.0));
    mk("transformer.h.0.ln_2.bias", vec![h], fp32_vector(h, 0.0));
    mk(
        "transformer.h.0.mlp.c_fc.weight",
        vec![GPT2_FFN, h],
        fp32_matrix(GPT2_FFN, h),
    );
    mk(
        "transformer.h.0.mlp.c_fc.bias",
        vec![GPT2_FFN],
        fp32_vector(GPT2_FFN, 0.0),
    );
    mk(
        "transformer.h.0.mlp.c_proj.weight",
        vec![h, GPT2_FFN],
        fp32_matrix(h, GPT2_FFN),
    );
    mk(
        "transformer.h.0.mlp.c_proj.bias",
        vec![h],
        fp32_vector(h, 0.0),
    );
    mk("transformer.ln_f.weight", vec![h], fp32_vector(h, 1.0));
    mk("transformer.ln_f.bias", vec![h], fp32_vector(h, 0.0));
    mk(
        "transformer.wte.weight",
        vec![GPT2_VOCAB, h],
        fp32_matrix(GPT2_VOCAB, h),
    );

    Model {
        name: "mini_gpt2".to_string(),
        architecture: "gpt2".to_string(),
        tensors,
        metadata: HashMap::new(),
    }
}

const LLAMA_HIDDEN: usize = 12;
const LLAMA_INTER: usize = 48;
const LLAMA_VOCAB: usize = 10;

/// Minimal but complete single-layer LLaMA fixture with explicit `lm_head`.
pub fn create_llama_test_model() -> Model {
    let h = LLAMA_HIDDEN;
    let mut tensors = HashMap::new();
    let mut mk = |name: &str, shape: Vec<usize>, data: Vec<u8>| {
        tensors.insert(
            name.to_string(),
            TensorData {
                shape,
                dtype: DataType::F32,
                data,
            },
        );
    };

    mk(
        "model.layers.0.input_layernorm.weight",
        vec![h],
        fp32_vector(h, 1.0),
    );
    mk(
        "model.layers.0.self_attn.q_proj.weight",
        vec![h, h],
        fp32_matrix(h, h),
    );
    mk(
        "model.layers.0.self_attn.k_proj.weight",
        vec![h, h],
        fp32_matrix(h, h),
    );
    mk(
        "model.layers.0.self_attn.v_proj.weight",
        vec![h, h],
        fp32_matrix(h, h),
    );
    mk(
        "model.layers.0.self_attn.o_proj.weight",
        vec![h, h],
        fp32_matrix(h, h),
    );
    mk(
        "model.layers.0.post_attention_layernorm.weight",
        vec![h],
        fp32_vector(h, 1.0),
    );
    mk(
        "model.layers.0.mlp.gate_proj.weight",
        vec![LLAMA_INTER, h],
        fp32_matrix(LLAMA_INTER, h),
    );
    mk(
        "model.layers.0.mlp.up_proj.weight",
        vec![LLAMA_INTER, h],
        fp32_matrix(LLAMA_INTER, h),
    );
    mk(
        "model.layers.0.mlp.down_proj.weight",
        vec![h, LLAMA_INTER],
        fp32_matrix(h, LLAMA_INTER),
    );
    mk(
        "model.embed_tokens.weight",
        vec![LLAMA_VOCAB, h],
        fp32_matrix(LLAMA_VOCAB, h),
    );
    mk("model.norm.weight", vec![h], fp32_vector(h, 1.0));
    mk(
        "lm_head.weight",
        vec![LLAMA_VOCAB, h],
        fp32_matrix(LLAMA_VOCAB, h),
    );

    let mut metadata = HashMap::new();
    metadata.insert("attention.head_count".to_string(), "2".to_string());

    Model {
        name: "mini_llama".to_string(),
        architecture: "llama".to_string(),
        tensors,
        metadata,
    }
}

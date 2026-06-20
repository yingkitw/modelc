mod common;

use modelc::model::{DataType, Model, TensorData};
use std::collections::HashMap;

#[test]
fn test_dtype_byte_size() {
    assert_eq!(DataType::F32.byte_size(), 4);
    assert_eq!(DataType::F16.byte_size(), 2);
    assert_eq!(DataType::BF16.byte_size(), 2);
    assert_eq!(DataType::I64.byte_size(), 8);
    assert_eq!(DataType::I32.byte_size(), 4);
    assert_eq!(DataType::I16.byte_size(), 2);
    assert_eq!(DataType::I8.byte_size(), 1);
    assert_eq!(DataType::U8.byte_size(), 1);
    assert_eq!(DataType::Bool.byte_size(), 1);
}

#[test]
fn test_dtype_element_count() {
    assert_eq!(DataType::F32.element_count(&[2, 3]), 6);
    assert_eq!(DataType::F32.element_count(&[1, 1, 1]), 1);
    assert_eq!(DataType::F32.element_count(&[4, 4, 4]), 64);
    assert_eq!(DataType::F32.element_count(&[]), 1);
}

#[test]
fn test_dtype_total_bytes() {
    assert_eq!(DataType::F32.total_bytes(&[2, 3]), 24);
    assert_eq!(DataType::F16.total_bytes(&[4, 5]), 40);
    assert_eq!(DataType::I64.total_bytes(&[2, 2]), 32);
}

#[test]
fn test_dtype_rust_type() {
    assert_eq!(DataType::F32.rust_type(), "f32");
    assert_eq!(DataType::F16.rust_type(), "u16");
    assert_eq!(DataType::BF16.rust_type(), "u16");
    assert_eq!(DataType::I64.rust_type(), "i64");
    assert_eq!(DataType::I32.rust_type(), "i32");
    assert_eq!(DataType::I16.rust_type(), "i16");
    assert_eq!(DataType::I8.rust_type(), "i8");
    assert_eq!(DataType::U8.rust_type(), "u8");
    assert_eq!(DataType::Bool.rust_type(), "bool");
}

#[test]
fn test_tensor_data_byte_len() {
    let td = TensorData {
        shape: vec![2, 3],
        dtype: DataType::F32,
        data: vec![0u8; 24],
    };
    assert_eq!(td.byte_len(), 24);
    assert_eq!(td.element_count(), 6);
}

#[test]
fn test_tensor_data_f16() {
    let td = TensorData {
        shape: vec![4, 4],
        dtype: DataType::F16,
        data: vec![0u8; 32],
    };
    assert_eq!(td.byte_len(), 32);
    assert_eq!(td.element_count(), 16);
}

#[test]
fn test_model_total_params() {
    let model = common::create_test_model();
    assert_eq!(model.total_params(), 8); // 6 (weight 2×3) + 2 bias
}

#[test]
fn test_model_total_bytes() {
    let model = common::create_test_model();
    assert_eq!(model.total_bytes(), 32); // 24 + 8
}

#[test]
fn test_model_empty() {
    let model = Model {
        name: "empty".to_string(),
        architecture: "none".to_string(),
        tensors: HashMap::new(),
        metadata: HashMap::new(),
    };
    assert_eq!(model.total_params(), 0);
    assert_eq!(model.total_bytes(), 0);
}

#[test]
fn test_model_large() {
    let model = common::create_large_test_model();
    assert_eq!(model.tensors.len(), 6);
    assert!(model.total_params() > 0);
    assert!(model.total_bytes() > 0);

    let hidden_dim = 64usize;
    let expected_params = hidden_dim * hidden_dim // layer0.weight
        + hidden_dim // layer0.bias
        + hidden_dim // layer0.ln_weight
        + hidden_dim // layer0.ln_bias
        + hidden_dim * hidden_dim // layer1.weight
        + hidden_dim; // layer1.bias
    assert_eq!(model.total_params(), expected_params);
}

#[test]
fn test_model_serialization_roundtrip() {
    let model = common::create_test_model();
    let json = serde_json::to_string(&model).unwrap();
    let deserialized: Model = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.name, model.name);
    assert_eq!(deserialized.architecture, model.architecture);
    assert_eq!(deserialized.tensors.len(), model.tensors.len());
    for (name, td) in &model.tensors {
        let d = deserialized.tensors.get(name).unwrap();
        assert_eq!(d.shape, td.shape);
        assert_eq!(d.dtype, td.dtype);
        assert_eq!(d.data, td.data);
    }
}

#[test]
fn test_model_binary_serialization_roundtrip() {
    let model = common::create_test_model();
    let bincode = serde_json::to_vec(&model).unwrap();
    let deserialized: Model = serde_json::from_slice(&bincode).unwrap();
    assert_eq!(deserialized.tensors["weight"].shape, vec![2, 3]);
    assert_eq!(deserialized.tensors["bias"].shape, vec![2]);
}

#[test]
fn test_dequantize_i8_in_place() {
    let mut model = Model {
        name: "q".to_string(),
        architecture: "mlp".to_string(),
        metadata: {
            let mut m = HashMap::new();
            m.insert("quant_scale.weight".to_string(), "0.5".to_string());
            m
        },
        tensors: {
            let mut t = HashMap::new();
            t.insert(
                "weight".to_string(),
                TensorData {
                    shape: vec![4],
                    dtype: DataType::I8,
                    data: vec![2i8 as u8, 4i8 as u8, -2i8 as u8, 0u8],
                },
            );
            t
        },
    };
    model.dequantize_in_place();
    let td = &model.tensors["weight"];
    assert_eq!(td.dtype, DataType::F32);
    let vals: Vec<f32> = td
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(
        (vals[0] - 1.0).abs() < 1e-5,
        "expected 1.0, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 2.0).abs() < 1e-5,
        "expected 2.0, got {}",
        vals[1]
    );
    assert!(
        (vals[2] - (-1.0)).abs() < 1e-5,
        "expected -1.0, got {}",
        vals[2]
    );
    assert!(
        (vals[3] - 0.0).abs() < 1e-5,
        "expected 0.0, got {}",
        vals[3]
    );
}

#[test]
fn test_dequantize_int4_in_place() {
    let mut model = Model {
        name: "q".to_string(),
        architecture: "mlp".to_string(),
        metadata: {
            let mut m = HashMap::new();
            m.insert("quant_scale.weight".to_string(), "1.0".to_string());
            m.insert("quant_mode.weight".to_string(), "int4".to_string());
            m
        },
        tensors: {
            let mut t = HashMap::new();
            // Pack two 4-bit values per byte: high=15, low=0 => (15-8)=7, (0-8)=-8
            t.insert(
                "weight".to_string(),
                TensorData {
                    shape: vec![4],
                    dtype: DataType::I8,
                    data: vec![0xF0u8, 0xF0u8],
                },
            );
            t
        },
    };
    model.dequantize_in_place();
    let td = &model.tensors["weight"];
    assert_eq!(td.dtype, DataType::F32);
    let vals: Vec<f32> = td
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals.len(), 4);
    // Each nibble: (15 - 8) = 7, (0 - 8) = -8
    assert!(
        (vals[0] - 7.0).abs() < 1e-5,
        "expected 7.0, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - (-8.0)).abs() < 1e-5,
        "expected -8.0, got {}",
        vals[1]
    );
}

#[test]
fn test_infer_architecture_gpt2() {
    let model = Model {
        name: "gpt2".to_string(),
        architecture: "unknown".to_string(),
        metadata: HashMap::new(),
        tensors: {
            let mut t = HashMap::new();
            t.insert(
                "transformer.wte.weight".to_string(),
                TensorData {
                    shape: vec![50257, 768],
                    dtype: DataType::F32,
                    data: vec![0u8; 50257 * 768 * 4],
                },
            );
            t.insert(
                "transformer.h.0.ln_1.weight".to_string(),
                TensorData {
                    shape: vec![768],
                    dtype: DataType::F32,
                    data: vec![0u8; 768 * 4],
                },
            );
            t
        },
    };
    assert_eq!(model.infer_architecture(), "gpt2");
}

#[test]
fn test_infer_architecture_llama() {
    let model = Model {
        name: "llama".to_string(),
        architecture: "unknown".to_string(),
        metadata: HashMap::new(),
        tensors: {
            let mut t = HashMap::new();
            t.insert(
                "model.layers.0.input_layernorm.weight".to_string(),
                TensorData {
                    shape: vec![4096],
                    dtype: DataType::F32,
                    data: vec![0u8; 4096 * 4],
                },
            );
            t.insert(
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                TensorData {
                    shape: vec![4096, 4096],
                    dtype: DataType::F32,
                    data: vec![0u8; 4096 * 4096 * 4],
                },
            );
            t
        },
    };
    assert_eq!(model.infer_architecture(), "llama");
}

#[test]
fn test_infer_architecture_bert() {
    let model = Model {
        name: "bert".to_string(),
        architecture: "unknown".to_string(),
        metadata: HashMap::new(),
        tensors: {
            let mut t = HashMap::new();
            t.insert(
                "embeddings.word_embeddings.weight".to_string(),
                TensorData {
                    shape: vec![30522, 768],
                    dtype: DataType::F32,
                    data: vec![0u8; 30522 * 768 * 4],
                },
            );
            t.insert(
                "encoder.layer.0.attention.self.query.weight".to_string(),
                TensorData {
                    shape: vec![768, 768],
                    dtype: DataType::F32,
                    data: vec![0u8; 768 * 768 * 4],
                },
            );
            t
        },
    };
    assert_eq!(model.infer_architecture(), "bert");
}

#[test]
fn test_infer_architecture_mlp() {
    let model = Model {
        name: "mlp".to_string(),
        architecture: "unknown".to_string(),
        metadata: HashMap::new(),
        tensors: {
            let mut t = HashMap::new();
            t.insert(
                "weight".to_string(),
                TensorData {
                    shape: vec![2, 3],
                    dtype: DataType::F32,
                    data: vec![0u8; 2 * 3 * 4],
                },
            );
            t.insert(
                "bias".to_string(),
                TensorData {
                    shape: vec![2],
                    dtype: DataType::F32,
                    data: vec![0u8; 2 * 4],
                },
            );
            t
        },
    };
    assert_eq!(model.infer_architecture(), "mlp");
}

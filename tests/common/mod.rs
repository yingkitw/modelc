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

    let bias_data: Vec<u8> = [0.1f32, 0.2, 0.3]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    tensors.insert(
        "bias".to_string(),
        TensorData {
            shape: vec![3],
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

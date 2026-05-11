mod common;

use modelc::model::{DataType, TensorData};
use modelc::runtime::serve::Runtime;
use std::collections::HashMap;

#[test]
fn test_runtime_from_raw_f32() {
    let model = common::create_test_model();
    let runtime = Runtime::from_raw(&model.tensors);

    let weight = runtime.get("weight").unwrap();
    assert_eq!(weight.shape, vec![2, 3]);
    let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(weight.data, expected);
}

#[test]
fn test_runtime_from_raw_bias() {
    let model = common::create_test_model();
    let runtime = Runtime::from_raw(&model.tensors);

    let bias = runtime.get("bias").unwrap();
    assert_eq!(bias.shape, vec![2]);
    let expected: Vec<f32> = vec![0.1, 0.2];
    for (a, b) in bias.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_runtime_from_raw_f16() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "h".to_string(),
        TensorData {
            shape: vec![2],
            dtype: DataType::F16,
            data: vec![0x00, 0x3c, 0x00, 0x40],
        },
    );
    let runtime = Runtime::from_raw(&tensors);
    let t = runtime.get("h").unwrap();
    assert!((t.data[0] - 1.0).abs() < 1e-3);
    assert!((t.data[1] - 2.0).abs() < 1e-3);
}

#[test]
fn test_runtime_mixed_dtypes_truncated_skipped() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "trunc_f32".to_string(),
        TensorData {
            shape: vec![2],
            dtype: DataType::F32,
            data: vec![0],
        },
    );
    tensors.insert(
        "ok_f32".to_string(),
        TensorData {
            shape: vec![2],
            dtype: DataType::F32,
            data: common::f32_to_bytes(&[1.0, 2.0]),
        },
    );

    let runtime = Runtime::from_raw(&tensors);
    assert!(runtime.get("ok_f32").is_some());
    assert!(runtime.get("trunc_f32").is_none());
}

#[test]
fn test_runtime_missing_tensor() {
    let model = common::create_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    assert!(runtime.get("nonexistent").is_none());
}

#[test]
fn test_runtime_tensor_names_sorted() {
    let model = common::create_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    let names = runtime.tensor_names();
    assert_eq!(names, vec!["bias", "weight"]);
}

#[test]
fn test_runtime_empty() {
    let tensors = HashMap::new();
    let runtime = Runtime::from_raw(&tensors);
    assert!(runtime.tensor_names().is_empty());
}

#[test]
fn test_runtime_large_model() {
    let model = common::create_large_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    let names = runtime.tensor_names();
    assert_eq!(names.len(), 6);
    assert!(runtime.get("layer0.weight").is_some());
    assert!(runtime.get("layer1.bias").is_some());
    assert_eq!(runtime.get("layer0.weight").unwrap().shape, vec![64, 64]);
}

//! Inline tests for `onnx_exec::helpers` (transpose, attribute getters,
//! element-wise ops, activations). Kept here because `helpers` is a private
//! submodule.

use std::collections::HashMap;

use crate::model::{DataType, TensorData};
use crate::onnx_exec::AttributeValue;
use crate::runtime::tensor::Tensor;

use super::helpers::{
    element_wise_div, element_wise_mul, element_wise_sub, get_attr_f32, get_attr_int,
    get_attr_ints, sigmoid, tanh_tensor, transpose, transpose_with_perm,
};

// --- attribute getters ---

#[test]
fn get_attr_f32_returns_default_when_missing() {
    let attrs: HashMap<String, AttributeValue> = HashMap::new();
    assert_eq!(get_attr_f32(&attrs, "x", 3.5), 3.5);
}

#[test]
fn get_attr_f32_reads_float_and_int_values() {
    let mut attrs = HashMap::new();
    attrs.insert("f".to_string(), AttributeValue::Float(1.5));
    attrs.insert("i".to_string(), AttributeValue::Int(7));
    assert_eq!(get_attr_f32(&attrs, "f", 0.0), 1.5);
    assert_eq!(get_attr_f32(&attrs, "i", 0.0), 7.0);
}

#[test]
fn get_attr_f32_falls_back_for_unsupported_variant() {
    let mut attrs = HashMap::new();
    attrs.insert("s".to_string(), AttributeValue::String("nope".to_string()));
    assert_eq!(get_attr_f32(&attrs, "s", 9.0), 9.0);
}

#[test]
fn get_attr_int_returns_default_when_missing() {
    let attrs: HashMap<String, AttributeValue> = HashMap::new();
    assert_eq!(get_attr_int(&attrs, "x", 42), 42);
}

#[test]
fn get_attr_int_reads_int_and_float_values() {
    let mut attrs = HashMap::new();
    attrs.insert("i".to_string(), AttributeValue::Int(-5));
    attrs.insert("f".to_string(), AttributeValue::Float(2.5));
    assert_eq!(get_attr_int(&attrs, "i", 0), -5);
    assert_eq!(get_attr_int(&attrs, "f", 0), 2);
}

#[test]
fn get_attr_ints_returns_vec_or_empty() {
    let mut attrs = HashMap::new();
    attrs.insert("arr".to_string(), AttributeValue::Ints(vec![1, 2, 3]));
    assert_eq!(get_attr_ints(&attrs, "arr"), Some(vec![1, 2, 3]));
    assert_eq!(get_attr_ints(&attrs, "missing"), None);
    attrs.insert("s".to_string(), AttributeValue::String("nope".to_string()));
    assert_eq!(get_attr_ints(&attrs, "s"), Some(vec![]));
}

// --- transpose ---

#[test]
fn transpose_non_2d_passes_through() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let out = transpose(&t);
    assert_eq!(out.data, t.data);
    assert_eq!(out.shape, vec![4]);
}

#[test]
fn transpose_2d_swaps_axes() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = transpose(&t);
    assert_eq!(out.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(out.shape, vec![3, 2]);
}

#[test]
fn transpose_with_perm_identity_is_identity() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = transpose_with_perm(&t, &[0, 1]).unwrap();
    assert_eq!(out.data, t.data);
}

#[test]
fn transpose_with_perm_reverses_axes() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = transpose_with_perm(&t, &[1, 0]).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(out.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn transpose_with_perm_length_mismatch_errors() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert!(transpose_with_perm(&t, &[0]).is_err());
}

// --- element-wise ---

#[test]
fn element_wise_mul_same_shape() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);
    let out = element_wise_mul(&a, &b).unwrap();
    assert_eq!(out.data, vec![4.0, 10.0, 18.0]);
}

#[test]
fn element_wise_mul_scalar_broadcast_left() {
    let a = Tensor::from_vec(vec![2.0, 4.0, 6.0], vec![3]);
    let b = Tensor::from_vec(vec![0.5], vec![1]);
    let out = element_wise_mul(&a, &b).unwrap();
    assert_eq!(out.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn element_wise_mul_scalar_left_operand_is_one() {
    let a = Tensor::from_vec(vec![0.5], vec![1]);
    let b = Tensor::from_vec(vec![2.0, 4.0, 6.0], vec![3]);
    let out = element_wise_mul(&a, &b).unwrap();
    assert_eq!(out.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn element_wise_mul_incompatible_errors() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(element_wise_mul(&a, &b).is_err());
}

#[test]
fn element_wise_div_same_shape() {
    let a = Tensor::from_vec(vec![4.0, 9.0, 16.0], vec![3]);
    let b = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);
    let out = element_wise_div(&a, &b).unwrap();
    assert_eq!(out.data, vec![2.0, 3.0, 4.0]);
}

#[test]
fn element_wise_div_scalar_broadcast() {
    let a = Tensor::from_vec(vec![4.0, 9.0, 16.0], vec![3]);
    let b = Tensor::from_vec(vec![2.0], vec![1]);
    let out = element_wise_div(&a, &b).unwrap();
    assert_eq!(out.data, vec![2.0, 4.5, 8.0]);
}

#[test]
fn element_wise_div_incompatible_errors() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(element_wise_div(&a, &b).is_err());
}

#[test]
fn element_wise_sub_same_shape() {
    let a = Tensor::from_vec(vec![10.0, 5.0, 1.0], vec![3]);
    let b = Tensor::from_vec(vec![3.0, 2.0, 1.0], vec![3]);
    let out = element_wise_sub(&a, &b).unwrap();
    assert_eq!(out.data, vec![7.0, 3.0, 0.0]);
}

#[test]
fn element_wise_sub_incompatible_errors() {
    let a = Tensor::from_vec(vec![1.0], vec![1]);
    let b = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    assert!(element_wise_sub(&a, &b).is_err());
}

// --- activations ---

#[test]
fn sigmoid_applies_per_element() {
    let t = Tensor::from_vec(vec![0.0, 100.0, -100.0], vec![3]);
    let out = sigmoid(&t);
    assert!((out.data[0] - 0.5).abs() < 1e-6);
    assert!((out.data[1] - 1.0).abs() < 1e-6);
    assert!(out.data[2].abs() < 1e-6);
}

#[test]
fn tanh_tensor_applies_per_element() {
    let t = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);
    let out = tanh_tensor(&t);
    assert_eq!(out.data[0], 0.0);
    assert!((out.data[1] - 0.7615942).abs() < 1e-5);
    assert!((out.data[2] + 0.7615942).abs() < 1e-5);
}

// --- sanity: helpers are reachable from outside via the parent module ---

#[test]
fn tensor_data_f32_helper() {
    let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let td = TensorData {
        shape: vec![2, 2],
        dtype: DataType::F32,
        data: bytes,
    };
    assert_eq!(td.element_count(), 4);
    assert_eq!(td.dtype, DataType::F32);
}

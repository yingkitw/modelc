use std::collections::HashMap;

use anyhow::{Result, ensure};

use crate::runtime::tensor::Tensor;

use super::AttributeValue;

pub(super) fn get_attr_f32(attrs: &HashMap<String, AttributeValue>, key: &str, default: f32) -> f32 {
    attrs.get(key).map(|v| match v {
        AttributeValue::Float(f) => *f,
        AttributeValue::Int(i) => *i as f32,
        _ => default,
    }).unwrap_or(default)
}

pub(super) fn get_attr_int(attrs: &HashMap<String, AttributeValue>, key: &str, default: i64) -> i64 {
    attrs.get(key).map(|v| match v {
        AttributeValue::Int(i) => *i,
        AttributeValue::Float(f) => *f as i64,
        _ => default,
    }).unwrap_or(default)
}

pub(super) fn get_attr_ints(attrs: &HashMap<String, AttributeValue>, key: &str) -> Option<Vec<i64>> {
    attrs.get(key).map(|v| match v {
        AttributeValue::Ints(v) => v.clone(),
        _ => Vec::new(),
    })
}

pub(super) fn transpose(a: &Tensor) -> Tensor {
    if a.shape.len() != 2 {
        return Tensor::from_vec(a.data.clone(), a.shape.clone());
    }
    let (m, n) = (a.shape[0], a.shape[1]);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a.data[i * n + j];
        }
    }
    Tensor::from_vec(out, vec![n, m])
}

pub(super) fn transpose_with_perm(a: &Tensor, perm: &[i64]) -> Result<Tensor> {
    ensure!(
        perm.len() == a.shape.len(),
        "transpose perm length mismatch"
    );
    let new_shape: Vec<usize> = perm.iter().map(|&p| a.shape[p as usize]).collect();
    let mut out = vec![0.0f32; a.data.len()];

    let rank = a.shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * a.shape[i + 1];
    }
    let mut new_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    for idx in 0..a.data.len() {
        let mut coords = vec![0usize; rank];
        let mut rem = idx;
        for i in 0..rank {
            coords[i] = rem / strides[i];
            rem %= strides[i];
        }
        let new_idx: usize = perm
            .iter()
            .enumerate()
            .map(|(i, &p)| coords[p as usize] * new_strides[i])
            .sum();
        out[new_idx] = a.data[idx];
    }

    Ok(Tensor::from_vec(out, new_shape))
}

pub(super) fn element_wise_mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape == b.shape {
        let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
        Ok(Tensor::from_vec(data, a.shape.clone()))
    } else if b.data.len() == 1 {
        Ok(crate::runtime::ops::mul_scalar(a, b.data[0]))
    } else if a.data.len() == 1 {
        Ok(crate::runtime::ops::mul_scalar(b, a.data[0]))
    } else {
        anyhow::bail!("Mul: incompatible shapes {:?} and {:?}", a.shape, b.shape)
    }
}

pub(super) fn element_wise_div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape == b.shape {
        let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x / y).collect();
        Ok(Tensor::from_vec(data, a.shape.clone()))
    } else if b.data.len() == 1 {
        let s = b.data[0];
        let data: Vec<f32> = a.data.iter().map(|x| x / s).collect();
        Ok(Tensor::from_vec(data, a.shape.clone()))
    } else {
        anyhow::bail!("Div: incompatible shapes {:?} and {:?}", a.shape, b.shape)
    }
}

pub(super) fn element_wise_sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape == b.shape {
        let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x - y).collect();
        Ok(Tensor::from_vec(data, a.shape.clone()))
    } else {
        anyhow::bail!("Sub: incompatible shapes {:?} and {:?}", a.shape, b.shape)
    }
}

pub(super) fn sigmoid(a: &Tensor) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    Tensor::from_vec(data, a.shape.clone())
}

pub(super) fn tanh_tensor(a: &Tensor) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|x| x.tanh()).collect();
    Tensor::from_vec(data, a.shape.clone())
}

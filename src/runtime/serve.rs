use std::collections::HashMap;

use half::bf16;
use half::f16;

use crate::model::{DataType, TensorData};
use crate::runtime::tensor::Tensor;

pub struct Runtime {
    tensors: HashMap<String, Tensor>,
}

fn bytes_to_f32_slice(td: &TensorData) -> Option<Vec<f32>> {
    let count = td.element_count();
    match td.dtype {
        DataType::F32 => {
            let byte_len = count * 4;
            if td.data.len() < byte_len {
                return None;
            }
            let mut out = vec![0.0f32; count];
            for (i, chunk) in td.data[..byte_len].chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Some(out)
        }
        DataType::F16 => {
            let byte_len = count * 2;
            if td.data.len() < byte_len {
                return None;
            }
            let mut out = vec![0.0f32; count];
            for (i, chunk) in td.data[..byte_len].chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = f16::from_bits(bits).to_f32();
            }
            Some(out)
        }
        DataType::BF16 => {
            let byte_len = count * 2;
            if td.data.len() < byte_len {
                return None;
            }
            let mut out = vec![0.0f32; count];
            for (i, chunk) in td.data[..byte_len].chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = bf16::from_bits(bits).to_f32();
            }
            Some(out)
        }
        DataType::I64 => {
            let byte_len = count * 8;
            if td.data.len() < byte_len {
                return None;
            }
            let mut out = vec![0.0f32; count];
            for (i, chunk) in td.data[..byte_len].chunks_exact(8).enumerate() {
                out[i] = i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32;
            }
            Some(out)
        }
        DataType::I32 => {
            let byte_len = count * 4;
            if td.data.len() < byte_len {
                return None;
            }
            let mut out = vec![0.0f32; count];
            for (i, chunk) in td.data[..byte_len].chunks_exact(4).enumerate() {
                out[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32;
            }
            Some(out)
        }
        DataType::I16 => {
            let byte_len = count * 2;
            if td.data.len() < byte_len {
                return None;
            }
            let mut out = vec![0.0f32; count];
            for (i, chunk) in td.data[..byte_len].chunks_exact(2).enumerate() {
                out[i] = i16::from_le_bytes([chunk[0], chunk[1]]) as f32;
            }
            Some(out)
        }
        DataType::I8 => {
            if td.data.len() < count {
                return None;
            }
            let mut out = vec![0.0f32; count];
            for (i, &b) in td.data[..count].iter().enumerate() {
                out[i] = b as i8 as f32;
            }
            Some(out)
        }
        DataType::U8 => {
            if td.data.len() < count {
                return None;
            }
            let mut out = vec![0.0f32; count];
            for (i, &b) in td.data[..count].iter().enumerate() {
                out[i] = b as f32;
            }
            Some(out)
        }
        DataType::Bool => {
            if td.data.len() < count {
                return None;
            }
            let mut out = vec![0.0f32; count];
            for (i, &b) in td.data[..count].iter().enumerate() {
                out[i] = if b != 0 { 1.0 } else { 0.0 };
            }
            Some(out)
        }
    }
}

impl Runtime {
    pub fn from_raw(raw: &HashMap<String, TensorData>) -> Self {
        let mut tensors = HashMap::new();
        for (name, td) in raw {
            if let Some(fdata) = bytes_to_f32_slice(td) {
                tensors.insert(name.clone(), Tensor::from_vec(fdata, td.shape.clone()));
            }
        }
        Self { tensors }
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> Vec<&String> {
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();
        names
    }
}

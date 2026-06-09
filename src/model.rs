use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    pub architecture: String,
    pub tensors: HashMap<String, TensorData>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl DataType {
    pub fn byte_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I64 => 8,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    pub fn element_count(&self, shape: &[usize]) -> usize {
        shape.iter().product::<usize>()
    }

    #[allow(dead_code)]
    pub fn rust_type(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "u16",
            Self::BF16 => "u16",
            Self::I64 => "i64",
            Self::I32 => "i32",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::Bool => "bool",
        }
    }

    pub fn total_bytes(&self, shape: &[usize]) -> usize {
        self.element_count(shape) * self.byte_size()
    }
}

impl TensorData {
    pub fn byte_len(&self) -> usize {
        self.dtype.total_bytes(&self.shape)
    }

    pub fn element_count(&self) -> usize {
        self.dtype.element_count(&self.shape)
    }
}

impl Model {
    pub fn total_params(&self) -> usize {
        self.tensors.values().map(|t| t.element_count()).sum()
    }

    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.byte_len()).sum()
    }

    /// Dequantize any I8 tensors that have `quant_scale.<name>` metadata back to F32 in place.
    pub fn dequantize_in_place(&mut self) {
        for (name, td) in self.tensors.iter_mut() {
            if td.dtype != DataType::I8 {
                continue;
            }
            let scale_key = format!("quant_scale.{}", name);
            if let Some(scale_str) = self.metadata.get(&scale_key) {
                if let Ok(scale) = scale_str.parse::<f32>() {
                    let count = td.element_count();
                    let mut new_data = Vec::with_capacity(count * 4);
                    for &b in &td.data {
                        let val = b as i8 as f32 * scale;
                        new_data.extend_from_slice(&val.to_le_bytes());
                    }
                    td.dtype = DataType::F32;
                    td.data = new_data;
                }
            }
        }
    }
}

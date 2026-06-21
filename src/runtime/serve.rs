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
        // GGUF quantized types are dequantized on-the-fly by Runtime::from_raw.
        DataType::Q4_0 | DataType::Q5_0 | DataType::Q8_0 | DataType::Q4_K | DataType::Q6_K => None,
    }
}

impl Runtime {
    pub fn from_raw(raw: &HashMap<String, TensorData>) -> Self {
        let mut tensors = HashMap::new();
        for (name, td) in raw {
            let fdata =
                bytes_to_f32_slice(td).or_else(|| crate::parsers::gguf::dequantize_gguf_tensor(td));
            if let Some(data) = fdata {
                tensors.insert(name.clone(), Tensor::from_vec(data, td.shape.clone()));
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use half::f16;

    use crate::model::{DataType, TensorData};
    use crate::runtime::serve::Runtime;

    #[test]
    fn runtime_dequantizes_q8_0_on_the_fly() {
        // Build a Q8_0 block: 32 elements, delta 2.0, first quant = 5 (=> 10.0), rest 0.
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());
        for j in 0..32 {
            block.push(if j == 0 { 5i8 as u8 } else { 0 });
        }

        let raw = HashMap::from([(
            "w".to_string(),
            TensorData {
                shape: vec![32],
                dtype: DataType::Q8_0,
                data: block,
            },
        )]);

        let rt = Runtime::from_raw(&raw);
        let t = rt.get("w").expect("tensor loaded");
        assert_eq!(t.data.len(), 32);
        assert!(
            (t.data[0] - 10.0).abs() < 1e-5,
            "first element: {}",
            t.data[0]
        );
        assert!(
            t.data[1..].iter().all(|&v| v.abs() < 1e-6),
            "rest should be zero"
        );
    }

    #[test]
    fn runtime_dequantizes_q4_0_on_the_fly() {
        // Build a Q4_0 block: 32 elements, delta 1.0, all nibbles = 0x88.
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        block.extend(vec![0x88u8; 16]);

        let raw = HashMap::from([(
            "w".to_string(),
            TensorData {
                shape: vec![32],
                dtype: DataType::Q4_0,
                data: block,
            },
        )]);

        let rt = Runtime::from_raw(&raw);
        let t = rt.get("w").expect("tensor loaded");
        assert_eq!(t.data.len(), 32);
    }
}

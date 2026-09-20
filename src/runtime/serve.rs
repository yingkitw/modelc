use std::collections::HashMap;
use std::sync::OnceLock;

use half::bf16;
use half::f16;

use crate::model::{DataType, Model, TensorData};
use crate::runtime::tensor::Tensor;

pub struct Runtime {
    tensors: HashMap<String, Tensor>,
    /// Lazily-built structural view (tensor names + shapes only) shared with the
    /// `arch` inspection helpers. Built once per `Runtime` instance — a new
    /// `Runtime` (e.g. after a LoRA swap) gets a fresh lock so the view refreshes.
    view: OnceLock<Model>,
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
        Self {
            tensors,
            view: OnceLock::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> Vec<&String> {
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();
        names
    }

    /// Cached structural view of this runtime as a synthetic [`Model`], carrying only
    /// tensor names and shapes (data is empty). Used by the `arch` inspection helpers
    /// so layer/hidden/head detection does not re-scan the tensor map on every forward
    /// pass. Cheap to build, free on subsequent calls.
    pub(crate) fn model_view(&self) -> &Model {
        let Runtime { tensors, view } = self;
        view.get_or_init(|| build_model_view(tensors))
    }
}

fn build_model_view(tensors: &HashMap<String, Tensor>) -> Model {
    let mut view = HashMap::with_capacity(tensors.len());
    for (name, t) in tensors {
        view.insert(
            name.clone(),
            TensorData {
                shape: t.shape.clone(),
                dtype: DataType::F32,
                data: Vec::new(),
            },
        );
    }
    Model {
        name: String::new(),
        architecture: String::new(),
        tensors: view,
        metadata: HashMap::new(),
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

    #[test]
    fn model_view_is_memoized_and_exposes_shapes() {
        let raw = HashMap::from([(
            "w".to_string(),
            TensorData {
                shape: vec![4, 8],
                dtype: DataType::F32,
                data: [0.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect(),
            },
        )]);

        let rt = Runtime::from_raw(&raw);
        let v1 = rt.model_view();
        let v2 = rt.model_view();
        // Same allocation → the structural view is built once and cached.
        assert!(std::ptr::eq(v1, v2));
        let td = v1.tensors.get("w").expect("tensor present in view");
        assert_eq!(td.shape, vec![4, 8]);
        // Data is intentionally empty in the structural view (shapes only).
        assert!(td.data.is_empty());
    }

    #[test]
    fn model_view_refreshes_after_runtime_swap() {
        // A fresh Runtime (e.g. after a LoRA swap) must rebuild its view rather
        // than serve a stale one from the previous instance.
        let raw = HashMap::from([(
            "a".to_string(),
            TensorData {
                shape: vec![2, 2],
                dtype: DataType::F32,
                data: vec![0u8; 16],
            },
        )]);
        let rt = Runtime::from_raw(&raw);
        assert!(rt.model_view().tensors.contains_key("a"));

        let raw2 = HashMap::from([(
            "b".to_string(),
            TensorData {
                shape: vec![2, 2],
                dtype: DataType::F32,
                data: vec![0u8; 16],
            },
        )]);
        let rt2 = Runtime::from_raw(&raw2);
        let view2 = rt2.model_view();
        assert!(!view2.tensors.contains_key("a"));
        assert!(view2.tensors.contains_key("b"));
    }

    // ---- bytes_to_f32_slice dtype coverage (via Runtime::from_raw) ----

    #[test]
    fn runtime_casts_every_numeric_dtype_to_f32() {
        use half::bf16;
        let raw = HashMap::from([
            // F16: 1.0, 2.0
            (
                "f16_t".to_string(),
                TensorData {
                    shape: vec![2],
                    dtype: DataType::F16,
                    data: [
                        f16::from_f32(1.0).to_bits().to_le_bytes(),
                        f16::from_f32(2.0).to_bits().to_le_bytes(),
                    ]
                    .concat(),
                },
            ),
            // BF16: 1.0, 2.0
            (
                "bf16_t".to_string(),
                TensorData {
                    shape: vec![2],
                    dtype: DataType::BF16,
                    data: [
                        bf16::from_f32(1.0).to_bits().to_le_bytes(),
                        bf16::from_f32(2.0).to_bits().to_le_bytes(),
                    ]
                    .concat(),
                },
            ),
            // I64: -1, 7
            (
                "i64_t".to_string(),
                TensorData {
                    shape: vec![2],
                    dtype: DataType::I64,
                    data: (-1i64)
                        .to_le_bytes()
                        .iter()
                        .chain(7i64.to_le_bytes().iter())
                        .copied()
                        .collect(),
                },
            ),
            // I32: -2, 8
            (
                "i32_t".to_string(),
                TensorData {
                    shape: vec![2],
                    dtype: DataType::I32,
                    data: (-2i32)
                        .to_le_bytes()
                        .iter()
                        .chain(8i32.to_le_bytes().iter())
                        .copied()
                        .collect(),
                },
            ),
            // I16: -3, 9
            (
                "i16_t".to_string(),
                TensorData {
                    shape: vec![2],
                    dtype: DataType::I16,
                    data: (-3i16)
                        .to_le_bytes()
                        .iter()
                        .chain(9i16.to_le_bytes().iter())
                        .copied()
                        .collect(),
                },
            ),
            // I8: -4, 10
            (
                "i8_t".to_string(),
                TensorData {
                    shape: vec![2],
                    dtype: DataType::I8,
                    data: vec![(-4i8) as u8, 10],
                },
            ),
            // U8: 0, 255
            (
                "u8_t".to_string(),
                TensorData {
                    shape: vec![2],
                    dtype: DataType::U8,
                    data: vec![0, 255],
                },
            ),
            // Bool: true, false
            (
                "bool_t".to_string(),
                TensorData {
                    shape: vec![2],
                    dtype: DataType::Bool,
                    data: vec![1, 0],
                },
            ),
        ]);

        let rt = Runtime::from_raw(&raw);
        assert_eq!(rt.get("f16_t").unwrap().data, vec![1.0, 2.0]);
        assert_eq!(rt.get("bf16_t").unwrap().data, vec![1.0, 2.0]);
        assert_eq!(rt.get("i64_t").unwrap().data, vec![-1.0, 7.0]);
        assert_eq!(rt.get("i32_t").unwrap().data, vec![-2.0, 8.0]);
        assert_eq!(rt.get("i16_t").unwrap().data, vec![-3.0, 9.0]);
        assert_eq!(rt.get("i8_t").unwrap().data, vec![-4.0, 10.0]);
        assert_eq!(rt.get("u8_t").unwrap().data, vec![0.0, 255.0]);
        assert_eq!(rt.get("bool_t").unwrap().data, vec![1.0, 0.0]);
    }

    #[test]
    fn runtime_skips_tensors_with_truncated_data() {
        // Data shorter than the declared element count → conversion returns
        // None → the tensor is silently omitted from the runtime.
        let raw = HashMap::from([
            (
                "short_f32".to_string(),
                TensorData {
                    shape: vec![4],
                    dtype: DataType::F32,
                    data: vec![0u8; 8], // needs 16
                },
            ),
            (
                "short_f16".to_string(),
                TensorData {
                    shape: vec![4],
                    dtype: DataType::F16,
                    data: vec![0u8; 4], // needs 8
                },
            ),
            (
                "short_i64".to_string(),
                TensorData {
                    shape: vec![2],
                    dtype: DataType::I64,
                    data: vec![0u8; 8], // needs 16
                },
            ),
            (
                "short_bool".to_string(),
                TensorData {
                    shape: vec![4],
                    dtype: DataType::Bool,
                    data: vec![0u8; 2], // needs 4
                },
            ),
        ]);
        let rt = Runtime::from_raw(&raw);
        assert!(rt.get("short_f32").is_none());
        assert!(rt.get("short_f16").is_none());
        assert!(rt.get("short_i64").is_none());
        assert!(rt.get("short_bool").is_none());
        assert!(rt.tensor_names().is_empty());
    }
}

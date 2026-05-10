use std::collections::HashMap;

use crate::model::TensorData;
use crate::runtime::tensor::Tensor;

pub struct Runtime {
    tensors: HashMap<String, Tensor>,
}

impl Runtime {
    pub fn from_raw(raw: &HashMap<String, TensorData>) -> Self {
        let mut tensors = HashMap::new();
        for (name, td) in raw {
            if td.dtype == crate::model::DataType::F32 {
                let count = td.element_count();
                let byte_len = count * 4;
                if td.data.len() >= byte_len {
                    let mut fdata = vec![0.0f32; count];
                    let src_bytes = &td.data[..byte_len];
                    for (i, chunk) in src_bytes.chunks_exact(4).enumerate() {
                        fdata[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                    tensors.insert(name.clone(), Tensor::from_vec(fdata, td.shape.clone()));
                }
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

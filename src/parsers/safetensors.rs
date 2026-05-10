use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use crate::model::{DataType, Model, TensorData};
use crate::parsers::WeightParser;

pub struct SafetensorsParser;

impl WeightParser for SafetensorsParser {
    fn parse(&self, path: &Path) -> Result<Model> {
        let data =
            std::fs::read(path).with_context(|| format!("failed to read {:?}", path))?;

        let st = safetensors::SafeTensors::deserialize(&data)
            .with_context(|| "failed to parse safetensors")?;

        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();

        for (name_str, tensor_view) in st.tensors() {
            let name = name_str;
            let shape: Vec<usize> = tensor_view.shape().to_vec();
            let dtype = match tensor_view.dtype() {
                safetensors::Dtype::F32 => DataType::F32,
                safetensors::Dtype::F16 => DataType::F16,
                safetensors::Dtype::BF16 => DataType::BF16,
                safetensors::Dtype::I64 => DataType::I64,
                safetensors::Dtype::I32 => DataType::I32,
                safetensors::Dtype::I16 => DataType::I16,
                safetensors::Dtype::I8 => DataType::I8,
                safetensors::Dtype::U8 => DataType::U8,
                safetensors::Dtype::BOOL => DataType::Bool,
                other => {
                    anyhow::bail!("unsupported safetensors dtype: {:?}", other);
                }
            };

            let tensor_data = tensor_view.data();
            let mut owned_data = vec![0u8; tensor_data.len()];
            owned_data.copy_from_slice(tensor_data);

            tensors.insert(
                name,
                TensorData {
                    shape,
                    dtype,
                    data: owned_data,
                },
            );
        }

        metadata.insert(
            "source".to_string(),
            path.file_name().unwrap().to_string_lossy().to_string(),
        );
        metadata.insert("format".to_string(), "safetensors".to_string());

        let name = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        Ok(Model {
            name,
            architecture: "unknown".to_string(),
            tensors,
            metadata,
        })
    }

    fn format_name(&self) -> &'static str {
        "safetensors"
    }
}

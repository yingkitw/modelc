use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use crate::model::{DataType, Model, TensorData};
use crate::parsers::WeightParser;

pub struct SafetensorsParser;

impl WeightParser for SafetensorsParser {
    fn parse(&self, path: &Path) -> Result<Model> {
        let data = std::fs::read(path).with_context(|| format!("failed to read {:?}", path))?;

        let header_meta = safetensors::SafeTensors::read_metadata(&data).ok();

        let architecture = header_meta
            .as_ref()
            .and_then(|(_, h)| h.metadata().as_ref())
            .and_then(|m| {
                m.get("architecture")
                    .or_else(|| m.get("model_type"))
                    .cloned()
            })
            .unwrap_or_else(|| "unknown".to_string());

        let st = safetensors::SafeTensors::deserialize(&data)
            .with_context(|| "failed to parse safetensors")?;

        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();

        if let Some((_, hdr)) = &header_meta
            && let Some(hm) = hdr.metadata().as_ref()
        {
            for (k, v) in hm {
                metadata.insert(k.clone(), v.clone());
            }
        }

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

        metadata
            .entry("source".to_string())
            .or_insert_with(|| path.file_name().unwrap().to_string_lossy().to_string());
        metadata
            .entry("format".to_string())
            .or_insert_with(|| "safetensors".to_string());

        let name = path.file_stem().unwrap().to_string_lossy().to_string();

        Ok(Model {
            name,
            architecture,
            tensors,
            metadata,
        })
    }

    fn format_name(&self) -> &'static str {
        "safetensors"
    }
}

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
#[allow(non_camel_case_types)]
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
    /// GGML Q4_0 block-quantized (18 bytes per 32 elements).
    Q4_0,
    /// GGML Q5_0 block-quantized (22 bytes per 32 elements).
    Q5_0,
    /// GGML Q8_0 block-quantized (34 bytes per 32 elements).
    Q8_0,
    /// GGML Q4_K super-block-quantized (144 bytes per 256 elements).
    Q4_K,
    /// GGML Q6_K super-block-quantized (210 bytes per 256 elements).
    Q6_K,
}

impl DataType {
    pub fn byte_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I64 => 8,
            Self::I8 | Self::U8 | Self::Bool => 1,
            // Block-quantized types have no fixed per-element byte size.
            Self::Q4_0 | Self::Q5_0 | Self::Q8_0 | Self::Q4_K | Self::Q6_K => 0,
        }
    }

    pub fn element_count(&self, shape: &[usize]) -> usize {
        shape.iter().product::<usize>()
    }

    pub fn total_bytes(&self, shape: &[usize]) -> usize {
        self.element_count(shape) * self.byte_size()
    }
}

impl TensorData {
    /// Returns the actual length of the stored data bytes. For quantized types this is the
    /// on-disk / on-wire byte count, not the dequantized element count × byte_size.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    pub fn element_count(&self) -> usize {
        self.dtype.element_count(&self.shape)
    }
}

/// A target quantization format for the `--quant-sizes` preview. Computes the byte
/// size a model would occupy if all its tensors were stored in this format, without
/// actually quantizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantPreview {
    /// 32-bit float (4 bytes/element).
    F32,
    /// 16-bit float (2 bytes/element).
    F16,
    /// 8-bit integer (1 byte/element).
    Int8,
    /// 4-bit packed (0.5 bytes/element, two signed nibbles per byte).
    Int4,
    /// GGML Q4_0 block-quantized: 18 bytes per block of 32 elements
    /// (one f16 scale + 32 × 4-bit weights).
    Q4_0,
}

impl QuantPreview {
    /// Bytes required to store `elements` values in this format. Block-quantized
    /// formats round up to a whole block per tensor.
    pub fn bytes_for(&self, elements: usize) -> usize {
        match self {
            Self::F32 => elements * 4,
            Self::F16 => elements * 2,
            Self::Int8 => elements,
            Self::Int4 => elements.div_ceil(2),
            Self::Q4_0 => elements.div_ceil(32) * 18,
        }
    }

    /// Human-readable label for display.
    pub fn label(&self) -> &'static str {
        match self {
            Self::F32 => "fp32",
            Self::F16 => "fp16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Q4_0 => "q4_0",
        }
    }

    /// All preview formats in display order.
    pub fn all() -> [QuantPreview; 5] {
        [Self::F32, Self::F16, Self::Int8, Self::Int4, Self::Q4_0]
    }
}

impl Model {
    pub fn total_params(&self) -> usize {
        self.tensors.values().map(|t| t.element_count()).sum()
    }

    /// Total size of all stored tensor bytes. For block-quantized GGUF tensors this is
    /// the on-disk byte count, not the dequantized size.
    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.byte_len()).sum()
    }

    /// Preview the total byte size of all tensors if quantized to `format`, **without**
    /// actually quantizing. Computed from element counts (shape-based), so it is accurate
    /// regardless of the tensors' current dtype.
    pub fn preview_size_bytes(&self, format: QuantPreview) -> usize {
        self.tensors
            .values()
            .map(|t| format.bytes_for(t.element_count()))
            .sum()
    }

    /// Dequantize any I8 (or INT4-packed-as-I8) tensors that have `quant_scale.<name>` metadata
    /// back to F32 in place. Also dequantizes GGUF-quantized tensors (Q4_0, Q5_0, Q8_0, Q4_K, Q6_K).
    pub fn dequantize_in_place(&mut self) {
        // Handle our own INT8 / INT4 quantization.
        for (name, td) in self.tensors.iter_mut() {
            if td.dtype != DataType::I8 {
                continue;
            }
            let scale_key = format!("quant_scale.{}", name);
            if let Some(scale_str) = self.metadata.get(&scale_key)
                && let Ok(scale) = scale_str.parse::<f32>()
            {
                let mode_key = format!("quant_mode.{}", name);
                let is_int4 = self
                    .metadata
                    .get(&mode_key)
                    .map(|m| m == "int4")
                    .unwrap_or(false);
                let count = td.element_count();
                let mut new_data = Vec::with_capacity(count * 4);
                if is_int4 {
                    // Unpack two signed nibbles per byte.
                    let mut remaining = count;
                    for &byte in &td.data {
                        let nibble0 = ((byte >> 4) & 0x0F) as i8 - 8;
                        let val0 = nibble0 as f32 * scale;
                        new_data.extend_from_slice(&val0.to_le_bytes());
                        remaining -= 1;
                        if remaining > 0 {
                            let nibble1 = (byte & 0x0F) as i8 - 8;
                            let val1 = nibble1 as f32 * scale;
                            new_data.extend_from_slice(&val1.to_le_bytes());
                            remaining -= 1;
                        }
                    }
                } else {
                    for &b in &td.data {
                        let val = b as i8 as f32 * scale;
                        new_data.extend_from_slice(&val.to_le_bytes());
                    }
                }
                td.dtype = DataType::F32;
                td.data = new_data;
            }
        }

        // Handle GGUF block-quantized types.
        for (_, td) in self.tensors.iter_mut() {
            if let Some(fdata) = crate::parsers::gguf::dequantize_gguf_tensor(td) {
                td.dtype = DataType::F32;
                td.data = fdata.into_iter().flat_map(|f| f.to_le_bytes()).collect();
            }
        }
    }

    /// Infer architecture from tensor naming patterns when no explicit hint is available.
    pub fn infer_architecture(&self) -> String {
        let names: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();

        // Llama-like: model.layers.*, layers.*, transformer.h.* with rotary/attention names
        if names
            .iter()
            .any(|n| n.starts_with("model.layers.") || n.starts_with("layers."))
            && names
                .iter()
                .any(|n| n.contains("rotary") || n.contains("attention") || n.contains("q_proj"))
        {
            return "llama".to_string();
        }

        // GPT2-like: transformer.h.*, transformer.wte, transformer.wpe
        if names.iter().any(|n| n.starts_with("transformer.h."))
            || (names.iter().any(|n| n.contains("wte")) && names.iter().any(|n| n.contains("wpe")))
        {
            return "gpt2".to_string();
        }

        // BERT-like: embeddings.*, encoder.layer.*
        if names.iter().any(|n| n.starts_with("embeddings."))
            && names.iter().any(|n| n.starts_with("encoder.layer."))
        {
            return "bert".to_string();
        }

        // MLP: layerN.weight / layerN.bias or single weight/bias pair
        if names
            .iter()
            .any(|n| n.starts_with("layer") && n.contains(".weight"))
            || (names.contains(&"weight") && names.contains(&"bias"))
        {
            return "mlp".to_string();
        }

        "generic".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(dtype: DataType, elements: usize) -> Model {
        let data = vec![0u8; dtype.byte_size() * elements];
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "w".to_string(),
            TensorData {
                shape: vec![elements],
                dtype,
                data,
            },
        );
        Model {
            name: "test".to_string(),
            architecture: "generic".to_string(),
            tensors,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn quant_preview_bytes_per_format() {
        let n = 256usize;
        assert_eq!(QuantPreview::F32.bytes_for(n), 1024);
        assert_eq!(QuantPreview::F16.bytes_for(n), 512);
        assert_eq!(QuantPreview::Int8.bytes_for(n), 256);
        assert_eq!(QuantPreview::Int4.bytes_for(n), 128);
        // Q4_0: 256 elements = 8 blocks × 18 bytes = 144.
        assert_eq!(QuantPreview::Q4_0.bytes_for(n), 144);
    }

    #[test]
    fn quant_preview_q4_0_rounds_up_per_block() {
        // 33 elements → 2 blocks (each holds up to 32) → 2 × 18 = 36 bytes.
        assert_eq!(QuantPreview::Q4_0.bytes_for(33), 36);
        assert_eq!(QuantPreview::Q4_0.bytes_for(32), 18);
        assert_eq!(QuantPreview::Q4_0.bytes_for(1), 18);
    }

    #[test]
    fn quant_preview_int4_rounds_up_odd() {
        assert_eq!(QuantPreview::Int4.bytes_for(3), 2);
        assert_eq!(QuantPreview::Int4.bytes_for(4), 2);
    }

    #[test]
    fn preview_size_bytes_matches_dtype_scaling() {
        // A 256-element F32 tensor (1024 bytes stored).
        let model = make_model(DataType::F32, 256);
        assert_eq!(model.preview_size_bytes(QuantPreview::F32), 1024);
        assert_eq!(model.preview_size_bytes(QuantPreview::F16), 512);
        assert_eq!(model.preview_size_bytes(QuantPreview::Int8), 256);
        assert_eq!(model.preview_size_bytes(QuantPreview::Q4_0), 144);
    }

    #[test]
    fn preview_size_bytes_is_shape_based_not_storage_based() {
        // Store as I8 (256 bytes) but the F32 preview reflects 256 elements × 4 = 1024.
        let model = make_model(DataType::I8, 256);
        assert_eq!(model.preview_size_bytes(QuantPreview::F32), 1024);
        assert_eq!(model.total_bytes(), 256, "stored bytes are the I8 size");
    }

    #[test]
    fn all_formats_returns_expected_set() {
        let all = QuantPreview::all();
        assert_eq!(all.len(), 5);
        assert_eq!(all[0], QuantPreview::F32);
        assert_eq!(all[4], QuantPreview::Q4_0);
        let labels: Vec<&str> = all.iter().map(|q| q.label()).collect();
        assert_eq!(labels, vec!["fp32", "fp16", "int8", "int4", "q4_0"]);
    }
}

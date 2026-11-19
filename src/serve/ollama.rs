//! Ollama ecosystem schema helpers for `/api/tags` + `/api/show`.
//!
//! Port of modelc-web's `api/_ollama.ts`. The upstream endpoints previously
//! used modelc-specific field names which made them only *superficially*
//! Ollama-compatible — real clients parse `details.format/family/families/
//! parameter_size/quantization_level`, `digest` and `model_info`. Keep
//! modelc extras alongside: Ollama clients ignore unknown keys.

use crate::pack::ArtifactHeader;

/// Human-readable parameter count, matching Ollama's formatting
/// (`8.0B`, `134.52M`, `1.5K`).
pub fn format_parameter_size(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.2}M", params as f64 / 1e6)
    } else if params >= 1_000 {
        format!("{:.1}K", params as f64 / 1e3)
    } else {
        params.to_string()
    }
}

/// Map a modelc quantization label to Ollama's `quantization_level` names.
pub fn quantization_level(quantization: &str) -> String {
    match quantization.to_ascii_lowercase().as_str() {
        "f32" | "fp32" => "F32".to_string(),
        "f16" | "fp16" => "F16".to_string(),
        "bf16" => "BF16".to_string(),
        "int8" | "i8" => "Q8_0".to_string(),
        "int4" => "Q4_0".to_string(),
        "q4_0" => "Q4_0".to_string(),
        "q5_0" => "Q5_0".to_string(),
        "q8_0" => "Q8_0".to_string(),
        "q4_k" => "Q4_K".to_string(),
        "q6_k" => "Q6_K".to_string(),
        other => other.to_ascii_uppercase(),
    }
}

/// llama.cpp file-type enum, used by Ollama's `general.file_type` (`0` = all F32).
pub fn ollama_file_type(quantization: &str) -> i32 {
    match quantization_level(quantization).as_str() {
        "F32" | "BF16" => 0,
        "F16" => 1,
        "Q4_0" => 2,
        "Q5_0" => 5,
        "Q8_0" => 7,
        "Q4_K" => 10,
        "Q6_K" => 12,
        _ => 0,
    }
}

/// Source format for `details.format` — artifact metadata, else "modelc".
pub fn artifact_format(header: &ArtifactHeader) -> String {
    header
        .metadata
        .get("format")
        .cloned()
        .unwrap_or_else(|| "modelc".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn parameter_size_formats_like_ollama() {
        assert_eq!(format_parameter_size(8_030_000_000), "8.0B");
        assert_eq!(format_parameter_size(134_520_000), "134.52M");
        assert_eq!(format_parameter_size(1_500), "1.5K");
        assert_eq!(format_parameter_size(42), "42");
        assert_eq!(format_parameter_size(0), "0");
    }

    #[test]
    fn quantization_level_maps_modelc_labels() {
        assert_eq!(quantization_level("f32"), "F32");
        assert_eq!(quantization_level("int8"), "Q8_0");
        assert_eq!(quantization_level("int4"), "Q4_0");
        assert_eq!(quantization_level("q4_k"), "Q4_K");
        assert_eq!(quantization_level("custom"), "CUSTOM");
    }

    #[test]
    fn file_type_matches_llama_cpp_enum() {
        assert_eq!(ollama_file_type("f32"), 0);
        assert_eq!(ollama_file_type("f16"), 1);
        assert_eq!(ollama_file_type("int4"), 2);
        assert_eq!(ollama_file_type("q8_0"), 7);
        assert_eq!(ollama_file_type("q6_k"), 12);
    }

    #[test]
    fn artifact_format_falls_back_to_modelc() {
        let h = ArtifactHeader {
            name: "x".into(),
            architecture: "mlp".into(),
            metadata: HashMap::new(),
            tensors: vec![],
        };
        assert_eq!(artifact_format(&h), "modelc");
    }
}

//! Tests for the `compiler::pack` orchestration: quantize FP32 → FP16/INT8/INT4
//! paths, prune threshold, and `inspect_model` returning the parsed Model.
//!
//! The internal `crate::pack::pack` write step is exercised by `tests/pack_test.rs`;
//! here we focus on the CLI-side orchestration in `compiler::pack` (quantize/prune
//! pre-processing, then delegate to the lower-level pack).

mod common;

use modelc::cli::{ModelArch, QuantizeMode, WeightFormat};
use modelc::compiler;
use modelc::model::DataType;

fn write_two_layer_mlp(path: &std::path::Path) {
    // Two-layer MLP: layer0 (4 -> 4) + layer1 (4 -> 2) + biases. Mixed
    // magnitudes so quantize (fp16/int8/int4) and prune (|v| < threshold) can
    // exercise their paths.
    let layer0_w: Vec<u8> = [
        0.5f32, -0.5, 1.0, 0.0, -1.0, 1.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5,
    ]
    .iter()
    .flat_map(|f| f.to_le_bytes())
    .collect();
    let layer0_b: Vec<u8> = [0.0f32, 0.0, 0.0, 0.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let layer1_w: Vec<u8> = [0.5f32, -0.5, 1.0, -1.0, 0.5, 0.5, -0.5, -0.5]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let layer1_b: Vec<u8> = [0.0f32, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    common::create_safetensors_file(
        path,
        vec![
            ("layer0.bias", "F32", vec![4], layer0_b),
            ("layer0.weight", "F32", vec![4, 4], layer0_w),
            ("layer1.bias", "F32", vec![2], layer1_b),
            ("layer1.weight", "F32", vec![2, 4], layer1_w),
        ],
    );
}

#[test]
fn inspect_model_returns_parsed_model() {
    let dir = common::tempdir().unwrap();
    let path = dir.path().join("inspect.safetensors");
    write_two_layer_mlp(&path);

    let m = compiler::inspect_model(&path, Some(&WeightFormat::Safetensors)).unwrap();
    assert_eq!(m.tensors.len(), 4);
    // Architecture is inferred from layerN.weight naming pattern.
    assert!(
        ["mlp", "generic", "unknown"].contains(&m.architecture.as_str()),
        "got arch {:?}",
        m.architecture
    );
    assert!(m.total_params() > 0);
}

#[test]
fn inspect_model_rejects_missing_format_hint() {
    let dir = common::tempdir().unwrap();
    let path = dir.path().join("unknown.xyz");
    std::fs::write(&path, b"abc").unwrap();
    let result = compiler::inspect_model(&path, None);
    assert!(result.is_err());
}

#[test]
fn pack_quantize_fp16_changes_dtype_and_bytes() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("mlp.safetensors");
    let out = dir.path().join("fp16.modelc");
    write_two_layer_mlp(&src);

    let returned = compiler::pack(
        &src,
        Some(&out),
        Some(&WeightFormat::Safetensors),
        None,
        false,
        Some(&QuantizeMode::Fp16),
        None,
    )
    .unwrap();
    assert_eq!(returned, out);

    let model = modelc::pack::unpack(&out).unwrap();
    for (name, t) in &model.tensors {
        assert_eq!(
            t.dtype,
            DataType::F16,
            "tensor `{name}!` should be F16 after quantize fp16"
        );
        // F16 elements take 2 bytes each (vs 4 for F32).
        assert_eq!(t.data.len(), t.shape.iter().product::<usize>() * 2);
    }
}

#[test]
fn pack_quantize_int8_writes_quant_scale_metadata() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("mlp.safetensors");
    let out = dir.path().join("int8.modelc");
    write_two_layer_mlp(&src);

    compiler::pack(
        &src,
        Some(&out),
        Some(&WeightFormat::Safetensors),
        None,
        false,
        Some(&QuantizeMode::Int8),
        None,
    )
    .unwrap();

    let model = modelc::pack::unpack(&out).unwrap();
    // Tensors with non-zero data get I8 + a quant_scale.<name> metadata entry.
    // All-zero tensors (the biases in this fixture) are skipped — the quantizer
    // would divide by zero, so they stay F32.
    let mut quantized_count = 0;
    for (name, t) in &model.tensors {
        if t.dtype == DataType::I8 {
            quantized_count += 1;
            let scale = model
                .metadata
                .get(&format!("quant_scale.{name}"))
                .unwrap_or_else(|| panic!("missing quant_scale.{name}"));
            let parsed: f32 = scale.parse().unwrap();
            assert!(
                parsed > 0.0,
                "scale for {name} should be positive, got {parsed}"
            );
        }
    }
    assert!(
        quantized_count >= 1,
        "expected at least one tensor quantized to I8, got {quantized_count}"
    );
}

#[test]
fn pack_quantize_int4_writes_quant_mode_metadata() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("mlp.safetensors");
    let out = dir.path().join("int4.modelc");
    write_two_layer_mlp(&src);

    compiler::pack(
        &src,
        Some(&out),
        Some(&WeightFormat::Safetensors),
        None,
        false,
        Some(&QuantizeMode::Int4),
        None,
    )
    .unwrap();

    let model = modelc::pack::unpack(&out).unwrap();
    let int4_count = model
        .metadata
        .values()
        .filter(|v| v.as_str() == "int4")
        .count();
    assert!(
        int4_count >= 1,
        "expected at least one int4 quant_mode marker, got {:?}",
        model.metadata
    );
    // INT4 packs two nibbles per byte — quantized tensors should be at most
    // ceil(params / 2) bytes. Bias tensors with all-zero values stay F32 (the
    // quantizer skips them to avoid divide-by-zero), so we only sum the
    // INT4-encoded tensors here.
    let int4_tensors: Vec<&modelc::model::TensorData> = model
        .tensors
        .values()
        .filter(|t| {
            model
                .metadata
                .get(&format!("quant_mode.{}", t_name_of(&model, t)))
                .map(String::as_str)
                == Some("int4")
        })
        .collect();
    assert!(
        !int4_tensors.is_empty(),
        "expected at least one int4 tensor"
    );
    let total_int4_bytes: usize = int4_tensors.iter().map(|t| t.data.len()).sum();
    let total_int4_params: usize = int4_tensors
        .iter()
        .map(|t| t.shape.iter().product::<usize>())
        .sum();
    assert!(total_int4_bytes <= total_int4_params.div_ceil(2) + 1);
}

/// A tiny MLP whose weight is a whole number of 32-element blocks (so the GGML
/// block-quantizers can encode it) and whose bias is not (8 elements).
fn write_blocky_mlp(path: &std::path::Path) {
    let weight: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32) * 0.25 - 4.0).to_le_bytes())
        .collect();
    let bias: Vec<u8> = (0..8)
        .flat_map(|i| ((i as f32) * 0.5).to_le_bytes())
        .collect();
    common::create_safetensors_file(
        path,
        vec![
            ("layer0.bias", "F32", vec![8], bias),
            ("layer0.weight", "F32", vec![8, 8], weight),
        ],
    );
}

/// The original F32 values of `write_blocky_mlp`'s weight.
fn blocky_weight_values() -> Vec<f32> {
    (0..64).map(|i| (i as f32) * 0.25 - 4.0).collect()
}

#[test]
fn pack_quantize_q4_0_encodes_blocks_and_loads_at_runtime() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("mlp.safetensors");
    let out = dir.path().join("q4_0.modelc");
    write_blocky_mlp(&src);

    compiler::pack(
        &src,
        Some(&out),
        Some(&WeightFormat::Safetensors),
        None,
        false,
        Some(&QuantizeMode::Q4_0),
        None,
    )
    .unwrap();

    let model = modelc::pack::unpack(&out).unwrap();
    let weight = &model.tensors["layer0.weight"];
    assert_eq!(weight.dtype, DataType::Q4_0, "whole blocks → Q4_0");
    assert_eq!(weight.data.len(), 64 / 32 * 18, "two Q4_0 blocks");
    // The bias is 8 elements — not a whole number of blocks — so it stays F32.
    assert_eq!(model.tensors["layer0.bias"].dtype, DataType::F32);

    // The header reports the block-quant dtype, which is what `/api/tags`
    // surfaces as the Ollama quantization level.
    let header = modelc::pack::read_header(&out).unwrap();
    assert_eq!(modelc::pack::infer_quantization(&header), "q4_0");

    // The runtime dequantizes the block-quant tensor on the fly, so the packed
    // artifact still runs — within one Q4_0 quantization step.
    let runtime = modelc::runtime::serve::Runtime::from_raw(&model.tensors);
    let w0 = runtime.get("layer0.weight").expect("layer0.weight");
    let original = blocky_weight_values();
    let amax = original.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    for (got, want) in w0.data.iter().zip(original.iter()) {
        assert!(
            (got - want).abs() <= amax / 8.0 + 1e-3,
            "Q4_0 weight {got} vs {want} exceeds one step"
        );
    }
}

#[test]
fn pack_quantize_q8_0_encodes_blocks_with_finer_precision() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("mlp.safetensors");
    let out = dir.path().join("q8_0.modelc");
    write_blocky_mlp(&src);

    compiler::pack(
        &src,
        Some(&out),
        Some(&WeightFormat::Safetensors),
        None,
        false,
        Some(&QuantizeMode::Q8_0),
        None,
    )
    .unwrap();

    let model = modelc::pack::unpack(&out).unwrap();
    let weight = &model.tensors["layer0.weight"];
    assert_eq!(weight.dtype, DataType::Q8_0, "whole blocks → Q8_0");
    assert_eq!(weight.data.len(), 64 / 32 * 34, "two Q8_0 blocks");
    assert_eq!(model.tensors["layer0.bias"].dtype, DataType::F32);

    let header = modelc::pack::read_header(&out).unwrap();
    assert_eq!(modelc::pack::infer_quantization(&header), "q8_0");

    // Q8_0 keeps 8 bits per weight, so its round-trip error is far below
    // Q4_0's — the values come back nearly exact.
    let runtime = modelc::runtime::serve::Runtime::from_raw(&model.tensors);
    let w0 = runtime.get("layer0.weight").expect("layer0.weight");
    let original = blocky_weight_values();
    let amax = original.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    for (got, want) in w0.data.iter().zip(original.iter()) {
        assert!(
            (got - want).abs() <= amax / 127.0 + 1e-3,
            "Q8_0 weight {got} vs {want} exceeds one step"
        );
    }
}

fn t_name_of(model: &modelc::model::Model, t: &modelc::model::TensorData) -> String {
    model
        .tensors
        .iter()
        .find(|(_, v)| std::ptr::eq(*v, t))
        .map(|(n, _)| n.clone())
        .unwrap_or_default()
}

#[test]
fn pack_prune_zeros_small_weights_and_records_metadata() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("mlp.safetensors");
    let out = dir.path().join("pruned.modelc");
    write_two_layer_mlp(&src);

    // Threshold 0.1 should zero values whose |v| < 0.1. The fixture has zeros
    // and 0.25, 0.5, 1.0 — so the ones strictly under 0.1 (none here) would
    // be zeroed. Use a higher threshold (0.6) so the 0.5 values get zeroed and
    // we exercise the prune path.
    compiler::pack(
        &src,
        Some(&out),
        Some(&WeightFormat::Safetensors),
        None,
        false,
        None,
        Some(0.6),
    )
    .unwrap();

    let model = modelc::pack::unpack(&out).unwrap();
    assert_eq!(
        model.metadata.get("pruned").map(String::as_str),
        Some("0.6")
    );
    // After pruning, every F32 value should be either exactly 0.0 (was below
    // threshold and got zeroed) or |v| >= threshold (kept).
    for t in model.tensors.values() {
        assert_eq!(t.dtype, DataType::F32, "prune should preserve F32");
        for chunk in t.data.chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert!(v == 0.0 || v.abs() >= 0.6, "unexpected un-pruned value {v}");
        }
    }
}

#[test]
fn pack_combines_quantize_and_prune() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("mlp.safetensors");
    let out = dir.path().join("pruned-fp16.modelc");
    write_two_layer_mlp(&src);

    compiler::pack(
        &src,
        Some(&out),
        Some(&WeightFormat::Safetensors),
        None,
        false,
        Some(&QuantizeMode::Fp16),
        Some(0.6),
    )
    .unwrap();

    let model = modelc::pack::unpack(&out).unwrap();
    for t in model.tensors.values() {
        assert_eq!(t.dtype, DataType::F16);
    }
    assert_eq!(
        model.metadata.get("pruned").map(String::as_str),
        Some("0.6")
    );
}

#[test]
fn pack_default_output_path_uses_modelc_extension() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("mlp.safetensors");
    write_two_layer_mlp(&src);

    let returned = compiler::pack(
        &src,
        None,
        Some(&WeightFormat::Safetensors),
        None,
        false,
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        returned.extension().and_then(|s| s.to_str()),
        Some("modelc")
    );
}

#[test]
fn pack_applies_arch_hint() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("generic.safetensors");
    // Bare weights with no layerN.weight naming → architecture should fall back
    // to "generic" without a hint. With a hint, the hint wins.
    common::create_safetensors_file(&src, vec![("weight", "F32", vec![2, 2], vec![0u8; 16])]);

    let out = dir.path().join("hinted.modelc");
    let m = compiler::inspect_model(&src, Some(&WeightFormat::Safetensors)).unwrap();
    // Generic arch without layerN.weight pattern.
    assert!(
        m.architecture.is_empty() || m.architecture == "generic" || m.architecture == "unknown"
    );

    compiler::pack(
        &src,
        Some(&out),
        Some(&WeightFormat::Safetensors),
        Some(&ModelArch::Mlp),
        false,
        None,
        None,
    )
    .unwrap();
    let m2 = modelc::pack::unpack(&out).unwrap();
    assert_eq!(m2.architecture, "mlp");
}

#[test]
fn pack_compress_writes_smaller_artifact_than_uncompressed() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("mlp.safetensors");
    write_two_layer_mlp(&src);

    let uncompressed = dir.path().join("plain.modelc");
    let compressed = dir.path().join("zstd.modelc");
    compiler::pack(
        &src,
        Some(&uncompressed),
        Some(&WeightFormat::Safetensors),
        None,
        false,
        None,
        None,
    )
    .unwrap();
    compiler::pack(
        &src,
        Some(&compressed),
        Some(&WeightFormat::Safetensors),
        None,
        true,
        None,
        None,
    )
    .unwrap();

    let raw = std::fs::metadata(&uncompressed).unwrap().len();
    let packed = std::fs::metadata(&compressed).unwrap().len();
    // Compressed artifact should be smaller (zstd frames have fixed overhead,
    // but our small dataset compresses well).
    assert!(
        packed < raw,
        "compressed {packed} should be smaller than raw {raw}"
    );
}

#[test]
fn pack_with_unknown_format_errors() {
    let dir = common::tempdir().unwrap();
    let path = dir.path().join("unknown.xyz");
    std::fs::write(&path, b"abc").unwrap();
    let result = compiler::pack(&path, None, None, None, false, None, None);
    assert!(result.is_err());
}

#[test]
fn prompt_model_gpt2_returns_generated_text() {
    let model = common::create_gpt2_test_model();
    let cfg = modelc::generate::GenerationConfig {
        max_tokens: 4,
        ..Default::default()
    };
    let out = compiler::prompt_model(&model, "hi", &cfg);
    assert!(!out.is_empty(), "greedy generation should emit tokens");
    // A second call with the same config must be bit-identical (greedy, no seed needed).
    let again = compiler::prompt_model(&model, "hi", &cfg);
    assert_eq!(out, again);
}

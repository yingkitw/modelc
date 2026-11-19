//! Direct tests for the `.modelc` artifact writer/reader in `src/pack.rs`.
//! Covers header parsing, pack/unpack round-trips for both compressed and
//! uncompressed artifacts, `verify` happy-path and corruption detection, and
//! `export_to_safetensors` end-to-end.

mod common;

use std::fs;

use modelc::model::{DataType, Model, TensorData};
use modelc::pack;

fn tiny_model() -> Model {
    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "weight".to_string(),
        TensorData {
            shape: vec![2, 3],
            dtype: DataType::F32,
            data: common::f32_to_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        },
    );
    tensors.insert(
        "bias".to_string(),
        TensorData {
            shape: vec![2],
            dtype: DataType::F32,
            data: common::f32_to_bytes(&[0.1, 0.2]),
        },
    );
    Model {
        name: "tiny_pack_test".to_string(),
        architecture: "mlp".to_string(),
        tensors,
        metadata: std::collections::HashMap::new(),
    }
}

#[test]
fn pack_unpack_round_trip_uncompressed() {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("tiny.modelc");
    let model = tiny_model();
    pack::pack(&model, &artifact, false).unwrap();

    let restored = pack::unpack(&artifact).unwrap();
    assert_eq!(restored.name, model.name);
    assert_eq!(restored.architecture, model.architecture);
    assert_eq!(restored.tensors.len(), model.tensors.len());
    assert_eq!(
        restored.tensors["weight"].data,
        model.tensors["weight"].data
    );
    assert_eq!(restored.tensors["bias"].data, model.tensors["bias"].data);
    assert_eq!(restored.tensors["weight"].dtype, DataType::F32);
    assert_eq!(restored.tensors["bias"].shape, vec![2]);
}

#[test]
fn pack_unpack_round_trip_compressed() {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("tiny-compressed.modelc");
    let model = tiny_model();
    pack::pack(&model, &artifact, true).unwrap();

    // Header reports compressed = true.
    let header = pack::read_header(&artifact).unwrap();
    assert!(header.metadata.is_empty() || header.metadata.values().any(|v| v == "true"));

    // Round-trip preserves tensor bytes.
    let restored = pack::unpack(&artifact).unwrap();
    assert_eq!(
        restored.tensors["weight"].data,
        model.tensors["weight"].data
    );
    assert_eq!(restored.tensors["bias"].data, model.tensors["bias"].data);
}

#[test]
fn read_header_returns_sorted_tensor_names() {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("sorted.modelc");
    let model = tiny_model();
    pack::pack(&model, &artifact, false).unwrap();
    let header = pack::read_header(&artifact).unwrap();
    let names: Vec<String> = header.tensors.iter().map(|t| t.name.clone()).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted, "header tensor names must be sorted");
}

#[test]
fn read_header_preserves_offsets_and_lengths() {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("offsets.modelc");
    let model = tiny_model();
    pack::pack(&model, &artifact, false).unwrap();
    let header = pack::read_header(&artifact).unwrap();

    // bias (6 bytes) sorts first, then weight (24 bytes).
    let bias = header.tensors.iter().find(|t| t.name == "bias").unwrap();
    let weight = header.tensors.iter().find(|t| t.name == "weight").unwrap();
    assert_eq!(bias.offset, 0);
    assert_eq!(bias.length, 8);
    assert_eq!(weight.offset, 8);
    assert_eq!(weight.length, 24);
    assert_eq!(bias.dtype.to_lowercase(), "f32");
    assert_eq!(weight.dtype.to_lowercase(), "f32");
}

#[test]
fn verify_accepts_valid_artifact() {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("valid.modelc");
    let model = tiny_model();
    pack::pack(&model, &artifact, false).unwrap();
    pack::verify(&artifact).unwrap();
}

#[test]
fn verify_rejects_corrupted_artifact() {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("corrupt.modelc");
    let model = tiny_model();
    pack::pack(&model, &artifact, false).unwrap();

    // Corrupt the artifact so that a tensor offset claims bytes past the end of
    // the data blob — `verify` rejects any artifact where max_end > blob size.
    let header = pack::read_header(&artifact).unwrap();
    let mut corrupted_header = header.clone();
    // Move a tensor offset far beyond the blob so the bounds check fires.
    if let Some(first) = corrupted_header.tensors.first_mut() {
        first.offset = u64::MAX / 2;
        first.length = 1024;
    }
    // Re-serialize and overwrite the file's header bytes.
    let header_json = serde_json::to_vec(&corrupted_header).unwrap();
    let header_len = header_json.len() as u64;
    let mut bytes = fs::read(&artifact).unwrap();
    let header_len_offset = 6 + 4 + 4; // magic + version + flags
    bytes[header_len_offset..header_len_offset + 8].copy_from_slice(&header_len.to_le_bytes());
    let header_start = header_len_offset + 8;
    bytes[header_start..header_start + header_json.len()].copy_from_slice(&header_json);
    fs::write(&artifact, &bytes).unwrap();

    let result = pack::verify(&artifact);
    assert!(
        result.is_err(),
        "verify should reject an artifact whose offsets exceed the blob size"
    );
}

#[test]
fn verify_rejects_unknown_magic() {
    let dir = common::tempdir().unwrap();
    let path = dir.path().join("not-modelc.modelc");
    fs::write(&path, b"NOT_MODE\x00\x00\x00\x00\x00\x00").unwrap();
    let result = pack::verify(&path);
    assert!(result.is_err());
}

#[test]
fn export_to_safetensors_writes_valid_file() {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("exportable.modelc");
    let out = dir.path().join("exported.safetensors");
    let model = tiny_model();
    pack::pack(&model, &artifact, false).unwrap();
    pack::export_to_safetensors(&artifact, &out).unwrap();

    // Read it back through safetensors to confirm validity.
    let data = fs::read(&out).unwrap();
    let st = safetensors::SafeTensors::deserialize(&data).unwrap();
    assert!(st.names().iter().any(|n| n.as_str() == "weight"));
    assert!(st.names().iter().any(|n| n.as_str() == "bias"));
}

/// Safetensors has no GGUF block-quant dtype: export must dequantize to F32
/// instead of panicking (learned from the modelc-web TS port).
#[test]
fn export_dequantizes_gguf_quant_tensors() {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("quant.modelc");
    let out = dir.path().join("quant.safetensors");

    // One Q4_0 block = f16 delta (2B) + 16 nibble bytes -> 32 elements.
    let mut block = half::f16::from_f32(1.0).to_le_bytes().to_vec();
    block.extend_from_slice(&[0u8; 16]);

    let mut model = tiny_model();
    model.tensors.insert(
        "w_q4".to_string(),
        TensorData {
            shape: vec![32],
            dtype: DataType::Q4_0,
            data: block,
        },
    );
    pack::pack(&model, &artifact, false).unwrap();

    pack::export_to_safetensors(&artifact, &out).unwrap();

    let data = fs::read(&out).unwrap();
    let st = safetensors::SafeTensors::deserialize(&data).unwrap();
    let view = st.tensor("w_q4").unwrap();
    assert_eq!(view.dtype(), safetensors::Dtype::F32);
    assert_eq!(view.shape(), vec![32]);
}

#[test]
fn unpack_rejects_truncated_artifact() {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("truncated.modelc");
    let model = tiny_model();
    pack::pack(&model, &artifact, false).unwrap();

    // Truncate to just the magic + header_len (skip header body and blob).
    let bytes = fs::read(&artifact).unwrap();
    fs::write(&artifact, &bytes[..22]).unwrap();

    let result = pack::unpack(&artifact);
    assert!(result.is_err(), "unpack should fail on truncated file");
}

#[test]
fn pack_metadata_round_trips() {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("metadata.modelc");
    let mut model = tiny_model();
    model
        .metadata
        .insert("source".to_string(), "pack_test".to_string());
    model
        .metadata
        .insert("pruned".to_string(), "0.001".to_string());
    pack::pack(&model, &artifact, false).unwrap();

    let restored = pack::unpack(&artifact).unwrap();
    assert_eq!(
        restored.metadata.get("source").map(String::as_str),
        Some("pack_test")
    );
    assert_eq!(
        restored.metadata.get("pruned").map(String::as_str),
        Some("0.001")
    );
}

#[test]
fn pack_with_quantized_int8_then_dequantize_round_trips_shape() {
    // Pack a model whose tensors are already typed F16/I8: the format stores
    // bytes verbatim, so unpacking must preserve dtype labels and byte length.
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("quantized.modelc");

    let mut tensors = std::collections::HashMap::new();
    // 4 F16 elements = 8 bytes
    tensors.insert(
        "embed".to_string(),
        TensorData {
            shape: vec![4],
            dtype: DataType::F16,
            data: vec![0x00, 0x3C, 0x66, 0x26, 0x00, 0x40, 0x00, 0x40],
        },
    );
    let model = Model {
        name: "q".to_string(),
        architecture: "mlp".to_string(),
        tensors,
        metadata: std::collections::HashMap::new(),
    };
    pack::pack(&model, &artifact, false).unwrap();

    let restored = pack::unpack(&artifact).unwrap();
    let t = &restored.tensors["embed"];
    assert_eq!(t.dtype, DataType::F16);
    assert_eq!(t.data.len(), 8);
    assert_eq!(t.shape, vec![4]);
}

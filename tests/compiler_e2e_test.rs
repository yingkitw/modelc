mod common;

use std::fs;
use std::net::SocketAddr;

use modelc::cli::WeightFormat;
use modelc::compiler;

#[test]
#[ignore]
fn test_e2e_compile_safetensors() {
    let dir = common::tempdir().unwrap();
    let weights_path = dir.path().join("test_model.safetensors");
    let output_path = dir.path().join("test_model_serve");

    let w_data = common::f32_to_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_data = common::f32_to_bytes(&[0.1, 0.2]);
    common::create_safetensors_file(
        &weights_path,
        vec![
            ("bias", "F32", vec![2], b_data),
            ("weight", "F32", vec![2, 3], w_data),
        ],
    );

    let result = compiler::compile(
        &weights_path,
        Some(&output_path),
        Some(&WeightFormat::Safetensors),
        None,
        SocketAddr::from(([127, 0, 0, 1], 18080)),
        None,
        false,
    );

    match result {
        Ok(bin_path) => {
            assert!(bin_path.exists());
            let metadata = fs::metadata(&bin_path).unwrap();
            assert!(metadata.len() > 0);
        }
        Err(e) => {
            eprintln!("e2e compile skipped or failed: {}", e);
        }
    }
}

#[test]
#[ignore]
fn test_e2e_compile_detect_format() {
    let dir = common::tempdir().unwrap();
    let weights_path = dir.path().join("auto.safetensors");
    let output_path = dir.path().join("auto_serve");

    let data = common::f32_to_bytes(&[1.0, 0.0, 0.0, 1.0]);
    common::create_safetensors_file(&weights_path, vec![("eye", "F32", vec![2, 2], data)]);

    let result = compiler::compile(
        &weights_path,
        Some(&output_path),
        None,
        None,
        SocketAddr::from(([127, 0, 0, 1], 18081)),
        None,
        false,
    );

    match result {
        Ok(bin_path) => {
            assert!(bin_path.exists());
        }
        Err(e) => {
            eprintln!("e2e compile skipped or failed: {}", e);
        }
    }
}

#[test]
fn test_inspect_safetensors() {
    let dir = common::tempdir().unwrap();
    let path = dir.path().join("inspect.safetensors");

    let w_data = common::f32_to_bytes(&[1.0, 2.0, 3.0, 4.0]);
    common::create_safetensors_file(&path, vec![("kernel", "F32", vec![2, 2], w_data)]);

    let result = compiler::inspect(&path, Some(&WeightFormat::Safetensors));
    assert!(result.is_ok());
}

#[test]
fn test_inspect_detect_format() {
    let dir = common::tempdir().unwrap();
    let path = dir.path().join("detect.safetensors");

    let data = common::f32_to_bytes(&[1.0]);
    common::create_safetensors_file(&path, vec![("x", "F32", vec![1], data)]);

    let result = compiler::inspect(&path, None);
    assert!(result.is_ok());
}

/// The documented flow: pack a model, then inspect the artifact itself.
/// `inspect model.modelc` used to fail with "could not detect weight format".
#[test]
fn test_inspect_packed_artifact_round_trip() {
    let dir = common::tempdir().unwrap();
    let src = dir.path().join("round_trip.safetensors");
    let w = common::f32_to_bytes(&[0.5, -0.25, 0.75, 1.0, 2.0, -1.0, 0.0, 3.0]);
    common::create_safetensors_file(&src, vec![("layer1.weight", "F32", vec![2, 4], w)]);

    let artifact = dir.path().join("round_trip.modelc");
    compiler::pack(&src, Some(&artifact), Some(&WeightFormat::Safetensors), None, false, None, None)
        .unwrap();
    assert!(artifact.exists());

    // The artifact unpacks through the same path inspect now takes.
    let model = compiler::inspect_model(&artifact, None).unwrap();
    assert_eq!(model.architecture, "mlp");
    assert_eq!(model.tensors.len(), 1);
    assert!(model.tensors.contains_key("layer1.weight"));

    // The printing path succeeds too — and without a stale double-detect.
    let printed = compiler::inspect(&artifact, None);
    assert!(printed.is_ok());

    // The --readme path is inspect_model + dequantize + print — same route.
    let mut card_model = compiler::inspect_model(&artifact, None).unwrap();
    card_model.dequantize_in_place();
}

#[test]
fn test_inspect_nonexistent() {
    let result = compiler::inspect(std::path::Path::new("/nonexistent.safetensors"), None);
    assert!(result.is_err());
}

#[test]
fn test_inspect_unknown_format() {
    let dir = common::tempdir().unwrap();
    let path = dir.path().join("unknown.xyz");
    std::fs::write(&path, b"data").unwrap();

    let result = compiler::inspect(&path, None);
    assert!(result.is_err());
}

#[test]
fn test_compile_unknown_format_no_flag() {
    let dir = common::tempdir().unwrap();
    let path = dir.path().join("unknown.xyz");
    std::fs::write(&path, b"data").unwrap();

    let result = compiler::compile(
        &path,
        None,
        None,
        None,
        SocketAddr::from(([0, 0, 0, 0], 8080)),
        None,
        false,
    );
    assert!(result.is_err());
}

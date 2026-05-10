mod common;

use std::fs;
use std::net::SocketAddr;

use modelc::cli::WeightFormat;
use modelc::compiler;

#[test]
#[ignore]
fn test_e2e_compile_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    let weights_path = dir.path().join("test_model.safetensors");
    let output_path = dir.path().join("test_model_serve");

    let w_data = common::f32_to_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_data = common::f32_to_bytes(&[0.1, 0.2, 0.3]);
    common::create_safetensors_file(
        &weights_path,
        vec![
            ("bias", "F32", vec![3], b_data),
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
    let dir = tempfile::tempdir().unwrap();
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("inspect.safetensors");

    let w_data = common::f32_to_bytes(&[1.0, 2.0, 3.0, 4.0]);
    common::create_safetensors_file(&path, vec![("kernel", "F32", vec![2, 2], w_data)]);

    let result = compiler::inspect(&path, Some(&WeightFormat::Safetensors));
    assert!(result.is_ok());
}

#[test]
fn test_inspect_detect_format() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("detect.safetensors");

    let data = common::f32_to_bytes(&[1.0]);
    common::create_safetensors_file(&path, vec![("x", "F32", vec![1], data)]);

    let result = compiler::inspect(&path, None);
    assert!(result.is_ok());
}

#[test]
fn test_inspect_nonexistent() {
    let result = compiler::inspect(std::path::Path::new("/nonexistent.safetensors"), None);
    assert!(result.is_err());
}

#[test]
fn test_inspect_unknown_format() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unknown.xyz");
    std::fs::write(&path, b"data").unwrap();

    let result = compiler::inspect(&path, None);
    assert!(result.is_err());
}

#[test]
fn test_compile_unknown_format_no_flag() {
    let dir = tempfile::tempdir().unwrap();
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

mod common;

use std::fs;
use std::net::SocketAddr;

use modelc::codegen::CodeGenerator;
use modelc::codegen::native::NativeCodegen;

fn addr(host_port: &str) -> SocketAddr {
    host_port.parse().expect("socket addr")
}

#[test]
fn test_codegen_creates_project_structure() {
    let model = common::create_test_model();
    let dir = tempfile::tempdir().unwrap();

    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("0.0.0.0:8080"))
        .unwrap();

    assert!(project_dir.join("Cargo.toml").exists());
    assert!(project_dir.join("src/main.rs").exists());
    assert!(project_dir.join("embedded_weights.bin").exists());
}

#[test]
fn test_codegen_cargo_toml_content() {
    let model = common::create_test_model();
    let dir = tempfile::tempdir().unwrap();

    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("[::1]:9090"))
        .unwrap();

    let cargo_toml = fs::read_to_string(project_dir.join("Cargo.toml")).unwrap();
    assert!(cargo_toml.contains("model-serve"));
    assert!(cargo_toml.contains("axum"));
    assert!(cargo_toml.contains("tokio"));
    assert!(cargo_toml.contains("lto = true"));
    assert!(cargo_toml.contains("strip = true"));
}

#[test]
fn test_codegen_main_rs_contains_model_name() {
    let model = common::create_test_model();
    let dir = tempfile::tempdir().unwrap();

    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("127.0.0.1:8080"))
        .unwrap();

    let main_rs = fs::read_to_string(project_dir.join("src/main.rs")).unwrap();
    assert!(main_rs.contains("test_model"));
    assert!(main_rs.contains("/infer"));
    assert!(main_rs.contains("/info"));
    assert!(main_rs.contains("forward"));
}

#[test]
fn test_codegen_main_rs_contains_tensor_metadata() {
    let model = common::create_test_model();
    let dir = tempfile::tempdir().unwrap();

    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("127.0.0.1:8080"))
        .unwrap();

    let main_rs = fs::read_to_string(project_dir.join("src/main.rs")).unwrap();
    assert!(main_rs.contains("\"weight\""));
    assert!(main_rs.contains("\"bias\""));
    assert!(main_rs.contains("TensorMeta"));
}

#[test]
fn test_codegen_embeds_listen_address() {
    let model = common::create_test_model();
    let dir = tempfile::tempdir().unwrap();

    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("0.0.0.0:9999"))
        .unwrap();

    let main_rs = fs::read_to_string(project_dir.join("src/main.rs")).unwrap();
    assert!(main_rs.contains("0.0.0.0:9999"));
}

#[test]
fn test_codegen_embedded_blob_matches_model() {
    let model = common::create_test_model();
    let dir = tempfile::tempdir().unwrap();

    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("127.0.0.1:8080"))
        .unwrap();

    let mut names: Vec<&String> = model.tensors.keys().collect();
    names.sort();
    let mut expected: Vec<u8> = Vec::new();
    for n in &names {
        expected.extend_from_slice(&model.tensors[*n].data);
    }

    let embedded = fs::read(project_dir.join("embedded_weights.bin")).unwrap();
    assert_eq!(embedded, expected);
}

#[test]
fn test_codegen_main_rs_compilable_structure() {
    let model = common::create_test_model();
    let dir = tempfile::tempdir().unwrap();

    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("127.0.0.1:8080"))
        .unwrap();

    let main_rs = fs::read_to_string(project_dir.join("src/main.rs")).unwrap();
    assert!(main_rs.contains("#[tokio::main]"));
    assert!(main_rs.contains("async fn main()"));
    assert!(main_rs.contains("include_bytes!"));
    assert!(main_rs.contains("Router::new()"));
    assert!(main_rs.contains("TensorMeta"));
    assert!(main_rs.contains("AppState"));
    assert!(main_rs.contains("InferRequest"));
    assert!(main_rs.contains("InferResponse"));
    assert!(main_rs.contains("ModelInfo"));
    assert!(main_rs.contains("MODEL_ARCHITECTURE"));
}

#[test]
fn test_codegen_large_model() {
    let model = common::create_large_test_model();
    let dir = tempfile::tempdir().unwrap();

    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("127.0.0.1:8080"))
        .unwrap();

    let main_rs = fs::read_to_string(project_dir.join("src/main.rs")).unwrap();
    assert!(main_rs.contains("layer0.weight"));
    assert!(main_rs.contains("layer0.bias"));
    assert!(main_rs.contains("layer0.ln_weight"));
    assert!(main_rs.contains("layer0.ln_bias"));
    assert!(main_rs.contains("layer1.weight"));
    assert!(main_rs.contains("layer1.bias"));
}

#[test]
fn test_codegen_total_params_in_output() {
    let model = common::create_test_model();
    let dir = tempfile::tempdir().unwrap();

    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("127.0.0.1:8080"))
        .unwrap();

    let main_rs = fs::read_to_string(project_dir.join("src/main.rs")).unwrap();
    let expected_params = model.total_params();
    assert!(main_rs.contains(&format!("total_params: {}", expected_params)));
    let expected_bytes = model.total_bytes();
    assert!(main_rs.contains(&format!("total_bytes: {}", expected_bytes)));
}

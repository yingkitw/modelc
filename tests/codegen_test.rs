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
    assert!(main_rs.contains("matmul_bias"));
    assert!(main_rs.contains("relu_inplace"));
}

#[test]
fn codegen_mlp_single_stack_emits_matmul() {
    let mut model = common::create_test_model();
    model.architecture = "mlp".to_string();
    let dir = tempfile::tempdir().unwrap();

    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("127.0.0.1:8080"))
        .unwrap();

    let main_rs = fs::read_to_string(project_dir.join("src/main.rs")).unwrap();
    assert!(main_rs.contains("matmul_bias"));
    assert!(main_rs.contains("decode_f32"));
}

#[test]
fn codegen_generic_arch_keeps_echo_forward() {
    let mut model = common::create_test_model();
    model.architecture = "generic".to_string();

    let dir = tempfile::tempdir().unwrap();
    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(&model, dir.path(), addr("127.0.0.1:8081"))
        .unwrap();

    let main_rs = fs::read_to_string(project_dir.join("src/main.rs")).unwrap();
    assert!(!main_rs.contains("matmul_bias"));
    assert!(main_rs.contains("input.to_vec()"));
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

// =============================================================================
// GPT-2 / LLaMA transformer codegen (previously broken stubs — now complete).
// =============================================================================

fn emitted_main(model: &modelc::model::Model) -> String {
    let dir = tempfile::tempdir().unwrap();
    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(model, dir.path(), addr("127.0.0.1:8080"))
        .unwrap();
    fs::read_to_string(project_dir.join("src/main.rs")).unwrap()
}

#[test]
fn codegen_gpt2_emits_complete_transformer_block() {
    let model = common::create_gpt2_test_model();
    let src = emitted_main(&model);

    // The original bug: stubs used `decode_f32` but never defined it. Now it must be defined.
    assert!(src.contains("fn decode_f32("));
    assert!(src.contains("fn decode_f32_le("));

    // Full FP32 toolbox required by the forward body.
    for needle in [
        "fn linear(",
        "fn layer_norm(",
        "fn gelu(",
        "fn add(",
        "fn single_token_attention(",
    ] {
        assert!(src.contains(needle), "gpt2 helpers missing `{needle}`");
    }

    // The entire per-layer op chain must be wired to concrete tensor names.
    for needle in [
        "transformer.h.0.ln_1.weight",
        "transformer.h.0.ln_1.bias",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_attn.bias",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.attn.c_proj.bias",
        "transformer.h.0.ln_2.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.ln_f.weight",
        "transformer.wte.weight",
    ] {
        assert!(
            src.contains(needle),
            "gpt2 forward missing tensor `{needle}`"
        );
    }

    // Structural correctness: QKV split, residual connections, GeLU activation.
    assert!(src.contains(".split_at("));
    assert!(src.contains("single_token_attention(q, k, v"));
    assert!(src.contains("let act = gelu(&fc);"));
    assert!(src.contains("add(&hidden, &proj)"));

    // Output projection via weight-tied wte (no lm_head.weight in this fixture).
    assert!(src.contains("\"transformer.wte.weight\""));
}

#[test]
fn codegen_llama_emits_complete_decoder_block() {
    let model = common::create_llama_test_model();
    let src = emitted_main(&model);

    assert!(src.contains("fn decode_f32("));

    for needle in [
        "fn linear(",
        "fn rms_norm(",
        "fn silu(",
        "fn rope(",
        "fn single_token_attention(",
        "fn add(",
    ] {
        assert!(src.contains(needle), "llama helpers missing `{needle}`");
    }

    for needle in [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ] {
        assert!(
            src.contains(needle),
            "llama forward missing tensor `{needle}`"
        );
    }

    // RoPE applied to both Q and K, SwiGLU gating, residual streams.
    assert!(src.contains("rope(&q, 0"));
    assert!(src.contains("rope(&k, 0"));
    assert!(src.contains("let act = silu(&gate);"));
    assert!(src.contains("a * b).collect()"));
    assert!(
        src.matches("add(&hidden, &proj)").count() >= 2,
        "llama forward should have >=2 residual adds (attn + mlp)"
    );
}

/// Real-world proof: the emitted standalone crate must `cargo check` cleanly. Ignored by default
/// since it requires `cargo` and (on first run) network access to fetch axum/tokio/serde.
#[test]
#[ignore]
fn codegen_gpt2_emitted_project_compiles() {
    check_emitted_project_compiles(&common::create_gpt2_test_model());
}

#[test]
#[ignore]
fn codegen_llama_emitted_project_compiles() {
    check_emitted_project_compiles(&common::create_llama_test_model());
}

fn check_emitted_project_compiles(model: &modelc::model::Model) {
    let dir = tempfile::tempdir().unwrap();
    let codegen = NativeCodegen;
    let project_dir = codegen
        .generate(model, dir.path(), addr("127.0.0.1:0"))
        .unwrap();

    let status = std::process::Command::new("cargo")
        .args(["check", "--offline"])
        .current_dir(&project_dir)
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("emitted project failed `cargo check` (exit {:?})", s.code()),
        Err(e) => eprintln!("skipping emitted-project compile check: cargo unavailable ({e})"),
    }
}

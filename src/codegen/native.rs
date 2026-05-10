use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::codegen::CodeGenerator;
use crate::model::Model;

const EMBEDDED_WEIGHTS_FILE: &str = "embedded_weights.bin";

pub struct NativeCodegen;

impl CodeGenerator for NativeCodegen {
    fn generate(&self, model: &Model, output_dir: &Path, listen: SocketAddr) -> Result<PathBuf> {
        let project_dir = output_dir.join("modelc_build");
        let src_dir = project_dir.join("src");

        fs::create_dir_all(&src_dir).with_context(|| "failed to create build directory")?;

        let mut names: Vec<&String> = model.tensors.keys().collect();
        names.sort();

        let mut blob: Vec<u8> = Vec::new();
        let mut tensor_loads = String::new();
        for name in &names {
            let tensor = model.tensors.get(*name).expect("tensor key mismatch");
            let offset = blob.len();
            let byte_len = tensor.data.len();
            blob.extend_from_slice(&tensor.data);
            let shape_fmt = format!("{:?}", tensor.shape);
            let dtype_size = tensor.dtype.byte_size();
            tensor_loads.push_str(&format!(
                "        ({:?}, TensorMeta {{ shape: &{shape_fmt}, dtype_size: {dtype_size}, byte_offset: {offset}, byte_len: {byte_len} }}),\n",
                name
            ));
        }

        fs::write(project_dir.join(EMBEDDED_WEIGHTS_FILE), &blob)
            .with_context(|| "failed to write embedded weight blob")?;

        let cargo_toml = generate_cargo_toml();
        let listen_str = listen.to_string();
        let main_rs = generate_main_rs(model, EMBEDDED_WEIGHTS_FILE, &tensor_loads, &listen_str);

        fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;
        fs::write(src_dir.join("main.rs"), main_rs)?;

        Ok(project_dir)
    }
}

fn generate_cargo_toml() -> String {
    r#"[package]
name = "model-serve"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[profile.release]
opt-level = 3
lto = true
strip = true
"#
    .to_string()
}

fn escape_rust_string_literal(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn generate_main_rs(model: &Model, weights_file: &str, tensor_loads: &str, listen: &str) -> String {
    let model_name_esc = escape_rust_string_literal(&model.name);
    let arch_esc = escape_rust_string_literal(&model.architecture);
    let listen_esc = escape_rust_string_literal(listen);

    let total_params = model.total_params();
    let total_bytes = model.total_bytes();

    format!(
        r##"use std::collections::HashMap;
use std::sync::Arc;

use axum::{{Json, Router, extract::State, routing::{{get, post}}}};
use serde::{{Deserialize, Serialize}};

struct TensorMeta {{
    shape: &'static [usize],
    dtype_size: usize,
    byte_offset: usize,
    byte_len: usize,
}}

struct AppState {{
    weights: &'static [u8],
    tensors: HashMap<&'static str, TensorMeta>,
}}

#[derive(Deserialize)]
struct InferRequest {{
    input: Vec<f32>,
}}

#[derive(Serialize)]
struct InferResponse {{
    output: Vec<f32>,
}}

#[derive(Serialize)]
struct ModelInfo {{
    name: &'static str,
    architecture: &'static str,
    total_params: usize,
    total_bytes: usize,
    tensors: Vec<String>,
}}

const MODEL_NAME: &str = "{model_name_esc}";
const MODEL_ARCHITECTURE: &str = "{arch_esc}";

#[tokio::main]
async fn main() {{
    let weights: &'static [u8] = include_bytes!("../{weights_file}");

    let mut tensors = HashMap::new();
    let tensor_defs: Vec<(&str, TensorMeta)> = vec![
{tensor_loads}    ];
    for (name, meta) in tensor_defs {{
        tensors.insert(name, meta);
    }}

    let state = Arc::new(AppState {{ weights, tensors }});

    let app = Router::new()
        .route("/infer", post(infer))
        .route("/info", get(model_info))
        .with_state(state);

    let addr = "{listen_esc}"
        .parse::<std::net::SocketAddr>()
        .expect("embedded listen address");

    let total_mb = {total_bytes} as f64 / (1024.0 * 1024.0);
    eprintln!(
        "model-serve: listening on http://{{}}\n  model: {{}}\n  architecture: {{}}\n  parameters: {total_params}\n  weight blob: {total_bytes} bytes (~{{:.4}} MB)",
        addr,
        MODEL_NAME,
        MODEL_ARCHITECTURE,
        total_mb,
    );

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}}

async fn infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> Json<InferResponse> {{
    let result = forward(&state, &req.input);
    Json(InferResponse {{ output: result }})
}}

async fn model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {{
    Json(ModelInfo {{
        name: MODEL_NAME,
        architecture: MODEL_ARCHITECTURE,
        total_params: {total_params},
        total_bytes: {total_bytes},
        tensors: state.tensors.keys().map(|k| k.to_string()).collect(),
    }})
}}

fn forward(_state: &AppState, input: &[f32]) -> Vec<f32> {{
    // Placeholder inference; tensors are reachable via `_state.weights` and `_state.tensors` metadata.
    input.to_vec()
}}
"##
    )
}

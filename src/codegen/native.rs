use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::codegen::CodeGenerator;
use crate::model::Model;

pub struct NativeCodegen;

impl CodeGenerator for NativeCodegen {
    fn generate(
        &self,
        model: &Model,
        weights_path: &Path,
        output_dir: &Path,
        port: u16,
    ) -> Result<PathBuf> {
        let project_dir = output_dir.join("modelc_build");
        let src_dir = project_dir.join("src");

        fs::create_dir_all(&src_dir)
            .with_context(|| "failed to create build directory")?;

        let weights_file_name = weights_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        fs::copy(weights_path, project_dir.join(&weights_file_name))
            .with_context(|| "failed to copy weights")?;

        let cargo_toml = generate_cargo_toml(port);
        let main_rs = generate_main_rs(model, &weights_file_name, port);

        fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;
        fs::write(src_dir.join("main.rs"), main_rs)?;

        Ok(project_dir)
    }
}

fn generate_cargo_toml(_port: u16) -> String {
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
"#.to_string()
}

fn generate_main_rs(model: &Model, weights_file: &str, port: u16) -> String {
    let _tensor_names: Vec<&String> = model.tensors.keys().collect();
    let tensor_info: Vec<(&String, &crate::model::TensorData)> =
        model.tensors.iter().collect();

    let mut tensor_loads = String::new();
    for (name, tensor) in &tensor_info {
        let _safe_ident = sanitize_ident(name);
        let shape_fmt = format!("{:?}", tensor.shape);
        let dtype_size = tensor.dtype.byte_size();
        let byte_len = tensor.byte_len();
        let offset = 0usize;
        tensor_loads.push_str(&format!(
            "        ({:?}, TensorMeta {{ shape: &{shape_fmt}, dtype_size: {dtype_size}, byte_offset: {offset}, byte_len: {byte_len} }}),\n",
            name
        ));
    }

    let model_name = &model.name;
    let total_params = model.total_params();
    let total_bytes = model.total_bytes();

    format!(
        r##"use std::collections::HashMap;
use std::sync::Arc;

use axum::{{Json, Router, routing::post, extract::State}};
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
    total_params: usize,
    total_bytes: usize,
    tensors: Vec<String>,
}}

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
        .route("/info", axum::routing::get(model_info))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], {port}));
    eprintln!("model-serve: serving '{model_name}' on http://{{}}", addr);
    eprintln!("  parameters: {total_params}");
    eprintln!("  size: {total_bytes} bytes ({{:.2}} MB)", {total_bytes} as f64 / (1024.0 * 1024.0));

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

async fn model_info(
    State(state): State<Arc<AppState>>,
) -> Json<ModelInfo> {{
    Json(ModelInfo {{
        name: "{model_name}",
        total_params: {total_params},
        total_bytes: {total_bytes},
        tensors: state.tensors.keys().map(|k| k.to_string()).collect(),
    }})
}}

fn forward(_state: &AppState, input: &[f32]) -> Vec<f32> {{
    // Placeholder: pass-through.
    // Replace with actual model forward pass (matmul, activation, etc.)
    input.to_vec()
}}
"##
    )
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

//! HTTP server for `modelc run` — loads a `.modelc` artifact and serves /info + /infer + /chat + /complete.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{Router, routing::{get, post}};
use serde::{Deserialize, Serialize};

use crate::model::Model;
use crate::runtime::serve::Runtime;

mod handlers;
mod infer;

pub async fn run_server(model: Model, addr: SocketAddr, profile: bool) -> anyhow::Result<()> {
    let app = build_router(model, profile);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("modelc run: listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

fn build_router(model: Model, profile: bool) -> Router {
    let onnx_plan = model
        .metadata
        .get("onnx.execution_plan")
        .and_then(|json| crate::onnx_exec::ExecutionPlan::from_json(json).ok());

    let state = Arc::new(AppState {
        name: model.name.clone(),
        architecture: model.architecture.clone(),
        total_params: model.total_params(),
        total_bytes: model.total_bytes(),
        tensor_names: {
            let mut names: Vec<String> = model.tensors.keys().cloned().collect();
            names.sort();
            names
        },
        runtime: Runtime::from_raw(&model.tensors),
        mlp_plan: infer_mlp_plan(&model),
        onnx_plan,
        profile,
    });

    Router::new()
        .route("/infer", post(handlers::infer))
        .route("/info", get(handlers::model_info))
        .route("/chat", post(handlers::chat))
        .route("/chat/stream", post(handlers::chat_stream))
        .route("/complete", post(handlers::complete))
        .with_state(state)
}

struct AppState {
    name: String,
    architecture: String,
    total_params: usize,
    total_bytes: usize,
    tensor_names: Vec<String>,
    runtime: Runtime,
    mlp_plan: Option<Vec<(String, String)>>,
    onnx_plan: Option<crate::onnx_exec::ExecutionPlan>,
    profile: bool,
}

#[derive(Deserialize)]
struct InferRequest {
    #[serde(default)]
    input: Vec<f32>,
    #[serde(default)]
    inputs: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct InferResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outputs: Option<Vec<Vec<f32>>>,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    architecture: String,
    total_params: usize,
    total_bytes: usize,
    tensors: Vec<String>,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
    #[serde(default)]
    #[allow(dead_code)]
    stream: bool,
}

#[derive(Deserialize, Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    message: Message,
}

#[derive(Deserialize)]
struct CompleteRequest {
    prompt: String,
}

#[derive(Serialize)]
struct CompleteResponse {
    completion: String,
}

#[derive(Serialize)]
struct StreamChunk {
    delta: String,
    done: bool,
}

fn infer_mlp_plan(model: &Model) -> Option<Vec<(String, String)>> {
    if model.architecture != "mlp" {
        return None;
    }

    layered_mlp_pairs(model).or_else(|| singleton_affine_pair(model))
}

fn singleton_affine_pair(model: &Model) -> Option<Vec<(String, String)>> {
    validate_affine_pair(model, "weight", "bias")?;
    Some(vec![("weight".to_string(), "bias".to_string())])
}

fn layered_mlp_pairs(model: &Model) -> Option<Vec<(String, String)>> {
    let mut ids: Vec<u32> = model
        .tensors
        .keys()
        .filter_map(|key| parse_layer_suffix(key.as_str()))
        .collect();
    if ids.is_empty() {
        return None;
    }

    ids.sort_unstable();
    ids.dedup();

    if !ids.windows(2).all(|pair| pair[1] == pair[0] + 1) {
        return None;
    }

    let mut seq = Vec::new();
    let mut prev_out_rows: Option<usize> = None;

    for id in ids {
        let weight_name = format!("layer{id}.weight");
        let bias_name = format!("layer{id}.bias");
        let (rows, cols) = affine_pair_shape(model, &weight_name, &bias_name)?;

        if let Some(out_prev) = prev_out_rows
            && out_prev != cols
        {
            return None;
        }

        seq.push((weight_name, bias_name));
        prev_out_rows = Some(rows);
    }

    Some(seq)
}

fn affine_pair_shape(model: &Model, weight_name: &str, bias_name: &str) -> Option<(usize, usize)> {
    validate_affine_pair(model, weight_name, bias_name)?;
    let w = model.tensors.get(weight_name)?;
    Some((*w.shape.first()?, *w.shape.get(1)?))
}

fn validate_affine_pair<'m>(
    model: &'m Model,
    weight_name: &str,
    bias_name: &str,
) -> Option<&'m crate::model::TensorData> {
    let w = model.tensors.get(weight_name)?;
    let b = model.tensors.get(bias_name)?;

    if w.dtype != crate::model::DataType::F32 || b.dtype != crate::model::DataType::F32 {
        return None;
    }

    let rows = *w.shape.first()?;
    if w.shape.len() != 2 || b.shape.len() != 1 {
        return None;
    }

    (b.shape[0] == rows).then_some(w)
}

fn parse_layer_suffix(name: &str) -> Option<u32> {
    let tail = name.strip_prefix("layer")?;
    let (idx, suf) = tail.split_once('.')?;
    if suf != "weight" {
        return None;
    }

    idx.parse::<u32>().ok()
}

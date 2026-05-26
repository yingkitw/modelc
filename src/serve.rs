//! HTTP server for `modelc run` — loads a `.modelc` artifact and serves /info + /infer.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use crate::model::Model;
use crate::runtime::serve::Runtime;
use crate::runtime::tensor::Tensor;

pub async fn run_server(model: Model, addr: SocketAddr) -> anyhow::Result<()> {
    let app = build_router(model);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("modelc run: listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

fn build_router(model: Model) -> Router {
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
    });

    Router::new()
        .route("/infer", post(infer))
        .route("/info", get(model_info))
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
}

#[derive(Deserialize)]
struct InferRequest {
    input: Vec<f32>,
}

#[derive(Serialize)]
struct InferResponse {
    output: Vec<f32>,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    architecture: String,
    total_params: usize,
    total_bytes: usize,
    tensors: Vec<String>,
}

async fn infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> Json<InferResponse> {
    let output = if let Some(plan) = &state.mlp_plan {
        run_mlp_forward(&state.runtime, plan, &req.input)
    } else {
        // Echo input for unsupported architectures.
        req.input.clone()
    };
    Json(InferResponse { output })
}

async fn model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {
    Json(ModelInfo {
        name: state.name.clone(),
        architecture: state.architecture.clone(),
        total_params: state.total_params,
        total_bytes: state.total_bytes,
        tensors: state.tensor_names.clone(),
    })
}

fn run_mlp_forward(runtime: &Runtime, plan: &[(String, String)], input: &[f32]) -> Vec<f32> {
    if plan.is_empty() {
        return input.to_vec();
    }

    let mut cur = input.to_vec();
    let last = plan.len() - 1;

    for (idx, (w_name, b_name)) in plan.iter().enumerate() {
        let w = runtime.get(w_name).expect("mlp weight missing");
        let b = runtime.get(b_name).expect("mlp bias missing");
        cur = gemv_bias(w, b, &cur);
        if idx != last {
            relu_inplace(&mut cur);
        }
    }

    cur
}

/// Matrix-vector multiply with bias. Weight shape is `[rows, cols]` (like the generated server).
fn gemv_bias(weight: &Tensor, bias: &Tensor, x: &[f32]) -> Vec<f32> {
    assert_eq!(weight.shape.len(), 2, "weight must be 2D");
    assert_eq!(bias.shape.len(), 1, "bias must be 1D");
    let rows = weight.shape[0];
    let cols = weight.shape[1];
    assert_eq!(cols, x.len(), "gemv: input size mismatch");
    assert_eq!(bias.shape[0], rows, "gemv: bias size mismatch");

    let mut out = vec![0.0f32; rows];
    for (r, out_v) in out.iter_mut().enumerate().take(rows) {
        let mut acc = bias.data[r];
        let row = &weight.data[r * cols..(r + 1) * cols];
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += wv * xv;
        }
        *out_v = acc;
    }
    out
}

fn relu_inplace(xs: &mut [f32]) {
    for v in xs {
        *v = v.max(0.0);
    }
}

// MLP plan detection (same logic as codegen/native.rs).
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

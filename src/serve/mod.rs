//! HTTP server for `modelc run` — loads a `.modelc` artifact and serves /info + /infer + /chat + /complete.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    Router,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use crate::model::Model;
use crate::runtime::serve::Runtime;

pub mod auth;
mod handlers;
mod infer;
mod metrics;
mod openai;

pub async fn run_server(
    model: Model,
    addr: SocketAddr,
    profile: bool,
    generation: crate::generate::GenerationConfig,
    auth: Option<auth::AuthConfig>,
) -> anyhow::Result<()> {
    let app = build_router(model, profile, generation);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("modelc run: listening on http://{}", addr);

    if let Some(auth_cfg) = auth {
        let svc = app
            .layer(axum::middleware::from_fn_with_state(
                auth_cfg,
                auth::middleware,
            ))
            .into_make_service_with_connect_info::<SocketAddr>();
        axum::serve(listener, svc).await?;
    } else {
        axum::serve(listener, app).await?;
    }
    Ok(())
}

fn build_router(
    model: Model,
    profile: bool,
    generation: crate::generate::GenerationConfig,
) -> Router {
    let onnx_plan = model
        .metadata
        .get("onnx.execution_plan")
        .and_then(|json| crate::onnx_exec::ExecutionPlan::from_json(json).ok());

    let chat_template = model.metadata.get("tokenizer.chat_template").cloned();

    let runtime = Runtime::from_raw(&model.tensors);
    let draft_model = transformer_hidden_dim(&model).and_then(|hidden| {
        let vocab_size = model.metadata.get("tokenizer.vocab_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        crate::draft::MlpDraftModel::from_runtime(&runtime, vocab_size, hidden, 64, generation.temperature, generation.top_p)
            .map(|dm| std::sync::Arc::new(dm) as std::sync::Arc<dyn crate::draft::DraftModel>)
    });

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
        runtime: std::sync::RwLock::new(runtime),
        base_tensors: model.tensors.clone(),
        mlp_plan: infer_mlp_plan(&model),
        onnx_plan,
        transformer_hidden: transformer_hidden_dim(&model),
        chat_template,
        profile,
        generation,
        prefix_cache: std::sync::RwLock::new(crate::prefix_cache::PrefixCache::new(
            PREFIX_CACHE_CAPACITY,
        )),
        metrics: metrics::Metrics::default(),
        draft_model,
    });

    Router::new()
        .route("/infer", post(handlers::infer))
        .route("/info", get(handlers::model_info))
        .route("/health", get(handlers::health))
        .route("/chat", post(handlers::chat))
        .route("/chat/stream", post(handlers::chat_stream))
        .route("/complete", post(handlers::complete))
        .route("/embeddings", post(handlers::embeddings))
        .route("/metrics", get(handlers::metrics_handler))
        .route("/v1/models", get(openai::list_models))
        .route("/v1/chat/completions", post(openai::chat_completion))
        .route("/v1/completions", post(openai::completions))
        .route("/lora/load", post(handlers::lora_load))
        .route("/lora/unload", post(handlers::lora_unload))
        .with_state(state)
}

struct AppState {
    name: String,
    architecture: String,
    total_params: usize,
    total_bytes: usize,
    tensor_names: Vec<String>,
    /// Mutable runtime so LoRA adapters can be swapped in/out without restarting.
    runtime: std::sync::RwLock<Runtime>,
    /// Original (unmodified) model tensors. Used to reset to base weights after unloading LoRA.
    base_tensors: std::collections::HashMap<String, crate::model::TensorData>,
    mlp_plan: Option<Vec<(String, String)>>,
    onnx_plan: Option<crate::onnx_exec::ExecutionPlan>,
    /// Hidden size for transformer (`gpt2`/`llama`) artifacts, so `/infer` inputs of the wrong
    /// length can be gracefully resized before the transformer forward. `None` for non-
    /// transformer architectures or when the hidden size cannot be determined.
    transformer_hidden: Option<usize>,
    /// Jinja2 chat template from model metadata (e.g. `tokenizer.chat_template`), used to
    /// format messages before tokenization for chat endpoints.
    chat_template: Option<String>,
    profile: bool,
    /// Default generation configuration for text inference endpoints.
    generation: crate::generate::GenerationConfig,
    /// Prompt-prefix KV cache shared across requests, so repeated or shared-prefix
    /// prompts (system prompts, tool descriptions) skip KV recomputation.
    prefix_cache: std::sync::RwLock<crate::prefix_cache::PrefixCache>,
    /// Prometheus-style metrics for request counts, latency, tokens generated, etc.
    metrics: metrics::Metrics,
    /// Optional neural draft model for speculative decoding.  Loaded from
    /// `draft.*` tensors in the runtime; falls back to n-gram drafting when
    /// absent.
    draft_model: Option<std::sync::Arc<dyn crate::draft::DraftModel>>,
}

/// Maximum number of distinct prompt prefixes retained in the prefix cache.
const PREFIX_CACHE_CAPACITY: usize = 32;

#[derive(Deserialize)]
struct InferRequest {
    #[serde(default)]
    input: Vec<f32>,
    #[serde(default)]
    inputs: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct LoraLoadRequest {
    /// Absolute or relative path to a Safetensors LoRA adapter file.
    path: String,
    /// LoRA alpha scaling factor (default 1.0).
    #[serde(default = "default_lora_alpha")]
    alpha: f32,
}

fn default_lora_alpha() -> f32 {
    1.0
}

#[derive(Serialize)]
struct LoraLoadResponse {
    applied: usize,
    skipped: usize,
    message: String,
}

#[derive(Serialize)]
struct LoraUnloadResponse {
    message: String,
}

#[derive(Serialize)]
struct InferResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outputs: Option<Vec<Vec<f32>>>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    architecture: String,
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
struct EmbeddingsRequest {
    #[serde(default)]
    input: String,
    #[serde(default)]
    inputs: Vec<String>,
}

#[derive(Serialize)]
struct EmbeddingEntry {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Serialize)]
struct EmbeddingsResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embeddings: Option<Vec<EmbeddingEntry>>,
    model: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// Optional regex grammar constraint applied during sampling.
    #[serde(default)]
    grammar: Option<String>,
    /// Optional JSON Schema object to validate generated output against.
    #[serde(default)]
    json_schema: Option<serde_json::Value>,
    /// Optional stop sequences that halt generation.
    #[serde(default)]
    stop: Vec<String>,
    /// Optional seed for reproducible sampling.
    #[serde(default)]
    seed: Option<u64>,
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
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// Optional regex grammar constraint applied during sampling.
    #[serde(default)]
    grammar: Option<String>,
    /// Optional JSON Schema object to validate generated output against.
    #[serde(default)]
    json_schema: Option<serde_json::Value>,
    /// Optional stop sequences that halt generation.
    #[serde(default)]
    stop: Vec<String>,
    /// Optional seed for reproducible sampling.
    #[serde(default)]
    seed: Option<u64>,
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

/// Resolve the hidden dimension for transformer architectures so `/infer` can resize inputs
/// to the expected single-vector width. Returns `None` for non-transformer models or when the
/// hidden size is zero / undetectable.
fn transformer_hidden_dim(model: &Model) -> Option<usize> {
    let hidden = match model.architecture.as_str() {
        "gpt2" => {
            let layers = crate::arch::detect_layers(model, "transformer.h.");
            crate::arch::gpt2_hidden_dim(model, &layers)
        }
        "llama" => {
            let layers = crate::arch::llama_layers(model);
            crate::arch::llama_hidden_dim(model, &layers)
        }
        _ => return None,
    };
    (hidden > 0).then_some(hidden)
}

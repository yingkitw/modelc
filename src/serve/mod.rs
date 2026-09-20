//! HTTP server for `modelc run` — loads a `.modelc` artifact and serves /info + /infer + /chat + /complete.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::{
    Router,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use crate::model::Model;
use crate::runtime::serve::Runtime;

pub mod auth;
mod handlers;
mod infer;
mod metrics;
mod openai;

/// A `Stream` wrapper that sets a cancellation flag when dropped.
///
/// Used by the SSE streaming endpoints: when the client disconnects, axum drops
/// the response body (and thus this stream), which trips the flag. The spawned
/// generation task checks the flag once per token via `GenerationConfig.cancel`
/// and stops early instead of running to completion.
pub(super) struct CancelOnDrop<S> {
    inner: S,
    cancel: Arc<AtomicBool>,
}

impl<S> CancelOnDrop<S> {
    pub(super) fn new(inner: S, cancel: Arc<AtomicBool>) -> Self {
        Self { inner, cancel }
    }
}

impl<S: Stream + Unpin> Stream for CancelOnDrop<S> {
    type Item = S::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

pub async fn run_server(
    model: Model,
    addr: SocketAddr,
    profile: bool,
    generation: crate::generate::GenerationConfig,
    auth: Option<auth::AuthConfig>,
    max_concurrent: Option<usize>,
) -> anyhow::Result<()> {
    run_server_with_shutdown(
        model,
        addr,
        profile,
        generation,
        auth,
        max_concurrent,
        shutdown_signal(),
    )
    .await
}

/// Like [`run_server`], but with a caller-supplied shutdown signal. When `shutdown`
/// resolves, axum stops accepting new connections and drains in-flight requests
/// before returning. Exposed publicly so embedders (and tests) can trigger a
/// graceful shutdown programmatically instead of relying on OS signals.
pub async fn run_server_with_shutdown<F>(
    model: Model,
    addr: SocketAddr,
    profile: bool,
    generation: crate::generate::GenerationConfig,
    auth: Option<auth::AuthConfig>,
    max_concurrent: Option<usize>,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let app = build_router(model, profile, generation, max_concurrent);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("modelc run: listening on http://{}", addr);

    let mut shutdown = Some(shutdown);
    if let Some(auth_cfg) = auth {
        let svc = app
            .layer(axum::middleware::from_fn_with_state(
                auth_cfg,
                auth::middleware,
            ))
            .into_make_service_with_connect_info::<SocketAddr>();
        axum::serve(listener, svc)
            .with_graceful_shutdown(shutdown.take().expect("shutdown future"))
            .await?;
    } else {
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown.take().expect("shutdown future"))
            .await?;
    }
    eprintln!("modelc run: server stopped");
    Ok(())
}

impl AppState {
    /// Build an `AppState` for tests without going through the full `run_server`
    /// bootstrap. Mirrors the construction in `build_router` but lets tests inject
    /// a hand-built `Model` + generation config without spawning a TCP listener.
    #[doc(hidden)]
    pub fn for_test(
        model: Model,
        profile: bool,
        generation: crate::generate::GenerationConfig,
        auth: Option<auth::AuthConfig>,
        max_concurrent: Option<usize>,
    ) -> std::sync::Arc<Self> {
        let _ = auth; // auth is wired at the router layer, not in AppState
        let architecture = model.architecture.clone();
        let transformer_hidden = crate::arch::transformer_hidden_dim(&model);
        let mlp_plan = infer_mlp_plan(&model);
        let onnx_plan = model
            .metadata
            .get("onnx.execution_plan")
            .and_then(|s| crate::onnx_exec::ExecutionPlan::from_json(s).ok());
        let chat_template = model.metadata.get("tokenizer.chat_template").cloned();
        let base_tensors = model.tensors.clone();
        let runtime = Runtime::from_raw(&model.tensors);
        // Try the neural draft model first; fall back to prompt-lookup for
        // transformer architectures without `draft.*` tensors. Mirrors the
        // production build_router path.
        let mut draft_model: Option<std::sync::Arc<dyn crate::draft::DraftModel>> =
            transformer_hidden.and_then(|hidden| {
                let vocab_size = model
                    .metadata
                    .get("tokenizer.vocab_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                crate::draft::MlpDraftModel::from_runtime(
                    &runtime,
                    vocab_size,
                    hidden,
                    64,
                    generation.temperature,
                    generation.top_p,
                )
                .map(|dm| std::sync::Arc::new(dm) as std::sync::Arc<dyn crate::draft::DraftModel>)
            });
        // Fall back to prompt lookup decoding when no neural draft weights are present.
        if draft_model.is_none() && (architecture == "gpt2" || architecture == "llama") {
            draft_model = Some(
                std::sync::Arc::new(crate::draft::PromptLookupDraftModel::new(3))
                    as std::sync::Arc<dyn crate::draft::DraftModel>,
            );
        }
        let max_concurrent = max_concurrent.map(|n| Arc::new(tokio::sync::Semaphore::new(n)));
        let tensor_names = {
            let mut names: Vec<String> = model.tensors.keys().cloned().collect();
            names.sort();
            names
        };
        std::sync::Arc::new(Self {
            name: model.name.clone(),
            architecture,
            total_params: model.total_params(),
            total_bytes: model.total_bytes(),
            tensor_names,
            runtime: std::sync::RwLock::new(runtime),
            base_tensors,
            mlp_plan,
            onnx_plan,
            transformer_hidden,
            chat_template,
            profile,
            generation,
            prefix_cache: std::sync::RwLock::new(crate::prefix_cache::PrefixCache::new(
                PREFIX_CACHE_CAPACITY,
            )),
            metrics: metrics::Metrics::default(),
            draft_model,
            max_concurrent,
        })
    }
}

/// Resolves on `SIGINT` (Ctrl-C) or, on Unix, `SIGTERM`. Used as the graceful-shutdown
/// trigger for [`run_server`].
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    eprintln!("modelc run: shutdown signal received, draining in-flight requests...");
}

fn build_router(
    model: Model,
    profile: bool,
    generation: crate::generate::GenerationConfig,
    max_concurrent: Option<usize>,
) -> Router {
    build_router_from_state(AppState::for_test(
        model,
        profile,
        generation,
        None,
        max_concurrent,
    ))
}

/// Variant that wires a pre-built `AppState` into the router. Used by both
/// `build_router` (production) and tests (which need to inject a synthetic
/// `AppState` to exercise handlers in isolation).
pub fn build_router_from_state(state: std::sync::Arc<AppState>) -> Router {
    let router = Router::new()
        .route("/", get(handlers::web_ui))
        .route("/infer", post(handlers::infer))
        .route("/info", get(handlers::model_info))
        .route("/props", get(handlers::server_props))
        .route("/api/version", get(handlers::version_info))
        .route("/api/tags", get(handlers::api_tags))
        .route("/api/show", post(handlers::api_show))
        .route("/health", get(handlers::health))
        .route("/chat", post(handlers::chat))
        .route("/chat/stream", post(handlers::chat_stream))
        .route("/complete", post(handlers::complete))
        .route("/infill", post(handlers::infill))
        .route("/embeddings", post(handlers::embeddings))
        .route("/tokenize", post(handlers::tokenize))
        .route("/detokenize", post(handlers::detokenize))
        .route("/metrics", get(handlers::metrics_handler))
        .route("/v1/models", get(openai::list_models))
        .route("/v1/models/{id}", get(openai::retrieve_model))
        .route("/v1/system", get(handlers::system_info))
        .route("/v1/chat/completions", post(openai::chat_completion))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/embeddings", post(openai::v1_embeddings))
        .route("/reranking", post(handlers::reranking))
        .route("/lora/load", post(handlers::lora_load))
        .route("/lora/unload", post(handlers::lora_unload))
        .route("/api/cache/stats", get(handlers::cache_stats))
        .route("/api/cache/clear", post(handlers::cache_clear))
        .with_state(state.clone());

    router.layer(axum::middleware::from_fn_with_state(
        state,
        backpressure_middleware,
    ))
}

pub struct AppState {
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
    /// Optional semaphore limiting concurrent inference requests.  When set and
    /// saturated, new inference requests receive 503 Service Unavailable.
    max_concurrent: Option<Arc<tokio::sync::Semaphore>>,
}

/// Axum middleware that rejects inference requests with 503 when the concurrent
/// request limit is reached.  Exempts `/health`, `/info`, and `/metrics` so
/// probes and observability still work under load.
async fn backpressure_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let path = req.uri().path();
    if path == "/health" || path == "/info" || path == "/props" || path == "/metrics" {
        return next.run(req).await;
    }

    if let Some(ref sem) = state.max_concurrent {
        match sem.try_acquire() {
            Ok(_permit) => next.run(req).await,
            Err(_) => axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response(),
        }
    } else {
        next.run(req).await
    }
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

#[derive(Serialize)]
struct ServerProps {
    model: String,
    architecture: String,
    total_params: usize,
    total_bytes: usize,
    chat_template: Option<String>,
    default_generation: DefaultGenerationProps,
}

#[derive(Serialize)]
struct DefaultGenerationProps {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    gamma: usize,
}

#[derive(Serialize)]
struct VersionInfo {
    version: String,
    git_sha: String,
}

#[derive(Deserialize)]
struct TokenizeRequest {
    #[serde(default)]
    input: String,
    #[serde(default)]
    inputs: Vec<String>,
}

#[derive(Serialize)]
struct TokenizeResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_batch: Option<Vec<Vec<u32>>>,
    /// Total number of tokens across all inputs.
    count: usize,
}

#[derive(Deserialize)]
struct DetokenizeRequest {
    #[serde(default)]
    tokens: Vec<u32>,
}

#[derive(Serialize)]
struct DetokenizeResponse {
    text: String,
}

/// System/hardware info exposed by `GET /v1/system`.
#[derive(Serialize)]
struct SystemInfo {
    model: String,
    architecture: String,
    total_params: usize,
    total_bytes: usize,
    /// Logical CPU cores available to the process.
    cpu_cores: usize,
    /// Operating system name (e.g. "macos", "linux", "windows").
    os: &'static str,
    /// CPU architecture (e.g. "aarch64", "x86_64").
    cpu_arch: &'static str,
    /// Pointer width in bits (32 or 64).
    pointer_width: usize,
    /// Whether Apple Silicon Metal GPU acceleration is compiled in.
    metal_available: bool,
    /// Best-effort total physical memory in bytes (`null` if unavailable).
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_total_bytes: Option<u64>,
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
struct RerankingRequest {
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    top_n: Option<usize>,
}

#[derive(Serialize)]
struct RerankingResponse {
    model: String,
    results: Vec<RerankingResult>,
}

#[derive(Serialize)]
struct RerankingResult {
    index: usize,
    relevance_score: f32,
    document: Document,
}

#[derive(Serialize)]
struct Document {
    text: String,
}

#[derive(Serialize)]
struct ApiTagsResponse {
    models: Vec<ApiTag>,
}

#[derive(Serialize)]
struct ApiTag {
    name: String,
    size: u64,
    details: ApiTagDetails,
}

#[derive(Serialize)]
struct ApiTagDetails {
    architecture: String,
    parameter_size: String,
    quantization: String,
}

#[derive(Deserialize)]
struct ApiShowRequest {
    name: String,
}

#[derive(Serialize)]
struct ApiShowResponse {
    name: String,
    architecture: String,
    size_bytes: u64,
    parameter_size: String,
    quantization: String,
    tensor_count: usize,
}

/// `GET /api/cache/stats` response.
#[derive(Serialize)]
struct CacheStatsResponse {
    entries: usize,
    capacity: usize,
    total_tokens: usize,
    total_kv_bytes: usize,
    /// `entries / capacity`, in the closed interval `[0.0, 1.0]`.
    utilization: f64,
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
    /// Optional min-p sampling threshold. Keeps tokens whose probability is at
    /// least `min_p` fraction of the max probability.
    #[serde(default)]
    min_p: Option<f32>,
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
    /// Penalty for repeated tokens. Values > 1.0 discourage repetition.
    #[serde(default)]
    repetition_penalty: Option<f32>,
    /// OpenAI-style presence penalty. Positive values discourage token reuse.
    #[serde(default)]
    presence_penalty: Option<f32>,
    /// OpenAI-style frequency penalty. Scales with token occurrence count.
    #[serde(default)]
    frequency_penalty: Option<f32>,
    /// OpenAI-style logit bias: map from token ID to additive bias (-100 to 100).
    #[serde(default)]
    logit_bias: Option<HashMap<u32, f32>>,
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
    /// Optional min-p sampling threshold. Keeps tokens whose probability is at
    /// least `min_p` fraction of the max probability.
    #[serde(default)]
    min_p: Option<f32>,
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
    /// Penalty for repeated tokens. Values > 1.0 discourage repetition.
    #[serde(default)]
    repetition_penalty: Option<f32>,
    /// OpenAI-style presence penalty. Positive values discourage token reuse.
    #[serde(default)]
    presence_penalty: Option<f32>,
    /// OpenAI-style frequency penalty. Scales with token occurrence count.
    #[serde(default)]
    frequency_penalty: Option<f32>,
    /// OpenAI-style logit bias: map from token ID to additive bias (-100 to 100).
    #[serde(default)]
    logit_bias: Option<HashMap<u32, f32>>,
}

#[derive(Serialize)]
struct CompleteResponse {
    completion: String,
}

#[derive(Deserialize)]
struct InfillRequest {
    prefix: String,
    suffix: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
}

#[derive(Serialize)]
struct InfillResponse {
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::sse::Event;

    /// Dropping the wrapped stream must trip the cancellation flag so the spawned
    /// generation task stops early.
    #[test]
    fn cancel_on_drop_sets_flag() {
        let (_tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(4);
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let _wrapped = CancelOnDrop::new(
                tokio_stream::wrappers::ReceiverStream::new(rx),
                cancel.clone(),
            );
            assert!(
                !cancel.load(Ordering::Relaxed),
                "flag clear while stream alive"
            );
        }
        assert!(
            cancel.load(Ordering::Relaxed),
            "flag must be set after the stream is dropped"
        );
    }
}

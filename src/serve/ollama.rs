//! Ollama ecosystem schema helpers for `/api/tags` + `/api/show`, plus the
//! inference endpoints `/api/generate` + `/api/chat` that real Ollama client
//! libraries drive.
//!
//! Port of modelc-web's `api/_ollama.ts`. The upstream endpoints previously
//! used modelc-specific field names which made them only *superficially*
//! Ollama-compatible — real clients parse `details.format/family/families/
//! parameter_size/quantization_level`, `digest` and `model_info`. Keep
//! modelc extras alongside: Ollama clients ignore unknown keys.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;

use super::infer::{
    run_embeddings, run_text_inference_token_ids, run_text_inference_with_config,
};
use super::{AppState, CancelOnDrop, Message};
use crate::generate::GenerationConfig;
use crate::pack::ArtifactHeader;

/// Human-readable parameter count, matching Ollama's formatting
/// (`8.0B`, `134.52M`, `1.5K`).
pub fn format_parameter_size(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.2}M", params as f64 / 1e6)
    } else if params >= 1_000 {
        format!("{:.1}K", params as f64 / 1e3)
    } else {
        params.to_string()
    }
}

/// Map a modelc quantization label to Ollama's `quantization_level` names.
pub fn quantization_level(quantization: &str) -> String {
    match quantization.to_ascii_lowercase().as_str() {
        "f32" | "fp32" => "F32".to_string(),
        "f16" | "fp16" => "F16".to_string(),
        "bf16" => "BF16".to_string(),
        "int8" | "i8" => "Q8_0".to_string(),
        "int4" => "Q4_0".to_string(),
        "q4_0" => "Q4_0".to_string(),
        "q5_0" => "Q5_0".to_string(),
        "q8_0" => "Q8_0".to_string(),
        "q4_k" => "Q4_K".to_string(),
        "q6_k" => "Q6_K".to_string(),
        other => other.to_ascii_uppercase(),
    }
}

/// llama.cpp file-type enum, used by Ollama's `general.file_type` (`0` = all F32).
pub fn ollama_file_type(quantization: &str) -> i32 {
    match quantization_level(quantization).as_str() {
        "F32" | "BF16" => 0,
        "F16" => 1,
        "Q4_0" => 2,
        "Q5_0" => 5,
        "Q8_0" => 7,
        "Q4_K" => 10,
        "Q6_K" => 12,
        _ => 0,
    }
}

/// Source format for `details.format` — artifact metadata, else "modelc".
pub fn artifact_format(header: &ArtifactHeader) -> String {
    header
        .metadata
        .get("format")
        .cloned()
        .unwrap_or_else(|| "modelc".to_string())
}

// ---------------------------------------------------------------------------
// `/api/generate` + `/api/chat` — Ollama inference endpoints
// ---------------------------------------------------------------------------

/// Ollama `options` sub-object. Only fields with a modelc equivalent are
/// declared; serde ignores the rest (`top_k`, `mirostat`, `keep_alive`, …).
#[derive(Default, Deserialize)]
pub struct OllamaOptions {
    #[serde(default)]
    pub num_predict: Option<i64>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Stop sequences — Ollama accepts a string or an array of strings.
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub num_ctx: Option<u64>,
}

#[derive(Deserialize)]
pub struct OllamaGenerateRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub raw: Option<bool>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub format: Option<Value>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Deserialize)]
pub struct OllamaChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub format: Option<Value>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

/// Ollama `/api/generate` final response shape (also the last NDJSON line
/// when streaming).
#[derive(Serialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub done_reason: String,
    pub context: Vec<u32>,
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: usize,
    pub prompt_eval_duration: u64,
    pub eval_count: usize,
    pub eval_duration: u64,
}

/// Ollama `/api/chat` final response shape.
#[derive(Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    pub done_reason: String,
    pub context: Vec<u32>,
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: usize,
    pub prompt_eval_duration: u64,
    pub eval_count: usize,
    pub eval_duration: u64,
}

fn stop_strings(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

/// Map an Ollama `options` object onto a `GenerationConfig`, layered on the
/// server defaults (same precedence as the `/chat` + `/complete` handlers).
fn options_config(base: &GenerationConfig, opts: Option<&OllamaOptions>) -> GenerationConfig {
    let mut cfg = base.clone();
    let Some(o) = opts else { return cfg };
    if let Some(n) = o.num_predict {
        // Ollama uses -1 for "unbounded"; our usize needs a concrete cap.
        if n > 0 {
            cfg.max_tokens = n as usize;
        }
    }
    if let Some(v) = o.temperature {
        cfg.temperature = v;
    }
    if let Some(v) = o.top_p {
        cfg.top_p = v;
    }
    if let Some(v) = o.min_p {
        cfg.min_p = v;
    }
    cfg.seed = o.seed.or(cfg.seed);
    if let Some(v) = o.repeat_penalty {
        cfg.repetition_penalty = v;
    }
    if let Some(v) = o.presence_penalty {
        cfg.presence_penalty = v;
    }
    if let Some(v) = o.frequency_penalty {
        cfg.frequency_penalty = v;
    }
    if let Some(v) = &o.stop {
        let stop = stop_strings(v);
        if !stop.is_empty() {
            cfg.stop = stop;
        }
    }
    if let Some(n) = o.num_ctx {
        cfg.max_context = Some(n as usize);
    }
    cfg
}

fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    crate::digest::rfc3339(secs)
}

/// `format: "json"` → object schema; `format: {…}` → the supplied schema.
fn schema_for(format: Option<&Value>) -> Option<Value> {
    match format {
        Some(Value::String(s)) if s == "json" => Some(serde_json::json!({"type": "object"})),
        Some(v @ Value::Object(_)) => Some(v.clone()),
        _ => None,
    }
}

/// `/api/generate` prompt assembly: `raw: true` sends the prompt verbatim
/// (unless a `system` override is present); otherwise messages pass through
/// the same chat template as `/chat`.
fn generate_prompt(state: &AppState, req: &OllamaGenerateRequest) -> String {
    if req.raw.unwrap_or(false) && req.system.is_none() {
        return req.prompt.clone();
    }
    let mut msgs = Vec::new();
    if let Some(system) = &req.system {
        msgs.push(crate::chat_template::ChatMessage {
            role: "system".to_string(),
            content: system.clone(),
        });
    }
    msgs.push(crate::chat_template::ChatMessage {
        role: "user".to_string(),
        content: req.prompt.clone(),
    });
    crate::chat_template::apply_chat_template(state.chat_template.as_deref(), &msgs)
}

struct GenResult {
    text: String,
    prompt_eval_count: usize,
    eval_count: usize,
    context: Vec<u32>,
    prompt_eval_ns: u64,
    eval_ns: u64,
}

/// Blocking generation shared by the non-streaming handlers and the
/// format-locked streaming fallback. Returns real token counts and
/// nanosecond durations for Ollama's stats fields.
fn run_generation(
    state: &AppState,
    prompt: &str,
    cfg: &GenerationConfig,
    format: Option<&Value>,
) -> GenResult {
    let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
    let t0 = Instant::now();
    let prompt_ids = tokenizer.encode(prompt);
    let prompt_eval_ns = t0.elapsed().as_nanos() as u64;

    let t1 = Instant::now();
    let (text, gen_ids) = if let Some(schema) = schema_for(format) {
        let text = crate::json_schema::generate_with_schema(
            |c| run_text_inference_with_config(state, prompt, c),
            &schema,
            cfg,
            3,
        );
        (text, Vec::new())
    } else {
        let ids = run_text_inference_token_ids(state, prompt, cfg);
        let full = tokenizer.decode(&[prompt_ids.as_slice(), &ids].concat());
        let base = tokenizer.decode(&prompt_ids);
        (full.strip_prefix(&base).unwrap_or(&full).to_string(), ids)
    };
    let eval_ns = t1.elapsed().as_nanos() as u64;
    let eval_count = if gen_ids.is_empty() {
        tokenizer.encode(&text).len()
    } else {
        gen_ids.len()
    };
    GenResult {
        text,
        prompt_eval_count: prompt_ids.len(),
        eval_count,
        context: [prompt_ids.as_slice(), &gen_ids].concat(),
        prompt_eval_ns,
        eval_ns,
    }
}

#[derive(Clone, Copy)]
enum StreamKind {
    Generate,
    Chat,
}

fn chunk_line(
    model: &str,
    created: &str,
    kind: StreamKind,
    delta: &str,
    done: bool,
    stats: Option<Value>,
) -> String {
    let mut v = match kind {
        StreamKind::Generate => serde_json::json!({
            "model": model,
            "created_at": created,
            "response": delta,
            "done": done,
        }),
        StreamKind::Chat => serde_json::json!({
            "model": model,
            "created_at": created,
            "message": {"role": "assistant", "content": delta},
            "done": done,
        }),
    };
    if done {
        v["done_reason"] = Value::String("stop".to_string());
        if let Some(s) = stats {
            for (k, val) in s.as_object().into_iter().flatten() {
                v[k] = val.clone();
            }
        }
    }
    format!("{}\n", serde_json::to_string(&v).unwrap_or_default())
}

/// NDJSON streaming (Ollama's wire format — newline-delimited JSON, not SSE).
fn ndjson_stream(
    state: Arc<AppState>,
    prompt: String,
    mut cfg: GenerationConfig,
    model: String,
    kind: StreamKind,
    format: Option<Value>,
) -> Response {
    let cancel = Arc::new(AtomicBool::new(false));
    cfg.cancel = Some(cancel.clone());
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, Infallible>>(8);

    tokio::spawn(async move {
        let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
        let prompt_ids = tokenizer.encode(&prompt);
        let created = now_rfc3339();
        let start = Instant::now();

        if let Some(schema) = schema_for(format.as_ref()) {
            // Schema-locked generation is whole-string; emit it as one chunk.
            let text = crate::json_schema::generate_with_schema(
                |c| run_text_inference_with_config(&state, &prompt, c),
                &schema,
                &cfg,
                3,
            );
            let eval_ns = start.elapsed().as_nanos() as u64;
            let _ = tx
                .send(Ok(chunk_line(&model, &created, kind, &text, false, None)))
                .await;
            let stats = serde_json::json!({
                "total_duration": start.elapsed().as_nanos() as u64,
                "load_duration": 0u64,
                "prompt_eval_count": prompt_ids.len(),
                "prompt_eval_duration": 0u64,
                "eval_count": tokenizer.encode(&text).len(),
                "eval_duration": eval_ns,
                "context": prompt_ids,
            });
            let _ = tx
                .send(Ok(chunk_line(&model, &created, kind, "", true, Some(stats))))
                .await;
            return;
        }

        let token_ids = run_text_inference_token_ids(&state, &prompt, &cfg);
        let eval_ns = start.elapsed().as_nanos() as u64;
        for delta in tokenizer.decode_deltas(&prompt_ids, &token_ids) {
            if tx
                .send(Ok(chunk_line(&model, &created, kind, &delta, false, None)))
                .await
                .is_err()
            {
                return;
            }
        }
        let context: Vec<u32> = [prompt_ids.as_slice(), &token_ids].concat();
        let stats = serde_json::json!({
            "total_duration": start.elapsed().as_nanos() as u64,
            "load_duration": 0u64,
            "prompt_eval_count": prompt_ids.len(),
            "prompt_eval_duration": 0u64,
            "eval_count": token_ids.len(),
            "eval_duration": eval_ns,
            "context": context,
        });
        let _ = tx
            .send(Ok(chunk_line(&model, &created, kind, "", true, Some(stats))))
            .await;
    });

    Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(CancelOnDrop::new(
            ReceiverStream::new(rx),
            cancel,
        )))
        .unwrap_or_else(|_| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "stream").into_response())
}

pub(super) async fn api_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaGenerateRequest>,
) -> Response {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let prompt = generate_prompt(&state, &req);
    let cfg = options_config(&state.generation, req.options.as_ref());
    let model = req.model.clone().unwrap_or_else(|| state.name.clone());

    if req.stream.unwrap_or(true) {
        return ndjson_stream(
            state.clone(),
            prompt,
            cfg,
            model,
            StreamKind::Generate,
            req.format,
        );
    }

    let start = Instant::now();
    let r = run_generation(&state, &prompt, &cfg, req.format.as_ref());
    Json(OllamaGenerateResponse {
        model,
        created_at: now_rfc3339(),
        response: r.text,
        done: true,
        done_reason: "stop".to_string(),
        context: r.context,
        total_duration: start.elapsed().as_nanos() as u64,
        load_duration: 0,
        prompt_eval_count: r.prompt_eval_count,
        prompt_eval_duration: r.prompt_eval_ns,
        eval_count: r.eval_count,
        eval_duration: r.eval_ns,
    })
    .into_response()
}

pub(super) async fn api_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaChatRequest>,
) -> Response {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let messages: Vec<crate::chat_template::ChatMessage> = req
        .messages
        .iter()
        .map(|m| crate::chat_template::ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    let prompt =
        crate::chat_template::apply_chat_template(state.chat_template.as_deref(), &messages);
    let cfg = options_config(&state.generation, req.options.as_ref());
    let model = req.model.clone().unwrap_or_else(|| state.name.clone());

    if req.stream.unwrap_or(true) {
        return ndjson_stream(state.clone(), prompt, cfg, model, StreamKind::Chat, req.format);
    }

    let start = Instant::now();
    let r = run_generation(&state, &prompt, &cfg, req.format.as_ref());
    Json(OllamaChatResponse {
        model,
        created_at: now_rfc3339(),
        message: Message {
            role: "assistant".to_string(),
            content: r.text,
        },
        done: true,
        done_reason: "stop".to_string(),
        context: r.context,
        total_duration: start.elapsed().as_nanos() as u64,
        load_duration: 0,
        prompt_eval_count: r.prompt_eval_count,
        prompt_eval_duration: r.prompt_eval_ns,
        eval_count: r.eval_count,
        eval_duration: r.eval_ns,
    })
    .into_response()
}

// ---------------------------------------------------------------------------
// `/api/embed` + `/api/embeddings` — Ollama embedding endpoints
// ---------------------------------------------------------------------------

/// Ollama `/api/embed` request. `input` accepts a string or an array of
/// strings; `truncate`, `options`, and `keep_alive` are ignored.
#[derive(Deserialize)]
pub struct OllamaEmbedRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub input: Value,
}

/// Legacy Ollama `/api/embeddings` request (single `prompt`).
#[derive(Deserialize)]
pub struct OllamaEmbeddingsRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: String,
}

/// Ollama `/api/embed` response shape.
#[derive(Serialize)]
pub struct OllamaEmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: usize,
}

/// Legacy `/api/embeddings` response shape.
#[derive(Serialize)]
pub struct OllamaEmbeddingsResponse {
    pub embedding: Vec<f32>,
}

/// Byte-normalized input vector — the same encoding `/embeddings` uses.
fn embed_input(text: &str) -> Vec<f32> {
    text.bytes().map(|b| b as f32 / 255.0).collect()
}

pub(super) async fn api_embed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaEmbedRequest>,
) -> Json<OllamaEmbedResponse> {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let start = Instant::now();
    let inputs = stop_strings(&req.input);
    let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
    let mut prompt_eval_count = 0usize;
    let embeddings: Vec<Vec<f32>> = inputs
        .iter()
        .map(|text| {
            prompt_eval_count += tokenizer.encode(text).len();
            run_embeddings(&state, &embed_input(text)).unwrap_or_default()
        })
        .collect();
    Json(OllamaEmbedResponse {
        model: req.model.unwrap_or_else(|| state.name.clone()),
        embeddings,
        total_duration: start.elapsed().as_nanos() as u64,
        load_duration: 0,
        prompt_eval_count,
    })
}

pub(super) async fn api_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaEmbeddingsRequest>,
) -> Json<OllamaEmbeddingsResponse> {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let _ = req.model; // single loaded artifact, like the other /api/* handlers
    Json(OllamaEmbeddingsResponse {
        embedding: run_embeddings(&state, &embed_input(&req.prompt)).unwrap_or_default(),
    })
}

// ---------------------------------------------------------------------------
// `/api/tags` + `/api/show` — Ollama model listing / details (registry surface)
// ---------------------------------------------------------------------------

/// `GET /api/tags` — Ollama-compatible model list.
///
/// Real Ollama clients parse `name`, `model`, `modified_at`, `size`, `digest`
/// and `details.*`. The extra modelc fields (`architecture`, `compressed`,
/// `tensor_count`) are ignored by clients.
#[derive(Serialize)]
pub(super) struct ApiTagsResponse {
    pub(super) models: Vec<ApiTag>,
}

#[derive(Serialize)]
pub(super) struct ApiTag {
    pub(super) name: String,
    pub(super) model: String,
    pub(super) modified_at: String,
    pub(super) size: u64,
    pub(super) digest: String,
    pub(super) details: OllamaDetails,
    // modelc extras
    pub(super) architecture: String,
    pub(super) tensor_count: usize,
    pub(super) compressed: bool,
}

/// Ollama `details` block (same shape for `/api/tags` and `/api/show`).
#[derive(Serialize)]
pub(super) struct OllamaDetails {
    pub(super) format: String,
    pub(super) family: String,
    pub(super) families: Vec<String>,
    pub(super) parameter_size: String,
    pub(super) quantization_level: String,
}

/// `POST /api/show` request — Ollama clients send `{"model": "..."}`, older
/// tooling (and previous modelc versions) send `{"name": "..."}`; accept both.
#[derive(Deserialize)]
pub(super) struct ApiShowRequest {
    pub(super) name: Option<String>,
    pub(super) model: Option<String>,
}

#[derive(Serialize)]
pub(super) struct ApiShowResponse {
    // Ollama schema
    pub(super) model: String,
    pub(super) modified_at: String,
    pub(super) size: u64,
    pub(super) digest: String,
    pub(super) details: OllamaDetails,
    pub(super) model_info: serde_json::Value,
    pub(super) capabilities: Vec<String>,
    pub(super) license: String,
    pub(super) modelfile: String,
    pub(super) parameters: String,
    pub(super) template: String,
    // modelc extras
    pub(super) name: String,
    pub(super) architecture: String,
    pub(super) size_bytes: u64,
    pub(super) parameter_size: String,
    pub(super) quantization: String,
    pub(super) tensor_count: usize,
}

/// `GET /api/tags` — Ollama-compatible model list over the local store.
///
/// Uses the real Ollama field names (`details.format/family/families/
/// parameter_size/quantization_level`, `digest`, `modified_at`) so Ollama
/// clients can enumerate the store; modelc extras are additive.
pub(super) async fn api_tags() -> Json<ApiTagsResponse> {
    let models = crate::store::list_models().unwrap_or_default();
    let tags: Vec<ApiTag> = models
        .iter()
        .map(|m| {
            let family = m.architecture.clone().unwrap_or_else(|| "generic".to_string());
            ApiTag {
                name: m.name.clone(),
                model: m.name.clone(),
                modified_at: crate::digest::file_mtime_rfc3339(&m.path),
                size: m.size_bytes,
                digest: crate::digest::digest_file(&m.path).unwrap_or_default(),
                details: OllamaDetails {
                    format: m.format.clone().unwrap_or_else(|| "modelc".to_string()),
                    family: family.clone(),
                    families: vec![family],
                    parameter_size: format_parameter_size(m.params.unwrap_or(0) as u64),
                    quantization_level: quantization_level(
                        m.quantization.as_deref().unwrap_or("f32"),
                    ),
                },
                architecture: m.architecture.clone().unwrap_or_default(),
                tensor_count: m.tensor_count,
                compressed: m.compressed,
            }
        })
        .collect();
    Json(ApiTagsResponse { models: tags })
}

/// `POST /api/show` — Ollama-compatible model details.
///
/// Accepts both `{"model": "..."}` (Ollama) and `{"name": "..."}` (legacy).
pub(super) async fn api_show(Json(req): Json<ApiShowRequest>) -> axum::response::Response {
    let requested = req.model.or(req.name).unwrap_or_default();
    if requested.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "model is required" })),
        )
            .into_response();
    }
    match crate::store::resolve_model_path(&requested) {
        Ok(path) => match crate::pack::read_header(&path) {
            Ok(header) => {
                let params: u64 = header
                    .tensors
                    .iter()
                    .map(|t| t.shape.iter().product::<usize>() as u64)
                    .sum();
                let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let quantization = crate::pack::infer_quantization(&header);
                let format = artifact_format(&header);
                let family = if header.architecture.is_empty() {
                    "generic".to_string()
                } else {
                    header.architecture.clone()
                };
                let model_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(header.name.as_str())
                    .to_string();
                Json(ApiShowResponse {
                    model: model_name,
                    modified_at: crate::digest::file_mtime_rfc3339(&path),
                    size: size_bytes,
                    digest: crate::digest::digest_file(&path).unwrap_or_default(),
                    details: OllamaDetails {
                        format,
                        family: family.clone(),
                        families: vec![family],
                        parameter_size: format_parameter_size(params),
                        quantization_level: quantization_level(&quantization),
                    },
                    model_info: serde_json::json!({
                        "general.architecture": header.architecture,
                        "general.parameter_count": params,
                        "general.file_type": ollama_file_type(&quantization),
                    }),
                    capabilities: vec![
                        "completion".to_string(),
                        "embedding".to_string(),
                        "insert".to_string(),
                        "tools".to_string(),
                    ],
                    license: header
                        .metadata
                        .get("license")
                        .cloned()
                        .unwrap_or_default(),
                    modelfile: String::new(),
                    parameters: String::new(),
                    template: String::new(),
                    name: header.name,
                    architecture: header.architecture,
                    size_bytes,
                    parameter_size: format_parameter_size(params),
                    quantization,
                    tensor_count: header.tensors.len(),
                })
                .into_response()
            }
            Err(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        Err(_) => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn parameter_size_formats_like_ollama() {
        assert_eq!(format_parameter_size(8_030_000_000), "8.0B");
        assert_eq!(format_parameter_size(134_520_000), "134.52M");
        assert_eq!(format_parameter_size(1_500), "1.5K");
        assert_eq!(format_parameter_size(42), "42");
        assert_eq!(format_parameter_size(0), "0");
    }

    #[test]
    fn quantization_level_maps_modelc_labels() {
        assert_eq!(quantization_level("f32"), "F32");
        assert_eq!(quantization_level("int8"), "Q8_0");
        assert_eq!(quantization_level("int4"), "Q4_0");
        assert_eq!(quantization_level("q4_k"), "Q4_K");
        assert_eq!(quantization_level("custom"), "CUSTOM");
    }

    #[test]
    fn file_type_matches_llama_cpp_enum() {
        assert_eq!(ollama_file_type("f32"), 0);
        assert_eq!(ollama_file_type("f16"), 1);
        assert_eq!(ollama_file_type("int4"), 2);
        assert_eq!(ollama_file_type("q8_0"), 7);
        assert_eq!(ollama_file_type("q6_k"), 12);
        assert_eq!(ollama_file_type("exotic"), 0, "unknown label → table fallback 0");
    }

    #[test]
    fn artifact_format_falls_back_to_modelc() {
        let h = ArtifactHeader {
            name: "x".into(),
            architecture: "mlp".into(),
            metadata: HashMap::new(),
            tensors: vec![],
        };
        assert_eq!(artifact_format(&h), "modelc");
    }
}

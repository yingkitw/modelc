//! Request/response DTOs for the native modelc HTTP API (`/infer`, `/chat`,
//! `/embeddings`, …). Kept in one module so `serve/mod.rs` stays focused on
//! state, routing, and middleware. Ollama-ecosystem DTOs live in `ollama.rs`
//! with the rest of that surface.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub(super) struct InferRequest {
    #[serde(default)]
    pub(super) input: Vec<f32>,
    #[serde(default)]
    pub(super) inputs: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
pub(super) struct LoraLoadRequest {
    /// Absolute or relative path to a Safetensors LoRA adapter file.
    pub(super) path: String,
    /// LoRA alpha scaling factor (default 1.0).
    #[serde(default = "default_lora_alpha")]
    pub(super) alpha: f32,
}

pub(super) fn default_lora_alpha() -> f32 {
    1.0
}

#[derive(Serialize)]
pub(super) struct LoraLoadResponse {
    pub(super) applied: usize,
    pub(super) skipped: usize,
    pub(super) message: String,
}

#[derive(Serialize)]
pub(super) struct LoraUnloadResponse {
    pub(super) message: String,
}

#[derive(Serialize)]
pub(super) struct InferResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) output: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) outputs: Option<Vec<Vec<f32>>>,
}

#[derive(Serialize)]
pub(super) struct HealthResponse {
    pub(super) status: String,
    pub(super) model: String,
    pub(super) architecture: String,
}

#[derive(Serialize)]
pub(super) struct ModelInfo {
    pub(super) name: String,
    pub(super) architecture: String,
    pub(super) total_params: usize,
    pub(super) total_bytes: usize,
    pub(super) tensors: Vec<String>,
}

#[derive(Serialize)]
pub(super) struct ServerProps {
    pub(super) model: String,
    pub(super) architecture: String,
    pub(super) total_params: usize,
    pub(super) total_bytes: usize,
    pub(super) chat_template: Option<String>,
    pub(super) default_generation: DefaultGenerationProps,
}

#[derive(Serialize)]
pub(super) struct DefaultGenerationProps {
    pub(super) max_tokens: usize,
    pub(super) temperature: f32,
    pub(super) top_p: f32,
    pub(super) min_p: f32,
    pub(super) repetition_penalty: f32,
    pub(super) presence_penalty: f32,
    pub(super) frequency_penalty: f32,
    pub(super) gamma: usize,
}

#[derive(Serialize)]
pub(super) struct VersionInfo {
    pub(super) version: String,
    pub(super) git_sha: String,
}

#[derive(Deserialize)]
pub(super) struct TokenizeRequest {
    #[serde(default)]
    pub(super) input: String,
    #[serde(default)]
    pub(super) inputs: Vec<String>,
}

#[derive(Serialize)]
pub(super) struct TokenizeResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tokens: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tokens_batch: Option<Vec<Vec<u32>>>,
    /// Total number of tokens across all inputs.
    pub(super) count: usize,
}

#[derive(Deserialize)]
pub(super) struct DetokenizeRequest {
    #[serde(default)]
    pub(super) tokens: Vec<u32>,
}

#[derive(Serialize)]
pub(super) struct DetokenizeResponse {
    pub(super) text: String,
}

/// System/hardware info exposed by `GET /v1/system`.
#[derive(Serialize)]
pub(super) struct SystemInfo {
    pub(super) model: String,
    pub(super) architecture: String,
    pub(super) total_params: usize,
    pub(super) total_bytes: usize,
    /// Logical CPU cores available to the process.
    pub(super) cpu_cores: usize,
    /// Operating system name (e.g. "macos", "linux", "windows").
    pub(super) os: &'static str,
    /// CPU architecture (e.g. "aarch64", "x86_64").
    pub(super) cpu_arch: &'static str,
    /// Pointer width in bits (32 or 64).
    pub(super) pointer_width: usize,
    /// Whether Apple Silicon Metal GPU acceleration is compiled in.
    pub(super) metal_available: bool,
    /// Best-effort total physical memory in bytes (`null` if unavailable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) memory_total_bytes: Option<u64>,
}

#[derive(Deserialize)]
pub(super) struct EmbeddingsRequest {
    #[serde(default)]
    pub(super) input: String,
    #[serde(default)]
    pub(super) inputs: Vec<String>,
}

#[derive(Serialize)]
pub(super) struct EmbeddingEntry {
    pub(super) embedding: Vec<f32>,
    pub(super) index: usize,
}

#[derive(Serialize)]
pub(super) struct EmbeddingsResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) embedding: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) embeddings: Option<Vec<EmbeddingEntry>>,
    pub(super) model: String,
}

#[derive(Deserialize)]
pub(super) struct RerankingRequest {
    pub(super) query: String,
    pub(super) documents: Vec<String>,
    #[serde(default)]
    pub(super) top_n: Option<usize>,
}

#[derive(Serialize)]
pub(super) struct RerankingResponse {
    pub(super) model: String,
    pub(super) results: Vec<RerankingResult>,
}

#[derive(Serialize)]
pub(super) struct RerankingResult {
    pub(super) index: usize,
    pub(super) relevance_score: f32,
    pub(super) document: Document,
}

#[derive(Serialize)]
pub(super) struct Document {
    pub(super) text: String,
}

/// `GET /api/cache/stats` response.
#[derive(Serialize)]
pub(super) struct CacheStatsResponse {
    pub(super) entries: usize,
    pub(super) capacity: usize,
    pub(super) total_tokens: usize,
    pub(super) total_kv_bytes: usize,
    /// `entries / capacity`, in the closed interval `[0.0, 1.0]`.
    pub(super) utilization: f64,
}

#[derive(Deserialize)]
pub(super) struct ChatRequest {
    pub(super) messages: Vec<Message>,
    #[serde(default)]
    pub(super) max_tokens: Option<usize>,
    #[serde(default)]
    pub(super) temperature: Option<f32>,
    #[serde(default)]
    pub(super) top_p: Option<f32>,
    /// Optional min-p sampling threshold. Keeps tokens whose probability is at
    /// least `min_p` fraction of the max probability.
    #[serde(default)]
    pub(super) min_p: Option<f32>,
    /// Optional regex grammar constraint applied during sampling.
    #[serde(default)]
    pub(super) grammar: Option<String>,
    /// Optional JSON Schema object to validate generated output against.
    #[serde(default)]
    pub(super) json_schema: Option<serde_json::Value>,
    /// Optional stop sequences that halt generation.
    #[serde(default)]
    pub(super) stop: Vec<String>,
    /// Optional seed for reproducible sampling.
    #[serde(default)]
    pub(super) seed: Option<u64>,
    /// Penalty for repeated tokens. Values > 1.0 discourage repetition.
    #[serde(default)]
    pub(super) repetition_penalty: Option<f32>,
    /// OpenAI-style presence penalty. Positive values discourage token reuse.
    #[serde(default)]
    pub(super) presence_penalty: Option<f32>,
    /// OpenAI-style frequency penalty. Scales with token occurrence count.
    #[serde(default)]
    pub(super) frequency_penalty: Option<f32>,
    /// OpenAI-style logit bias: map from token ID to additive bias (-100 to 100).
    #[serde(default)]
    pub(super) logit_bias: Option<HashMap<u32, f32>>,
}

#[derive(Deserialize, Serialize, Clone)]
pub(super) struct Message {
    pub(super) role: String,
    pub(super) content: String,
}

#[derive(Serialize)]
pub(super) struct ChatResponse {
    pub(super) message: Message,
}

#[derive(Deserialize)]
pub(super) struct CompleteRequest {
    pub(super) prompt: String,
    #[serde(default)]
    pub(super) max_tokens: Option<usize>,
    #[serde(default)]
    pub(super) temperature: Option<f32>,
    #[serde(default)]
    pub(super) top_p: Option<f32>,
    /// Optional min-p sampling threshold. Keeps tokens whose probability is at
    /// least `min_p` fraction of the max probability.
    #[serde(default)]
    pub(super) min_p: Option<f32>,
    /// Optional regex grammar constraint applied during sampling.
    #[serde(default)]
    pub(super) grammar: Option<String>,
    /// Optional JSON Schema object to validate generated output against.
    #[serde(default)]
    pub(super) json_schema: Option<serde_json::Value>,
    /// Optional stop sequences that halt generation.
    #[serde(default)]
    pub(super) stop: Vec<String>,
    /// Optional seed for reproducible sampling.
    #[serde(default)]
    pub(super) seed: Option<u64>,
    /// Penalty for repeated tokens. Values > 1.0 discourage repetition.
    #[serde(default)]
    pub(super) repetition_penalty: Option<f32>,
    /// OpenAI-style presence penalty. Positive values discourage token reuse.
    #[serde(default)]
    pub(super) presence_penalty: Option<f32>,
    /// OpenAI-style frequency penalty. Scales with token occurrence count.
    #[serde(default)]
    pub(super) frequency_penalty: Option<f32>,
    /// OpenAI-style logit bias: map from token ID to additive bias (-100 to 100).
    #[serde(default)]
    pub(super) logit_bias: Option<HashMap<u32, f32>>,
}

#[derive(Serialize)]
pub(super) struct CompleteResponse {
    pub(super) completion: String,
}

#[derive(Deserialize)]
pub(super) struct InfillRequest {
    pub(super) prefix: String,
    pub(super) suffix: String,
    #[serde(default)]
    pub(super) prompt: Option<String>,
    #[serde(default)]
    pub(super) max_tokens: Option<usize>,
    #[serde(default)]
    pub(super) temperature: Option<f32>,
    #[serde(default)]
    pub(super) top_p: Option<f32>,
    #[serde(default)]
    pub(super) min_p: Option<f32>,
    #[serde(default)]
    pub(super) seed: Option<u64>,
    #[serde(default)]
    pub(super) repetition_penalty: Option<f32>,
    #[serde(default)]
    pub(super) presence_penalty: Option<f32>,
    #[serde(default)]
    pub(super) frequency_penalty: Option<f32>,
}

#[derive(Serialize)]
pub(super) struct InfillResponse {
    pub(super) completion: String,
}

#[derive(Serialize)]
pub(super) struct StreamChunk {
    pub(super) delta: String,
    pub(super) done: bool,
}

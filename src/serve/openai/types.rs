//! Serde DTOs for the OpenAI-compatible surface (`/v1/models`,
//! `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`). Handler logic
//! stays in `serve/openai.rs`; this module holds the wire schema only.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub(in crate::serve) struct ModelsList {
    pub(in crate::serve)object: &'static str,
    pub(in crate::serve)data: Vec<ModelEntry>,
}

#[derive(Serialize)]
pub(in crate::serve) struct ModelEntry {
    pub(in crate::serve)id: String,
    pub(in crate::serve)object: &'static str,
    pub(in crate::serve)created: u64,
    pub(in crate::serve)owned_by: &'static str,
}

/// Sampling parameters shared by `/v1/chat/completions` and `/v1/completions`.
/// Serde-flattened into both request structs, so the JSON surface is unchanged.
#[derive(Deserialize, Default)]
pub(in crate::serve) struct SamplingParams {
    #[serde(default)]
    pub(in crate::serve)max_tokens: Option<usize>,
    /// Newer OpenAI parameter name for max_tokens (preferred if both given).
    #[serde(default)]
    pub(in crate::serve)max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub(in crate::serve)temperature: Option<f32>,
    #[serde(default)]
    pub(in crate::serve)top_p: Option<f32>,
    /// Optional min-p sampling threshold. Keeps tokens whose probability is at
    /// least `min_p` fraction of the max probability.
    #[serde(default)]
    pub(in crate::serve)min_p: Option<f32>,
    /// Optional regex grammar constraint applied during sampling.
    #[serde(default)]
    pub(in crate::serve)grammar: Option<String>,
    /// Optional stop sequences that halt generation.
    #[serde(default)]
    pub(in crate::serve)stop: Vec<String>,
    /// Optional seed for reproducible sampling.
    #[serde(default)]
    pub(in crate::serve)seed: Option<u64>,
    /// Penalty for repeated tokens. Values > 1.0 discourage repetition.
    #[serde(default)]
    pub(in crate::serve)repetition_penalty: Option<f32>,
    /// OpenAI-style presence penalty. Positive values discourage token reuse.
    #[serde(default)]
    pub(in crate::serve)presence_penalty: Option<f32>,
    /// OpenAI-style frequency penalty. Scales with token occurrence count.
    #[serde(default)]
    pub(in crate::serve)frequency_penalty: Option<f32>,
    /// OpenAI-style logit bias: map from token ID to additive bias (-100 to 100).
    #[serde(default)]
    pub(in crate::serve)logit_bias: Option<HashMap<u32, f32>>,
}

impl SamplingParams {
    /// Layer the request's sampling overrides onto the server defaults.
    pub(in crate::serve) fn to_generation_config(
        &self,
        base: &crate::generate::GenerationConfig,
    ) -> crate::generate::GenerationConfig {
        crate::generate::GenerationConfig {
            max_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(base.max_tokens),
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            min_p: self.min_p.unwrap_or(base.min_p),
            gamma: base.gamma,
            use_int8_kv: base.use_int8_kv,
            use_mixed_kv: base.use_mixed_kv,
            constraint: self
                .grammar
                .as_ref()
                .and_then(|pat| {
                    crate::constraint::RegexConstraint::new(pat).map(|c| {
                        std::sync::Arc::new(c)
                            as std::sync::Arc<dyn crate::constraint::Constraint>
                    })
                })
                .or_else(|| base.constraint.clone()),
            max_context: base.max_context,
            anchor_tokens: base.anchor_tokens,
            stop: if self.stop.is_empty() {
                base.stop.clone()
            } else {
                self.stop.clone()
            },
            seed: self.seed.or(base.seed),
            repetition_penalty: self.repetition_penalty.unwrap_or(base.repetition_penalty),
            presence_penalty: self.presence_penalty.unwrap_or(base.presence_penalty),
            frequency_penalty: self.frequency_penalty.unwrap_or(base.frequency_penalty),
            logit_bias: self
                .logit_bias
                .clone()
                .unwrap_or_else(|| base.logit_bias.clone()),
            cancel: None,
        }
    }
}

#[derive(Deserialize)]
pub(in crate::serve) struct ChatCompletionRequest {
    #[serde(default)]
    pub(in crate::serve)model: String,
    pub(in crate::serve)messages: Vec<ChatMessage>,
    #[serde(default)]
    pub(in crate::serve)stream: bool,
    #[serde(default)]
    pub(in crate::serve)response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub(in crate::serve)tools: Vec<Tool>,
    /// Optional tool choice: "auto", "none", or {"type":"function","function":{"name":"..."}}
    #[serde(default)]
    pub(in crate::serve)tool_choice: Option<serde_json::Value>,
    /// Whether to return log probabilities of the output tokens.
    #[serde(default)]
    pub(in crate::serve)logprobs: Option<bool>,
    /// How many (0–20) top tokens to return per output position. Requires `logprobs: true`.
    #[serde(default)]
    pub(in crate::serve)top_logprobs: Option<u8>,
    /// Optional JSON Schema object to validate generated output against.
    #[serde(default)]
    pub(in crate::serve)json_schema: Option<serde_json::Value>,
    /// How many chat completions to generate. Defaults to 1.
    #[serde(default)]
    pub(in crate::serve)n: Option<u32>,
    /// Optional streaming options (OpenAI-standard).
    #[serde(default)]
    pub(in crate::serve)stream_options: Option<StreamOptions>,
    /// Optional user identifier (OpenAI-standard, accepted but ignored).
    #[serde(default)]
    #[allow(dead_code)]
    pub(in crate::serve)user: Option<String>,
    #[serde(flatten)]
    pub(in crate::serve)params: SamplingParams,
}

#[derive(Deserialize)]
pub(in crate::serve) struct StreamOptions {
    #[serde(default)]
    pub(in crate::serve)include_usage: bool,
}

#[derive(Deserialize)]
pub(in crate::serve) struct ResponseFormat {
    #[serde(rename = "type")]
    pub(in crate::serve)format_type: String,
    /// Optional json_schema definition (for type = "json_schema").
    #[serde(default)]
    pub(in crate::serve)json_schema: Option<JsonSchemaDef>,
}

#[derive(Deserialize)]
pub(in crate::serve) struct JsonSchemaDef {
    #[serde(default)]
    #[allow(dead_code)] // OpenAI field accepted for API fidelity, ignored.
    pub(in crate::serve)name: Option<String>,
    #[serde(default)]
    pub(in crate::serve)schema: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)] // OpenAI field accepted for API fidelity, ignored.
    pub(in crate::serve)strict: Option<bool>,
}

#[derive(Deserialize, Clone)]
pub(in crate::serve) struct Tool {
    pub(in crate::serve)function: ToolFunction,
}

#[derive(Deserialize, Clone)]
pub(in crate::serve) struct ToolFunction {
    pub(in crate::serve)name: String,
    #[serde(default)]
    pub(in crate::serve)description: String,
}

#[derive(Deserialize, Serialize, Clone)]
pub(in crate::serve) struct ChatMessage {
    pub(in crate::serve)role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::serve)content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::serve)name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::serve)tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(in crate::serve) struct ToolCall {
    pub(in crate::serve)id: String,
    #[serde(rename = "type")]
    pub(in crate::serve)call_type: String,
    pub(in crate::serve)function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(in crate::serve) struct ToolCallFunction {
    pub(in crate::serve)name: String,
    pub(in crate::serve)arguments: String,
}

#[derive(Serialize)]
pub(in crate::serve) struct ChatCompletionResponse {
    pub(in crate::serve)id: String,
    pub(in crate::serve)object: &'static str,
    pub(in crate::serve)created: u64,
    pub(in crate::serve)model: String,
    pub(in crate::serve)choices: Vec<Choice>,
    pub(in crate::serve)usage: Usage,
    /// OpenAI-compatible fingerprint identifying the backend build.
    pub(in crate::serve)system_fingerprint: String,
}

#[derive(Serialize)]
pub(in crate::serve) struct Choice {
    pub(in crate::serve)index: usize,
    pub(in crate::serve)message: ChatMessage,
    pub(in crate::serve)finish_reason: &'static str,
    /// `null` when `logprobs` is not requested; an object with a `content` array otherwise.
    pub(in crate::serve)logprobs: Option<Logprobs>,
}

/// OpenAI-shaped logprobs payload (`choices[].logprobs.content`).
#[derive(Serialize)]
pub(in crate::serve) struct Logprobs {
    pub(in crate::serve)content: Vec<ContentLogprob>,
}

#[derive(Serialize)]
pub(in crate::serve) struct ContentLogprob {
    pub(in crate::serve)token: String,
    pub(in crate::serve)logprob: f64,
    /// Raw UTF-8 bytes of `token` (OpenAI emits this as an array of integers).
    pub(in crate::serve)bytes: Vec<u8>,
    /// Up to `top_logprobs` alternative tokens for this position, sorted by
    /// descending probability. Empty unless `top_logprobs > 0` was requested.
    pub(in crate::serve)top_logprobs: Vec<TopLogprob>,
}

#[derive(Serialize)]
pub(in crate::serve) struct TopLogprob {
    pub(in crate::serve)token: String,
    pub(in crate::serve)logprob: f64,
    pub(in crate::serve)bytes: Vec<u8>,
}

#[derive(Serialize)]
pub(in crate::serve) struct Usage {
    pub(in crate::serve)prompt_tokens: usize,
    pub(in crate::serve)completion_tokens: usize,
    pub(in crate::serve)total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Streaming chunk types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(in crate::serve) struct ChatCompletionChunk {
    pub(in crate::serve)id: String,
    pub(in crate::serve)object: &'static str,
    pub(in crate::serve)created: u64,
    pub(in crate::serve)model: String,
    pub(in crate::serve)choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::serve)usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::serve)system_fingerprint: Option<String>,
}

#[derive(Serialize)]
pub(in crate::serve) struct ChunkChoice {
    pub(in crate::serve)index: usize,
    pub(in crate::serve)delta: ChunkDelta,
    pub(in crate::serve)finish_reason: Option<&'static str>,
}

#[derive(Serialize, Default)]
pub(in crate::serve) struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::serve)role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::serve)content: Option<String>,
}

/// Accepts prompt as either a single string or an array of strings (batch mode).
#[derive(Deserialize)]
#[serde(untagged)]
pub(in crate::serve) enum PromptInput {
    Single(String),
    Batch(Vec<String>),
}

impl PromptInput {
    pub(in crate::serve) fn into_vec(self) -> Vec<String> {
        match self {
            PromptInput::Single(s) => vec![s],
            PromptInput::Batch(v) => v,
        }
    }
}

impl Default for PromptInput {
    fn default() -> Self {
        PromptInput::Single(String::new())
    }
}

#[derive(Deserialize)]
pub(in crate::serve) struct CompletionRequest {
    #[serde(default)]
    pub(in crate::serve)model: String,
    /// Prompt can be a single string or an array of strings (batch mode).
    #[serde(default)]
    pub(in crate::serve)prompt: PromptInput,
    /// Whether to return log probabilities of the output tokens.
    #[serde(default)]
    pub(in crate::serve)logprobs: Option<bool>,
    /// How many (0–20) top tokens to return per output position. Requires `logprobs: true`.
    #[serde(default)]
    pub(in crate::serve)top_logprobs: Option<u8>,
    /// Echo back the prompt in addition to the completion.
    #[serde(default)]
    pub(in crate::serve)echo: Option<bool>,
    /// How many completions to generate. Defaults to 1.
    #[serde(default)]
    pub(in crate::serve)n: Option<u32>,
    /// Generate best_of completions server-side and return the top n by length.
    /// Defaults to n (or 1). Cannot be used with stream.
    #[serde(default)]
    pub(in crate::serve)best_of: Option<u32>,
    /// Optional suffix — text to append after the generated completion.
    #[serde(default)]
    pub(in crate::serve)suffix: Option<String>,
    /// Optional streaming options (OpenAI-standard).
    #[serde(default)]
    pub(in crate::serve)stream_options: Option<StreamOptions>,
    /// Optional user identifier (OpenAI-standard, accepted but ignored).
    #[serde(default)]
    #[allow(dead_code)]
    pub(in crate::serve)user: Option<String>,
    #[serde(default)]
    pub(in crate::serve)stream: bool,
    #[serde(flatten)]
    pub(in crate::serve)params: SamplingParams,
}

#[derive(Serialize)]
pub(in crate::serve) struct CompletionResponse {
    pub(in crate::serve)id: String,
    pub(in crate::serve)object: &'static str,
    pub(in crate::serve)created: u64,
    pub(in crate::serve)model: String,
    pub(in crate::serve)choices: Vec<CompletionChoice>,
    pub(in crate::serve)usage: Usage,
    /// OpenAI-compatible fingerprint identifying the backend build.
    pub(in crate::serve)system_fingerprint: String,
}

#[derive(Serialize)]
pub(in crate::serve) struct CompletionChoice {
    pub(in crate::serve)index: usize,
    pub(in crate::serve)text: String,
    pub(in crate::serve)finish_reason: &'static str,
    pub(in crate::serve)logprobs: Option<Logprobs>,
}

#[derive(Serialize)]
pub(in crate::serve) struct CompletionChunk {
    pub(in crate::serve)id: String,
    pub(in crate::serve)object: &'static str,
    pub(in crate::serve)created: u64,
    pub(in crate::serve)model: String,
    pub(in crate::serve)choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::serve)usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::serve)system_fingerprint: Option<String>,
}

#[derive(Serialize)]
pub(in crate::serve) struct CompletionChunkChoice {
    pub(in crate::serve)index: usize,
    pub(in crate::serve)text: String,
    pub(in crate::serve)finish_reason: Option<&'static str>,
}

#[derive(Deserialize)]
pub(in crate::serve) struct V1EmbeddingsRequest {
    /// A string or array of strings to embed.
    pub(in crate::serve)input: serde_json::Value,
    #[serde(default)]
    pub(in crate::serve)model: String,
    /// `"float"` (default) or `"base64"`.
    #[serde(default = "default_encoding_format")]
    pub(in crate::serve)encoding_format: String,
}

pub(in crate::serve) fn default_encoding_format() -> String {
    "float".to_string()
}

/// Standard base64 encoder (RFC 4648, no padding-stripping).
/// Inline to avoid pulling in the `base64` crate.
pub(in crate::serve) fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(n & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

#[derive(Serialize)]
pub(in crate::serve) struct V1EmbeddingsResponse {
    pub(in crate::serve)object: &'static str,
    pub(in crate::serve)data: Vec<V1Embedding>,
    pub(in crate::serve)model: String,
    pub(in crate::serve)usage: Usage,
}

#[derive(Serialize)]
pub(in crate::serve) struct V1Embedding {
    pub(in crate::serve)object: &'static str,
    pub(in crate::serve)embedding: serde_json::Value,
    pub(in crate::serve)index: usize,
    pub(in crate::serve)dimensions: usize,
}

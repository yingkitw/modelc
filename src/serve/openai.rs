//! OpenAI-compatible API endpoints (`/v1/models`, `/v1/chat/completions`, `/v1/completions`).

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::{IntoResponse, Sse};
use axum::{Json, extract::State, response::sse::Event};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;

use super::AppState;
use super::infer::{run_text_inference_with_config, run_text_inference_with_logprobs};

// ---------------------------------------------------------------------------
// /v1/models
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(super) struct ModelsList {
    object: &'static str,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
pub(super) struct ModelEntry {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

pub(super) async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsList> {
    Json(ModelsList {
        object: "list",
        data: vec![ModelEntry {
            id: state.name.clone(),
            object: "model",
            created: 0,
            owned_by: "modelc",
        }],
    })
}

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(super) struct ChatCompletionRequest {
    #[serde(default)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    #[serde(default)]
    tools: Vec<Tool>,
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
    /// Whether to return log probabilities of the output tokens.
    #[serde(default)]
    logprobs: Option<bool>,
    /// How many (0–20) top tokens to return per output position. Requires `logprobs: true`.
    #[serde(default)]
    top_logprobs: Option<u8>,
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
}

#[derive(Deserialize)]
pub(super) struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Deserialize, Clone)]
pub(super) struct Tool {
    function: ToolFunction,
}

#[derive(Deserialize, Clone)]
pub(super) struct ToolFunction {
    name: String,
    #[serde(default)]
    description: String,
}

#[derive(Deserialize, Serialize, Clone)]
pub(super) struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(super) struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(super) struct ToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
pub(super) struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
pub(super) struct Choice {
    index: usize,
    message: ChatMessage,
    finish_reason: &'static str,
    /// `null` when `logprobs` is not requested; an object with a `content` array otherwise.
    logprobs: Option<Logprobs>,
}

/// OpenAI-shaped logprobs payload (`choices[].logprobs.content`).
#[derive(Serialize)]
pub(super) struct Logprobs {
    content: Vec<ContentLogprob>,
}

#[derive(Serialize)]
pub(super) struct ContentLogprob {
    token: String,
    logprob: f64,
    /// Raw UTF-8 bytes of `token` (OpenAI emits this as an array of integers).
    bytes: Vec<u8>,
    /// Up to `top_logprobs` alternative tokens for this position, sorted by
    /// descending probability. Empty unless `top_logprobs > 0` was requested.
    top_logprobs: Vec<TopLogprob>,
}

#[derive(Serialize)]
pub(super) struct TopLogprob {
    token: String,
    logprob: f64,
    bytes: Vec<u8>,
}

#[derive(Serialize)]
pub(super) struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Streaming chunk types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(super) struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
pub(super) struct ChunkChoice {
    index: usize,
    delta: ChunkDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize, Default)]
pub(super) struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

pub(super) async fn chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let is_json_mode = req
        .response_format
        .as_ref()
        .is_some_and(|rf| rf.format_type == "json_object");
    let has_tools = !req.tools.is_empty();
    let stream = req.stream;

    let mut messages: Vec<crate::chat_template::ChatMessage> = req
        .messages
        .iter()
        .map(|m| crate::chat_template::ChatMessage {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
        })
        .collect();

    if is_json_mode {
        messages.push(crate::chat_template::ChatMessage {
            role: "system".to_string(),
            content: "Respond with valid JSON only. Do not include any explanatory text before or after the JSON.".to_string(),
        });
    }

    if has_tools {
        let tool_desc = build_tool_description(&req.tools);
        messages.push(crate::chat_template::ChatMessage {
            role: "system".to_string(),
            content: tool_desc,
        });
    }

    let prompt =
        crate::chat_template::apply_chat_template(state.chat_template.as_deref(), &messages)
            .unwrap_or_else(|_| {
                req.messages
                    .last()
                    .and_then(|m| m.content.clone())
                    .unwrap_or_default()
            });

    let constraint = req
        .grammar
        .and_then(|pat| {
            crate::constraint::RegexConstraint::new(&pat).map(|c| {
                std::sync::Arc::new(c) as std::sync::Arc<dyn crate::constraint::Constraint>
            })
        })
        .or_else(|| state.generation.constraint.clone());
    let gen_cfg = crate::generate::GenerationConfig {
        max_tokens: req.max_tokens.unwrap_or(state.generation.max_tokens),
        temperature: req.temperature.unwrap_or(state.generation.temperature),
        top_p: req.top_p.unwrap_or(state.generation.top_p),
        min_p: req.min_p.unwrap_or(state.generation.min_p),
        gamma: state.generation.gamma,
        use_int8_kv: state.generation.use_int8_kv,
        use_mixed_kv: state.generation.use_mixed_kv,
        constraint,
        max_context: state.generation.max_context,
        anchor_tokens: state.generation.anchor_tokens,
        stop: if req.stop.is_empty() {
            state.generation.stop.clone()
        } else {
            req.stop
        },
        seed: req.seed.or(state.generation.seed),
        repetition_penalty: req
            .repetition_penalty
            .unwrap_or(state.generation.repetition_penalty),
        presence_penalty: req
            .presence_penalty
            .unwrap_or(state.generation.presence_penalty),
        frequency_penalty: req
            .frequency_penalty
            .unwrap_or(state.generation.frequency_penalty),
    };

    let model_name = if req.model.is_empty() {
        state.name.clone()
    } else {
        req.model
    };
    let id = format!("chatcmpl-{}-0", state.name);

    // ------------------------------------------------------------------
    // Streaming path
    // ------------------------------------------------------------------
    if stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(4);
        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            let token_ids =
                super::infer::run_text_inference_token_ids(&state_clone, &prompt, &gen_cfg);
            let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
            let prompt_ids = tokenizer.encode(&prompt);
            let mut prev_text = String::new();
            let mut first = true;

            for (idx, &_token_id) in token_ids.iter().enumerate() {
                let cumulative = [prompt_ids.as_slice(), &token_ids[..=idx]].concat();
                let text = tokenizer.decode(&cumulative);
                if let Some(delta) = text.strip_prefix(&prev_text) {
                    if !delta.is_empty() {
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created: 0,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: if first {
                                        Some("assistant".to_string())
                                    } else {
                                        None
                                    },
                                    content: Some(delta.to_string()),
                                },
                                finish_reason: None,
                            }],
                        };
                        let _ = tx
                            .send(Ok(
                                Event::default().data(serde_json::to_string(&chunk).unwrap())
                            ))
                            .await;
                        first = false;
                    }
                    prev_text = text;
                }
            }

            // Final chunk with finish_reason.
            let final_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created: 0,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta::default(),
                    finish_reason: Some("stop"),
                }],
            };
            let _ = tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&final_chunk).unwrap())
                ))
                .await;
            // OpenAI streams end with [DONE].
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        return Sse::new(ReceiverStream::new(rx)).into_response();
    }

    // ------------------------------------------------------------------
    // Non-streaming path (existing logic)
    // ------------------------------------------------------------------
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_n = req.top_logprobs.unwrap_or(0).min(20) as usize;

    let (raw_output, logprobs_field): (String, Option<Logprobs>) = if want_logprobs {
        match run_text_inference_with_logprobs(&state, &prompt, &gen_cfg, top_n) {
            Some((ids, lps)) => {
                let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
                let text = tokenizer.decode(&ids);
                let content = build_logprobs_content(&tokenizer, &lps);
                (text, Some(Logprobs { content }))
            }
            None => {
                let text = run_text_inference_with_config(&state, &prompt, &gen_cfg);
                (
                    text,
                    Some(Logprobs {
                        content: Vec::new(),
                    }),
                )
            }
        }
    } else {
        let text = if let Some(ref schema) = req.json_schema {
            crate::json_schema::generate_with_schema(
                |cfg| run_text_inference_with_config(&state, &prompt, cfg),
                schema,
                &gen_cfg,
                3,
            )
        } else {
            run_text_inference_with_config(&state, &prompt, &gen_cfg)
        };
        (text, None)
    };

    let (message, finish_reason) = if has_tools {
        if let Some(tool_calls) = parse_tool_calls(&raw_output) {
            (
                ChatMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(tool_calls),
                },
                "tool_calls",
            )
        } else {
            let content = extract_content(&raw_output, is_json_mode);
            (
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(content),
                    tool_calls: None,
                },
                "stop",
            )
        }
    } else {
        let content = extract_content(&raw_output, is_json_mode);
        (
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(content),
                tool_calls: None,
            },
            "stop",
        )
    };

    let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
    let prompt_tokens = tokenizer.encode(&prompt).len();
    let completion_tokens = tokenizer.encode(&raw_output).len();

    Json(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: 0,
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason,
            logprobs: logprobs_field,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response()
}

fn extract_content(raw: &str, is_json_mode: bool) -> String {
    if is_json_mode {
        extract_json_object(raw).unwrap_or_else(|| raw.to_string())
    } else {
        raw.to_string()
    }
}

/// Build the OpenAI-shaped `choices[].logprobs.content` array from per-token logprobs.
///
/// Each entry includes the decoded `token` text, its `logprob`, its raw UTF-8 `bytes`,
/// and up to N `top_logprobs` alternatives (decoded the same way).
fn build_logprobs_content(
    tokenizer: &crate::tokenizer::BpeTokenizer,
    lps: &[crate::generate::TokenLogprob],
) -> Vec<ContentLogprob> {
    lps.iter()
        .map(|lp| {
            let bytes = tokenizer
                .token_bytes(lp.token)
                .map(|b| b.to_vec())
                .unwrap_or_default();
            let token_str = String::from_utf8_lossy(&bytes).into_owned();
            let top_logprobs = lp
                .top_logprobs
                .iter()
                .map(|(id, logprob)| {
                    let tb = tokenizer
                        .token_bytes(*id)
                        .map(|b| b.to_vec())
                        .unwrap_or_default();
                    TopLogprob {
                        token: String::from_utf8_lossy(&tb).into_owned(),
                        logprob: *logprob as f64,
                        bytes: tb,
                    }
                })
                .collect();
            ContentLogprob {
                token: token_str,
                logprob: lp.logprob as f64,
                bytes,
                top_logprobs,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// /v1/completions (legacy non-chat completions API)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(super) struct CompletionRequest {
    #[serde(default)]
    model: String,
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
    /// Whether to return log probabilities of the output tokens.
    #[serde(default)]
    logprobs: Option<bool>,
    /// How many (0–20) top tokens to return per output position. Requires `logprobs: true`.
    #[serde(default)]
    top_logprobs: Option<u8>,
    /// Optional regex grammar constraint applied during sampling.
    #[serde(default)]
    grammar: Option<String>,
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
    #[serde(default)]
    stream: bool,
}

#[derive(Serialize)]
pub(super) struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
pub(super) struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: &'static str,
    logprobs: Option<Logprobs>,
}

#[derive(Serialize)]
pub(super) struct CompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChunkChoice>,
}

#[derive(Serialize)]
pub(super) struct CompletionChunkChoice {
    index: usize,
    text: String,
    finish_reason: Option<&'static str>,
}

pub(super) async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let stream = req.stream;

    let constraint = req
        .grammar
        .and_then(|pat| {
            crate::constraint::RegexConstraint::new(&pat).map(|c| {
                std::sync::Arc::new(c) as std::sync::Arc<dyn crate::constraint::Constraint>
            })
        })
        .or_else(|| state.generation.constraint.clone());
    let gen_cfg = crate::generate::GenerationConfig {
        max_tokens: req.max_tokens.unwrap_or(state.generation.max_tokens),
        temperature: req.temperature.unwrap_or(state.generation.temperature),
        top_p: req.top_p.unwrap_or(state.generation.top_p),
        min_p: req.min_p.unwrap_or(state.generation.min_p),
        gamma: state.generation.gamma,
        use_int8_kv: state.generation.use_int8_kv,
        use_mixed_kv: state.generation.use_mixed_kv,
        constraint,
        max_context: state.generation.max_context,
        anchor_tokens: state.generation.anchor_tokens,
        stop: if req.stop.is_empty() {
            state.generation.stop.clone()
        } else {
            req.stop
        },
        seed: req.seed.or(state.generation.seed),
        repetition_penalty: req
            .repetition_penalty
            .unwrap_or(state.generation.repetition_penalty),
        presence_penalty: req
            .presence_penalty
            .unwrap_or(state.generation.presence_penalty),
        frequency_penalty: req
            .frequency_penalty
            .unwrap_or(state.generation.frequency_penalty),
    };

    let model_name = if req.model.is_empty() {
        state.name.clone()
    } else {
        req.model
    };
    let id = format!("cmpl-{}-0", state.name);

    // ------------------------------------------------------------------
    // Streaming path
    // ------------------------------------------------------------------
    if stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(4);
        let state_clone = Arc::clone(&state);
        let prompt_clone = req.prompt.clone();
        tokio::spawn(async move {
            let token_ids =
                super::infer::run_text_inference_token_ids(&state_clone, &prompt_clone, &gen_cfg);
            let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
            let prompt_ids = tokenizer.encode(&prompt_clone);
            let mut prev_text = String::new();

            for (idx, &_token_id) in token_ids.iter().enumerate() {
                let cumulative = [prompt_ids.as_slice(), &token_ids[..=idx]].concat();
                let text = tokenizer.decode(&cumulative);
                if let Some(delta) = text.strip_prefix(&prev_text) {
                    if !delta.is_empty() {
                        let chunk = CompletionChunk {
                            id: id.clone(),
                            object: "text_completion",
                            created: 0,
                            model: model_name.clone(),
                            choices: vec![CompletionChunkChoice {
                                index: 0,
                                text: delta.to_string(),
                                finish_reason: None,
                            }],
                        };
                        let _ = tx
                            .send(Ok(
                                Event::default().data(serde_json::to_string(&chunk).unwrap())
                            ))
                            .await;
                    }
                    prev_text = text;
                }
            }

            let final_chunk = CompletionChunk {
                id: id.clone(),
                object: "text_completion",
                created: 0,
                model: model_name.clone(),
                choices: vec![CompletionChunkChoice {
                    index: 0,
                    text: String::new(),
                    finish_reason: Some("stop"),
                }],
            };
            let _ = tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&final_chunk).unwrap())
                ))
                .await;
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        return Sse::new(ReceiverStream::new(rx)).into_response();
    }

    // ------------------------------------------------------------------
    // Non-streaming path
    // ------------------------------------------------------------------
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_n = req.top_logprobs.unwrap_or(0).min(20) as usize;

    let (raw_output, logprobs_field): (String, Option<Logprobs>) = if want_logprobs {
        match run_text_inference_with_logprobs(&state, &req.prompt, &gen_cfg, top_n) {
            Some((ids, lps)) => {
                let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
                let text = tokenizer.decode(&ids);
                let content = build_logprobs_content(&tokenizer, &lps);
                (text, Some(Logprobs { content }))
            }
            None => {
                let text = run_text_inference_with_config(&state, &req.prompt, &gen_cfg);
                (
                    text,
                    Some(Logprobs {
                        content: Vec::new(),
                    }),
                )
            }
        }
    } else {
        let text = run_text_inference_with_config(&state, &req.prompt, &gen_cfg);
        (text, None)
    };

    let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
    let prompt_tokens = tokenizer.encode(&req.prompt).len();
    let completion_tokens = tokenizer.encode(&raw_output).len();

    Json(CompletionResponse {
        id,
        object: "text_completion",
        created: 0,
        model: model_name,
        choices: vec![CompletionChoice {
            index: 0,
            text: raw_output,
            finish_reason: "stop",
            logprobs: logprobs_field,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response()
}

/// Build a system prompt that describes available tools to the model.
fn build_tool_description(tools: &[Tool]) -> String {
    let mut lines = vec![
        "You have access to the following tools. Respond with a JSON object containing 'tool_calls', an array of objects with 'name' and 'arguments' (a JSON object).".to_string(),
    ];
    for tool in tools {
        let desc = if tool.function.description.is_empty() {
            format!("- {}: no description", tool.function.name)
        } else {
            format!("- {}: {}", tool.function.name, tool.function.description)
        };
        lines.push(desc);
    }
    lines.join("\n")
}

/// Parse generated text for tool calls when tools are active.
/// Expects JSON with a `tool_calls` array, each item having `name` and `arguments`.
fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let json_str = extract_json_object(text)?;
    let val: serde_json::Value = serde_json::from_str(&json_str).ok()?;
    let arr = val.get("tool_calls")?.as_array()?;
    let mut calls = Vec::new();
    for (idx, item) in arr.iter().enumerate() {
        let name = item.get("name")?.as_str()?;
        let args = item.get("arguments")?;
        let args_str = if args.is_string() {
            args.as_str()?.to_string()
        } else {
            serde_json::to_string(args).ok()?
        };
        calls.push(ToolCall {
            id: format!("call-{}", idx),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: args_str,
            },
        });
    }
    if calls.is_empty() { None } else { Some(calls) }
}

/// Extract the first well-formed JSON object or array from generated text.
/// Returns `None` if no valid JSON is found.
fn extract_json_object(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if (trimmed.starts_with('{') || trimmed.starts_with('['))
        && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    {
        return Some(trimmed.to_string());
    }
    // Try to find a JSON object or array substring.
    for (start, ch) in trimmed.char_indices() {
        if ch == '{' || ch == '[' {
            for (end, end_ch) in trimmed[start..].char_indices().rev() {
                let abs_end = start + end;
                if end_ch == '}' || end_ch == ']' {
                    let candidate = &trimmed[start..=abs_end];
                    if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                        return Some(candidate.to_string());
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_parses_object() {
        let s = "Here is the result: {\"answer\": 42} thanks!";
        assert_eq!(
            extract_json_object(s),
            Some(r#"{"answer": 42}"#.to_string())
        );
    }

    #[test]
    fn extract_json_parses_array() {
        let s = "Some text [1, 2, 3] more text";
        assert_eq!(extract_json_object(s), Some("[1, 2, 3]".to_string()));
    }

    #[test]
    fn extract_json_returns_none_for_no_json() {
        assert_eq!(extract_json_object("no json here"), None);
    }

    #[test]
    fn extract_json_uses_whole_string_when_valid() {
        let s = r#"{"key": "value"}"#;
        assert_eq!(
            extract_json_object(s),
            Some(r#"{"key": "value"}"#.to_string())
        );
    }

    #[test]
    fn build_tool_description_lists_tools() {
        let tools = vec![Tool {
            function: ToolFunction {
                name: "get_weather".to_string(),
                description: "Get current weather".to_string(),
            },
        }];
        let desc = build_tool_description(&tools);
        assert!(desc.contains("get_weather"));
        assert!(desc.contains("Get current weather"));
    }

    #[test]
    fn parse_tool_calls_extracts_name_and_args() {
        let text = r#"{"tool_calls": [{"name": "get_weather", "arguments": {"city": "NYC"}}]}"#;
        let calls = parse_tool_calls(text).expect("parses");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"NYC"}"#);
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn parse_tool_calls_returns_none_for_plain_json() {
        assert_eq!(parse_tool_calls(r#"{"answer": 42}"#), None);
    }

    #[test]
    fn parse_tool_calls_handles_arguments_as_string() {
        let text = r#"{"tool_calls": [{"name": "foo", "arguments": "{\"x\":1}"}]}"#;
        let calls = parse_tool_calls(text).expect("parses");
        assert_eq!(calls[0].function.arguments, "{\"x\":1}");
    }
}

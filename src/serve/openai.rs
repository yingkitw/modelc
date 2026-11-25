//! OpenAI-compatible API endpoints (`/v1/models`, `/v1/chat/completions`, `/v1/completions`).

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::{IntoResponse, Sse};
use axum::{Json, extract::State, response::sse::Event};
use tokio_stream::wrappers::ReceiverStream;

use super::AppState;
use super::CancelOnDrop;
mod types;

use types::*;

use super::infer::{
    run_embeddings, run_text_inference_with_config, run_text_inference_with_logprobs,
};

/// Current Unix timestamp in seconds. OpenAI's `created` field on every
/// response object is "The Unix timestamp (in seconds) of when the completion
/// was created" — client libraries and logging pipelines rely on it being a
/// plausible timestamp, not the epoch placeholder.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Stable `system_fingerprint` suffix derived from the crate version + git SHA.
/// OpenAI includes this field on chat/completion responses; some clients log it
/// or use it to detect backend changes.
fn system_fingerprint() -> String {
    format!("fp_modelc_{}", crate::GIT_SHA)
}

// ---------------------------------------------------------------------------
// /v1/models
// ---------------------------------------------------------------------------

pub(super) async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsList> {
    Json(ModelsList {
        object: "list",
        data: vec![ModelEntry {
            id: state.name.clone(),
            object: "model",
            created: unix_now(),
            owned_by: "modelc",
        }],
    })
}

pub(super) async fn retrieve_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    if id != state.name {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("The model '{}' does not exist", id),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response();
    }
    Json(ModelEntry {
        id: state.name.clone(),
        object: "model",
        created: unix_now(),
        owned_by: "modelc",
    })
    .into_response()
}

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

/// Request-derived flags for `/v1/chat/completions`: which structured-output
/// modes and tool constraints apply.
struct ChatFlags {
    json_mode: bool,
    json_schema_mode: bool,
    response_schema: Option<serde_json::Value>,
    has_tools: bool,
    tool_choice_required: bool,
    tool_choice_specific: Option<String>,
}

impl ChatFlags {
    fn from_request(req: &ChatCompletionRequest) -> Self {
        let json_mode = req
            .response_format
            .as_ref()
            .is_some_and(|rf| rf.format_type == "json_object");
        let json_schema_mode = req
            .response_format
            .as_ref()
            .is_some_and(|rf| rf.format_type == "json_schema");
        let response_schema = req
            .response_format
            .as_ref()
            .and_then(|rf| rf.json_schema.as_ref())
            .and_then(|js| js.schema.clone());
        let tool_choice_none = req
            .tool_choice
            .as_ref()
            .and_then(|v| v.as_str())
            .is_some_and(|s| s == "none");
        let tool_choice_required = req
            .tool_choice
            .as_ref()
            .and_then(|v| v.as_str())
            .is_some_and(|s| s == "required");
        let tool_choice_specific = req
            .tool_choice
            .as_ref()
            .and_then(|v| v.get("function"))
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .map(|s| s.to_string());
        ChatFlags {
            json_mode,
            json_schema_mode,
            response_schema,
            has_tools: !req.tools.is_empty() && !tool_choice_none,
            tool_choice_required,
            tool_choice_specific,
        }
    }
}

/// Messages + injected system prompts (JSON mode, JSON schema, tool
/// descriptions) ready for the chat template.
fn templated_chat_messages(
    req: &ChatCompletionRequest,
    flags: &ChatFlags,
) -> Vec<crate::chat_template::ChatMessage> {
    let mut messages: Vec<crate::chat_template::ChatMessage> = req
        .messages
        .iter()
        .map(|m| crate::chat_template::ChatMessage {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
        })
        .collect();

    if flags.json_mode {
        messages.push(crate::chat_template::ChatMessage {
            role: "system".to_string(),
            content: "Respond with valid JSON only. Do not include any explanatory text before or after the JSON.".to_string(),
        });
    }

    if flags.json_schema_mode {
        let schema_str = flags
            .response_schema
            .as_ref()
            .map(|s| serde_json::to_string_pretty(s).unwrap_or_default())
            .unwrap_or_default();
        messages.push(crate::chat_template::ChatMessage {
            role: "system".to_string(),
            content: format!(
                "Respond with valid JSON that conforms to the following JSON Schema. Do not include any explanatory text before or after the JSON.\n\n{}",
                schema_str
            ),
        });
    }

    if flags.has_tools {
        let mut content = build_tool_description(&req.tools);
        if flags.tool_choice_required {
            content.push_str(
                "\nYou must call one of the available tools. Do not respond with text only.",
            );
        }
        if let Some(ref name) = flags.tool_choice_specific {
            content.push_str(&format!(
                "\nYou must call the tool named \"{}\". Do not call any other tool.",
                name
            ));
        }
        messages.push(crate::chat_template::ChatMessage {
            role: "system".to_string(),
            content,
        });
    }

    messages
}

pub(super) async fn chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let flags = ChatFlags::from_request(&req);
    let stream = req.stream;

    let messages = templated_chat_messages(&req, &flags);
    let prompt =
        crate::chat_template::apply_chat_template(state.chat_template.as_deref(), &messages);

    let gen_cfg = req.params.to_generation_config(&state.generation);

    let model_name = if req.model.is_empty() {
        state.name.clone()
    } else {
        req.model
    };
    let id = format!("chatcmpl-{}-0", state.name);

    // ------------------------------------------------------------------
    // Streaming path
    // ------------------------------------------------------------------
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|so| so.include_usage);
    if stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(4);
        let state_clone = Arc::clone(&state);
        // Cancellation flag: set when the SSE client disconnects (stream dropped).
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut gen_cfg = gen_cfg;
        gen_cfg.cancel = Some(cancel.clone());
        tokio::spawn(async move {
            let token_ids =
                super::infer::run_text_inference_token_ids(&state_clone, &prompt, &gen_cfg);
            let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
            let prompt_ids = tokenizer.encode(&prompt);

            for (i, delta) in tokenizer
                .decode_deltas(&prompt_ids, &token_ids)
                .into_iter()
                .enumerate()
            {
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created: unix_now(),
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: if i == 0 {
                                Some("assistant".to_string())
                            } else {
                                None
                            },
                            content: Some(delta),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                    system_fingerprint: Some(system_fingerprint()),
                };
                // If the client is gone, stop dripping (generation has already
                // been cancelled via the flag).
                if tx
                    .send(Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap())
                    ))
                    .await
                    .is_err()
                {
                    break;
                }
            }

            // Final chunk with finish_reason.
            let final_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created: unix_now(),
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta::default(),
                    finish_reason: Some("stop"),
                }],
                usage: None,
                system_fingerprint: Some(system_fingerprint()),
            };
            let _ = tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&final_chunk).unwrap())
                ))
                .await;

            // When stream_options.include_usage is true, emit a separate
            // chunk with empty choices and the usage object (OpenAI protocol).
            if include_usage {
                let usage_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created: unix_now(),
                    model: model_name.clone(),
                    choices: vec![],
                    usage: Some(Usage {
                        prompt_tokens: prompt_ids.len(),
                        completion_tokens: token_ids.len(),
                        total_tokens: prompt_ids.len() + token_ids.len(),
                    }),
                    system_fingerprint: Some(system_fingerprint()),
                };
                let _ = tx
                    .send(Ok(
                        Event::default().data(serde_json::to_string(&usage_chunk).unwrap())
                    ))
                    .await;
            }

            // OpenAI streams end with [DONE].
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        return Sse::new(CancelOnDrop::new(ReceiverStream::new(rx), cancel)).into_response();
    }

    // ------------------------------------------------------------------
    // Non-streaming path
    // ------------------------------------------------------------------
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_n = req.top_logprobs.unwrap_or(0).min(20) as usize;
    let n = req.n.unwrap_or(1).clamp(1, 20) as usize;

    let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
    let prompt_tokens = tokenizer.encode(&prompt).len();
    let mut choices: Vec<Choice> = Vec::with_capacity(n);
    let mut total_completion_tokens = 0usize;

    for i in 0..n {
        let (raw_output, logprobs_field): (String, Option<Logprobs>) = if want_logprobs {
            match run_text_inference_with_logprobs(&state, &prompt, &gen_cfg, top_n) {
                Some((ids, lps)) => {
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
            } else if flags.json_schema_mode {
                if let Some(ref schema) = flags.response_schema {
                    crate::json_schema::generate_with_schema(
                        |cfg| run_text_inference_with_config(&state, &prompt, cfg),
                        schema,
                        &gen_cfg,
                        3,
                    )
                } else {
                    run_text_inference_with_config(&state, &prompt, &gen_cfg)
                }
            } else {
                run_text_inference_with_config(&state, &prompt, &gen_cfg)
            };
            (text, None)
        };

        let (message, finish_reason) = if flags.has_tools {
            if let Some(tool_calls) = parse_tool_calls(&raw_output) {
                (
                    ChatMessage {
                        role: "assistant".to_string(),
                        content: None,
                        name: None,
                        tool_calls: Some(tool_calls),
                    },
                    "tool_calls",
                )
            } else {
                let content = extract_content(&raw_output, flags.json_mode);
                (
                    ChatMessage {
                        role: "assistant".to_string(),
                        content: Some(content),
                        name: None,
                        tool_calls: None,
                    },
                    "stop",
                )
            }
        } else {
            let content = extract_content(&raw_output, flags.json_mode);
            (
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(content),
                    name: None,
                    tool_calls: None,
                },
                "stop",
            )
        };

        total_completion_tokens += tokenizer.encode(&raw_output).len();

        choices.push(Choice {
            index: i,
            message,
            finish_reason,
            logprobs: logprobs_field,
        });
    }

    Json(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: unix_now(),
        model: model_name,
        choices,
        usage: Usage {
            prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: prompt_tokens + total_completion_tokens,
        },
        system_fingerprint: system_fingerprint(),
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

pub(super) async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let stream = req.stream;
    let gen_cfg = req.params.to_generation_config(&state.generation);

    let model_name = if req.model.is_empty() {
        state.name.clone()
    } else {
        req.model
    };
    let id = format!("cmpl-{}-0", state.name);

    let prompts = req.prompt.into_vec();
    let prompt = prompts[0].clone();

    // ------------------------------------------------------------------
    // Streaming path
    // ------------------------------------------------------------------
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|so| so.include_usage);
    if stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(4);
        let state_clone = Arc::clone(&state);
        let prompt_clone = prompt.clone();
        // Cancellation flag: set when the SSE client disconnects (stream dropped).
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut gen_cfg = gen_cfg;
        gen_cfg.cancel = Some(cancel.clone());
        let echo_prompt = req.echo.unwrap_or(false);
        tokio::spawn(async move {
            // When echo is true, emit the prompt as the first chunk.
            if echo_prompt && !prompt_clone.is_empty() {
                let chunk = CompletionChunk {
                    id: id.clone(),
                    object: "text_completion",
                    created: unix_now(),
                    model: model_name.clone(),
                    choices: vec![CompletionChunkChoice {
                        index: 0,
                        text: prompt_clone.clone(),
                        finish_reason: None,
                    }],
                    usage: None,
                    system_fingerprint: Some(system_fingerprint()),
                };
                if tx
                    .send(Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap())
                    ))
                    .await
                    .is_err()
                {
                    return;
                }
            }

            let token_ids =
                super::infer::run_text_inference_token_ids(&state_clone, &prompt_clone, &gen_cfg);
            let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
            let prompt_ids = tokenizer.encode(&prompt_clone);
            for delta in tokenizer.decode_deltas(&prompt_ids, &token_ids) {
                let chunk = CompletionChunk {
                    id: id.clone(),
                    object: "text_completion",
                    created: unix_now(),
                    model: model_name.clone(),
                    choices: vec![CompletionChunkChoice {
                        index: 0,
                        text: delta,
                        finish_reason: None,
                    }],
                    usage: None,
                    system_fingerprint: Some(system_fingerprint()),
                };
                if tx
                    .send(Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap())
                    ))
                    .await
                    .is_err()
                {
                    break;
                }
            }

            let final_chunk = CompletionChunk {
                id: id.clone(),
                object: "text_completion",
                created: unix_now(),
                model: model_name.clone(),
                choices: vec![CompletionChunkChoice {
                    index: 0,
                    text: String::new(),
                    finish_reason: Some("stop"),
                }],
                usage: None,
                system_fingerprint: Some(system_fingerprint()),
            };
            let _ = tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&final_chunk).unwrap())
                ))
                .await;

            // When stream_options.include_usage is true, emit a separate
            // chunk with empty choices and the usage object (OpenAI protocol).
            if include_usage {
                let usage_chunk = CompletionChunk {
                    id: id.clone(),
                    object: "text_completion",
                    created: unix_now(),
                    model: model_name.clone(),
                    choices: vec![],
                    usage: Some(Usage {
                        prompt_tokens: prompt_ids.len(),
                        completion_tokens: token_ids.len(),
                        total_tokens: prompt_ids.len() + token_ids.len(),
                    }),
                    system_fingerprint: Some(system_fingerprint()),
                };
                let _ = tx
                    .send(Ok(
                        Event::default().data(serde_json::to_string(&usage_chunk).unwrap())
                    ))
                    .await;
            }

            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        return Sse::new(CancelOnDrop::new(ReceiverStream::new(rx), cancel)).into_response();
    }

    // ------------------------------------------------------------------
    // Non-streaming path
    // ------------------------------------------------------------------
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_n = req.top_logprobs.unwrap_or(0).min(20) as usize;
    let n = req.n.unwrap_or(1).clamp(1, 20) as usize;
    let best_of = req.best_of.unwrap_or(n as u32).max(n as u32).min(20) as usize;
    let echo_prompt = req.echo.unwrap_or(false);
    let suffix = req.suffix.as_deref().unwrap_or("");

    let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
    let mut total_prompt_tokens = 0usize;

    let mut choices: Vec<CompletionChoice> = Vec::with_capacity(n * prompts.len());
    let mut total_completion_tokens = 0usize;
    let mut choice_index = 0usize;

    for p in &prompts {
        total_prompt_tokens += tokenizer.encode(p).len();

        // Generate best_of completions, then pick the top n by token count.
        let mut all: Vec<(String, Option<Logprobs>, usize)> = Vec::with_capacity(best_of);
        for _ in 0..best_of {
            let (raw_output, logprobs_field): (String, Option<Logprobs>) = if want_logprobs {
                match run_text_inference_with_logprobs(&state, p, &gen_cfg, top_n) {
                    Some((ids, lps)) => {
                        let text = tokenizer.decode(&ids);
                        let content = build_logprobs_content(&tokenizer, &lps);
                        (text, Some(Logprobs { content }))
                    }
                    None => {
                        let text = run_text_inference_with_config(&state, p, &gen_cfg);
                        (
                            text,
                            Some(Logprobs {
                                content: Vec::new(),
                            }),
                        )
                    }
                }
            } else {
                let text = run_text_inference_with_config(&state, p, &gen_cfg);
                (text, None)
            };
            let tok_count = tokenizer.encode(&raw_output).len();
            all.push((raw_output, logprobs_field, tok_count));
        }

        // Sort by token count descending (proxy for highest logprob) and take top n.
        all.sort_by_key(|b| std::cmp::Reverse(b.2));
        all.truncate(n);

        for (raw_output, logprobs_field, _) in all.into_iter() {
            let display_text = if echo_prompt {
                format!("{}{}{}", p, raw_output, suffix)
            } else {
                format!("{}{}", raw_output, suffix)
            };
            total_completion_tokens += tokenizer.encode(&display_text).len();

            choices.push(CompletionChoice {
                index: choice_index,
                text: display_text,
                finish_reason: "stop",
                logprobs: logprobs_field,
            });
            choice_index += 1;
        }
    }

    Json(CompletionResponse {
        id,
        object: "text_completion",
        created: unix_now(),
        model: model_name,
        choices,
        usage: Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
        },
        system_fingerprint: system_fingerprint(),
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

// ---------------------------------------------------------------------------
// /v1/embeddings
// ---------------------------------------------------------------------------

pub(super) async fn v1_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<V1EmbeddingsRequest>,
) -> Json<V1EmbeddingsResponse> {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);

    let inputs = extract_embedding_inputs(&req.input);
    let use_base64 = req.encoding_format == "base64";

    let mut entries = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0usize;

    for (idx, text) in inputs.iter().enumerate() {
        let input_f32: Vec<f32> = text.bytes().map(|b| b as f32 / 255.0).collect();
        total_tokens += text.len();
        let embedding = run_embeddings(&state, &input_f32).unwrap_or_default();
        let dimensions = embedding.len();

        let embedding_value = if use_base64 {
            let bytes: Vec<u8> = embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
            serde_json::Value::String(base64_encode(&bytes))
        } else {
            serde_json::to_value(embedding).unwrap_or(serde_json::Value::Null)
        };

        entries.push(V1Embedding {
            object: "embedding",
            embedding: embedding_value,
            index: idx,
            dimensions,
        });
    }

    Json(V1EmbeddingsResponse {
        object: "list",
        data: entries,
        model: if req.model.is_empty() {
            state.name.clone()
        } else {
            req.model
        },
        usage: Usage {
            prompt_tokens: total_tokens,
            completion_tokens: 0,
            total_tokens,
        },
    })
}

/// Flatten the `input` field (string or array of strings) into a Vec of texts.
fn extract_embedding_inputs(input: &serde_json::Value) -> Vec<String> {
    if let Some(arr) = input.as_array() {
        arr.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect()
    } else if let Some(s) = input.as_str() {
        vec![s.to_string()]
    } else {
        Vec::new()
    }
}

/// Orders `(index, cosine)` pairs by descending raw similarity (stable, so ties
/// keep input order), truncates to `top_n`, then clamps the reported score into
/// Cohere's documented `[0, 1]` range. Clamping happens *after* ordering, so
/// documents whose similarity is ≤ 0 stay ordered by raw similarity even
/// though they all report `0.0`.
fn rank_results(mut scored: Vec<(usize, f32)>, top_n: Option<usize>) -> Vec<(usize, f32)> {
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(top_n) = top_n {
        scored.truncate(top_n);
    }
    scored
        .into_iter()
        .map(|(index, cos)| (index, cos.clamp(0.0, 1.0)))
        .collect()
}

/// Rank `documents` against `query` with the model's embedding path: cosine
/// similarity, ordered by descending similarity, truncated to `top_n`, and
/// clamped into the `[0, 1]` range Cohere's contract documents.
///
/// Shared by the HTTP `/v1/rerank` endpoint and the MCP `rerank` tool so both
/// rank identically.
pub(super) fn rerank_documents(
    state: &AppState,
    query: &str,
    documents: &[String],
    top_n: Option<usize>,
) -> Vec<(usize, f32)> {
    let query_input: Vec<f32> = query.bytes().map(|b| b as f32 / 255.0).collect();
    let query_emb = run_embeddings(state, &query_input).unwrap_or_default();

    let mut scored = Vec::with_capacity(documents.len());
    for (idx, doc) in documents.iter().enumerate() {
        let doc_input: Vec<f32> = doc.bytes().map(|b| b as f32 / 255.0).collect();
        let doc_emb = run_embeddings(state, &doc_input).unwrap_or_default();
        scored.push((
            idx,
            super::handlers::cosine_similarity(&query_emb, &doc_emb),
        ));
    }
    rank_results(scored, top_n)
}

/// `POST /v1/rerank` + `/v2/rerank` — Cohere-compatible reranking over the same
/// embedding + cosine machinery as the native `/reranking` endpoint. Both API
/// versions' paths share one envelope: results are ordered by descending raw
/// similarity, truncated to `top_n`, scored in Cohere's documented `[0, 1]`
/// range, and carry the document only when `return_documents` is true.
pub(super) async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CohereRerankRequest>,
) -> axum::response::Response {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);

    // Object documents rank on their `text` field; one without a string
    // `text` is a missing-required-fields request.
    let mut texts = Vec::with_capacity(req.documents.len());
    for (idx, doc) in req.documents.iter().enumerate() {
        let Some(text) = doc.text() else {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "message": format!(
                        "documents[{idx}] is an object without a string `text` field"
                    )
                })),
            )
                .into_response();
        };
        texts.push(text.to_string());
    }

    let results: Vec<CohereRerankResult> =
        rerank_documents(&state, &req.query, &texts, req.top_n)
            .into_iter()
            .map(|(index, relevance_score)| CohereRerankResult {
                index,
                relevance_score,
                document: req
                    .return_documents
                    .then(|| req.documents[index].echoed()),
            })
            .collect();

    Json(CohereRerankResponse {
        id: format!("rerank-{}", state.name),
        results,
    })
    .into_response()
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

    #[test]
    fn base64_encode_matches_known_vectors() {
        // RFC 4648 §10 test vectors.
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_encode_handles_arbitrary_bytes() {
        // 3-byte chunks should not produce padding.
        let bytes: Vec<u8> = (0u8..=255).collect();
        let encoded = base64_encode(&bytes);
        assert_eq!(encoded.len() % 4, 0);
        // Round-trip via the inverse mapping should yield the original.
        let mut decoded = Vec::with_capacity(bytes.len());
        let chars: Vec<u8> = encoded.bytes().collect();
        for chunk in chars.chunks(4) {
            let mut vals = [0u32; 4];
            for (i, &c) in chunk.iter().enumerate() {
                vals[i] = match c {
                    b'A'..=b'Z' => (c - b'A') as u32,
                    b'a'..=b'z' => (c - b'a' + 26) as u32,
                    b'0'..=b'9' => (c - b'0' + 52) as u32,
                    b'+' => 62,
                    b'/' => 63,
                    b'=' => 0,
                    _ => panic!("unexpected base64 char: {}", c as char),
                };
            }
            let n = (vals[0] << 18) | (vals[1] << 12) | (vals[2] << 6) | vals[3];
            decoded.push((n >> 16) as u8);
            if chunk[2] != b'=' {
                decoded.push((n >> 8) as u8);
            }
            if chunk[3] != b'=' {
                decoded.push(n as u8);
            }
        }
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn rank_results_orders_by_raw_similarity_then_clamps() {
        // Ordering uses the raw cosine (1.4 outranks 0.9) while the reported
        // score lands in Cohere's documented [0, 1] range.
        let ranked = rank_results(vec![(0, -0.5), (1, 0.9), (2, 1.4)], None);
        assert_eq!(ranked, vec![(2, 1.0), (1, 0.9), (0, 0.0)]);
    }

    #[test]
    fn rank_results_truncates_to_top_n() {
        let scored = vec![(0, 0.2), (1, 0.8), (2, 0.5)];
        assert_eq!(rank_results(scored.clone(), Some(1)), vec![(1, 0.8)]);
        assert_eq!(
            rank_results(scored.clone(), Some(10)),
            vec![(1, 0.8), (2, 0.5), (0, 0.2)]
        );
        assert_eq!(rank_results(scored, Some(0)), Vec::new());
    }

    #[test]
    fn rank_results_ties_keep_input_order() {
        // Stable sort: equal scores stay in original-index order.
        let ranked = rank_results(vec![(0, 0.5), (1, 0.5), (2, 0.5)], None);
        let indices: Vec<usize> = ranked.into_iter().map(|(i, _)| i).collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn rerank_document_accepts_strings_and_objects() {
        let text: RerankDocument = serde_json::from_str(r#""washington""#).unwrap();
        assert_eq!(text.text(), Some("washington"));
        assert_eq!(text.echoed(), serde_json::json!({ "text": "washington" }));

        let object: RerankDocument =
            serde_json::from_str(r#"{"text": "paris", "title": "FR"}"#).unwrap();
        assert_eq!(object.text(), Some("paris"));
        // Cohere v1 preserves extra object fields in the response.
        assert_eq!(
            object.echoed(),
            serde_json::json!({ "text": "paris", "title": "FR" })
        );
    }

    #[test]
    fn rerank_document_without_string_text_is_not_rankable() {
        let no_text: RerankDocument = serde_json::from_str(r#"{"title": "FR"}"#).unwrap();
        assert_eq!(no_text.text(), None);
        let non_string: RerankDocument = serde_json::from_str(r#"{"text": 42}"#).unwrap();
        assert_eq!(non_string.text(), None);
    }
}

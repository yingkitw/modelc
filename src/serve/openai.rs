//! OpenAI-compatible API endpoints (`/v1/models`, `/v1/chat/completions`).

use std::sync::Arc;

use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

use super::AppState;
use super::infer::run_text_inference;

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
    #[allow(dead_code)]
    stream: bool,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    #[serde(default)]
    tools: Vec<Tool>,
}

#[derive(Deserialize)]
pub(super) struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Deserialize, Clone)]
pub(super) struct Tool {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    tool_type: String,
    function: ToolFunction,
}

#[derive(Deserialize, Clone)]
pub(super) struct ToolFunction {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    #[allow(dead_code)]
    parameters: serde_json::Value,
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
}

#[derive(Serialize)]
pub(super) struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

pub(super) async fn chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    let is_json_mode = req
        .response_format
        .as_ref()
        .is_some_and(|rf| rf.format_type == "json_object");
    let has_tools = !req.tools.is_empty();

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

    let prompt = crate::chat_template::apply_chat_template(state.chat_template.as_deref(), &messages)
        .unwrap_or_else(|_| {
            req.messages
                .last()
                .and_then(|m| m.content.clone())
                .unwrap_or_default()
        });

    let raw_output = run_text_inference(&state, &prompt);

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

    let prompt_tokens = prompt.split_whitespace().count();
    let completion_tokens = raw_output.split_whitespace().count();

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}-0", state.name),
        object: "chat.completion",
        created: 0,
        model: if req.model.is_empty() {
            state.name.clone()
        } else {
            req.model
        },
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

fn extract_content(raw: &str, is_json_mode: bool) -> String {
    if is_json_mode {
        extract_json_object(raw).unwrap_or_else(|| raw.to_string())
    } else {
        raw.to_string()
    }
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
    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
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
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "get_weather".to_string(),
                description: "Get current weather".to_string(),
                parameters: serde_json::Value::Null,
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

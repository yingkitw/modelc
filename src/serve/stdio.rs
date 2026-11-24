//! `modelc serve --stdio` — MCP-compatible JSON-RPC 2.0 server on stdin/stdout.
//!
//! Newline-delimited JSON-RPC 2.0, the same wire format the Model Context
//! Protocol (MCP) uses over stdio, so the artifact can be attached directly to
//! editors and agent runtimes. stdout carries **only** protocol messages — all
//! logs go to stderr. Synchronous by design: a blocking line loop over stdin
//! with the same `AppState` inference plumbing the HTTP endpoints use.
//!
//! Methods: `initialize`, `ping`, `tools/list`, `tools/call`
//! (`chat` / `complete` / `embeddings` tools), `notifications/initialized`
//! (ignored). Unknown methods return JSON-RPC error `-32601`; malformed lines
//! return `-32700`; tool failures surface as MCP error *results*
//! (`isError: true`), not protocol errors.

use std::io::{BufRead, Write};

use serde_json::{json, Value};

use super::infer::{run_embeddings, run_text_inference_with_config};
use super::types::{ChatRequest, CompleteRequest, EmbeddingEntry, EmbeddingsRequest, EmbeddingsResponse};

/// Highest MCP protocol version this server speaks.
const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
/// Protocol versions accepted for `initialize` (echoed back verbatim).
const SUPPORTED_PROTOCOL_VERSIONS: [&str; 3] = ["2024-11-05", "2025-03-26", "2025-06-18"];

/// Serve `model` over stdin/stdout as a newline-delimited JSON-RPC 2.0 server.
/// Returns when stdin closes (EOF). `--api-key` / `--rate-limit` are HTTP-mode
/// only and ignored here.
pub fn run_stdio_server(
    model: crate::model::Model,
    profile: bool,
    generation: crate::generate::GenerationConfig,
    max_concurrent: Option<usize>,
) -> anyhow::Result<()> {
    // The production bootstrap is the same one `build_router` uses, so stdio
    // serves exactly what the HTTP router would serve (MLP plan, ONNX plan,
    // draft model, prefix cache included).
    let state = super::AppState::for_test(model, profile, generation, None, max_concurrent);
    eprintln!(
        "modelc serve: MCP stdio server ready — model '{}' | architecture: {} | {} tensor(s)",
        state.name,
        state.architecture,
        state.tensor_names.len()
    );
    eprintln!("modelc serve: newline-delimited JSON-RPC 2.0 on stdin/stdout (MCP-compatible)");

    let stdin = std::io::stdin();
    let locked = stdin.lock();
    let mut out = std::io::stdout().lock();
    for line in locked.lines() {
        let line = line?;
        if let Some(response) = handle_message(&state, &line) {
            writeln!(out, "{}", response)?;
            out.flush()?;
        }
    }
    eprintln!("modelc serve: stdin closed, exiting");
    Ok(())
}

/// Handle one protocol line. `None` means "no response" — notifications and
/// non-protocol lines are silent by design.
pub(super) fn handle_message(state: &super::AppState, line: &str) -> Option<Value> {
    if line.trim().is_empty() {
        return None;
    }
    let msg: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => {
            return Some(rpc_error(&Value::Null, -32700, "parse error: invalid JSON"));
        }
    };
    let id = msg.get("id").cloned().unwrap_or(Value::Null);
    let is_notification = msg.get("id").is_none();
    let Some(method) = msg.get("method").and_then(|m| m.as_str()) else {
        // Responses arriving without a method are treated as notifications.
        return if is_notification {
            None
        } else {
            Some(rpc_error(&id, -32600, "invalid request: missing method"))
        };
    };

    match method {
        "initialize" => Some(rpc_result(&id, initialize_result(&msg))),
        "notifications/initialized" => None,
        "ping" => Some(rpc_result(&id, json!({}))),
        "tools/list" => Some(rpc_result(&id, json!({ "tools": tools_list() }))),
        "tools/call" => {
            let name = msg
                .pointer("/params/name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let arguments = msg.pointer("/params/arguments").cloned();
            Some(rpc_result(&id, call_tool(state, name, arguments)))
        }
        other => {
            if is_notification {
                return None;
            }
            Some(rpc_error(
                &id,
                -32601,
                &format!("method not found: {other}"),
            ))
        }
    }
}

fn rpc_result(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_error(id: &Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn tool_result(text: String, is_error: bool) -> Value {
    json!({
        "content": [ { "type": "text", "text": text } ],
        "isError": is_error
    })
}

fn initialize_result(msg: &Value) -> Value {
    let requested = msg
        .pointer("/params/protocolVersion")
        .and_then(|v| v.as_str())
        .unwrap_or(MCP_PROTOCOL_VERSION);
    let version = if SUPPORTED_PROTOCOL_VERSIONS.contains(&requested) {
        requested
    } else {
        MCP_PROTOCOL_VERSION
    };
    json!({
        "protocolVersion": version,
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": { "name": "modelc", "version": crate::CLI_VERSION }
    })
}

/// The three inference tools, with JSON Schema `inputSchema`s mirroring the
/// native `/chat`, `/complete`, and `/embeddings` request shapes.
fn tools_list() -> Value {
    fn sampling_props() -> Value {
        json!({
            "max_tokens": { "type": "integer" },
            "temperature": { "type": "number" },
            "top_p": { "type": "number" },
            "min_p": { "type": "number" },
            "seed": { "type": "integer" },
            "stop": { "type": "array", "items": { "type": "string" } },
            "grammar": { "type": "string" }
        })
    }
    let mut chat = json!({
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": { "role": { "type": "string" }, "content": { "type": "string" } },
                "required": ["role", "content"]
            }
        },
        "json_schema": { "type": "object" }
    });
    merge_object(&mut chat, sampling_props());
    let mut complete = json!({ "prompt": { "type": "string" } });
    merge_object(&mut complete, sampling_props());
    json!([
        {
            "name": "chat",
            "description": "Generate an assistant reply for a list of chat messages",
            "inputSchema": { "type": "object", "properties": chat, "required": ["messages"] }
        },
        {
            "name": "complete",
            "description": "Complete a raw prompt string",
            "inputSchema": { "type": "object", "properties": complete, "required": ["prompt"] }
        },
        {
            "name": "embeddings",
            "description": "Embed one string (`input`) or a batch (`inputs`)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string" },
                    "inputs": { "type": "array", "items": { "type": "string" } }
                }
            }
        }
    ])
}

/// Merge `add`'s top-level keys into `base` (both JSON objects).
fn merge_object(base: &mut Value, add: Value) {
    if let (Some(b), Some(a)) = (base.as_object_mut(), add.as_object()) {
        for (k, v) in a {
            b.insert(k.clone(), v.clone());
        }
    }
}

fn call_tool(state: &super::AppState, name: &str, arguments: Option<Value>) -> Value {
    let arguments = arguments.unwrap_or_else(|| json!({}));
    match name {
        "chat" => tool_chat(state, &arguments),
        "complete" => tool_complete(state, &arguments),
        "embeddings" => tool_embeddings(state, &arguments),
        other => tool_result(format!("Unknown tool: {other}"), true),
    }
}

fn tool_chat(state: &super::AppState, arguments: &Value) -> Value {
    let req: ChatRequest = match serde_json::from_value(arguments.clone()) {
        Ok(r) => r,
        Err(e) => return tool_result(format!("invalid chat arguments: {e}"), true),
    };
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
    let gen_cfg = super::handlers::make_generation_config(
        &state.generation,
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.min_p,
        req.grammar.clone(),
        req.stop.clone(),
        req.seed,
        req.repetition_penalty,
        req.presence_penalty,
        req.frequency_penalty,
        req.logit_bias.clone(),
    );
    let output = if let Some(ref schema) = req.json_schema {
        crate::json_schema::generate_with_schema(
            |cfg| run_text_inference_with_config(state, &prompt, cfg),
            schema,
            &gen_cfg,
            3,
        )
    } else {
        run_text_inference_with_config(state, &prompt, &gen_cfg)
    };
    tool_result(output, false)
}

fn tool_complete(state: &super::AppState, arguments: &Value) -> Value {
    let req: CompleteRequest = match serde_json::from_value(arguments.clone()) {
        Ok(r) => r,
        Err(e) => return tool_result(format!("invalid complete arguments: {e}"), true),
    };
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let gen_cfg = super::handlers::make_generation_config(
        &state.generation,
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.min_p,
        req.grammar.clone(),
        req.stop.clone(),
        req.seed,
        req.repetition_penalty,
        req.presence_penalty,
        req.frequency_penalty,
        req.logit_bias.clone(),
    );
    let output = if let Some(ref schema) = req.json_schema {
        crate::json_schema::generate_with_schema(
            |cfg| run_text_inference_with_config(state, &req.prompt, cfg),
            schema,
            &gen_cfg,
            3,
        )
    } else {
        run_text_inference_with_config(state, &req.prompt, &gen_cfg)
    };
    tool_result(output, false)
}

fn tool_embeddings(state: &super::AppState, arguments: &Value) -> Value {
    let req: EmbeddingsRequest = match serde_json::from_value(arguments.clone()) {
        Ok(r) => r,
        Err(e) => return tool_result(format!("invalid embeddings arguments: {e}"), true),
    };
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    if !req.inputs.is_empty() {
        let entries: Vec<EmbeddingEntry> = req
            .inputs
            .iter()
            .enumerate()
            .map(|(idx, text)| {
                let input: Vec<f32> = text.bytes().map(|b| b as f32 / 255.0).collect();
                EmbeddingEntry {
                    embedding: run_embeddings(state, &input).unwrap_or_default(),
                    index: idx,
                }
            })
            .collect();
        let response = EmbeddingsResponse {
            embedding: None,
            embeddings: Some(entries),
            model: state.name.clone(),
        };
        return tool_result(
            serde_json::to_string(&response).unwrap_or_else(|_| "[]".to_string()),
            false,
        );
    }
    let input: Vec<f32> = req.input.bytes().map(|b| b as f32 / 255.0).collect();
    let embedding = run_embeddings(state, &input).unwrap_or_default();
    let response = EmbeddingsResponse {
        embedding: Some(embedding),
        embeddings: None,
        model: state.name.clone(),
    };
    tool_result(
        serde_json::to_string(&response).unwrap_or_else(|_| "[]".to_string()),
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DataType, Model, TensorData};
    use std::collections::HashMap;

    /// 2-layer MLP fixture with the same weights as `tests/pybridge_test.rs`:
    /// input [1, 2] → hidden [2.1, 3.4] → output 1.3.
    fn serve_fixture() -> Model {
        let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut tensors = HashMap::new();
        tensors.insert(
            "layer0.weight".into(),
            TensorData {
                shape: vec![2, 2],
                dtype: DataType::F32,
                data: bytes(&[1.0, 0.5, -0.5, 2.0]),
            },
        );
        tensors.insert(
            "layer0.bias".into(),
            TensorData {
                shape: vec![2],
                dtype: DataType::F32,
                data: bytes(&[0.1, -0.1]),
            },
        );
        tensors.insert(
            "layer1.weight".into(),
            TensorData {
                shape: vec![1, 2],
                dtype: DataType::F32,
                data: bytes(&[2.0, -1.0]),
            },
        );
        tensors.insert(
            "layer1.bias".into(),
            TensorData {
                shape: vec![1],
                dtype: DataType::F32,
                data: bytes(&[0.5]),
            },
        );
        Model {
            name: "serve_fixture".into(),
            architecture: "mlp".into(),
            tensors,
            metadata: HashMap::new(),
        }
    }

    fn state() -> std::sync::Arc<super::super::AppState> {
        super::super::AppState::for_test(
            serve_fixture(),
            false,
            crate::generate::GenerationConfig::default(),
            None,
            None,
        )
    }

    fn call(state: &super::super::AppState, line: &str) -> Value {
        handle_message(state, line).expect("expected a response")
    }

    #[test]
    fn initialize_echoes_protocol_version_and_server_info() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}"#,
        );
        assert_eq!(resp["id"], 1);
        assert_eq!(
            resp["result"]["protocolVersion"],
            "2025-03-26",
            "supported requested version is echoed"
        );
        assert_eq!(resp["result"]["serverInfo"]["name"], "modelc");
        assert_eq!(resp["result"]["capabilities"]["tools"]["listChanged"], false);
    }

    #[test]
    fn initialize_unknown_version_falls_back_to_latest() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1999-01-01"}}"#,
        );
        assert_eq!(resp["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);
    }

    #[test]
    fn notifications_are_silent() {
        assert!(handle_message(
            &state(),
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#
        )
        .is_none());
        assert!(handle_message(&state(), r#"{"jsonrpc":"2.0","method":"some/other/notification"}"#)
            .is_none());
    }

    #[test]
    fn ping_returns_empty_result() {
        let resp = call(&state(), r#"{"jsonrpc":"2.0","id":9,"method":"ping"}"#);
        assert_eq!(resp["id"], 9);
        assert_eq!(resp["result"], json!({}));
    }

    #[test]
    fn tools_list_exposes_chat_complete_embeddings() {
        let resp = call(&state(), r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().expect("tool name"))
            .collect();
        assert_eq!(names, vec!["chat", "complete", "embeddings"]);
        assert!(tools[1]["inputSchema"]["required"]
            .as_array()
            .expect("required")
            .contains(&json!("prompt")));
    }

    #[test]
    fn tools_call_complete_runs_inference() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"complete","arguments":{"prompt":"AB"}}}"#,
        );
        assert_eq!(resp["result"]["isError"], false);
        let text = resp["result"]["content"][0]["text"].as_str().expect("text");
        let output: Vec<f32> = serde_json::from_str(text).expect("float array for mlp model");
        assert_eq!(output.len(), 1);
        // Reference: byte-normalized "AB" → [65, 66] / 255 through the fixture weights.
        let (x0, x1) = (b'A' as f32 / 255.0, b'B' as f32 / 255.0);
        let h0 = x0 * 1.0 + x1 * 0.5 + 0.1;
        let h1 = x0 * -0.5 + x1 * 2.0 - 0.1;
        let expected = h0 * 2.0 - h1 + 0.5;
        assert!((output[0] - expected).abs() < 1e-4, "got {output:?}");
    }

    #[test]
    fn tools_call_chat_formats_messages_and_runs() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"chat","arguments":{"messages":[{"role":"user","content":"AB"}]}}}"#,
        );
        assert_eq!(resp["result"]["isError"], false);
        let text = resp["result"]["content"][0]["text"].as_str().expect("text");
        let output: Vec<f32> = serde_json::from_str(text).expect("float array for mlp model");
        // The chat template renders `user: AB\n`; the MLP path feeds the first
        // `hidden` bytes. Reference computation with the fixture weights:
        let xb: Vec<f32> = "user: AB\n"
            .bytes()
            .take(2)
            .map(|b| b as f32 / 255.0)
            .collect();
        let h0 = xb[0] * 1.0 + xb[1] * 0.5 + 0.1;
        let h1 = xb[0] * -0.5 + xb[1] * 2.0 - 0.1;
        let expected = h0 * 2.0 - h1 + 0.5;
        assert!((output[0] - expected).abs() < 1e-4, "got {output:?}");
    }

    #[test]
    fn tools_call_chat_invalid_arguments_is_error_result() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"chat","arguments":{}}}"#,
        );
        assert_eq!(resp["result"]["isError"], true);
        assert!(resp["result"]["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains("invalid chat arguments"));
    }

    #[test]
    fn tools_call_unknown_tool_is_error_result_not_protocol_error() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"warp-speed"}}"#,
        );
        assert_eq!(resp["id"], 6);
        assert_eq!(resp["result"]["isError"], true);
        assert!(resp["result"]["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains("Unknown tool"));
        assert!(resp.get("error").is_none(), "protocol must stay clean");
    }

    #[test]
    fn tools_call_embeddings_returns_json_envelope() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"embeddings","arguments":{"input":"AB"}}}"#,
        );
        assert_eq!(resp["result"]["isError"], false);
        let text = resp["result"]["content"][0]["text"].as_str().expect("text");
        let envelope: Value = serde_json::from_str(text).expect("json envelope");
        assert_eq!(envelope["model"], "serve_fixture");
        // MLP artifacts have no transformer hidden state → empty embedding vector.
        assert_eq!(envelope["embedding"], json!([]));
    }

    #[test]
    fn unknown_method_returns_jsonrpc_error() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":8,"method":"resources/list"}"#,
        );
        assert_eq!(resp["error"]["code"], -32601);
        assert!(resp["error"]["message"].as_str().unwrap().contains("resources/list"));
    }

    #[test]
    fn malformed_line_yields_parse_error_with_null_id() {
        let resp = call(&state(), "{not json");
        assert_eq!(resp["error"]["code"], -32700);
        assert_eq!(resp["id"], Value::Null);
    }

    #[test]
    fn missing_method_with_id_yields_invalid_request() {
        let resp = call(&state(), r#"{"jsonrpc":"2.0","id":42}"#);
        assert_eq!(resp["error"]["code"], -32600);
        assert_eq!(resp["id"], 42);
    }

    #[test]
    fn blank_lines_are_ignored() {
        assert!(handle_message(&state(), "   ").is_none());
    }

    /// Conversation-level capability check: a canonical MCP session — initialize,
    /// notification silence, ping, tools discovery, every tool, and the error
    /// paths — replayed in order against the fixture model.
    #[test]
    fn mcp_session_replays_end_to_end() {
        type SessionStep<'a> = (&'a str, fn(&Value));
        let state = state();
        let session: &[SessionStep] = &[
            (
                r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}"#,
                |r| {
                    assert_eq!(r["result"]["protocolVersion"], "2025-03-26");
                    assert_eq!(r["result"]["serverInfo"]["name"], "modelc");
                },
            ),
            (
                r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
                |r| assert!(r.is_null(), "notification must stay silent, got {r}"),
            ),
            (r#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#, |r| {
                assert_eq!(r["result"], json!({}));
            }),
            (
                r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#,
                |r| {
                    let tools = r["result"]["tools"].as_array().expect("tools");
                    assert_eq!(tools.len(), 3);
                },
            ),
            (
                r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"complete","arguments":{"prompt":"AB"}}}"#,
                |r| {
                    assert_eq!(r["result"]["isError"], false);
                    let text = r["result"]["content"][0]["text"].as_str().unwrap();
                    assert!(text.starts_with('['), "mlp output is a float array: {text}");
                },
            ),
            (
                r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"chat","arguments":{"messages":[{"role":"user","content":"AB"}]}}}"#,
                |r| assert_eq!(r["result"]["isError"], false),
            ),
            (
                r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"embeddings","arguments":{"inputs":["A","BB"]}}}"#,
                |r| {
                    assert_eq!(r["result"]["isError"], false);
                    let envelope: Value =
                        serde_json::from_str(r["result"]["content"][0]["text"].as_str().unwrap())
                            .unwrap();
                    assert_eq!(envelope["embeddings"].as_array().unwrap().len(), 2);
                },
            ),
            (
                r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"warp-speed"}}"#,
                |r| assert_eq!(r["result"]["isError"], true),
            ),
            ("{oops", |r| assert_eq!(r["error"]["code"], -32700)),
            (
                r#"{"jsonrpc":"2.0","id":8,"method":"resources/list"}"#,
                |r| assert_eq!(r["error"]["code"], -32601),
            ),
        ];
        for (line, check) in session {
            let response = handle_message(&state, line);
            check(response.as_ref().unwrap_or(&Value::Null));
        }
    }
}

//! `modelc serve --stdio` — MCP-compatible JSON-RPC 2.0 server on stdin/stdout.
//!
//! Newline-delimited JSON-RPC 2.0, the same wire format the Model Context
//! Protocol (MCP) uses over stdio, so the artifact can be attached directly to
//! editors and agent runtimes. stdout carries **only** protocol messages — all
//! logs go to stderr. Synchronous by design: a blocking line loop over stdin
//! with the same `AppState` inference plumbing the HTTP endpoints use.
//!
//! Methods: `initialize`, `ping`, `tools/list`, `tools/call`
//! (`chat` / `complete` / `embeddings` tools), `resources/list`,
//! `resources/read` (read-only model metadata), `notifications/initialized`
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
/// Earliest MCP version that defines the resources capability.
const MCP_RESOURCES_PROTOCOL_VERSION: &str = "2025-03-26";

/// URI of the model-overview resource (name, architecture, sizes, metadata).
const RESOURCE_MODEL: &str = "modelc://model";
/// URI of the tensor-table resource (name, shape, dtype, bytes per tensor).
const RESOURCE_TENSORS: &str = "modelc://model/tensors";
/// URI of the tokenizer-metadata resource (model type, special ids, template).
const RESOURCE_TOKENIZER: &str = "modelc://model/tokenizer";
/// Longest metadata value a resource returns before truncating it.
const METADATA_VALUE_MAX_CHARS: usize = 1024;

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
    handle_value(state, &msg)
}

/// Handle one already-parsed JSON-RPC message. Shared by both MCP transports
/// (stdio and Streamable HTTP) so they answer identically.
pub(super) fn handle_value(state: &super::AppState, msg: &Value) -> Option<Value> {
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
        "initialize" => Some(rpc_result(&id, initialize_result(msg))),
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
        "resources/list" => Some(rpc_result(&id, resources_list())),
        "resources/read" => {
            let Some(uri) = msg.pointer("/params/uri").and_then(|v| v.as_str()) else {
                return Some(rpc_error(
                    &id,
                    -32602,
                    "invalid params: resources/read requires a `uri` string",
                ));
            };
            match resources_read(state, uri) {
                Ok(result) => Some(rpc_result(&id, result)),
                Err(message) => Some(rpc_error(&id, -32002, &message)),
            }
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
    let mut capabilities = json!({ "tools": { "listChanged": false } });
    // Resources entered MCP in 2025-03-26; a 2024-11-05 client does not know
    // them, so the capability is only advertised from that version on.
    if version >= MCP_RESOURCES_PROTOCOL_VERSION {
        capabilities["resources"] = json!({ "subscribe": false, "listChanged": false });
    }
    json!({
        "protocolVersion": version,
        "capabilities": capabilities,
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
        },
        {
            "name": "rerank",
            "description": "Rank documents by relevance to a query (Cohere-style reranking)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "documents": { "type": "array", "items": { "type": "string" } },
                    "top_n": { "type": "integer" },
                    "return_documents": { "type": "boolean" }
                },
                "required": ["query", "documents"]
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

/// The read-only resources this server exposes: model identity, the tensor
/// table, and the tokenizer metadata the artifact carried.
fn resources_list() -> Value {
    json!({
        "resources": [
            {
                "uri": RESOURCE_MODEL,
                "name": "model",
                "description": "Model overview: name, architecture, parameter and byte counts, quantization, artifact metadata",
                "mimeType": "application/json"
            },
            {
                "uri": RESOURCE_TENSORS,
                "name": "model-tensors",
                "description": "Tensor table: name, shape, dtype, element count and stored bytes",
                "mimeType": "application/json"
            },
            {
                "uri": RESOURCE_TOKENIZER,
                "name": "model-tokenizer",
                "description": "Tokenizer metadata: model type, BOS/EOS token ids, chat template, raw tokenizer entries",
                "mimeType": "application/json"
            }
        ]
    })
}

/// Read one resource. `Err` carries the message for a JSON-RPC error.
fn resources_read(state: &super::AppState, uri: &str) -> Result<Value, String> {
    let content = match uri {
        RESOURCE_MODEL => model_overview(state),
        RESOURCE_TENSORS => tensor_table(state),
        RESOURCE_TOKENIZER => tokenizer_metadata(state),
        other => return Err(format!("resource not found: {other}")),
    };
    let text = serde_json::to_string_pretty(&content).unwrap_or_else(|_| "{}".to_string());
    Ok(json!({
        "contents": [ { "uri": uri, "mimeType": "application/json", "text": text } ]
    }))
}

/// Model overview: identity, size, and the artifact's own metadata.
fn model_overview(state: &super::AppState) -> Value {
    json!({
        "name": state.name,
        "architecture": state.architecture,
        "tensor_count": state.tensor_names.len(),
        "total_params": state.total_params,
        "total_bytes": state.total_bytes,
        "quantization": quantization_label(state),
        "metadata": state.metadata,
    })
}

/// The tensor table, sorted by tensor name so the output is deterministic.
fn tensor_table(state: &super::AppState) -> Value {
    let mut tensors: Vec<Value> = state
        .base_tensors
        .iter()
        .map(|(name, td)| {
            json!({
                "name": name,
                "shape": td.shape,
                "dtype": format!("{:?}", td.dtype),
                "elements": td.element_count(),
                "bytes": td.byte_len(),
            })
        })
        .collect();
    tensors.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    json!({ "tensors": tensors })
}

/// Tokenizer metadata: model type, special token ids, chat template size, and
/// the raw `tokenizer.*` entries the artifact carried.
fn tokenizer_metadata(state: &super::AppState) -> Value {
    let mut entries = serde_json::Map::new();
    for (key, value) in state
        .metadata
        .iter()
        .filter(|(key, _)| key.starts_with("tokenizer."))
    {
        entries.insert(key.clone(), Value::String(truncate(value, METADATA_VALUE_MAX_CHARS)));
    }
    json!({
        "model_type": state.metadata.get("tokenizer.ggml.model"),
        "bos_token_id": state.metadata.get("tokenizer.ggml.bos_token_id").and_then(|v| v.parse::<i64>().ok()),
        "eos_token_id": state.metadata.get("tokenizer.ggml.eos_token_id").and_then(|v| v.parse::<i64>().ok()),
        "chat_template_chars": state.chat_template.as_ref().map(|t| t.chars().count()),
        "entries": entries,
    })
}

/// The artifact's quantization label: "f32" for plain float weights, the
/// dtype name when the model is quantized, "mixed" when it carries several.
fn quantization_label(state: &super::AppState) -> String {
    let mut dtypes: Vec<String> = state
        .base_tensors
        .values()
        .map(|td| format!("{:?}", td.dtype))
        .collect();
    dtypes.sort();
    dtypes.dedup();
    match dtypes.len() {
        0 => "none".to_string(),
        1 => dtypes[0].to_lowercase(),
        _ => {
            if dtypes.iter().all(|d| d == "F32") {
                "f32".to_string()
            } else {
                "mixed".to_string()
            }
        }
    }
}

/// Cap a metadata value so a resource never carries an unbounded blob.
fn truncate(value: &str, max_chars: usize) -> String {
    let total = value.chars().count();
    if total <= max_chars {
        return value.to_string();
    }
    let head: String = value.chars().take(max_chars).collect();
    format!("{head}… [truncated, {total} chars total]")
}

fn call_tool(state: &super::AppState, name: &str, arguments: Option<Value>) -> Value {
    let arguments = arguments.unwrap_or_else(|| json!({}));
    match name {
        "chat" => tool_chat(state, &arguments),
        "complete" => tool_complete(state, &arguments),
        "embeddings" => tool_embeddings(state, &arguments),
        "rerank" => tool_rerank(state, &arguments),
        other => tool_result(format!("Unknown tool: {other}"), true),
    }
}

/// Rank documents against a query with the same core the `/v1/rerank` endpoint
/// uses, so an agent reranks through the same numbers the HTTP API returns.
fn tool_rerank(state: &super::AppState, arguments: &Value) -> Value {
    let query = arguments
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let documents: Vec<String> = arguments
        .get("documents")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if query.is_empty() || documents.is_empty() {
        return tool_result(
            "invalid rerank arguments: `query` and `documents` are required".to_string(),
            true,
        );
    }
    let top_n = arguments
        .get("top_n")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let return_documents = arguments
        .get("return_documents")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let ranked = super::openai::rerank_documents(state, query, &documents, top_n);
    let results: Vec<Value> = ranked
        .iter()
        .map(|(index, relevance_score)| {
            let mut entry = json!({ "index": index, "relevance_score": relevance_score });
            if return_documents {
                entry["document"] = json!({ "text": documents[*index] });
            }
            entry
        })
        .collect();
    tool_result(
        serde_json::to_string(&json!({ "results": results })).unwrap_or_default(),
        false,
    )
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
            metadata: HashMap::from([
                ("general.architecture".to_string(), "llama".to_string()),
                ("tokenizer.ggml.model".to_string(), "llama".to_string()),
                ("tokenizer.ggml.bos_token_id".to_string(), "1".to_string()),
                ("tokenizer.ggml.eos_token_id".to_string(), "2".to_string()),
                ("tokenizer.chat_template".to_string(), "{{ bos }}".to_string()),
            ]),
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
        assert_eq!(names, vec!["chat", "complete", "embeddings", "rerank"]);
        assert!(tools[1]["inputSchema"]["required"]
            .as_array()
            .expect("required")
            .contains(&json!("prompt")));
        // The rerank tool requires a query and documents.
        let rerank = tools
            .iter()
            .find(|t| t["name"] == "rerank")
            .expect("rerank tool");
        let required = rerank["inputSchema"]["required"].as_array().expect("required");
        assert!(required.contains(&json!("query")));
        assert!(required.contains(&json!("documents")));
    }

    #[test]
    fn tools_call_rerank_ranks_documents() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":18,"method":"tools/call","params":{"name":"rerank","arguments":{"query":"capital cities","documents":["washington","paris","tokyo"],"top_n":2}}}"#,
        );
        assert_eq!(resp["result"]["isError"], false);
        let text = resp["result"]["content"][0]["text"].as_str().expect("text");
        let envelope: Value = serde_json::from_str(text).expect("json envelope");
        let results = envelope["results"].as_array().expect("results");
        assert_eq!(results.len(), 2, "top_n truncates");
        for r in results {
            let score = r["relevance_score"].as_f64().expect("score");
            assert!(
                (0.0..=1.0).contains(&score),
                "score is clamped into Cohere's documented range: {score}"
            );
            assert!(r["index"].as_u64().is_some());
            assert!(
                r.get("document").is_none(),
                "document text is opt-in, like the HTTP endpoint"
            );
        }
    }

    #[test]
    fn tools_call_rerank_can_return_the_document_text() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"rerank","arguments":{"query":"q","documents":["washington","paris"],"return_documents":true}}}"#,
        );
        assert_eq!(resp["result"]["isError"], false);
        let text = resp["result"]["content"][0]["text"].as_str().expect("text");
        let envelope: Value = serde_json::from_str(text).expect("json envelope");
        let results = envelope["results"].as_array().expect("results");
        assert_eq!(results.len(), 2);
        for r in results {
            let idx = r["index"].as_u64().unwrap() as usize;
            assert_eq!(
                r["document"]["text"].as_str().unwrap(),
                ["washington", "paris"][idx],
                "document text matches the original index"
            );
        }
    }

    #[test]
    fn tools_call_rerank_invalid_arguments_is_error_result() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":19,"method":"tools/call","params":{"name":"rerank","arguments":{"query":"q"}}}"#,
        );
        assert_eq!(resp["result"]["isError"], true);
        assert!(resp["result"]["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains("invalid rerank arguments"));
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
    fn resources_list_exposes_the_model_resources() {
        let resp = call(&state(), r#"{"jsonrpc":"2.0","id":10,"method":"resources/list"}"#);
        let resources = resp["result"]["resources"].as_array().expect("resources");
        let uris: Vec<&str> = resources
            .iter()
            .map(|r| r["uri"].as_str().expect("uri"))
            .collect();
        assert_eq!(
            uris,
            vec![
                "modelc://model",
                "modelc://model/tensors",
                "modelc://model/tokenizer"
            ]
        );
        for r in resources {
            assert_eq!(r["mimeType"], "application/json");
            assert!(!r["name"].as_str().unwrap().is_empty());
            assert!(!r["description"].as_str().unwrap().is_empty());
        }
    }

    #[test]
    fn resources_read_model_overview_reports_identity_and_sizes() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":11,"method":"resources/read","params":{"uri":"modelc://model"}}"#,
        );
        assert_eq!(resp["result"]["contents"][0]["uri"], "modelc://model");
        assert_eq!(
            resp["result"]["contents"][0]["mimeType"],
            "application/json"
        );
        let overview: Value =
            serde_json::from_str(resp["result"]["contents"][0]["text"].as_str().expect("text"))
                .expect("overview json");
        assert_eq!(overview["name"], "serve_fixture");
        assert_eq!(overview["architecture"], "mlp");
        assert_eq!(overview["tensor_count"], 4);
        assert_eq!(overview["total_params"], 9);
        assert_eq!(overview["quantization"], "f32");
        // The artifact's own metadata rides along.
        assert_eq!(overview["metadata"]["general.architecture"], "llama");
    }

    #[test]
    fn resources_read_tensor_table_is_sorted_and_typed() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":12,"method":"resources/read","params":{"uri":"modelc://model/tensors"}}"#,
        );
        let table: Value =
            serde_json::from_str(resp["result"]["contents"][0]["text"].as_str().expect("text"))
                .expect("tensor table json");
        let tensors = table["tensors"].as_array().expect("tensors");
        let names: Vec<&str> = tensors
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec![
                "layer0.bias",
                "layer0.weight",
                "layer1.bias",
                "layer1.weight"
            ],
            "sorted by name"
        );
        assert_eq!(tensors[1]["shape"], json!([2, 2]));
        assert_eq!(tensors[1]["dtype"], "F32");
        assert_eq!(tensors[1]["elements"], 4);
        assert_eq!(tensors[1]["bytes"], 16);
    }

    #[test]
    fn resources_read_tokenizer_reports_special_ids_and_entries() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":13,"method":"resources/read","params":{"uri":"modelc://model/tokenizer"}}"#,
        );
        let tokenizer: Value =
            serde_json::from_str(resp["result"]["contents"][0]["text"].as_str().expect("text"))
                .expect("tokenizer json");
        assert_eq!(tokenizer["model_type"], "llama");
        assert_eq!(tokenizer["bos_token_id"], 1);
        assert_eq!(tokenizer["eos_token_id"], 2);
        assert_eq!(tokenizer["chat_template_chars"], 9);
        assert_eq!(tokenizer["entries"]["tokenizer.ggml.model"], "llama");
        assert!(
            tokenizer["entries"].get("general.architecture").is_none(),
            "only tokenizer.* entries are exposed"
        );
    }

    #[test]
    fn resources_read_unknown_uri_is_resource_not_found() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":14,"method":"resources/read","params":{"uri":"modelc://nope"}}"#,
        );
        assert_eq!(resp["error"]["code"], -32002);
        assert!(
            resp["error"]["message"]
                .as_str()
                .unwrap()
                .contains("modelc://nope")
        );
    }

    #[test]
    fn resources_read_without_uri_is_invalid_params() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":15,"method":"resources/read","params":{}}"#,
        );
        assert_eq!(resp["error"]["code"], -32602);
    }

    #[test]
    fn initialize_advertises_resources_only_for_clients_that_know_them() {
        let modern = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":16,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}"#,
        );
        assert_eq!(
            modern["result"]["capabilities"]["resources"]["subscribe"],
            false
        );

        // The 2024-11-05 spec predates resources — advertising them would point
        // clients at a method they do not know.
        let legacy = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":17,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#,
        );
        assert!(
            legacy["result"]["capabilities"].get("resources").is_none(),
            "legacy clients must not see the resources capability"
        );
    }

    #[test]
    fn unknown_method_returns_jsonrpc_error() {
        let resp = call(
            &state(),
            r#"{"jsonrpc":"2.0","id":8,"method":"prompts/list"}"#,
        );
        assert_eq!(resp["error"]["code"], -32601);
        assert!(resp["error"]["message"].as_str().unwrap().contains("prompts/list"));
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
                    assert_eq!(tools.len(), 4);
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
            (
                r#"{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"rerank","arguments":{"query":"capital cities","documents":["washington","paris"],"top_n":1}}}"#,
                |r| {
                    assert_eq!(r["result"]["isError"], false);
                    let envelope: Value =
                        serde_json::from_str(r["result"]["content"][0]["text"].as_str().unwrap())
                            .unwrap();
                    assert_eq!(envelope["results"].as_array().unwrap().len(), 1);
                },
            ),
            (
                r#"{"jsonrpc":"2.0","id":8,"method":"resources/list"}"#,
                |r| {
                    let resources = r["result"]["resources"].as_array().expect("resources");
                    assert_eq!(resources.len(), 3);
                },
            ),
            (
                r#"{"jsonrpc":"2.0","id":9,"method":"resources/read","params":{"uri":"modelc://model"}}"#,
                |r| {
                    assert_eq!(r["result"]["contents"][0]["uri"], "modelc://model");
                },
            ),
            (
                r#"{"jsonrpc":"2.0","id":10,"method":"resources/read","params":{"uri":"modelc://nope"}}"#,
                |r| assert_eq!(r["error"]["code"], -32002),
            ),
            ("{oops", |r| assert_eq!(r["error"]["code"], -32700)),
            (
                r#"{"jsonrpc":"2.0","id":11,"method":"prompts/list"}"#,
                |r| assert_eq!(r["error"]["code"], -32601),
            ),
        ];
        for (line, check) in session {
            let response = handle_message(&state, line);
            check(response.as_ref().unwrap_or(&Value::Null));
        }
    }
}

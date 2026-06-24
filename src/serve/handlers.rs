use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    response::sse::{Event, Sse},
};
use tokio_stream::wrappers::ReceiverStream;

use super::infer::{
    run_embeddings, run_inference, run_mlp_forward_batched, run_text_inference_token_ids,
    run_text_inference_with_config,
};
use super::{
    AppState, CancelOnDrop, ChatRequest, ChatResponse, CompleteRequest, CompleteResponse,
    EmbeddingEntry, EmbeddingsRequest, EmbeddingsResponse, HealthResponse, InferRequest,
    InferResponse, LoraLoadRequest, LoraLoadResponse, LoraUnloadResponse, Message, ModelInfo,
    StreamChunk, SystemInfo, TokenizeRequest, TokenizeResponse,
};

pub(super) async fn infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> Json<InferResponse> {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    if !req.inputs.is_empty() {
        let _timer = super::metrics::InferenceTimer::new(&state.metrics);
        let start = std::time::Instant::now();

        // Use the batched MLP path when we have multiple inputs and a known MLP plan.
        let outs = if req.inputs.len() > 1 {
            if let Some(plan) = &state.mlp_plan {
                let runtime = state.runtime.read().expect("runtime lock poisoned");
                run_mlp_forward_batched(&runtime, plan, &req.inputs, state.profile)
            } else {
                req.inputs
                    .iter()
                    .map(|inp| run_inference(&state, inp))
                    .collect()
            }
        } else {
            req.inputs
                .iter()
                .map(|inp| run_inference(&state, inp))
                .collect()
        };

        if state.profile {
            eprintln!(
                "  batch infer: {} items in {:.3} ms",
                outs.len(),
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        return Json(InferResponse {
            output: None,
            outputs: Some(outs),
        });
    }

    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let start = std::time::Instant::now();
    let output = run_inference(&state, &req.input);
    if state.profile {
        eprintln!("  infer: {:.3} ms", start.elapsed().as_secs_f64() * 1000.0);
    }
    Json(InferResponse {
        output: Some(output),
        outputs: None,
    })
}

pub(super) async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.name.clone(),
        architecture: state.architecture.clone(),
    })
}

pub(super) async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Json<EmbeddingsResponse> {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    if !req.inputs.is_empty() {
        let entries: Vec<EmbeddingEntry> = req
            .inputs
            .iter()
            .enumerate()
            .map(|(idx, text)| {
                let input: Vec<f32> = text.bytes().map(|b| b as f32 / 255.0).collect();
                let embedding = run_embeddings(&state, &input).unwrap_or_default();
                EmbeddingEntry {
                    embedding,
                    index: idx,
                }
            })
            .collect();
        return Json(EmbeddingsResponse {
            embedding: None,
            embeddings: Some(entries),
            model: state.name.clone(),
        });
    }

    let input: Vec<f32> = req.input.bytes().map(|b| b as f32 / 255.0).collect();
    let embedding = run_embeddings(&state, &input).unwrap_or_default();
    Json(EmbeddingsResponse {
        embedding: Some(embedding),
        embeddings: None,
        model: state.name.clone(),
    })
}

pub(super) async fn model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {
    Json(ModelInfo {
        name: state.name.clone(),
        architecture: state.architecture.clone(),
        total_params: state.total_params,
        total_bytes: state.total_bytes,
        tensors: state.tensor_names.clone(),
    })
}

/// `POST /tokenize` — encode text into token IDs using the model's tokenizer
/// (byte-level BPE fallback, the same one used by `/chat` and `/complete`).
/// Accepts `{ "input": "..." }` (single) or `{ "inputs": ["...", "..."] }` (batch).
pub(super) async fn tokenize(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> Json<TokenizeResponse> {
    let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
    if !req.inputs.is_empty() {
        let batch: Vec<Vec<u32>> = req.inputs.iter().map(|s| tokenizer.encode(s)).collect();
        let count = batch.iter().map(|t| t.len()).sum();
        return Json(TokenizeResponse {
            tokens: None,
            tokens_batch: Some(batch),
            count,
        });
    }
    let tokens = tokenizer.encode(&req.input);
    let count = tokens.len();
    Json(TokenizeResponse {
        tokens: Some(tokens),
        tokens_batch: None,
        count,
    })
}

/// `GET /v1/system` — best-effort system/hardware info for orchestration and
/// debugging (CPU cores, OS, architecture, Metal availability, total memory).
pub(super) async fn system_info(State(state): State<Arc<AppState>>) -> Json<SystemInfo> {
    Json(SystemInfo {
        model: state.name.clone(),
        architecture: state.architecture.clone(),
        total_params: state.total_params,
        total_bytes: state.total_bytes,
        cpu_cores: cpu_core_count(),
        os: std::env::consts::OS,
        cpu_arch: std::env::consts::ARCH,
        pointer_width: std::mem::size_of::<usize>() * 8,
        metal_available: cfg!(target_os = "macos"),
        memory_total_bytes: total_memory_bytes(),
    })
}

/// Number of logical CPU cores available to the process.
fn cpu_core_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Best-effort total physical memory in bytes. Reads `/proc/meminfo` on Linux,
/// shells out to `sysctl hw.memsize` on macOS, and returns `None` elsewhere.
fn total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

pub(super) async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
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
        crate::chat_template::apply_chat_template(state.chat_template.as_deref(), &messages)
            .unwrap_or_else(|_| {
                req.messages
                    .last()
                    .map(|m| m.content.clone())
                    .unwrap_or_default()
            });
    let gen_cfg = make_generation_config(
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
    );
    let output = if let Some(ref schema) = req.json_schema {
        crate::json_schema::generate_with_schema(
            |cfg| run_text_inference_with_config(&state, &prompt, cfg),
            schema,
            &gen_cfg,
            3,
        )
    } else {
        run_text_inference_with_config(&state, &prompt, &gen_cfg)
    };
    Json(ChatResponse {
        message: Message {
            role: "assistant".to_string(),
            content: output,
        },
    })
}

pub(super) async fn complete(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompleteRequest>,
) -> Json<CompleteResponse> {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);
    let gen_cfg = make_generation_config(
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
    );
    let output = if let Some(ref schema) = req.json_schema {
        crate::json_schema::generate_with_schema(
            |cfg| run_text_inference_with_config(&state, &req.prompt, cfg),
            schema,
            &gen_cfg,
            3,
        )
    } else {
        run_text_inference_with_config(&state, &req.prompt, &gen_cfg)
    };
    Json(CompleteResponse { completion: output })
}

pub(super) async fn chat_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Sse<CancelOnDrop<ReceiverStream<Result<Event, Infallible>>>> {
    let messages: Vec<crate::chat_template::ChatMessage> = req
        .messages
        .iter()
        .map(|m| crate::chat_template::ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    let prompt =
        crate::chat_template::apply_chat_template(state.chat_template.as_deref(), &messages)
            .unwrap_or_else(|_| {
                req.messages
                    .last()
                    .map(|m| m.content.clone())
                    .unwrap_or_default()
            });
    let mut gen_cfg = make_generation_config(
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
    );

    // Cancellation flag: set when the SSE client disconnects (stream dropped).
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    gen_cfg.cancel = Some(cancel.clone());

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(4);

    tokio::spawn(async move {
        let token_ids = run_text_inference_token_ids(&state, &prompt, &gen_cfg);
        let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
        let prompt_ids = tokenizer.encode(&prompt);
        let mut prev_text = String::new();

        for (idx, &_token_id) in token_ids.iter().enumerate() {
            let cumulative = [prompt_ids.as_slice(), &token_ids[..=idx]].concat();
            let text = tokenizer.decode(&cumulative);
            if let Some(delta) = text.strip_prefix(&prev_text) {
                if !delta.is_empty() {
                    let chunk = serde_json::to_string(&StreamChunk {
                        delta: delta.to_string(),
                        done: false,
                    })
                    .unwrap();
                    // Stop dripping once the client is gone.
                    if tx.send(Ok(Event::default().data(chunk))).await.is_err() {
                        break;
                    }
                }
                prev_text = text;
            }
        }

        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::to_string(&StreamChunk {
                    delta: String::new(),
                    done: true,
                })
                .unwrap(),
            )))
            .await;
    });

    Sse::new(CancelOnDrop::new(ReceiverStream::new(rx), cancel))
}

/// Build a generation config, applying per-request overrides on top of server defaults.
#[allow(clippy::too_many_arguments)]
fn make_generation_config(
    base: &crate::generate::GenerationConfig,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    grammar: Option<String>,
    stop: Vec<String>,
    seed: Option<u64>,
    repetition_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
) -> crate::generate::GenerationConfig {
    let constraint = grammar.and_then(|pat| {
        crate::constraint::RegexConstraint::new(&pat)
            .map(|c| std::sync::Arc::new(c) as std::sync::Arc<dyn crate::constraint::Constraint>)
    });
    crate::generate::GenerationConfig {
        max_tokens: max_tokens.unwrap_or(base.max_tokens),
        temperature: temperature.unwrap_or(base.temperature),
        top_p: top_p.unwrap_or(base.top_p),
        min_p: min_p.unwrap_or(base.min_p),
        gamma: base.gamma,
        use_int8_kv: base.use_int8_kv,
        use_mixed_kv: base.use_mixed_kv,
        constraint: constraint.or_else(|| base.constraint.clone()),
        max_context: base.max_context,
        anchor_tokens: base.anchor_tokens,
        stop: if stop.is_empty() {
            base.stop.clone()
        } else {
            stop
        },
        seed: seed.or(base.seed),
        repetition_penalty: repetition_penalty.unwrap_or(base.repetition_penalty),
        presence_penalty: presence_penalty.unwrap_or(base.presence_penalty),
        frequency_penalty: frequency_penalty.unwrap_or(base.frequency_penalty),
        cancel: None,
    }
}

pub(super) async fn lora_load(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoraLoadRequest>,
) -> Json<LoraLoadResponse> {
    let path = std::path::Path::new(&req.path);
    let mut model = crate::model::Model {
        name: state.name.clone(),
        architecture: state.architecture.clone(),
        tensors: state.base_tensors.clone(),
        metadata: std::collections::HashMap::new(),
    };

    match crate::lora::apply_lora(&mut model, path, req.alpha) {
        Ok(()) => {
            let mut runtime = state.runtime.write().expect("runtime lock poisoned");
            *runtime = crate::runtime::serve::Runtime::from_raw(&model.tensors);
            Json(LoraLoadResponse {
                applied: model.tensors.len(), // lora.rs doesn't return counts directly, so we approximate
                skipped: 0,
                message: format!("LoRA loaded from {:?}", path),
            })
        }
        Err(e) => Json(LoraLoadResponse {
            applied: 0,
            skipped: 0,
            message: format!("Failed to load LoRA: {e}"),
        }),
    }
}

pub(super) async fn lora_unload(State(state): State<Arc<AppState>>) -> Json<LoraUnloadResponse> {
    let mut runtime = state.runtime.write().expect("runtime lock poisoned");
    *runtime = crate::runtime::serve::Runtime::from_raw(&state.base_tensors);
    Json(LoraUnloadResponse {
        message: "LoRA unloaded; base model restored".to_string(),
    })
}

pub(super) async fn metrics_handler(
    State(state): State<Arc<AppState>>,
) -> axum::response::Response<String> {
    let body = state.metrics.render();
    axum::response::Response::builder()
        .header("Content-Type", "text/plain; version=0.0.4")
        .body(body)
        .unwrap()
}

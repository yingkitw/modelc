use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    response::IntoResponse,
    response::sse::{Event, Sse},
};
use tokio_stream::wrappers::ReceiverStream;

use super::infer::{
    run_embeddings, run_inference, run_mlp_forward_batched, run_text_inference_token_ids,
    run_text_inference_with_config,
};
use super::{
    AppState, CancelOnDrop, ChatRequest, ChatResponse, CompleteRequest, CompleteResponse,
    DefaultGenerationProps, DetokenizeRequest, DetokenizeResponse, Document, EmbeddingEntry,
    EmbeddingsRequest, EmbeddingsResponse, HealthResponse, InfillRequest, InfillResponse,
    InferRequest, InferResponse, LoraLoadRequest, LoraLoadResponse, LoraUnloadResponse, Message,
    ModelInfo, RerankingRequest, RerankingResponse, RerankingResult, ServerProps, StreamChunk,
    SystemInfo, TokenizeRequest, TokenizeResponse, VersionInfo,
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

/// `GET /props` — llama.cpp-compatible server properties endpoint.
/// Exposes the chat template, default generation parameters, and model metadata
/// so clients can auto-configure without trial-and-error.
pub(super) async fn server_props(State(state): State<Arc<AppState>>) -> Json<ServerProps> {
    let g = &state.generation;
    Json(ServerProps {
        model: state.name.clone(),
        architecture: state.architecture.clone(),
        total_params: state.total_params,
        total_bytes: state.total_bytes,
        chat_template: state.chat_template.clone(),
        default_generation: DefaultGenerationProps {
            max_tokens: g.max_tokens,
            temperature: g.temperature,
            top_p: g.top_p,
            min_p: g.min_p,
            repetition_penalty: g.repetition_penalty,
            presence_penalty: g.presence_penalty,
            frequency_penalty: g.frequency_penalty,
            gamma: g.gamma,
        },
    })
}

/// `GET /api/version` — expose CLI version and git SHA for orchestration.
pub(super) async fn version_info() -> Json<VersionInfo> {
    Json(VersionInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: crate::GIT_SHA.to_string(),
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

/// `POST /detokenize` — decode token IDs back to text using the model's tokenizer
/// (byte-level BPE fallback). Accepts `{ "tokens": [id, ...] }`.
pub(super) async fn detokenize(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<DetokenizeRequest>,
) -> Json<DetokenizeResponse> {
    let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
    let text = tokenizer.decode(&req.tokens);
    Json(DetokenizeResponse { text })
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
    let prompt = crate::chat_template::apply_chat_template(
        state.chat_template.as_deref(),
        &messages,
    );
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
        req.logit_bias.clone(),
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
        req.logit_bias.clone(),
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

pub(super) async fn infill(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InfillRequest>,
) -> Json<InfillResponse> {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);

    let prompt = if let Some(ref extra) = req.prompt {
        format!("{extra}\n{prefix}", prefix = req.prefix)
    } else {
        req.prefix.clone()
    };

    let mut stop = vec![req.suffix.clone(), "\n```".to_string()];
    if let Some(ref extra) = req.prompt {
        stop.push(extra.clone());
    }

    let gen_cfg = make_generation_config(
        &state.generation,
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.min_p,
        None,
        stop,
        req.seed,
        req.repetition_penalty,
        req.presence_penalty,
        req.frequency_penalty,
        None,
    );

    let output = run_text_inference_with_config(&state, &prompt, &gen_cfg);
    Json(InfillResponse { completion: output })
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
    let prompt = crate::chat_template::apply_chat_template(
        state.chat_template.as_deref(),
        &messages,
    );
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
        req.logit_bias.clone(),
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
    logit_bias: Option<HashMap<u32, f32>>,
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
        logit_bias: logit_bias.unwrap_or_else(|| base.logit_bias.clone()),
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

pub(super) async fn reranking(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RerankingRequest>,
) -> Json<RerankingResponse> {
    let _guard = super::metrics::ActiveRequestGuard::new(&state.metrics);
    let _timer = super::metrics::InferenceTimer::new(&state.metrics);

    let query_input: Vec<f32> = req.query.bytes().map(|b| b as f32 / 255.0).collect();
    let query_emb = run_embeddings(&state, &query_input).unwrap_or_default();

    let mut results: Vec<RerankingResult> = req
        .documents
        .iter()
        .enumerate()
        .map(|(idx, doc)| {
            let doc_input: Vec<f32> = doc.bytes().map(|b| b as f32 / 255.0).collect();
            let doc_emb = run_embeddings(&state, &doc_input).unwrap_or_default();
            let score = cosine_similarity(&query_emb, &doc_emb);
            RerankingResult {
                index: idx,
                relevance_score: score,
                document: Document {
                    text: doc.clone(),
                },
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(top_n) = req.top_n {
        results.truncate(top_n);
    }

    Json(RerankingResponse {
        model: state.name.clone(),
        results,
    })
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

pub(super) async fn web_ui() -> axum::response::Html<&'static str> {
    let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>modelc — Local LLM Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
#header{padding:12px 20px;background:#16213e;border-bottom:1px solid #0f3460}
#header h1{font-size:18px;font-weight:600}
#header .model{font-size:12px;color:#8899aa;margin-top:2px}
#messages{flex:1;overflow-y:auto;padding:20px;max-width:900px;margin:0 auto;width:100%}
.msg{margin-bottom:16px;max-width:80%}
.msg.user{margin-left:auto}
.msg .role{font-size:11px;color:#667788;margin-bottom:4px;text-transform:uppercase}
.msg .bubble{padding:12px 16px;border-radius:12px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}
.msg.user .bubble{background:#0f3460}
.msg.assistant .bubble{background:#222244;border:1px solid #333}
#input-area{padding:16px 20px;background:#16213e;border-top:1px solid #0f3460;max-width:900px;margin:0 auto;width:100%}
#input-row{display:flex;gap:8px}
#prompt{flex:1;padding:12px 16px;border:1px solid #0f3460;border-radius:8px;background:#1a1a2e;color:#e0e0e0;font-size:14px;resize:none;height:48px;max-height:120px}
#prompt:focus{outline:none;border-color:#533483}
#send{padding:12px 24px;background:#533483;border:none;border-radius:8px;color:#fff;font-size:14px;cursor:pointer;white-space:nowrap}
#send:hover{background:#6a4493}
#send:disabled{opacity:0.5;cursor:not-allowed}
#status{font-size:11px;color:#667788;margin-top:8px;text-align:center}
</style>
</head>
<body>
<div id="header">
<h1>modelc</h1>
<div class="model" id="model-name">Loading…</div>
</div>
<div id="messages"></div>
<div id="input-area">
<div id="input-row">
<textarea id="prompt" placeholder="Send a message… (Enter to send, Shift+Enter for newline)" rows="1"></textarea>
<button id="send">Send</button>
</div>
<div id="status"></div>
</div>
<script>
const promptEl=document.getElementById('prompt');
const sendBtn=document.getElementById('send');
const messagesEl=document.getElementById('messages');
const statusEl=document.getElementById('status');
const modelNameEl=document.getElementById('model-name');
let busy=false;
function addMsg(role,text){
  const d=document.createElement('div');
  d.className='msg '+role;
  const r=document.createElement('div');
  r.className='role';r.textContent=role;
  const b=document.createElement('div');
  b.className='bubble';b.textContent=text;
  d.appendChild(r);d.appendChild(b);
  messagesEl.appendChild(d);
  messagesEl.scrollTop=messagesEl.scrollHeight;
  return b;
}
async function send(){
  if(busy)return;
  const text=promptEl.value.trim();
  if(!text)return;
  busy=true;sendBtn.disabled=true;
  promptEl.value='';promptEl.style.height='48px';
  addMsg('user',text);
  const bubble=addMsg('assistant','');
  statusEl.textContent='Generating…';
  try{
    const res=await fetch('/v1/chat/completions',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'',messages:[{role:'user',content:text}],stream:true})
    });
    const reader=res.body.getReader();
    const decoder=new TextDecoder();
    let buf='';
    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\n');
      buf=lines.pop();
      for(const line of lines){
        if(line.startsWith('data: ')){
          const data=line.slice(6);
          if(data==='[DONE]')continue;
          try{
            const j=JSON.parse(data);
            const delta=j.choices?.[0]?.delta?.content||'';
            bubble.textContent+=delta;
            messagesEl.scrollTop=messagesEl.scrollHeight;
          }catch(e){}
        }
      }
    }
  }catch(e){bubble.textContent='Error: '+e.message;}
  if(!bubble.textContent)bubble.textContent='(empty response)';
  statusEl.textContent='';
  busy=false;sendBtn.disabled=false;promptEl.focus();
}
sendBtn.addEventListener('click',send);
promptEl.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}
});
promptEl.addEventListener('input',()=>{
  promptEl.style.height='48px';
  promptEl.style.height=Math.min(promptEl.scrollHeight,120)+'px';
});
fetch('/info').then(r=>r.json()).then(j=>{
  modelNameEl.textContent=j.name+' · '+j.architecture+' · '+(j.total_params/1e6).toFixed(1)+'M params';
}).catch(()=>{modelNameEl.textContent='modelc';});
promptEl.focus();
</script>
</body>
</html>"#;
    axum::response::Html(html)
}

pub(super) async fn api_tags() -> Json<super::ApiTagsResponse> {
    let models = crate::store::list_models().unwrap_or_default();
    let tags: Vec<super::ApiTag> = models
        .iter()
        .map(|m| super::ApiTag {
            name: m.name.clone(),
            size: m.size_bytes,
            details: super::ApiTagDetails {
                architecture: m.architecture.clone().unwrap_or_default(),
                parameter_size: format!("{}", m.params.unwrap_or(0)),
                quantization: if m.compressed { "compressed" } else { "f32" }.to_string(),
            },
        })
        .collect();
    Json(super::ApiTagsResponse { models: tags })
}

pub(super) async fn api_show(
    Json(req): Json<super::ApiShowRequest>,
) -> axum::response::Response {
    match crate::store::resolve_model_path(&req.name) {
        Ok(path) => match crate::pack::read_header(&path) {
            Ok(header) => {
                let params: usize = header
                    .tensors
                    .iter()
                    .map(|t| t.shape.iter().product::<usize>())
                    .sum();
                let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                Json(super::ApiShowResponse {
                    name: header.name,
                    architecture: header.architecture,
                    size_bytes,
                    parameter_size: format!("{}", params),
                    quantization: header
                        .metadata
                        .get("quantization")
                        .cloned()
                        .unwrap_or_else(|| "f32".to_string()),
                    tensor_count: header.tensors.len(),
                })
                .into_response()
            }
            Err(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR
                .into_response(),
        },
        Err(_) => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

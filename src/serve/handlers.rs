use std::convert::Infallible;
use std::sync::Arc;

use axum::{Json, extract::State, response::sse::{Event, Sse}};
use tokio_stream::wrappers::ReceiverStream;

use super::{AppState, ChatRequest, ChatResponse, CompleteRequest, CompleteResponse, InferRequest, InferResponse, Message, ModelInfo, StreamChunk};
use super::infer::{run_inference, run_text_inference};

pub(super) async fn infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> Json<InferResponse> {
    if !req.inputs.is_empty() {
        let start = std::time::Instant::now();
        let outs: Vec<Vec<f32>> = req
            .inputs
            .iter()
            .map(|inp| run_inference(&state, inp))
            .collect();
        if state.profile {
            eprintln!("  batch infer: {} items in {:.3} ms", outs.len(), start.elapsed().as_secs_f64() * 1000.0);
        }
        return Json(InferResponse {
            output: None,
            outputs: Some(outs),
        });
    }

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

pub(super) async fn model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {
    Json(ModelInfo {
        name: state.name.clone(),
        architecture: state.architecture.clone(),
        total_params: state.total_params,
        total_bytes: state.total_bytes,
        tensors: state.tensor_names.clone(),
    })
}

pub(super) async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let prompt = req
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let output = run_text_inference(&state, &prompt);
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
    let output = run_text_inference(&state, &req.prompt);
    Json(CompleteResponse { completion: output })
}

pub(super) async fn chat_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let prompt = req
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let output = run_text_inference(&state, &prompt);

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(4);

    tokio::spawn(async move {
        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::to_string(&StreamChunk {
                    delta: output,
                    done: true,
                })
                .unwrap(),
            )))
            .await;
    });

    Sse::new(ReceiverStream::new(rx))
}

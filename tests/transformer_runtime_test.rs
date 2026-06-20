mod common;

use std::net::SocketAddr;

use modelc::runtime::serve::Runtime;
use modelc::runtime::transformer;
use modelc::serve::run_server;

const GPT2_HIDDEN: usize = 12;
const GPT2_VOCAB: usize = 10;
const LLAMA_HIDDEN: usize = 12;
const LLAMA_VOCAB: usize = 10;

#[test]
fn forward_gpt2_produces_vocab_sized_logits() {
    let model = common::create_gpt2_test_model();
    let runtime = Runtime::from_raw(&model.tensors);

    let input = vec![0.1f32; GPT2_HIDDEN];
    let out = transformer::forward_gpt2(&runtime, &input, false).expect("gpt2 forward");

    // Tied `wte` output projection → one logit per vocab entry.
    assert_eq!(out.len(), GPT2_VOCAB);
    assert!(out.iter().all(|v| v.is_finite()), "non-finite logits");
    // Must be a real transform, not an echo of the (12-dim) input.
    assert_ne!(out, input);
    assert!(out.iter().any(|v| *v != 0.0), "output is all zeros");
}

#[test]
fn forward_llama_produces_vocab_sized_logits() {
    let model = common::create_llama_test_model();
    let runtime = Runtime::from_raw(&model.tensors);

    let input = vec![0.05f32; LLAMA_HIDDEN];
    let out = transformer::forward_llama(&runtime, &input, false).expect("llama forward");

    assert_eq!(out.len(), LLAMA_VOCAB);
    assert!(out.iter().all(|v| v.is_finite()));
    assert_ne!(out, input);
    assert!(out.iter().any(|v| *v != 0.0));
}

#[test]
fn forward_gpt2_is_deterministic() {
    let model = common::create_gpt2_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    let input = vec![0.2f32; GPT2_HIDDEN];

    let a = transformer::forward_gpt2(&runtime, &input, false).unwrap();
    let b = transformer::forward_gpt2(&runtime, &input, false).unwrap();
    assert_eq!(a, b);
}

#[test]
fn forward_is_input_sensitive() {
    let model = common::create_gpt2_test_model();
    let runtime = Runtime::from_raw(&model.tensors);

    let out_a = transformer::forward_gpt2(&runtime, &[0.1f32; GPT2_HIDDEN], false).unwrap();
    let out_b = transformer::forward_gpt2(&runtime, &[0.9f32; GPT2_HIDDEN], false).unwrap();
    assert_ne!(out_a, out_b, "forward must respond to input changes");
}

#[test]
fn forward_gpt2_returns_none_without_output_head() {
    let model = common::create_gpt2_test_model();
    let mut tensors = model.tensors.clone();
    // Remove both possible output projections (this fixture has no lm_head, only tied wte).
    tensors.remove("transformer.wte.weight");

    let runtime = Runtime::from_raw(&tensors);
    let input = vec![0.0f32; GPT2_HIDDEN];
    assert!(transformer::forward_gpt2(&runtime, &input, false).is_none());
}

#[test]
fn forward_llama_returns_none_without_output_head() {
    let model = common::create_llama_test_model();
    let mut tensors = model.tensors.clone();
    tensors.remove("lm_head.weight");
    tensors.remove("model.embed_tokens.weight");

    let runtime = Runtime::from_raw(&tensors);
    let input = vec![0.0f32; LLAMA_HIDDEN];
    assert!(transformer::forward_llama(&runtime, &input, false).is_none());
}

// ---------------------------------------------------------------------------
// End-to-end: `modelc run` must serve real transformer inference over HTTP.
// ---------------------------------------------------------------------------

/// Grab a free ephemeral port by briefly binding to `:0`, then releasing it.
fn ephemeral_addr() -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

/// Retry a request against the freshly booted server until it responds (it may still be
/// binding the listener when we first try).
fn wait_for_server(url: &str) -> serde_json::Value {
    for _ in 0..50 {
        if let Ok(resp) = ureq::get(url).call()
            && let Ok(bytes) = read_body(resp)
            && let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes)
        {
            return val;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("server never came up at {url}");
}

fn read_body(resp: ureq::http::Response<ureq::Body>) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut buf = Vec::new();
    resp.into_body().into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

fn post_json(url: &str, body: &serde_json::Value) -> serde_json::Value {
    let body_str = serde_json::to_string(body).unwrap();
    let resp = ureq::post(url)
        .content_type("application/json")
        .send(&body_str)
        .expect("POST failed");
    let bytes = read_body(resp).expect("read body");
    serde_json::from_slice(&bytes).expect("parse json")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_gpt2_inference() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    // Boot the runtime server in the background.
    tokio::spawn(async move {
        let _ = run_server(model, addr, false).await;
    });

    let info = wait_for_server(&format!("{base}/info"));
    assert_eq!(info["architecture"], "gpt2");

    // /infer with a hidden-sized vector must return vocab-sized logits, not an echo.
    let body = serde_json::json!({ "input": vec![0.1f32; GPT2_HIDDEN] });
    let val = post_json(&format!("{base}/infer"), &body);
    let out = val["output"].as_array().expect("output array");
    assert_eq!(out.len(), GPT2_VOCAB, "expected vocab-sized logits");
    assert!(out.iter().any(|v| v.as_f64().unwrap_or(0.0) != 0.0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_resizes_oversized_infer_input() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(model, addr, false).await;
    });

    wait_for_server(&format!("{base}/info"));

    // An input longer than hidden must be gracefully truncated (not a 500).
    let oversized: Vec<f32> = vec![0.3; GPT2_HIDDEN * 4];
    let body = serde_json::json!({ "input": oversized });
    let val = post_json(&format!("{base}/infer"), &body);
    assert_eq!(val["output"].as_array().unwrap().len(), GPT2_VOCAB);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_health() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(model, addr, false).await;
    });

    wait_for_server(&format!("{base}/info"));

    let val = wait_for_server(&format!("{base}/health"));
    assert_eq!(val["status"], "ok");
    assert_eq!(val["model"], "mini_gpt2");
    assert_eq!(val["architecture"], "gpt2");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_openai_models() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(model, addr, false).await;
    });

    wait_for_server(&format!("{base}/info"));

    let val = wait_for_server(&format!("{base}/v1/models"));
    assert_eq!(val["object"], "list");
    let data = val["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "model");
    assert_eq!(data[0]["owned_by"], "modelc");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_embeddings() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(model, addr, false).await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({ "input": "hello world" });
    let val = post_json(&format!("{base}/embeddings"), &body);

    let embedding = val["embedding"].as_array().expect("embedding array");
    assert_eq!(embedding.len(), GPT2_HIDDEN, "embedding must be hidden-size");
    assert!(embedding.iter().any(|v| v.as_f64().unwrap_or(0.0) != 0.0));
    assert_eq!(val["model"], "mini_gpt2");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_openai_chat_completions() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(model, addr, false).await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let val = post_json(&format!("{base}/v1/chat/completions"), &body);

    assert_eq!(val["object"], "chat.completion");
    let choices = val["choices"].as_array().expect("choices array");
    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0]["message"]["role"], "assistant");
    assert!(choices[0]["message"]["content"].as_str().is_some());
    assert!(val["usage"]["total_tokens"].as_u64().unwrap_or(0) > 0);
}

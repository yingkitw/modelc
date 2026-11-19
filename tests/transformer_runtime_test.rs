mod common;

use std::net::SocketAddr;

use common::http::{self, HttpError, Response};

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
        if let Ok(resp) = http::get(url).call() {
            if let Ok(bytes) = read_body(resp) {
                if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    return val;
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("server never came up at {url}");
}

fn read_body(resp: Response) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut buf = Vec::new();
    resp.into_body().into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

fn post_json(url: &str, body: &serde_json::Value) -> serde_json::Value {
    let body_str = serde_json::to_string(body).unwrap();
    let resp = http::post(url)
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
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
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
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
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
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
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
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
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
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({ "input": "hello world" });
    let val = post_json(&format!("{base}/embeddings"), &body);

    let embedding = val["embedding"].as_array().expect("embedding array");
    assert_eq!(
        embedding.len(),
        GPT2_HIDDEN,
        "embedding must be hidden-size"
    );
    assert!(embedding.iter().any(|v| v.as_f64().unwrap_or(0.0) != 0.0));
    assert_eq!(val["model"], "mini_gpt2");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_tokenize() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Single input → flat `tokens` array.
    let body = serde_json::json!({ "input": "hello world" });
    let val = post_json(&format!("{base}/tokenize"), &body);
    let tokens = val["tokens"].as_array().expect("tokens array");
    assert!(!tokens.is_empty(), "should tokenize non-empty text");
    assert_eq!(val["count"].as_u64().unwrap(), tokens.len() as u64);
    assert!(val.get("tokens_batch").is_none() || val["tokens_batch"].is_null());

    // Batch input → `tokens_batch` array of arrays, `count` is the total.
    let body = serde_json::json!({ "inputs": ["hello", "world"] });
    let val = post_json(&format!("{base}/tokenize"), &body);
    let batch = val["tokens_batch"].as_array().expect("tokens_batch array");
    assert_eq!(batch.len(), 2, "batch should have one token list per input");
    let total: u64 = batch
        .iter()
        .map(|t| t.as_array().map(|a| a.len() as u64).unwrap_or(0))
        .sum();
    assert_eq!(val["count"].as_u64().unwrap(), total);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_system_info() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let val = wait_for_server(&format!("{base}/v1/system"));
    assert_eq!(val["model"], "mini_gpt2");
    assert_eq!(val["architecture"], "gpt2");
    assert!(val["cpu_cores"].as_u64().unwrap() >= 1, "at least one core");
    assert!(!val["os"].as_str().unwrap().is_empty(), "os string");
    assert!(!val["cpu_arch"].as_str().unwrap().is_empty(), "arch string");
    assert_eq!(val["pointer_width"].as_u64().unwrap(), 64);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_openai_chat_completions() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
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
    // Without `logprobs` requested, the field must serialize to null.
    assert!(choices[0]["logprobs"].is_null());
    assert!(val["usage"]["total_tokens"].as_u64().unwrap_or(0) > 0);
    // `created` must be a plausible Unix timestamp (OpenAI protocol), not 0.
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let created = val["created"].as_u64().expect("created timestamp");
    assert!(
        created > 1_600_000_000 && created <= now_secs + 5,
        "created {created} should be a recent Unix timestamp (now={now_secs})"
    );
    // `system_fingerprint` identifies the backend build.
    let fp = val["system_fingerprint"].as_str().expect("fingerprint");
    assert!(fp.starts_with("fp_modelc_"), "unexpected fingerprint {fp}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_openai_chat_logprobs() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Request logprobs with the top-3 alternatives per position.
    let body = serde_json::json!({
        "model": "mini_gpt2",
        "messages": [{"role": "user", "content": "hello"}],
        "logprobs": true,
        "top_logprobs": 3,
        "max_tokens": 5
    });
    let val = post_json(&format!("{base}/v1/chat/completions"), &body);

    let logprobs = &val["choices"][0]["logprobs"];
    assert!(
        logprobs.is_object(),
        "logprobs should be an object when requested"
    );
    let content = logprobs["content"].as_array().expect("content array");
    assert!(
        !content.is_empty(),
        "should produce at least one token's logprobs"
    );
    assert!(content.len() <= 5, "should respect max_tokens");

    // Each entry must carry token/logprob/bytes/top_logprobs with the right shape.
    for entry in content {
        assert!(entry["token"].is_string(), "token is a string");
        assert!(entry["bytes"].is_array(), "bytes is an array");
        let lp = entry["logprob"].as_f64().expect("logprob is a number");
        assert!(
            lp <= 0.0,
            "logprob must be <= 0.0 (it is ln of a probability)"
        );
        let top = entry["top_logprobs"]
            .as_array()
            .expect("top_logprobs array");
        assert!(
            top.len() <= 3,
            "top_logprobs must respect the requested count"
        );
        // top_logprobs are sorted by descending probability (ascending ln-magnitude).
        let mut prev = f64::INFINITY;
        for alt in top {
            assert!(alt["token"].is_string());
            assert!(alt["bytes"].is_array());
            let p = alt["logprob"].as_f64().unwrap();
            assert!(p <= 0.0);
            assert!(p <= prev + 1e-6, "top_logprobs must be sorted descending");
            prev = p;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_openai_chat_logprobs_no_top() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // `logprobs: true` with no `top_logprobs` → entries present, top_logprobs empty.
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": true,
        "max_tokens": 2
    });
    let val = post_json(&format!("{base}/v1/chat/completions"), &body);

    let content = val["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("content array");
    for entry in content {
        assert!(
            entry["top_logprobs"]
                .as_array()
                .map(|a| a.is_empty())
                .unwrap_or(true),
            "top_logprobs must be empty when not requested"
        );
        assert!(entry["logprob"].as_f64().is_some());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_lora_unload_restores_base() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Unload with no LoRA loaded should still succeed (restores base).
    let body = serde_json::json!({});
    let val = post_json(&format!("{base}/lora/unload"), &body);
    assert!(val["message"].as_str().unwrap().contains("unloaded"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_lora_load_bad_path_returns_error() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({"path": "/nonexistent/lora.safetensors"});
    let val = post_json(&format!("{base}/lora/load"), &body);
    assert!(val["message"].as_str().unwrap().contains("Failed"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_metrics_returns_prometheus_text() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Make a chat request so metrics are populated.
    let chat_body = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2
    });
    let _ = post_json(&format!("{base}/chat"), &chat_body);

    // Fetch metrics in Prometheus text format.
    let mut resp = http::get(&format!("{base}/metrics"))
        .call()
        .expect("metrics request");
    let text = resp.body_mut().read_to_string().expect("read metrics body");

    assert!(
        text.contains("modelc_requests_total"),
        "should expose request counter"
    );
    assert!(
        text.contains("modelc_inference_duration_seconds_count"),
        "should expose histogram count"
    );
    assert!(
        text.contains("modelc_active_requests"),
        "should expose active request gauge"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_chat_accepts_json_schema() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // The test model is random so it won't produce valid JSON, but the endpoint
    // should accept the schema parameter and process it (retrying up to 3 times).
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "json_schema": {"type": "object"}
    });
    let val = post_json(&format!("{base}/chat"), &body);
    // Response should contain a message even if JSON validation ultimately fails.
    assert!(
        val["message"].is_object(),
        "response should have message field"
    );
    assert!(
        val["message"]["content"].as_str().is_some(),
        "response should have content string"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_openai_chat_stream() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
        "stream": true
    });
    let body_str = serde_json::to_string(&body).unwrap();

    let resp = http::post(&format!("{base}/v1/chat/completions"))
        .content_type("application/json")
        .send(&body_str)
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);

    let text = read_body(resp)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    assert!(
        text.contains("chat.completion.chunk"),
        "should emit OpenAI chunk objects"
    );
    assert!(text.contains("[DONE]"), "should end with [DONE]");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_openai_completions() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "prompt": "hello",
        "max_tokens": 4
    });
    let val = post_json(&format!("{base}/v1/completions"), &body);
    assert_eq!(val["object"].as_str(), Some("text_completion"));
    assert!(val["choices"].is_array(), "should have choices array");
    assert_eq!(val["choices"][0]["index"].as_i64(), Some(0));
    assert!(
        val["choices"][0]["text"].as_str().is_some(),
        "should have text"
    );
    assert_eq!(val["choices"][0]["finish_reason"].as_str(), Some("stop"));
    assert!(val["usage"]["prompt_tokens"].is_number());
    assert!(val["usage"]["completion_tokens"].is_number());
    // OpenAI protocol: `created` is a real Unix timestamp; `system_fingerprint`
    // identifies the backend build.
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let created = val["created"].as_u64().expect("created timestamp");
    assert!(
        created > 1_600_000_000 && created <= now_secs + 5,
        "created {created} should be a recent Unix timestamp (now={now_secs})"
    );
    assert!(
        val["system_fingerprint"]
            .as_str()
            .unwrap_or_default()
            .starts_with("fp_modelc_")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_completions_echo() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "prompt": "hello",
        "max_tokens": 4,
        "echo": true
    });
    let val = post_json(&format!("{base}/v1/completions"), &body);
    let text = val["choices"][0]["text"]
        .as_str()
        .expect("should have text");
    assert!(
        text.starts_with("hello"),
        "echo=true should prepend prompt to completion, got: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_completions_n() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "prompt": "hello",
        "max_tokens": 4,
        "n": 3
    });
    let val = post_json(&format!("{base}/v1/completions"), &body);
    let choices = val["choices"].as_array().expect("choices should be array");
    assert_eq!(choices.len(), 3, "n=3 should produce 3 choices");
    for (i, choice) in choices.iter().enumerate() {
        assert_eq!(
            choice["index"].as_i64(),
            Some(i as i64),
            "choice index should match"
        );
        assert!(
            choice["text"].as_str().is_some(),
            "each choice should have text"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_api_key_rejects_unauthenticated() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    let auth = modelc::serve::auth::AuthConfig::new(Some("secret".to_string()), None);
    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            Some(auth),
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Request without Authorization header → 401
    let status: u16 = match http::get(&format!("{base}/metrics")).call() {
        Ok(r) => r.status(),
        Err(HttpError::StatusCode(code)) => code,
        Err(_) => 0,
    };
    assert_eq!(status, 401, "unauthenticated request should be rejected");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_api_key_accepts_authenticated() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    let auth = modelc::serve::auth::AuthConfig::new(Some("secret".to_string()), None);
    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            Some(auth),
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Request with correct Bearer token → 200
    let resp = http::get(&format!("{base}/metrics"))
        .header("Authorization", "Bearer secret")
        .call()
        .expect("request should succeed");
    assert_eq!(
        resp.status(),
        200,
        "authenticated request should pass through"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_rate_limit_rejects_over_limit() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    // Allow only 1 request per minute.
    let auth = modelc::serve::auth::AuthConfig::new(None, Some(1));
    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            Some(auth),
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // First request should succeed.
    let resp1 = http::get(&format!("{base}/metrics"))
        .call()
        .expect("first request should succeed");
    assert_eq!(resp1.status(), 200, "first request should pass");

    // Second immediate request should be rate-limited (429).
    let status: u16 = match http::get(&format!("{base}/metrics")).call() {
        Ok(r) => r.status(),
        Err(HttpError::StatusCode(code)) => code,
        Err(_) => 0,
    };
    assert_eq!(status, 429, "second request should be rate limited");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn run_server_handles_concurrent_transformer_requests() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Fire two concurrent chat requests; both should complete successfully.
    let body_a = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4
    });
    let body_b = serde_json::json!({
        "messages": [{"role": "user", "content": "world"}],
        "max_tokens": 4
    });

    let base_a = base.clone();
    let base_b = base.clone();
    let (resp_a, resp_b) = tokio::join!(
        tokio::task::spawn_blocking(move || {
            http::post(&format!("{base_a}/chat"))
                .content_type("application/json")
                .send(&serde_json::to_string(&body_a).unwrap())
                .expect("concurrent request A should succeed")
        }),
        tokio::task::spawn_blocking(move || {
            http::post(&format!("{base_b}/chat"))
                .content_type("application/json")
                .send(&serde_json::to_string(&body_b).unwrap())
                .expect("concurrent request B should succeed")
        }),
    );

    let resp_a = resp_a.expect("spawn A should not panic");
    let resp_b = resp_b.expect("spawn B should not panic");

    assert_eq!(
        resp_a.status(),
        200,
        "concurrent request A should return 200"
    );
    assert_eq!(
        resp_b.status(),
        200,
        "concurrent request B should return 200"
    );

    let val_a: serde_json::Value =
        serde_json::from_reader(resp_a.into_body().as_reader()).expect("A should be JSON");
    let val_b: serde_json::Value =
        serde_json::from_reader(resp_b.into_body().as_reader()).expect("B should be JSON");

    assert!(
        val_a["message"]["content"].as_str().is_some(),
        "A should have content"
    );
    assert!(
        val_b["message"]["content"].as_str().is_some(),
        "B should have content"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn run_server_max_concurrent_rejects_with_503() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            Some(1),
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Fire three concurrent chat requests with max_concurrent=1;
    // at least one should be rejected with 503.
    let body_str = serde_json::to_string(&serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4
    }))
    .unwrap();

    let (ra, rb, rc) = tokio::join!(
        tokio::task::spawn_blocking({
            let body = body_str.clone();
            let url = format!("{base}/chat");
            move || {
                http::post(&url)
                    .content_type("application/json")
                    .send(&body)
            }
        }),
        tokio::task::spawn_blocking({
            let body = body_str.clone();
            let url = format!("{base}/chat");
            move || {
                http::post(&url)
                    .content_type("application/json")
                    .send(&body)
            }
        }),
        tokio::task::spawn_blocking({
            let body = body_str.clone();
            let url = format!("{base}/chat");
            move || {
                http::post(&url)
                    .content_type("application/json")
                    .send(&body)
            }
        }),
    );

    let statuses = [
        match ra.expect("spawn should not panic") {
            Ok(r) => r.status(),
            Err(HttpError::StatusCode(code)) => code,
            Err(_) => 0,
        },
        match rb.expect("spawn should not panic") {
            Ok(r) => r.status(),
            Err(HttpError::StatusCode(code)) => code,
            Err(_) => 0,
        },
        match rc.expect("spawn should not panic") {
            Ok(r) => r.status(),
            Err(HttpError::StatusCode(code)) => code,
            Err(_) => 0,
        },
    ];
    let ok_count = statuses.iter().filter(|&&s| s == 200).count();
    let rejected_count = statuses.iter().filter(|&&s| s == 503).count();

    assert!(
        ok_count >= 1,
        "at least one request should succeed (200), got {:?}",
        statuses
    );
    assert!(
        rejected_count >= 1,
        "at least one request should be rejected with 503, got {:?}",
        statuses
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_openai_v1_embeddings() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Single input, float format (default).
    let body = serde_json::json!({
        "input": "hello",
        "encoding_format": "float"
    });
    let val = post_json(&format!("{base}/v1/embeddings"), &body);
    assert_eq!(val["object"].as_str(), Some("list"));
    let data = val["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"].as_str(), Some("embedding"));
    assert!(
        data[0]["embedding"].is_array(),
        "float embedding should be array"
    );
    let dims = data[0]["dimensions"].as_u64().expect("dimensions present") as usize;
    let emb = data[0]["embedding"].as_array().unwrap();
    assert_eq!(emb.len(), dims, "dimensions should match embedding length");
    assert!(val["usage"]["total_tokens"].is_number());

    // Batch input, base64 format.
    let body = serde_json::json!({
        "input": ["hello", "world"],
        "encoding_format": "base64"
    });
    let val = post_json(&format!("{base}/v1/embeddings"), &body);
    let data = val["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
    assert!(
        data[0]["embedding"].as_str().is_some(),
        "base64 embedding should be a string"
    );
    assert_eq!(data[0]["index"].as_u64(), Some(0));
    assert_eq!(data[1]["index"].as_u64(), Some(1));
    assert!(data[0]["dimensions"].as_u64().is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_api_version() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let resp = http::get(&format!("{base}/api/version"))
        .call()
        .expect("version request should succeed");
    let bytes = read_body(resp).expect("read body");
    let val: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");
    assert!(
        val["version"].as_str().is_some(),
        "version should be present"
    );
    assert!(
        val["git_sha"].as_str().is_some(),
        "git_sha should be present"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_props() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let resp = http::get(&format!("{base}/props"))
        .call()
        .expect("props request should succeed");
    let bytes = read_body(resp).expect("read body");
    let val: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");
    assert!(val["model"].as_str().is_some(), "model should be present");
    assert!(
        val["architecture"].as_str().is_some(),
        "architecture should be present"
    );
    assert!(
        val["total_params"].is_number(),
        "total_params should be present"
    );
    assert!(
        val["default_generation"]["max_tokens"].is_number(),
        "default_generation.max_tokens should be present"
    );
    assert!(
        val["default_generation"]["temperature"].is_number(),
        "default_generation.temperature should be present"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_detokenize() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // First tokenize to get valid token IDs, then detokenize to verify round-trip.
    let tok_body = serde_json::json!({ "input": "hello" });
    let tok_val = post_json(&format!("{base}/tokenize"), &tok_body);
    let tokens = tok_val["tokens"]
        .as_array()
        .expect("tokens should be array");
    assert!(!tokens.is_empty(), "should have at least one token");

    let detok_body = serde_json::json!({ "tokens": tokens });
    let detok_val = post_json(&format!("{base}/detokenize"), &detok_body);
    let text = detok_val["text"].as_str().expect("text should be present");
    assert!(
        text.contains("hello"),
        "detokenized text should contain original input, got: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_infill() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "prefix": "fn main() {",
        "suffix": "}",
        "max_tokens": 4,
        "temperature": 0.0
    });
    let val = post_json(&format!("{base}/infill"), &body);
    let completion = val["completion"]
        .as_str()
        .expect("infill response must include completion");
    // Tiny GPT-2 fixture still runs the FIM path; output may be empty if a stop
    // sequence matches immediately, but the field must be present and UTF-8.
    let _ = completion;

    let with_prompt = serde_json::json!({
        "prefix": "let x = ",
        "suffix": ";",
        "prompt": "complete the rust snippet",
        "max_tokens": 2,
        "seed": 1
    });
    let val = post_json(&format!("{base}/infill"), &with_prompt);
    assert!(
        val["completion"].is_string(),
        "prompt+prefix infill should still return a completion string"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_completions_suffix() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "prompt": "hello",
        "max_tokens": 4,
        "suffix": " END"
    });
    let val = post_json(&format!("{base}/v1/completions"), &body);
    let text = val["choices"][0]["text"]
        .as_str()
        .expect("should have text");
    assert!(
        text.ends_with(" END"),
        "suffix should be appended to completion, got: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_chat_completions_n() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
        "n": 3
    });
    let val = post_json(&format!("{base}/v1/chat/completions"), &body);
    let choices = val["choices"].as_array().expect("choices should be array");
    assert_eq!(choices.len(), 3, "n=3 should produce 3 choices");
    for (i, choice) in choices.iter().enumerate() {
        assert_eq!(
            choice["index"].as_i64(),
            Some(i as i64),
            "choice index should match"
        );
        assert!(
            choice["message"]["content"].as_str().is_some(),
            "each choice should have message content"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_chat_stream_include_usage() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let body_str = serde_json::to_string(&body).unwrap();

    let resp = http::post(&format!("{base}/v1/chat/completions"))
        .content_type("application/json")
        .send(&body_str)
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);

    let text = read_body(resp)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    assert!(
        text.contains("chat.completion.chunk"),
        "should emit OpenAI chunk objects"
    );
    assert!(text.contains("[DONE]"), "should end with [DONE]");
    assert!(
        text.contains("\"usage\""),
        "should include usage object when stream_options.include_usage is true"
    );
    assert!(
        text.contains("\"prompt_tokens\""),
        "usage should contain prompt_tokens"
    );
    assert!(
        text.contains("\"completion_tokens\""),
        "usage should contain completion_tokens"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_completions_stream_include_usage() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "prompt": "hello",
        "max_tokens": 3,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let body_str = serde_json::to_string(&body).unwrap();

    let resp = http::post(&format!("{base}/v1/completions"))
        .content_type("application/json")
        .send(&body_str)
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);

    let text = read_body(resp)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    assert!(
        text.contains("text_completion"),
        "should emit text_completion chunk objects"
    );
    assert!(text.contains("[DONE]"), "should end with [DONE]");
    assert!(
        text.contains("\"usage\""),
        "should include usage object when stream_options.include_usage is true"
    );
    assert!(
        text.contains("\"prompt_tokens\""),
        "usage should contain prompt_tokens"
    );
    assert!(
        text.contains("\"completion_tokens\""),
        "usage should contain completion_tokens"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_retrieve_model() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let list = http::get(&format!("{base}/v1/models"))
        .call()
        .expect("list models should succeed");
    let list_bytes = read_body(list).expect("read body");
    let list_val: serde_json::Value = serde_json::from_slice(&list_bytes).expect("parse json");
    let model_id = list_val["data"][0]["id"]
        .as_str()
        .expect("model id should be present");

    let resp = http::get(&format!("{base}/v1/models/{model_id}"))
        .call()
        .expect("retrieve model should succeed");
    let bytes = read_body(resp).expect("read body");
    let val: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");
    assert_eq!(val["id"].as_str(), Some(model_id));
    assert_eq!(val["object"].as_str(), Some("model"));
    assert_eq!(val["owned_by"].as_str(), Some("modelc"));

    let not_found = http::get(&format!("{base}/v1/models/nonexistent")).call();
    assert!(not_found.is_err(), "non-existent model should return error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_completions_best_of() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "prompt": "hello",
        "max_tokens": 4,
        "best_of": 3,
        "n": 1
    });
    let val = post_json(&format!("{base}/v1/completions"), &body);
    let choices = val["choices"].as_array().expect("choices should be array");
    assert_eq!(
        choices.len(),
        1,
        "best_of=3, n=1 should return 1 choice (the best)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_chat_with_user_field() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // The user field should be accepted without error.
    let body = serde_json::json!({
        "model": "mini_gpt2",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
        "user": "test-user-123"
    });
    let val = post_json(&format!("{base}/v1/chat/completions"), &body);
    assert!(
        val["choices"].as_array().is_some(),
        "should return choices even with user field"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_chat_response_format_json_schema() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 10,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "number"}
                    },
                    "required": ["answer"]
                },
                "strict": true
            }
        }
    });
    let val = post_json(&format!("{base}/v1/chat/completions"), &body);
    assert!(
        val["choices"].as_array().is_some(),
        "should return choices with json_schema response_format"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_chat_tool_choice_none() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}}}],
        "tool_choice": "none"
    });
    let val = post_json(&format!("{base}/v1/chat/completions"), &body);
    let choices = val["choices"].as_array().expect("choices should be array");
    assert_eq!(choices.len(), 1);
    // With tool_choice: none, finish_reason should be "stop", not "tool_calls"
    assert_eq!(
        choices[0]["finish_reason"].as_str(),
        Some("stop"),
        "tool_choice=none should not produce tool_calls"
    );
    assert!(
        choices[0]["message"]["tool_calls"].is_null(),
        "tool_choice=none should not return tool_calls"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_chat_tool_choice_required() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 5,
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}}}],
        "tool_choice": "required"
    });
    let val = post_json(&format!("{base}/v1/chat/completions"), &body);
    let choices = val["choices"].as_array().expect("choices should be array");
    assert_eq!(choices.len(), 1);
    // Should still get a valid response (tool injection happened)
    assert!(
        choices[0]["message"]["content"].as_str().is_some()
            || choices[0]["message"]["tool_calls"].is_array(),
        "should return either content or tool_calls"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_chat_with_name_field() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // The name field on messages should be accepted without error.
    let body = serde_json::json!({
        "model": "mini_gpt2",
        "messages": [{"role": "user", "content": "hello", "name": "alice"}],
        "max_tokens": 3
    });
    let val = post_json(&format!("{base}/v1/chat/completions"), &body);
    assert!(
        val["choices"].as_array().is_some(),
        "should return choices even with name field"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_chat_max_completion_tokens() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 3
    });
    let val = post_json(&format!("{base}/v1/chat/completions"), &body);
    let choices = val["choices"].as_array().expect("choices should be array");
    assert_eq!(choices.len(), 1);
    // Verify usage reflects the max_completion_tokens limit
    let usage = &val["usage"];
    assert!(
        usage["completion_tokens"].as_u64().is_some(),
        "usage should contain completion_tokens"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_completions_batch_prompt() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "model": "mini_gpt2",
        "prompt": ["hello", "world"],
        "max_tokens": 3
    });
    let val = post_json(&format!("{base}/v1/completions"), &body);
    let choices = val["choices"].as_array().expect("choices should be array");
    assert_eq!(
        choices.len(),
        2,
        "batch prompt with 2 strings should return 2 choices"
    );
    assert_eq!(choices[0]["index"].as_u64(), Some(0));
    assert_eq!(choices[1]["index"].as_u64(), Some(1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_reranking() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let body = serde_json::json!({
        "query": "what is rust",
        "documents": [
            "Rust is a systems programming language",
            "Python is a scripting language",
            "Rust is memory-safe without garbage collection"
        ],
        "top_n": 2
    });
    let val = post_json(&format!("{base}/reranking"), &body);
    assert!(val["model"].as_str().is_some(), "model field present");
    let results = val["results"].as_array().expect("results should be array");
    assert_eq!(results.len(), 2, "top_n=2 should return 2 results");
    assert!(
        results[0]["index"].as_u64().is_some(),
        "index field present"
    );
    assert!(
        results[0]["relevance_score"].as_f64().is_some(),
        "relevance_score field present"
    );
    assert!(
        results[0]["document"]["text"].as_str().is_some(),
        "document.text field present"
    );
    assert!(
        results[0]["relevance_score"].as_f64() >= results[1]["relevance_score"].as_f64(),
        "results should be sorted by descending relevance"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_web_ui() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let resp = http::get(&format!("{base}/"))
        .call()
        .expect("GET / should succeed");
    let bytes = read_body(resp).expect("read body");
    let html = String::from_utf8_lossy(&bytes);
    assert!(html.contains("<!DOCTYPE html>"), "should return HTML");
    assert!(html.contains("modelc"), "should contain modelc title");
    assert!(
        html.contains("/v1/chat/completions"),
        "should reference the chat completions endpoint"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_api_tags() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    // Install a model so /api/tags has an entry with full Ollama fields.
    let tmp = common::tempfile().unwrap();
    modelc::pack::pack(&model, tmp.path(), false).unwrap();
    let test_name = "__modelc_test_api_tags__";
    modelc::store::install(tmp.path(), test_name).unwrap();

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    let resp = http::get(&format!("{base}/api/tags"))
        .call()
        .expect("GET /api/tags should succeed");
    let bytes = read_body(resp).expect("read body");
    let val: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");
    let models = val["models"].as_array().expect("models should be an array");
    let entry = models
        .iter()
        .find(|m| m["name"].as_str() == Some(test_name))
        .expect("installed model should appear in /api/tags");

    // Ollama ecosystem fields (learned from modelc-web): real clients parse these.
    assert!(entry["model"].as_str().is_some(), "model field present");
    assert!(
        entry["modified_at"].as_str().is_some_and(|s| s.ends_with('Z')),
        "modified_at RFC3339 present"
    );
    assert!(
        entry["digest"].as_str().is_some_and(|s| s.starts_with("sha256:")),
        "digest sha256:<hex> present"
    );
    let details = &entry["details"];
    for key in ["format", "family", "families", "parameter_size", "quantization_level"] {
        assert!(details[key].is_string() || details[key].is_array(),
            "details.{key} present");
    }
    modelc::store::remove_model(test_name, false, true).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_server_serves_api_show() {
    let model = common::create_gpt2_test_model();
    let addr = ephemeral_addr();
    let base = format!("http://{addr}");

    // Install a model into the store so /api/show can find it.
    let tmp = common::tempfile().unwrap();
    modelc::pack::pack(&model, tmp.path(), false).unwrap();
    let test_name = "__modelc_test_api_show__";
    modelc::store::install(tmp.path(), test_name).unwrap();

    tokio::spawn(async move {
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Ollama clients send {"model": "..."}; legacy tooling sends {"name": "..."}.
    let body = serde_json::json!({ "model": test_name });
    let val = post_json(&format!("{base}/api/show"), &body);
    assert!(val["name"].as_str().is_some(), "name field present");
    assert!(
        val["architecture"].as_str().is_some(),
        "architecture field present"
    );
    assert!(
        val["size_bytes"].as_u64().is_some(),
        "size_bytes field present"
    );
    assert!(
        val["tensor_count"].as_u64().is_some(),
        "tensor_count field present"
    );
    // Ollama ecosystem fields.
    assert!(val["model"].as_str().is_some(), "model field present");
    assert!(
        val["digest"].as_str().is_some_and(|s| s.starts_with("sha256:")),
        "digest sha256:<hex> present"
    );
    let details = &val["details"];
    for key in ["format", "family", "families", "parameter_size", "quantization_level"] {
        assert!(details[key].is_string() || details[key].is_array(),
            "details.{key} present");
    }
    let info = &val["model_info"];
    assert!(info["general.architecture"].is_string(), "model_info.architecture");
    assert!(info["general.parameter_count"].is_number(), "model_info.parameter_count");
    assert!(info["general.file_type"].is_number(), "model_info.file_type");
    assert!(val["capabilities"].is_array(), "capabilities array present");

    // Legacy {"name": ...} still works.
    let val2 = post_json(&format!("{base}/api/show"), &serde_json::json!({ "name": test_name }));
    assert!(val2["model"].as_str().is_some(), "name-keyed request still works");

    // Clean up.
    modelc::store::remove_model(test_name, false, true).unwrap();
}

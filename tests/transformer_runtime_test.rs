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
        let _ = run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
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
    let mut resp = ureq::get(&format!("{base}/metrics"))
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

    let resp = ureq::post(&format!("{base}/v1/chat/completions"))
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
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Request without Authorization header → 401
    let status: u16 = match ureq::get(&format!("{base}/metrics")).call() {
        Ok(r) => r.status().as_u16(),
        Err(ureq::Error::StatusCode(code)) => code,
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
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // Request with correct Bearer token → 200
    let resp = ureq::get(&format!("{base}/metrics"))
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
        )
        .await;
    });

    wait_for_server(&format!("{base}/info"));

    // First request should succeed.
    let resp1 = ureq::get(&format!("{base}/metrics"))
        .call()
        .expect("first request should succeed");
    assert_eq!(resp1.status(), 200, "first request should pass");

    // Second immediate request should be rate-limited (429).
    let status: u16 = match ureq::get(&format!("{base}/metrics")).call() {
        Ok(r) => r.status().as_u16(),
        Err(ureq::Error::StatusCode(code)) => code,
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
            ureq::post(&format!("{base_a}/chat"))
                .content_type("application/json")
                .send(&serde_json::to_string(&body_a).unwrap())
                .expect("concurrent request A should succeed")
        }),
        tokio::task::spawn_blocking(move || {
            ureq::post(&format!("{base_b}/chat"))
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

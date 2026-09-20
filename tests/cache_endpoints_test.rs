//! Integration tests for the prefix-cache management endpoints:
//! `GET /api/cache/stats` and `POST /api/cache/clear`. Both routes are
//! exempt from auth/backpressure so dashboards can probe them under load.

mod common;

use common::http;

fn read_body(resp: http::Response) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut buf = Vec::new();
    resp.into_body().into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

fn get_json(url: &str) -> serde_json::Value {
    let resp = http::get(url).call().expect("GET failed");
    let bytes = read_body(resp).expect("read body");
    serde_json::from_slice(&bytes).expect("parse json")
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

fn wait_for_server(url: &str) {
    for _ in 0..50 {
        if http::get(url).call().is_ok() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("server never came up at {url}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cache_stats_reports_empty_state() {
    let model = common::create_test_model();
    // Bind an ephemeral port first so we know the URL.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let base = format!("http://{addr}");
    tokio::spawn(async move {
        let _ = modelc::serve::run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/health"));
    let body = get_json(&format!("{base}/api/cache/stats"));
    assert_eq!(body["entries"], 0);
    assert!(body["capacity"].as_u64().unwrap() > 0);
    assert_eq!(body["total_tokens"], 0);
    assert_eq!(body["total_kv_bytes"], 0);
    assert_eq!(body["utilization"], 0.0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cache_clear_succeeds_and_reports_zero() {
    let model = common::create_test_model();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let base = format!("http://{addr}");
    tokio::spawn(async move {
        let _ = modelc::serve::run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/health"));
    let body = post_json(&format!("{base}/api/cache/clear"), &serde_json::json!({}));
    assert_eq!(body["cleared_entries"], 0);
    assert!(body["message"].as_str().unwrap().contains("cleared"));

    // Stats still report the (now-empty) cache cleanly.
    let stats = get_json(&format!("{base}/api/cache/stats"));
    assert_eq!(stats["entries"], 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cache_endpoints_coexist_with_inference() {
    let model = common::create_test_model();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let base = format!("http://{addr}");
    tokio::spawn(async move {
        let _ = modelc::serve::run_server(
            model,
            addr,
            false,
            modelc::generate::GenerationConfig::default(),
            None,
            None,
        )
        .await;
    });

    wait_for_server(&format!("{base}/health"));

    // Run inference (MLP model → prefix cache bypassed but must not panic).
    let resp = post_json(
        &format!("{base}/complete"),
        &serde_json::json!({
            "prompt": "hello world",
            "max_tokens": 4,
            "temperature": 0.0,
        }),
    );
    assert!(resp["completion"].is_string());

    // Cache endpoints still respond cleanly.
    let stats = get_json(&format!("{base}/api/cache/stats"));
    assert!(stats["capacity"].as_u64().unwrap() > 0);

    let cleared = post_json(&format!("{base}/api/cache/clear"), &serde_json::json!({}));
    assert_eq!(cleared["cleared_entries"], 0);
}

//! End-to-end `modelc serve` tests: spawn the real binary as a child process.
//!
//! - `--stdio`: drives a full MCP-style JSON-RPC session (initialize →
//!   notifications/initialized → tools/list → tools/call → resources/list →
//!   resources/read) through pipes.
//! - default (HTTP) mode: binds a port and answers `/info` with the same shape
//!   `modelc run` serves — they share the router.

mod common;

use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::process::{Command, Stdio};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_modelc")
}

/// Pack the shared `weight`/`bias` MLP fixture into a temp artifact.
/// The TempDir must outlive the spawned server.
fn pack_fixture() -> (modelc::tempdir::TempDir, std::path::PathBuf) {
    let dir = modelc::tempdir::tempdir().expect("tempdir");
    let artifact = dir.path().join("serve_fixture.modelc");
    modelc::pack::pack(&common::create_test_model(), &artifact, false).expect("pack fixture");
    (dir, artifact)
}

fn send_msg(stdin: &mut std::process::ChildStdin, value: &Value) {
    writeln!(stdin, "{value}").unwrap();
    stdin.flush().unwrap();
}

fn read_msg(reader: &mut std::io::BufReader<std::process::ChildStdout>) -> Value {
    let mut line = String::new();
    reader.read_line(&mut line).expect("read response line");
    serde_json::from_str(line.trim()).expect("valid JSON-RPC response")
}

#[test]
fn serve_stdio_speaks_mcp_jsonrpc() {
    let (_dir, artifact) = pack_fixture();

    let mut child = Command::new(bin())
        .args(["serve", artifact.to_str().unwrap(), "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn modelc serve --stdio");
    let mut stdin = child.stdin.take().expect("child stdin");
    let stdout = child.stdout.take().expect("child stdout");
    let mut reader = std::io::BufReader::new(stdout);

    // 1. initialize → server info + capabilities.
    send_msg(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":1,"method":"initialize",
                "params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"modelc-test","version":"0"}}}),
    );
    let resp = read_msg(&mut reader);
    assert_eq!(resp["id"], 1);
    assert_eq!(resp["result"]["serverInfo"]["name"], "modelc");
    assert_eq!(resp["result"]["protocolVersion"], "2025-06-18");

    // 2. notification — must not produce a response line.
    send_msg(&mut stdin, &json!({"jsonrpc":"2.0","method":"notifications/initialized"}));

    // 3. tools/list → the inference tools plus rerank.
    send_msg(&mut stdin, &json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}));
    let resp = read_msg(&mut reader);
    let tools = resp["result"]["tools"].as_array().expect("tools array");
    let names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().expect("tool name"))
        .collect();
    assert_eq!(names, vec!["chat", "complete", "embeddings", "rerank"]);

    // 4. tools/call complete on the fixture model → float array, no error.
    send_msg(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":3,"method":"tools/call",
                "params":{"name":"complete","arguments":{"prompt":"AB"}}}),
    );
    let resp = read_msg(&mut reader);
    assert_eq!(resp["id"], 3);
    assert_eq!(resp["result"]["isError"], false);
    let text = resp["result"]["content"][0]["text"].as_str().expect("text content");
    let output: Vec<f32> = serde_json::from_str(text).expect("mlp output is a float array");
    assert_eq!(output.len(), 2, "weight [2,3] projects 3 inputs to 2 outputs");

    // 5. Unknown tool → MCP error *result*, not a protocol error.
    send_msg(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"warp-speed"}}),
    );
    let resp = read_msg(&mut reader);
    assert_eq!(resp["result"]["isError"], true);
    assert!(resp["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("Unknown tool"));

    // 6. resources/list → the read-only model resources.
    send_msg(&mut stdin, &json!({"jsonrpc":"2.0","id":5,"method":"resources/list"}));
    let resp = read_msg(&mut reader);
    let resources = resp["result"]["resources"].as_array().expect("resources array");
    let uris: Vec<&str> = resources
        .iter()
        .map(|r| r["uri"].as_str().expect("resource uri"))
        .collect();
    assert_eq!(
        uris,
        vec![
            "modelc://model",
            "modelc://model/tensors",
            "modelc://model/tokenizer"
        ]
    );

    // 7. resources/read → the model overview, as JSON inside the text content.
    send_msg(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":6,"method":"resources/read",
                "params":{"uri":"modelc://model"}}),
    );
    let resp = read_msg(&mut reader);
    let contents = resp["result"]["contents"].as_array().expect("contents");
    assert_eq!(contents[0]["uri"], "modelc://model");
    let overview: Value =
        serde_json::from_str(contents[0]["text"].as_str().expect("text")).expect("overview json");
    assert_eq!(overview["name"], "test_model");
    assert_eq!(overview["architecture"], "mlp");
    assert_eq!(overview["tensor_count"], 2);

    // 8. The tensor table lists every tensor of the artifact.
    send_msg(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":7,"method":"resources/read",
                "params":{"uri":"modelc://model/tensors"}}),
    );
    let resp = read_msg(&mut reader);
    let table: Value = serde_json::from_str(
        resp["result"]["contents"][0]["text"].as_str().expect("text"),
    )
    .expect("tensor table json");
    let tensors = table["tensors"].as_array().expect("tensors");
    let names: Vec<&str> = tensors
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["bias", "weight"], "sorted by name");

    // 9. tools/call rerank → the same ranking the HTTP endpoint returns.
    send_msg(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":10,"method":"tools/call",
                "params":{"name":"rerank","arguments":{
                    "query":"capital cities",
                    "documents":["washington","paris","tokyo"],
                    "top_n":2,
                    "return_documents":true}}}),
    );
    let resp = read_msg(&mut reader);
    assert_eq!(resp["result"]["isError"], false);
    let envelope: Value =
        serde_json::from_str(resp["result"]["content"][0]["text"].as_str().expect("text"))
            .expect("rerank envelope");
    let results = envelope["results"].as_array().expect("results");
    assert_eq!(results.len(), 2, "top_n truncates");
    for r in results {
        let score = r["relevance_score"].as_f64().expect("score");
        assert!((0.0..=1.0).contains(&score), "score in [0, 1]: {score}");
        assert!(r["document"]["text"].as_str().is_some());
    }

    // 10. An unknown resource URI → JSON-RPC error -32002 (resource not found).
    send_msg(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":11,"method":"resources/read",
                "params":{"uri":"modelc://nope"}}),
    );
    let resp = read_msg(&mut reader);
    assert_eq!(resp["error"]["code"], -32002);

    // 11. Unknown method → JSON-RPC protocol error.
    send_msg(&mut stdin, &json!({"jsonrpc":"2.0","id":12,"method":"prompts/list"}));
    let resp = read_msg(&mut reader);
    assert_eq!(resp["error"]["code"], -32601);

    child.kill().expect("kill child");
    child.wait().expect("reap child");
}

#[test]
fn serve_http_mode_serves_the_run_router() {
    let (_dir, artifact) = pack_fixture();

    // Unique per test-run port to avoid collisions with parallel suites.
    let port: u16 = 18000 + (std::process::id() % 2000) as u16;
    let mut child = Command::new(bin())
        .args(["serve", artifact.to_str().unwrap(), "-p", &port.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn modelc serve (http)");

    let mut info: Option<String> = None;
    for _ in 0..100 {
        if let Ok(resp) = common::http::get(&format!("http://127.0.0.1:{port}/info")).send("") {
            let body = common::http::read_body(resp).expect("read /info body");
            info = Some(String::from_utf8(body).expect("utf8 body"));
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    // Kill before asserting so a failed assertion can't leak the child process.
    child.kill().expect("kill child");
    child.wait().expect("reap child");

    let info = info.expect("serve (http) did not come up before the poll budget ran out");
    assert!(info.contains("\"architecture\":\"mlp\""), "info: {info}");
    assert!(info.contains("\"name\":\"test_model\""), "info: {info}");
}

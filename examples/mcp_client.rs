//! Drive `modelc serve <artifact> --stdio` as an MCP client — the runnable
//! counterpart of the `serve --stdio` test cases.
//!
//! 1. packs a tiny 2-layer MLP into a temp artifact,
//! 2. spawns `modelc serve <artifact> --stdio` as a child process,
//! 3. performs an MCP session over stdin/stdout: `initialize` →
//!    `notifications/initialized` (silent) → `tools/list` → `tools/call chat` →
//!    `resources/list` → `resources/read`,
//! 4. validates every response and prints the transcript.
//!
//! Run with: `cargo run --example mcp_client` (uses `modelc` from PATH or the
//! local `target/debug/modelc`).

use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::process::{Command, Stdio};

fn modelc_bin() -> String {
    for candidate in ["modelc", &format!("{}/target/debug/modelc", env!("CARGO_MANIFEST_DIR"))] {
        let ok = Command::new(candidate)
            .arg("--version")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false);
        if ok {
            return candidate.to_string();
        }
    }
    panic!("modelc binary not found — build with `cargo build` first");
}

fn fixture() -> (modelc::tempdir::TempDir, std::path::PathBuf) {
    let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "layer0.weight".to_string(),
        modelc::model::TensorData {
            shape: vec![2, 2],
            dtype: modelc::model::DataType::F32,
            data: bytes(&[1.0, 0.5, -0.5, 2.0]),
        },
    );
    tensors.insert(
        "layer0.bias".to_string(),
        modelc::model::TensorData {
            shape: vec![2],
            dtype: modelc::model::DataType::F32,
            data: bytes(&[0.1, -0.1]),
        },
    );
    tensors.insert(
        "layer1.weight".to_string(),
        modelc::model::TensorData {
            shape: vec![1, 2],
            dtype: modelc::model::DataType::F32,
            data: bytes(&[2.0, -1.0]),
        },
    );
    tensors.insert(
        "layer1.bias".to_string(),
        modelc::model::TensorData {
            shape: vec![1],
            dtype: modelc::model::DataType::F32,
            data: bytes(&[0.5]),
        },
    );
    let model = modelc::model::Model {
        name: "mcp_demo".into(),
        architecture: "mlp".into(),
        tensors,
        metadata: Default::default(),
    };
    let dir = modelc::tempdir::tempdir().expect("temp dir");
    let artifact = dir.path().join("mcp_demo.modelc");
    modelc::pack::pack(&model, &artifact, false).expect("pack fixture");
    let path = artifact.clone();
    (dir, path)
}

fn main() {
    let (_dir, artifact) = fixture();
    println!("1. spawning `modelc serve {} --stdio`...", artifact.display());
    let mut child = Command::new(modelc_bin())
        .args(["serve", artifact.to_str().unwrap(), "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn modelc serve --stdio");
    let mut stdin = child.stdin.take().expect("child stdin");
    let mut reader = std::io::BufReader::new(child.stdout.take().expect("child stdout"));

    let send = |stdin: &mut std::process::ChildStdin, request: &Value| {
        println!("→ {request}");
        writeln!(stdin, "{request}").unwrap();
        stdin.flush().unwrap();
    };
    let mut line = String::new();
    let mut read = |reader: &mut std::io::BufReader<std::process::ChildStdout>| -> Value {
        line.clear();
        reader.read_line(&mut line).expect("read response");
        let response: Value = serde_json::from_str(line.trim()).expect("JSON-RPC response");
        println!("← {response}");
        response
    };

    println!("2. initialize...");
    send(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":1,"method":"initialize",
                "params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"modelc-mcp-demo","version":"0"}}}),
    );
    let response = read(&mut reader);
    assert_eq!(response["id"], 1);
    assert_eq!(response["result"]["serverInfo"]["name"], "modelc");
    assert_eq!(response["result"]["protocolVersion"], "2025-06-18");

    println!("3. notifications/initialized (no response expected)...");
    send(&mut stdin, &json!({"jsonrpc":"2.0","method":"notifications/initialized"}));

    println!("4. tools/list...");
    send(&mut stdin, &json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}));
    let response = read(&mut reader);
    let names: Vec<&str> = response["result"]["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| t["name"].as_str().expect("tool name"))
        .collect();
    assert_eq!(
        names,
        vec!["chat", "complete", "embeddings", "rerank"]
    );

    println!("5. tools/call chat...");
    send(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":3,"method":"tools/call",
                "params":{"name":"chat","arguments":{"messages":[{"role":"user","content":"hello world"}]}}}),
    );
    let response = read(&mut reader);
    assert_eq!(response["id"], 3);
    assert_eq!(response["result"]["isError"], false);
    let output = response["result"]["content"][0]["text"]
        .as_str()
        .expect("text content")
        .to_string();

    println!("6. tools/call rerank...");
    send(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":4,"method":"tools/call",
                "params":{"name":"rerank","arguments":{
                    "query":"capital cities",
                    "documents":["washington","paris","tokyo"],
                    "top_n":2,
                    "return_documents":true}}}),
    );
    let response = read(&mut reader);
    assert_eq!(response["result"]["isError"], false);
    let ranked: Value =
        serde_json::from_str(response["result"]["content"][0]["text"].as_str().expect("text"))
            .expect("rerank envelope");
    assert_eq!(ranked["results"].as_array().expect("results").len(), 2);

    println!("7. resources/list...");
    send(&mut stdin, &json!({"jsonrpc":"2.0","id":5,"method":"resources/list"}));
    let response = read(&mut reader);
    let uris: Vec<&str> = response["result"]["resources"]
        .as_array()
        .expect("resources array")
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

    println!("8. resources/read modelc://model...");
    send(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":6,"method":"resources/read",
                "params":{"uri":"modelc://model"}}),
    );
    let response = read(&mut reader);
    let overview: Value = serde_json::from_str(
        response["result"]["contents"][0]["text"]
            .as_str()
            .expect("text"),
    )
    .expect("overview json");
    assert_eq!(overview["name"], "mcp_demo");
    assert_eq!(overview["architecture"], "mlp");
    println!("   model overview: {overview}");

    println!("9. shutting down...");
    child.kill().expect("kill child");
    child.wait().expect("reap child");

    println!(
        "\nMCP session validated: model responded through stdio with {output}\n(pipeline: python/numpy/lists → weights → artifact → MCP stdio server)"
    );
}

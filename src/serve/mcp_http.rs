//! MCP **Streamable HTTP** transport (`POST /mcp`) — the second standard
//! transport defined by the Model Context Protocol (the first being stdio,
//! which `serve/stdio.rs` implements).
//!
//! The specification's requirements this transport implements:
//! - one endpoint that accepts POST carrying a single JSON-RPC request,
//!   notification or response;
//! - `202 Accepted` with no body for notifications and client responses;
//! - an `Mcp-Session-Id` issued at initialization and echoed on later requests;
//! - the `MCP-Protocol-Version` header validated on every request;
//! - `Origin` validation, so a page on the open internet cannot drive a server
//!   bound to the local machine (DNS-rebinding protection).
//!
//! The JSON-RPC core is shared with the stdio transport — every message goes
//! through `stdio::handle_value` — so both transports answer identically.
//!
//! `GET /mcp` is answered with `405 Method Not Allowed`: this server has no
//! server→client stream, and the specification allows exactly that.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use axum::body::Bytes;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use super::AppState;

/// MCP protocol versions this transport accepts in the `MCP-Protocol-Version`
/// header (mirrors the stdio transport's supported set).
const SUPPORTED_PROTOCOL_VERSIONS: [&str; 3] = ["2024-11-05", "2025-03-26", "2025-06-18"];

/// Session ids this process has issued.
fn sessions() -> &'static Mutex<HashSet<String>> {
    static SESSIONS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Fallback counter for session ids on platforms without `/dev/urandom`.
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A fresh session id: 128 bits of randomness, shaped like a UUID v4 — the
/// form MCP clients expect from an `Mcp-Session-Id`.
fn new_session_id() -> String {
    let mut bytes = random_bytes(16).unwrap_or_else(|| {
        // Fallback: mix the clock, a counter and the process id — unique
        // enough for a local session id.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let counter = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut seed = vec![0u8; 16];
        seed[..8].copy_from_slice(&nanos.to_le_bytes());
        seed[8..].copy_from_slice(&counter.to_le_bytes());
        seed
    });
    // Version 4, variant RFC 4122.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Cryptographically random bytes, when the platform offers them.
fn random_bytes(n: usize) -> Option<Vec<u8>> {
    #[cfg(unix)]
    {
        use std::io::Read;
        let mut buf = vec![0u8; n];
        std::fs::File::open("/dev/urandom")
            .ok()?
            .read_exact(&mut buf)
            .ok()?;
        Some(buf)
    }
    #[cfg(not(unix))]
    {
        let _ = n;
        None
    }
}

/// The host part of a `Host` / `Origin` value, without scheme or port.
fn host_of(value: &str) -> String {
    let value = value.trim();
    // `Origin` carries a scheme; `Host` does not.
    let value = value
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(value);
    // IPv6 literal, e.g. `[::1]:8080`.
    if let Some(rest) = value.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            return format!("[{}]", &rest[..end]);
        }
    }
    value.split(':').next().unwrap_or("").to_string()
}

/// True for host names that name the local machine.
fn is_loopback_host(host: &str) -> bool {
    matches!(host, "localhost" | "127.0.0.1" | "[::1]" | "::1")
}

/// Whether a request's `Origin` may drive this server.
///
/// The specification requires servers to validate `Origin` to prevent DNS
/// rebinding: a page on the open internet must not be able to talk to a
/// server bound to localhost. A request is allowed when it is same-origin,
/// or when both the origin and the server address are loopback.
fn origin_allowed(origin: &str, host_header: Option<&str>) -> bool {
    let origin_host = host_of(origin);
    if origin_host.is_empty() {
        return true;
    }
    match host_header {
        None => is_loopback_host(&origin_host),
        Some(host) => {
            let host = host_of(host);
            origin_host.eq_ignore_ascii_case(&host)
                || (is_loopback_host(&origin_host) && is_loopback_host(&host))
        }
    }
}

/// `POST /mcp` — the MCP Streamable HTTP endpoint.
pub(super) async fn mcp_endpoint(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Origin validation (DNS-rebinding protection).
    if let Some(origin) = headers.get("origin").and_then(|v| v.to_str().ok()) {
        let host = headers.get("host").and_then(|v| v.to_str().ok());
        if !origin_allowed(origin, host) {
            return (
                StatusCode::FORBIDDEN,
                Json(json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": { "code": -32600, "message": "cross-origin MCP requests are refused" }
                })),
            )
                .into_response();
        }
    }

    // The protocol version header is validated on every request.
    if let Some(version) = headers
        .get("mcp-protocol-version")
        .and_then(|v| v.to_str().ok())
    {
        if !SUPPORTED_PROTOCOL_VERSIONS.contains(&version) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": {
                        "code": -32600,
                        "message": format!("unsupported MCP protocol version: {version}")
                    }
                })),
            )
                .into_response();
        }
    }

    // A session id we did not issue is refused (the specification allows
    // servers to require one on every request after initialization).
    let session = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    if let Some(ref id) = session {
        if !sessions().lock().expect("session lock").contains(id) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": { "code": -32600, "message": "unknown Mcp-Session-Id" }
                })),
            )
                .into_response();
        }
    }

    let message: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": { "code": -32700, "message": format!("parse error: {e}") }
                })),
            )
                .into_response();
        }
    };

    // Notifications and client responses are accepted without a reply.
    if message.get("id").is_none() {
        return StatusCode::ACCEPTED.into_response();
    }

    let Some(result) = super::stdio::handle_value(&state, &message) else {
        return StatusCode::ACCEPTED.into_response();
    };

    // A session id is issued at initialization and echoed on later requests.
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json");
    if message.get("method").and_then(|m| m.as_str()) == Some("initialize") {
        let id = new_session_id();
        sessions().lock().expect("session lock").insert(id.clone());
        response = response.header("mcp-session-id", id);
    } else if let Some(id) = session {
        response = response.header("mcp-session-id", id);
    }
    response
        .body(axum::body::Body::from(result.to_string()))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// `GET /mcp` — this server offers no server→client stream, which the
/// specification allows as long as the endpoint answers `405`.
pub(super) async fn mcp_get_not_allowed() -> Response {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        Json(json!({
            "jsonrpc": "2.0",
            "id": Value::Null,
            "error": {
                "code": -32600,
                "message": "this server offers no server-to-client stream; use POST"
            }
        })),
    )
        .into_response()
}

/// `DELETE /mcp` — session termination is not offered; the specification
/// allows answering `405`.
pub(super) async fn mcp_delete_not_allowed() -> Response {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        Json(json!({
            "jsonrpc": "2.0",
            "id": Value::Null,
            "error": {
                "code": -32600,
                "message": "session termination is not supported; use POST"
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DataType, Model, TensorData};

    fn fixture_state() -> std::sync::Arc<AppState> {
        let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut tensors = std::collections::HashMap::new();
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
        let model = Model {
            name: "mcp_http_fixture".into(),
            architecture: "mlp".into(),
            tensors,
            metadata: Default::default(),
        };
        AppState::for_test(
            model,
            false,
            crate::generate::GenerationConfig::default(),
            None,
            None,
        )
    }

    async fn post(state: &std::sync::Arc<AppState>, body: &str, headers: &[(&str, &str)]) -> Response {
        let mut map = HeaderMap::new();
        for (name, value) in headers {
            map.insert(
                axum::http::HeaderName::from_bytes(name.as_bytes()).expect("header name"),
                axum::http::HeaderValue::from_str(value).expect("header value"),
            );
        }
        mcp_endpoint(
            axum::extract::State(state.clone()),
            map,
            Bytes::from(body.to_string()),
        )
        .await
    }

    async fn json_of(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    const INITIALIZE: &str =
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#;

    #[tokio::test]
    async fn initialize_returns_a_result_and_a_session_id() {
        let state = fixture_state();
        let response = post(&state, INITIALIZE, &[]).await;
        assert_eq!(response.status(), StatusCode::OK);
        let session = response
            .headers()
            .get("mcp-session-id")
            .expect("session id header")
            .to_str()
            .unwrap()
            .to_string();
        assert_eq!(session.len(), 36, "uuid-shaped session id: {session}");
        assert_eq!(session.chars().filter(|c| *c == '-').count(), 4);
        assert!(
            session.chars().all(|c| c.is_ascii_graphic()),
            "visible ASCII only: {session}"
        );
        let json = json_of(response).await;
        assert_eq!(json["id"], 1);
        assert_eq!(json["result"]["serverInfo"]["name"], "modelc");
        assert_eq!(
            json["result"]["capabilities"]["resources"]["subscribe"],
            false
        );
    }

    #[tokio::test]
    async fn tools_list_works_over_http() {
        let state = fixture_state();
        let response = post(
            &state,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            &[],
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_of(response).await;
        let names: Vec<&str> = json["result"]["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .map(|t| t["name"].as_str().expect("name"))
            .collect();
        assert_eq!(names, vec!["chat", "complete", "embeddings", "rerank"]);
    }

    #[tokio::test]
    async fn resources_are_readable_over_http() {
        let state = fixture_state();
        let response = post(
            &state,
            r#"{"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":"modelc://model"}}"#,
            &[],
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_of(response).await;
        assert_eq!(json["result"]["contents"][0]["uri"], "modelc://model");
    }

    #[tokio::test]
    async fn notifications_are_accepted_without_a_body() {
        let state = fixture_state();
        let response = post(
            &state,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            &[],
        )
        .await;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        assert!(bytes.is_empty(), "notifications get no body");
    }

    #[tokio::test]
    async fn malformed_body_is_rejected() {
        let state = fixture_state();
        let response = post(&state, "{not json", &[]).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let json = json_of(response).await;
        assert_eq!(json["error"]["code"], -32700);
    }

    #[tokio::test]
    async fn unsupported_protocol_version_is_rejected() {
        let state = fixture_state();
        let response = post(
            &state,
            INITIALIZE,
            &[("mcp-protocol-version", "1999-01-01")],
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let json = json_of(response).await;
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("unsupported MCP protocol version")
        );
    }

    #[tokio::test]
    async fn supported_protocol_version_is_accepted() {
        let state = fixture_state();
        let response = post(
            &state,
            INITIALIZE,
            &[("mcp-protocol-version", "2025-03-26")],
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn cross_origin_request_is_refused() {
        let state = fixture_state();
        let response = post(
            &state,
            INITIALIZE,
            &[("origin", "http://evil.example"), ("host", "127.0.0.1")],
        )
        .await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn same_origin_request_is_allowed() {
        let state = fixture_state();
        let response = post(
            &state,
            INITIALIZE,
            &[("origin", "http://127.0.0.1:8080"), ("host", "127.0.0.1")],
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_session_id_is_refused() {
        let state = fixture_state();
        let response = post(
            &state,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/list"}"#,
            &[("mcp-session-id", "not-a-session-we-issued")],
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn issued_session_id_is_echoed_back() {
        let state = fixture_state();
        let first = post(&state, INITIALIZE, &[]).await;
        let session = first
            .headers()
            .get("mcp-session-id")
            .expect("session id")
            .to_str()
            .unwrap()
            .to_string();
        let second = post(
            &state,
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/list"}"#,
            &[("mcp-session-id", &session)],
        )
        .await;
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(
            second.headers().get("mcp-session-id").unwrap().to_str().unwrap(),
            session
        );
    }

    #[tokio::test]
    async fn get_is_not_allowed() {
        let response = mcp_get_not_allowed().await;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn delete_is_not_allowed() {
        let response = mcp_delete_not_allowed().await;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[test]
    fn host_of_strips_scheme_and_port() {
        assert_eq!(host_of("http://127.0.0.1:8080"), "127.0.0.1");
        assert_eq!(host_of("127.0.0.1:8080"), "127.0.0.1");
        assert_eq!(host_of("http://[::1]:8080"), "[::1]");
        assert_eq!(host_of("https://example.com"), "example.com");
    }

    #[test]
    fn loopback_origins_are_recognized() {
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("[::1]"));
        assert!(!is_loopback_host("evil.example"));
    }
}

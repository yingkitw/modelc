//! Minimal HTTP client for integration tests. Replaces the `ureq` crate.
//!
//! Supports plain `http://` GET and POST with arbitrary headers and a request
//! body. Returns the response status code plus the raw body bytes. Not for
//! production use — no TLS, no redirect handling, no streaming.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub struct Response {
    pub status: u16,
    body: Vec<u8>,
}

impl Response {
    pub fn status(&self) -> u16 {
        self.status
    }

    pub fn body_mut(&mut self) -> BodyRef<'_> {
        BodyRef(&mut self.body)
    }

    pub fn into_body(self) -> Body {
        Body(self.body)
    }

    pub fn as_reader(&self) -> std::io::Cursor<&[u8]> {
        std::io::Cursor::new(self.body.as_slice())
    }
}

pub struct Body(Vec<u8>);

impl Body {
    pub fn into_reader(self) -> std::io::Cursor<Vec<u8>> {
        std::io::Cursor::new(self.0)
    }

    pub fn as_reader(&self) -> std::io::Cursor<&[u8]> {
        std::io::Cursor::new(self.0.as_slice())
    }
}

pub struct BodyRef<'a>(&'a mut Vec<u8>);

impl BodyRef<'_> {
    pub fn read_to_string(&mut self) -> std::io::Result<String> {
        Ok(String::from_utf8_lossy(self.0).into_owned())
    }
}

pub struct Request {
    method: &'static str,
    url: String,
    headers: Vec<(String, String)>,
    body: Option<Vec<u8>>,
}

pub fn get(url: &str) -> Request {
    Request {
        method: "GET",
        url: url.to_string(),
        headers: Vec::new(),
        body: None,
    }
}

pub fn post(url: &str) -> Request {
    Request {
        method: "POST",
        url: url.to_string(),
        headers: Vec::new(),
        body: None,
    }
}

impl Request {
    pub fn header(mut self, k: &str, v: &str) -> Self {
        self.headers.push((k.to_string(), v.to_string()));
        self
    }

    pub fn content_type(self, ct: &str) -> Self {
        self.header("Content-Type", ct)
    }

    pub fn send(self, body: &str) -> Result<Response, HttpError> {
        self.send_bytes(body.as_bytes())
    }

    pub fn send_bytes(mut self, body: &[u8]) -> Result<Response, HttpError> {
        self.body = Some(body.to_vec());
        // Move into a FnMut that reconstructs each attempt. We rebuild the Request
        // from its captured parts; this is cheap (just an allocation).
        let method = self.method;
        let url = self.url.clone();
        let headers = self.headers.clone();
        let body = self.body.clone();
        with_retry(move || do_request(method, &url, &headers, body.as_deref()))
    }

    pub fn call(self) -> Result<Response, HttpError> {
        // Move into a FnMut that reconstructs each attempt. We rebuild the Request
        // from its captured parts; this is cheap (just an allocation).
        let method = self.method;
        let url = self.url.clone();
        let headers = self.headers.clone();
        let body = self.body.clone();
        with_retry(move || do_request(method, &url, &headers, body.as_deref()))
    }
}

#[derive(Debug)]
pub enum HttpError {
    StatusCode(u16),
    Io(std::io::Error),
}

impl From<std::io::Error> for HttpError {
    fn from(e: std::io::Error) -> Self {
        HttpError::Io(e)
    }
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpError::StatusCode(c) => write!(f, "HTTP status {}", c),
            HttpError::Io(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for HttpError {}

fn do_request(
    method: &str,
    url: &str,
    headers: &[(String, String)],
    body: Option<&[u8]>,
) -> Result<Response, HttpError> {
    let rest = url.strip_prefix("http://").ok_or_else(|| {
        HttpError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "only http:// URLs are supported in tests",
        ))
    })?;
    let (host_port, path) = match rest.split_once('/') {
        Some((h, p)) => (h, format!("/{}", p)),
        None => (rest, "/".to_string()),
    };
    let (host, port) = match host_port.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse::<u16>().unwrap_or(80)),
        None => (host_port.to_string(), 80u16),
    };

    let mut stream = TcpStream::connect((host.as_str(), port))?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;

    let body = body.unwrap_or(&[]);
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: modelc-test\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");

    stream.write_all(req.as_bytes())?;
    if !body.is_empty() {
        stream.write_all(body)?;
    }

    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;

    let header_end = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| {
            HttpError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HTTP response missing header terminator",
            ))
        })?;
    let headers_str = std::str::from_utf8(&response[..header_end])
        .map_err(|e| HttpError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    let status_line = headers_str.lines().next().ok_or_else(|| {
        HttpError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "empty HTTP response",
        ))
    })?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            HttpError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid HTTP status line",
            ))
        })?;
    let body = response[header_end + 4..].to_vec();

    if status >= 400 {
        Err(HttpError::StatusCode(status))
    } else {
        Ok(Response { status, body })
    }
}

pub fn read_body(resp: Response) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut buf = Vec::new();
    resp.into_body().into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

/// True when an I/O error looks like a transient connection-level failure worth retrying.
fn is_transient_io(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::TimedOut
            | std::io::ErrorKind::UnexpectedEof
    )
}

const RETRY_ATTEMPTS: usize = 10;
const RETRY_BACKOFF: std::time::Duration = std::time::Duration::from_millis(20);

/// Wrap a single request attempt with retry on transient I/O errors.
fn with_retry<F>(mut f: F) -> Result<Response, HttpError>
where
    F: FnMut() -> Result<Response, HttpError>,
{
    let mut last_err: Option<HttpError> = None;
    for attempt in 0..RETRY_ATTEMPTS {
        match f() {
            Ok(resp) => return Ok(resp),
            Err(HttpError::Io(e)) if attempt + 1 < RETRY_ATTEMPTS && is_transient_io(&e) => {
                last_err = Some(HttpError::Io(e));
                std::thread::sleep(RETRY_BACKOFF);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.expect("retry loop ran without recording an error"))
}

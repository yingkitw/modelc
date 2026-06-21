//! Authentication and rate-limiting middleware for the HTTP server.
//!
//! When `--api-key` is set, every request must include an `Authorization: Bearer <key>`
//! header.  When `--rate-limit` is set, a per-IP token bucket limits requests to N
//! per minute.  Both can be active simultaneously.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};

/// Shared state for auth + rate limiting.
#[derive(Clone)]
pub struct AuthConfig {
    /// Expected `Authorization: Bearer <key>` value.  `None` = no API key check.
    pub api_key: Option<String>,
    /// Max requests per minute per client IP.  `None` or `0` = no limit.
    pub rate_limit_per_minute: Option<u32>,
    /// In-memory token buckets keyed by client IP.
    buckets: Arc<Mutex<HashMap<String, TokenBucket>>>,
}

impl AuthConfig {
    pub fn new(api_key: Option<String>, rate_limit_per_minute: Option<u32>) -> Self {
        Self {
            api_key,
            rate_limit_per_minute,
            buckets: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// Simple token bucket for rate limiting.
pub(crate) struct TokenBucket {
    tokens: f64,
    last_update: std::time::Instant,
    rate: f64,   // tokens per second
    capacity: f64,
}

impl TokenBucket {
    fn new(per_minute: u32) -> Self {
        let capacity = per_minute.max(1) as f64;
        Self {
            tokens: capacity,
            last_update: std::time::Instant::now(),
            rate: capacity / 60.0,
            capacity,
        }
    }

    /// Try to consume one token.  Returns `true` if allowed.
    fn consume(&mut self) -> bool {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rate).min(self.capacity);
        self.last_update = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Axum middleware that checks API key and rate limit.
/// /health and /info are exempt so load-balancer probes work without auth.
pub async fn middleware(
    axum::extract::State(config): axum::extract::State<AuthConfig>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path();
    if path == "/health" || path == "/info" {
        return next.run(req).await;
    }

    // API key check
    if let Some(ref expected) = config.api_key {
        let auth = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let bearer = auth.strip_prefix("Bearer ").unwrap_or("");
        if bearer != expected {
            return StatusCode::UNAUTHORIZED.into_response();
        }
    }

    // Rate limiting check (uses a simple peer-addr fallback; the exact IP is
    // best-effort in test/localhost environments).
    if let Some(limit) = config.rate_limit_per_minute && limit > 0 {
        let ip = req
            .extensions()
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
            .map(|ci| ci.0.ip().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let allowed = {
            let mut buckets = config.buckets.lock().unwrap();
            let bucket = buckets
                .entry(ip)
                .or_insert_with(|| TokenBucket::new(limit));
            bucket.consume()
        };

        if !allowed {
            return StatusCode::TOO_MANY_REQUESTS.into_response();
        }
    }

    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_bucket_allows_within_limit() {
        let mut bucket = TokenBucket::new(60);
        assert!(bucket.consume(), "first request should be allowed");
        assert!(bucket.consume(), "second request should be allowed");
    }

    #[test]
    fn token_bucket_refills_over_time() {
        let mut bucket = TokenBucket::new(1);
        assert!(bucket.consume(), "first request should be allowed");
        assert!(!bucket.consume(), "second request should be denied immediately");
        // Simulate time passing
        bucket.last_update = std::time::Instant::now() - std::time::Duration::from_secs(60);
        assert!(bucket.consume(), "should be allowed after refill period");
    }
}

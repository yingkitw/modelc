//! Minimal Prometheus-style metrics using std atomics.
//!
//! Tracks request counts, latency histograms, active requests, and tokens generated.
//! Formatted on-the-fly in Prometheus text exposition format for `GET /metrics`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Simplified histogram with fixed buckets (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, +Inf).
/// Each bucket stores the count of observations <= its upper bound.
#[derive(Default)]
pub struct Histogram {
    buckets: [AtomicU64; 9],
    sum: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    const BOUNDS: [f64; 9] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, f64::INFINITY];

    pub fn observe(&self, seconds: f64) {
        let nanos = (seconds * 1_000_000_000.0) as u64;
        self.sum.fetch_add(nanos, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        for (i, bound) in Self::BOUNDS.iter().enumerate() {
            if seconds <= *bound {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    pub fn sum(&self) -> f64 {
        let nanos = self.sum.load(Ordering::Relaxed);
        nanos as f64 / 1_000_000_000.0
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn bucket_counts(&self) -> [u64; 9] {
        std::array::from_fn(|i| self.buckets[i].load(Ordering::Relaxed))
    }
}

/// Global metrics state shared across handlers.
#[derive(Default)]
pub struct Metrics {
    requests_total: AtomicU64,
    active_requests: AtomicU64,
    inference_duration: Histogram,
}

impl Metrics {
    pub fn increment_requests(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_active(&self) {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn decrement_active(&self) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn observe_inference_duration(&self, seconds: f64) {
        self.inference_duration.observe(seconds);
    }

    /// Render metrics in Prometheus text format.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("# HELP modelc_requests_total Total number of HTTP requests.\n");
        out.push_str("# TYPE modelc_requests_total counter\n");
        out.push_str(&format!(
            "modelc_requests_total {}\n",
            self.requests_total.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP modelc_active_requests Number of requests currently in flight.\n");
        out.push_str("# TYPE modelc_active_requests gauge\n");
        out.push_str(&format!(
            "modelc_active_requests {}\n",
            self.active_requests.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP modelc_inference_duration_seconds Time spent in inference.\n");
        out.push_str("# TYPE modelc_inference_duration_seconds histogram\n");
        let counts = self.inference_duration.bucket_counts();
        for (i, bound) in Histogram::BOUNDS.iter().enumerate() {
            let le = if bound.is_infinite() {
                "+Inf".to_string()
            } else {
                format!("{}", bound)
            };
            out.push_str(&format!(
                "modelc_inference_duration_seconds_bucket{{le=\"{}\"}} {}\n",
                le, counts[i]
            ));
        }
        out.push_str(&format!(
            "modelc_inference_duration_seconds_sum {}\n",
            self.inference_duration.sum()
        ));
        out.push_str(&format!(
            "modelc_inference_duration_seconds_count {}\n",
            self.inference_duration.count()
        ));

        out
    }
}

/// RAII guard that increments active requests on creation and decrements on drop.
pub struct ActiveRequestGuard<'a> {
    metrics: &'a Metrics,
}

impl<'a> ActiveRequestGuard<'a> {
    pub fn new(metrics: &'a Metrics) -> Self {
        metrics.increment_active();
        metrics.increment_requests();
        Self { metrics }
    }
}

impl Drop for ActiveRequestGuard<'_> {
    fn drop(&mut self) {
        self.metrics.decrement_active();
    }
}

/// RAII timer that records duration on drop.
pub struct InferenceTimer<'a> {
    metrics: &'a Metrics,
    start: Instant,
}

impl<'a> InferenceTimer<'a> {
    pub fn new(metrics: &'a Metrics) -> Self {
        Self {
            metrics,
            start: Instant::now(),
        }
    }
}

impl Drop for InferenceTimer<'_> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        self.metrics.observe_inference_duration(elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_observes_into_buckets() {
        let h = Histogram::default();
        h.observe(0.03); // bucket 0 (0.01) no, bucket 1 (0.05) yes
        h.observe(0.07); // bucket 2 (0.1) yes
        h.observe(1.5);  // bucket 6 (2.5) yes
        h.observe(10.0); // bucket 8 (+Inf) yes
        let counts = h.bucket_counts();
        assert_eq!(counts[0], 0, "0.03 > 0.01");
        assert_eq!(counts[1], 1, "0.03 <= 0.05");
        assert_eq!(counts[2], 1, "0.07 <= 0.1");
        assert_eq!(counts[3], 0);
        assert_eq!(counts[4], 0);
        assert_eq!(counts[5], 0);
        assert_eq!(counts[6], 1, "1.5 <= 2.5");
        assert_eq!(counts[7], 0);
        assert_eq!(counts[8], 1, "10.0 <= +Inf");
        assert_eq!(h.count(), 4);
        assert!((h.sum() - 11.6).abs() < 0.01);
    }

    #[test]
    fn metrics_render_includes_all_series() {
        let m = Metrics::default();
        m.increment_requests();
        m.increment_requests();
        m.observe_inference_duration(0.05);
        let text = m.render();
        assert!(text.contains("modelc_requests_total 2"));
        assert!(text.contains("modelc_inference_duration_seconds_bucket{le=\"0.05\"} 1"));
        assert!(text.contains("modelc_inference_duration_seconds_count 1"));
    }

    #[test]
    fn active_request_guard_changes_gauge() {
        let m = Metrics::default();
        assert_eq!(m.active_requests.load(Ordering::Relaxed), 0);
        {
            let _guard = ActiveRequestGuard::new(&m);
            assert_eq!(m.active_requests.load(Ordering::Relaxed), 1);
        }
        assert_eq!(m.active_requests.load(Ordering::Relaxed), 0);
    }
}

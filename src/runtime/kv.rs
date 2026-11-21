//! Key/value cache containers for incremental transformer decoding.
//!
//! `KvLayer` holds one layer's K/V tensors in FP32, INT8-quantized, or Mixed
//! (anchor FP32 + INT8 tail) layout; `KvCache` is the per-model stack of them.
//! Used by `transformer.rs` forward passes, `generate.rs` sampling, and
//! `prefix_cache.rs` snapshots.

use std::borrow::Cow;

/// Can store either FP32, INT8-quantized data (4x memory reduction), or a
/// mixed-precision mode where recent tokens stay in FP32 and older tokens are
/// quantized to INT8 (combines accuracy for active context with memory savings).
#[derive(Clone)]
pub enum KvLayer {
    Fp32 {
        k: Vec<f32>,
        v: Vec<f32>,
        hidden: usize,
    },
    Int8 {
        k_q: Vec<i8>,
        v_q: Vec<i8>,
        k_scales: Vec<f32>,
        v_scales: Vec<f32>,
        hidden: usize,
    },
    /// Recent `window` tokens in FP32 ("hot"), older tokens in INT8 ("cold").
    Mixed {
        k_hot: Vec<f32>,
        v_hot: Vec<f32>,
        k_cold_q: Vec<i8>,
        v_cold_q: Vec<i8>,
        k_cold_scales: Vec<f32>,
        v_cold_scales: Vec<f32>,
        hidden: usize,
        window: usize,
    },
}

impl KvLayer {
    fn quantize_scale(xs: &[f32]) -> f32 {
        let max = xs.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max > 0.0 { max / 127.0 } else { 1.0 }
    }

    fn quantize(xs: &[f32], scale: f32) -> Vec<i8> {
        xs.iter()
            .map(|x| (*x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect()
    }

    pub fn new_fp32() -> Self {
        KvLayer::Fp32 {
            k: Vec::new(),
            v: Vec::new(),
            hidden: 0,
        }
    }

    pub fn new_int8(hidden: usize) -> Self {
        KvLayer::Int8 {
            k_q: Vec::new(),
            v_q: Vec::new(),
            k_scales: Vec::new(),
            v_scales: Vec::new(),
            hidden,
        }
    }

    pub fn new_mixed(hidden: usize, window: usize) -> Self {
        KvLayer::Mixed {
            k_hot: Vec::new(),
            v_hot: Vec::new(),
            k_cold_q: Vec::new(),
            v_cold_q: Vec::new(),
            k_cold_scales: Vec::new(),
            v_cold_scales: Vec::new(),
            hidden,
            window,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            KvLayer::Fp32 { k, hidden, .. } => {
                if *hidden > 0 {
                    k.len() / hidden
                } else {
                    0
                }
            }
            KvLayer::Int8 { k_q, hidden, .. } => {
                if *hidden > 0 {
                    k_q.len() / hidden
                } else {
                    0
                }
            }
            KvLayer::Mixed {
                k_hot,
                k_cold_q,
                hidden,
                ..
            } => {
                if *hidden > 0 {
                    k_hot.len() / hidden + k_cold_q.len() / hidden
                } else {
                    0
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrowed view of the accumulated K data.
    ///
    /// For [`KvLayer::Fp32`] (the default) this returns a borrowed slice into the
    /// backing buffer — zero-copy, so attention pays nothing per token as the cache
    /// grows. Quantized variants ([`KvLayer::Int8`] / [`KvLayer::Mixed`]) still
    /// materialize owned FP32 via [`k_all`](Self::k_all), preserving their behavior.
    pub fn k_all_cow(&self) -> Cow<'_, [f32]> {
        match self {
            KvLayer::Fp32 { k, .. } => Cow::Borrowed(k),
            _ => Cow::Owned(self.k_all()),
        }
    }

    /// Borrowed view of the accumulated V data. See [`k_all_cow`](Self::k_all_cow).
    pub fn v_all_cow(&self) -> Cow<'_, [f32]> {
        match self {
            KvLayer::Fp32 { v, .. } => Cow::Borrowed(v),
            _ => Cow::Owned(self.v_all()),
        }
    }

    pub fn k_all(&self) -> Vec<f32> {
        match self {
            KvLayer::Fp32 { k, .. } => k.clone(),
            KvLayer::Int8 {
                k_q,
                k_scales,
                hidden,
                ..
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let n = k_q.len() / h;
                let mut result = Vec::with_capacity(k_q.len());
                for i in 0..n {
                    let scale = k_scales[i];
                    for j in 0..*hidden {
                        result.push(k_q[i * hidden + j] as f32 * scale);
                    }
                }
                result
            }
            KvLayer::Mixed {
                k_hot,
                k_cold_q,
                k_cold_scales,
                hidden,
                ..
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let n = k_cold_q.len() / h;
                let mut result = Vec::with_capacity(k_hot.len() + k_cold_q.len());
                for i in 0..n {
                    let scale = k_cold_scales[i];
                    for j in 0..*hidden {
                        result.push(k_cold_q[i * hidden + j] as f32 * scale);
                    }
                }
                result.extend_from_slice(k_hot);
                result
            }
        }
    }

    pub fn v_all(&self) -> Vec<f32> {
        match self {
            KvLayer::Fp32 { v, .. } => v.clone(),
            KvLayer::Int8 {
                v_q,
                v_scales,
                hidden,
                ..
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let n = v_q.len() / h;
                let mut result = Vec::with_capacity(v_q.len());
                for i in 0..n {
                    let scale = v_scales[i];
                    for j in 0..*hidden {
                        result.push(v_q[i * hidden + j] as f32 * scale);
                    }
                }
                result
            }
            KvLayer::Mixed {
                v_hot,
                v_cold_q,
                v_cold_scales,
                hidden,
                ..
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let n = v_cold_q.len() / h;
                let mut result = Vec::with_capacity(v_hot.len() + v_cold_q.len());
                for i in 0..n {
                    let scale = v_cold_scales[i];
                    for j in 0..*hidden {
                        result.push(v_cold_q[i * hidden + j] as f32 * scale);
                    }
                }
                result.extend_from_slice(v_hot);
                result
            }
        }
    }

    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        match self {
            KvLayer::Fp32 {
                k: ck,
                v: cv,
                hidden,
            } => {
                if *hidden == 0 {
                    *hidden = k.len();
                }
                ck.extend_from_slice(k);
                cv.extend_from_slice(v);
            }
            KvLayer::Int8 {
                k_q,
                v_q,
                k_scales,
                v_scales,
                hidden,
            } => {
                *hidden = k.len();
                let k_scale = Self::quantize_scale(k);
                let v_scale = Self::quantize_scale(v);
                k_scales.push(k_scale);
                v_scales.push(v_scale);
                k_q.extend(Self::quantize(k, k_scale));
                v_q.extend(Self::quantize(v, v_scale));
            }
            KvLayer::Mixed {
                k_hot,
                v_hot,
                k_cold_q,
                v_cold_q,
                k_cold_scales,
                v_cold_scales,
                hidden,
                window,
            } => {
                if *hidden == 0 {
                    *hidden = k.len();
                }
                let h = *hidden;
                k_hot.extend_from_slice(k);
                v_hot.extend_from_slice(v);
                // If hot buffer exceeds window, move oldest tokens to cold.
                let hot_tokens = k_hot.len() / h;
                if hot_tokens > *window {
                    let to_move = hot_tokens - *window;
                    let move_elements = to_move * h;
                    for i in 0..to_move {
                        let slice_k = &k_hot[i * h..(i + 1) * h];
                        let slice_v = &v_hot[i * h..(i + 1) * h];
                        let k_scale = Self::quantize_scale(slice_k);
                        let v_scale = Self::quantize_scale(slice_v);
                        k_cold_scales.push(k_scale);
                        v_cold_scales.push(v_scale);
                        k_cold_q.extend(Self::quantize(slice_k, k_scale));
                        v_cold_q.extend(Self::quantize(slice_v, v_scale));
                    }
                    *k_hot = k_hot.split_off(move_elements);
                    *v_hot = v_hot.split_off(move_elements);
                }
            }
        }
    }

    /// Remove the oldest `n` tokens from the cache, shifting remaining tokens left.
    /// If `n` >= current length, the cache is cleared.
    pub fn shift(&mut self, n: usize) {
        match self {
            KvLayer::Fp32 { k, v, hidden } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let keep = k.len() / h;
                let remove = n.min(keep);
                let remove_elements = remove * h;
                *k = k.split_off(remove_elements);
                *v = v.split_off(remove_elements);
            }
            KvLayer::Int8 {
                k_q,
                v_q,
                k_scales,
                v_scales,
                hidden,
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let keep = k_q.len() / h;
                let remove = n.min(keep);
                let remove_elements = remove * h;
                *k_q = k_q.split_off(remove_elements);
                *v_q = v_q.split_off(remove_elements);
                let scale_remove = k_scales.len().min(remove);
                *k_scales = k_scales.split_off(scale_remove);
                *v_scales = v_scales.split_off(scale_remove);
            }
            KvLayer::Mixed {
                k_hot,
                v_hot,
                k_cold_q,
                v_cold_q,
                k_cold_scales,
                v_cold_scales,
                hidden,
                ..
            } => {
                let h = if *hidden == 0 { 1 } else { *hidden };
                let cold_tokens = k_cold_q.len() / h;
                let hot_tokens = k_hot.len() / h;
                let total = cold_tokens + hot_tokens;
                let remove = n.min(total);

                if remove <= cold_tokens {
                    // Remove only from cold.
                    let remove_elements = remove * h;
                    *k_cold_q = k_cold_q.split_off(remove_elements);
                    *v_cold_q = v_cold_q.split_off(remove_elements);
                    let scale_remove = k_cold_scales.len().min(remove);
                    *k_cold_scales = k_cold_scales.split_off(scale_remove);
                    *v_cold_scales = v_cold_scales.split_off(scale_remove);
                } else {
                    // Remove all cold, then some from hot.
                    k_cold_q.clear();
                    v_cold_q.clear();
                    k_cold_scales.clear();
                    v_cold_scales.clear();
                    let hot_remove = remove - cold_tokens;
                    let remove_elements = hot_remove * h;
                    *k_hot = k_hot.split_off(remove_elements);
                    *v_hot = v_hot.split_off(remove_elements);
                }
            }
        }
    }

    /// Remove `n` tokens from the middle of the cache, preserving the first `anchor`
    /// tokens at the beginning. This implements StreamingLLM-style eviction where
    /// "attention sink" anchor tokens (e.g., the first 4 tokens of a prompt) are
    /// retained while older non-anchor tokens are evicted to make room for new ones.
    ///
    /// If `anchor` is 0 or there are not enough non-anchor tokens to remove,
    /// falls back to regular `shift(n)`.
    pub fn shift_anchored(&mut self, n: usize, anchor: usize) {
        if anchor == 0 || n == 0 {
            self.shift(n);
            return;
        }
        let len = self.len();
        if n >= len {
            self.shift(n);
            return;
        }
        let keep = len - n;
        if keep <= anchor {
            self.shift(n);
            return;
        }

        let h = match self {
            KvLayer::Fp32 { hidden, .. } => *hidden,
            KvLayer::Int8 { hidden, .. } => *hidden,
            KvLayer::Mixed { hidden, .. } => *hidden,
        };

        let full_k = self.k_all();
        let full_v = self.v_all();

        let prefix_k = &full_k[..anchor * h];
        let prefix_v = &full_v[..anchor * h];
        let suffix_k = &full_k[(anchor + n) * h..];
        let suffix_v = &full_v[(anchor + n) * h..];

        // Clear current state while preserving variant and window.
        match self {
            KvLayer::Fp32 { k, v, hidden } => {
                k.clear();
                v.clear();
                *hidden = h;
            }
            KvLayer::Int8 {
                k_q,
                v_q,
                k_scales,
                v_scales,
                hidden,
            } => {
                k_q.clear();
                v_q.clear();
                k_scales.clear();
                v_scales.clear();
                *hidden = h;
            }
            KvLayer::Mixed {
                k_hot,
                v_hot,
                k_cold_q,
                v_cold_q,
                k_cold_scales,
                v_cold_scales,
                hidden,
                window,
            } => {
                let win = *window;
                k_hot.clear();
                v_hot.clear();
                k_cold_q.clear();
                v_cold_q.clear();
                k_cold_scales.clear();
                v_cold_scales.clear();
                *hidden = h;
                *window = win;
            }
        }

        // Re-append anchor tokens.
        for i in 0..anchor {
            let s = i * h;
            let e = s + h;
            self.append(&prefix_k[s..e], &prefix_v[s..e]);
        }

        // Re-append remaining suffix tokens.
        let suffix_tokens = keep - anchor;
        for i in 0..suffix_tokens {
            let s = i * h;
            let e = s + h;
            self.append(&suffix_k[s..e], &suffix_v[s..e]);
        }
    }
}

/// A cache of Key/Value vectors for each transformer layer.
#[derive(Clone)]
pub struct KvCache {
    pub layers: Vec<Option<KvLayer>>,
}

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: vec![None; n_layers],
        }
    }

    pub fn new_quantized(n_layers: usize, hidden: usize, use_int8: bool) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(Some(if use_int8 {
                KvLayer::new_int8(hidden)
            } else {
                KvLayer::new_fp32()
            }));
        }
        Self { layers }
    }

    pub fn new_mixed(n_layers: usize, hidden: usize, window: usize) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(Some(KvLayer::new_mixed(hidden, window)));
        }
        Self { layers }
    }

    /// Remove the oldest `n` tokens from every layer in the cache.
    pub fn shift(&mut self, n: usize) {
        for l in self.layers.iter_mut().flatten() {
            l.shift(n);
        }
    }

    /// Remove `n` tokens from the middle of every layer, preserving the first `anchor` tokens.
    /// See [`KvLayer::shift_anchored`] for details.
    pub fn shift_anchored(&mut self, n: usize, anchor: usize) {
        for l in self.layers.iter_mut().flatten() {
            l.shift_anchored(n, anchor);
        }
    }

    /// Total number of cached tokens (assumes all layers are in sync).
    pub fn len(&self) -> usize {
        self.layers
            .first()
            .and_then(|l| l.as_ref().map(|layer| layer.len()))
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Per-layer steps. Each mutates the residual stream `h` in place and returns whether the
// block had any effect (so callers can skip profiling when the block was a no-op).
// ---------------------------------------------------------------------------

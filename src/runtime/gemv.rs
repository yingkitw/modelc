//! SIMD-accelerated GEMV row dot-product.
//!
//! THIS FILE IS COMPILED INTO `modelc` (used by the `modelc run` runtime) AND EMITTED
//! VERBATIM INTO `modelc compile`-generated binaries (via `include_str!` in
//! `codegen/native/helpers.rs`). The two inference paths therefore share a single source
//! of truth, so `modelc run` and `modelc compile` produce bit-identical logits even though
//! the reduction is vectorized (and therefore reassociated) over the K dimension. Vector
//! width, FMA vs mul+add, the horizontal-reduction order, and the scalar tail are all fixed
//! per architecture so both paths round the same way on the same host.
//!
//! If you edit this file, you are editing both paths at once — that is the point.

/// Row dot-product `sum_i w[i] * x[i]`, vectorized over the reduction dimension when the
/// host supports it, with a scalar fallback. The accumulation order is fixed per arch so
/// callers (runtime and emitted binaries) round identically.
#[allow(dead_code)]
pub fn gemv_dot(w: &[f32], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let n = w.len();

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && n >= 4 {
            return unsafe { gemv_dot_neon(w, x) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && n >= 8 {
            return unsafe { gemv_dot_avx(w, x) };
        }
    }

    gemv_dot_scalar(w, x)
}

fn gemv_dot_scalar(w: &[f32], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0;
    while i < w.len() {
        acc += w[i] * x[i];
        i += 1;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
unsafe fn gemv_dot_neon(w: &[f32], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = w.len();
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    let main = n - (n % 4);
    let mut i = 0usize;
    while i < main {
        let wv = unsafe { vld1q_f32(w.as_ptr().add(i)) };
        let xv = unsafe { vld1q_f32(x.as_ptr().add(i)) };
        acc = unsafe { vfmaq_f32(acc, wv, xv) };
        i += 4;
    }
    let mut lanes = [0f32; 4];
    unsafe { vst1q_f32(lanes.as_mut_ptr(), acc) };
    let mut s = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < n {
        s += w[i] * x[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
unsafe fn gemv_dot_avx(w: &[f32], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = w.len();
    let mut acc = unsafe { _mm256_setzero_ps() };
    let main = n - (n % 8);
    let mut i = 0usize;
    while i < main {
        let wv = unsafe { _mm256_loadu_ps(w.as_ptr().add(i)) };
        let xv = unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) };
        acc = unsafe { _mm256_add_ps(acc, _mm256_mul_ps(wv, xv)) };
        i += 8;
    }
    let mut lanes = [0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let lo = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    let hi = (lanes[4] + lanes[5]) + (lanes[6] + lanes[7]);
    let mut s = lo + hi;
    while i < n {
        s += w[i] * x[i];
        i += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_dot_empty_is_zero() {
        assert_eq!(gemv_dot(&[], &[]), 0.0);
    }

    #[test]
    fn gemv_dot_matches_reference_for_small_exact_values() {
        // Values chosen so every product and the total are exactly representable in f32;
        // SIMD (reassociated) and scalar must therefore agree exactly here.
        let w: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5).collect();
        let expected: f32 = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let got = gemv_dot(&w, &x);
        assert!((got - expected).abs() < 1e-5, "got {got}, expected {expected}");
    }

    #[test]
    fn gemv_dot_handles_non_multiple_of_lane_width() {
        // 13 elements exercises the scalar tail on every architecture.
        let w: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let x: Vec<f32> = (0..13).map(|i| (i as f32) + 1.0).collect();
        let expected: f32 = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let got = gemv_dot(&w, &x);
        assert!((got - expected).abs() < 1e-4, "got {got}, expected {expected}");
    }
}

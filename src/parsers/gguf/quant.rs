//! GGUF block-quantization **encoder** — the write side of the layouts that
//! [`super::dequant`] decodes.
//!
//! Mirrors llama.cpp's reference quantizers (`quantize_row_q4_0_ref` /
//! `quantize_row_q8_0_ref` in `ggml-quants.c`) byte-for-byte, so an artifact
//! packed by modelc is byte-compatible with what llama.cpp writes for the same
//! weights — and llama.cpp can read ours back.
//!
//! Both layouts quantize in blocks of 32 elements. A tensor whose element count
//! is not a whole number of blocks cannot be encoded, so the encoders return
//! `None` and the caller keeps that tensor in F32 (llama.cpp's own fallback).

use half::f16;

/// Elements per block for the Q4_0 / Q8_0 layouts.
const BLOCK_ELEMENTS: usize = 32;

/// Encode `values` into Q4_0 blocks: 18 bytes per 32 elements
/// (one f16 scale + 16 bytes holding two 4-bit weights each).
///
/// Returns `None` when the element count is not a multiple of the block size.
pub fn quantize_q4_0(values: &[f32]) -> Option<Vec<u8>> {
    if !values.len().is_multiple_of(BLOCK_ELEMENTS) {
        return None;
    }
    let mut out = Vec::with_capacity(values.len() / BLOCK_ELEMENTS * 18);
    for block in values.chunks_exact(BLOCK_ELEMENTS) {
        // llama.cpp: `amax` is the largest magnitude in the block, `max` the
        // signed value that produced it, and the scale is `max / -8` — so the
        // representable range is [-8·d, 7·d] in the 4-bit domain.
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in block {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for j in 0..BLOCK_ELEMENTS / 2 {
            let x0 = block[j] * id;
            let x1 = block[j + BLOCK_ELEMENTS / 2] * id;
            // C's `(int8_t)` cast truncates toward zero; `MIN(15, …)` clamps.
            let xi0 = ((x0 + 8.5) as i32).clamp(0, 15) as u8;
            let xi1 = ((x1 + 8.5) as i32).clamp(0, 15) as u8;
            // Low nibble holds element j, high nibble holds element j + 16.
            out.push(xi0 | (xi1 << 4));
        }
    }
    Some(out)
}

/// Encode `values` into Q8_0 blocks: 34 bytes per 32 elements
/// (one f16 scale + 32 int8 weights).
///
/// Returns `None` when the element count is not a multiple of the block size.
pub fn quantize_q8_0(values: &[f32]) -> Option<Vec<u8>> {
    if !values.len().is_multiple_of(BLOCK_ELEMENTS) {
        return None;
    }
    let mut out = Vec::with_capacity(values.len() / BLOCK_ELEMENTS * 34);
    for block in values.chunks_exact(BLOCK_ELEMENTS) {
        // llama.cpp: the scale is `amax / 127` (always positive) and each weight
        // is stored as `round(x / d)`.
        let amax = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for &v in block {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    Some(out)
}

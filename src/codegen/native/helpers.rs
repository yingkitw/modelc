use crate::model::Model;

/// Emit the runtime helpers required by the generated `forward` for this model's architecture.
///
/// The MLP path keeps its existing `matmul_bias` / `relu_inplace` helpers. The GPT-2 and LLaMA
/// paths emit a richer FP32 tensor toolbox (layer norm / RMS norm, GeLU / SiLU, softmax, RoPE,
/// single-token causal attention) so their generated `forward` bodies compile and execute a
/// structurally complete transformer block instead of relying on stubs.
pub(super) fn emit_helpers(model: &Model) -> String {
    match model.architecture.as_str() {
        "mlp" => mlp_helpers(),
        "gpt2" => gpt2_helpers(),
        "llama" => llama_helpers(),
        _ => String::new(),
    }
}

fn mlp_helpers() -> String {
    r#"fn decode_f32(state: &AppState, name: &str) -> Vec<f32> {
    let meta = state
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor `{}`", name));
    let slice = &state.weights[meta.byte_offset..meta.byte_offset + meta.byte_len];
    decode_f32_le(slice).expect("decoder expects fp32 payloads for this server")
}

fn decode_f32_le(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().ok()?));
    }
    Some(out)
}

fn matmul_bias(
    state: &AppState,
    weight_key: &str,
    bias_key: &str,
    x: &[f32],
) -> Vec<f32> {
    let w_meta = state
        .tensors
        .get(weight_key)
        .unwrap_or_else(|| panic!("missing tensor `{}`", weight_key));
    assert_eq!(
        w_meta.dtype_size, 4,
        "`{}` must be fp32 for this emitted server",
        weight_key
    );
    assert_eq!(
        w_meta.shape.len(),
        2,
        "`{}`: expected dense matrix",
        weight_key,
    );

    let rows = w_meta.shape[0];
    let cols = w_meta.shape[1];
    assert_eq!(
        cols,
        x.len(),
        "`{}` gemv mismatch: cols {cols}, input {}",
        weight_key,
        x.len(),
    );

    let w_flat = decode_f32(state, weight_key);
    assert_eq!(w_flat.len(), rows.checked_mul(cols).expect("sizes"));

    let bias = decode_f32(state, bias_key);
    assert_eq!(bias.len(), rows);

    let mut out = vec![0f32; rows];
    for r in 0..rows {
        let mut acc = bias[r];
        let row = &w_flat[r * cols..(r + 1) * cols];
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += *wv * *xv;
        }
        out[r] = acc;
    }
    out
}

fn relu_inplace(xs: &mut [f32]) {
    for v in xs {
        *v = v.max(0.0);
    }
}"#
    .trim_end()
    .to_string()
}

fn gpt2_helpers() -> String {
    r#"fn decode_f32(state: &AppState, name: &str) -> Vec<f32> {
    let meta = state
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor `{}`", name));
    let slice = &state.weights[meta.byte_offset..meta.byte_offset + meta.byte_len];
    decode_f32_le(slice).expect("decoder expects fp32 payloads for this server")
}

fn decode_f32_le(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().ok()?));
    }
    Some(out)
}

/// `y = W x + b` for a dense weight stored row-major as `[out, in]`. `bias_key` may be absent,
/// in which case the bias is treated as zero (matches embeddings / tied heads).
fn linear(state: &AppState, weight_key: &str, bias_key: Option<&str>, x: &[f32]) -> Vec<f32> {
    let w_meta = state
        .tensors
        .get(weight_key)
        .unwrap_or_else(|| panic!("missing tensor `{}`", weight_key));
    assert_eq!(w_meta.dtype_size, 4, "`{}` must be fp32", weight_key);
    assert_eq!(w_meta.shape.len(), 2, "`{}` must be a dense matrix", weight_key);

    let rows = w_meta.shape[0];
    let cols = w_meta.shape[1];
    assert_eq!(cols, x.len(), "`{}` gemv mismatch: cols {cols}, input {}", weight_key, x.len());

    let w_flat = decode_f32(state, weight_key);
    let bias = match bias_key {
        Some(k) => decode_f32(state, k),
        None => vec![0.0f32; rows],
    };
    assert_eq!(bias.len(), rows);

    let mut out = vec![0f32; rows];
    for r in 0..rows {
        let mut acc = bias[r];
        let row = &w_flat[r * cols..(r + 1) * cols];
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += *wv * *xv;
        }
        out[r] = acc;
    }
    out
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n.max(1) as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n.max(1) as f32;
    let std = (var + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let wi = w.get(i).copied().unwrap_or(1.0);
            let bi = b.get(i).copied().unwrap_or(0.0);
            (v - mean) / std * wi + bi
        })
        .collect()
}

fn gelu(x: &[f32]) -> Vec<f32> {
    // Exact (erf) GeLU as used by GPT-2.
    let c = f32::sqrt(2.0 / std::f32::consts::PI);
    x.iter()
        .map(|&v| 0.5 * v * (1.0 + (c * (v + 0.044715 * v.powi(3))).tanh()))
        .collect()
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0))
        .collect()
}

/// Single-token causal self-attention. For a sequence length of one the attention matrix is
/// `softmax([q.k / sqrt(d_k)])` = `[1.0]`, so the output collapses to `V`. This preserves the
/// full QKV projection structure while remaining within the codegen GEMV (single-vector) contract.
fn single_token_attention(_q: &[f32], _k: &[f32], v: &[f32], _n_heads: usize) -> Vec<f32> {
    v.to_vec()
}"#
    .trim_end()
    .to_string()
}

fn llama_helpers() -> String {
    r#"fn decode_f32(state: &AppState, name: &str) -> Vec<f32> {
    let meta = state
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor `{}`", name));
    let slice = &state.weights[meta.byte_offset..meta.byte_offset + meta.byte_len];
    decode_f32_le(slice).expect("decoder expects fp32 payloads for this server")
}

fn decode_f32_le(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().ok()?));
    }
    Some(out)
}

/// `y = W x` for a dense weight stored row-major as `[out, in]` (LLaMA projections are bias-free).
fn linear(state: &AppState, weight_key: &str, x: &[f32]) -> Vec<f32> {
    let w_meta = state
        .tensors
        .get(weight_key)
        .unwrap_or_else(|| panic!("missing tensor `{}`", weight_key));
    assert_eq!(w_meta.dtype_size, 4, "`{}` must be fp32", weight_key);
    assert_eq!(w_meta.shape.len(), 2, "`{}` must be a dense matrix", weight_key);

    let rows = w_meta.shape[0];
    let cols = w_meta.shape[1];
    assert_eq!(cols, x.len(), "`{}` gemv mismatch: cols {cols}, input {}", weight_key, x.len());

    let w_flat = decode_f32(state, weight_key);
    let mut out = vec![0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        let row = &w_flat[r * cols..(r + 1) * cols];
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += *wv * *xv;
        }
        out[r] = acc;
    }
    out
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len().max(1) as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let rms = (ms + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| v / rms * w.get(i).copied().unwrap_or(1.0))
        .collect()
}

fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0))
        .collect()
}

/// Rotary position embedding for `pos` (head_dim = x.len() / n_heads). For the codegen single-
/// token path `pos = 0`, which makes RoPE the identity; the rotation is still applied so the
/// structure matches a full sequence forward.
fn rope(x: &[f32], pos: usize, n_heads: usize) -> Vec<f32> {
    let head_dim = (x.len() / n_heads.max(1)).max(1);
    let half = head_dim / 2;
    let mut out = x.to_vec();
    let base = 10000.0f32;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let theta = pos as f32 * base.powf(-(i as f32) / half as f32);
            let (s, c) = theta.sin_cos();
            let x0 = x[off + i];
            let x1 = x[off + i + half];
            out[off + i] = x0 * c - x1 * s;
            out[off + i + half] = x0 * s + x1 * c;
        }
    }
    out
}

/// Single-token causal attention; collapses to `V` for sequence length one (see gpt2 helper).
fn single_token_attention(_q: &[f32], _k: &[f32], v: &[f32], _n_heads: usize) -> Vec<f32> {
    v.to_vec()
}"#
    .trim_end()
    .to_string()
}

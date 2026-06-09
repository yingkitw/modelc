use crate::codegen::native::infer_mlp_plan;
use crate::model::Model;

pub(super) fn emit_mlp_helpers(model: &Model) -> String {
    if infer_mlp_plan(model).is_none() {
        return String::new();
    }

    r#"fn decode_f32(state: &AppState, name: &'static str) -> Vec<f32> {
    let meta = state.tensors.get(name).expect("tensor lookup");
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
    weight_key: &'static str,
    bias_key: &'static str,
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

use crate::model::Model;

pub(super) fn emit_forward_fn(model: &Model) -> String {
    match model.architecture.as_str() {
        "mlp" => {
            if let Some(plan) = super::infer_mlp_plan(model) {
                build_forward_from_plan(plan)
            } else {
                placeholder_forward()
            }
        }
        "gpt2" => build_gpt2_forward(model),
        "llama" => build_llama_forward(model),
        _ => placeholder_forward(),
    }
}

fn placeholder_forward() -> String {
    r#"fn forward(_state: &AppState, input: &[f32]) -> Vec<f32> {
    input.to_vec()
}"#
    .to_string()
}

fn build_gpt2_forward(_model: &Model) -> String {
    r#"fn forward(state: &AppState, input: &[f32]) -> Vec<f32> {
    let mut hidden = input.to_vec();
    let n_layers = state.tensors.keys().filter(|k| k.starts_with("transformer.h.")).count() / 12;
    for layer in 0..n_layers {
        let prefix = format!("transformer.h.{}", layer);
        if let Some(ln1) = state.tensors.get(&format!("{}.ln_1.weight", prefix)) {
            let w = decode_f32(state, &format!("{}.ln_1.weight", prefix));
            let b = decode_f32(state, &format!("{}.ln_1.bias", prefix));
            hidden = layer_norm_1d(&hidden, &w, &b);
        }
    }
    if let Some(wte) = state.tensors.get("transformer.wte.weight") {
        let w = decode_f32(state, "transformer.wte.weight");
        gemv(&w, wte.shape[1], &hidden)
    } else {
        hidden
    }
}

fn layer_norm_1d(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len().max(1) as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len().max(1) as f32;
    let std = (var + 1e-5).sqrt();
    x.iter().enumerate().map(|(i, v)| {
        let wi = w.get(i).copied().unwrap_or(1.0);
        let bi = b.get(i).copied().unwrap_or(0.0);
        (v - mean) / std * wi + bi
    }).collect()
}

fn gemv(weights: &[f32], cols: usize, x: &[f32]) -> Vec<f32> {
    let rows = weights.len() / cols.max(1);
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols.min(x.len()) {
            acc += weights[r * cols + c] * x[c];
        }
        out[r] = acc;
    }
    out
}"#
    .to_string()
}

fn build_llama_forward(_model: &Model) -> String {
    r#"fn forward(state: &AppState, input: &[f32]) -> Vec<f32> {
    let mut hidden = input.to_vec();
    let n_layers = state.tensors.keys().filter(|k| k.starts_with("model.layers.")).count() / 10;
    for layer in 0..n_layers {
        let prefix = format!("model.layers.{}", layer);
        if let Some(_) = state.tensors.get(&format!("{}.input_layernorm.weight", prefix)) {
            let w = decode_f32(state, &format!("{}.input_layernorm.weight", prefix));
            hidden = rms_norm_1d(&hidden, &w);
        }
        if let Some(_) = state.tensors.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
            let q = gemv_t(&decode_f32(state, &format!("{}.self_attn.q_proj.weight", prefix)), &hidden);
            let v = gemv_t(&decode_f32(state, &format!("{}.self_attn.v_proj.weight", prefix)), &hidden);
            hidden = q.iter().zip(v.iter()).map(|(a, b)| a + b).collect();
        }
        if let Some(_) = state.tensors.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
            let gate = gemv_t(&decode_f32(state, &format!("{}.mlp.gate_proj.weight", prefix)), &hidden);
            let up = gemv_t(&decode_f32(state, &format!("{}.mlp.up_proj.weight", prefix)), &hidden);
            let silu: Vec<f32> = gate.iter().map(|v| v * (1.0 / (1.0 + (-v).exp()))).collect();
            hidden = silu.iter().zip(up.iter()).map(|(a, b)| a * b).collect();
            let down = gemv_t(&decode_f32(state, &format!("{}.mlp.down_proj.weight", prefix)), &hidden);
            hidden = down;
        }
    }
    hidden
}

fn rms_norm_1d(x: &[f32], w: &[f32]) -> Vec<f32> {
    let sum_sq = x.iter().map(|v| v * v).sum::<f32>();
    let rms = (sum_sq / x.len().max(1) as f32 + 1e-6).sqrt();
    x.iter().enumerate().map(|(i, v)| {
        let wi = w.get(i).copied().unwrap_or(1.0);
        v / rms * wi
    }).collect()
}

fn gemv_t(weights: &[f32], x: &[f32]) -> Vec<f32> {
    let cols = x.len();
    let rows = weights.len() / cols.max(1);
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += weights[r * cols + c] * x[c];
        }
        out[r] = acc;
    }
    out
}"#
    .to_string()
}

fn build_forward_from_plan(plan: Vec<(String, String)>) -> String {
    debug_assert!(!plan.is_empty());

    if plan.len() == 1 {
        let (w, b) = &plan[0];
        return finalize_forward(&format!(
            "    matmul_bias(state, {:?}, {:?}, input)",
            w.as_str(),
            b.as_str()
        ));
    }

    let (w0, b0) = &plan[0];
    let mut body = format!(
        "    let mut cur = matmul_bias(state, {:?}, {:?}, input);\n    relu_inplace(&mut cur);\n",
        w0.as_str(),
        b0.as_str(),
    );

    let last_global = plan.len() - 1;
    for (global_idx, (w, b)) in plan.iter().enumerate().skip(1) {
        body.push_str(&format!(
            "    cur = matmul_bias(state, {:?}, {:?}, &cur);\n",
            w.as_str(),
            b.as_str(),
        ));

        if global_idx != last_global {
            body.push_str("    relu_inplace(&mut cur);\n");
        }
    }

    body.push_str("    cur");
    finalize_forward(&body)
}

fn finalize_forward(body: &str) -> String {
    format!(
        "fn forward(state: &AppState, input: &[f32]) -> Vec<f32> {{\n{}\n}}",
        body.trim_end()
    )
}

use crate::arch::{
    detect_layers, gpt2_head_count, gpt2_hidden_dim, has_pair, llama_head_count, llama_hidden_dim,
    llama_layers,
};
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

fn build_gpt2_forward(model: &Model) -> String {
    let layers = detect_layers(model, "transformer.h.");
    let hidden = gpt2_hidden_dim(model, &layers);
    let n_heads = gpt2_head_count(hidden);

    let mut body = String::new();
    body.push_str("    let mut hidden = input.to_vec();\n");

    for &layer in &layers {
        let p = format!("transformer.h.{layer}.");
        let attn_block = gpt2_attention_block(model, &p, hidden, n_heads);
        let mlp_block = gpt2_mlp_block(model, &p);
        if attn_block.is_empty() && mlp_block.is_empty() {
            continue;
        }
        body.push_str(&format!("    // --- transformer block {layer} ---\n"));
        body.push_str(&attn_block);
        body.push_str(&mlp_block);
    }

    // Final layer norm + output projection (lm_head, or tied via wte).
    if has_pair(model, "transformer.ln_f.weight", "transformer.ln_f.bias") {
        body.push_str("    {\n");
        body.push_str("        let w = decode_f32(state, \"transformer.ln_f.weight\");\n");
        body.push_str("        let b = decode_f32(state, \"transformer.ln_f.bias\");\n");
        body.push_str("        hidden = layer_norm(&hidden, &w, &b, 1e-5);\n");
        body.push_str("    }\n");
    }

    if model.tensors.contains_key("lm_head.weight") {
        let bias = if model.tensors.contains_key("lm_head.bias") {
            "Some(\"lm_head.bias\")"
        } else {
            "None"
        };
        body.push_str(&format!(
            "    hidden = linear(state, \"lm_head.weight\", {bias}, &hidden);\n"
        ));
    } else if model.tensors.contains_key("transformer.wte.weight") {
        // Weight-tied output projection: lm_head = wte^T.
        body.push_str("    hidden = linear(state, \"transformer.wte.weight\", None, &hidden);\n");
    }

    body.push_str("    hidden");
    finalize_forward(&body)
}

fn gpt2_attention_block(model: &Model, prefix: &str, hidden: usize, n_heads: usize) -> String {
    let ln = has_pair(
        model,
        &format!("{prefix}ln_1.weight"),
        &format!("{prefix}ln_1.bias"),
    );
    let c_attn = has_pair(
        model,
        &format!("{prefix}attn.c_attn.weight"),
        &format!("{prefix}attn.c_attn.bias"),
    );
    let c_proj = has_pair(
        model,
        &format!("{prefix}attn.c_proj.weight"),
        &format!("{prefix}attn.c_proj.bias"),
    );

    if !c_attn {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("    {\n");
    if ln {
        out.push_str(&format!(
            "        let w = decode_f32(state, \"{prefix}ln_1.weight\");\n"
        ));
        out.push_str(&format!(
            "        let b = decode_f32(state, \"{prefix}ln_1.bias\");\n"
        ));
        out.push_str("        let h = layer_norm(&hidden, &w, &b, 1e-5);\n");
    } else {
        out.push_str("        let h = hidden.clone();\n");
    }

    out.push_str(&format!(
        "        let qkv = linear(state, \"{prefix}attn.c_attn.weight\", Some(\"{prefix}attn.c_attn.bias\"), &h);\n"
    ));
    out.push_str(&format!(
        "        let (q, rest) = qkv.split_at({hidden});\n"
    ));
    out.push_str(&format!("        let (k, v) = rest.split_at({hidden});\n"));
    out.push_str(&format!(
        "        let attn = single_token_attention(q, k, v, {n_heads});\n"
    ));

    if c_proj {
        out.push_str(&format!(
            "        let proj = linear(state, \"{prefix}attn.c_proj.weight\", Some(\"{prefix}attn.c_proj.bias\"), &attn);\n"
        ));
        out.push_str("        hidden = add(&hidden, &proj);\n");
    } else {
        out.push_str("        hidden = add(&hidden, &attn);\n");
    }
    out.push_str("    }\n");
    out
}

fn gpt2_mlp_block(model: &Model, prefix: &str) -> String {
    let ln = has_pair(
        model,
        &format!("{prefix}ln_2.weight"),
        &format!("{prefix}ln_2.bias"),
    );
    let c_fc = has_pair(
        model,
        &format!("{prefix}mlp.c_fc.weight"),
        &format!("{prefix}mlp.c_fc.bias"),
    );
    let c_proj = has_pair(
        model,
        &format!("{prefix}mlp.c_proj.weight"),
        &format!("{prefix}mlp.c_proj.bias"),
    );

    if !c_fc || !c_proj {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("    {\n");
    if ln {
        out.push_str(&format!(
            "        let w = decode_f32(state, \"{prefix}ln_2.weight\");\n"
        ));
        out.push_str(&format!(
            "        let b = decode_f32(state, \"{prefix}ln_2.bias\");\n"
        ));
        out.push_str("        let h = layer_norm(&hidden, &w, &b, 1e-5);\n");
    } else {
        out.push_str("        let h = hidden.clone();\n");
    }

    out.push_str(&format!(
        "        let fc = linear(state, \"{prefix}mlp.c_fc.weight\", Some(\"{prefix}mlp.c_fc.bias\"), &h);\n"
    ));
    out.push_str("        let act = gelu(&fc);\n");
    out.push_str(&format!(
        "        let proj = linear(state, \"{prefix}mlp.c_proj.weight\", Some(\"{prefix}mlp.c_proj.bias\"), &act);\n"
    ));
    out.push_str("        hidden = add(&hidden, &proj);\n");
    out.push_str("    }\n");
    out
}

fn build_llama_forward(model: &Model) -> String {
    let layers = llama_layers(model);
    let hidden = llama_hidden_dim(model, &layers);
    let n_heads = llama_head_count(model, hidden);

    let mut body = String::new();
    body.push_str("    let mut hidden = input.to_vec();\n");

    for &layer in &layers {
        let p = format!("model.layers.{layer}.");
        let attn = llama_attention_block(model, &p, n_heads);
        let mlp = llama_mlp_block(model, &p);
        if attn.is_empty() && mlp.is_empty() {
            continue;
        }
        body.push_str(&format!("    // --- decoder layer {layer} ---\n"));
        body.push_str(&attn);
        body.push_str(&mlp);
    }

    if model.tensors.contains_key("model.norm.weight") {
        body.push_str("    {\n");
        body.push_str("        let w = decode_f32(state, \"model.norm.weight\");\n");
        body.push_str("        hidden = rms_norm(&hidden, &w, 1e-6);\n");
        body.push_str("    }\n");
    }

    if model.tensors.contains_key("lm_head.weight") {
        body.push_str("    hidden = linear(state, \"lm_head.weight\", &hidden);\n");
    } else if model.tensors.contains_key("model.embed_tokens.weight") {
        // Tied embeddings: project back through the token table.
        body.push_str("    hidden = linear(state, \"model.embed_tokens.weight\", &hidden);\n");
    }

    body.push_str("    hidden");
    finalize_forward(&body)
}

fn llama_attention_block(model: &Model, prefix: &str, n_heads: usize) -> String {
    let has_norm = model
        .tensors
        .contains_key(&format!("{prefix}input_layernorm.weight"));
    let q = model
        .tensors
        .contains_key(&format!("{prefix}self_attn.q_proj.weight"));
    let k = model
        .tensors
        .contains_key(&format!("{prefix}self_attn.k_proj.weight"));
    let v = model
        .tensors
        .contains_key(&format!("{prefix}self_attn.v_proj.weight"));
    let o = model
        .tensors
        .contains_key(&format!("{prefix}self_attn.o_proj.weight"));

    if !(q && k && v && o) {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("    {\n");
    if has_norm {
        out.push_str(&format!(
            "        let w = decode_f32(state, \"{prefix}input_layernorm.weight\");\n"
        ));
        out.push_str("        let h = rms_norm(&hidden, &w, 1e-6);\n");
    } else {
        out.push_str("        let h = hidden.clone();\n");
    }

    out.push_str(&format!(
        "        let q = linear(state, \"{prefix}self_attn.q_proj.weight\", &h);\n"
    ));
    out.push_str(&format!(
        "        let k = linear(state, \"{prefix}self_attn.k_proj.weight\", &h);\n"
    ));
    out.push_str(&format!(
        "        let v = linear(state, \"{prefix}self_attn.v_proj.weight\", &h);\n"
    ));
    // Position 0 → RoPE is identity, but keeps the full sequence structure intact.
    out.push_str(&format!("        let q = rope(&q, 0, {n_heads});\n"));
    out.push_str(&format!("        let k = rope(&k, 0, {n_heads});\n"));
    out.push_str(&format!(
        "        let attn = single_token_attention(&q, &k, &v, {n_heads});\n"
    ));
    out.push_str(&format!(
        "        let proj = linear(state, \"{prefix}self_attn.o_proj.weight\", &attn);\n"
    ));
    out.push_str("        hidden = add(&hidden, &proj);\n");
    out.push_str("    }\n");
    out
}

fn llama_mlp_block(model: &Model, prefix: &str) -> String {
    let has_norm = model
        .tensors
        .contains_key(&format!("{prefix}post_attention_layernorm.weight"));
    let gate = model
        .tensors
        .contains_key(&format!("{prefix}mlp.gate_proj.weight"));
    let up = model
        .tensors
        .contains_key(&format!("{prefix}mlp.up_proj.weight"));
    let down = model
        .tensors
        .contains_key(&format!("{prefix}mlp.down_proj.weight"));

    if !(gate && up && down) {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("    {\n");
    if has_norm {
        out.push_str(&format!(
            "        let w = decode_f32(state, \"{prefix}post_attention_layernorm.weight\");\n"
        ));
        out.push_str("        let h = rms_norm(&hidden, &w, 1e-6);\n");
    } else {
        out.push_str("        let h = hidden.clone();\n");
    }

    // SwiGLU: down(silu(gate(x)) * up(x))
    out.push_str(&format!(
        "        let gate = linear(state, \"{prefix}mlp.gate_proj.weight\", &h);\n"
    ));
    out.push_str(&format!(
        "        let up = linear(state, \"{prefix}mlp.up_proj.weight\", &h);\n"
    ));
    out.push_str("        let act = silu(&gate);\n");
    out.push_str(
        "        let prod: Vec<f32> = act.iter().zip(up.iter()).map(|(a, b)| a * b).collect();\n",
    );
    out.push_str(&format!(
        "        let proj = linear(state, \"{prefix}mlp.down_proj.weight\", &prod);\n"
    ));
    out.push_str("        hidden = add(&hidden, &proj);\n");
    out.push_str("    }\n");
    out
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

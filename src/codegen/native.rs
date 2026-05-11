use std::fs;
use std::io::{BufWriter, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::codegen::CodeGenerator;
use crate::model::{DataType, Model, TensorData};

const EMBEDDED_WEIGHTS_FILE: &str = "embedded_weights.bin";

pub struct NativeCodegen;

impl CodeGenerator for NativeCodegen {
    fn generate(&self, model: &Model, output_dir: &Path, listen: SocketAddr) -> Result<PathBuf> {
        let project_dir = output_dir.join("modelc_build");
        let src_dir = project_dir.join("src");

        fs::create_dir_all(&src_dir).with_context(|| "failed to create build directory")?;

        let mut names: Vec<&String> = model.tensors.keys().collect();
        names.sort();

        let blob_path = project_dir.join(EMBEDDED_WEIGHTS_FILE);
        let blob_file = fs::File::create(&blob_path)
            .with_context(|| "failed to create embedded weight blob")?;
        let mut writer = BufWriter::new(blob_file);

        let mut tensor_loads = String::new();
        let mut blob_offset = 0usize;

        for name in &names {
            let tensor = model.tensors.get(*name).expect("tensor key mismatch");
            let offset = blob_offset;
            let byte_len = tensor.data.len();
            writer
                .write_all(&tensor.data)
                .with_context(|| format!("writing tensor {name} into blob"))?;

            blob_offset = blob_offset.saturating_add(byte_len);

            let shape_fmt = format!("{:?}", tensor.shape);
            let dtype_size = tensor.dtype.byte_size();
            tensor_loads.push_str(&format!(
                "        ({:?}, TensorMeta {{ shape: &{shape_fmt}, dtype_size: {dtype_size}, byte_offset: {offset}, byte_len: {byte_len} }}),\n",
                name
            ));
        }

        writer
            .flush()
            .with_context(|| "flushing embedded weight blob")?;

        let cargo_toml = generate_cargo_toml();
        let listen_str = listen.to_string();
        let optional_helpers = emit_mlp_helpers(model);
        let forward_fn = emit_forward_fn(model);
        let main_rs = generate_main_rs(
            model,
            EMBEDDED_WEIGHTS_FILE,
            &tensor_loads,
            &listen_str,
            optional_helpers.trim_end(),
            forward_fn.trim_end(),
        );

        fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;
        fs::write(src_dir.join("main.rs"), main_rs)?;

        Ok(project_dir)
    }
}

fn generate_cargo_toml() -> String {
    r#"[package]
name = "model-serve"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[profile.release]
opt-level = 3
lto = true
strip = true
"#
    .to_string()
}

fn escape_rust_string_literal(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn generate_main_rs(
    model: &Model,
    weights_file: &str,
    tensor_loads: &str,
    listen: &str,
    optional_helpers: &str,
    forward_fn: &str,
) -> String {
    let model_name_esc = escape_rust_string_literal(&model.name);
    let arch_esc = escape_rust_string_literal(&model.architecture);
    let listen_esc = escape_rust_string_literal(listen);

    let total_params = model.total_params();
    let total_bytes = model.total_bytes();

    format!(
        r##"use std::collections::HashMap;
use std::sync::Arc;

use axum::{{Json, Router, extract::State, routing::{{get, post}}}};
use serde::{{Deserialize, Serialize}};

struct TensorMeta {{
    shape: &'static [usize],
    dtype_size: usize,
    byte_offset: usize,
    byte_len: usize,
}}

struct AppState {{
    weights: &'static [u8],
    tensors: HashMap<&'static str, TensorMeta>,
}}

#[derive(Deserialize)]
struct InferRequest {{
    input: Vec<f32>,
}}

#[derive(Serialize)]
struct InferResponse {{
    output: Vec<f32>,
}}

#[derive(Serialize)]
struct ModelInfo {{
    name: &'static str,
    architecture: &'static str,
    total_params: usize,
    total_bytes: usize,
    tensors: Vec<String>,
}}

const MODEL_NAME: &str = "{model_name_esc}";
const MODEL_ARCHITECTURE: &str = "{arch_esc}";

#[tokio::main]
async fn main() {{
    let weights: &'static [u8] = include_bytes!("../{weights_file}");

    let mut tensors = HashMap::new();
    let tensor_defs: Vec<(&str, TensorMeta)> = vec![
{tensor_loads}    ];
    for (name, meta) in tensor_defs {{
        tensors.insert(name, meta);
    }}

    let state = Arc::new(AppState {{ weights, tensors }});

    let app = Router::new()
        .route("/infer", post(infer))
        .route("/info", get(model_info))
        .with_state(state);

    let addr = "{listen_esc}"
        .parse::<std::net::SocketAddr>()
        .expect("embedded listen address");

    let total_mb = {total_bytes} as f64 / (1024.0 * 1024.0);
    eprintln!(
        "model-serve: listening on http://{{}}\n  model: {{}}\n  architecture: {{}}\n  parameters: {total_params}\n  weight blob: {total_bytes} bytes (~{{:.4}} MB)",
        addr,
        MODEL_NAME,
        MODEL_ARCHITECTURE,
        total_mb,
    );

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}}

async fn infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> Json<InferResponse> {{
    let result = forward(&state, &req.input);
    Json(InferResponse {{ output: result }})
}}

async fn model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {{
    Json(ModelInfo {{
        name: MODEL_NAME,
        architecture: MODEL_ARCHITECTURE,
        total_params: {total_params},
        total_bytes: {total_bytes},
        tensors: state.tensors.keys().map(|k| k.to_string()).collect(),
    }})
}}

{optional_helpers}

{forward_fn}
"##
    )
}

fn emit_mlp_helpers(model: &Model) -> String {
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

fn emit_forward_fn(model: &Model) -> String {
    if let Some(plan) = infer_mlp_plan(model) {
        build_forward_from_plan(plan)
    } else {
        placeholder_forward()
    }
}

fn placeholder_forward() -> String {
    r#"fn forward(_state: &AppState, input: &[f32]) -> Vec<f32> {
    input.to_vec()
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

fn infer_mlp_plan(model: &Model) -> Option<Vec<(String, String)>> {
    if model.architecture != "mlp" {
        return None;
    }

    layered_mlp_pairs(model).or_else(|| singleton_affine_pair(model))
}

fn singleton_affine_pair(model: &Model) -> Option<Vec<(String, String)>> {
    validate_affine_pair(model, "weight", "bias")?;
    Some(vec![("weight".to_string(), "bias".to_string())])
}

fn layered_mlp_pairs(model: &Model) -> Option<Vec<(String, String)>> {
    let mut ids: Vec<u32> = model
        .tensors
        .keys()
        .filter_map(|key| parse_layer_suffix(key.as_str()))
        .collect();
    if ids.is_empty() {
        return None;
    }

    ids.sort_unstable();
    ids.dedup();

    if !ids.windows(2).all(|pair| pair[1] == pair[0] + 1) {
        return None;
    }

    let mut seq = Vec::new();
    let mut prev_out_rows: Option<usize> = None;

    for id in ids {
        let weight_name = format!("layer{id}.weight");
        let bias_name = format!("layer{id}.bias");
        let (rows, cols) = affine_pair_shape(model, &weight_name, &bias_name)?;

        if let Some(out_prev) = prev_out_rows
            && out_prev != cols
        {
            return None;
        }

        seq.push((weight_name, bias_name));
        prev_out_rows = Some(rows);
    }

    Some(seq)
}

fn affine_pair_shape(model: &Model, weight_name: &str, bias_name: &str) -> Option<(usize, usize)> {
    validate_affine_pair(model, weight_name, bias_name)?;
    let w = model.tensors.get(weight_name)?;
    Some((*w.shape.first()?, *w.shape.get(1)?))
}

fn validate_affine_pair<'m>(
    model: &'m Model,
    weight_name: &str,
    bias_name: &str,
) -> Option<&'m TensorData> {
    let w = model.tensors.get(weight_name)?;
    let b = model.tensors.get(bias_name)?;

    if w.dtype != DataType::F32 || b.dtype != DataType::F32 {
        return None;
    }

    let rows = *w.shape.first()?;
    if w.shape.len() != 2 || b.shape.len() != 1 {
        return None;
    }

    (b.shape[0] == rows).then_some(w)
}

fn parse_layer_suffix(name: &str) -> Option<u32> {
    let tail = name.strip_prefix("layer")?;
    let (idx, suf) = tail.split_once('.')?;
    if suf != "weight" {
        return None;
    }

    idx.parse::<u32>().ok()
}

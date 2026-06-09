use std::collections::HashMap;

use crate::model::{DataType, TensorData};
use crate::runtime::serve::Runtime;
use crate::runtime::tensor::Tensor;

use super::AppState;

pub(super) fn runtime_to_tensor_data(runtime: &Runtime) -> HashMap<String, TensorData> {
    let names = runtime.tensor_names();
    let mut map = HashMap::new();
    for name in &names {
        if let Some(tensor) = runtime.get(*name) {
            let mut data = Vec::with_capacity(tensor.data.len() * 4);
            for &v in &tensor.data {
                data.extend_from_slice(&v.to_le_bytes());
            }
            map.insert(name.to_string(), TensorData {
                shape: tensor.shape.clone(),
                dtype: DataType::F32,
                data,
            });
        }
    }
    map
}

pub(super) fn run_inference(state: &AppState, input: &[f32]) -> Vec<f32> {
    if let Some(plan) = &state.onnx_plan {
        let runtime_tensors = runtime_to_tensor_data(&state.runtime);
        match crate::onnx_exec::execute_plan(plan, &runtime_tensors, input) {
            Ok(result) => return result,
            Err(e) => {
                eprintln!("  ONNX execution failed: {}, falling back", e);
            }
        }
    }
    if let Some(plan) = &state.mlp_plan {
        run_mlp_forward(&state.runtime, plan, input, state.profile)
    } else {
        input.to_vec()
    }
}

pub(super) fn run_text_inference(state: &AppState, prompt: &str) -> String {
    let input: Vec<f32> = prompt.bytes().map(|b| b as f32 / 255.0).collect();

    if let Some(plan) = &state.onnx_plan {
        let runtime_tensors = runtime_to_tensor_data(&state.runtime);
        if let Ok(result) = crate::onnx_exec::execute_plan(plan, &runtime_tensors, &input) {
            return serde_json::to_string(&result).unwrap_or_else(|_| "[]".to_string());
        }
    }

    let plan = state.mlp_plan.as_ref();
    let output = if let Some(plan) = plan {
        if let Some(w) = state.runtime.get(&plan[0].0) {
            let input_size = w.shape.get(1).copied().unwrap_or(input.len());
            let vec = if input.len() >= input_size {
                input[..input_size].to_vec()
            } else {
                let mut v = input;
                v.resize(input_size, 0.0);
                v
            };
            run_mlp_forward(&state.runtime, plan, &vec, state.profile)
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    serde_json::to_string(&output).unwrap_or_else(|_| "[]".to_string())
}

pub(super) fn run_mlp_forward(runtime: &Runtime, plan: &[(String, String)], input: &[f32], profile: bool) -> Vec<f32> {
    if plan.is_empty() {
        return input.to_vec();
    }

    let mut cur = input.to_vec();
    let last = plan.len() - 1;

    for (idx, (w_name, b_name)) in plan.iter().enumerate() {
        let w = runtime.get(w_name).expect("mlp weight missing");
        let b = runtime.get(b_name).expect("mlp bias missing");
        let start = std::time::Instant::now();
        cur = gemv_bias(w, b, &cur);
        if profile {
            eprintln!("    matmul+bias ({}): {:.3} ms", w_name, start.elapsed().as_secs_f64() * 1000.0);
        }
        if idx != last {
            let r_start = std::time::Instant::now();
            relu_inplace(&mut cur);
            if profile {
                eprintln!("    relu: {:.3} ms", r_start.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }

    cur
}

fn gemv_bias(weight: &Tensor, bias: &Tensor, x: &[f32]) -> Vec<f32> {
    assert_eq!(weight.shape.len(), 2, "weight must be 2D");
    assert_eq!(bias.shape.len(), 1, "bias must be 1D");
    let rows = weight.shape[0];
    let cols = weight.shape[1];
    assert_eq!(cols, x.len(), "gemv: input size mismatch");
    assert_eq!(bias.shape[0], rows, "gemv: bias size mismatch");

    let mut out = vec![0.0f32; rows];
    for (r, out_v) in out.iter_mut().enumerate().take(rows) {
        let mut acc = bias.data[r];
        let row = &weight.data[r * cols..(r + 1) * cols];
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += wv * xv;
        }
        *out_v = acc;
    }
    out
}

fn relu_inplace(xs: &mut [f32]) {
    for v in xs {
        *v = v.max(0.0);
    }
}

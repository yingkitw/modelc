use std::collections::HashMap;

use crate::model::{DataType, TensorData};
use crate::runtime::serve::Runtime;
use crate::runtime::tensor::Tensor;

use super::AppState;

pub(super) fn runtime_to_tensor_data(runtime: &Runtime) -> HashMap<String, TensorData> {
    let names = runtime.tensor_names();
    let mut map = HashMap::new();
    for name in &names {
        if let Some(tensor) = runtime.get(name) {
            let mut data = Vec::with_capacity(tensor.data.len() * 4);
            for &v in &tensor.data {
                data.extend_from_slice(&v.to_le_bytes());
            }
            map.insert(
                name.to_string(),
                TensorData {
                    shape: tensor.shape.clone(),
                    dtype: DataType::F32,
                    data,
                },
            );
        }
    }
    map
}

pub(super) fn run_inference(state: &AppState, input: &[f32]) -> Vec<f32> {
    let runtime = state.runtime.read().expect("runtime lock poisoned");
    if let Some(plan) = &state.onnx_plan {
        let runtime_tensors = runtime_to_tensor_data(&runtime);
        match crate::onnx_exec::execute_plan(plan, &runtime_tensors, input) {
            Ok(result) => return result,
            Err(e) => {
                eprintln!("  ONNX execution failed: {}, falling back", e);
            }
        }
    }
    if let Some(plan) = &state.mlp_plan {
        return run_mlp_forward(&runtime, plan, input, state.profile);
    }
    if let Some(output) = run_transformer(state, &runtime, input) {
        return output;
    }
    input.to_vec()
}

/// Execute a GPT-2 / LLaMA forward when the artifact is a transformer. Inputs are resized to
/// the model's hidden size (truncated or zero-padded) so `/infer` is tolerant of length
/// mismatches instead of erroring. Returns `None` for non-transformer models or when the
/// forward has no usable output head (caller then falls back to echo).
fn run_transformer(
    state: &AppState,
    runtime: &Runtime,
    input: &[f32],
) -> Option<Vec<f32>> {
    let hidden = state.transformer_hidden?;
    let resized = resize_to_hidden(input, hidden);
    let start = std::time::Instant::now();
    let out = match state.architecture.as_str() {
        "gpt2" => {
            crate::runtime::transformer::forward_gpt2(runtime, &resized, state.profile)
        }
        "llama" => {
            crate::runtime::transformer::forward_llama(runtime, &resized, state.profile)
        }
        _ => None,
    };
    if state.profile
        && let Some(o) = &out
    {
        eprintln!(
            "  transformer forward ({}): {} out dims in {:.3} ms",
            state.architecture,
            o.len(),
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    out
}

fn resize_to_hidden(input: &[f32], hidden: usize) -> Vec<f32> {
    if input.len() == hidden {
        return input.to_vec();
    }
    if input.len() > hidden {
        input[..hidden].to_vec()
    } else {
        let mut v = input.to_vec();
        v.resize(hidden, 0.0);
        v
    }
}

pub(super) fn run_embeddings(state: &AppState, input: &[f32]) -> Option<Vec<f32>> {
    let hidden = state.transformer_hidden?;
    let resized = resize_to_hidden(input, hidden);
    let runtime = state.runtime.read().expect("runtime lock poisoned");
    match state.architecture.as_str() {
        "gpt2" => crate::runtime::transformer::embed_gpt2(&runtime, &resized, state.profile),
        "llama" => {
            crate::runtime::transformer::embed_llama(&runtime, &resized, state.profile)
        }
        _ => None,
    }
}

pub(super) fn run_text_inference_with_config(
    state: &AppState,
    prompt: &str,
    config: &crate::generate::GenerationConfig,
) -> String {
    let input: Vec<f32> = prompt.bytes().map(|b| b as f32 / 255.0).collect();

    // For transformer models, use autoregressive generation (with the prefix cache).
    if (state.architecture == "gpt2" || state.architecture == "llama")
        && let Some(hidden) = state.transformer_hidden
    {
        let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
        let prompt_ids = tokenizer.encode(prompt);
        // Brief read lock for prefix cache lookup.
        let lookup = {
            let guard = state.prefix_cache.read().expect("prefix cache poisoned");
            guard.lookup(&prompt_ids)
        };
        let runtime = state.runtime.read().expect("runtime lock poisoned");
        let (ids, maybe_kv) = if config.gamma > 0 {
            let (text, kv) = crate::generate::speculative_generate(
                &runtime,
                &state.architecture,
                hidden,
                &tokenizer,
                prompt,
                config,
                lookup,
                state.draft_model.as_ref().map(|dm| dm.as_ref()),
            );
            let prompt_ids = tokenizer.encode(prompt);
            let all_ids = tokenizer.encode(&(prompt.to_string() + &text));
            let ids = if all_ids.len() > prompt_ids.len() {
                all_ids[prompt_ids.len()..].to_vec()
            } else {
                Vec::new()
            };
            (ids, kv)
        } else {
            let (ids, _, kv) = crate::generate::generate_core(
                &runtime,
                &state.architecture,
                hidden,
                &tokenizer,
                prompt,
                config,
                lookup,
                false,
                0,
            );
            (ids, kv)
        };
        // Brief write lock for prefix cache insertion.
        if let Some(kv) = maybe_kv {
            let mut guard = state.prefix_cache.write().expect("prefix cache poisoned");
            let mut full_ids = prompt_ids.clone();
            full_ids.extend_from_slice(&ids);
            guard.insert(
                full_ids,
                crate::prefix_cache::CachedPrefix {
                    kv,
                    last_logits: Vec::new(),
                },
            );
        }
        return tokenizer.decode(&ids);
    }

    let runtime = state.runtime.read().expect("runtime lock poisoned");
    if let Some(plan) = &state.onnx_plan {
        let runtime_tensors = runtime_to_tensor_data(&runtime);
        if let Ok(result) = crate::onnx_exec::execute_plan(plan, &runtime_tensors, &input) {
            return serde_json::to_string(&result).unwrap_or_else(|_| "[]".to_string());
        }
    }

    if let Some(output) = run_transformer(state, &runtime, &input) {
        return serde_json::to_string(&output).unwrap_or_else(|_| "[]".to_string());
    }

    let plan = state.mlp_plan.as_ref();
    let output = if let Some(plan) = plan {
        if let Some(w) = runtime.get(&plan[0].0) {
            let input_size = w.shape.get(1).copied().unwrap_or(input.len());
            let vec = if input.len() >= input_size {
                input[..input_size].to_vec()
            } else {
                let mut v = input;
                v.resize(input_size, 0.0);
                v
            };
            run_mlp_forward(&runtime, plan, &vec, state.profile)
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    serde_json::to_string(&output).unwrap_or_else(|_| "[]".to_string())
}

/// Generate token IDs for transformer models, returning the newly generated IDs.
/// Non-transformer models return an empty Vec.
pub(super) fn run_text_inference_token_ids(
    state: &AppState,
    prompt: &str,
    config: &crate::generate::GenerationConfig,
) -> Vec<u32> {
    if (state.architecture == "gpt2" || state.architecture == "llama")
        && let Some(hidden) = state.transformer_hidden
    {
        let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
        let prompt_ids = tokenizer.encode(prompt);
        let lookup = {
            let guard = state.prefix_cache.read().expect("prefix cache poisoned");
            guard.lookup(&prompt_ids)
        };
        let runtime = state.runtime.read().expect("runtime lock poisoned");
        let (ids, maybe_kv) = if config.gamma > 0 {
            let (text, kv) = crate::generate::speculative_generate(
                &runtime,
                &state.architecture,
                hidden,
                &tokenizer,
                prompt,
                config,
                lookup,
                state.draft_model.as_ref().map(|dm| dm.as_ref()),
            );
            let prompt_ids = tokenizer.encode(prompt);
            let all_ids = tokenizer.encode(&(prompt.to_string() + &text));
            let ids = if all_ids.len() > prompt_ids.len() {
                all_ids[prompt_ids.len()..].to_vec()
            } else {
                Vec::new()
            };
            (ids, kv)
        } else {
            let (ids, _, kv) = crate::generate::generate_core(
                &runtime,
                &state.architecture,
                hidden,
                &tokenizer,
                prompt,
                config,
                lookup,
                false,
                0,
            );
            (ids, kv)
        };
        if let Some(kv) = maybe_kv {
            let mut guard = state.prefix_cache.write().expect("prefix cache poisoned");
            let mut full_ids = prompt_ids.clone();
            full_ids.extend_from_slice(&ids);
            guard.insert(
                full_ids,
                crate::prefix_cache::CachedPrefix {
                    kv,
                    last_logits: Vec::new(),
                },
            );
        }
        return ids;
    }
    Vec::new()
}

/// Generate token IDs for a transformer model, also returning per-token logprobs.
///
/// Returns `None` for non-transformer architectures (caller should fall back to
/// `run_text_inference_with_config`). When successful, returns the newly
/// generated token IDs and one `TokenLogprob` per generated token.
pub(super) fn run_text_inference_with_logprobs(
    state: &AppState,
    prompt: &str,
    config: &crate::generate::GenerationConfig,
    top_logprobs: usize,
) -> Option<(Vec<u32>, Vec<crate::generate::TokenLogprob>)> {
    let hidden = state.transformer_hidden?;
    if state.architecture != "gpt2" && state.architecture != "llama" {
        return None;
    }
    let tokenizer = crate::tokenizer::BpeTokenizer::byte_fallback();
    let prompt_ids = tokenizer.encode(prompt);
    let lookup = {
        let guard = state.prefix_cache.read().expect("prefix cache poisoned");
        guard.lookup(&prompt_ids)
    };
    let runtime = state.runtime.read().expect("runtime lock poisoned");
    let (ids, logprobs, maybe_kv) = crate::generate::generate_core(
        &runtime,
        &state.architecture,
        hidden,
        &tokenizer,
        prompt,
        config,
        lookup,
        true,
        top_logprobs,
    );
    if let Some(kv) = maybe_kv {
        let mut guard = state.prefix_cache.write().expect("prefix cache poisoned");
        let mut full_ids = prompt_ids.clone();
        full_ids.extend_from_slice(&ids);
        guard.insert(
            full_ids,
            crate::prefix_cache::CachedPrefix {
                kv,
                last_logits: Vec::new(),
            },
        );
    }
    Some((ids, logprobs))
}

pub(super) fn run_mlp_forward(
    runtime: &Runtime,
    plan: &[(String, String)],
    input: &[f32],
    profile: bool,
) -> Vec<f32> {
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
            eprintln!(
                "    matmul+bias ({}): {:.3} ms",
                w_name,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        if idx != last {
            let r_start = std::time::Instant::now();
            relu_inplace(&mut cur);
            if profile {
                eprintln!(
                    "    relu: {:.3} ms",
                    r_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
    }

    cur
}

/// Batched MLP forward: run the same MLP plan on multiple inputs in one pass.
/// Each input must have the same length. Returns one output vector per input.
pub(super) fn run_mlp_forward_batched(
    runtime: &Runtime,
    plan: &[(String, String)],
    inputs: &[Vec<f32>],
    profile: bool,
) -> Vec<Vec<f32>> {
    if plan.is_empty() {
        return inputs.to_vec();
    }

    let mut cur = inputs.to_vec();
    let last = plan.len() - 1;

    for (idx, (w_name, b_name)) in plan.iter().enumerate() {
        let w = runtime.get(w_name).expect("mlp weight missing");
        let b = runtime.get(b_name).expect("mlp bias missing");
        let start = std::time::Instant::now();
        let refs: Vec<&[f32]> = cur.iter().map(|v| v.as_slice()).collect();
        cur = batched_gemv_bias(w, b, &refs);
        if profile {
            eprintln!(
                "    batched matmul+bias ({}): {} items, {:.3} ms",
                w_name,
                inputs.len(),
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        if idx != last {
            let r_start = std::time::Instant::now();
            for v in &mut cur {
                relu_inplace(v);
            }
            if profile {
                eprintln!(
                    "    batched relu: {:.3} ms",
                    r_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
    }

    cur
}

fn batched_gemv_bias(weight: &Tensor, bias: &Tensor, inputs: &[&[f32]]) -> Vec<Vec<f32>> {
    assert_eq!(weight.shape.len(), 2, "weight must be 2D");
    assert_eq!(bias.shape.len(), 1, "bias must be 1D");
    let rows = weight.shape[0];
    let cols = weight.shape[1];
    let batch = inputs.len();
    assert_eq!(bias.shape[0], rows, "batched gemv: bias size mismatch");
    for (i, inp) in inputs.iter().enumerate() {
        assert_eq!(inp.len(), cols, "batched gemv: input {i} size mismatch");
    }

    let mut outs = vec![vec![0.0f32; rows]; batch];
    for (r, row) in weight.data.chunks_exact(cols).enumerate().take(rows) {
        let b = bias.data[r];
        for (batch_idx, inp) in inputs.iter().enumerate() {
            let mut acc = b;
            for (wv, xv) in row.iter().zip(inp.iter()) {
                acc += wv * xv;
            }
            outs[batch_idx][r] = acc;
        }
    }
    outs
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

#[cfg(test)]
mod tests {
    use crate::runtime::tensor::Tensor;

    use super::batched_gemv_bias;

    #[test]
    fn batched_gemv_two_inputs() {
        let weight = Tensor::from_vec(
            vec![
                1.0, 0.0, // row 0
                0.0, 1.0, // row 1
            ],
            vec![2, 2],
        );
        let bias = Tensor::from_vec(vec![10.0, 20.0], vec![2]);
        let inputs: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let outs = batched_gemv_bias(&weight, &bias, &refs);
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0], vec![11.0, 22.0]); // [10+1, 20+2]
        assert_eq!(outs[1], vec![13.0, 24.0]); // [10+3, 20+4]
    }

    #[test]
    fn batched_gemv_single_input() {
        let weight = Tensor::from_vec(vec![2.0, 3.0], vec![1, 2]);
        let bias = Tensor::from_vec(vec![5.0], vec![1]);
        let inputs: Vec<Vec<f32>> = vec![vec![1.0, 1.0]];
        let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let outs = batched_gemv_bias(&weight, &bias, &refs);
        assert_eq!(outs.len(), 1);
        assert!(
            (outs[0][0] - 10.0).abs() < 1e-6,
            "expected 10.0, got {}",
            outs[0][0]
        );
    }
}

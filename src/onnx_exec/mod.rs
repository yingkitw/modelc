//! ONNX graph execution engine.
//!
//! Parses ONNX graph nodes into an execution plan and runs inference
//! using the existing tensor runtime ops.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::model::{DataType, TensorData};
use crate::runtime::ops;
use crate::runtime::tensor::Tensor;

mod helpers;

/// A single operation in the ONNX execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Op {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    #[serde(default)]
    pub attrs: HashMap<String, AttributeValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<String>),
}

/// Execution plan built from an ONNX graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub ops: Vec<Op>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl ExecutionPlan {
    /// Serialize the plan to a JSON string for storage in model metadata.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("failed to serialize execution plan")
    }

    /// Deserialize from a JSON string stored in model metadata.
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("failed to deserialize execution plan")
    }
}

/// Build an execution plan from an ONNX graph.
pub fn build_plan(graph: &onnx_rs::ast::Graph) -> Result<ExecutionPlan> {
    use onnx_rs::ast::AttributeType;

    let mut ops = Vec::new();

    for node in &graph.node {
        let op_type = node.op_type.as_str().to_string();
        let inputs: Vec<String> = node.input.iter().map(|s| s.to_string()).collect();
        let outputs: Vec<String> = node.output.iter().map(|s| s.to_string()).collect();

        let mut attrs = HashMap::new();
        for attr in &node.attribute {
            let key = attr.name.to_string();
            let val = match attr.r#type {
                AttributeType::Int => AttributeValue::Int(attr.i),
                AttributeType::Float => AttributeValue::Float(attr.f),
                AttributeType::String => {
                    AttributeValue::String(String::from_utf8_lossy(attr.s).to_string())
                }
                AttributeType::Ints => AttributeValue::Ints(attr.ints.clone()),
                AttributeType::Floats => AttributeValue::Floats(attr.floats.clone()),
                AttributeType::Strings => AttributeValue::Strings(
                    attr.strings
                        .iter()
                        .map(|s| String::from_utf8_lossy(s).to_string())
                        .collect(),
                ),
                _ => continue,
            };
            attrs.insert(key, val);
        }

        ops.push(Op {
            op_type,
            inputs,
            outputs,
            attrs,
        });
    }

    let plan_inputs: Vec<String> = graph.input.iter().map(|v| v.name.to_string()).collect();
    let plan_outputs: Vec<String> = graph.output.iter().map(|v| v.name.to_string()).collect();

    Ok(ExecutionPlan {
        ops,
        inputs: plan_inputs,
        outputs: plan_outputs,
    })
}

/// Run the execution plan with the given weight tensors and input vector.
pub fn execute_plan(
    plan: &ExecutionPlan,
    weights: &HashMap<String, TensorData>,
    input: &[f32],
) -> Result<Vec<f32>> {
    let mut env: HashMap<String, Tensor> = HashMap::new();

    // Load all weight tensors into the environment.
    for (name, td) in weights {
        let count = td.element_count();
        let fdata: Option<Vec<f32>> = match td.dtype {
            DataType::F32 => {
                let byte_len = count * 4;
                if td.data.len() < byte_len {
                    None
                } else {
                    Some(
                        td.data[..byte_len]
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                            .collect(),
                    )
                }
            }
            DataType::F16 => {
                let byte_len = count * 2;
                if td.data.len() < byte_len {
                    None
                } else {
                    Some(
                        td.data[..byte_len]
                            .chunks_exact(2)
                            .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                            .collect(),
                    )
                }
            }
            DataType::I8 => {
                if td.data.len() < count {
                    None
                } else {
                    Some(td.data[..count].iter().map(|&b| b as i8 as f32).collect())
                }
            }
            _ => None,
        };
        if let Some(fdata) = fdata {
            env.insert(name.clone(), Tensor::from_vec(fdata, td.shape.clone()));
        }
    }

    // Bind the first input to the provided data (may override a weight if names collide).
    if let Some(first_input) = plan.inputs.first() {
        let input_tensor = if let Some(w) = weights.values().next() {
            let target_size = w.shape.last().copied().unwrap_or(input.len());
            let vec = if input.len() >= target_size {
                input[..target_size].to_vec()
            } else {
                let mut v = input.to_vec();
                v.resize(target_size, 0.0);
                v
            };
            Tensor::from_vec(vec, vec![1, target_size])
        } else {
            Tensor::from_vec(input.to_vec(), vec![1, input.len()])
        };
        env.insert(first_input.clone(), input_tensor);
    }

    for op in &plan.ops {
        match op.op_type.as_str() {
            "MatMul" | "Matmul" => {
                ensure!(op.inputs.len() >= 2, "MatMul needs 2 inputs");
                let a = env.get(&op.inputs[0]).context("MatMul missing A")?;
                let b = env.get(&op.inputs[1]).context("MatMul missing B")?;
                let out = ops::matmul(a, b);
                env.insert(op.outputs[0].clone(), out);
            }
            "Gemm" => {
                ensure!(op.inputs.len() >= 2, "Gemm needs at least 2 inputs");
                let a = env.get(&op.inputs[0]).context("Gemm missing A")?;
                let b = env.get(&op.inputs[1]).context("Gemm missing B")?;
                let alpha = helpers::get_attr_f32(&op.attrs, "alpha", 1.0);
                let beta = helpers::get_attr_f32(&op.attrs, "beta", 1.0);
                let trans_a = helpers::get_attr_int(&op.attrs, "transA", 0) != 0;
                let trans_b = helpers::get_attr_int(&op.attrs, "transB", 0) != 0;

                let a_t = if trans_a {
                    helpers::transpose(a)
                } else {
                    Tensor::from_vec(a.data.clone(), a.shape.clone())
                };
                let b_t = if trans_b {
                    helpers::transpose(b)
                } else {
                    Tensor::from_vec(b.data.clone(), b.shape.clone())
                };
                let mut out = ops::matmul(&a_t, &b_t);
                if alpha != 1.0 {
                    out = ops::mul_scalar(&out, alpha);
                }
                if op.inputs.len() >= 3 {
                    let c = env.get(&op.inputs[2]).context("Gemm missing C")?;
                    let bias = if beta != 1.0 {
                        ops::mul_scalar(c, beta)
                    } else {
                        Tensor::from_vec(c.data.clone(), c.shape.clone())
                    };
                    out = ops::add(&out, &bias);
                }
                env.insert(op.outputs[0].clone(), out);
            }
            "Add" => {
                ensure!(op.inputs.len() >= 2, "Add needs 2 inputs");
                let a = env.get(&op.inputs[0]).context("Add missing A")?;
                let b = env.get(&op.inputs[1]).context("Add missing B")?;
                let out = ops::add(a, b);
                env.insert(op.outputs[0].clone(), out);
            }
            "Mul" => {
                ensure!(op.inputs.len() >= 2, "Mul needs 2 inputs");
                let a = env.get(&op.inputs[0]).context("Mul missing A")?;
                let b = env.get(&op.inputs[1]).context("Mul missing B")?;
                let out = helpers::element_wise_mul(a, b)?;
                env.insert(op.outputs[0].clone(), out);
            }
            "Div" => {
                ensure!(op.inputs.len() >= 2, "Div needs 2 inputs");
                let a = env.get(&op.inputs[0]).context("Div missing A")?;
                let b = env.get(&op.inputs[1]).context("Div missing B")?;
                let out = helpers::element_wise_div(a, b)?;
                env.insert(op.outputs[0].clone(), out);
            }
            "Sub" => {
                ensure!(op.inputs.len() >= 2, "Sub needs 2 inputs");
                let a = env.get(&op.inputs[0]).context("Sub missing A")?;
                let b = env.get(&op.inputs[1]).context("Sub missing B")?;
                let out = helpers::element_wise_sub(a, b)?;
                env.insert(op.outputs[0].clone(), out);
            }
            "Relu" => {
                ensure!(!op.inputs.is_empty(), "Relu needs input");
                let a = env.get(&op.inputs[0]).context("Relu missing input")?;
                let out = ops::relu(a);
                env.insert(op.outputs[0].clone(), out);
            }
            "Softmax" => {
                ensure!(!op.inputs.is_empty(), "Softmax needs input");
                let a = env.get(&op.inputs[0]).context("Softmax missing input")?;
                let axis = helpers::get_attr_int(&op.attrs, "axis", -1);
                let axis_usize = if axis < 0 {
                    a.shape.len().saturating_sub(1)
                } else {
                    axis as usize
                };
                let out = ops::softmax(a, axis_usize.min(a.shape.len().saturating_sub(1)));
                env.insert(op.outputs[0].clone(), out);
            }
            "LayerNormalization" | "LayerNorm" => {
                ensure!(op.inputs.len() >= 3, "LayerNorm needs 3 inputs");
                let x = env.get(&op.inputs[0]).context("LayerNorm missing X")?;
                let scale = env.get(&op.inputs[1]).context("LayerNorm missing scale")?;
                let bias = env.get(&op.inputs[2]).context("LayerNorm missing bias")?;
                let eps = helpers::get_attr_f32(&op.attrs, "epsilon", 1e-5);
                let out = ops::layer_norm(x, scale, bias, eps);
                env.insert(op.outputs[0].clone(), out);
            }
            "Transpose" => {
                ensure!(!op.inputs.is_empty(), "Transpose needs input");
                let a = env.get(&op.inputs[0]).context("Transpose missing input")?;
                let perm = helpers::get_attr_ints(&op.attrs, "perm");
                let out = if let Some(p) = perm {
                    helpers::transpose_with_perm(a, &p)?
                } else {
                    helpers::transpose(a)
                };
                env.insert(op.outputs[0].clone(), out);
            }
            "Reshape" => {
                ensure!(op.inputs.len() >= 2, "Reshape needs 2 inputs");
                let a = env.get(&op.inputs[0]).context("Reshape missing data")?;
                let shape = if let Some(shape_tensor) = env.get(&op.inputs[1]) {
                    shape_tensor.data.iter().map(|v| *v as usize).collect()
                } else {
                    a.shape.clone()
                };
                let out = a.reshape(shape);
                env.insert(op.outputs[0].clone(), out);
            }
            "Constant" => {
                continue;
            }
            "Identity" => {
                ensure!(!op.inputs.is_empty(), "Identity needs input");
                let a = env.get(&op.inputs[0]).context("Identity missing input")?;
                env.insert(
                    op.outputs[0].clone(),
                    Tensor::from_vec(a.data.clone(), a.shape.clone()),
                );
            }
            "Cast" => {
                ensure!(!op.inputs.is_empty(), "Cast needs input");
                let a = env.get(&op.inputs[0]).context("Cast missing input")?;
                env.insert(
                    op.outputs[0].clone(),
                    Tensor::from_vec(a.data.clone(), a.shape.clone()),
                );
            }
            "Sigmoid" => {
                ensure!(!op.inputs.is_empty(), "Sigmoid needs input");
                let a = env.get(&op.inputs[0]).context("Sigmoid missing input")?;
                let out = helpers::sigmoid(a);
                env.insert(op.outputs[0].clone(), out);
            }
            "Tanh" => {
                ensure!(!op.inputs.is_empty(), "Tanh needs input");
                let a = env.get(&op.inputs[0]).context("Tanh missing input")?;
                let out = helpers::tanh_tensor(a);
                env.insert(op.outputs[0].clone(), out);
            }
            unsupported => {
                eprintln!("  onnx_exec: unsupported op '{}' — skipping", unsupported);
                continue;
            }
        }
    }

    // Collect outputs
    let mut result = Vec::new();
    for out_name in &plan.outputs {
        if let Some(tensor) = env.get(out_name) {
            result.extend_from_slice(&tensor.data);
        } else {
            anyhow::bail!("output tensor '{}' not produced by graph", out_name);
        }
    }

    Ok(result)
}

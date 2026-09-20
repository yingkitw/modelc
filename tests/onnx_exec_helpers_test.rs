//! Direct tests for `onnx_exec::execute_plan` — every supported op is
//! dispatched against a hand-built `ExecutionPlan` so the dispatch table
//! stays exercised end-to-end. The helpers (transpose, attribute getters,
//! element-wise ops, activations) are tested inline in `src/onnx_exec/helpers_tests.rs`.

use std::collections::HashMap;

use modelc::model::{DataType, TensorData};
use modelc::onnx_exec::{AttributeValue, ExecutionPlan, Op};

fn tensor_data_f32(values: &[f32], shape: Vec<usize>) -> TensorData {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorData {
        shape,
        dtype: DataType::F32,
        data: bytes,
    }
}

fn plan_with(
    op_type: &str,
    inputs: Vec<&str>,
    outputs: Vec<&str>,
    attrs: HashMap<String, AttributeValue>,
) -> ExecutionPlan {
    let outputs: Vec<String> = outputs.into_iter().map(String::from).collect();
    ExecutionPlan {
        ops: vec![Op {
            op_type: op_type.to_string(),
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.clone(),
            attrs,
        }],
        inputs: vec!["x".to_string()],
        outputs,
    }
}

#[test]
fn execute_plan_matmul() {
    let mut weights = HashMap::new();
    weights.insert(
        "w".to_string(),
        tensor_data_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    );
    let plan = plan_with("MatMul", vec!["x", "w"], vec!["y"], HashMap::new());
    let out = modelc::onnx_exec::execute_plan(&plan, &weights, &[1.0, 2.0]).unwrap();
    // [[1,2],[3,4]] (2x2) · [1,2]ᵀ (1x2 → 2x1) would normally give [5,11],
    // but execute_plan resizes the input as [1, target_size] so the math is
    // [1,2] × [[1,2],[3,4]] = [1*1+2*3, 1*2+2*4] = [7, 10].
    assert_eq!(out, vec![7.0, 10.0]);
}

#[test]
fn execute_plan_gemm_with_alpha_beta_and_transpose() {
    let mut weights = HashMap::new();
    // a (1x2) = [1, 2]; b (2x2) = identity; c (1x2) = [10, 20]
    weights.insert("a".to_string(), tensor_data_f32(&[1.0, 2.0], vec![1, 2]));
    weights.insert(
        "b".to_string(),
        tensor_data_f32(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]),
    );
    weights.insert("c".to_string(), tensor_data_f32(&[10.0, 20.0], vec![1, 2]));
    let mut attrs = HashMap::new();
    attrs.insert("alpha".to_string(), AttributeValue::Float(2.0));
    attrs.insert("beta".to_string(), AttributeValue::Float(0.5));
    let plan = plan_with("Gemm", vec!["a", "b", "c"], vec!["y"], attrs);
    let out = modelc::onnx_exec::execute_plan(&plan, &weights, &[0.0]).unwrap();
    // α·a·b + β·c = 2·[1,2] + 0.5·[10,20] = [2,4] + [5,10] = [7,14]
    assert_eq!(out, vec![7.0, 14.0]);
}

#[test]
fn execute_plan_add_sub_mul_div() {
    let mut weights = HashMap::new();
    weights.insert("a".to_string(), tensor_data_f32(&[2.0, 4.0, 6.0], vec![3]));
    weights.insert("b".to_string(), tensor_data_f32(&[1.0, 2.0, 3.0], vec![3]));

    let p_add = plan_with("Add", vec!["a", "b"], vec!["y"], HashMap::new());
    assert_eq!(
        modelc::onnx_exec::execute_plan(&p_add, &weights, &[0.0]).unwrap(),
        vec![3.0, 6.0, 9.0]
    );

    let p_sub = plan_with("Sub", vec!["a", "b"], vec!["y"], HashMap::new());
    assert_eq!(
        modelc::onnx_exec::execute_plan(&p_sub, &weights, &[0.0]).unwrap(),
        vec![1.0, 2.0, 3.0]
    );

    let p_mul = plan_with("Mul", vec!["a", "b"], vec!["y"], HashMap::new());
    assert_eq!(
        modelc::onnx_exec::execute_plan(&p_mul, &weights, &[0.0]).unwrap(),
        vec![2.0, 8.0, 18.0]
    );

    let p_div = plan_with("Div", vec!["a", "b"], vec!["y"], HashMap::new());
    assert_eq!(
        modelc::onnx_exec::execute_plan(&p_div, &weights, &[0.0]).unwrap(),
        vec![2.0, 2.0, 2.0]
    );
}

#[test]
fn execute_plan_relu_softmax_sigmoid_tanh() {
    let mut weights = HashMap::new();
    weights.insert(
        "a".to_string(),
        tensor_data_f32(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]),
    );

    let p_relu = plan_with("Relu", vec!["a"], vec!["y"], HashMap::new());
    assert_eq!(
        modelc::onnx_exec::execute_plan(&p_relu, &weights, &[0.0]).unwrap(),
        vec![0.0, 0.0, 0.0, 1.0, 2.0]
    );

    let p_softmax = plan_with("Softmax", vec!["a"], vec!["y"], HashMap::new());
    let sm = modelc::onnx_exec::execute_plan(&p_softmax, &weights, &[0.0]).unwrap();
    let sum: f32 = sm.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sums to 1, got {sum}");

    let p_sig = plan_with("Sigmoid", vec!["a"], vec!["y"], HashMap::new());
    let sg = modelc::onnx_exec::execute_plan(&p_sig, &weights, &[0.0]).unwrap();
    assert!((sg[2] - 0.5).abs() < 1e-6);

    let p_tanh = plan_with("Tanh", vec!["a"], vec!["y"], HashMap::new());
    let th = modelc::onnx_exec::execute_plan(&p_tanh, &weights, &[0.0]).unwrap();
    assert_eq!(th[2], 0.0);
}

#[test]
fn execute_plan_layer_norm() {
    let mut weights = HashMap::new();
    weights.insert(
        "x".to_string(),
        tensor_data_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]),
    );
    weights.insert(
        "scale".to_string(),
        tensor_data_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]),
    );
    weights.insert(
        "bias".to_string(),
        tensor_data_f32(&[0.0, 0.0, 0.0, 0.0], vec![4]),
    );
    let plan = plan_with(
        "LayerNormalization",
        vec!["x", "scale", "bias"],
        vec!["y"],
        HashMap::new(),
    );
    let out = modelc::onnx_exec::execute_plan(&plan, &weights, &[0.0]).unwrap();
    let mean: f32 = out.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5);
}

#[test]
fn execute_plan_transpose_default_and_with_perm() {
    let mut weights = HashMap::new();
    weights.insert(
        "a".to_string(),
        tensor_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
    );
    // Default transpose (no `perm` attr) → swap axes for 2-D input.
    let p_default = plan_with("Transpose", vec!["a"], vec!["y"], HashMap::new());
    assert_eq!(
        modelc::onnx_exec::execute_plan(&p_default, &weights, &[0.0]).unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    // With perm [1, 0] — same outcome.
    let mut attrs = HashMap::new();
    attrs.insert("perm".to_string(), AttributeValue::Ints(vec![1, 0]));
    let p_perm = plan_with("Transpose", vec!["a"], vec!["y"], attrs);
    assert_eq!(
        modelc::onnx_exec::execute_plan(&p_perm, &weights, &[0.0]).unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn execute_plan_reshape_uses_second_input_as_shape() {
    let mut weights = HashMap::new();
    weights.insert(
        "a".to_string(),
        tensor_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
    );
    weights.insert("shape".to_string(), tensor_data_f32(&[3.0, 2.0], vec![2]));
    let plan = plan_with("Reshape", vec!["a", "shape"], vec!["y"], HashMap::new());
    let out = modelc::onnx_exec::execute_plan(&plan, &weights, &[0.0]).unwrap();
    assert_eq!(out.len(), 6);
}

#[test]
fn execute_plan_identity_passes_through() {
    let mut weights = HashMap::new();
    weights.insert("a".to_string(), tensor_data_f32(&[1.0, 2.0, 3.0], vec![3]));
    let plan = plan_with("Identity", vec!["a"], vec!["y"], HashMap::new());
    assert_eq!(
        modelc::onnx_exec::execute_plan(&plan, &weights, &[0.0]).unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn execute_plan_cast_passes_through() {
    let mut weights = HashMap::new();
    weights.insert("a".to_string(), tensor_data_f32(&[4.0, 5.0, 6.0], vec![3]));
    let plan = plan_with("Cast", vec!["a"], vec!["y"], HashMap::new());
    assert_eq!(
        modelc::onnx_exec::execute_plan(&plan, &weights, &[0.0]).unwrap(),
        vec![4.0, 5.0, 6.0]
    );
}

#[test]
fn execute_plan_constant_is_skipped() {
    let mut weights = HashMap::new();
    weights.insert("a".to_string(), tensor_data_f32(&[1.0, 2.0, 3.0], vec![3]));
    let plan = ExecutionPlan {
        ops: vec![
            Op {
                op_type: "Constant".to_string(),
                inputs: vec![],
                outputs: vec!["unused".to_string()],
                attrs: HashMap::new(),
            },
            Op {
                op_type: "Identity".to_string(),
                inputs: vec!["a".to_string()],
                outputs: vec!["y".to_string()],
                attrs: HashMap::new(),
            },
        ],
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
    };
    assert_eq!(
        modelc::onnx_exec::execute_plan(&plan, &weights, &[0.0]).unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn execute_plan_unsupported_op_is_skipped() {
    let mut weights = HashMap::new();
    weights.insert("a".to_string(), tensor_data_f32(&[1.0, 2.0, 3.0], vec![3]));
    let plan = ExecutionPlan {
        ops: vec![
            Op {
                op_type: "FakeOp".to_string(),
                inputs: vec![],
                outputs: vec!["unused".to_string()],
                attrs: HashMap::new(),
            },
            Op {
                op_type: "Identity".to_string(),
                inputs: vec!["a".to_string()],
                outputs: vec!["y".to_string()],
                attrs: HashMap::new(),
            },
        ],
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
    };
    assert_eq!(
        modelc::onnx_exec::execute_plan(&plan, &weights, &[0.0]).unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn execute_plan_missing_input_for_matmul_errors() {
    let weights: HashMap<String, TensorData> = HashMap::new();
    let plan = plan_with(
        "MatMul",
        vec!["missing1", "missing2"],
        vec!["y"],
        HashMap::new(),
    );
    let result = modelc::onnx_exec::execute_plan(&plan, &weights, &[1.0, 2.0]);
    assert!(result.is_err());
}

#[test]
fn execute_plan_f16_weight_roundtrip() {
    let mut weights = HashMap::new();
    // F16: 1.0 = 0x3C00, 2.0 = 0x4000
    weights.insert(
        "w".to_string(),
        TensorData {
            shape: vec![2, 2],
            dtype: DataType::F16,
            data: vec![0x00, 0x3C, 0x00, 0x40, 0x00, 0x40, 0x00, 0x3C],
        },
    );
    let plan = plan_with("MatMul", vec!["x", "w"], vec!["y"], HashMap::new());
    // input [1,2] matmul [[1,2],[2,1]] = [5,4]
    let out = modelc::onnx_exec::execute_plan(&plan, &weights, &[1.0, 2.0]).unwrap();
    assert_eq!(out, vec![5.0, 4.0]);
}

#[test]
fn execute_plan_i8_weight_roundtrip() {
    let mut weights = HashMap::new();
    weights.insert(
        "w".to_string(),
        TensorData {
            shape: vec![2, 2],
            dtype: DataType::I8,
            data: vec![1, 2, 3, 4],
        },
    );
    let plan = plan_with("MatMul", vec!["x", "w"], vec!["y"], HashMap::new());
    let out = modelc::onnx_exec::execute_plan(&plan, &weights, &[1.0, 1.0]).unwrap();
    assert_eq!(out, vec![4.0, 6.0]);
}

#[test]
fn execute_plan_short_input_is_padded() {
    let mut weights = HashMap::new();
    weights.insert(
        "w".to_string(),
        tensor_data_f32(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]),
    );
    let plan = plan_with("MatMul", vec!["x", "w"], vec!["y"], HashMap::new());
    // Input is 1 element but the target shape (last dim of first weight) is 2.
    let out = modelc::onnx_exec::execute_plan(&plan, &weights, &[5.0]).unwrap();
    assert_eq!(out, vec![5.0, 0.0]);
}

#[test]
fn execute_plan_serializes_via_execution_plan_methods() {
    let plan = ExecutionPlan {
        ops: vec![Op {
            op_type: "Relu".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: HashMap::new(),
        }],
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
    };
    let json = plan.to_json().unwrap();
    let round_tripped = ExecutionPlan::from_json(&json).unwrap();
    assert_eq!(round_tripped.ops.len(), 1);
    assert_eq!(round_tripped.ops[0].op_type, "Relu");
}

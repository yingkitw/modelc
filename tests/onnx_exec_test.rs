use std::collections::HashMap;

use anyhow::Result;

use modelc::model::{DataType, TensorData};
use modelc::onnx_exec::{AttributeValue, ExecutionPlan, Op, execute_plan};

fn make_f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> TensorData {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    TensorData {
        shape,
        dtype: DataType::F32,
        data: bytes,
    }
}

#[test]
fn execute_plan_matmul_add() -> Result<()> {
    let plan = ExecutionPlan {
        inputs: vec!["input".to_string()],
        outputs: vec!["out".to_string()],
        ops: vec![
            Op {
                op_type: "MatMul".to_string(),
                inputs: vec!["input".to_string(), "W".to_string()],
                outputs: vec!["mm".to_string()],
                attrs: HashMap::new(),
            },
            Op {
                op_type: "Add".to_string(),
                inputs: vec!["mm".to_string(), "B".to_string()],
                outputs: vec!["out".to_string()],
                attrs: HashMap::new(),
            },
        ],
    };

    let mut weights = HashMap::new();
    // W: 2x2 identity
    weights.insert(
        "W".to_string(),
        make_f32_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
    );
    // B: bias [1.0, 2.0]
    weights.insert("B".to_string(), make_f32_tensor(vec![1.0, 2.0], vec![2]));

    let result = execute_plan(&plan, &weights, &[3.0, 4.0])?;
    // input [1, 2] * [2, 2] identity = [3,4] + [1,2] = [4,6]
    assert_eq!(result.len(), 2);
    assert!((result[0] - 4.0).abs() < 1e-5, "got {}", result[0]);
    assert!((result[1] - 6.0).abs() < 1e-5, "got {}", result[1]);
    Ok(())
}

#[test]
fn execute_plan_gemm_alpha_beta() -> Result<()> {
    let mut attrs = HashMap::new();
    attrs.insert("alpha".to_string(), AttributeValue::Float(2.0));
    attrs.insert("beta".to_string(), AttributeValue::Float(0.5));

    let plan = ExecutionPlan {
        inputs: vec!["input".to_string()],
        outputs: vec!["out".to_string()],
        ops: vec![Op {
            op_type: "Gemm".to_string(),
            inputs: vec!["input".to_string(), "W".to_string(), "C".to_string()],
            outputs: vec!["out".to_string()],
            attrs,
        }],
    };

    let mut weights = HashMap::new();
    // W: 2x2 identity
    weights.insert(
        "W".to_string(),
        make_f32_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
    );
    // C: bias [2.0, 2.0]
    weights.insert("C".to_string(), make_f32_tensor(vec![2.0, 2.0], vec![2]));

    let result = execute_plan(&plan, &weights, &[1.0, 1.0])?;
    // alpha*input*W + beta*C = 2*[1,1]*I + 0.5*[2,2] = [2,2] + [1,1] = [3,3]
    assert_eq!(result.len(), 2);
    assert!((result[0] - 3.0).abs() < 1e-5, "got {}", result[0]);
    assert!((result[1] - 3.0).abs() < 1e-5, "got {}", result[1]);
    Ok(())
}

#[test]
fn execute_plan_relu() -> Result<()> {
    let plan = ExecutionPlan {
        inputs: vec!["input".to_string()],
        outputs: vec!["out".to_string()],
        ops: vec![Op {
            op_type: "Relu".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out".to_string()],
            attrs: HashMap::new(),
        }],
    };

    let weights = HashMap::new();
    let result = execute_plan(&plan, &weights, &[-1.0, 0.0, 2.0])?;
    assert_eq!(result.len(), 3);
    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 0.0).abs() < 1e-5);
    assert!((result[2] - 2.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn execute_plan_sigmoid() -> Result<()> {
    let plan = ExecutionPlan {
        inputs: vec!["input".to_string()],
        outputs: vec!["out".to_string()],
        ops: vec![Op {
            op_type: "Sigmoid".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out".to_string()],
            attrs: HashMap::new(),
        }],
    };

    let weights = HashMap::new();
    let result = execute_plan(&plan, &weights, &[0.0])?;
    assert_eq!(result.len(), 1);
    assert!(
        (result[0] - 0.5).abs() < 1e-5,
        "sigmoid(0) should be 0.5, got {}",
        result[0]
    );
    Ok(())
}

#[test]
fn execute_plan_softmax() -> Result<()> {
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(1));

    let plan = ExecutionPlan {
        inputs: vec!["input".to_string()],
        outputs: vec!["out".to_string()],
        ops: vec![Op {
            op_type: "Softmax".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out".to_string()],
            attrs,
        }],
    };

    let weights = HashMap::new();
    let result = execute_plan(&plan, &weights, &[0.0, 0.0])?;
    assert_eq!(result.len(), 2);
    assert!((result[0] - 0.5).abs() < 1e-5);
    assert!((result[1] - 0.5).abs() < 1e-5);
    Ok(())
}

#[test]
fn execute_plan_identity() -> Result<()> {
    let plan = ExecutionPlan {
        inputs: vec!["input".to_string()],
        outputs: vec!["out".to_string()],
        ops: vec![Op {
            op_type: "Identity".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out".to_string()],
            attrs: HashMap::new(),
        }],
    };

    let weights = HashMap::new();
    let result = execute_plan(&plan, &weights, &[1.0, 2.0, 3.0])?;
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
fn execute_plan_mul_div() -> Result<()> {
    let plan = ExecutionPlan {
        inputs: vec!["input".to_string()],
        outputs: vec!["out".to_string()],
        ops: vec![
            Op {
                op_type: "Mul".to_string(),
                inputs: vec!["input".to_string(), "scale".to_string()],
                outputs: vec!["mul".to_string()],
                attrs: HashMap::new(),
            },
            Op {
                op_type: "Div".to_string(),
                inputs: vec!["mul".to_string(), "scale".to_string()],
                outputs: vec!["out".to_string()],
                attrs: HashMap::new(),
            },
        ],
    };

    let mut weights = HashMap::new();
    weights.insert("scale".to_string(), make_f32_tensor(vec![2.0], vec![1]));

    let result = execute_plan(&plan, &weights, &[3.0])?;
    // mul = 3 * 2 = 6, div = 6 / 2 = 3
    assert_eq!(result.len(), 1);
    assert!((result[0] - 3.0).abs() < 1e-5, "got {}", result[0]);
    Ok(())
}

#[test]
fn execute_plan_sub() -> Result<()> {
    let plan = ExecutionPlan {
        inputs: vec!["input".to_string()],
        outputs: vec!["out".to_string()],
        ops: vec![Op {
            op_type: "Sub".to_string(),
            inputs: vec!["input".to_string(), "bias".to_string()],
            outputs: vec!["out".to_string()],
            attrs: HashMap::new(),
        }],
    };

    let mut weights = HashMap::new();
    weights.insert(
        "bias".to_string(),
        make_f32_tensor(vec![1.0, 2.0], vec![1, 2]),
    );

    let result = execute_plan(&plan, &weights, &[5.0, 5.0])?;
    assert_eq!(result, vec![4.0, 3.0]);
    Ok(())
}

#[test]
fn execution_plan_serde_roundtrip() -> Result<()> {
    let plan = ExecutionPlan {
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        ops: vec![Op {
            op_type: "Relu".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: HashMap::new(),
        }],
    };

    let json = plan.to_json()?;
    let restored = ExecutionPlan::from_json(&json)?;
    assert_eq!(restored.inputs, vec!["x"]);
    assert_eq!(restored.outputs, vec!["y"]);
    assert_eq!(restored.ops.len(), 1);
    assert_eq!(restored.ops[0].op_type, "Relu");
    Ok(())
}

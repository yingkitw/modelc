//! End-to-end: a Python program → weights → runnable model.
//!
//! `compiler::pack` on a `.py` file executes it via the python bridge
//! (`src/pybridge.rs`), packs the extracted tensors into a `.modelc` artifact,
//! and the artifact runs a real MLP forward pass through the runtime — the same
//! path `modelc run` serves. Skips gracefully when no Python is on PATH.

fn python_available() -> bool {
    for candidate in ["python3", "python"] {
        if std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

const PROGRAM: &str = r#"
# y = W1 @ relu(W0 @ x + b0) + b1, hand-written weights
layer0_weight = [[1.0, 0.5], [-0.5, 2.0]]
layer0_bias = [0.1, -0.1]
layer1_weight = [[2.0, -1.0]]
layer1_bias = [0.5]
"#;

/// Mirrors `serve::infer::gemv_bias` (row-major `[out, in]` weight): the exact
/// math `modelc run` applies to `mlp` artifacts.
fn gemv(weight: &modelc::runtime::tensor::Tensor, bias: &modelc::runtime::tensor::Tensor, x: &[f32]) -> Vec<f32> {
    let dot = modelc::runtime::gemv::gemv_dot;
    let (rows, cols) = (weight.shape[0], weight.shape[1]);
    (0..rows)
        .map(|r| bias.data[r] + dot(&weight.data[r * cols..(r + 1) * cols], x))
        .collect()
}

#[test]
fn pack_python_program_produces_runnable_mlp_artifact() {
    if !python_available() {
        eprintln!("skipping: no python3/python on PATH");
        return;
    }
    let dir = modelc::tempdir::tempdir().expect("tempdir");
    let program = dir.path().join("tiny_net.py");
    std::fs::write(&program, PROGRAM).expect("write program");
    let artifact = dir.path().join("tiny_net.modelc");

    let out = modelc::compiler::pack(&program, Some(&artifact), None, None, false, None, None)
        .expect("pack python program");
    assert_eq!(out, artifact);

    let model = modelc::pack::unpack(&artifact).expect("unpack artifact");
    assert_eq!(model.name, "tiny_net", "artifact inherits the program stem");
    assert_eq!(model.architecture, "mlp", "layer{{N}} naming infers mlp");
    assert_eq!(model.tensors.len(), 4);
    for name in ["layer0.weight", "layer0.bias", "layer1.weight", "layer1.bias"] {
        assert!(model.tensors.contains_key(name), "missing {name}");
    }
    assert_eq!(
        model.metadata.get("format").map(String::as_str),
        Some("modelc-python")
    );
    assert_eq!(
        model.metadata.get("source").map(String::as_str),
        Some("tiny_net.py")
    );

    // Run it as a model: y = W1 @ relu(W0 @ x + b0) + b1.
    let runtime = modelc::runtime::serve::Runtime::from_raw(&model.tensors);
    let w0 = runtime.get("layer0.weight").expect("layer0.weight");
    let b0 = runtime.get("layer0.bias").expect("layer0.bias");
    let w1 = runtime.get("layer1.weight").expect("layer1.weight");
    let b1 = runtime.get("layer1.bias").expect("layer1.bias");

    let h = gemv(w0, b0, &[1.0, 2.0]); // [2.1, 3.4], positive so relu is identity
    assert!((h[0] - 2.1).abs() < 1e-4 && (h[1] - 3.4).abs() < 1e-4);
    let h: Vec<f32> = h.into_iter().map(|v| v.max(0.0)).collect();
    let y = gemv(w1, b1, &h);
    assert!(
        (y[0] - 1.3).abs() < 1e-4,
        "expected 1.3 from the compiled python weights, got {}",
        y[0]
    );
}

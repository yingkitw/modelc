//! Compile a Python program into weights and run it as a model.
//!
//! The library-level mirror of the CLI flow `modelc pack model.py` → `modelc run`:
//! 1. writes a tiny Python program that defines an MLP as plain lists,
//! 2. executes it via the python bridge (`src/pybridge.rs`) into a Safetensors file,
//! 3. packs the weights into a `.modelc` artifact,
//! 4. runs an MLP forward pass from the unpacked artifact.
//!
//! Run with: `cargo run --example python_model` (requires Python on PATH).

const PROGRAM: &str = r#"
# A tiny 2-layer MLP, hand-written as plain Python lists.
layer0_weight = [[1.0, 0.5], [-0.5, 2.0]]
layer0_bias = [0.1, -0.1]
layer1_weight = [[2.0, -1.0]]
layer1_bias = [0.5]
"#;

fn gemv(weight: &modelc::runtime::tensor::Tensor, bias: &modelc::runtime::tensor::Tensor, x: &[f32]) -> Vec<f32> {
    let dot = modelc::runtime::gemv::gemv_dot;
    let (rows, cols) = (weight.shape[0], weight.shape[1]);
    (0..rows)
        .map(|r| bias.data[r] + dot(&weight.data[r * cols..(r + 1) * cols], x))
        .collect()
}

fn main() {
    let dir = modelc::tempdir::tempdir().expect("temp dir");
    let program = dir.path().join("tiny_net.py");
    std::fs::write(&program, PROGRAM).expect("write program");
    let artifact = dir.path().join("tiny_net.modelc");

    println!("1. executing {} via the python bridge...", program.display());
    modelc::compiler::pack(&program, Some(&artifact), None, None, false, None, None)
        .expect("compiling the python program (is Python on PATH?)");

    let model = modelc::pack::unpack(&artifact).expect("unpack artifact");
    println!(
        "2. packed {} — architecture: {}, tensors: {:?}",
        artifact.display(),
        model.architecture,
        {
            let mut names: Vec<&str> = model.tensors.keys().map(String::as_str).collect();
            names.sort();
            names
        }
    );

    println!("3. running the compiled weights as a model...");
    let runtime = modelc::runtime::serve::Runtime::from_raw(&model.tensors);
    let get = |name: &str| runtime.get(name).expect(name);
    let w0 = get("layer0.weight");
    let b0 = get("layer0.bias");
    let w1 = get("layer1.weight");
    let b1 = get("layer1.bias");

    let x = [1.0f32, 2.0];
    let h = gemv(w0, b0, &x).into_iter().map(|v| v.max(0.0)).collect::<Vec<_>>();
    let y = gemv(w1, b1, &h);
    println!("4. output for input {x:?}: {:?}", y);

    assert!((y[0] - 1.3).abs() < 1e-4, "expected ~1.3, got {}", y[0]);
    println!("done — the python program now runs as a model.");
}

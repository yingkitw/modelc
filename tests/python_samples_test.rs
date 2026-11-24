//! Validate the python-bridge capability against the sample programs in
//! `examples/python/` — the same files a user can read, run, and modify.
//!
//! Valid samples are packed end-to-end (`compiler::pack` on the `.py`), their
//! artifacts inspected (architecture, tensor names, shapes, dtypes), and their
//! forward pass executed against golden values computed independently here.
//! Invalid samples must fail with the documented error. Self-skips when no
//! Python interpreter is on PATH.

mod common;

use modelc::model::{DataType, Model};

const SAMPLES: &str = "examples/python";

fn sample(name: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(SAMPLES)
        .join(name);
    assert!(path.exists(), "sample {} is missing from {SAMPLES}/", path.display());
    path
}

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

/// Mirror of `serve::infer::gemv_bias` (row-major `[out, in]` weights) — the
/// exact math `modelc run` applies to `mlp` artifacts, with ReLU between layers.
/// Accepts both `layer{N}` stacks and singleton `weight`/`bias` naming.
fn mlp_forward(model: &Model, x: &[f32]) -> Vec<f32> {
    let runtime = modelc::runtime::serve::Runtime::from_raw(&model.tensors);
    let dot = modelc::runtime::gemv::gemv_dot;

    let mut ids: Vec<u32> = model
        .tensors
        .keys()
        .filter_map(|k| k.strip_prefix("layer")?.split_once('.')?.0.parse().ok())
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let pairs: Vec<(String, String)> = if ids.is_empty() {
        vec![("weight".to_string(), "bias".to_string())]
    } else {
        ids.iter()
            .map(|id| (format!("layer{id}.weight"), format!("layer{id}.bias")))
            .collect()
    };
    assert!(!pairs.is_empty(), "fixture has no runnable affine stack");

    let mut cur = x.to_vec();
    let last = pairs.len() - 1;
    for (idx, (w_name, b_name)) in pairs.iter().enumerate() {
        let w = runtime.get(w_name).expect(w_name);
        let b = runtime.get(b_name).expect(b_name);
        let (rows, cols) = (w.shape[0], w.shape[1]);
        cur = (0..rows)
            .map(|r| b.data[r] + dot(&w.data[r * cols..(r + 1) * cols], &cur))
            .collect();
        if idx != last {
            cur = cur.into_iter().map(|v| v.max(0.0)).collect();
        }
    }
    cur
}

#[test]
fn python_samples_compile_and_run_with_golden_values() {
    if !python_available() {
        eprintln!("skipping: no python3/python on PATH");
        return;
    }
    type TensorSpec = (&'static str, &'static [usize], DataType);
    type SampleCase = (
        &'static str,
        &'static str,
        &'static [TensorSpec],
        &'static [f32],
        Option<&'static [f32]>,
    );
    let cases: &[SampleCase] = &[
        (
            // (file, architecture, (tensor, shape, dtype)[], input, expected output)
            "mlp_lists.py",
            "mlp",
            &[
                ("layer0.weight", &[2usize, 2], DataType::F32),
                ("layer0.bias", &[2], DataType::F32),
                ("layer1.weight", &[1, 2], DataType::F32),
                ("layer1.bias", &[1], DataType::F32),
            ],
            &[1.0, 2.0],
            Some(&[1.3]),
        ),
        (
            "numpy_mlp.py",
            "mlp",
            &[
                ("layer0.weight", &[2, 2], DataType::F32),
                ("layer0.bias", &[2], DataType::F32),
                ("layer1.weight", &[1, 2], DataType::F32),
                ("layer1.bias", &[1], DataType::F32),
            ],
            &[1.0, 2.0],
            Some(&[1.3]),
        ),
        (
            "torch_state_dict.py",
            "mlp",
            &[
                ("layer0.weight", &[1, 2], DataType::F32),
                ("layer0.bias", &[1], DataType::F32),
                ("layer1.weight", &[2, 1], DataType::F32),
                ("layer1.bias", &[2], DataType::F32),
            ],
            &[1.0, 1.0],
            Some(&[11.5, 16.0]),
        ),
        (
            "main_block.py",
            "mlp",
            &[
                ("layer0.weight", &[2, 2], DataType::F32),
                ("layer0.bias", &[2], DataType::F32),
            ],
            &[1.0, 1.0],
            Some(&[3.0, 3.0]),
        ),
        (
            "mixed_globals.py",
            "mlp",
            &[("weight", &[1, 2], DataType::F32), ("bias", &[1], DataType::F32)],
            &[1.0, 1.0],
            Some(&[3.5]),
        ),
        (
            "int_matrix.py",
            "mlp",
            &[
                ("layer0.weight", &[2, 2], DataType::I32),
                ("layer0.bias", &[2], DataType::I32),
            ],
            &[],
            None, // I32 tensors are not MLP-plan eligible — dtype preservation is the case
        ),
    ];

    for (file, architecture, tensors, input, expected) in cases {
        let program = sample(file);
        let dir = modelc::tempdir::tempdir().expect("tempdir");
        let artifact = dir.path().join("out.modelc");
        modelc::compiler::pack(&program, Some(&artifact), None, None, false, None, None)
            .unwrap_or_else(|e| panic!("{file}: pack failed: {e:#}"));

        let model = modelc::pack::unpack(&artifact)
            .unwrap_or_else(|e| panic!("{file}: unpack failed: {e:#}"));
        assert_eq!(model.architecture, *architecture, "{file}: architecture");
        assert_eq!(
            model.tensors.len(),
            tensors.len(),
            "{file}: tensor count (scalars must not leak in)"
        );
        for (name, shape, dtype) in *tensors {
            let td = model
                .tensors
                .get(*name)
                .unwrap_or_else(|| panic!("{file}: missing tensor {name}"));
            assert_eq!(td.shape, shape.to_vec(), "{file}: {name} shape");
            assert_eq!(td.dtype, *dtype, "{file}: {name} dtype");
        }

        if let Some(expected) = expected {
            let output = mlp_forward(&model, input);
            assert_eq!(
                output.len(),
                expected.len(),
                "{file}: output length"
            );
            for (got, want) in output.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < 1e-4,
                    "{file}: expected {expected:?}, got {output:?}"
                );
            }
        }
    }
}

#[test]
fn invalid_python_samples_fail_with_documented_errors() {
    if !python_available() {
        eprintln!("skipping: no python3/python on PATH");
        return;
    }
    let cases: &[(&str, &str)] = &[
        ("invalid_ragged.py", "ragged nesting"),
        ("invalid_no_tensors.py", "no tensors found"),
    ];
    for (file, expected_fragment) in cases {
        let program = sample(file);
        let dir = modelc::tempdir::tempdir().expect("tempdir");
        let artifact = dir.path().join("out.modelc");
        let err = modelc::compiler::pack(&program, Some(&artifact), None, None, false, None, None)
            .expect_err(&format!("{file}: must fail"));
        let msg = format!("{err:#}");
        assert!(
            msg.contains(expected_fragment),
            "{file}: expected error containing {expected_fragment:?}, got: {msg}"
        );
        assert!(!artifact.exists(), "{file}: no artifact may be written");
    }
}

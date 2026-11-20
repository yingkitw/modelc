//! Recursive self-improvement round-trip: a reference model defined in code
//! produces target outputs; candidate weights travel through the `.modelc`
//! artifact pipeline (`pack` → `verify` → `unpack`) every iteration, influence
//! inference through the real runtime (`ops::linear`), and are then adjusted
//! toward the code-defined spec via a gradient step. The loop asserts that the
//! artifact round-trip is lossless on every cycle while the loss decreases
//! monotonically until the weights align with the reference.

mod common;

use modelc::digest;
use modelc::model::{DataType, Model, TensorData};
use modelc::pack;
use modelc::runtime::ops;
use modelc::runtime::tensor::Tensor;

const IN: usize = 3;
const OUT: usize = 2;
const N: usize = 8;
const LR: f32 = 0.3;
const MAX_ITERS: usize = 80;
const TOL: f32 = 1e-4;

/// The "code" spec: the true linear model the weights must align to.
const W_TRUE: [f32; IN * OUT] = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5];
const B_TRUE: [f32; OUT] = [0.3, -0.2];

fn inputs() -> Vec<f32> {
    (0..N * IN)
        .map(|i| (i % 5) as f32 * 0.5 - 1.0)
        .collect()
}

fn targets(x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; N * OUT];
    for n in 0..N {
        for o in 0..OUT {
            y[n * OUT + o] = B_TRUE[o]
                + (0..IN).map(|j| x[n * IN + j] * W_TRUE[j * OUT + o]).sum::<f32>();
        }
    }
    y
}

fn model_from(w: &[f32], b: &[f32]) -> Model {
    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "weight".to_string(),
        TensorData {
            shape: vec![IN, OUT],
            dtype: DataType::F32,
            data: common::f32_to_bytes(w),
        },
    );
    tensors.insert(
        "bias".to_string(),
        TensorData {
            shape: vec![OUT],
            dtype: DataType::F32,
            data: common::f32_to_bytes(b),
        },
    );
    Model {
        name: "self_improve".to_string(),
        architecture: "linear".to_string(),
        tensors,
        metadata: std::collections::HashMap::new(),
    }
}

/// One full cycle: pack weights → verify → unpack → infer → gradient step.
/// Returns (loss before the update, updated weights, updated bias).
fn cycle(
    artifact: &std::path::Path,
    w: &[f32],
    b: &[f32],
    x: &[f32],
    y: &[f32],
    compress: bool,
) -> (f32, Vec<f32>, Vec<f32>) {
    pack::pack(&model_from(w, b), artifact, compress).unwrap();
    pack::verify(artifact).unwrap();
    let restored = pack::unpack(artifact).unwrap();

    // Round-trip invariant: F32 weights survive the artifact byte-for-byte.
    let rw = &restored.tensors["weight"];
    let rb = &restored.tensors["bias"];
    assert_eq!(rw.dtype, DataType::F32);
    assert_eq!(rw.shape, vec![IN, OUT]);
    assert_eq!(rw.data, common::f32_to_bytes(w));
    assert_eq!(rb.data, common::f32_to_bytes(b));

    // Influence: inference uses only what the artifact materialized.
    let pred = ops::linear(
        &Tensor::from_vec(x.to_vec(), vec![N, IN]),
        &Tensor::from_vec(common::bytes_to_f32(&rw.data), rw.shape.clone()),
        Some(&Tensor::from_vec(
            common::bytes_to_f32(&rb.data),
            rb.shape.clone(),
        )),
    );

    let scale = 2.0 / (N * OUT) as f32;
    let mut loss = 0.0f32;
    let mut dw = vec![0.0f32; IN * OUT];
    let mut db = vec![0.0f32; OUT];
    for n in 0..N {
        for o in 0..OUT {
            let e = pred.data[n * OUT + o] - y[n * OUT + o];
            loss += e * e;
            for j in 0..IN {
                dw[j * OUT + o] += scale * x[n * IN + j] * e;
            }
            db[o] += scale * e;
        }
    }
    loss /= (N * OUT) as f32;

    // Adjust toward the code spec.
    let new_w: Vec<f32> = w.iter().zip(&dw).map(|(w, g)| w - LR * g).collect();
    let new_b: Vec<f32> = b.iter().zip(&db).map(|(b, g)| b - LR * g).collect();
    (loss, new_w, new_b)
}

fn self_improve(compress: bool) {
    let dir = common::tempdir().unwrap();
    let artifact = dir.path().join("self_improve.modelc");
    let x = inputs();
    let y = targets(&x);

    let mut w = vec![0.0f32; IN * OUT]; // start misaligned
    let mut b = vec![0.0f32; OUT];
    let mut prev = f32::INFINITY;
    let mut digests = Vec::new();
    let mut final_loss = f32::INFINITY;
    let mut converged = false;

    for _ in 0..MAX_ITERS {
        let (loss, nw, nb) = cycle(&artifact, &w, &b, &x, &y, compress);
        digests.push(digest::digest_file(&artifact).unwrap());

        // Recursive self-improvement: every cycle must do better than the last.
        assert!(loss < prev, "loss must decrease each iteration: {loss} !< {prev}");
        prev = loss;
        w = nw;
        b = nb;
        if loss < 1e-10 {
            final_loss = loss;
            converged = true;
            break;
        }
        final_loss = loss;
    }

    assert!(converged || final_loss < TOL, "failed to converge: {final_loss}");
    assert!(final_loss < TOL);
    // The weights changed (digests differ) while every artifact stayed valid.
    assert_ne!(digests[0], *digests.last().unwrap());
    let wt = Tensor::from_vec(w, vec![IN, OUT]);
    for (i, (&got, &want)) in wt.data.iter().zip(W_TRUE.iter()).enumerate() {
        assert!(
            (got - want).abs() < 0.05,
            "weight[{i}] = {got}, expected ~{want}"
        );
    }
    for (i, (&got, &want)) in b.iter().zip(B_TRUE.iter()).enumerate() {
        assert!((got - want).abs() < 0.05, "bias[{i}] = {got}, expected ~{want}");
    }
}

#[test]
fn self_improving_weights_converge_through_artifact_round_trip() {
    self_improve(false);
}

#[test]
fn self_improving_weights_converge_through_compressed_artifact() {
    self_improve(true);
}

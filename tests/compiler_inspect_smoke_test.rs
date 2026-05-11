//! Smoke tests for `compiler::inspect` on real temp files (library API + format detection).

mod common;

use tempfile::tempdir;

use modelc::compiler;

#[test]
fn inspect_safetensors_via_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let data = common::f32_to_bytes(&[0.25, 0.5, 0.75, 1.0]);
    common::create_safetensors_file(&path, vec![("t", "F32", vec![2, 2], data)]);

    compiler::inspect(&path, None).expect("inspect should print summary without error");
}

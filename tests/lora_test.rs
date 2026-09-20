use std::collections::HashMap;

use anyhow::Result;

mod common;

use modelc::model::{DataType, Model, TensorData};

/// Build a minimal safetensors file containing a single f32 tensor.
fn make_minimal_lora_safetensors() -> Vec<u8> {
    use safetensors::tensor::TensorView;

    // Tensor shape: [2, 2] = 4 f32 values = 16 bytes.
    let shape: Vec<usize> = vec![2, 2];
    let data: Vec<u8> = vec![1.0f32, 0.0, 0.0, 1.0]
        .into_iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let tv = TensorView::new(safetensors::Dtype::F32, shape, &data).unwrap();
    let mut metadata: HashMap<String, TensorView> = HashMap::new();
    metadata.insert("weight".to_string(), tv);

    safetensors::serialize(&metadata, &None).unwrap()
}

#[test]
fn lora_apply_changes_base_tensor() -> Result<()> {
    let mut model = Model {
        name: "base".to_string(),
        architecture: "mlp".to_string(),
        metadata: HashMap::new(),
        tensors: {
            let mut t = HashMap::new();
            t.insert(
                "weight".to_string(),
                TensorData {
                    shape: vec![2, 2],
                    dtype: DataType::F32,
                    data: vec![0.0f32, 0.0, 0.0, 0.0]
                        .into_iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect(),
                },
            );
            t
        },
    };

    // Build a minimal safetensors file with a single tensor.
    let lora_buf = make_minimal_lora_safetensors();
    let temp_dir = common::tempdir()?;
    let lora_path = temp_dir.path().join("adapter.safetensors");
    std::fs::write(&lora_path, &lora_buf)?;

    modelc::lora::apply_lora(&mut model, &lora_path, 1.0)?;

    let updated = &model.tensors["weight"];
    let values: Vec<f32> = updated
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // Base was all zeros, LoRA adds scaled values.
    // The first element should be non-zero after LoRA application.
    assert!(
        values[0].abs() > 0.001,
        "expected LoRA to modify weight, got {:?}",
        values
    );
    Ok(())
}

#[test]
fn lora_apply_skips_non_f32() -> Result<()> {
    let mut model = Model {
        name: "base".to_string(),
        architecture: "mlp".to_string(),
        metadata: HashMap::new(),
        tensors: {
            let mut t = HashMap::new();
            t.insert(
                "weight".to_string(),
                TensorData {
                    shape: vec![2, 2],
                    dtype: DataType::I8,
                    data: vec![0, 0, 0, 0],
                },
            );
            t
        },
    };

    let lora_buf = make_minimal_lora_safetensors();
    let temp_dir = common::tempdir()?;
    let lora_path = temp_dir.path().join("adapter.safetensors");
    std::fs::write(&lora_path, &lora_buf)?;

    modelc::lora::apply_lora(&mut model, &lora_path, 1.0)?;

    // I8 tensor should remain unchanged.
    assert_eq!(model.tensors["weight"].dtype, DataType::I8);
    Ok(())
}

/// Build a safetensors buffer with several F16 tensors under arbitrary names.
fn make_f16_lora_safetensors(names: &[&str]) -> Vec<u8> {
    use safetensors::tensor::TensorView;

    let mut metadata: HashMap<String, TensorView> = HashMap::new();
    // TensorViews borrow their data, so the buffers must outlive the views.
    // Pre-allocate the buffer slot, then borrow it immutably for the view.
    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(names.len());
    for _name in names {
        // [2, 2] F16 tensor: values 0.5, 0.25, 0.125, 0.0625.
        let data: Vec<u8> = [0.5f32, 0.25, 0.125, 0.0625]
            .into_iter()
            .flat_map(|f| half::f16::from_f32(f).to_bits().to_le_bytes())
            .collect();
        buffers.push(data);
    }
    for (i, name) in names.iter().enumerate() {
        let tv = TensorView::new(safetensors::Dtype::F16, vec![2, 2], &buffers[i]).unwrap();
        metadata.insert(name.to_string(), tv);
    }
    safetensors::serialize(&metadata, &None).unwrap()
}

#[test]
fn lora_apply_skips_tensors_without_matching_base() -> Result<()> {
    // Adapter tensor names that don't match any base tensor are skipped
    // without error (exercises the `no matching base tensor` branch).
    let mut model = Model {
        name: "base".to_string(),
        architecture: "mlp".to_string(),
        metadata: HashMap::new(),
        tensors: HashMap::new(), // no tensors at all
    };

    let lora_buf = make_f16_lora_safetensors(&["unknown.lora_A", "unknown.lora_B"]);
    let temp_dir = common::tempdir()?;
    let lora_path = temp_dir.path().join("adapter.safetensors");
    std::fs::write(&lora_path, &lora_buf)?;

    // Should succeed (skip) rather than error.
    modelc::lora::apply_lora(&mut model, &lora_path, 1.0)?;
    Ok(())
}

#[test]
fn lora_apply_accepts_f16_adapter_tensors() -> Result<()> {
    // Adapter carrying F16 tensors whose stripped names match the base
    // (exercises the F16 branch of view_to_f32 + the apply path).
    let mut model = Model {
        name: "base".to_string(),
        architecture: "mlp".to_string(),
        metadata: HashMap::new(),
        tensors: {
            let mut t = HashMap::new();
            t.insert(
                "weight".to_string(),
                TensorData {
                    shape: vec![2, 2],
                    dtype: DataType::F32,
                    data: vec![0.0f32, 0.0, 0.0, 0.0]
                        .into_iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect(),
                },
            );
            t
        },
    };

    let lora_buf = make_f16_lora_safetensors(&["weight"]);
    let temp_dir = common::tempdir()?;
    let lora_path = temp_dir.path().join("adapter.safetensors");
    std::fs::write(&lora_path, &lora_buf)?;

    modelc::lora::apply_lora(&mut model, &lora_path, 1.0)?;

    let values: Vec<f32> = model.tensors["weight"]
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(
        values.iter().any(|v| v.abs() > 1e-3),
        "F16 adapter values should be applied, got {values:?}"
    );
    Ok(())
}

#[test]
fn list_lora_adapters_finds_only_safetensors_files() -> Result<()> {
    let dir = common::tempdir()?;
    std::fs::write(dir.path().join("a.safetensors"), b"x")?;
    std::fs::write(dir.path().join("b.safetensors"), b"x")?;
    std::fs::write(dir.path().join("ignored.bin"), b"x")?;
    std::fs::write(dir.path().join("ignored.txt"), b"x")?;

    let adapters = modelc::lora::list_lora_adapters(dir.path())?;
    assert_eq!(adapters.len(), 2, "only .safetensors files: {adapters:?}");
    assert!(adapters[0].ends_with("a.safetensors"));
    assert!(adapters[1].ends_with("b.safetensors"));
    Ok(())
}

#[test]
fn list_lora_adapters_missing_dir_returns_empty() -> Result<()> {
    let adapters = modelc::lora::list_lora_adapters(std::path::Path::new("/nonexistent/lora/dir"))?;
    assert!(adapters.is_empty());
    Ok(())
}

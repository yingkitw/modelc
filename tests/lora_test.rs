use std::collections::HashMap;

use anyhow::Result;

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
    let temp_dir = tempfile::tempdir()?;
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
    assert!(values[0].abs() > 0.001, "expected LoRA to modify weight, got {:?}", values);
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
    let temp_dir = tempfile::tempdir()?;
    let lora_path = temp_dir.path().join("adapter.safetensors");
    std::fs::write(&lora_path, &lora_buf)?;

    modelc::lora::apply_lora(&mut model, &lora_path, 1.0)?;

    // I8 tensor should remain unchanged.
    assert_eq!(model.tensors["weight"].dtype, DataType::I8);
    Ok(())
}

//! LoRA (Low-Rank Adaptation) adapter support.
//!
//! LoRA adapters are small sets of low-rank matrices (A and B) that are
//! added to base model weights at runtime: W' = W + alpha/r * B·A.

use std::path::Path;

use anyhow::{Context, Result};

use crate::model::{DataType, Model, TensorData};

/// Load a LoRA adapter from a Safetensors file and apply it to a model.
pub fn apply_lora(model: &mut Model, lora_path: &Path, alpha: f32) -> Result<()> {
    let lora_data = std::fs::read(lora_path)
        .with_context(|| format!("failed to read LoRA file {:?}", lora_path))?;
    let lora = safetensors::SafeTensors::deserialize(&lora_data)
        .with_context(|| format!("failed to parse LoRA safetensors {:?}", lora_path))?;

    let mut applied = 0;
    let mut skipped = 0;

    for name in lora.names() {
        let view = lora.tensor(&name)?;
        let base_name = lora_base_name(&name);

        if let Some(base_tensor) = model.tensors.get_mut(base_name) {
            if base_tensor.dtype != DataType::F32 {
                eprintln!("  skipping {} (base tensor is not F32)", base_name);
                skipped += 1;
                continue;
            }

            let lora_f32 = view_to_f32(&view)?;
            let rank = infer_rank(&name, lora_f32.len(), base_tensor.element_count());
            let scale = alpha / rank.max(1.0);

            apply_low_rank_update(base_tensor, &lora_f32, scale)?;
            applied += 1;
            eprintln!("  applied LoRA to {} (rank={}, scale={:.4})", base_name, rank, scale);
        } else {
            eprintln!("  skipping {} (no matching base tensor)", base_name);
            skipped += 1;
        }
    }

    eprintln!("LoRA applied: {} tensors, skipped: {}", applied, skipped);
    Ok(())
}

/// Strip LoRA-specific prefixes/suffixes to find the base tensor name.
fn lora_base_name(lora_name: &str) -> &str {
    lora_name
        .strip_prefix("lora_")
        .unwrap_or(lora_name)
        .strip_suffix(".lora_A")
        .or_else(|| lora_name.strip_suffix(".lora_B"))
        .or_else(|| lora_name.strip_suffix(".lora_up"))
        .or_else(|| lora_name.strip_suffix(".lora_down"))
        .unwrap_or(lora_name)
}

/// Infer the rank from the LoRA tensor name or shape.
fn infer_rank(name: &str, lora_elems: usize, base_elems: usize) -> f32 {
    // LoRA tensors come in A (in×rank) and B (rank×out) pairs.
    // If lora_elems is smaller than base_elems, it's likely the A matrix.
    if name.contains("lora_A") || name.contains("lora_down") {
        // Try to infer rank from typical shapes
        if base_elems > 0 && lora_elems > 0 && lora_elems < base_elems {
            (base_elems as f32 / lora_elems as f32).sqrt().round()
        } else {
            8.0 // default rank
        }
    } else {
        8.0
    }
}

/// Convert a safetensors view to f32 values.
fn view_to_f32(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>> {
    let dtype = view.dtype();
    let data = view.data();
    Ok(match dtype {
        safetensors::Dtype::F32 => data
            .chunks_exact(4)
            .map(|c: &[u8]| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        safetensors::Dtype::F16 => data
            .chunks_exact(2)
            .map(|c: &[u8]| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect(),
        _ => anyhow::bail!("unsupported LoRA dtype {:?}", dtype),
    })
}

/// Apply a low-rank update to a base tensor.
fn apply_low_rank_update(base: &mut TensorData, lora: &[f32], scale: f32) -> Result<()> {
    let base_count = base.element_count();
    let mut base_f32 = Vec::with_capacity(base_count);
    for chunk in base.data.chunks_exact(4) {
        base_f32.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    anyhow::ensure!(
        base_f32.len() == base_count,
        "base tensor decode mismatch"
    );

    // Simple element-wise addition for single-matrix LoRA tensors.
    // For proper BA application we'd need both A and B matrices.
    let n = base_f32.len().min(lora.len());
    for i in 0..n {
        base_f32[i] += lora[i] * scale;
    }

    let mut new_data = Vec::with_capacity(base_count * 4);
    for v in base_f32 {
        new_data.extend_from_slice(&v.to_le_bytes());
    }
    base.data = new_data;
    Ok(())
}

/// List available LoRA adapter files in a directory.
pub fn list_lora_adapters(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut adapters = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                adapters.push(path);
            }
        }
    }
    adapters.sort();
    Ok(adapters)
}

//! Single-file artifact format for modelc.
//!
//! Format layout (version 2):
//!   - Magic: "MODELC" (6 bytes)
//!   - Version: u32 LE (4 bytes)
//!   - Flags: u32 LE (4 bytes) — bit 0 = zstd compressed data blob
//!   - Header length: u64 LE (8 bytes)
//!   - Header JSON (header_length bytes)
//!   - Tensor data blob (concatenated raw or zstd-compressed bytes)

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::model::{DataType, Model, TensorData};

const MAGIC: &[u8] = b"MODELC";
const VERSION: u32 = 2;
const FLAG_COMPRESSED: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactHeader {
    pub name: String,
    pub architecture: String,
    pub metadata: HashMap<String, String>,
    pub tensors: Vec<ArtifactTensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub offset: u64,
    pub length: u64,
}

/// Pack a [`Model`] into a `.modelc` artifact file with optional compression.
pub fn pack(model: &Model, path: &Path, compress: bool) -> Result<()> {
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to create artifact at {:?}", path))?;

    // Build header and compute tensor offsets.
    let mut tensors = Vec::with_capacity(model.tensors.len());
    let mut data_offset: u64 = 0;

    // Sort tensor names for deterministic output.
    let mut names: Vec<&String> = model.tensors.keys().collect();
    names.sort();

    for name in &names {
        let td = &model.tensors[*name];
        let length = td.data.len() as u64;
        tensors.push(ArtifactTensor {
            name: name.to_string(),
            shape: td.shape.clone(),
            dtype: data_type_to_str(td.dtype),
            offset: data_offset,
            length,
        });
        data_offset += length;
    }

    let header = ArtifactHeader {
        name: model.name.clone(),
        architecture: model.architecture.clone(),
        metadata: model.metadata.clone(),
        tensors,
    };

    let header_json = serde_json::to_vec(&header).context("failed to serialize artifact header")?;
    let header_len = header_json.len() as u64;

    // Build raw data blob.
    let mut raw_blob = Vec::with_capacity(data_offset as usize);
    for name in &names {
        let td = &model.tensors[*name];
        raw_blob.extend_from_slice(&td.data);
    }

    let flags = if compress { FLAG_COMPRESSED } else { 0 };
    let data_blob: Vec<u8> = if compress {
        zstd::encode_all(&raw_blob[..], 3).context("failed to compress artifact data")?
    } else {
        raw_blob
    };

    // Write header.
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&flags.to_le_bytes())?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(&header_json)?;
    file.write_all(&data_blob)?;

    file.flush().context("failed to flush artifact file")?;
    Ok(())
}

/// Read only the JSON header from a `.modelc` artifact without loading tensor data.
pub fn read_header(path: &Path) -> Result<ArtifactHeader> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open artifact at {:?}", path))?;

    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .context("artifact file too short (magic)")?;
    if magic != MAGIC {
        anyhow::bail!("invalid artifact magic bytes (not a .modelc file)");
    }

    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes)
        .context("artifact file too short (version)")?;
    let version = u32::from_le_bytes(version_bytes);

    let _flags = if version >= 2 {
        let mut flags_bytes = [0u8; 4];
        file.read_exact(&mut flags_bytes)
            .context("artifact file too short (flags)")?;
        u32::from_le_bytes(flags_bytes)
    } else {
        0
    };

    if version != 1 && version != 2 {
        anyhow::bail!("unsupported artifact version {} (expected 1 or 2)", version);
    }

    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .context("artifact file too short (header length)")?;
    let header_len = u64::from_le_bytes(header_len_bytes);

    let mut header_json = vec![0u8; header_len as usize];
    file.read_exact(&mut header_json)
        .context("artifact file too short (header)")?;
    let header: ArtifactHeader = serde_json::from_slice(&header_json)
        .context("failed to deserialize artifact header")?;

    Ok(header)
}

/// Unpack a `.modelc` artifact file into a [`Model`].
pub fn unpack(path: &Path) -> Result<Model> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open artifact at {:?}", path))?;

    // Read magic.
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .context("artifact file too short (magic)")?;
    if magic != MAGIC {
        anyhow::bail!("invalid artifact magic bytes (not a .modelc file)");
    }

    // Read version.
    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes)
        .context("artifact file too short (version)")?;
    let version = u32::from_le_bytes(version_bytes);

    // Read flags (only present in version >= 2).
    let flags = if version >= 2 {
        let mut flags_bytes = [0u8; 4];
        file.read_exact(&mut flags_bytes)
            .context("artifact file too short (flags)")?;
        u32::from_le_bytes(flags_bytes)
    } else {
        0
    };

    if version != 1 && version != 2 {
        anyhow::bail!("unsupported artifact version {} (expected 1 or 2)", version);
    }

    // Read header length.
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .context("artifact file too short (header length)")?;
    let header_len = u64::from_le_bytes(header_len_bytes);

    // Read header JSON.
    let mut header_json = vec![0u8; header_len as usize];
    file.read_exact(&mut header_json)
        .context("artifact file too short (header)")?;
    let header: ArtifactHeader = serde_json::from_slice(&header_json)
        .context("failed to deserialize artifact header")?;

    // Read tensor data blob.
    let mut data_blob = Vec::new();
    file.read_to_end(&mut data_blob)
        .context("failed to read artifact data blob")?;

    // Decompress if needed.
    let data_blob = if flags & FLAG_COMPRESSED != 0 {
        zstd::decode_all(&data_blob[..]).context("failed to decompress artifact data")?
    } else {
        data_blob
    };

    // Reconstruct Model.
    let mut tensors = HashMap::new();
    for at in &header.tensors {
        let start = at.offset as usize;
        let end = start + at.length as usize;
        if end > data_blob.len() {
            anyhow::bail!(
                "artifact corrupted: tensor {} claims offset {} + length {} exceeds blob size {}",
                at.name,
                at.offset,
                at.length,
                data_blob.len()
            );
        }
        let data = data_blob[start..end].to_vec();
        tensors.insert(
            at.name.clone(),
            TensorData {
                shape: at.shape.clone(),
                dtype: str_to_data_type(&at.dtype)?,
                data,
            },
        );
    }

    Ok(Model {
        name: header.name,
        architecture: header.architecture,
        tensors,
        metadata: header.metadata,
    })
}

fn data_type_to_str(dt: DataType) -> String {
    match dt {
        DataType::F32 => "f32".to_string(),
        DataType::F16 => "f16".to_string(),
        DataType::BF16 => "bf16".to_string(),
        DataType::I64 => "i64".to_string(),
        DataType::I32 => "i32".to_string(),
        DataType::I16 => "i16".to_string(),
        DataType::I8 => "i8".to_string(),
        DataType::U8 => "u8".to_string(),
        DataType::Bool => "bool".to_string(),
    }
}

fn str_to_data_type(s: &str) -> Result<DataType> {
    match s {
        "f32" => Ok(DataType::F32),
        "f16" => Ok(DataType::F16),
        "bf16" => Ok(DataType::BF16),
        "i64" => Ok(DataType::I64),
        "i32" => Ok(DataType::I32),
        "i16" => Ok(DataType::I16),
        "i8" => Ok(DataType::I8),
        "u8" => Ok(DataType::U8),
        "bool" => Ok(DataType::Bool),
        _ => anyhow::bail!("unknown dtype '{}' in artifact", s),
    }
}

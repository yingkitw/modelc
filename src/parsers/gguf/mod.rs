//! GGUF reader (little-endian). Loads dense tensors as-is and preserves **Q4_0 / Q5_0 / Q8_0 /
//! Q4_K / Q6_K** GGML block-quantized layouts in IR (dequantized at runtime by `Runtime::from_raw`
//! or on demand via `dequantize_gguf_tensor`). Other quant types still error with a descriptive
//! message. See [GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};

use crate::model::{DataType, Model, TensorData};
use crate::parsers::WeightParser;

mod cursor;
mod dequant;
mod quant;

pub use quant::{quantize_q4_0, quantize_q8_0};

pub struct GgufParser;

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q5_0: u32 = 6;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_I8: u32 = 24;
const GGML_TYPE_I16: u32 = 25;
const GGML_TYPE_I32: u32 = 26;
const GGML_TYPE_I64: u32 = 27;
const GGML_TYPE_F64: u32 = 28;
const GGML_TYPE_BF16: u32 = 30;

fn is_gguf_quant_type(ty: u32) -> bool {
    matches!(
        ty,
        GGML_TYPE_Q4_0 | GGML_TYPE_Q5_0 | GGML_TYPE_Q8_0 | GGML_TYPE_Q4_K | GGML_TYPE_Q6_K
    )
}

// gguf_metadata_value_type
const VAL_UINT8: u32 = 0;
const VAL_INT8: u32 = 1;
const VAL_UINT16: u32 = 2;
const VAL_INT16: u32 = 3;
const VAL_UINT32: u32 = 4;
const VAL_INT32: u32 = 5;
const VAL_FLOAT32: u32 = 6;
const VAL_BOOL: u32 = 7;
const VAL_STRING: u32 = 8;
const VAL_ARRAY: u32 = 9;
const VAL_UINT64: u32 = 10;
const VAL_INT64: u32 = 11;
const VAL_FLOAT64: u32 = 12;

const MAX_STRING_BYTES: usize = 4 * 1024 * 1024;

impl WeightParser for GgufParser {
    fn parse(&self, path: &Path) -> Result<Model> {
        let data = std::fs::read(path).with_context(|| format!("failed to read {:?}", path))?;
        parse_gguf_bytes(&data, path)
    }

    fn format_name(&self) -> &'static str {
        "gguf"
    }
}

/// Tokenizer metadata extracted from a GGUF file.
#[derive(Debug, Default)]
pub struct GgufTokenizerMetadata {
    pub model: String,
    pub vocab: Vec<String>,
    pub merges: Vec<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub chat_template: Option<String>,
}

fn parse_gguf_bytes(data: &[u8], path: &Path) -> Result<Model> {
    let mut cursor = cursor::Cursor::new(data);
    cursor.expect_bytes(b"GGUF").context("missing GGUF magic")?;

    let _version = cursor.read_u32()?;
    let tensor_count_u64 = cursor.read_u64()?;
    let kv_count_u64 = cursor.read_u64()?;
    let tensor_count = usize::try_from(tensor_count_u64)
        .map_err(|_| anyhow!("tensor_count does not fit in usize"))?;
    let kv_count =
        usize::try_from(kv_count_u64).map_err(|_| anyhow!("kv_count does not fit in usize"))?;

    let mut metadata: HashMap<String, String> = HashMap::new();

    for _ in 0..kv_count {
        let key = cursor.read_string()?;
        let value_type = cursor.read_u32()?;
        let value_str = cursor.read_kv_value_formatted(value_type)?;
        metadata.insert(key, value_str);
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(32);
    let alignment_usize = usize::try_from(alignment).map_err(|_| anyhow!("invalid alignment"))?;

    let mut infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cursor.read_string()?;
        let n_dims = cursor.read_u32()? as usize;
        if n_dims > 128 {
            anyhow::bail!("GGUF tensor has unrealistic n_dimensions {n_dims}");
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(cursor.read_u64()?);
        }
        let ggml_ty = cursor.read_u32()?;
        let ti_offset = cursor.read_u64()?;
        infos.push(TensorInfoRaw {
            name,
            dims,
            ggml_ty,
            offset: ti_offset,
        });
    }

    let mut tensor_base = cursor.pos();
    tensor_base = align_offset(tensor_base, alignment_usize);
    if tensor_base > data.len() {
        anyhow::bail!("GGUF tensor data offset past end of file");
    }

    let mut tensors = HashMap::new();
    for ti in &infos {
        let mut nelem: usize = 1;
        for d in &ti.dims {
            let du =
                usize::try_from(*d).map_err(|_| anyhow!("tensor dim does not fit in usize"))?;
            nelem = nelem
                .checked_mul(du)
                .ok_or_else(|| anyhow!("tensor {:?} dimension product overflow", ti.name))?;
        }

        let (dtype, expected) = descriptor_for_ggml(ti.ggml_ty, nelem).with_context(|| {
            format!(
                "tensor {:?}: {} (ggml type id {})",
                ti.name,
                ggml_type_hint(ti.ggml_ty),
                ti.ggml_ty
            )
        })?;
        let ti_off_usize =
            usize::try_from(ti.offset).map_err(|_| anyhow!("tensor offset overflow"))?;
        let absolute = tensor_base
            .checked_add(ti_off_usize)
            .context("tensor absolute offset overflow")?;
        let end = absolute
            .checked_add(expected)
            .context("tensor end overflow")?;
        if end > data.len() {
            anyhow::bail!(
                "tensor {:?} slice [{absolute}, {end}) exceeds file length {}",
                ti.name,
                data.len(),
            );
        }

        let mut raw_slice = data[absolute..end].to_vec();
        let shape_usize = ti
            .dims
            .iter()
            .map(|d| usize::try_from(*d).map_err(|_| anyhow!("tensor dim does not fit in usize")))
            .collect::<Result<Vec<_>>>()?;

        // Only unpack/dequantize non-quantized payloads. GGUF block-quantized types are kept
        // as raw bytes and dequantized at runtime by `Runtime::from_raw`.
        if !is_gguf_quant_type(ti.ggml_ty) {
            raw_slice = unpack_ggml_payload(ti.ggml_ty, &raw_slice, nelem).with_context(|| {
                format!("tensor {:?} (type {}): payload decode", ti.name, ti.ggml_ty)
            })?;
        }

        tensors.insert(
            ti.name.clone(),
            TensorData {
                shape: shape_usize,
                dtype,
                data: raw_slice,
            },
        );
    }

    let architecture = metadata
        .get("general.architecture")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    metadata
        .entry("format".to_string())
        .or_insert_with(|| "gguf".to_string());
    let name = path.file_stem().unwrap().to_string_lossy().to_string();

    Ok(Model {
        name,
        architecture,
        tensors,
        metadata,
    })
}

/// Read tokenizer metadata from a GGUF file without loading tensors.
pub fn extract_tokenizer_metadata(path: &Path) -> Result<GgufTokenizerMetadata> {
    let data = std::fs::read(path).with_context(|| format!("failed to read {:?}", path))?;
    let mut cursor = cursor::Cursor::new(&data);
    cursor.expect_bytes(b"GGUF").context("missing GGUF magic")?;

    let _version = cursor.read_u32()?;
    let tensor_count_u64 = cursor.read_u64()?;
    let kv_count_u64 = cursor.read_u64()?;
    let kv_count = usize::try_from(kv_count_u64).map_err(|_| anyhow!("kv_count overflow"))?;
    let _tensor_count =
        usize::try_from(tensor_count_u64).map_err(|_| anyhow!("tensor_count overflow"))?;

    let mut meta = GgufTokenizerMetadata::default();

    for _ in 0..kv_count {
        let key = cursor.read_string()?;
        let value_type = cursor.read_u32()?;

        match key.as_str() {
            "tokenizer.ggml.model" => {
                if value_type == VAL_STRING {
                    meta.model = cursor.read_string()?;
                } else {
                    let _ = cursor.read_kv_value_formatted(value_type);
                }
            }
            "tokenizer.ggml.tokens" | "tokenizer.ggml.vocab" => {
                if value_type == VAL_ARRAY {
                    meta.vocab = cursor.read_string_array()?;
                } else {
                    let _ = cursor.read_kv_value_formatted(value_type);
                }
            }
            "tokenizer.ggml.merges" => {
                if value_type == VAL_ARRAY {
                    meta.merges = cursor.read_string_array()?;
                } else {
                    let _ = cursor.read_kv_value_formatted(value_type);
                }
            }
            "tokenizer.ggml.bos_token_id" => {
                if value_type == VAL_UINT32 {
                    meta.bos_token_id = Some(cursor.read_u32()?);
                } else {
                    let _ = cursor.read_kv_value_formatted(value_type);
                }
            }
            "tokenizer.ggml.eos_token_id" => {
                if value_type == VAL_UINT32 {
                    meta.eos_token_id = Some(cursor.read_u32()?);
                } else {
                    let _ = cursor.read_kv_value_formatted(value_type);
                }
            }
            "tokenizer.chat_template" if value_type == VAL_STRING => {
                meta.chat_template = Some(cursor.read_string()?);
            }
            _ => {
                let _ = cursor.read_kv_value_formatted(value_type);
            }
        }
    }

    Ok(meta)
}

struct TensorInfoRaw {
    name: String,
    dims: Vec<u64>,
    ggml_ty: u32,
    offset: u64,
}

fn align_offset(pos: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return pos;
    }
    let r = pos % alignment;
    if r == 0 { pos } else { pos + alignment - r }
}

/// `(IR dtype, on-disk byte span for this tensor)`
fn descriptor_for_ggml(ggml_ty: u32, nelem: usize) -> Result<(DataType, usize)> {
    Ok(match ggml_ty {
        GGML_TYPE_F32 => (DataType::F32, dense_byte_len(4, nelem)?),
        GGML_TYPE_F16 | GGML_TYPE_BF16 => (dense_dtype(ggml_ty)?, dense_byte_len(2, nelem)?),
        GGML_TYPE_I8 => (DataType::I8, dense_byte_len(1, nelem)?),
        GGML_TYPE_I16 => (DataType::I16, dense_byte_len(2, nelem)?),
        GGML_TYPE_I32 => (DataType::I32, dense_byte_len(4, nelem)?),
        GGML_TYPE_I64 => (DataType::I64, dense_byte_len(8, nelem)?),
        GGML_TYPE_F64 => (DataType::F32, dense_byte_len(8, nelem)?),
        GGML_TYPE_Q4_0 => {
            ensure_multiple_of(nelem, GGML_BLOCK_ELEMENTS, ggml_ty)?;
            let bytes = quant_block_byte_len(ggml_ty, nelem)?;
            (DataType::Q4_0, bytes)
        }
        GGML_TYPE_Q5_0 => {
            ensure_multiple_of(nelem, GGML_BLOCK_ELEMENTS, ggml_ty)?;
            let bytes = quant_block_byte_len(ggml_ty, nelem)?;
            (DataType::Q5_0, bytes)
        }
        GGML_TYPE_Q8_0 => {
            ensure_multiple_of(nelem, GGML_BLOCK_ELEMENTS, ggml_ty)?;
            let bytes = quant_block_byte_len(ggml_ty, nelem)?;
            (DataType::Q8_0, bytes)
        }
        GGML_TYPE_Q4_K => {
            const QK_K: usize = 256;
            ensure_multiple_of(nelem, QK_K, ggml_ty)?;
            let bytes = nelem / QK_K * 144;
            (DataType::Q4_K, bytes)
        }
        GGML_TYPE_Q6_K => {
            const QK_K: usize = 256;
            ensure_multiple_of(nelem, QK_K, ggml_ty)?;
            let bytes = nelem / QK_K * 210;
            (DataType::Q6_K, bytes)
        }
        _ => anyhow::bail!(
            "unsupported GGML type {ty} ({hint}) — convert to F32/F16/Q4_0/Q8_0 GGUF or strip weights",
            ty = ggml_ty,
            hint = ggml_type_hint(ggml_ty),
        ),
    })
}

const GGML_BLOCK_ELEMENTS: usize = 32;

fn dense_byte_len(width: usize, nelem: usize) -> Result<usize> {
    nelem
        .checked_mul(width)
        .filter(|&b| b > 0 || nelem == 0)
        .context("dense tensor byte size overflow")
}

fn dense_dtype(ty: u32) -> Result<DataType> {
    Ok(match ty {
        GGML_TYPE_F16 => DataType::F16,
        GGML_TYPE_BF16 => DataType::BF16,
        _ => anyhow::bail!("internal: not a 2-byte dense type"),
    })
}

fn ensure_multiple_of(nelem: usize, k: usize, ty: u32) -> Result<()> {
    if nelem == 0 {
        return Ok(());
    }
    if !nelem.is_multiple_of(k) {
        anyhow::bail!(
            "element count {nelem} is not divisible by block size {k} for {}",
            ggml_type_hint(ty)
        );
    }
    Ok(())
}

fn quant_block_byte_len(ty: u32, nelem: usize) -> Result<usize> {
    let blocks = nelem / GGML_BLOCK_ELEMENTS;
    let per_block = match ty {
        GGML_TYPE_Q4_0 => 18,
        GGML_TYPE_Q5_0 => 22,
        GGML_TYPE_Q8_0 => 34,
        _ => anyhow::bail!("not a handled block-quant layout"),
    };
    blocks
        .checked_mul(per_block)
        .context("quant tensor byte overflow")
}

fn unpack_ggml_payload(ty: u32, blob: &[u8], nelem: usize) -> Result<Vec<u8>> {
    Ok(match ty {
        GGML_TYPE_F64 => dequant::f64_blob_to_f32(blob)?,
        GGML_TYPE_Q4_0 => dequant::f32_blob_bytes(&dequant::dequantize_q4_0(blob, nelem)?),
        GGML_TYPE_Q5_0 => dequant::f32_blob_bytes(&dequant::dequantize_q5_0(blob, nelem)?),
        GGML_TYPE_Q8_0 => dequant::f32_blob_bytes(&dequant::dequantize_q8_0(blob, nelem)?),
        GGML_TYPE_Q4_K => dequant::f32_blob_bytes(&dequant::dequantize_q4_k(blob, nelem)?),
        GGML_TYPE_Q6_K => dequant::f32_blob_bytes(&dequant::dequantize_q6_k(blob, nelem)?),
        _ => blob.to_vec(),
    })
}

/// Dequantize a single `TensorData` that carries a GGUF-quantized dtype into an `f32` Vec.
/// Returns `None` if the dtype is not a supported GGUF quantization type.
pub fn dequantize_gguf_tensor(td: &TensorData) -> Option<Vec<f32>> {
    let nelem = td.element_count();
    match td.dtype {
        DataType::Q4_0 => dequant::dequantize_q4_0(&td.data, nelem).ok(),
        DataType::Q5_0 => dequant::dequantize_q5_0(&td.data, nelem).ok(),
        DataType::Q8_0 => dequant::dequantize_q8_0(&td.data, nelem).ok(),
        DataType::Q4_K => dequant::dequantize_q4_k(&td.data, nelem).ok(),
        DataType::Q6_K => dequant::dequantize_q6_k(&td.data, nelem).ok(),
        _ => None,
    }
}

fn ggml_type_hint(ty: u32) -> &'static str {
    match ty {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        12 => "Q4_K",
        14 => "Q6_K",
        28 => "F64",
        24 => "I8",
        30 => "BF16",
        _ => "see ggml.ggml_type docs",
    }
}

#[cfg(test)]
mod tests;

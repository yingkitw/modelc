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
            "tokenizer.chat_template" => {
                if value_type == VAL_STRING {
                    meta.chat_template = Some(cursor.read_string()?);
                } else {
                    let _ = cursor.read_kv_value_formatted(value_type);
                }
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
mod tests {
    use half::f16;

    use super::*;

    #[test]
    fn roundtrip_minimal_gguf_f32() -> Result<()> {
        let kv_count: u64 = 1;
        let tensor_count: u64 = 1;

        fn gguf_payload_string(s: &str) -> Vec<u8> {
            let mut b = Vec::new();
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
            b
        }

        let key = gguf_payload_string("general.architecture");

        let mut kv_block = Vec::new();
        kv_block.extend_from_slice(&key);
        kv_block.extend_from_slice(&VAL_STRING.to_le_bytes());
        kv_block.extend_from_slice(&gguf_payload_string("gpt2"));

        let w_name_block = gguf_payload_string("w");

        let mut ti = Vec::new();
        ti.extend_from_slice(&w_name_block);
        ti.extend_from_slice(&1u32.to_le_bytes()); // one dim
        ti.extend_from_slice(&2u64.to_le_bytes()); // inner length = 2
        ti.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        ti.extend_from_slice(&0u64.to_le_bytes());

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());

        buf.extend_from_slice(&kv_block);
        buf.extend_from_slice(&ti);

        let alignment = 32usize;
        let pad = align_offset(buf.len(), alignment) - buf.len();
        buf.extend(std::iter::repeat_n(0, pad));
        debug_assert_eq!(buf.len() % alignment, 0);

        let bytes: Vec<u8> = [11.3f32, 11.4f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        buf.extend_from_slice(&bytes);

        let m = parse_gguf_bytes(&buf, Path::new("fixture.gguf"))?;
        assert_eq!(m.architecture, "gpt2");
        assert_eq!(m.tensors.len(), 1);
        assert_eq!(m.tensors["w"].data, bytes);
        Ok(())
    }

    #[test]
    fn quantized_tensor_rejected() {
        /// GGML_TYPE_Q8_1 — not yet supported (Q4_0 / Q8_0 are expanded to F32).
        const GGML_TYPE_Q8_1: u32 = 9;

        let kv_block = encode_minimal_kv_generic();
        let w_name_block = gguf_string_bytes("w");
        let mut ti = Vec::new();
        ti.extend_from_slice(&w_name_block);
        ti.extend_from_slice(&1u32.to_le_bytes());
        ti.extend_from_slice(&128u64.to_le_bytes());
        ti.extend_from_slice(&GGML_TYPE_Q8_1.to_le_bytes());
        ti.extend_from_slice(&0u64.to_le_bytes());

        let r = build_minimal_roundtrip(kv_block, ti, &[0u8; 512]);
        let err = r.expect_err("Q8_1 tensors should fail");
        let s = format!("{err:#}");
        assert!(
            s.contains("unsupported GGML type 9"),
            "unexpected diagnostic: {s}"
        );
    }

    #[test]
    fn roundtrip_q4_0_zeros() -> Result<()> {
        let kv_block = encode_minimal_kv_generic();
        let w_name_block = gguf_string_bytes("w");
        let mut ti = Vec::new();
        ti.extend_from_slice(&w_name_block);
        ti.extend_from_slice(&1u32.to_le_bytes());
        ti.extend_from_slice(&32u64.to_le_bytes());
        ti.extend_from_slice(&GGML_TYPE_Q4_0.to_le_bytes());
        ti.extend_from_slice(&0u64.to_le_bytes());

        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());
        block.extend(vec![0x88u8; 16]);

        let tensor_count: u64 = 1;
        let kv_count: u64 = 1;
        let alignment = 32usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf.extend_from_slice(&kv_block);
        buf.extend_from_slice(&ti);
        buf.extend(std::iter::repeat_n(
            0u8,
            align_offset(buf.len(), alignment) - buf.len(),
        ));
        buf.extend_from_slice(&block);

        let m = parse_gguf_bytes(&buf, Path::new("q4.gguf"))?;
        let t = &m.tensors["w"];
        assert_eq!(t.dtype, DataType::Q4_0);
        assert_eq!(t.shape, vec![32usize]);
        assert_eq!(t.data.len(), 18);
        let f = dequantize_gguf_tensor(t).expect("dequantize");
        assert_eq!(f.len(), 32);
        assert!(f.iter().all(|&v| v.abs() < 1e-5), "Q4_0 zeros expected");
        Ok(())
    }

    /// Q8_0: one 32-wide block, delta 1.0, first int8 quant = 3 ⇒ first dequant value 3.0, rest 0.
    #[test]
    fn roundtrip_q8_0_scaled_first_element() -> Result<()> {
        let kv_block = encode_minimal_kv_generic();
        let w_name_block = gguf_string_bytes("w");
        let mut ti = Vec::new();
        ti.extend_from_slice(&w_name_block);
        ti.extend_from_slice(&1u32.to_le_bytes());
        ti.extend_from_slice(&32u64.to_le_bytes());
        ti.extend_from_slice(&GGML_TYPE_Q8_0.to_le_bytes());
        ti.extend_from_slice(&0u64.to_le_bytes());

        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        for j in 0..GGML_BLOCK_ELEMENTS {
            block.push(if j == 0 { 3i8 as u8 } else { 0 });
        }

        let tensor_count: u64 = 1;
        let kv_count: u64 = 1;
        let alignment = 32usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf.extend_from_slice(&kv_block);
        buf.extend_from_slice(&ti);
        buf.extend(std::iter::repeat_n(
            0u8,
            align_offset(buf.len(), alignment) - buf.len(),
        ));
        buf.extend_from_slice(&block);

        let m = parse_gguf_bytes(&buf, Path::new("q8.gguf"))?;
        let t = &m.tensors["w"];
        assert_eq!(t.dtype, DataType::Q8_0);
        assert_eq!(t.shape, vec![32usize]);
        assert_eq!(t.data.len(), 34);
        let f = dequantize_gguf_tensor(t).expect("dequantize");
        assert_eq!(f.len(), 32);
        assert!((f[0] - 3.0).abs() < 1e-5, "got {}", f[0]);
        for (i, &v) in f.iter().enumerate().skip(1) {
            assert!((v - 0.0).abs() < 1e-6, "idx {i} got {v}");
        }
        Ok(())
    }

    /// Q5_0: one 32-wide block, delta 1.0, all quant nibbles = 0x88 (high=1, low=0).
    /// For Q5_0: low nibble = 0x0 or 0x8 => (8 or 0) - 16 = -8 or -8.
    /// Actually qs byte 0x88: low nibble = 8, high nibble = 8.
    /// qh byte: bit=1 for first element.
    /// First value: (1<<4 | 8) = 24 - 16 = 8.
    /// Remaining: high=0, low=8 => 8 - 16 = -8.
    #[test]
    fn roundtrip_q5_0_scaled() -> Result<()> {
        let kv_block = encode_minimal_kv_generic();
        let w_name_block = gguf_string_bytes("w");
        let mut ti = Vec::new();
        ti.extend_from_slice(&w_name_block);
        ti.extend_from_slice(&1u32.to_le_bytes());
        ti.extend_from_slice(&32u64.to_le_bytes());
        ti.extend_from_slice(&GGML_TYPE_Q5_0.to_le_bytes());
        ti.extend_from_slice(&0u64.to_le_bytes());

        let mut block = Vec::with_capacity(22);
        block.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        // qs: 16 bytes, each 0x88 means low=8, high=8 for pairs
        block.extend(vec![0x88u8; 16]);
        // qh: 4 bytes, all 0 so only first 8 elements have high_bit=1
        block.extend(vec![0xFFu8; 4]);

        let tensor_count: u64 = 1;
        let kv_count: u64 = 1;
        let alignment = 32usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf.extend_from_slice(&kv_block);
        buf.extend_from_slice(&ti);
        buf.extend(std::iter::repeat_n(
            0u8,
            align_offset(buf.len(), alignment) - buf.len(),
        ));
        buf.extend_from_slice(&block);

        let m = parse_gguf_bytes(&buf, Path::new("q5.gguf"))?;
        let t = &m.tensors["w"];
        assert_eq!(t.dtype, DataType::Q5_0);
        assert_eq!(t.shape, vec![32usize]);
        assert_eq!(t.data.len(), 22);
        let f = dequantize_gguf_tensor(t).expect("dequantize");
        assert_eq!(f.len(), 32);
        // First few elements should be non-zero because high bit=1
        assert!(
            f[0].abs() > 0.01,
            "expected non-zero Q5_0 value, got {}",
            f[0]
        );
        Ok(())
    }

    /// Q4_K: 256-element superblock, 144 bytes.
    #[test]
    fn roundtrip_q4_k_scaled() -> Result<()> {
        let kv_block = encode_minimal_kv_generic();
        let w_name_block = gguf_string_bytes("w");
        let mut ti = Vec::new();
        ti.extend_from_slice(&w_name_block);
        ti.extend_from_slice(&1u32.to_le_bytes());
        ti.extend_from_slice(&256u64.to_le_bytes());
        ti.extend_from_slice(&GGML_TYPE_Q4_K.to_le_bytes());
        ti.extend_from_slice(&0u64.to_le_bytes());

        let mut block = Vec::with_capacity(144);
        // d = 1.0, dmin = 0.0
        block.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        block.extend_from_slice(&f16::from_f32(0.0).to_bits().to_le_bytes());
        // scales: 12 bytes
        block.extend(vec![1u8; 12]);
        // qs: 128 bytes of nibbles
        block.extend(vec![0x88u8; 128]);

        let tensor_count: u64 = 1;
        let kv_count: u64 = 1;
        let alignment = 32usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf.extend_from_slice(&kv_block);
        buf.extend_from_slice(&ti);
        buf.extend(std::iter::repeat_n(
            0u8,
            align_offset(buf.len(), alignment) - buf.len(),
        ));
        buf.extend_from_slice(&block);

        let m = parse_gguf_bytes(&buf, Path::new("q4k.gguf"))?;
        let t = &m.tensors["w"];
        assert_eq!(t.dtype, DataType::Q4_K);
        assert_eq!(t.shape, vec![256usize]);
        assert_eq!(t.data.len(), 144);
        let f = dequantize_gguf_tensor(t).expect("dequantize");
        assert_eq!(f.len(), 256);
        assert!(f[0].abs() < 100.0, "Q4_K sanity check");
        Ok(())
    }

    /// Q6_K: 256-element superblock, 210 bytes.
    #[test]
    fn roundtrip_q6_k_scaled() -> Result<()> {
        let kv_block = encode_minimal_kv_generic();
        let w_name_block = gguf_string_bytes("w");
        let mut ti = Vec::new();
        ti.extend_from_slice(&w_name_block);
        ti.extend_from_slice(&1u32.to_le_bytes());
        ti.extend_from_slice(&256u64.to_le_bytes());
        ti.extend_from_slice(&GGML_TYPE_Q6_K.to_le_bytes());
        ti.extend_from_slice(&0u64.to_le_bytes());

        let mut block = Vec::with_capacity(210);
        // d = 1.0
        block.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        // scales: 16 bytes
        block.extend(vec![1u8; 16]);
        // ql: 128 bytes of nibbles
        block.extend(vec![0x88u8; 128]);
        // qh: 64 bytes, all 0xFF for high 2 bits
        block.extend(vec![0xFFu8; 64]);

        let tensor_count: u64 = 1;
        let kv_count: u64 = 1;
        let alignment = 32usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf.extend_from_slice(&kv_block);
        buf.extend_from_slice(&ti);
        buf.extend(std::iter::repeat_n(
            0u8,
            align_offset(buf.len(), alignment) - buf.len(),
        ));
        buf.extend_from_slice(&block);

        let m = parse_gguf_bytes(&buf, Path::new("q6k.gguf"))?;
        let t = &m.tensors["w"];
        assert_eq!(t.dtype, DataType::Q6_K);
        assert_eq!(t.shape, vec![256usize]);
        assert_eq!(t.data.len(), 210);
        let f = dequantize_gguf_tensor(t).expect("dequantize");
        assert_eq!(f.len(), 256);
        assert!(f[0].abs() < 100.0, "Q6_K sanity check");
        Ok(())
    }

    #[test]
    fn roundtrip_f64_tensor_to_f32_payload() -> Result<()> {
        let kv_count: u64 = 1;
        let tensor_count: u64 = 1;

        let mut kv_block = Vec::new();
        kv_block.extend_from_slice(&gguf_string_bytes("general.architecture"));
        kv_block.extend_from_slice(&VAL_STRING.to_le_bytes());
        kv_block.extend_from_slice(&gguf_string_bytes("custom"));

        let mut ti = Vec::new();
        ti.extend_from_slice(&gguf_string_bytes("d"));
        ti.extend_from_slice(&1u32.to_le_bytes());
        ti.extend_from_slice(&2u64.to_le_bytes());
        ti.extend_from_slice(&GGML_TYPE_F64.to_le_bytes());
        ti.extend_from_slice(&0u64.to_le_bytes());

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf.extend_from_slice(&kv_block);
        buf.extend_from_slice(&ti);
        let alignment = 32usize;
        buf.extend(std::iter::repeat_n(
            0u8,
            align_offset(buf.len(), alignment) - buf.len(),
        ));

        let f64_blob: Vec<u8> = [1.25f64, -0.5f64]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        buf.extend_from_slice(&f64_blob);

        let m = parse_gguf_bytes(&buf, Path::new("f64.gguf"))?;
        assert_eq!(m.architecture, "custom");
        let t = &m.tensors["d"];
        assert_eq!(t.dtype, DataType::F32);
        let out: Vec<f32> = t
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!((out[0] - 1.25f32).abs() < 1e-5);
        assert!((out[1] - (-0.5f32)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn extract_tokenizer_metadata_reads_vocab_and_merges() -> Result<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());

        let tensor_count: u64 = 0;
        let kv_count: u64 = 4;
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());

        // tokenizer.ggml.model
        buf.extend_from_slice(&gguf_string_bytes("tokenizer.ggml.model"));
        buf.extend_from_slice(&VAL_STRING.to_le_bytes());
        buf.extend_from_slice(&gguf_string_bytes("gpt2"));

        // tokenizer.ggml.tokens (array of 3 strings)
        buf.extend_from_slice(&gguf_string_bytes("tokenizer.ggml.tokens"));
        buf.extend_from_slice(&VAL_ARRAY.to_le_bytes());
        buf.extend_from_slice(&VAL_STRING.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&gguf_string_bytes("a"));
        buf.extend_from_slice(&gguf_string_bytes("b"));
        buf.extend_from_slice(&gguf_string_bytes("ab"));

        // tokenizer.ggml.merges (array of 1 string)
        buf.extend_from_slice(&gguf_string_bytes("tokenizer.ggml.merges"));
        buf.extend_from_slice(&VAL_ARRAY.to_le_bytes());
        buf.extend_from_slice(&VAL_STRING.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&gguf_string_bytes("a b"));

        // tokenizer.ggml.bos_token_id
        buf.extend_from_slice(&gguf_string_bytes("tokenizer.ggml.bos_token_id"));
        buf.extend_from_slice(&VAL_UINT32.to_le_bytes());
        buf.extend_from_slice(&100u32.to_le_bytes());

        let tmp = tempfile::NamedTempFile::new()?;
        std::fs::write(tmp.path(), &buf)?;

        let meta = extract_tokenizer_metadata(tmp.path())?;
        assert_eq!(meta.model, "gpt2");
        assert_eq!(meta.vocab, vec!["a", "b", "ab"]);
        assert_eq!(meta.merges, vec!["a b"]);
        assert_eq!(meta.bos_token_id, Some(100));
        assert_eq!(meta.eos_token_id, None);
        Ok(())
    }

    fn gguf_string_bytes(s: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
        b
    }

    fn encode_minimal_kv_generic() -> Vec<u8> {
        let mut kv_block = Vec::new();
        kv_block.extend_from_slice(&gguf_string_bytes("general.architecture"));
        kv_block.extend_from_slice(&VAL_STRING.to_le_bytes());
        kv_block.extend_from_slice(&gguf_string_bytes("llama"));
        kv_block
    }

    fn build_minimal_roundtrip(kv_block: Vec<u8>, ti: Vec<u8>, tensor_blob: &[u8]) -> Result<()> {
        let tensor_count: u64 = 1;
        let kv_count: u64 = 1;
        let alignment = 32usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf.extend_from_slice(&kv_block);
        buf.extend_from_slice(&ti);
        buf.extend(std::iter::repeat_n(
            0u8,
            align_offset(buf.len(), alignment) - buf.len(),
        ));
        buf.extend_from_slice(tensor_blob);
        parse_gguf_bytes(&buf, Path::new("x.gguf"))?;
        Ok(())
    }
}

//! GGUF reader (little-endian). Loads dense tensors and **Q4_0 / Q8_0** GGML blocks expanded to
//! **`f32`** payloads in IR. Other quant types still error with a descriptive message. See
//! [GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use byteorder::{ByteOrder, LittleEndian};
use half::f16;

use crate::model::{DataType, Model, TensorData};
use crate::parsers::WeightParser;

pub struct GgufParser;

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_I8: u32 = 24;
const GGML_TYPE_I16: u32 = 25;
const GGML_TYPE_I32: u32 = 26;
const GGML_TYPE_I64: u32 = 27;
const GGML_TYPE_F64: u32 = 28;
const GGML_TYPE_BF16: u32 = 30;

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

fn parse_gguf_bytes(data: &[u8], path: &Path) -> Result<Model> {
    let mut cursor = Cursor::new(data);
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

    let mut tensor_base = cursor.pos;
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

        raw_slice = unpack_ggml_payload(ti.ggml_ty, &raw_slice, nelem).with_context(|| {
            format!("tensor {:?} (type {}): payload decode", ti.name, ti.ggml_ty)
        })?;

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
        GGML_TYPE_Q4_0 | GGML_TYPE_Q8_0 => {
            ensure_multiple_of(nelem, GGML_BLOCK_ELEMENTS, ggml_ty)?;
            let bytes = quant_block_byte_len(ggml_ty, nelem)?;
            (DataType::F32, bytes)
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
        GGML_TYPE_Q8_0 => 34,
        _ => anyhow::bail!("not a handled block-quant layout"),
    };
    blocks
        .checked_mul(per_block)
        .context("quant tensor byte overflow")
}

fn unpack_ggml_payload(ty: u32, blob: &[u8], nelem: usize) -> Result<Vec<u8>> {
    Ok(match ty {
        GGML_TYPE_F64 => f64_blob_to_f32(blob)?,
        GGML_TYPE_Q4_0 => f32_blob_bytes(&dequantize_q4_0(blob, nelem)?),
        GGML_TYPE_Q8_0 => f32_blob_bytes(&dequantize_q8_0(blob, nelem)?),
        _ => blob.to_vec(),
    })
}

fn f64_blob_to_f32(blob: &[u8]) -> Result<Vec<u8>> {
    anyhow::ensure!(
        blob.len().is_multiple_of(8),
        "unexpected f64 payload length {}",
        blob.len()
    );
    let mut out = Vec::with_capacity(blob.len() / 2);
    for chunk in blob.chunks_exact(8) {
        let f = f64::from_le_bytes(chunk.try_into().unwrap()) as f32;
        out.extend_from_slice(&f.to_le_bytes());
    }
    Ok(out)
}

fn f32_blob_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn dequantize_q4_0(src: &[u8], nelem: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(nelem.is_multiple_of(GGML_BLOCK_ELEMENTS), "q4_0 uneven row");
    const BLK_BYTES: usize = 18;
    let nb = nelem / GGML_BLOCK_ELEMENTS;
    anyhow::ensure!(src.len() == nb * BLK_BYTES);
    let mut y = vec![0f32; nelem];

    // Matches `dequantize_row_q4_0` in llama.cpp / ggml.
    for i in 0..nb {
        let bo = i * BLK_BYTES;
        let d = f16::from_bits(LittleEndian::read_u16(&src[bo..bo + 2])).to_f32();
        let qs = &src[bo + 2..bo + BLK_BYTES];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            y[i * GGML_BLOCK_ELEMENTS + j] = x0 as f32 * d;
            y[i * GGML_BLOCK_ELEMENTS + j + GGML_BLOCK_ELEMENTS / 2] = x1 as f32 * d;
        }
    }
    Ok(y)
}

fn dequantize_q8_0(src: &[u8], nelem: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(nelem.is_multiple_of(GGML_BLOCK_ELEMENTS), "q8_0 uneven row");
    const BLK_BYTES: usize = 34;
    let nb = nelem / GGML_BLOCK_ELEMENTS;
    anyhow::ensure!(src.len() == nb * BLK_BYTES);
    let mut y = vec![0f32; nelem];
    // Matches `dequantize_row_q8_0`: `half` delta × int8 quants (ggml-common.h layout).
    for i in 0..nb {
        let bo = i * BLK_BYTES;
        let d = f16::from_bits(LittleEndian::read_u16(&src[bo..bo + 2])).to_f32();
        for j in 0..GGML_BLOCK_ELEMENTS {
            let q = src[bo + 2 + j] as i8;
            y[i * GGML_BLOCK_ELEMENTS + j] = q as f32 * d;
        }
    }
    Ok(y)
}

fn ggml_type_hint(ty: u32) -> &'static str {
    match ty {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        8 => "Q8_0",
        28 => "F64",
        24 => "I8",
        30 => "BF16",
        _ => "see ggml.ggml_type docs",
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn expect_bytes(&mut self, pat: &[u8]) -> Result<()> {
        let end = self.pos.checked_add(pat.len()).context("oob")?;
        if end > self.data.len() || &self.data[self.pos..end] != pat {
            anyhow::bail!("expected {:?}", pat);
        }
        self.pos = end;
        Ok(())
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let end = self.pos.checked_add(4).context("read past end")?;
        if end > self.data.len() {
            anyhow::bail!("unexpected EOF reading u32");
        }
        let v = LittleEndian::read_u32(&self.data[self.pos..end]);
        self.pos = end;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let end = self.pos.checked_add(8).context("read past end")?;
        if end > self.data.len() {
            anyhow::bail!("unexpected EOF reading u64");
        }
        let v = LittleEndian::read_u64(&self.data[self.pos..end]);
        self.pos = end;
        Ok(v)
    }

    fn read_string(&mut self) -> Result<String> {
        let len_u = self.read_u64()?;
        let len = usize::try_from(len_u).map_err(|_| anyhow!("string length overflow"))?;
        if len > MAX_STRING_BYTES {
            anyhow::bail!("GGUF string too large ({len} bytes)");
        }
        let end = self.pos.checked_add(len).context("string bounds")?;
        if end > self.data.len() {
            anyhow::bail!("unexpected EOF in GGUF string");
        }
        let s = std::str::from_utf8(&self.data[self.pos..end])
            .context("invalid UTF-8 in GGUF string")?;
        self.pos = end;
        Ok(s.to_string())
    }

    fn read_kv_value_formatted(&mut self, vt: u32) -> Result<String> {
        Ok(match vt {
            VAL_UINT8 => format!("{}", self.read_u8()?),
            VAL_INT8 => format!("{}", self.read_i8()?),
            VAL_UINT16 => format!("{}", self.read_u16()?),
            VAL_INT16 => format!("{}", self.read_i16()?),
            VAL_UINT32 => format!("{}", self.read_u32()?),
            VAL_INT32 => format!("{}", self.read_i32()?),
            VAL_FLOAT32 => format!("{}", f32::from_bits(self.read_u32()?)),
            VAL_BOOL => format!("{}", self.read_u8()? != 0),
            VAL_STRING => self.read_string()?,
            VAL_ARRAY => self.read_kv_array_summary()?,
            VAL_UINT64 => format!("{}", self.read_u64()?),
            VAL_INT64 => format!("{}", self.read_i64()?),
            VAL_FLOAT64 => format!("{}", f64::from_bits(self.read_u64()?)),
            _ => anyhow::bail!("unknown GGUF metadata value type {vt}"),
        })
    }

    fn read_kv_array_summary(&mut self) -> Result<String> {
        if self.remaining() < 12 {
            anyhow::bail!("EOF reading array header");
        }
        let inner_type = self.read_u32()?;
        let len_u = self.read_u64()?;
        let n = usize::try_from(len_u).map_err(|_| anyhow!("array length overflow"))?;
        let preview = n.min(32);
        let mut parts = Vec::new();
        for i in 0..n {
            if i < preview {
                parts.push(self.read_kv_value_formatted(inner_type)?);
                continue;
            }
            self.consume_kv_value(inner_type)?;
        }
        if n > preview {
            parts.push(format!("…(+{} omitted)", len_u - preview as u64));
        }
        Ok(format!("[{}]", parts.join(",")))
    }

    fn consume_kv_value(&mut self, vt: u32) -> Result<()> {
        match vt {
            VAL_UINT8 => {
                let _ = self.read_u8()?;
            }
            VAL_INT8 => {
                let _ = self.read_i8()?;
            }
            VAL_UINT16 => {
                let _ = self.read_u16()?;
            }
            VAL_INT16 => {
                let _ = self.read_i16()?;
            }
            VAL_UINT32 => {
                let _ = self.read_u32()?;
            }
            VAL_INT32 => {
                let _ = self.read_i32()?;
            }
            VAL_FLOAT32 => {
                let _ = self.read_u32()?;
            }
            VAL_BOOL => {
                let _ = self.read_u8()?;
            }
            VAL_STRING => {
                let _ = self.read_string()?;
            }
            VAL_UINT64 => {
                let _ = self.read_u64()?;
            }
            VAL_INT64 => {
                let _ = self.read_i64()?;
            }
            VAL_FLOAT64 => {
                let _ = self.read_u64()?;
            }
            VAL_ARRAY => {
                let inner = self.read_u32()?;
                let len_u = self.read_u64()?;
                let n = usize::try_from(len_u).map_err(|_| anyhow!("array length overflow"))?;
                for _ in 0..n {
                    self.consume_kv_value(inner)?;
                }
            }
            _ => anyhow::bail!("unknown GGUF metadata value type for skip {vt}"),
        }
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            anyhow::bail!("EOF u8");
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let end = self.pos.checked_add(2).context("read_u16 bounds")?;
        if end > self.data.len() {
            anyhow::bail!("unexpected EOF reading u16");
        }
        let v = LittleEndian::read_u16(&self.data[self.pos..end]);
        self.pos = end;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_i32(&mut self) -> Result<i32> {
        let v = self.read_u32()?;
        Ok(v as i32)
    }

    fn read_i64(&mut self) -> Result<i64> {
        let v = self.read_u64()?;
        Ok(v as i64)
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
        /// GGML_TYPE_Q4_K — still unsupported here (Q4_0 / Q8_0 are expanded to F32).
        const GGML_TYPE_Q4_K: u32 = 12;

        let kv_block = encode_minimal_kv_generic();
        let w_name_block = gguf_string_bytes("w");
        let mut ti = Vec::new();
        ti.extend_from_slice(&w_name_block);
        ti.extend_from_slice(&1u32.to_le_bytes());
        ti.extend_from_slice(&128u64.to_le_bytes());
        ti.extend_from_slice(&GGML_TYPE_Q4_K.to_le_bytes());
        ti.extend_from_slice(&0u64.to_le_bytes());

        let r = build_minimal_roundtrip(kv_block, ti, &[0u8; 512]);
        let err = r.expect_err("Q4_K tensors should fail");
        let s = format!("{err:#}");
        assert!(
            s.contains("unsupported GGML type 12"),
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
        assert_eq!(t.dtype, DataType::F32);
        assert_eq!(t.shape, vec![32usize]);
        assert_eq!(t.data, vec![0u8; 32 * 4]);
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
        assert_eq!(t.dtype, DataType::F32);
        assert_eq!(t.shape, vec![32usize]);
        let first = f32::from_le_bytes(std::convert::TryInto::try_into(&t.data[0..4]).unwrap());
        assert!((first - 3.0).abs() < 1e-5, "got {first}");
        for i in 1..32 {
            let v = f32::from_le_bytes(
                std::convert::TryInto::try_into(&t.data[i * 4..(i + 1) * 4]).unwrap(),
            );
            assert!((v - 0.0).abs() < 1e-6, "idx {i} got {v}");
        }
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

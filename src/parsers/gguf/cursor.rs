use anyhow::{Context, Result, anyhow};
use byteorder::{ByteOrder, LittleEndian};

use super::{
    MAX_STRING_BYTES, VAL_ARRAY, VAL_BOOL, VAL_FLOAT32, VAL_FLOAT64, VAL_INT8, VAL_INT16,
    VAL_INT32, VAL_INT64, VAL_STRING, VAL_UINT8, VAL_UINT16, VAL_UINT32, VAL_UINT64,
};

pub(super) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(super) fn pos(&self) -> usize {
        self.pos
    }

    pub(super) fn expect_bytes(&mut self, pat: &[u8]) -> Result<()> {
        let end = self.pos.checked_add(pat.len()).context("oob")?;
        if end > self.data.len() || &self.data[self.pos..end] != pat {
            anyhow::bail!("expected {:?}", pat);
        }
        self.pos = end;
        Ok(())
    }

    pub(super) fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub(super) fn read_u32(&mut self) -> Result<u32> {
        let end = self.pos.checked_add(4).context("read past end")?;
        if end > self.data.len() {
            anyhow::bail!("unexpected EOF reading u32");
        }
        let v = LittleEndian::read_u32(&self.data[self.pos..end]);
        self.pos = end;
        Ok(v)
    }

    pub(super) fn read_u64(&mut self) -> Result<u64> {
        let end = self.pos.checked_add(8).context("read past end")?;
        if end > self.data.len() {
            anyhow::bail!("unexpected EOF reading u64");
        }
        let v = LittleEndian::read_u64(&self.data[self.pos..end]);
        self.pos = end;
        Ok(v)
    }

    pub(super) fn read_string(&mut self) -> Result<String> {
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

    pub(super) fn read_kv_value_formatted(&mut self, vt: u32) -> Result<String> {
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

    /// Read a GGUF metadata array of strings and return all elements.
    /// `inner_type` must be `VAL_STRING`.
    pub(super) fn read_string_array(&mut self) -> Result<Vec<String>> {
        let inner_type = self.read_u32()?;
        if inner_type != VAL_STRING {
            anyhow::bail!("expected string array, got inner type {inner_type}");
        }
        let len_u = self.read_u64()?;
        let n = usize::try_from(len_u).map_err(|_| anyhow!("array length overflow"))?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_string()?);
        }
        Ok(out)
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

    pub(super) fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            anyhow::bail!("EOF u8");
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    pub(super) fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    pub(super) fn read_u16(&mut self) -> Result<u16> {
        let end = self.pos.checked_add(2).context("read_u16 bounds")?;
        if end > self.data.len() {
            anyhow::bail!("unexpected EOF reading u16");
        }
        let v = LittleEndian::read_u16(&self.data[self.pos..end]);
        self.pos = end;
        Ok(v)
    }

    pub(super) fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    pub(super) fn read_i32(&mut self) -> Result<i32> {
        let v = self.read_u32()?;
        Ok(v as i32)
    }

    pub(super) fn read_i64(&mut self) -> Result<i64> {
        let v = self.read_u64()?;
        Ok(v as i64)
    }
}

use anyhow::{Result, ensure};
use byteorder::{ByteOrder, LittleEndian};
use half::f16;

pub(super) fn f64_blob_to_f32(blob: &[u8]) -> Result<Vec<u8>> {
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

pub(super) fn f32_blob_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub(super) fn dequantize_q4_0(src: &[u8], nelem: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(nelem.is_multiple_of(super::GGML_BLOCK_ELEMENTS), "q4_0 uneven row");
    const BLK_BYTES: usize = 18;
    let nb = nelem / super::GGML_BLOCK_ELEMENTS;
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
            y[i * super::GGML_BLOCK_ELEMENTS + j] = x0 as f32 * d;
            y[i * super::GGML_BLOCK_ELEMENTS + j + super::GGML_BLOCK_ELEMENTS / 2] = x1 as f32 * d;
        }
    }
    Ok(y)
}

pub(super) fn dequantize_q8_0(src: &[u8], nelem: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(nelem.is_multiple_of(super::GGML_BLOCK_ELEMENTS), "q8_0 uneven row");
    const BLK_BYTES: usize = 34;
    let nb = nelem / super::GGML_BLOCK_ELEMENTS;
    anyhow::ensure!(src.len() == nb * BLK_BYTES);
    let mut y = vec![0f32; nelem];
    // Matches `dequantize_row_q8_0`: `half` delta × int8 quants (ggml-common.h layout).
    for i in 0..nb {
        let bo = i * BLK_BYTES;
        let d = f16::from_bits(LittleEndian::read_u16(&src[bo..bo + 2])).to_f32();
        for j in 0..super::GGML_BLOCK_ELEMENTS {
            let q = src[bo + 2 + j] as i8;
            y[i * super::GGML_BLOCK_ELEMENTS + j] = q as f32 * d;
        }
    }
    Ok(y)
}

pub(super) fn dequantize_q5_0(src: &[u8], nelem: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(nelem.is_multiple_of(super::GGML_BLOCK_ELEMENTS), "q5_0 uneven row");
    const BLK_BYTES: usize = 22;
    let nb = nelem / super::GGML_BLOCK_ELEMENTS;
    anyhow::ensure!(src.len() == nb * BLK_BYTES);
    let mut y = vec![0f32; nelem];

    for i in 0..nb {
        let bo = i * BLK_BYTES;
        let d = f16::from_bits(LittleEndian::read_u16(&src[bo..bo + 2])).to_f32();
        let qs = &src[bo + 2..bo + 18];
        let qh = &src[bo + 18..bo + 22];

        for j in 0..super::GGML_BLOCK_ELEMENTS {
            let qs_idx = j / 2;
            let low_nibble = if j % 2 == 0 {
                qs[qs_idx] & 0x0F
            } else {
                qs[qs_idx] >> 4
            };
            let qh_byte_idx = j / 8;
            let qh_bit_idx = j % 8;
            let high_bit = (qh[qh_byte_idx] >> qh_bit_idx) & 1;
            let value = (((high_bit << 4) | low_nibble) as i32 - 16) as f32 * d;
            y[i * super::GGML_BLOCK_ELEMENTS + j] = value;
        }
    }
    Ok(y)
}

pub(super) fn dequantize_q4_k(src: &[u8], nelem: usize) -> Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLK_BYTES: usize = 144;
    anyhow::ensure!(nelem.is_multiple_of(QK_K), "q4_k uneven superblock");
    let nb = nelem / QK_K;
    anyhow::ensure!(src.len() == nb * BLK_BYTES);
    let mut y = vec![0f32; nelem];

    for i in 0..nb {
        let bo = i * BLK_BYTES;
        let d = f16::from_bits(LittleEndian::read_u16(&src[bo..bo + 2])).to_f32();
        let dmin = f16::from_bits(LittleEndian::read_u16(&src[bo + 2..bo + 4])).to_f32();
        let scales = &src[bo + 4..bo + 16];
        let qs = &src[bo + 16..bo + 16 + 128];

        for sb in 0..8 {
            let scale = scales[sb] as f32 * d;
            let min = scales[sb] as f32 * dmin;
            for j in 0..32 {
                let qs_idx = sb * 16 + j / 2;
                let nibble = if j % 2 == 0 {
                    qs[qs_idx] & 0x0F
                } else {
                    qs[qs_idx] >> 4
                };
                let value = (nibble as f32 - 8.0) * scale + min;
                y[i * QK_K + sb * 32 + j] = value;
            }
        }
    }
    Ok(y)
}

pub(super) fn dequantize_q6_k(src: &[u8], nelem: usize) -> Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLK_BYTES: usize = 210;
    anyhow::ensure!(nelem.is_multiple_of(QK_K), "q6_k uneven superblock");
    let nb = nelem / QK_K;
    anyhow::ensure!(src.len() == nb * BLK_BYTES);
    let mut y = vec![0f32; nelem];

    for i in 0..nb {
        let bo = i * BLK_BYTES;
        let d = f16::from_bits(LittleEndian::read_u16(&src[bo..bo + 2])).to_f32();
        let scales = &src[bo + 2..bo + 18];
        let ql = &src[bo + 18..bo + 146];
        let qh = &src[bo + 146..bo + 210];

        for sb in 0..16 {
            let scale = scales[sb] as f32 * d;
            for j in 0..16 {
                let idx = sb * 16 + j;
                let ql_idx = idx / 2;
                let low_nibble = if idx % 2 == 0 {
                    ql[ql_idx] & 0x0F
                } else {
                    ql[ql_idx] >> 4
                };
                let qh_byte_idx = idx / 4;
                let qh_shift = (idx % 4) * 2;
                let high_2bit = (qh[qh_byte_idx] >> qh_shift) & 0x03;
                let combined = ((high_2bit << 4) | low_nibble) as i32 - 32;
                y[i * QK_K + idx] = combined as f32 * scale;
            }
        }
    }
    Ok(y)
}

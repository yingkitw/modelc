
use half::f16;

use super::cursor::Cursor;
use super::dequant::{dequantize_q4_0, dequantize_q8_0};
use super::*;
use crate::tempdir::tempfile;

#[test]
fn roundtrip_minimal_gguf_f32() -> Result<()> {
    let buf = minimal_f32_gguf();
    let m = parse_gguf_bytes(&buf, Path::new("fixture.gguf"))?;
    assert_eq!(m.architecture, "gpt2");
    assert_eq!(m.tensors.len(), 1);
    assert_eq!(m.tensors["w"].data, [11.3f32, 11.4f32].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>());
    Ok(())
}

/// A one-tensor F32 GGUF with `general.architecture = "gpt2"` — reused by the
/// trait-path and error-path tests below.
fn minimal_f32_gguf() -> Vec<u8> {
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
    buf
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

    let tmp = tempfile()?;
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

// ---- Cursor unit tests (src/parsers/gguf/cursor.rs) ----

#[test]
fn cursor_read_u32_roundtrip() {
    let bytes = 0xDEADBEEFu32.to_le_bytes();
    let mut c = Cursor::new(&bytes);
    assert_eq!(c.read_u32().unwrap(), 0xDEADBEEFu32);
    assert_eq!(c.remaining(), 0);
}

#[test]
fn cursor_read_u32_eof_errors() {
    let bytes = [1u8, 2, 3]; // 1 byte short
    let mut c = Cursor::new(&bytes);
    let err = c.read_u32().unwrap_err();
    assert!(format!("{err:#}").contains("EOF"));
}

#[test]
fn cursor_read_u64_roundtrip() {
    let bytes = 0x0102_0304_0506_0708u64.to_le_bytes();
    let mut c = Cursor::new(&bytes);
    assert_eq!(c.read_u64().unwrap(), 0x0102_0304_0506_0708u64);
}

#[test]
fn cursor_read_u64_eof_errors() {
    let bytes = [1u8; 4]; // 4 bytes short
    let mut c = Cursor::new(&bytes);
    assert!(c.read_u64().is_err());
}

#[test]
fn cursor_read_u16_roundtrip() {
    let bytes = 0xBEEFu16.to_le_bytes();
    let mut c = Cursor::new(&bytes);
    assert_eq!(c.read_u16().unwrap(), 0xBEEFu16);
}

#[test]
fn cursor_read_u8_and_i8() {
    let bytes = [0xFFu8, 0x00, 0x7F];
    let mut c = Cursor::new(&bytes);
    assert_eq!(c.read_u8().unwrap(), 0xFF);
    assert_eq!(c.read_i8().unwrap(), 0x00);
    assert_eq!(c.read_i8().unwrap(), 0x7F);
}

#[test]
fn cursor_read_u8_eof_errors() {
    let mut c = Cursor::new(&[]);
    assert!(c.read_u8().is_err());
}

#[test]
fn cursor_i32_i64_i16_via_u32() {
    // Layout: [u32 -1][u64 0]
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // -1 as i32
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let mut c = Cursor::new(&bytes);
    assert_eq!(c.read_i32().unwrap(), -1);
    assert_eq!(c.read_i64().unwrap(), 0);
}

#[test]
fn cursor_expect_bytes_match_and_advance() {
    let bytes = b"GGUF\x03\x00";
    let mut c = Cursor::new(bytes);
    c.expect_bytes(b"GGUF").unwrap();
    assert_eq!(c.pos(), 4);
}

#[test]
fn cursor_expect_bytes_mismatch_errors() {
    let bytes = b"GGUF";
    let mut c = Cursor::new(bytes);
    assert!(c.expect_bytes(b"GGU2").is_err());
}

#[test]
fn cursor_expect_bytes_oob_errors() {
    let bytes = b"GG";
    let mut c = Cursor::new(bytes);
    assert!(c.expect_bytes(b"GGUF").is_err());
}

#[test]
fn cursor_remaining_saturates() {
    let mut c = Cursor::new(&[1, 2, 3]);
    assert_eq!(c.remaining(), 3);
    c.read_u8().unwrap();
    c.read_u8().unwrap();
    c.read_u8().unwrap();
    assert_eq!(c.remaining(), 0);
    // After EOF, further reads fail but `remaining()` stays 0 (saturating_sub).
    assert!(c.read_u8().is_err());
    assert_eq!(c.remaining(), 0);
}

#[test]
fn cursor_read_string_roundtrip() {
    let mut bytes = Vec::new();
    let s = "hello world";
    bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
    bytes.extend_from_slice(s.as_bytes());
    let mut c = Cursor::new(&bytes);
    assert_eq!(c.read_string().unwrap(), s);
}

#[test]
fn cursor_read_string_eof_errors() {
    // Length says 6 bytes follow but only 3 are present.
    let bytes = [6u8, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', b'c'];
    let mut c = Cursor::new(&bytes);
    assert!(c.read_string().is_err());
}

#[test]
fn cursor_read_string_invalid_utf8_errors() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(3u64).to_le_bytes());
    bytes.extend_from_slice(&[0xFF, 0xFE, 0xFD]); // not UTF-8
    let mut c = Cursor::new(&bytes);
    assert!(c.read_string().is_err());
}

#[test]
fn cursor_read_string_too_large_errors() {
    // Claim a 1GB+ length — MAX_STRING_BYTES should reject.
    let mut bytes = Vec::new();
    let huge_len: u64 = 2 * 1024 * 1024 * 1024;
    bytes.extend_from_slice(&huge_len.to_le_bytes());
    let mut c = Cursor::new(&bytes);
    let err = c.read_string().unwrap_err();
    assert!(format!("{err:#}").contains("too large"));
}

#[test]
fn cursor_read_kv_value_formatted_all_scalar_types() {
    // Build a buffer where each value is preceded by its size matching
    // the kv-value-formatted parser's expectations. read_kv_value_formatted
    // reads the value *body* (not the type tag).
    let mut bytes = Vec::new();
    bytes.push(1u8); // u8=1
    bytes.push((-2i8) as u8); // i8=-2
    bytes.extend_from_slice(&3u16.to_le_bytes()); // u16=3
    bytes.extend_from_slice(&(-4i16).to_le_bytes()); // i16=-4
    bytes.extend_from_slice(&5u32.to_le_bytes()); // u32=5
    bytes.extend_from_slice(&(-6i32).to_le_bytes()); // i32=-6
    bytes.extend_from_slice(&0.5f32.to_bits().to_le_bytes()); // f32
    bytes.push(1u8); // bool=true
    bytes.extend_from_slice(&3u64.to_le_bytes()); // string len
    bytes.extend_from_slice(b"yes");
    bytes.extend_from_slice(&42u64.to_le_bytes()); // u64
    bytes.extend_from_slice(&(-1i64).to_le_bytes()); // i64
    bytes.extend_from_slice(&2.5f64.to_bits().to_le_bytes()); // f64

    let mut c = Cursor::new(&bytes);
    assert_eq!(c.read_kv_value_formatted(VAL_UINT8).unwrap(), "1");
    assert_eq!(c.read_kv_value_formatted(VAL_INT8).unwrap(), "-2");
    assert_eq!(c.read_kv_value_formatted(VAL_UINT16).unwrap(), "3");
    assert_eq!(c.read_kv_value_formatted(VAL_INT16).unwrap(), "-4");
    assert_eq!(c.read_kv_value_formatted(VAL_UINT32).unwrap(), "5");
    assert_eq!(c.read_kv_value_formatted(VAL_INT32).unwrap(), "-6");
    assert!(
        c.read_kv_value_formatted(VAL_FLOAT32)
            .unwrap()
            .contains("0.5")
    );
    assert_eq!(c.read_kv_value_formatted(VAL_BOOL).unwrap(), "true");
    assert_eq!(c.read_kv_value_formatted(VAL_STRING).unwrap(), "yes");
    assert_eq!(c.read_kv_value_formatted(VAL_UINT64).unwrap(), "42");
    assert_eq!(c.read_kv_value_formatted(VAL_INT64).unwrap(), "-1");
    assert_eq!(c.read_kv_value_formatted(VAL_FLOAT64).unwrap(), "2.5");
}

#[test]
fn cursor_read_kv_value_formatted_unknown_type_errors() {
    let mut c = Cursor::new(&[]);
    let err = c.read_kv_value_formatted(999).unwrap_err();
    assert!(format!("{err:#}").contains("unknown GGUF metadata value type"));
}

#[test]
fn cursor_read_string_array_roundtrip() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&VAL_STRING.to_le_bytes());
    bytes.extend_from_slice(&3u64.to_le_bytes());
    for s in ["one", "two", "three"] {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
    }
    let mut c = Cursor::new(&bytes);
    assert_eq!(c.read_string_array().unwrap(), vec!["one", "two", "three"]);
}

#[test]
fn cursor_read_string_array_wrong_inner_type_errors() {
    let bytes = VAL_UINT32.to_le_bytes(); // not VAL_STRING
    let mut c = Cursor::new(&bytes);
    assert!(c.read_string_array().is_err());
}

/// Encode a GGUF metadata array body: inner_type tag + u64 len + element payloads.
fn kv_array_bytes(inner_type: u32, elements: &[Vec<u8>]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&inner_type.to_le_bytes());
    b.extend_from_slice(&(elements.len() as u64).to_le_bytes());
    for e in elements {
        b.extend_from_slice(e);
    }
    b
}

fn gguf_str_bytes(s: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
    b
}

#[test]
fn cursor_kv_array_summary_with_truncation() {
    // 33-element u8 array → 32 previewed + 1 consumed → "…(+1 omitted)".
    let elems: Vec<Vec<u8>> = (0u8..33).map(|i| vec![i]).collect();
    let bytes = kv_array_bytes(VAL_UINT8, &elems);
    let mut c = Cursor::new(&bytes);
    let summary = c.read_kv_value_formatted(VAL_ARRAY).unwrap();
    assert!(summary.ends_with(",…(+1 omitted)]"), "got {summary}");
    assert!(summary.starts_with("[0,1,2,"), "got {summary}");
}

#[test]
fn cursor_kv_array_summary_short() {
    let elems: Vec<Vec<u8>> = vec![7u32.to_le_bytes().to_vec(); 3];
    let bytes = kv_array_bytes(VAL_UINT32, &elems);
    let mut c = Cursor::new(&bytes);
    assert_eq!(c.read_kv_value_formatted(VAL_ARRAY).unwrap(), "[7,7,7]");
}

#[test]
fn cursor_kv_array_summary_eof_errors() {
    // Fewer than 12 bytes remaining → header can't be read.
    let mut c = Cursor::new(&[0u8; 8]);
    assert!(c.read_kv_value_formatted(VAL_ARRAY).is_err());
}

#[test]
fn cursor_consume_kv_value_skips_string_array() {
    // 33 strings → element 33 goes through consume_kv_value's VAL_STRING arm.
    let elems: Vec<Vec<u8>> = (0..33).map(|i| gguf_str_bytes(&format!("s{i}"))).collect();
    let bytes = kv_array_bytes(VAL_STRING, &elems);
    let mut c = Cursor::new(&bytes);
    let summary = c.read_kv_value_formatted(VAL_ARRAY).unwrap();
    assert!(summary.contains("…(+1 omitted)"), "got {summary}");
}

#[test]
fn cursor_consume_kv_value_covers_all_scalar_arms() {
    // One 33-element array per scalar type → element 33 exercises that type's
    // consume_kv_value arm.
    let cases: &[(u32, Vec<u8>)] = &[
        (VAL_UINT8, vec![9u8]),
        (VAL_INT8, vec![(-9i8) as u8]),
        (VAL_UINT16, 9u16.to_le_bytes().to_vec()),
        (VAL_INT16, (-9i16).to_le_bytes().to_vec()),
        (VAL_UINT32, 9u32.to_le_bytes().to_vec()),
        (VAL_INT32, (-9i32).to_le_bytes().to_vec()),
        (VAL_FLOAT32, 0.25f32.to_bits().to_le_bytes().to_vec()),
        (VAL_BOOL, vec![1u8]),
        (VAL_UINT64, 9u64.to_le_bytes().to_vec()),
        (VAL_INT64, (-9i64).to_le_bytes().to_vec()),
        (VAL_FLOAT64, 0.25f64.to_bits().to_le_bytes().to_vec()),
    ];
    for &(vt, ref elem) in cases {
        let elems = vec![elem.clone(); 33];
        let bytes = kv_array_bytes(vt, &elems);
        let mut c = Cursor::new(&bytes);
        let summary = c
            .read_kv_value_formatted(VAL_ARRAY)
            .unwrap_or_else(|e| panic!("type {vt}: {e}"));
        assert!(summary.contains("…(+1 omitted)"), "type {vt}: {summary}");
    }
}

#[test]
fn cursor_consume_kv_value_nested_array_arm() {
    // Outer array of 33 elements where each element is itself a 1-element
    // u8 array → element 33 hits consume_kv_value's VAL_ARRAY arm.
    let inner = kv_array_bytes(VAL_UINT8, &[vec![5u8]]);
    let elems = vec![inner; 33];
    let bytes = kv_array_bytes(VAL_ARRAY, &elems);
    let mut c = Cursor::new(&bytes);
    let summary = c.read_kv_value_formatted(VAL_ARRAY).unwrap();
    assert!(summary.contains("[5]"), "got {summary}");
    assert!(summary.contains("…(+1 omitted)"), "got {summary}");
}

#[test]
fn cursor_read_string_array_rejects_impossible_count() {
    // Claim 1M strings with almost no data — the pre-allocation guard must
    // reject before Vec::with_capacity can attempt a giant alloc.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&VAL_STRING.to_le_bytes());
    bytes.extend_from_slice(&(1_000_000u64).to_le_bytes());
    let mut c = Cursor::new(&bytes);
    assert!(c.read_string_array().is_err());
}

#[test]
fn cursor_read_i16_roundtrip() {
    let bytes = (-1i16).to_le_bytes();
    let mut c = Cursor::new(&bytes);
    assert_eq!(c.read_i16().unwrap(), -1);
}

// ---- Parser entry + error paths (gguf/mod.rs) ----

/// Assemble a GGUF file from raw blocks (kv entries, tensor infos, blob).
fn gguf_bytes(
    kv_block: &[u8],
    kv_count: u64,
    ti: &[u8],
    tensor_count: u64,
    blob: &[u8],
    alignment: usize,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&kv_count.to_le_bytes());
    buf.extend_from_slice(kv_block);
    buf.extend_from_slice(ti);
    let pad = align_offset(buf.len(), alignment) - buf.len();
    buf.extend(std::iter::repeat_n(0u8, pad));
    buf.extend_from_slice(blob);
    buf
}

/// One 2-element F32 tensor named "w" plus the architecture KV.
fn one_tensor_ti(name: &str, dims: &[u64], ty: u32) -> Vec<u8> {
    let mut ti = gguf_string_bytes(name);
    ti.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        ti.extend_from_slice(&d.to_le_bytes());
    }
    ti.extend_from_slice(&ty.to_le_bytes());
    ti.extend_from_slice(&0u64.to_le_bytes());
    ti
}

#[test]
fn weight_parser_trait_reads_from_disk_and_reports_format() -> Result<()> {
    let tmp = tempfile()?;
    std::fs::write(tmp.path(), minimal_f32_gguf())?;
    let parser = GgufParser;
    assert_eq!(parser.format_name(), "gguf");
    let m = parser.parse(tmp.path())?;
    assert_eq!(m.architecture, "gpt2");
    assert_eq!(m.tensors.len(), 1);
    Ok(())
}

#[test]
fn n_dims_over_128_rejected() {
    let mut ti = gguf_string_bytes("w");
    ti.extend_from_slice(&129u32.to_le_bytes());
    let buf = gguf_bytes(&encode_minimal_kv_generic(), 1, &ti, 1, &[], 32);
    let err = parse_gguf_bytes(&buf, Path::new("x.gguf")).unwrap_err();
    assert!(format!("{err:#}").contains("unrealistic n_dimensions"), "{err:#}");
}

#[test]
fn tensor_base_past_end_rejected() {
    // `general.alignment` = 1 MiB pushes the aligned tensor base past EOF.
    let mut kv = gguf_string_bytes("general.alignment");
    kv.extend_from_slice(&VAL_STRING.to_le_bytes());
    kv.extend_from_slice(&gguf_string_bytes("1048576"));
    let buf = gguf_bytes(&kv, 1, &[], 0, &[], 32);
    let err = parse_gguf_bytes(&buf, Path::new("x.gguf")).unwrap_err();
    assert!(format!("{err:#}").contains("past end of file"), "{err:#}");
}

#[test]
fn truncated_tensor_payload_rejected() {
    // ti advertises [4,4] F32 (64 bytes) but the blob holds only 8.
    let ti = one_tensor_ti("w", &[4, 4], GGML_TYPE_F32);
    let blob: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let buf = gguf_bytes(&encode_minimal_kv_generic(), 1, &ti, 1, &blob, 32);
    let err = parse_gguf_bytes(&buf, Path::new("x.gguf")).unwrap_err();
    assert!(format!("{err:#}").contains("exceeds file length"), "{err:#}");
}

#[test]
fn alignment_1_short_circuits_padding() -> Result<()> {
    let mut kv = gguf_string_bytes("general.alignment");
    kv.extend_from_slice(&VAL_STRING.to_le_bytes());
    kv.extend_from_slice(&gguf_string_bytes("1"));
    let ti = one_tensor_ti("w", &[2], GGML_TYPE_F32);
    let blob: Vec<u8> = [3.5f32, -1.25].iter().flat_map(|f| f.to_le_bytes()).collect();
    let buf = gguf_bytes(&kv, 1, &ti, 1, &blob, 1);
    let m = parse_gguf_bytes(&buf, Path::new("x.gguf"))?;
    assert_eq!(m.tensors["w"].data, blob);
    Ok(())
}

#[test]
fn dense_dtypes_map_to_ir() -> Result<()> {
    let dtypes: &[(u32, DataType, usize, usize)] = &[
        // (ggml type, IR dtype, on-disk bytes/element, IR bytes/element)
        (GGML_TYPE_F16, DataType::F16, 2, 2),
        (GGML_TYPE_BF16, DataType::BF16, 2, 2),
        (GGML_TYPE_I8, DataType::I8, 1, 1),
        (GGML_TYPE_I16, DataType::I16, 2, 2),
        (GGML_TYPE_I32, DataType::I32, 4, 4),
        (GGML_TYPE_I64, DataType::I64, 8, 8),
        (GGML_TYPE_F64, DataType::F32, 8, 4), // widened to F32 in IR
    ];
    let mut ti = Vec::new();
    let mut blob = Vec::new();
    for (i, (ggml_ty, _dtype, width, _ir_width)) in dtypes.iter().enumerate() {
        let offset = blob.len() as u64;
        ti.extend_from_slice(&gguf_string_bytes(&format!("t{i}")));
        ti.extend_from_slice(&1u32.to_le_bytes());
        ti.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
        ti.extend_from_slice(&ggml_ty.to_le_bytes());
        ti.extend_from_slice(&offset.to_le_bytes());
        blob.extend(vec![0u8; 2 * width]);
    }
    let buf = gguf_bytes(&encode_minimal_kv_generic(), 1, &ti, dtypes.len() as u64, &blob, 32);
    let m = parse_gguf_bytes(&buf, Path::new("x.gguf"))?;
    for (i, (_ggml_ty, dtype, _width, ir_width)) in dtypes.iter().enumerate() {
        let t = &m.tensors[&format!("t{i}")];
        assert_eq!(t.dtype, *dtype, "t{i} dtype");
        assert_eq!(t.data.len(), 2 * ir_width, "t{i} payload length");
    }
    Ok(())
}

#[test]
fn block_quant_nelem_not_multiple_of_32_rejected() {
    // Q4_0 with 30 elements → descriptor error naming the block size.
    let ti = one_tensor_ti("w", &[30], GGML_TYPE_Q4_0);
    let buf = gguf_bytes(&encode_minimal_kv_generic(), 1, &ti, 1, &[0u8; 18], 32);
    let err = parse_gguf_bytes(&buf, Path::new("x.gguf")).unwrap_err();
    assert!(format!("{err:#}").contains("not divisible by block size"), "{err:#}");
}

#[test]
fn zero_element_block_quant_tensor_is_accepted() -> Result<()> {
    // dims [0] → nelem 0 → block-multiple check short-circuits, empty payload.
    let ti = one_tensor_ti("w", &[0], GGML_TYPE_Q4_0);
    let buf = gguf_bytes(&encode_minimal_kv_generic(), 1, &ti, 1, &[], 32);
    let m = parse_gguf_bytes(&buf, Path::new("x.gguf"))?;
    assert_eq!(m.tensors["w"].dtype, DataType::Q4_0);
    assert_eq!(m.tensors["w"].data.len(), 0);
    Ok(())
}

#[test]
fn unsupported_type_error_names_ggml_type() {
    // ggml type 3 = Q4_1 → the error message reaches the type-hint table.
    let ti = one_tensor_ti("w", &[32], 3);
    let buf = gguf_bytes(&encode_minimal_kv_generic(), 1, &ti, 1, &[0u8; 18], 32);
    let err = parse_gguf_bytes(&buf, Path::new("x.gguf")).unwrap_err();
    assert!(
        format!("{err:#}").contains("unsupported GGML type 3 (Q4_1)"),
        "{err:#}"
    );
}

#[test]
fn extract_tokenizer_metadata_tolerates_mismatched_types() -> Result<()> {
    let mut kv = Vec::new();
    let push_kv = |kv: &mut Vec<u8>, key: &str, value_type: u32, payload: &[u8]| {
        kv.extend_from_slice(&gguf_string_bytes(key));
        kv.extend_from_slice(&value_type.to_le_bytes());
        kv.extend_from_slice(payload);
    };
    // model as UINT32 (wrong type) → skipped via the formatted read.
    push_kv(&mut kv, "tokenizer.ggml.model", VAL_UINT32, &7u32.to_le_bytes());
    // tokens / merges / bos as STRING (wrong types) → skipped.
    push_kv(
        &mut kv,
        "tokenizer.ggml.tokens",
        VAL_STRING,
        &gguf_string_bytes("not-an-array"),
    );
    push_kv(
        &mut kv,
        "tokenizer.ggml.merges",
        VAL_STRING,
        &gguf_string_bytes("nope"),
    );
    push_kv(
        &mut kv,
        "tokenizer.ggml.bos_token_id",
        VAL_STRING,
        &gguf_string_bytes("3"),
    );
    // eos as UINT32 (correct) → harvested.
    push_kv(&mut kv, "tokenizer.ggml.eos_token_id", VAL_UINT32, &5u32.to_le_bytes());
    // eos again as STRING (wrong type) → skipped.
    push_kv(
        &mut kv,
        "tokenizer.ggml.eos_token_id",
        VAL_STRING,
        &gguf_string_bytes("5"),
    );
    // chat template as STRING → harvested.
    push_kv(
        &mut kv,
        "tokenizer.chat_template",
        VAL_STRING,
        &gguf_string_bytes("{{sys}}"),
    );
    // unknown key → default arm consumes it.
    push_kv(
        &mut kv,
        "general.size_label",
        VAL_STRING,
        &gguf_string_bytes("1B"),
    );
    let buf = gguf_bytes(&kv, 8, &[], 0, &[], 32);
    let tmp = tempfile()?;
    std::fs::write(tmp.path(), &buf)?;
    let meta = extract_tokenizer_metadata(tmp.path())?;
    assert_eq!(meta.model, "");
    assert!(meta.vocab.is_empty());
    assert!(meta.merges.is_empty());
    assert_eq!(meta.bos_token_id, None);
    assert_eq!(meta.eos_token_id, Some(5));
    assert_eq!(meta.chat_template.as_deref(), Some("{{sys}}"));
    Ok(())
}

// ---------------------------------------------------------------------------
// Block-quant encoder — the write side used by `modelc pack --quantize q4_0|q8_0`
// ---------------------------------------------------------------------------

/// 64 elements with mixed magnitudes, used by the round-trip tests below.
fn blocky_values() -> Vec<f32> {
    (0..64).map(|i| (i as f32) * 0.125 - 2.0).collect()
}

#[test]
fn q4_0_encoder_matches_llama_cpp_reference_block() {
    // llama.cpp reference values: amax = 1.0 (element 0) → d = 1.0 / -8 = -0.125
    // (exactly representable in f16 → 0xB000), id = -8.
    //   element 0  ( 1.0) → x·id = -8 → trunc(-8 + 8.5) =  0        → nibble 0
    //   element 1  (-1.0) → x·id =  8 → trunc( 8 + 8.5) = 16 → MIN 15 → nibble 15
    //   elements 2+ ( 0.0) → 8.5 → trunc 8 → nibble 8
    // The high nibble of byte j holds element j + 16 (all zeros → nibble 8).
    let mut values = vec![0.0f32; 32];
    values[0] = 1.0;
    values[1] = -1.0;

    let bytes = quantize_q4_0(&values).expect("32 elements is a whole block");
    assert_eq!(bytes.len(), 18, "one Q4_0 block is 18 bytes");
    assert_eq!(&bytes[0..2], &[0x00, 0xB0], "f16 scale -0.125");
    assert_eq!(bytes[2], 0x80, "element 0 → nibble 0, element 16 → nibble 8");
    assert_eq!(
        bytes[3], 0x8F,
        "element 1 saturates to nibble 15, element 17 → nibble 8"
    );
    assert!(
        bytes[4..].iter().all(|&b| b == 0x88),
        "zero elements encode to nibble 8"
    );
}

#[test]
fn q8_0_encoder_matches_llama_cpp_reference_block() {
    // amax = 127.0 → d = 127.0 / 127 = 1.0 (f16 → 0x3C00), id = 1.0, and each
    // weight is stored as round(x / d).
    let mut values = vec![0.0f32; 32];
    values[0] = 127.0;
    values[1] = -127.0;
    values[2] = 64.0;

    let bytes = quantize_q8_0(&values).expect("32 elements is a whole block");
    assert_eq!(bytes.len(), 34, "one Q8_0 block is 34 bytes");
    assert_eq!(&bytes[0..2], &[0x00, 0x3C], "f16 scale 1.0");
    assert_eq!(bytes[2], 127u8, "127.0 → 127");
    assert_eq!(bytes[3], (-127i8) as u8, "-127.0 → -127");
    assert_eq!(bytes[4], 64, "64.0 → 64");
    assert!(bytes[5..].iter().all(|&b| b == 0), "zeros → 0");
}

#[test]
fn q4_0_round_trips_through_the_gguf_decoder() {
    // Encode → decode with the *existing* GGUF decoder: both sides must agree,
    // and the error stays within one quantization step (amax / 8).
    let values = blocky_values();
    let bytes = quantize_q4_0(&values).expect("64 elements is two whole blocks");
    assert_eq!(bytes.len(), 2 * 18);
    let decoded = dequantize_q4_0(&bytes, values.len()).expect("decode");
    let amax = values.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    for (got, want) in decoded.iter().zip(values.iter()) {
        assert!(
            (got - want).abs() <= amax / 8.0 + 1e-6,
            "Q4_0 error {got} vs {want} exceeds one step (amax/8 = {})",
            amax / 8.0
        );
    }
}

#[test]
fn q8_0_round_trips_through_the_gguf_decoder() {
    // Q8_0 keeps 8 bits per weight — its step (amax / 127) is far finer than Q4_0's.
    let values = blocky_values();
    let bytes = quantize_q8_0(&values).expect("64 elements is two whole blocks");
    assert_eq!(bytes.len(), 2 * 34);
    let decoded = dequantize_q8_0(&bytes, values.len()).expect("decode");
    let amax = values.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    for (got, want) in decoded.iter().zip(values.iter()) {
        assert!(
            (got - want).abs() <= amax / 127.0 + 1e-3,
            "Q8_0 error {got} vs {want} exceeds one step"
        );
    }
}

#[test]
fn block_quantizers_reject_partial_blocks() {
    // Not a whole number of 32-element blocks → the caller keeps the tensor F32.
    assert!(quantize_q4_0(&[0.0f32; 33]).is_none());
    assert!(quantize_q8_0(&[0.0f32; 33]).is_none());
    // Whole blocks are fine; an empty tensor encodes to no bytes.
    assert!(quantize_q4_0(&[0.0f32; 64]).is_some());
    assert_eq!(quantize_q4_0(&[]).expect("empty tensor").len(), 0);
}

#[test]
fn q4_0_zero_block_round_trips_to_zero() {
    // An all-zero block has no scale (d = -0.0) and every weight decodes to 0.
    let values = vec![0.0f32; 32];
    let bytes = quantize_q4_0(&values).expect("whole block");
    let decoded = dequantize_q4_0(&bytes, 32).expect("decode");
    assert!(decoded.iter().all(|&v| v == 0.0), "zeros stay zeros");
}

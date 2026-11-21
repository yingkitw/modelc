
use half::f16;

use super::cursor::Cursor;
use super::*;
use crate::tempdir::tempfile;

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

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};

use onnx_rs::ast::{DataType as OnnxDt, TensorProto};

use crate::model::DataType;

pub(super) fn onnx_initializer_bytes(t: &TensorProto<'_>, elem: DataType) -> Result<Vec<u8>> {
    if let Some(raw) = t.as_raw()
        && !raw.is_empty()
    {
        return Ok(raw.to_vec());
    }

    Ok(match elem {
        DataType::F32 => {
            if let Some(fs) = t.as_f32() {
                fs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()
            } else if let Some(ds) = t.as_f64() {
                ds.iter()
                    .flat_map(|x| (*x as f32).to_le_bytes())
                    .collect::<Vec<u8>>()
            } else {
                anyhow::bail!("{:?} missing float payload", t.name());
            }
        }
        DataType::I32 => {
            let Some(ix) = t.as_i32() else {
                anyhow::bail!("{:?} missing int32 payload", t.name());
            };
            ix.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()
        }
        DataType::I64 => {
            let Some(ix) = t.as_i64() else {
                anyhow::bail!("{:?} missing int64 payload", t.name());
            };
            ix.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()
        }
        DataType::F16
        | DataType::BF16
        | DataType::U8
        | DataType::I8
        | DataType::I16
        | DataType::Bool
        | DataType::Q4_0
        | DataType::Q5_0
        | DataType::Q8_0
        | DataType::Q4_K
        | DataType::Q6_K => {
            anyhow::bail!(
                "{:?} missing raw_data for packed dtype {:?}",
                t.name(),
                elem,
            );
        }
    })
}

pub(super) fn onnx_tensor_storage_width(dt: OnnxDt) -> Result<usize> {
    Ok(match dt {
        OnnxDt::Float => 4,
        OnnxDt::Double => 8,
        OnnxDt::Float16 | OnnxDt::Bfloat16 => 2,
        OnnxDt::Int32 => 4,
        OnnxDt::Int64 => 8,
        OnnxDt::Uint8 => 1,
        OnnxDt::Int8 => 1,
        OnnxDt::Uint16 | OnnxDt::Int16 => 2,
        OnnxDt::Bool => 1,
        dt => anyhow::bail!("ONNX dtype {dt:?} cannot be loaded from external initializer bytes",),
    })
}

pub(super) fn load_onnx_external_bytes(
    model_path: &Path,
    init: &TensorProto<'_>,
) -> Result<Vec<u8>> {
    let mut location: Option<&str> = None;
    let mut offset: u64 = 0;
    let mut length: Option<u64> = None;

    for e in init.external_data() {
        match e.key {
            "location" => location = Some(e.value),
            "offset" => {
                let v: i64 = e.value.parse().with_context(|| {
                    format!(
                        "initializer {:?}: external_data `offset` invalid integer",
                        init.name(),
                    )
                })?;
                ensure!(
                    v >= 0,
                    "initializer {:?}: negative external offset",
                    init.name(),
                );
                offset = u64::try_from(v).context(format!(
                    "initializer {:?}: external offset overflow",
                    init.name(),
                ))?;
            }
            "length" => {
                let v: i64 = e.value.parse().with_context(|| {
                    format!(
                        "initializer {:?}: external_data `length` invalid integer",
                        init.name(),
                    )
                })?;
                ensure!(
                    v >= 0,
                    "initializer {:?}: negative external length",
                    init.name(),
                );
                length = Some(u64::try_from(v).context(format!(
                    "initializer {:?}: external length overflow",
                    init.name(),
                ))?);
            }
            _ => {}
        }
    }

    let loc = location.filter(|s| !s.is_empty()).ok_or_else(|| {
        anyhow!(
            "initializer {:?}: external_data missing non-empty `location`",
            init.name(),
        )
    })?;

    let base = model_path.parent().unwrap_or_else(|| Path::new("."));
    let ext_path = base.join(loc);

    let mut file = File::open(&ext_path).with_context(|| format!("opening {ext_path:?}"))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("stat {ext_path:?}"))?
        .len();

    let nbytes = if let Some(len) = length {
        len
    } else {
        file_len.checked_sub(offset).ok_or_else(|| {
            anyhow!(
                "initializer {:?}: external offset {} at or beyond file length {}",
                init.name(),
                offset,
                file_len,
            )
        })?
    };

    let span_end = offset
        .checked_add(nbytes)
        .ok_or_else(|| anyhow!("initializer {:?}: external byte span overflow", init.name(),))?;
    ensure!(
        span_end <= file_len,
        "initializer {:?}: external slice [{}..{}) exceeds file length {}",
        init.name(),
        offset,
        span_end,
        file_len,
    );

    let n_usize = usize::try_from(nbytes)
        .map_err(|_| anyhow!("initializer {:?}: span too large", init.name()))?;

    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; n_usize];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

pub(super) fn onnx_external_bytes_to_ir(
    disk: &[u8],
    onnx_ty: OnnxDt,
    nelem: usize,
    tensor_name: &str,
) -> Result<Vec<u8>> {
    let elem = super::onnx_elem_to_model(onnx_ty)?;
    let w = onnx_tensor_storage_width(onnx_ty)?;
    let expected = nelem
        .checked_mul(w)
        .with_context(|| format!("{tensor_name}: external tensor payload size overflow"))?;

    ensure!(
        disk.len() == expected,
        "{tensor_name}: external ONNX raw length {} but expected {expected} ({nelem} elems × {w} bytes — {onnx_ty:?})",
        disk.len(),
    );

    let out = match onnx_ty {
        OnnxDt::Float => disk.to_vec(),
        OnnxDt::Double => disk
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<u8>>(),
        OnnxDt::Float16
        | OnnxDt::Bfloat16
        | OnnxDt::Int32
        | OnnxDt::Int64
        | OnnxDt::Uint8
        | OnnxDt::Int8
        | OnnxDt::Uint16
        | OnnxDt::Int16
        | OnnxDt::Bool => disk.to_vec(),
        _ => anyhow::bail!(
            "{tensor_name}: ONNX dtype {onnx_ty:?} unsupported for external raw tensor decoding",
        ),
    };

    ensure!(
        out.len() == elem.byte_size().saturating_mul(nelem),
        "{tensor_name}: decoded payload length mismatches IR dtype {elem:?}",
    );
    Ok(out)
}

pub(super) fn onnx_slice_bytes_to_ir(
    slice: &[u8],
    elem: DataType,
    nelem: usize,
    tensor_name: &str,
) -> Result<Vec<u8>> {
    let expected = elem.byte_size().saturating_mul(nelem);
    ensure!(
        slice.len() == expected,
        "{tensor_name}: segmented slice length {} but expected {expected} ({nelem} elems × {:?})",
        slice.len(),
        elem,
    );
    Ok(slice.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_rs::ast::TensorProto;

    // ---- onnx_initializer_bytes ----

    #[test]
    fn raw_data_passthrough_when_present() {
        // When raw_data is present, it is returned verbatim regardless of the
        // requested element type.
        let raw = [42u8, 0, 0, 0, 43, 0, 0, 0];
        let t = TensorProto::from_raw("w", vec![2], OnnxDt::Float, &raw);
        let out = onnx_initializer_bytes(&t, DataType::F32).unwrap();
        assert_eq!(out, raw.to_vec());
    }

    #[test]
    fn f32_payload_serializes_to_le_bytes() {
        let t = TensorProto::from_f32("w", vec![2], vec![1.0f32, -2.0]);
        let out = onnx_initializer_bytes(&t, DataType::F32).unwrap();
        let v: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(v, vec![1.0, -2.0]);
    }

    #[test]
    fn f64_payload_downcasts_to_f32_when_f32_requested() {
        let t = TensorProto::from_f64("w", vec![2], vec![3.5f64, -0.25]);
        let out = onnx_initializer_bytes(&t, DataType::F32).unwrap();
        let v: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(v, vec![3.5f32, -0.25]);
    }

    #[test]
    fn f32_requested_but_no_payload_errors() {
        let t = TensorProto::from_raw("empty", vec![1], OnnxDt::Float, &[]);
        let err = onnx_initializer_bytes(&t, DataType::F32).unwrap_err();
        assert!(format!("{err:#}").contains("missing float payload"));
    }

    #[test]
    fn i32_and_i64_payloads_serialize() {
        let t32 = TensorProto::from_i32("w", vec![2], vec![7i32, -8]);
        let out32 = onnx_initializer_bytes(&t32, DataType::I32).unwrap();
        let v32: Vec<i32> = out32
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(v32, vec![7, -8]);

        let t64 = TensorProto::from_i64("w", vec![2], vec![1i64 << 40, -1]);
        let out64 = onnx_initializer_bytes(&t64, DataType::I64).unwrap();
        let v64: Vec<i64> = out64
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(v64, vec![1i64 << 40, -1]);
    }

    #[test]
    fn missing_int_payloads_error() {
        let t = TensorProto::from_raw("i", vec![1], OnnxDt::Int32, &[]);
        assert!(
            onnx_initializer_bytes(&t, DataType::I32)
                .unwrap_err()
                .to_string()
                .contains("missing int32")
        );

        let t64 = TensorProto::from_raw("i", vec![1], OnnxDt::Int64, &[]);
        assert!(
            onnx_initializer_bytes(&t64, DataType::I64)
                .unwrap_err()
                .to_string()
                .contains("missing int64")
        );
    }

    #[test]
    fn packed_dtypes_without_raw_error() {
        let t = TensorProto::from_raw("p", vec![1], OnnxDt::Float16, &[]);
        let err = onnx_initializer_bytes(&t, DataType::F16).unwrap_err();
        assert!(format!("{err:#}").contains("missing raw_data for packed dtype"));
    }

    // ---- onnx_tensor_storage_width ----

    #[test]
    fn storage_widths_match_onnx_spec() {
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Float).unwrap(), 4);
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Double).unwrap(), 8);
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Float16).unwrap(), 2);
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Bfloat16).unwrap(), 2);
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Int32).unwrap(), 4);
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Int64).unwrap(), 8);
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Uint8).unwrap(), 1);
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Int8).unwrap(), 1);
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Uint16).unwrap(), 2);
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Int16).unwrap(), 2);
        assert_eq!(onnx_tensor_storage_width(OnnxDt::Bool).unwrap(), 1);
        // Unsupported dtype errors with a named hint.
        let err = onnx_tensor_storage_width(OnnxDt::Uint64).unwrap_err();
        assert!(format!("{err:#}").contains("cannot be loaded from external initializer"));
    }

    // ---- onnx_external_bytes_to_ir ----

    #[test]
    fn external_float_passthrough_and_double_downcast() {
        let floats = [1.0f32, 2.0];
        let mut bytes = Vec::new();
        for f in floats {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        let out = onnx_external_bytes_to_ir(&bytes, OnnxDt::Float, 2, "w").unwrap();
        assert_eq!(out, bytes);

        let doubles = [1.5f64, -2.5];
        let mut dbytes = Vec::new();
        for d in doubles {
            dbytes.extend_from_slice(&d.to_le_bytes());
        }
        let out = onnx_external_bytes_to_ir(&dbytes, OnnxDt::Double, 2, "w").unwrap();
        let v: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(v, vec![1.5f32, -2.5]);

        // Int64 passthrough (IR keeps raw bytes; runtime casts).
        let mut ibytes = Vec::new();
        for i in [5i64, -6] {
            ibytes.extend_from_slice(&i.to_le_bytes());
        }
        let out = onnx_external_bytes_to_ir(&ibytes, OnnxDt::Int64, 2, "w").unwrap();
        assert_eq!(out, ibytes);
    }

    #[test]
    fn external_length_mismatch_errors() {
        let bytes = [0u8; 4];
        let err = onnx_external_bytes_to_ir(&bytes, OnnxDt::Float, 2, "w").unwrap_err();
        assert!(format!("{err:#}").contains("external ONNX raw length"));
    }

    #[test]
    fn external_unsupported_dtype_errors() {
        // Uint64 is unsupported for external decoding (width check passes at 8
        // only for Int64; Uint64 falls into the bail arm).
        let bytes = [0u8; 8];
        let err = onnx_external_bytes_to_ir(&bytes, OnnxDt::Uint64, 1, "w");
        assert!(err.is_err());
    }

    // ---- onnx_slice_bytes_to_ir ----

    #[test]
    fn slice_bytes_roundtrip_and_mismatch() {
        let bytes = [1u8, 2, 3, 4];
        let out = onnx_slice_bytes_to_ir(&bytes, DataType::F32, 1, "w").unwrap();
        assert_eq!(out, bytes.to_vec());

        let err = onnx_slice_bytes_to_ir(&bytes, DataType::F32, 2, "w").unwrap_err();
        assert!(format!("{err:#}").contains("segmented slice length"));
    }
}

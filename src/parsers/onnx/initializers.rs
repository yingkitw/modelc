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

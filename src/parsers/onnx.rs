use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};

use onnx_rs::ast::{DataLocation, DataType as OnnxDt, TensorProto};

use crate::model::{DataType, Model, TensorData};
use crate::parsers::WeightParser;

pub struct OnnxParser;

impl WeightParser for OnnxParser {
    fn parse(&self, path: &Path) -> Result<Model> {
        let data = std::fs::read(path).with_context(|| format!("failed to read {:?}", path))?;
        parse_onnx_bytes(&data, path)
    }

    fn format_name(&self) -> &'static str {
        "onnx"
    }
}

fn parse_onnx_bytes(bytes: &[u8], path: &Path) -> Result<Model> {
    let m = onnx_rs::parse(bytes).map_err(|e| anyhow!("{e}"))?;
    let graph = m.graph.as_ref().context("ONNX ModelProto missing Graph")?;

    let mut tensors = HashMap::new();
    for init in &graph.initializer {
        if init.segment().is_some() {
            anyhow::bail!(
                "initializer {:?} is segmented; flatten before import",
                init.name()
            );
        }

        let name = init.name().to_string();
        let onnx_ty = init.data_type();
        let elem = onnx_elem_to_model(onnx_ty)
            .with_context(|| format!("{name}: unsupported ONNX element type {:?}", onnx_ty,))?;

        let shape_dims = init.dims();
        let mut shape_usize = Vec::with_capacity(shape_dims.len());
        for d in shape_dims {
            ensure!(*d >= 0, "{name}: unexpected negative dim {d}");
            shape_usize.push(usize::try_from(*d).context("initializer dim overflow")?);
        }

        let nelem_expected = elem.element_count(&shape_usize);

        let blob = if let Some(raw) = init.as_raw()
            && !raw.is_empty()
        {
            onnx_initializer_bytes(init, elem)
                .with_context(|| format!("initializer {name} payload"))?
        } else if !init.external_data().is_empty() {
            let disk = load_onnx_external_bytes(path, init)
                .with_context(|| format!("initializer {name} external_data"))?;
            onnx_external_bytes_to_ir(&disk, onnx_ty, nelem_expected, init.name())?
        } else if init.data_location() == DataLocation::External {
            anyhow::bail!(
                "initializer {name:?}: data_location is EXTERNAL but raw_data is empty and external_data has no `location` entry",
            );
        } else {
            onnx_initializer_bytes(init, elem)
                .with_context(|| format!("initializer {name} payload"))?
        };

        ensure!(
            blob.len() == elem.byte_size().saturating_mul(nelem_expected),
            "{name}: byte mismatch (expected {}, got {})",
            elem.byte_size().saturating_mul(nelem_expected),
            blob.len(),
        );

        tensors.insert(
            name,
            TensorData {
                shape: shape_usize,
                dtype: elem,
                data: blob,
            },
        );
    }

    if tensors.is_empty() {
        anyhow::bail!("no embedded ONNX initializers; export with inlined weights");
    }

    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "onnx".to_string());
    if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
        metadata.insert("source".to_string(), fname.to_string());
    }
    metadata.insert(
        "onnx.producer_name".to_string(),
        m.producer_name.to_string(),
    );
    metadata.insert(
        "onnx.producer_version".to_string(),
        m.producer_version.to_string(),
    );

    Ok(Model {
        name: path.file_stem().unwrap().to_string_lossy().to_string(),
        architecture: "onnx_initializer_dump".to_string(),
        tensors,
        metadata,
    })
}

fn onnx_elem_to_model(dt: OnnxDt) -> Result<DataType> {
    Ok(match dt {
        OnnxDt::Float => DataType::F32,
        OnnxDt::Float16 => DataType::F16,
        OnnxDt::Bfloat16 => DataType::BF16,
        OnnxDt::Int32 => DataType::I32,
        OnnxDt::Int64 => DataType::I64,
        OnnxDt::Uint8 => DataType::U8,
        OnnxDt::Int8 => DataType::I8,
        OnnxDt::Int16 => DataType::I16,
        OnnxDt::Uint16 => DataType::I16,
        OnnxDt::Bool => DataType::Bool,
        OnnxDt::Double => DataType::F32,
        OnnxDt::Uint32
        | OnnxDt::Uint64
        | OnnxDt::String
        | OnnxDt::Undefined
        | OnnxDt::Complex64
        | OnnxDt::Complex128
        | OnnxDt::Float8e4m3fn
        | OnnxDt::Float8e4m3fnuz
        | OnnxDt::Float8e5m2
        | OnnxDt::Float8e5m2fnuz
        | OnnxDt::Uint4
        | OnnxDt::Int4
        | OnnxDt::Float4e2m1 => {
            anyhow::bail!("ONNX dtype {dt:?} not supported for initializer import");
        }
    })
}

fn onnx_initializer_bytes(t: &TensorProto<'_>, elem: DataType) -> Result<Vec<u8>> {
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
        | DataType::Bool => {
            anyhow::bail!(
                "{:?} missing raw_data for packed dtype {:?}",
                t.name(),
                elem,
            );
        }
    })
}

fn onnx_tensor_storage_width(dt: OnnxDt) -> Result<usize> {
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

fn load_onnx_external_bytes(model_path: &Path, init: &TensorProto<'_>) -> Result<Vec<u8>> {
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

fn onnx_external_bytes_to_ir(
    disk: &[u8],
    onnx_ty: OnnxDt,
    nelem: usize,
    tensor_name: &str,
) -> Result<Vec<u8>> {
    let elem = onnx_elem_to_model(onnx_ty)?;
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

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use tempfile::tempdir;

    use onnx_rs::ast::{
        DataLocation, DataType as Odt, Graph, Model, OperatorSetId, StringStringEntry, TensorProto,
    };
    use onnx_rs::encode;

    use crate::parsers::{WeightParser, onnx::OnnxParser};

    #[test]
    fn external_float_initializer_roundtrip() -> Result<()> {
        let dir = tempdir()?;
        let onnx_path = dir.path().join("m.onnx");
        let w = [1.0f32, -2.0f32, 3.5f32];
        let raw: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(dir.path().join("weights.bin"), &raw)?;

        let tensor = TensorProto::from_raw("w", vec![3], Odt::Float, &[])
            .with_data_location(DataLocation::External)
            .with_external_data(vec![StringStringEntry {
                key: "location",
                value: "weights.bin",
            }]);

        let model = Model {
            ir_version: 9,
            opset_import: vec![OperatorSetId {
                domain: "",
                version: 19,
            }],
            producer_name: "test",
            producer_version: "0",
            graph: Some(Graph {
                name: "g",
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };

        std::fs::write(&onnx_path, encode(&model))?;

        let m = OnnxParser.parse(&onnx_path)?;
        let t = m.tensors.get("w").context("missing w tensor")?;
        assert_eq!(t.shape, vec![3usize]);
        assert_eq!(t.data, raw);
        Ok(())
    }

    #[test]
    fn external_initializer_respects_offset() -> Result<()> {
        let dir = tempdir()?;
        let onnx_path = dir.path().join("m.onnx");
        let mut padded = vec![0u8; 16];
        padded.extend_from_slice(&5.0f32.to_le_bytes());
        padded.extend_from_slice(&[0xaa, 0xbb]); // trailing noise
        std::fs::write(dir.path().join("blob.bin"), &padded)?;

        let tensor = TensorProto::from_raw("b", vec![1], Odt::Float, &[])
            .with_data_location(DataLocation::External)
            .with_external_data(vec![
                StringStringEntry {
                    key: "location",
                    value: "blob.bin",
                },
                StringStringEntry {
                    key: "offset",
                    value: "16",
                },
                StringStringEntry {
                    key: "length",
                    value: "4",
                },
            ]);

        let model = Model {
            ir_version: 9,
            opset_import: vec![OperatorSetId {
                domain: "",
                version: 19,
            }],
            producer_name: "test",
            producer_version: "0",
            graph: Some(Graph {
                name: "g",
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };

        std::fs::write(&onnx_path, encode(&model))?;

        let m = OnnxParser.parse(&onnx_path)?;
        let t = m.tensors["b"].data.clone();
        assert_eq!(t, 5.0f32.to_le_bytes().to_vec());
        Ok(())
    }

    #[test]
    fn external_tensor_without_location_errors() {
        let dir = tempdir().unwrap();
        let onnx_path = dir.path().join("m.onnx");
        let tensor = TensorProto::from_raw("x", vec![1], Odt::Float, &[])
            .with_data_location(DataLocation::External);

        let model = Model {
            ir_version: 9,
            opset_import: vec![OperatorSetId {
                domain: "",
                version: 19,
            }],
            producer_name: "test",
            producer_version: "0",
            graph: Some(Graph {
                name: "g",
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };

        std::fs::write(&onnx_path, encode(&model)).unwrap();
        let err = OnnxParser.parse(&onnx_path).unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("EXTERNAL") || s.contains("location"), "{s}");
    }

    #[test]
    fn external_tensor_missing_file_errors() {
        let dir = tempdir().unwrap();
        let onnx_path = dir.path().join("m.onnx");

        let tensor = TensorProto::from_raw("x", vec![1], Odt::Float, &[])
            .with_data_location(DataLocation::External)
            .with_external_data(vec![StringStringEntry {
                key: "location",
                value: "does_not_exist.bin",
            }]);

        let model = Model {
            ir_version: 9,
            opset_import: vec![OperatorSetId {
                domain: "",
                version: 19,
            }],
            producer_name: "test",
            producer_version: "0",
            graph: Some(Graph {
                name: "g",
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };

        std::fs::write(&onnx_path, encode(&model)).unwrap();
        let err = OnnxParser.parse(&onnx_path).unwrap_err();
        assert!(format!("{err:#}").contains("does_not_exist"), "{err:#}");
    }

    #[test]
    fn external_slice_not_within_file_errors() {
        let dir = tempdir().unwrap();
        let onnx_path = dir.path().join("m.onnx");
        std::fs::write(dir.path().join("tiny.bin"), [0u8; 4]).unwrap();

        let tensor = TensorProto::from_raw("x", vec![1], Odt::Float, &[])
            .with_data_location(DataLocation::External)
            .with_external_data(vec![
                StringStringEntry {
                    key: "location",
                    value: "tiny.bin",
                },
                StringStringEntry {
                    key: "offset",
                    value: "4",
                },
                StringStringEntry {
                    key: "length",
                    value: "4",
                },
            ]);

        let model = Model {
            ir_version: 9,
            opset_import: vec![OperatorSetId {
                domain: "",
                version: 19,
            }],
            producer_name: "test",
            producer_version: "0",
            graph: Some(Graph {
                name: "g",
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };

        std::fs::write(&onnx_path, encode(&model)).unwrap();
        assert!(OnnxParser.parse(&onnx_path).is_err());
    }

    #[test]
    fn external_double_initializer_converts_to_f32() -> Result<()> {
        let dir = tempdir()?;
        let onnx_path = dir.path().join("m.onnx");
        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(&std::f64::consts::PI.to_le_bytes());
        raw.extend_from_slice(&(-1.5f64).to_le_bytes());
        std::fs::write(dir.path().join("d.bin"), &raw)?;

        let tensor = TensorProto::from_raw("d", vec![2], Odt::Double, &[])
            .with_data_location(DataLocation::External)
            .with_external_data(vec![StringStringEntry {
                key: "location",
                value: "d.bin",
            }]);

        let model = Model {
            ir_version: 9,
            opset_import: vec![OperatorSetId {
                domain: "",
                version: 19,
            }],
            producer_name: "test",
            producer_version: "0",
            graph: Some(Graph {
                name: "g",
                initializer: vec![tensor],
                ..Default::default()
            }),
            ..Default::default()
        };

        std::fs::write(&onnx_path, encode(&model))?;

        let m = OnnxParser.parse(&onnx_path)?;
        let t = m.tensors["d"].clone();
        assert_eq!(t.shape, vec![2usize]);
        let a = f32::from_le_bytes(std::convert::TryInto::try_into(&t.data[0..4]).unwrap());
        let b = f32::from_le_bytes(std::convert::TryInto::try_into(&t.data[4..8]).unwrap());
        assert!((a - std::f64::consts::PI as f32).abs() < 1e-5);
        assert!((b - (-1.5f32)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn no_initializers_errors() -> Result<()> {
        let dir = tempdir()?;
        let onnx_path = dir.path().join("empty.onnx");
        let model = Model {
            ir_version: 9,
            opset_import: vec![OperatorSetId {
                domain: "",
                version: 19,
            }],
            producer_name: "test",
            producer_version: "0",
            graph: Some(Graph {
                name: "g",
                ..Default::default()
            }),
            ..Default::default()
        };

        std::fs::write(&onnx_path, encode(&model))?;
        assert!(OnnxParser.parse(&onnx_path).is_err());
        Ok(())
    }
}

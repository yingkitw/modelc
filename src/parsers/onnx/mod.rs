use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};

use onnx_rs::ast::{DataLocation, DataType as OnnxDt};

use crate::model::{DataType, Model, TensorData};
use crate::parsers::WeightParser;

mod initializers;

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
            if let Some(seg) = init.segment() {
                let begin = usize::try_from(seg.begin)
                    .with_context(|| format!("{name}: segment begin overflow"))?;
                let end = usize::try_from(seg.end)
                    .with_context(|| format!("{name}: segment end overflow"))?;
                ensure!(
                    end <= raw.len(),
                    "{name}: segment [{begin}, {end}) exceeds raw_data length {}",
                    raw.len()
                );
                let slice = &raw[begin..end];
                initializers::onnx_slice_bytes_to_ir(slice, elem, nelem_expected, &name)?
            } else {
                initializers::onnx_initializer_bytes(init, elem)
                    .with_context(|| format!("initializer {name} payload"))?
            }
        } else if !init.external_data().is_empty() {
            let disk = initializers::load_onnx_external_bytes(path, init)
                .with_context(|| format!("initializer {name} external_data"))?;
            initializers::onnx_external_bytes_to_ir(&disk, onnx_ty, nelem_expected, init.name())?
        } else if init.data_location() == DataLocation::External {
            anyhow::bail!(
                "initializer {name:?}: data_location is EXTERNAL but raw_data is empty and external_data has no `location` entry",
            );
        } else {
            initializers::onnx_initializer_bytes(init, elem)
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

    // Build execution plan from graph nodes if any exist.
    if !graph.node.is_empty() {
        match crate::onnx_exec::build_plan(graph) {
            Ok(plan) => {
                if let Ok(json) = plan.to_json() {
                    metadata.insert("onnx.execution_plan".to_string(), json);
                }
            }
            Err(e) => {
                eprintln!("  warning: failed to build ONNX execution plan: {}", e);
            }
        }
    }

    Ok(Model {
        name: path.file_stem().unwrap().to_string_lossy().to_string(),
        architecture: "onnx".to_string(),
        tensors,
        metadata,
    })
}

pub(super) fn onnx_elem_to_model(dt: OnnxDt) -> Result<DataType> {
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
    fn segmented_initializer_roundtrip() -> Result<()> {
        let dir = tempdir()?;
        let onnx_path = dir.path().join("m.onnx");

        // Full raw_data: 6 floats = 24 bytes. Take segment [8, 16) = 2 floats.
        let full_raw: Vec<u8> = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tensor = TensorProto::from_raw("w", vec![2], Odt::Float, &full_raw)
            .with_segment(onnx_rs::ast::TensorSegment { begin: 8, end: 16 });

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
        let t = &m.tensors["w"];
        assert_eq!(t.shape, vec![2usize]);
        let vals: Vec<f32> = t
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![3.0, 4.0]);
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

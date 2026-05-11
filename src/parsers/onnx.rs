use std::collections::HashMap;
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
        if init.data_location() == DataLocation::External || !init.external_data().is_empty() {
            anyhow::bail!(
                "initializer {:?} uses external data; merge weights into the .onnx file first",
                init.name(),
            );
        }
        if init.segment().is_some() {
            anyhow::bail!(
                "initializer {:?} is segmented; flatten before import",
                init.name()
            );
        }

        let name = init.name().to_string();
        let elem = onnx_elem_to_model(init.data_type()).with_context(|| {
            format!(
                "{name}: unsupported ONNX element type {:?}",
                init.data_type(),
            )
        })?;

        let shape_dims = init.dims();
        let mut shape_usize = Vec::with_capacity(shape_dims.len());
        for d in shape_dims {
            ensure!(*d >= 0, "{name}: unexpected negative dim {d}");
            shape_usize.push(usize::try_from(*d).context("initializer dim overflow")?);
        }

        let nelem_expected = elem.element_count(&shape_usize);
        let blob = onnx_initializer_bytes(init, elem)
            .with_context(|| format!("initializer {name} payload"))?;

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

//! PyTorch checkpoints are pickled (often inside a ZIP). Practical Rust support without an embedded Python
//! interpreter is limited, so we accept:
//! - **Mislabeled plain Safetensors** files carrying a `.pt` / `.pth` suffix.
//! - **ZIP-backed archives** (`PK\x03\x04`) containing one or more `*.safetensors` payloads.

use std::collections::HashMap;
use std::io::Cursor;
use std::io::Read as _;
use std::path::Path;

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use zip::ZipArchive;

use crate::model::{DataType, Model, TensorData};
use crate::parsers::WeightParser;
use crate::parsers::safetensors::SafetensorsParser;

pub struct PytorchParser;

impl WeightParser for PytorchParser {
    fn parse(&self, path: &Path) -> Result<Model> {
        pytorch_parse(path).with_context(|| {
            format!(
                "{:?}: expected plain Safetensors bytes or ZIP with embedded *.safetensors tensors.",
                path
            )
        })
    }

    fn format_name(&self) -> &'static str {
        "pytorch"
    }
}

fn pytorch_parse(path: &Path) -> Result<Model> {
    let raw = std::fs::read(path).with_context(|| format!("failed to read {:?}", path))?;

    let looks_like_st = SafeTensors::deserialize(&raw)
        .map(|st| !st.tensors().is_empty())
        .unwrap_or(false);
    if looks_like_st {
        return SafetensorsParser.parse(path);
    }

    if raw.starts_with(b"PK\x03\x04") {
        return parse_torch_zip(path, &raw);
    }

    anyhow::bail!(
        "not ZIP and not Safetensors; export via Safetensors, GGUF, or ONNX bundles first",
    );
}

fn parse_torch_zip(path: &Path, raw: &[u8]) -> Result<Model> {
    let cursor = Cursor::new(raw);
    let mut archive = ZipArchive::new(cursor).context("unable to parse ZIP archive")?;

    let mut payloads: Vec<usize> = Vec::new();

    for idx in 0..archive.len() {
        let Ok(entry) = archive.by_index(idx) else {
            continue;
        };

        let name_owned = entry.name().to_owned();
        if entry.is_dir() || name_owned.ends_with('/') {
            continue;
        }
        if nested_safetensors_candidate(&name_owned) {
            payloads.push(idx);
        }
    }

    if payloads.is_empty() {
        anyhow::bail!("no nested *.safetensors tensors discovered inside ZIP");
    }

    let prefix_entries = payloads.len() > 1;
    let mut tensors: HashMap<String, TensorData> = HashMap::new();
    let mut sources = Vec::new();

    for idx in payloads {
        let mut entry = archive
            .by_index(idx)
            .with_context(|| format!("ZIP entry #{idx} missing"))?;

        let name_owned = entry.name().to_owned();
        let mut blob = Vec::new();
        entry
            .read_to_end(&mut blob)
            .with_context(|| format!("failed reading zipped {name_owned}"))?;

        let st = SafeTensors::deserialize(&blob)
            .with_context(|| format!("{name_owned}: not Safetensors"))?;

        sources.push(name_owned.clone());
        let stem = sanitize_stem(&name_owned);

        for (tensor_name, view) in st.tensors() {
            let dtype = safetensors_dtype(view.dtype())
                .with_context(|| format!("unsupported dtype @{name_owned}:{tensor_name}"))?;

            let fq = if prefix_entries {
                format!("{stem}.{tensor_name}")
            } else {
                tensor_name
            };

            let shape: Vec<usize> = view.shape().to_vec();
            let src = view.data();
            let mut owned = vec![0u8; src.len()];
            owned.copy_from_slice(src);

            tensors.insert(
                fq,
                TensorData {
                    shape,
                    dtype,
                    data: owned,
                },
            );
        }
    }

    Ok(Model {
        name: path.file_stem().unwrap().to_string_lossy().into_owned(),
        architecture: "pytorch_zip_bundle".into(),
        tensors,
        metadata: HashMap::from([
            ("format".into(), "pytorch_zip_safe".into()),
            ("pytorch.sources".into(), sources.join(", ")),
        ]),
    })
}

fn nested_safetensors_candidate(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".safetensors") || lower.ends_with(".st") || lower.contains(".safetensors.")
}

fn sanitize_stem(zip_path: &str) -> String {
    Path::new(zip_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(zip_path)
        .chars()
        .map(|ch| match ch {
            '.' | '/' | '\\' => '_',
            c => c,
        })
        .collect()
}

fn safetensors_dtype(dt: safetensors::Dtype) -> Result<DataType> {
    Ok(match dt {
        safetensors::Dtype::F32 => DataType::F32,
        safetensors::Dtype::F16 => DataType::F16,
        safetensors::Dtype::BF16 => DataType::BF16,
        safetensors::Dtype::I64 => DataType::I64,
        safetensors::Dtype::I32 => DataType::I32,
        safetensors::Dtype::I16 => DataType::I16,
        safetensors::Dtype::I8 => DataType::I8,
        safetensors::Dtype::U8 => DataType::U8,
        safetensors::Dtype::BOOL => DataType::Bool,
        other => anyhow::bail!("dtype {other:?} not mapped"),
    })
}

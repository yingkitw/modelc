use std::path::Path;

use anyhow::Result;

use crate::model::Model;
use crate::parsers::WeightParser;

pub struct OnnxParser;

impl WeightParser for OnnxParser {
    fn parse(&self, _path: &Path) -> Result<Model> {
        anyhow::bail!(
            "ONNX parsing is not implemented here (spec: \
             https://onnx.ai/onnx/intro/). \
             Export to safetensors or use the onnx format with a future parser."
        )
    }

    fn format_name(&self) -> &'static str {
        "onnx"
    }
}

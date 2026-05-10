use std::path::Path;

use anyhow::Result;

use crate::model::Model;
use crate::parsers::WeightParser;

pub struct OnnxParser;

impl WeightParser for OnnxParser {
    fn parse(&self, _path: &Path) -> Result<Model> {
        anyhow::bail!(
            "ONNX parsing is not yet implemented. \
             Contribute or use safetensors format for now."
        )
    }

    fn format_name(&self) -> &'static str {
        "onnx"
    }
}

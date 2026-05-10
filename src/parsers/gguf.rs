use std::path::Path;

use anyhow::Result;

use crate::model::Model;
use crate::parsers::WeightParser;

pub struct GgufParser;

impl WeightParser for GgufParser {
    fn parse(&self, _path: &Path) -> Result<Model> {
        anyhow::bail!(
            "GGUF parsing is not implemented in this tree (format: \
             https://github.com/ggerganov/ggml/blob/master/docs/gguf.md). \
             Use safetensors or pass a different path."
        )
    }

    fn format_name(&self) -> &'static str {
        "gguf"
    }
}

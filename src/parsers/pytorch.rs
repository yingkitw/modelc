use std::path::Path;

use anyhow::Result;

use crate::model::Model;
use crate::parsers::WeightParser;

pub struct PytorchParser;

impl WeightParser for PytorchParser {
    fn parse(&self, _path: &Path) -> Result<Model> {
        anyhow::bail!(
            "PyTorch (.pt/.pth/.pkl) parsing is not implemented (format is Python pickle/zip; \
             prefer safetensors exports: https://github.com/huggingface/safetensors)."
        )
    }

    fn format_name(&self) -> &'static str {
        "pytorch"
    }
}

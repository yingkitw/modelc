pub mod gguf;
pub mod onnx;
pub mod pytorch;
pub mod safetensors;

use anyhow::Result;
use std::path::Path;

use crate::model::Model;

pub trait WeightParser: Send + Sync {
    fn parse(&self, path: &Path) -> Result<Model>;
    fn format_name(&self) -> &'static str;
}

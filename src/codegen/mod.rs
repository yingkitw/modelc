pub mod native;

use anyhow::Result;

use crate::model::Model;

pub trait CodeGenerator: Send + Sync {
    fn generate(
        &self,
        model: &Model,
        weights_path: &std::path::Path,
        output_dir: &std::path::Path,
        port: u16,
    ) -> Result<std::path::PathBuf>;
}

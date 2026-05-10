pub mod native;

use std::net::SocketAddr;
use std::path::Path;

use anyhow::Result;

use crate::model::Model;

pub trait CodeGenerator: Send + Sync {
    fn generate(
        &self,
        model: &Model,
        output_dir: &Path,
        listen: SocketAddr,
    ) -> Result<std::path::PathBuf>;
}

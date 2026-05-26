//! Local model package store for `modelc list` / `modelc pull`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Returns the platform-specific store directory for modelc packages.
///
/// - macOS: `~/Library/Application Support/modelc/models`
/// - Windows: `%APPDATA%/modelc/models`
/// - Linux: `~/.local/share/modelc/models`
pub fn store_dir() -> Result<PathBuf> {
    let base = dirs::data_dir().context("could not determine data directory")?;
    let dir = base.join("modelc").join("models");
    std::fs::create_dir_all(&dir).context("failed to create model store directory")?;
    Ok(dir)
}

/// List installed models (files ending with `.modelc` in the store).
pub fn list_models() -> Result<Vec<InstalledModel>> {
    let dir = store_dir()?;
    let mut models = Vec::new();

    for entry in std::fs::read_dir(&dir).context("failed to read store directory")? {
        let entry = entry.context("failed to read directory entry")?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("modelc") {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            let size = std::fs::metadata(&path)
                .map(|m| m.len())
                .unwrap_or(0);
            models.push(InstalledModel {
                name,
                path,
                size_bytes: size,
            });
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

/// Resolve a model name or path to an actual `.modelc` file path.
///
/// If `input` is an existing file path, return it directly.
/// Otherwise, look for `<name>.modelc` in the store.
pub fn resolve_model_path(input: &str) -> Result<PathBuf> {
    let path = Path::new(input);
    if path.is_file() {
        return Ok(path.to_path_buf());
    }

    let dir = store_dir()?;
 let candidate = dir.join(format!("{}.modelc", input));
    if candidate.is_file() {
        return Ok(candidate);
    }

    anyhow::bail!("model not found: {} (searched local path and store)", input)
}

/// Install a `.modelc` file into the store with the given name.
pub fn install(source: &Path, name: &str) -> Result<PathBuf> {
    let dir = store_dir()?;
    let dest = dir.join(format!("{}.modelc", name));
    std::fs::copy(source, &dest)
        .with_context(|| format!("failed to copy {:?} to {:?}", source, dest))?;
    Ok(dest)
}

#[derive(Debug)]
pub struct InstalledModel {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
}

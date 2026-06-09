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

            let mut architecture = None;
            let mut params = None;
            let mut compressed = false;

            if let Ok(header) = crate::pack::read_header(&path) {
                architecture = Some(header.architecture);
                let p: usize = header.tensors.iter().map(|t| {
                    t.shape.iter().product::<usize>()
                }).sum();
                params = Some(p);
            }

            // Determine compression by reading flags (version 2+ has flags at offset 10).
            if let Ok(meta) = std::fs::metadata(&path) {
                if meta.len() > 14 {
                    if let Ok(mut file) = std::fs::File::open(&path) {
                        use std::io::{Read, Seek};
                        let _ = file.seek(std::io::SeekFrom::Start(10));
                        let mut flags_bytes = [0u8; 4];
                        if file.read_exact(&mut flags_bytes).is_ok() {
                            let flags = u32::from_le_bytes(flags_bytes);
                            compressed = flags & 1 != 0;
                        }
                    }
                }
            }

            models.push(InstalledModel {
                name,
                path,
                size_bytes: size,
                architecture,
                params,
                compressed,
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

/// Download a `.modelc` file from a URL into the store with the given name.
pub fn download(url: &str, name: &str) -> Result<PathBuf> {
    let dir = store_dir()?;
    let dest = dir.join(format!("{}.modelc", name));

    let mut body = ureq::get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("download failed: {}", e))?
        .into_body();
    let mut reader = body.as_reader();

    let mut file = std::fs::File::create(&dest)
        .with_context(|| format!("failed to create {:?}", dest))?;
    std::io::copy(&mut reader, &mut file)
        .with_context(|| format!("failed to write downloaded data to {:?}", dest))?;

    Ok(dest)
}

#[derive(Debug)]
pub struct InstalledModel {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub architecture: Option<String>,
    pub params: Option<usize>,
    pub compressed: bool,
}

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

/// Search installed models by name or architecture substring.
pub fn search_models(query: &str) -> Result<Vec<InstalledModel>> {
    let q = query.to_lowercase();
    let all = list_models()?;
    let filtered: Vec<InstalledModel> = all
        .into_iter()
        .filter(|m| {
            m.name.to_lowercase().contains(&q)
                || m.architecture
                    .as_ref()
                    .map(|a| a.to_lowercase().contains(&q))
                    .unwrap_or(false)
        })
        .collect();
    Ok(filtered)
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
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

            let mut architecture = None;
            let mut params = None;
            let mut compressed = false;

            if let Ok(header) = crate::pack::read_header(&path) {
                architecture = Some(header.architecture);
                let p: usize = header
                    .tensors
                    .iter()
                    .map(|t| t.shape.iter().product::<usize>())
                    .sum();
                params = Some(p);
            }

            // Determine compression by reading flags (version 2+ has flags at offset 10).
            if let Ok(meta) = std::fs::metadata(&path)
                && meta.len() > 14
                && let Ok(mut file) = std::fs::File::open(&path)
            {
                use std::io::{Read, Seek};
                let _ = file.seek(std::io::SeekFrom::Start(10));
                let mut flags_bytes = [0u8; 4];
                if file.read_exact(&mut flags_bytes).is_ok() {
                    let flags = u32::from_le_bytes(flags_bytes);
                    compressed = flags & 1 != 0;
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

    let mut file =
        std::fs::File::create(&dest).with_context(|| format!("failed to create {:?}", dest))?;
    std::io::copy(&mut reader, &mut file)
        .with_context(|| format!("failed to write downloaded data to {:?}", dest))?;

    Ok(dest)
}

/// List versioned copies of a model (files matching `<name>.v<N>.modelc`).
pub fn list_versions(name: &str) -> Result<Vec<(u32, PathBuf)>> {
    let dir = store_dir()?;
    let mut versions = Vec::new();
    let prefix = format!("{}.v", name);

    for entry in std::fs::read_dir(&dir).context("failed to read store directory")? {
        let entry = entry.context("failed to read directory entry")?;
        let path = entry.path();
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            && let Some(tail) = stem.strip_prefix(&prefix)
            && let Some(ver_str) = tail.split('.').next()
            && let Ok(ver) = ver_str.parse::<u32>()
        {
            versions.push((ver, path));
        }
    }
    versions.sort_by_key(|(v, _)| *v);
    Ok(versions)
}

/// Remove a model from the store.
///
/// Deletes `<name>.modelc`. If `all` is true, also deletes all `<name>.v<N>.modelc` versioned copies.
/// If `force` is false and versioned copies exist (but `all` is false), refuses to delete.
pub fn remove_model(name: &str, all: bool, force: bool) -> Result<()> {
    let dir = store_dir()?;
    let main_path = dir.join(format!("{}.modelc", name));
    if !main_path.is_file() {
        anyhow::bail!("model '{}' not found in store", name);
    }

    if !all && !force {
        let prefix = format!("{}.v", name);
        let has_versions = std::fs::read_dir(&dir)
            .context("failed to read store directory")?
            .filter_map(|e| e.ok())
            .any(|entry| {
                let path = entry.path();
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|stem| stem.starts_with(&prefix) && path.is_file())
            });
        if has_versions {
            anyhow::bail!(
                "model '{}' has versioned copies; use --all to delete them or --force to proceed",
                name
            );
        }
    }

    std::fs::remove_file(&main_path)
        .with_context(|| format!("failed to remove {:?}", main_path))?;
    println!("Removed '{}'.", name);

    if all {
        let prefix = format!("{}.v", name);
        for entry in std::fs::read_dir(&dir).context("failed to read store directory")? {
            let entry = entry.context("failed to read directory entry")?;
            let path = entry.path();
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && stem.starts_with(&prefix)
                && path.is_file()
            {
                std::fs::remove_file(&path)
                    .with_context(|| format!("failed to remove {:?}", path))?;
                println!(
                    "Removed versioned copy {:?}.",
                    path.file_name().unwrap_or_default()
                );
            }
        }
    }

    Ok(())
}

/// Switch the active model to a specific version.
pub fn switch_version(name: &str, version: u32) -> Result<PathBuf> {
    let dir = store_dir()?;
    let source = dir.join(format!("{}.v{}.modelc", name, version));
    if !source.is_file() {
        anyhow::bail!(
            "version {} of '{}' not found at {:?}",
            version,
            name,
            source
        );
    }
    let dest = dir.join(format!("{}.modelc", name));
    std::fs::copy(&source, &dest)
        .with_context(|| format!("failed to copy {:?} to {:?}", source, dest))?;
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

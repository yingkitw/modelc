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
            let mut format = None;
            let mut quantization = None;
            let mut tensor_count = 0usize;

            if let Ok(header) = crate::pack::read_header(&path) {
                tensor_count = header.tensors.len();
                architecture = Some(header.architecture.clone());
                let p: usize = header
                    .tensors
                    .iter()
                    .map(|t| t.shape.iter().product::<usize>())
                    .sum();
                params = Some(p);
                format = header.metadata.get("format").cloned();
                quantization = Some(crate::pack::infer_quantization(&header));
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
                format,
                quantization,
                tensor_count,
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

/// Download a `.modelc` file from an `http://` or `https://` URL into the store.
///
/// - `https://` is delegated to the system `curl` (present on macOS, Linux and
///   Windows 10+) — keeps the binary free of a TLS stack.
/// - Authentication: `Authorization: Bearer <token>` is sent when a token is
///   provided (`download_auth`) or via the `MODELC_TOKEN` env var — needed to
///   pull from token-gated endpoints like modelc-web.
/// - Integrity: when the server sends `X-Modelc-Digest` (or an `ETag`) of the
///   form `sha256:<hex>` — the modelc-web store convention — the downloaded
///   bytes are verified against it; a mismatch is an error.
pub fn download(url: &str, name: &str) -> Result<PathBuf> {
    download_auth(url, name, None)
}

/// Like [`download`], with an explicit bearer token (env `MODELC_TOKEN` as fallback).
pub fn download_auth(url: &str, name: &str, token: Option<&str>) -> Result<PathBuf> {
    let dir = store_dir()?;
    let dest = dir.join(format!("{}.modelc", name));

    let token = token
        .map(str::to_string)
        .or_else(|| std::env::var("MODELC_TOKEN").ok())
        .filter(|t| !t.is_empty());

    let (body, digest_hdr) = if url.starts_with("https://") {
        https_get(url, token.as_deref())?
    } else {
        http_get(url, token.as_deref())?
    };

    verify_response_digest(&body, digest_hdr.as_deref())?;

    std::fs::write(&dest, &body)
        .with_context(|| format!("failed to write downloaded data to {:?}", dest))?;

    Ok(dest)
}

/// Verify the downloaded blob against a `sha256:<hex>` digest header, when present.
/// No header → nothing to check. Unparseable header → warn, don't fail.
fn verify_response_digest(body: &[u8], digest_hdr: Option<&str>) -> Result<()> {
    let Some(expected) = digest_hdr else { return Ok(()) };
    match crate::digest::verify_digest(body, expected) {
        Ok(true) => {
            eprintln!("digest verified: {expected}");
            Ok(())
        }
        Ok(false) => anyhow::bail!(
            "download corrupted: server digest {expected} != sha256 of downloaded bytes"
        ),
        Err(e) => {
            eprintln!("warning: could not verify digest header {expected:?}: {e}");
            Ok(())
        }
    }
}

/// Extract a `sha256:<hex>` digest from response headers — `X-Modelc-Digest`
/// first, then `ETag` (which may be quoted). Case-insensitive names.
fn extract_digest_header(headers: &str) -> Option<String> {
    let mut etag = None;
    for line in headers.lines().skip(1) {
        if let Some((k, v)) = line.split_once(':') {
            let v = v.trim();
            match k.trim().to_ascii_lowercase().as_str() {
                "x-modelc-digest" => return Some(v.to_string()),
                "etag" => etag = Some(v.trim_matches('"').to_string()),
                _ => {}
            }
        }
    }
    etag.filter(|e| e.starts_with("sha256:"))
}

/// Minimal HTTP/1.1 GET over `std::net::TcpStream`. Supports plain `http://` only.
/// Returns (body, digest-header-if-any).
fn http_get(url: &str, token: Option<&str>) -> Result<(Vec<u8>, Option<String>)> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    let rest = url
        .strip_prefix("http://")
        .context("unsupported URL scheme (use http:// or https://)")?;
    let (host_port, path) = match rest.split_once('/') {
        Some((h, p)) => (h, format!("/{}", p)),
        None => (rest, "/".to_string()),
    };
    let (host, port) = match host_port.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse::<u16>().unwrap_or(80)),
        None => (host_port.to_string(), 80u16),
    };

    let mut stream = TcpStream::connect((host.as_str(), port))
        .with_context(|| format!("failed to connect to {}:{}", host, port))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(300)))
        .context("failed to set read timeout")?;

    let auth = token
        .map(|t| format!("Authorization: Bearer {t}\r\n"))
        .unwrap_or_default();
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: modelc/0.1\r\nAccept: */*\r\n{auth}Connection: close\r\n\r\n"
    );
    stream
        .write_all(req.as_bytes())
        .context("failed to write HTTP request")?;

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .context("failed to read HTTP response")?;

    let header_end = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .context("invalid HTTP response: missing header terminator")?;
    let headers =
        std::str::from_utf8(&response[..header_end]).context("HTTP headers are not valid UTF-8")?;

    let status_line = headers.lines().next().context("empty HTTP response")?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .context("invalid HTTP status line")?;
    if status >= 400 {
        anyhow::bail!("HTTP request failed with status {}", status);
    }

    let digest = extract_digest_header(headers);
    Ok((response[header_end + 4..].to_vec(), digest))
}

/// HTTPS GET via the system `curl`: `curl` writes the body to a temp file and
/// dumps response headers to another. Returns (body, digest-header-if-any).
fn https_get(url: &str, token: Option<&str>) -> Result<(Vec<u8>, Option<String>)> {
    use std::process::Command;

    let hdr_path = std::env::temp_dir().join(format!(
        "modelc-hdr-{}-{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let out_path = hdr_path.with_extension("bin");

    let mut cmd = Command::new("curl");
    cmd.args([
        "-fsSL",
        "--connect-timeout",
        "30",
        "--max-time",
        "1800",
        "-A",
        concat!("modelc/", env!("CARGO_PKG_VERSION")),
        "-D",
    ])
    .arg(&hdr_path)
    .arg("-o")
    .arg(&out_path);
    if let Some(t) = token {
        cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
    }
    cmd.arg(url);

    let output = cmd.output().context(
        "curl not found — needed for https:// pulls; install curl or download manually and use `modelc install`",
    )?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = std::fs::remove_file(&hdr_path);
        let _ = std::fs::remove_file(&out_path);
        anyhow::bail!("https download failed (curl exit {:?}): {}", output.status.code(), stderr.trim());
    }

    let headers = std::fs::read_to_string(&hdr_path).unwrap_or_default();
    let body = std::fs::read(&out_path)
        .with_context(|| format!("curl produced no output file {:?}", out_path))?;
    let _ = std::fs::remove_file(&hdr_path);
    let _ = std::fs::remove_file(&out_path);

    Ok((body, extract_digest_header(&headers)))
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

/// Copy a model in the store to a new name.
///
/// Resolves `source` as a model name or file path, then copies the `.modelc`
/// artifact to `<dest>.modelc` in the store.
pub fn copy_model(source: &str, dest: &str) -> Result<PathBuf> {
    let src_path = resolve_model_path(source)?;
    let dir = store_dir()?;
    let dest_path = dir.join(format!("{}.modelc", dest));
    if dest_path.is_file() {
        anyhow::bail!("destination '{}' already exists in store", dest);
    }
    std::fs::copy(&src_path, &dest_path)
        .with_context(|| format!("failed to copy {:?} to {:?}", src_path, dest_path))?;
    Ok(dest_path)
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
    /// Source weight format (`safetensors`/`gguf`/…) from artifact metadata.
    pub format: Option<String>,
    /// Quantization label inferred from the artifact (`pack::infer_quantization`).
    pub quantization: Option<String>,
    /// Number of tensors in the artifact header.
    pub tensor_count: usize,
}

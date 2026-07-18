//! Minimal RAII temporary-directory helper.
//!
//! Replaces the `tempfile` crate with a tiny stdlib-only implementation:
//! `std::env::temp_dir()` plus a process-unique suffix. The directory is removed
//! on drop. Not secure against adversarial local users — fine for tests, build
//! scratch dirs, and short-lived downloads.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_path(label: &str) -> PathBuf {
    let pid = std::process::id();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("modelc-{}-{}-{}-{}", label, pid, nanos, seq))
}

/// A directory under the system temp dir. Removed recursively on drop.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn into_path(self) -> PathBuf {
        let path = self.path.clone();
        // Forget self so the Drop doesn't remove it.
        std::mem::forget(self);
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// A file path under the system temp dir. Removed on drop.
pub struct TempFile {
    path: PathBuf,
}

impl TempFile {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn tempdir() -> std::io::Result<TempDir> {
    let path = unique_path("dir");
    std::fs::create_dir_all(&path)?;
    Ok(TempDir { path })
}

pub fn tempfile() -> std::io::Result<TempFile> {
    Ok(TempFile {
        path: unique_path("file"),
    })
}

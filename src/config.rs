//! Configuration file support for modelc.
//!
//! Reads `~/.modelc/config.toml` for user-level defaults.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub bind: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    pub store_path: Option<PathBuf>,
    #[serde(default)]
    pub compress: Option<bool>,
}

impl Config {
    /// Load config from `~/.modelc/config.toml` if it exists.
    pub fn load() -> Result<Self> {
        let path = config_path()?;
        Self::load_from(&path)
    }

    /// Load config from an explicit path. Returns `Ok(default)` if the file does not exist.
    pub fn load_from(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config at {:?}", path))?;
        let cfg: Config = toml::from_str(&contents)
            .with_context(|| format!("failed to parse config at {:?}", path))?;
        Ok(cfg)
    }

    /// Save config to `~/.modelc/config.toml`.
    pub fn save(&self) -> Result<()> {
        let path = config_path()?;
        self.save_to(&path)
    }

    /// Save config to an explicit path. Creates parent directories as needed.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        let dir = path
            .parent()
            .context("config path has no parent directory")?;
        std::fs::create_dir_all(dir).context("failed to create config directory")?;
        let contents = toml::to_string_pretty(self).context("failed to serialize config")?;
        std::fs::write(path, contents)
            .with_context(|| format!("failed to write config to {:?}", path))?;
        Ok(())
    }
}

fn config_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not determine home directory")?;
    Ok(home.join(".modelc").join("config.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_tmp(name: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("modelc-config-{name}-{pid}-{nanos}"))
    }

    #[test]
    fn default_is_all_none() {
        let cfg = Config::default();
        assert_eq!(cfg.bind, None);
        assert_eq!(cfg.port, None);
        assert_eq!(cfg.store_path, None);
        assert_eq!(cfg.compress, None);
    }

    #[test]
    fn load_from_missing_file_returns_default() {
        let path = unique_tmp("missing").join("config.toml");
        let cfg = Config::load_from(&path).unwrap();
        assert_eq!(cfg, Config::default());
    }

    #[test]
    fn save_then_load_round_trips() {
        let path = unique_tmp("roundtrip").join("config.toml");
        let want = Config {
            bind: Some("127.0.0.1".to_string()),
            port: Some(9090),
            store_path: Some(PathBuf::from("/tmp/models")),
            compress: Some(true),
        };
        want.save_to(&path).unwrap();
        let got = Config::load_from(&path).unwrap();
        assert_eq!(want, got);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_partial_file_fills_defaults() {
        let path = unique_tmp("partial").join("config.toml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "port = 8080\n").unwrap();
        let cfg = Config::load_from(&path).unwrap();
        assert_eq!(cfg.port, Some(8080));
        assert_eq!(cfg.bind, None);
        assert_eq!(cfg.store_path, None);
        assert_eq!(cfg.compress, None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_creates_missing_parent_directories() {
        let path = unique_tmp("nested")
            .join("deep")
            .join("dir")
            .join("config.toml");
        let cfg = Config {
            port: Some(1234),
            ..Config::default()
        };
        cfg.save_to(&path).unwrap();
        assert!(path.is_file());
        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn load_invalid_toml_returns_error() {
        let path = unique_tmp("invalid").join("config.toml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "this is = not valid toml ====").unwrap();
        let err = Config::load_from(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("failed to parse config"),
            "unexpected error: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_wrong_type_returns_error() {
        let path = unique_tmp("wrongtype").join("config.toml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // `port` must be u16; "abc" is not a valid integer.
        std::fs::write(&path, "port = \"abc\"\n").unwrap();
        let err = Config::load_from(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("failed to parse config"), "unexpected: {msg}");
        let _ = std::fs::remove_file(&path);
    }
}

//! Configuration file support for modelc.
//!
//! Reads `~/.modelc/config.toml` for user-level defaults.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
        if !path.is_file() {
            return Ok(Self::default());
        }
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read config at {:?}", path))?;
        let cfg: Config = toml::from_str(&contents)
            .with_context(|| format!("failed to parse config at {:?}", path))?;
        Ok(cfg)
    }

    /// Save config to `~/.modelc/config.toml`.
    pub fn save(&self) -> Result<()> {
        let path = config_path()?;
        let dir = path
            .parent()
            .context("config path has no parent directory")?;
        std::fs::create_dir_all(dir).context("failed to create config directory")?;
        let contents = toml::to_string_pretty(self).context("failed to serialize config")?;
        std::fs::write(&path, contents)
            .with_context(|| format!("failed to write config to {:?}", path))?;
        Ok(())
    }
}

fn config_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not determine home directory")?;
    Ok(home.join(".modelc").join("config.toml"))
}

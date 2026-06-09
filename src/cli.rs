//! Command-line parsing and helpers shared with the compiler.

use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, ValueHint};

use crate::model::Model;

/// Maximum weights file size (bytes) to fully read when sniffing ambiguous paths for Safetensors.
const MAX_FULL_SNIFF_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Parser, Debug)]
#[command(
    name = "modelc",
    version = crate::CLI_VERSION,
    about = "Package and run model files — single-file artifacts, local inference"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    #[command(about = "Compile a model to a standalone executable")]
    Compile {
        #[arg(help = "Path to model weights file", value_hint = ValueHint::FilePath)]
        input: PathBuf,

        #[arg(short, long, help = "Output binary path", value_hint = ValueHint::FilePath)]
        output: Option<PathBuf>,

        #[arg(short = 'f', long = "format", help = "Input weight format", value_enum)]
        format: Option<WeightFormat>,

        #[arg(
            long,
            help = "Model architecture hint (overrides parsed value when set)",
            value_enum
        )]
        arch: Option<ModelArch>,

        #[arg(
            short = 'p',
            long,
            default_value_t = 8080u16,
            help = "Listening port (with --bind unless --listen is set)"
        )]
        port: u16,

        #[arg(
            long,
            default_value = "0.0.0.0",
            value_name = "IP",
            help = "IP address the generated binary binds (ignored if --listen is set)"
        )]
        bind: String,

        #[arg(
            long = "listen",
            value_name = "ADDR:PORT",
            help = "Full bind address for the generated server (overrides --bind and --port)"
        )]
        listen: Option<String>,

        #[arg(long, help = "Target triple for cross-compilation")]
        target: Option<String>,

        #[arg(
            long,
            help = "Build generated `model-serve` without `cargo --release` (debug artifacts)"
        )]
        debug: bool,
    },

    #[command(about = "Inspect a model weight file")]
    Inspect {
        #[arg(help = "Path to model weights file", value_hint = ValueHint::FilePath)]
        input: PathBuf,

        #[arg(short = 'f', long = "format", help = "Input weight format", value_enum)]
        format: Option<WeightFormat>,

        #[arg(long, help = "Generate a Markdown model card from metadata")]
        readme: bool,
    },

    #[command(about = "Pack model weights into a single .modelc artifact")]
    Pack {
        #[arg(help = "Path to model weights file", value_hint = ValueHint::FilePath)]
        input: PathBuf,

        #[arg(short, long, help = "Output artifact path", value_hint = ValueHint::FilePath)]
        output: Option<PathBuf>,

        #[arg(short = 'f', long = "format", help = "Input weight format", value_enum)]
        format: Option<WeightFormat>,

        #[arg(
            long,
            help = "Model architecture hint (overrides parsed value when set)",
            value_enum
        )]
        arch: Option<ModelArch>,

        #[arg(long, help = "Compress tensor data with zstd")]
        compress: bool,

        #[arg(long, help = "Quantize FP32 tensors (fp16, int8, int4)", value_enum)]
        quantize: Option<QuantizeMode>,

        #[arg(long, help = "Prune weights with abs(value) < threshold", value_name = "THRESHOLD")]
        prune: Option<f32>,
    },

    #[command(about = "Run a .modelc artifact (starts local HTTP server)")]
    Run {
        #[arg(help = "Path to .modelc artifact or model name", value_hint = ValueHint::FilePath)]
        input: String,

        #[arg(
            short = 'p',
            long,
            default_value_t = 8080u16,
            help = "Listening port"
        )]
        port: u16,

        #[arg(
            long,
            default_value = "127.0.0.1",
            value_name = "IP",
            help = "IP address to bind"
        )]
        bind: String,

        #[arg(long, help = "Print per-operation timing for each inference request")]
        profile: bool,
    },

    #[command(about = "List installed model packages")]
    List,

    #[command(about = "Search installed models by name or architecture")]
    Search {
        #[arg(help = "Query string (matches name or architecture)")]
        query: String,
    },

    #[command(about = "Pull a model package from a URL or path into the local store")]
    Pull {
        #[arg(help = "Source URL or file path")]
        source: String,

        #[arg(short, long, help = "Local name for the model")]
        name: Option<String>,

        #[arg(short, long, help = "Version tag (saves as <name>.v<version>.modelc)")]
        version: Option<u32>,
    },

    #[command(about = "Benchmark inference latency on a .modelc artifact")]
    Bench {
        #[arg(help = "Path to .modelc artifact or model name")]
        input: String,

        #[arg(short, long, default_value_t = 100, help = "Number of warmup iterations")]
        warmup: usize,

        #[arg(short, long, default_value_t = 1000, help = "Number of benchmark iterations")]
        iterations: usize,
    },

    #[command(about = "Verify a .modelc artifact integrity")]
    Verify {
        #[arg(help = "Path to .modelc artifact or model name")]
        input: String,
    },

    #[command(about = "Export a .modelc artifact to Safetensors")]
    Export {
        #[arg(help = "Path to .modelc artifact or model name")]
        input: String,

        #[arg(short, long, help = "Output path", value_hint = ValueHint::FilePath)]
        output: Option<PathBuf>,
    },

    #[command(about = "Generate shell completions")]
    Completions {
        #[arg(help = "Shell: bash, zsh, fish, elvish, powershell")]
        shell: String,
    },

    #[command(about = "List versions of an installed model")]
    Versions {
        #[arg(help = "Model name")]
        name: String,
    },

    #[command(about = "Switch active version of a model")]
    Switch {
        #[arg(help = "Model name")]
        name: String,

        #[arg(help = "Version number (e.g. 1, 2)")]
        version: u32,
    },

    #[command(about = "Generate a minimal Docker image for a .modelc artifact")]
    Containerize {
        #[arg(help = "Path to .modelc artifact or model name")]
        input: String,

        #[arg(short, long, help = "Output directory", value_hint = ValueHint::DirPath)]
        output: Option<PathBuf>,

        #[arg(long, help = "Base image", default_value = "debian:bookworm-slim")]
        base_image: String,
    },

    #[command(about = "Apply a LoRA adapter to a model artifact")]
    Lora {
        #[arg(help = "Path to .modelc artifact or model name")]
        model: String,

        #[arg(help = "Path to LoRA adapter (.safetensors)")]
        adapter: PathBuf,

        #[arg(short, long, default_value_t = 1.0, help = "LoRA alpha scaling factor")]
        alpha: f32,

        #[arg(short, long, help = "Output artifact path")]
        output: Option<PathBuf>,
    },
}

/// Resolve `--listen` vs `--bind` + `--port` before calling [`crate::compiler::compile`].
pub fn compile_listen(
    bind: &str,
    port: u16,
    listen: Option<&str>,
) -> anyhow::Result<std::net::SocketAddr> {
    if let Some(s) = listen {
        return s.trim().parse().with_context(|| {
            format!(
                "invalid socket address {:?} (expected e.g. 127.0.0.1:8080 or [::1]:8080)",
                s.trim()
            )
        });
    }
    let addr: IpAddr = bind
        .trim()
        .parse()
        .with_context(|| format!("invalid bind IP {:?}", bind.trim()))?;
    Ok(std::net::SocketAddr::new(addr, port))
}

/// Apply `--arch` CLI hint after parsing weights.
/// If no hint is provided and the current architecture is empty or "generic",
/// attempts to infer from tensor naming patterns.
pub fn apply_arch_hint(model: &mut Model, arch: Option<&ModelArch>) {
    if let Some(a) = arch {
        model.architecture = a.as_str().to_string();
        return;
    }
    if model.architecture.is_empty()
        || model.architecture == "generic"
        || model.architecture == "unknown"
    {
        let inferred = model.infer_architecture();
        if inferred != "generic" {
            model.architecture = inferred;
        }
    }
}

/// Generate shell completions for the given shell name.
pub fn generate_completions(shell_name: &str) -> anyhow::Result<()> {
    let shell = match shell_name.to_lowercase().as_str() {
        "bash" => clap_complete::Shell::Bash,
        "zsh" => clap_complete::Shell::Zsh,
        "fish" => clap_complete::Shell::Fish,
        "elvish" => clap_complete::Shell::Elvish,
        "powershell" | "pwsh" | "ps" => clap_complete::Shell::PowerShell,
        _ => anyhow::bail!("unsupported shell '{}'. Supported: bash, zsh, fish, elvish, powershell", shell_name),
    };
    let mut cmd = <Cli as clap::CommandFactory>::command();
    clap_complete::generate(shell, &mut cmd, "modelc", &mut std::io::stdout());
    Ok(())
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    Safetensors,
    Gguf,
    Onnx,
    Pytorch,
}

impl WeightFormat {
    pub fn detect(path: &Path) -> Option<Self> {
        let name = path.to_string_lossy().to_lowercase();

        if name.ends_with(".safetensors") {
            return Some(Self::Safetensors);
        }
        if name.ends_with(".gguf") {
            return Some(Self::Gguf);
        }
        if name.ends_with(".onnx") {
            return Some(Self::Onnx);
        }
        if name.ends_with(".pt") || name.ends_with(".pth") {
            return Some(Self::Pytorch);
        }
        if name.ends_with(".bin") {
            if name.contains("ggml") {
                return Some(Self::Gguf);
            }
            if name.contains("pytorch") {
                return Some(Self::Pytorch);
            }
            return sniff_path(path).ok().flatten();
        }

        sniff_path(path).ok().flatten()
    }
}

fn sniff_path(path: &std::path::Path) -> std::io::Result<Option<WeightFormat>> {
    let meta = std::fs::metadata(path)?;
    let len = meta.len();

    let read_n = (len as usize).min(512);
    let mut head = vec![0u8; read_n];
    if read_n > 0 {
        use std::io::Read;
        std::fs::File::open(path)?.read_exact(&mut head)?;
    }

    if head.starts_with(b"GGUF") {
        return Ok(Some(WeightFormat::Gguf));
    }
    if head.starts_with(b"PK\x03\x04") {
        return Ok(Some(WeightFormat::Pytorch));
    }

    if len <= MAX_FULL_SNIFF_BYTES {
        let data = std::fs::read(path)?;
        if safetensors::SafeTensors::deserialize(&data).is_ok() {
            return Ok(Some(WeightFormat::Safetensors));
        }
    }

    Ok(None)
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantizeMode {
    Fp16,
    Int8,
    Int4,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Gpt2,
    Bert,
    Mlp,
    Generic,
}

impl ModelArch {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Gpt2 => "gpt2",
            Self::Bert => "bert",
            Self::Mlp => "mlp",
            Self::Generic => "generic",
        }
    }
}

#[cfg(test)]
mod tests_listen {
    use super::*;
    #[test]
    fn compile_listen_bind_port() {
        let a = compile_listen("127.0.0.1", 9000, None).unwrap();
        assert_eq!(a.port(), 9000);
    }
}

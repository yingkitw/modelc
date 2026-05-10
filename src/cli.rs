use std::path::PathBuf;

use clap::{Parser, ValueHint};

#[derive(Parser, Debug)]
#[command(name = "modelc", version, about = "Compile model weight files to standalone executable binaries")]
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

        #[arg(long, help = "Model architecture hint", value_enum)]
        arch: Option<ModelArch>,

        #[arg(long, default_value = "8080", help = "HTTP serving port for compiled binary")]
        port: u16,

        #[arg(long, help = "Target triple for cross-compilation")]
        target: Option<String>,

        #[arg(long, help = "Build in release mode (default: true)")]
        release: Option<bool>,
    },

    #[command(about = "Inspect a model weight file")]
    Inspect {
        #[arg(help = "Path to model weights file", value_hint = ValueHint::FilePath)]
        input: PathBuf,

        #[arg(short = 'f', long = "format", help = "Input weight format", value_enum)]
        format: Option<WeightFormat>,
    },
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    Safetensors,
    Gguf,
    Onnx,
    Pytorch,
}

impl WeightFormat {
    pub fn detect(path: &PathBuf) -> Option<Self> {
        let name = path.to_string_lossy().to_lowercase();
        if name.ends_with(".safetensors") {
            Some(Self::Safetensors)
        } else if name.ends_with(".gguf") || name.ends_with(".bin") && name.contains("ggml") {
            Some(Self::Gguf)
        } else if name.ends_with(".onnx") {
            Some(Self::Onnx)
        } else if name.ends_with(".pt") || name.ends_with(".bin") || name.ends_with(".pth") {
            Some(Self::Pytorch)
        } else {
            None
        }
    }
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum ModelArch {
    Llama,
    Gpt2,
    Bert,
    Mlp,
    Generic,
}

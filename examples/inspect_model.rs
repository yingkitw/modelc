use std::path::PathBuf;

use modelc::cli::WeightFormat;
use modelc::compiler;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: inspect_model <model_file> [--format <format>]");
        eprintln!("  Supported formats: safetensors, gguf, onnx, pytorch");
        std::process::exit(1);
    }

    let input = PathBuf::from(&args[1]);
    let format: Option<WeightFormat> = if args.len() > 3 && args[2] == "--format" {
        match args[3].as_str() {
            "safetensors" => Some(WeightFormat::Safetensors),
            "gguf" => Some(WeightFormat::Gguf),
            "onnx" => Some(WeightFormat::Onnx),
            "pytorch" => Some(WeightFormat::Pytorch),
            _ => {
                eprintln!("Unknown format: {}", args[3]);
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    if let Err(e) = compiler::inspect(&input, format.as_ref()) {
        eprintln!("Error: {:#}", e);
        std::process::exit(1);
    }
}

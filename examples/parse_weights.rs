//! Parse a weight file into [`modelc::model::Model`] using the same parsers as the CLI.
//!
//! Run from the repository root:
//!
//! ```text
//! cargo run --example parse_weights -- path/to/weights.safetensors safetensors
//! cargo run --example parse_weights -- path/to/model.gguf gguf
//! cargo run --example parse_weights -- path/to/model.onnx onnx
//! cargo run --example parse_weights -- path/to/bundle.pt pytorch
//! ```
//!
//! The second argument is the format name (`safetensors`, `gguf`, `onnx`, `pytorch`).

use std::env;
use std::path::Path;

use anyhow::{Result, bail};

use modelc::model::Model;
use modelc::parsers::WeightParser;
use modelc::parsers::gguf::GgufParser;
use modelc::parsers::onnx::OnnxParser;
use modelc::parsers::pytorch::PytorchParser;
use modelc::parsers::safetensors::SafetensorsParser;

fn main() -> Result<()> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 2 {
        bail!(
            "usage: cargo run --example parse_weights -- <path> <format>\nformats: safetensors | gguf | onnx | pytorch"
        );
    }
    let path = args.remove(0);
    let fmt = args.remove(0);
    let p = Path::new(&path);

    let model: Model = match fmt.as_str() {
        "safetensors" => SafetensorsParser.parse(p)?,
        "gguf" => GgufParser.parse(p)?,
        "onnx" => OnnxParser.parse(p)?,
        "pytorch" => PytorchParser.parse(p)?,
        other => bail!("unknown format {other:?}"),
    };

    println!("name: {}", model.name);
    println!("architecture: {}", model.architecture);
    println!(
        "tensors: {} (≈ {:.2} MiB parameters as counted by modelc)",
        model.tensors.len(),
        model.total_params() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "raw bytes in IR: {:.2} KiB",
        model.total_bytes() as f64 / 1024.0
    );

    let mut names: Vec<&String> = model.tensors.keys().collect();
    names.sort();

    println!("\nfirst tensors (sorted by name, up to 8):");
    for name in names.iter().take(8).copied() {
        let t = &model.tensors[name];
        println!("  {name:?} dtype={:?} shape={:?}", t.dtype, t.shape);
    }
    if names.len() > 8 {
        println!("  …(+{} more)", names.len().saturating_sub(8));
    }

    if !model.metadata.is_empty() {
        println!("\nmetadata (subset):");
        for (k, v) in model.metadata.iter().take(6) {
            println!("  {k}: {v}");
        }
    }

    Ok(())
}

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::cli::WeightFormat;
use crate::codegen::native::NativeCodegen;
use crate::codegen::CodeGenerator;
use crate::parsers::gguf::GgufParser;
use crate::parsers::onnx::OnnxParser;
use crate::parsers::pytorch::PytorchParser;
use crate::parsers::safetensors::SafetensorsParser;
use crate::parsers::WeightParser;

fn get_parser(format: &WeightFormat) -> Box<dyn WeightParser> {
    match format {
        WeightFormat::Safetensors => Box::new(SafetensorsParser),
        WeightFormat::Gguf => Box::new(GgufParser),
        WeightFormat::Onnx => Box::new(OnnxParser),
        WeightFormat::Pytorch => Box::new(PytorchParser),
    }
}

pub fn compile(
    input: &Path,
    output: Option<&Path>,
    format: Option<&WeightFormat>,
    port: u16,
    target: Option<&str>,
    release: bool,
) -> Result<PathBuf> {
    let weight_format = format
        .cloned()
        .or_else(|| WeightFormat::detect(&input.to_path_buf()))
        .context("could not detect weight format; specify with -f/--format")?;

    eprintln!("modelc: parsing {:?} ({:?})...", input, weight_format);
    let start = Instant::now();

    let parser = get_parser(&weight_format);
    let model = parser.parse(input).with_context(|| {
        format!("failed to parse {:?} as {}", input, parser.format_name())
    })?;

    eprintln!(
        "  parsed {} tensors ({} params, {:.2} MB) in {:.2}s",
        model.tensors.len(),
        model.total_params(),
        model.total_bytes() as f64 / (1024.0 * 1024.0),
        start.elapsed().as_secs_f64(),
    );

    eprintln!("modelc: generating native binary...");
    let gen_start = Instant::now();

    let output_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| {
            let mut p = input.to_path_buf();
            p.set_extension("");
            PathBuf::from(format!("{}_serve", p.display()))
        });

    let build_dir = tempfile::tempdir().context("failed to create temp dir")?;
    let codegen = NativeCodegen;
    let project_dir =
        codegen.generate(&model, input, build_dir.path(), port)?;

    eprintln!(
        "  generated project in {:.2}s",
        gen_start.elapsed().as_secs_f64(),
    );

    eprintln!("modelc: compiling...");
    let compile_start = Instant::now();

    let mut cmd = Command::new("cargo");
    cmd.arg("build");
    if release {
        cmd.arg("--release");
    }
    if let Some(t) = target {
        cmd.args(["--target", t]);
    }
    cmd.current_dir(&project_dir);

    let status = cmd.status().context("failed to run cargo")?;
    if !status.success() {
        anyhow::bail!("cargo build failed with status {}", status);
    }

    let bin_path = if release {
        project_dir.join("target/release/model-serve")
    } else {
        project_dir.join("target/debug/model-serve")
    };

    if let Some(t) = target {
        let with_target = if release {
            project_dir.join(format!("target/{}/release/model-serve", t))
        } else {
            project_dir.join(format!("target/{}/debug/model-serve", t))
        };
        if with_target.exists() {
            std::fs::copy(&with_target, &output_path)
                .context("failed to copy binary to output")?;
        } else {
            std::fs::copy(&bin_path, &output_path)
                .context("failed to copy binary to output")?;
        }
    } else {
        std::fs::copy(&bin_path, &output_path)
            .context("failed to copy binary to output")?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&output_path, std::fs::Permissions::from_mode(0o755))
            .context("failed to set executable permissions")?;
    }

    eprintln!(
        "  compiled in {:.2}s -> {:?}",
        compile_start.elapsed().as_secs_f64(),
        output_path,
    );
    eprintln!("modelc: done.");

    Ok(output_path)
}

pub fn inspect(input: &Path, format: Option<&WeightFormat>) -> Result<()> {
    let weight_format = format
        .cloned()
        .or_else(|| WeightFormat::detect(&input.to_path_buf()))
        .context("could not detect weight format; specify with -f/--format")?;

    let parser = get_parser(&weight_format);
    let model = parser.parse(input).with_context(|| {
        format!("failed to parse {:?} as {}", input, parser.format_name())
    })?;

    println!("Model: {}", model.name);
    println!("Architecture: {}", model.architecture);
    println!("Format: {}", parser.format_name());
    println!(
        "Total parameters: {} ({:.2}M)",
        model.total_params(),
        model.total_params() as f64 / 1e6
    );
    println!(
        "Total size: {:.2} MB",
        model.total_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!("\nTensors ({}):", model.tensors.len());

    let mut names: Vec<&String> = model.tensors.keys().collect();
    names.sort();

    let max_name_len = names.iter().map(|n| n.len()).max().unwrap_or(0);
    for name in &names {
        let t = &model.tensors[*name];
        println!(
            "  {:width$}  {:?}  {:?}  {:.2} KB",
            name,
            t.shape,
            t.dtype,
            t.byte_len() as f64 / 1024.0,
            width = max_name_len,
        );
    }

    if !model.metadata.is_empty() {
        println!("\nMetadata:");
        for (k, v) in &model.metadata {
            println!("  {}: {}", k, v);
        }
    }

    Ok(())
}

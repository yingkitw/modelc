use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::cli::{ModelArch, QuantizeMode, WeightFormat, apply_arch_hint};
use crate::codegen::CodeGenerator;
use crate::codegen::native::NativeCodegen;
use crate::model::{DataType, Model};
use crate::parsers::WeightParser;
use crate::parsers::gguf::GgufParser;
use crate::parsers::onnx::OnnxParser;
use crate::parsers::pytorch::PytorchParser;
use crate::parsers::safetensors::SafetensorsParser;

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
    arch: Option<&ModelArch>,
    listen: SocketAddr,
    target: Option<&str>,
    release: bool,
) -> Result<PathBuf> {
    let weight_format = format
        .cloned()
        .or_else(|| WeightFormat::detect(input))
        .context("could not detect weight format; specify with -f/--format")?;

    eprintln!("modelc: parsing {:?} ({:?})...", input, weight_format);
    let start = Instant::now();

    let parser = get_parser(&weight_format);
    let mut model = parser
        .parse(input)
        .with_context(|| format!("failed to parse {:?} as {}", input, parser.format_name()))?;

    apply_arch_hint(&mut model, arch);

    eprintln!(
        "  parsed {} tensors ({} params, {:.2} MB) in {:.2}s",
        model.tensors.len(),
        model.total_params(),
        model.total_bytes() as f64 / (1024.0 * 1024.0),
        start.elapsed().as_secs_f64(),
    );

    eprintln!("modelc: generating native binary (listen: {})...", listen);
    let gen_start = Instant::now();

    let output_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        let mut p = input.to_path_buf();
        p.set_extension("");
        PathBuf::from(format!("{}_serve", p.display()))
    });

    let build_dir = tempfile::tempdir().context("failed to create temp dir")?;
    let codegen = NativeCodegen;
    let project_dir = codegen.generate(&model, build_dir.path(), listen)?;

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
            std::fs::copy(&with_target, &output_path).context("failed to copy binary to output")?;
        } else {
            std::fs::copy(&bin_path, &output_path).context("failed to copy binary to output")?;
        }
    } else {
        std::fs::copy(&bin_path, &output_path).context("failed to copy binary to output")?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&output_path, std::fs::Permissions::from_mode(0o755))
            .context("failed to set executable permissions")?;
    }
    #[cfg(windows)]
    {
        // Windows does not use POSIX executable bits; the file is already runnable.
    }

    eprintln!(
        "  compiled in {:.2}s -> {:?}",
        compile_start.elapsed().as_secs_f64(),
        output_path,
    );
    eprintln!("modelc: done.");

    Ok(output_path)
}

pub fn pack(
    input: &Path,
    output: Option<&Path>,
    format: Option<&WeightFormat>,
    arch: Option<&ModelArch>,
    compress: bool,
    quantize: Option<&QuantizeMode>,
    prune: Option<f32>,
) -> Result<PathBuf> {
    let weight_format = format
        .cloned()
        .or_else(|| WeightFormat::detect(input))
        .context("could not detect weight format; specify with -f/--format")?;

    let parser = get_parser(&weight_format);
    let mut model = parser
        .parse(input)
        .with_context(|| format!("failed to parse {:?} as {}", input, parser.format_name()))?;

    apply_arch_hint(&mut model, arch);

    if let Some(threshold) = prune {
        eprintln!("modelc: pruning weights with |value| < {}...", threshold);
        let mut pruned = 0;
        for (_name, td) in model.tensors.iter_mut() {
            if td.dtype != DataType::F32 {
                continue;
            }
            let mut changed = false;
            for chunk in td.data.chunks_exact_mut(4) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if val.abs() < threshold {
                    chunk.copy_from_slice(&0.0f32.to_le_bytes());
                    changed = true;
                }
            }
            if changed {
                pruned += 1;
            }
        }
        model.metadata.insert("pruned".to_string(), threshold.to_string());
        eprintln!("  pruned {} tensors", pruned);
    }

    if let Some(mode) = quantize {
        eprintln!("modelc: quantizing tensors to {:?}...", mode);
        let mut quantized = 0;
        for (name, td) in model.tensors.iter_mut() {
            if td.dtype != DataType::F32 {
                continue;
            }
            match mode {
                QuantizeMode::Fp16 => {
                    let count = td.element_count();
                    let mut new_data = Vec::with_capacity(count * 2);
                    for chunk in td.data.chunks_exact(4) {
                        let bits = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        let f16_val = half::f16::from_f32(bits);
                        new_data.extend_from_slice(&f16_val.to_le_bytes());
                    }
                    td.dtype = DataType::F16;
                    td.data = new_data;
                    quantized += 1;
                }
                QuantizeMode::Int8 => {
                    let count = td.element_count();
                    let mut floats = Vec::with_capacity(count);
                    for chunk in td.data.chunks_exact(4) {
                        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        floats.push(val);
                    }
                    let max_abs = floats.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    if max_abs > 0.0 {
                        let scale = max_abs / 127.0;
                        let mut new_data = Vec::with_capacity(count);
                        for val in floats {
                            let q = (val / scale).clamp(-127.0, 127.0).round() as i8;
                            new_data.push(q as u8);
                        }
                        td.dtype = DataType::I8;
                        td.data = new_data;
                        model.metadata.insert(format!("quant_scale.{}", name), scale.to_string());
                        quantized += 1;
                    }
                }
                QuantizeMode::Int4 => {
                    let count = td.element_count();
                    let mut floats = Vec::with_capacity(count);
                    for chunk in td.data.chunks_exact(4) {
                        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        floats.push(val);
                    }
                    let max_abs = floats.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    if max_abs > 0.0 {
                        let scale = max_abs / 7.0;
                        let mut new_data = Vec::with_capacity((count + 1) / 2);
                        let mut idx = 0;
                        while idx < count {
                            let q0 = (floats[idx] / scale).clamp(-7.0, 7.0).round() as i8;
                            let nibble0 = (q0 + 8) as u8 & 0x0F;
                            let nibble1 = if idx + 1 < count {
                                let q1 = (floats[idx + 1] / scale).clamp(-7.0, 7.0).round() as i8;
                                (q1 + 8) as u8 & 0x0F
                            } else {
                                0
                            };
                            new_data.push((nibble0 << 4) | nibble1);
                            idx += 2;
                        }
                        td.dtype = DataType::I8;
                        td.data = new_data;
                        model.metadata.insert(format!("quant_scale.{}", name), scale.to_string());
                        model.metadata.insert(format!("quant_mode.{}", name), "int4".to_string());
                        quantized += 1;
                    }
                }
            }
        }
        eprintln!("  quantized {} tensors", quantized);
    }

    let output_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        let mut p = input.to_path_buf();
        p.set_extension("modelc");
        p
    });

    eprintln!("modelc: packing {} tensors -> {:?}...", model.tensors.len(), output_path);
    crate::pack::pack(&model, &output_path, compress)?;
    if compress {
        let raw = model.total_bytes();
        let packed = std::fs::metadata(&output_path)?.len();
        eprintln!("  compressed: {} bytes -> {} bytes ({:.1}%)", raw, packed, 100.0 * packed as f64 / raw.max(1) as f64);
    }
    eprintln!("modelc: done -> {:?}", output_path);

    Ok(output_path)
}

/// Parse a model file and return the [`Model`] for programmatic inspection.
pub fn inspect_model(input: &Path, format: Option<&WeightFormat>) -> Result<Model> {
    let weight_format = format
        .cloned()
        .or_else(|| WeightFormat::detect(input))
        .context("could not detect weight format; specify with -f/--format")?;

    let parser = get_parser(&weight_format);
    let model = parser
        .parse(input)
        .with_context(|| format!("failed to parse {:?} as {}", input, parser.format_name()))?;

    Ok(model)
}

pub fn inspect(input: &Path, format: Option<&WeightFormat>) -> Result<()> {
    let model = inspect_model(input, format)?;
    let weight_format = format
        .cloned()
        .or_else(|| WeightFormat::detect(input))
        .context("could not detect weight format; specify with -f/--format")?;
    let parser = get_parser(&weight_format);

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

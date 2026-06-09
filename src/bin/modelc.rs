//! `modelc` CLI entrypoint (`cargo install` / `cargo run --bin modelc`).

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let cli = modelc::cli::Cli::parse();

    match &cli.command {
        modelc::cli::Commands::Compile {
            input,
            output,
            format,
            arch,
            port,
            bind,
            listen,
            target,
            debug,
        } => {
            let listen_sa = modelc::cli::compile_listen(bind.as_str(), *port, listen.as_deref())?;
            let embedded_release = !debug;

            eprintln!("modelc {}", modelc::CLI_VERSION);

            modelc::compiler::compile(
                input,
                output.as_deref(),
                format.as_ref(),
                arch.as_ref(),
                listen_sa,
                target.as_deref(),
                embedded_release,
            )?;
        }
        modelc::cli::Commands::Inspect { input, format } => {
            modelc::compiler::inspect(input, format.as_ref())?;
        }
        modelc::cli::Commands::Pack {
            input,
            output,
            format,
            arch,
            compress,
            quantize,
        } => {
            eprintln!("modelc {}", modelc::CLI_VERSION);
            let output_path = output.clone().unwrap_or_else(|| {
                let mut p = input.to_path_buf();
                p.set_extension("modelc");
                p
            });
            modelc::compiler::pack(
                input,
                Some(&output_path),
                format.as_ref(),
                arch.as_ref(),
                *compress,
                quantize.as_ref(),
            )?;
        }
        modelc::cli::Commands::Run { input, port, bind } => {
            let path = modelc::store::resolve_model_path(input)?;
            eprintln!("modelc run: loading {:?}...", path);
            let mut model = modelc::pack::unpack(&path)?;
            model.dequantize_in_place();
            eprintln!(
                "  model: {} | architecture: {} | tensors: {} | params: {}",
                model.name,
                model.architecture,
                model.tensors.len(),
                model.total_params(),
            );

            let addr: SocketAddr = format!("{}:{}", bind.trim(), port)
                .parse()
                .map_err(|e| anyhow::anyhow!("invalid bind address: {}", e))?;

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(modelc::serve::run_server(model, addr))?;
        }
        modelc::cli::Commands::List => {
            let models = modelc::store::list_models()?;
            if models.is_empty() {
                println!("No models installed. Use `modelc pull <source>` to add one.");
            } else {
                println!("Installed models:");
                for m in models {
                    let size_mb = m.size_bytes as f64 / (1024.0 * 1024.0);
                    let arch = m.architecture.as_deref().unwrap_or("unknown");
                    let params = m.params.map(|p| format!("{} params", p)).unwrap_or_default();
                    let comp = if m.compressed { " [zstd]" } else { "" };
                    println!(
                        "  {:20} {:>8.2} MB  {:12} {:>14}{}",
                        m.name, size_mb, arch, params, comp
                    );
                }
            }
        }
        modelc::cli::Commands::Pull { source, name } => {
            let is_url = source.starts_with("http://") || source.starts_with("https://");

            let model_name = name.clone().unwrap_or_else(|| {
                if is_url {
                    source
                        .trim_end_matches(".modelc")
                        .rsplit('/')
                        .next()
                        .unwrap_or("model")
                        .to_string()
                } else {
                    PathBuf::from(source)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("model")
                        .to_string()
                }
            });

            let dest = if is_url {
                modelc::store::download(source, &model_name)?
            } else {
                let source_path = PathBuf::from(source);
                if !source_path.is_file() {
                    anyhow::bail!("source not found: {:?}", source);
                }
                modelc::store::install(&source_path, &model_name)?
            };
            println!("Installed '{}' -> {:?}", model_name, dest);
        }
        modelc::cli::Commands::Bench {
            input,
            warmup,
            iterations,
        } => {
            let path = modelc::store::resolve_model_path(input)?;
            let mut model = modelc::pack::unpack(&path)?;
            model.dequantize_in_place();
            eprintln!(
                "Benchmarking {} (arch: {}, tensors: {}, params: {})",
                model.name,
                model.architecture,
                model.tensors.len(),
                model.total_params(),
            );
            run_benchmark(&model, *warmup, *iterations)?;
        }
        modelc::cli::Commands::Verify { input } => {
            let path = modelc::store::resolve_model_path(input)?;
            modelc::pack::verify(&path)?;
        }
        modelc::cli::Commands::Export { input, output } => {
            let path = modelc::store::resolve_model_path(input)?;
            let output_path = output.clone().unwrap_or_else(|| {
                let mut p = path.clone();
                p.set_extension("safetensors");
                p
            });
            modelc::pack::export_to_safetensors(&path, &output_path)?;
        }
    }

    Ok(())
}

fn run_benchmark(model: &modelc::model::Model, warmup: usize, iterations: usize) -> Result<()> {
    use modelc::runtime::serve::Runtime;
    use std::time::Instant;

    let runtime = Runtime::from_raw(&model.tensors);

    // Determine input size from first mlp layer or default to 4
    let input_size = if model.architecture == "mlp" {
        if let Some(w) = runtime.get("weight") {
            w.shape.get(1).copied().unwrap_or(4)
        } else {
            // Find first layerN.weight
            let mut sizes: Vec<usize> = model.tensors
                .keys()
                .filter(|k| k.starts_with("layer") && k.ends_with(".weight"))
                .filter_map(|k| runtime.get(k).map(|t| t.shape.get(1).copied().unwrap_or(4)))
                .collect();
            sizes.sort();
            sizes.first().copied().unwrap_or(4)
        }
    } else {
        4
    };

    let input: Vec<f32> = (0..input_size).map(|i| i as f32 * 0.1).collect();

    // Warmup
    for _ in 0..warmup {
        let _ = benchmark_inference(model, &runtime, &input);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = benchmark_inference(model, &runtime, &input);
    }
    let elapsed = start.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / iterations as f64;
    let throughput = iterations as f64 / elapsed.as_secs_f64();

    println!("Benchmark results ({} iterations):", iterations);
    println!("  total time: {:.3} ms", total_ms);
    println!("  avg latency: {:.3} ms", avg_ms);
    println!("  throughput: {:.1} inferences/sec", throughput);

    Ok(())
}

fn benchmark_inference(
    model: &modelc::model::Model,
    runtime: &modelc::runtime::serve::Runtime,
    input: &[f32],
) -> Vec<f32> {
    use modelc::runtime::tensor::Tensor;
    use modelc::runtime::ops;

    if model.architecture == "mlp" {
        // Try single weight/bias first
        if let (Some(w), Some(b)) = (runtime.get("weight"), runtime.get("bias")) {
            let x = Tensor::from_vec(input.to_vec(), vec![1, input.len()]);
            let mut out = ops::matmul(&x, w);
            out = ops::add(&out, b);
            return out.data;
        }
        // Try layerN pairs
        let mut ids: Vec<u32> = model
            .tensors
            .keys()
            .filter_map(|key| {
                let tail = key.strip_prefix("layer")?;
                let (idx, suf) = tail.split_once('.')?;
                if suf == "weight" {
                    idx.parse::<u32>().ok()
                } else {
                    None
                }
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        if !ids.is_empty() {
            let mut cur = input.to_vec();
            let last = ids.len() - 1;
            for (i, id) in ids.iter().enumerate() {
                let w_name = format!("layer{id}.weight");
                let b_name = format!("layer{id}.bias");
                if let (Some(w), Some(b)) = (runtime.get(&w_name), runtime.get(&b_name)) {
                    let x = Tensor::from_vec(cur, vec![1, w.shape[1]]);
                    let mut out = ops::matmul(&x, w);
                    out = ops::add(&out, b);
                    cur = out.data;
                    if i != last {
                        cur = ops::relu(&Tensor::from_vec(cur.clone(), vec![cur.len()])).data;
                    }
                }
            }
            return cur;
        }
    }

    // Fallback: simple matmul with first 2D tensor
    if let Some(name) = runtime.tensor_names().first() {
        if let Some(t) = runtime.get(name) {
            if t.shape.len() == 2 {
                let x = Tensor::from_vec(input.to_vec(), vec![1, input.len()]);
                let out = ops::matmul(&x, t);
                return out.data;
            }
        }
    }

    input.to_vec()
}

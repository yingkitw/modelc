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
        modelc::cli::Commands::Inspect { input, format, readme } => {
            if *readme {
                let mut model = modelc::compiler::inspect_model(input, format.as_ref())?;
                model.dequantize_in_place();
                println!("# {} Model Card\n", model.name);
                println!("## Overview\n");
                println!("- **Architecture**: {}", model.architecture);
                println!("- **Tensors**: {}", model.tensors.len());
                println!("- **Parameters**: {}", model.total_params());
                println!("- **Total bytes**: {}", model.total_bytes());
                println!("\n## Metadata\n");
                if model.metadata.is_empty() {
                    println!("_No metadata._");
                } else {
                    for (k, v) in &model.metadata {
                        println!("- **{}**: {}", k, v);
                    }
                }
                println!("\n## Tensors\n");
                let mut names: Vec<&str> = model.tensors.keys().map(|s| s.as_str()).collect();
                names.sort_unstable();
                for name in names {
                    let td = &model.tensors[name];
                    println!(
                        "- `{}`: shape={:?}, dtype={:?}, bytes={}",
                        name, td.shape, td.dtype, td.byte_len()
                    );
                }
            } else {
                modelc::compiler::inspect(input, format.as_ref())?;
            }
        }
        modelc::cli::Commands::Pack {
            input,
            output,
            format,
            arch,
            compress,
            quantize,
            prune,
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
                compress,
                quantize.as_ref(),
                *prune,
            )?;
        }
        modelc::cli::Commands::Run {
            input,
            port,
            bind,
            profile,
            max_tokens,
            temperature,
            seed,
            max_context,
            anchor_tokens,
            grammar,
            api_key,
            rate_limit,
        } => {
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
            if *profile {
                eprintln!("  profiling enabled");
            }

            let addr: SocketAddr = format!("{}:{}", bind.trim(), port)
                .parse()
                .map_err(|e| anyhow::anyhow!("invalid bind address: {}", e))?;

            let mut generation = modelc::generate::GenerationConfig::default();
            if let Some(n) = max_tokens {
                generation.max_tokens = *n;
            }
            if let Some(t) = temperature {
                generation.temperature = *t;
            }
            if let Some(s) = seed {
                generation.seed = Some(*s);
            }
            if let Some(c) = max_context {
                generation.max_context = Some(*c);
            }
            if let Some(a) = anchor_tokens {
                generation.anchor_tokens = *a;
            }
            if let Some(g) = grammar
                && let Some(c) = modelc::constraint::RegexConstraint::new(g)
            {
                generation.constraint = Some(std::sync::Arc::new(c));
            }

            let auth = modelc::serve::auth::AuthConfig::new(api_key.clone(), *rate_limit);
            let auth_opt = if api_key.is_some() || rate_limit.is_some() {
                Some(auth)
            } else {
                None
            };

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(modelc::serve::run_server(model, addr, *profile, generation, auth_opt))?;
        }
        modelc::cli::Commands::List => {
            let models = modelc::store::list_models()?;
            print_model_list(&models);
        }
        modelc::cli::Commands::Search { query } => {
            let models = modelc::store::search_models(query)?;
            if models.is_empty() {
                println!("No models matching '{}'.", query);
            } else {
                println!("Models matching '{}':", query);
                print_model_list(&models);
            }
        }
        modelc::cli::Commands::Pull { source, name, version } => {
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

            if let Some(ver) = version {
                let versioned = modelc::store::store_dir()?
                    .join(format!("{}.v{}.modelc", model_name, ver));
                std::fs::copy(&dest, &versioned)
                    .with_context(|| format!("failed to create versioned copy {:?}", versioned))?;
                println!("Installed '{}' v{} -> {:?}", model_name, ver, versioned);
            }

            println!("Installed '{}' -> {:?}", model_name, dest);
        }
        modelc::cli::Commands::Versions { name } => {
            let versions = modelc::store::list_versions(name)?;
            if versions.is_empty() {
                println!("No versions found for '{}'.", name);
            } else {
                println!("Versions for '{}':", name);
                for (ver, path) in versions {
                    let size_mb = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) as f64
                        / (1024.0 * 1024.0);
                    println!("  v{}  {:>8.2} MB  {:?}", ver, size_mb, path);
                }
            }
        }
        modelc::cli::Commands::Switch { name, version } => {
            let dest = modelc::store::switch_version(name, *version)?;
            println!("Switched '{}' to v{} -> {:?}", name, version, dest);
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
        modelc::cli::Commands::Completions { shell } => {
            modelc::cli::generate_completions(shell)?;
        }
        modelc::cli::Commands::Containerize { input, output, base_image } => {
            let path = modelc::store::resolve_model_path(input)?;
            let out_dir = output.clone().unwrap_or_else(|| {
                let mut p = std::env::current_dir().unwrap_or_default();
                p.push("modelc-docker");
                p
            });
            modelc::containerize::containerize(&path, &out_dir, base_image)?;
        }
        modelc::cli::Commands::Lora { model, adapter, alpha, output } => {
            let path = modelc::store::resolve_model_path(model)?;
            eprintln!("Loading model from {:?}...", path);
            let mut m = modelc::pack::unpack(&path)?;
            m.dequantize_in_place();
            eprintln!("Applying LoRA adapter {:?} (alpha={})...", adapter, alpha);
            modelc::lora::apply_lora(&mut m, adapter, *alpha)?;
            let out_path = output.clone().unwrap_or_else(|| {
                let mut p = path.clone();
                p.set_file_name(format!("{}.lora.modelc", p.file_stem().unwrap().to_string_lossy()));
                p
            });
            modelc::pack::pack(&m, &out_path, false)?;
            eprintln!("Saved adapted model -> {:?}", out_path);
        }
        modelc::cli::Commands::Rm { name, all, force } => {
            modelc::store::remove_model(name, *all, *force)?;
        }
    }

    Ok(())
}

fn run_benchmark(model: &modelc::model::Model, warmup: usize, iterations: usize) -> anyhow::Result<()> {
    use modelc::runtime::serve::Runtime;
    use std::time::Instant;

    let runtime = Runtime::from_raw(&model.tensors);

    let input_size = if model.architecture == "mlp" {
        if let Some(w) = runtime.get("weight") {
            w.shape.get(1).copied().unwrap_or(4)
        } else {
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

    for _ in 0..warmup {
        let _ = benchmark_inference(model, &runtime, &input);
    }

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
        if let (Some(w), Some(b)) = (runtime.get("weight"), runtime.get("bias")) {
            let x = Tensor::from_vec(input.to_vec(), vec![1, input.len()]);
            let mut out = ops::matmul(&x, w);
            out = ops::add(&out, b);
            return out.data;
        }
        let mut ids: Vec<u32> = model
            .tensors
            .keys()
            .filter_map(|key| {
                let tail = key.strip_prefix("layer")?;
                let (idx, suf) = tail.split_once('.')?;
                if suf == "weight" { idx.parse::<u32>().ok() } else { None }
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

fn print_model_list(models: &[modelc::store::InstalledModel]) {
    if models.is_empty() {
        println!("No models installed. Use `modelc pull <source>` to add one.");
        return;
    }
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

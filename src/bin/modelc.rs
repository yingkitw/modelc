//! `modelc` CLI entrypoint (`cargo install` / `cargo run --bin modelc`).

use std::path::PathBuf;

use anyhow::{Context, Result};
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
        modelc::cli::Commands::Inspect {
            input,
            format,
            readme,
            quant_sizes,
        } => {
            if *quant_sizes {
                modelc::compiler::print_quant_sizes(input, format.as_ref())?;
            } else if *readme {
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
                        name,
                        td.shape,
                        td.dtype,
                        td.byte_len()
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
                *compress,
                quantize.as_ref(),
                *prune,
            )?;
        }
        modelc::cli::Commands::Run { input, flags } => {
            let model = load_artifact(input, "run")?;
            start_http_server(model, flags)?;
        }
        modelc::cli::Commands::Serve { input, stdio, flags } => {
            let model = load_artifact(input, "serve")?;
            if *stdio {
                modelc::serve::run_stdio_server(
                    model,
                    flags.profile,
                    generation_from_flags(flags),
                    flags.max_concurrent,
                )?;
            } else {
                start_http_server(model, flags)?;
            }
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
        modelc::cli::Commands::Diff { source, target, json } => {
            modelc::diff::diff_command(source, target, *json)?;
        }

        modelc::cli::Commands::Pull {
            source,
            name,
            version,
            token,
        } => {
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
                modelc::store::download_auth(source, &model_name, token.as_deref())?
            } else {
                let source_path = PathBuf::from(source);
                if !source_path.is_file() {
                    anyhow::bail!("source not found: {:?}", source);
                }
                modelc::store::install(&source_path, &model_name)?
            };

            if let Some(ver) = version {
                let versioned =
                    modelc::store::store_dir()?.join(format!("{}.v{}.modelc", model_name, ver));
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
        modelc::cli::Commands::Containerize {
            input,
            output,
            base_image,
        } => {
            let path = modelc::store::resolve_model_path(input)?;
            let out_dir = output.clone().unwrap_or_else(|| {
                let mut p = std::env::current_dir().unwrap_or_default();
                p.push("modelc-docker");
                p
            });
            modelc::containerize::containerize(&path, &out_dir, base_image)?;
        }
        modelc::cli::Commands::Lora {
            model,
            adapter,
            alpha,
            output,
        } => {
            let path = modelc::store::resolve_model_path(model)?;
            eprintln!("Loading model from {:?}...", path);
            let mut m = modelc::pack::unpack(&path)?;
            m.dequantize_in_place();
            eprintln!("Applying LoRA adapter {:?} (alpha={})...", adapter, alpha);
            modelc::lora::apply_lora(&mut m, adapter, *alpha)?;
            let out_path = output.clone().unwrap_or_else(|| {
                let mut p = path.clone();
                p.set_file_name(format!(
                    "{}.lora.modelc",
                    p.file_stem().unwrap().to_string_lossy()
                ));
                p
            });
            modelc::pack::pack(&m, &out_path, false)?;
            eprintln!("Saved adapted model -> {:?}", out_path);
        }
        modelc::cli::Commands::Cp { source, dest } => {
            let dest_path = modelc::store::copy_model(source, dest)?;
            println!("Copied '{}' -> '{}' ({:?})", source, dest, dest_path);
        }
        modelc::cli::Commands::Rm { name, all, force } => {
            modelc::store::remove_model(name, *all, *force)?;
        }
        modelc::cli::Commands::Prompt {
            input,
            prompt,
            max_tokens,
            temperature,
            seed,
            top_p,
            min_p,
        } => {
            let path = modelc::store::resolve_model_path(input)?;
            let mut model = modelc::pack::unpack(&path)?;
            model.dequantize_in_place();
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
            if let Some(p) = top_p {
                generation.top_p = *p;
            }
            if let Some(p) = min_p {
                generation.min_p = *p;
            }
            let output = modelc::compiler::prompt_model(&model, prompt, &generation);
            print!("{output}");
            if !output.ends_with('\n') {
                println!();
            }
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
            let mut sizes: Vec<usize> = model
                .tensors
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
    use modelc::runtime::ops;
    use modelc::runtime::tensor::Tensor;

    use modelc::runtime::gemv::gemv_dot;

    // Row-major `[out, in]` weights through the shared gemv kernel — the same math
    // (and the same reduction order) as the serving path, so benchmarks measure
    // representative work instead of a transposed matmul.
    let affine = |w: &modelc::runtime::tensor::Tensor, b: Option<&modelc::runtime::tensor::Tensor>, x: &[f32]| -> Vec<f32> {
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let mut x = x.to_vec();
        x.resize(cols, 0.0);
        (0..rows)
            .map(|r| {
                let bias = b.and_then(|bb| bb.data.get(r).copied()).unwrap_or(0.0);
                bias + gemv_dot(&w.data[r * cols..(r + 1) * cols], &x)
            })
            .collect()
    };

    if model.architecture == "mlp" {
        // Try single weight/bias first
        if let (Some(w), Some(b)) = (runtime.get("weight"), runtime.get("bias")) {
            return affine(w, Some(b), input);
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
                    cur = affine(w, Some(b), &cur);
                    if i != last {
                        for v in &mut cur {
                            *v = v.max(0.0);
                        }
                    }
                }
            }
            return cur;
        }
    }

    // Fallback: simple matmul with first 2D tensor
    if let Some(t) = runtime
        .tensor_names()
        .first()
        .and_then(|name| runtime.get(name))
        .filter(|t| t.shape.len() == 2)
    {
        let x = Tensor::from_vec(input.to_vec(), vec![1, input.len()]);
        let out = ops::matmul(&x, t);
        return out.data;
    }

    input.to_vec()
}

/// Resolve and load a `.modelc` artifact (path or store name) for `run` / `serve`.
fn load_artifact(input: &str, label: &str) -> Result<modelc::model::Model> {
    let path = modelc::store::resolve_model_path(input)?;
    eprintln!("modelc {label}: loading {:?}...", path);
    let mut model = modelc::pack::unpack(&path)?;
    model.dequantize_in_place();
    eprintln!(
        "  model: {} | architecture: {} | tensors: {} | params: {}",
        model.name,
        model.architecture,
        model.tensors.len(),
        model.total_params(),
    );
    Ok(model)
}

/// Shared `run` / `serve` HTTP mode: bind `flags.bind:flags.port` and serve until shutdown.
fn start_http_server(model: modelc::model::Model, flags: &modelc::cli::ServeFlags) -> Result<()> {
    use std::net::SocketAddr;

    if flags.profile {
        eprintln!("  profiling enabled");
    }

    let addr: SocketAddr = format!("{}:{}", flags.bind.trim(), flags.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid bind address: {e}"))?;

    let generation = generation_from_flags(flags);
    let auth = modelc::serve::auth::AuthConfig::new(flags.api_key.clone(), flags.rate_limit);
    let auth_opt = if flags.api_key.is_some() || flags.rate_limit.is_some() {
        Some(auth)
    } else {
        None
    };

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(modelc::serve::run_server(
        model,
        addr,
        flags.profile,
        generation,
        auth_opt,
        flags.max_concurrent,
    ))?;
    Ok(())
}

/// Apply the `--max-tokens` / `--temperature` / … default flags onto [`modelc::generate::GenerationConfig`].
fn generation_from_flags(flags: &modelc::cli::ServeFlags) -> modelc::generate::GenerationConfig {
    let mut generation = modelc::generate::GenerationConfig::default();
    if let Some(n) = flags.max_tokens {
        generation.max_tokens = n;
    }
    if let Some(t) = flags.temperature {
        generation.temperature = t;
    }
    if let Some(s) = flags.seed {
        generation.seed = Some(s);
    }
    if let Some(rp) = flags.repetition_penalty {
        generation.repetition_penalty = rp;
    }
    if let Some(pp) = flags.presence_penalty {
        generation.presence_penalty = pp;
    }
    if let Some(fp) = flags.frequency_penalty {
        generation.frequency_penalty = fp;
    }
    if let Some(mp) = flags.min_p {
        generation.min_p = mp;
    }
    if let Some(c) = flags.max_context {
        generation.max_context = Some(c);
    }
    if let Some(a) = flags.anchor_tokens {
        generation.anchor_tokens = a;
    }
    if let Some(c) = flags
        .grammar
        .as_deref()
        .and_then(modelc::constraint::RegexConstraint::new)
    {
        generation.constraint = Some(std::sync::Arc::new(c));
    }
    generation
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
        let params = m
            .params
            .map(|p| format!("{} params", p))
            .unwrap_or_default();
        let comp = if m.compressed { " [zstd]" } else { "" };
        let quant = m
            .quantization
            .as_deref()
            .filter(|q| !q.eq_ignore_ascii_case("f32"))
            .map(|q| format!(" [{q}]"))
            .unwrap_or_default();
        println!(
            "  {:20} {:>8.2} MB  {:12} {:>14}{}{}",
            m.name, size_mb, arch, params, comp, quant
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelc::model::{DataType, Model, TensorData};
    use modelc::runtime::serve::Runtime;
    use std::collections::HashMap;

    /// The pybridge golden fixture: input [1, 2] → layer0 → relu → layer1 → 1.3
    /// (same numbers as `tests/pybridge_test.rs`).
    fn layer_stack_model() -> Model {
        let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut tensors = HashMap::new();
        tensors.insert(
            "layer0.weight".into(),
            TensorData { shape: vec![2, 2], dtype: DataType::F32, data: bytes(&[1.0, 0.5, -0.5, 2.0]) },
        );
        tensors.insert(
            "layer0.bias".into(),
            TensorData { shape: vec![2], dtype: DataType::F32, data: bytes(&[0.1, -0.1]) },
        );
        tensors.insert(
            "layer1.weight".into(),
            TensorData { shape: vec![1, 2], dtype: DataType::F32, data: bytes(&[2.0, -1.0]) },
        );
        tensors.insert(
            "layer1.bias".into(),
            TensorData { shape: vec![1], dtype: DataType::F32, data: bytes(&[0.5]) },
        );
        Model { name: "bench_fixture".into(), architecture: "mlp".into(), tensors, metadata: HashMap::new() }
    }

    /// Regression: `benchmark_inference` must use the shared gemv kernel with
    /// row-major `[out, in]` weights (the serving math). The old implementation
    /// fed `[1, in] @ [out, in]` into `ops::matmul`, which panics on the
    /// rectangular second layer (`[1, 2]` with input length 2).
    #[test]
    fn benchmark_inference_matches_serving_math_on_layer_stack() {
        let model = layer_stack_model();
        let runtime = Runtime::from_raw(&model.tensors);
        let out = benchmark_inference(&model, &runtime, &[1.0, 2.0]);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.3).abs() < 1e-4, "expected 1.3, got {out:?}");
    }

    #[test]
    fn benchmark_inference_handles_singleton_weight_bias_rows() {
        let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".into(),
            TensorData { shape: vec![2, 3], dtype: DataType::F32, data: bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) },
        );
        tensors.insert(
            "bias".into(),
            TensorData { shape: vec![2], dtype: DataType::F32, data: bytes(&[0.1, 0.2]) },
        );
        let model = Model { name: "w".into(), architecture: "mlp".into(), tensors, metadata: HashMap::new() };
        let runtime = Runtime::from_raw(&model.tensors);
        // Row r = bias[r] + dot(weight row r, x): [0·1 + 0.1·2 + 0.2·3] + 0.1 = 0.9,
        // [0·4 + 0.1·5 + 0.2·6] + 0.2 = 1.9. The old matmul path would have
        // asserted on the [1,3] @ [2,3] shape mismatch.
        let out = benchmark_inference(&model, &runtime, &[0.0, 0.1, 0.2]);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.9).abs() < 1e-4 && (out[1] - 1.9).abs() < 1e-4, "got {out:?}");
    }

    #[test]
    fn benchmark_inference_zero_pads_short_inputs() {
        let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut tensors = HashMap::new();
        tensors.insert(
            "layer0.weight".into(),
            TensorData { shape: vec![2, 2], dtype: DataType::F32, data: bytes(&[1.0, 2.0, 3.0, 4.0]) },
        );
        tensors.insert(
            "layer0.bias".into(),
            TensorData { shape: vec![2], dtype: DataType::F32, data: bytes(&[1.0, 2.0]) },
        );
        let model = Model { name: "pad".into(), architecture: "mlp".into(), tensors, metadata: HashMap::new() };
        let runtime = Runtime::from_raw(&model.tensors);
        // x = [1, 0] (zero-padded): [1·1 + 0·2] + 1 = 2, [1·3 + 0·4] + 2 = 5.
        let out = benchmark_inference(&model, &runtime, &[1.0]);
        assert!((out[0] - 2.0).abs() < 1e-4 && (out[1] - 5.0).abs() < 1e-4, "got {out:?}");
    }

    #[test]
    fn benchmark_inference_falls_back_to_first_2d_tensor() {
        let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut tensors = HashMap::new();
        tensors.insert(
            "m".into(),
            TensorData { shape: vec![2, 2], dtype: DataType::F32, data: bytes(&[1.0, 2.0, 3.0, 4.0]) },
        );
        let model = Model { name: "g".into(), architecture: "generic".into(), tensors, metadata: HashMap::new() };
        let runtime = Runtime::from_raw(&model.tensors);
        // matmul([1, 2], [[1, 2], [3, 4]]) → [1·1 + 2·3, 1·2 + 2·4] = [7, 10].
        let out = benchmark_inference(&model, &runtime, &[1.0, 2.0]);
        assert!((out[0] - 7.0).abs() < 1e-4 && (out[1] - 10.0).abs() < 1e-4, "got {out:?}");
    }

    #[test]
    fn benchmark_command_runs_end_to_end() {
        let model = layer_stack_model();
        run_benchmark(&model, 1, 2).expect("bench runs");
    }

    #[test]
    fn generation_from_flags_applies_every_override() {
        let flags = modelc::cli::ServeFlags {
            port: 9000,
            bind: "127.0.0.1".into(),
            profile: true,
            max_tokens: Some(5),
            temperature: Some(0.5),
            seed: Some(7),
            repetition_penalty: Some(1.1),
            presence_penalty: Some(0.2),
            frequency_penalty: Some(0.3),
            min_p: Some(0.1),
            max_context: Some(256),
            anchor_tokens: Some(4),
            grammar: Some("^[a-z]+$".into()),
            api_key: Some("k".into()),
            rate_limit: Some(60),
            max_concurrent: Some(2),
        };
        let g = generation_from_flags(&flags);
        assert_eq!(g.max_tokens, 5);
        assert_eq!(g.temperature, 0.5);
        assert_eq!(g.seed, Some(7));
        assert_eq!(g.repetition_penalty, 1.1);
        assert_eq!(g.presence_penalty, 0.2);
        assert_eq!(g.frequency_penalty, 0.3);
        assert_eq!(g.min_p, 0.1);
        assert_eq!(g.max_context, Some(256));
        assert_eq!(g.anchor_tokens, 4);
        assert!(g.constraint.is_some(), "valid grammar becomes a constraint");
    }

    #[test]
    fn generation_from_flags_defaults_and_rejects_invalid_grammar() {
        let flags = modelc::cli::ServeFlags {
            port: 8080,
            bind: "127.0.0.1".into(),
            profile: false,
            max_tokens: None,
            temperature: None,
            seed: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            min_p: None,
            max_context: None,
            anchor_tokens: None,
            grammar: Some("[invalid".into()),
            api_key: None,
            rate_limit: None,
            max_concurrent: None,
        };
        let g = generation_from_flags(&flags);
        let default = modelc::generate::GenerationConfig::default();
        assert_eq!(g.max_tokens, default.max_tokens);
        assert_eq!(g.temperature, default.temperature);
        assert_eq!(g.seed, None);
        assert!(g.constraint.is_none(), "unparseable grammar → no constraint");
    }

    #[test]
    fn load_artifact_resolves_paths_and_reports() {
        let model = layer_stack_model();
        let dir = modelc::tempdir::tempdir().unwrap();
        let artifact = dir.path().join("bench_fixture.modelc");
        modelc::pack::pack(&model, &artifact, false).unwrap();

        let loaded = load_artifact(artifact.to_str().unwrap(), "test").unwrap();
        assert_eq!(loaded.name, "bench_fixture");
        assert_eq!(loaded.architecture, "mlp");
        assert_eq!(loaded.tensors.len(), 4);
    }

    #[test]
    fn print_model_list_handles_empty_and_populated_stores() {
        print_model_list(&[]);

        let models = vec![modelc::store::InstalledModel {
            name: "demo".into(),
            path: "demo.modelc".into(),
            size_bytes: 1024,
            architecture: Some("mlp".into()),
            params: Some(6),
            compressed: true,
            format: Some("safetensors".into()),
            quantization: Some("int8".into()),
            tensor_count: 2,
        }];
        print_model_list(&models);
    }
}

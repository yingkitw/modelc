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
    }

    Ok(())
}

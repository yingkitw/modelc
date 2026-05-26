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
            )?;
        }
        modelc::cli::Commands::Run { input, port, bind } => {
            let path = modelc::store::resolve_model_path(input)?;
            eprintln!("modelc run: loading {:?}...", path);
            let model = modelc::pack::unpack(&path)?;
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
                    println!("  {:20} {:.2} MB  {:?}", m.name, size_mb, m.path);
                }
            }
        }
        modelc::cli::Commands::Pull { source, name } => {
            let source_path = PathBuf::from(source);
            if !source_path.is_file() {
                anyhow::bail!("source not found: {:?}", source);
            }

            let model_name = name.clone().unwrap_or_else(|| {
                source_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
                    .to_string()
            });

            let dest = modelc::store::install(&source_path, &model_name)?;
            println!("Installed '{}' -> {:?}", model_name, dest);
        }
    }

    Ok(())
}

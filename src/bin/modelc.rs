//! `modelc` CLI entrypoint (`cargo install` / `cargo run --bin modelc`).

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
    }

    Ok(())
}

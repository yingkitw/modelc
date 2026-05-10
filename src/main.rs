use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let cli = modelc::cli::Cli::parse();

    match &cli.command {
        modelc::cli::Commands::Compile {
            input,
            output,
            format,
            port,
            target,
            release,
            ..
        } => {
            let rel = release.unwrap_or(true);
            modelc::compiler::compile(
                input,
                output.as_deref(),
                format.as_ref(),
                *port,
                target.as_deref(),
                rel,
            )?;
        }
        modelc::cli::Commands::Inspect { input, format } => {
            modelc::compiler::inspect(input, format.as_ref())?;
        }
    }

    Ok(())
}

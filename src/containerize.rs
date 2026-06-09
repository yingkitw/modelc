//! Generate a minimal Docker/OCI image for a .modelc artifact.

use std::path::Path;

use anyhow::{Context, Result};

/// Generate a Dockerfile and entrypoint script for the given artifact.
pub fn containerize(
    artifact_path: &Path,
    output_dir: &Path,
    base_image: &str,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output directory {:?}", output_dir))?;

    let artifact_name = artifact_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.modelc");

    let dockerfile = format!(
        r#"FROM {base_image}
WORKDIR /app
COPY {artifact_name} /app/model.modelc
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
EXPOSE 8080
CMD ["/app/entrypoint.sh"]
"#,
        base_image = base_image,
        artifact_name = artifact_name,
    );

    let entrypoint = r#"#!/bin/sh
set -e
echo "modelc serve: starting..."
exec modelc run /app/model.modelc --bind 0.0.0.0 --port 8080
"#;

    std::fs::write(output_dir.join("Dockerfile"), dockerfile)
        .with_context(|| "failed to write Dockerfile")?;
    std::fs::write(output_dir.join("entrypoint.sh"), entrypoint)
        .with_context(|| "failed to write entrypoint.sh")?;

    // Copy the artifact into the build context
    let dest_artifact = output_dir.join(artifact_name);
    std::fs::copy(artifact_path, &dest_artifact)
        .with_context(|| format!("failed to copy artifact to {:?}", dest_artifact))?;

    eprintln!("Generated Docker build context in {:?}", output_dir);
    eprintln!("  Dockerfile");
    eprintln!("  entrypoint.sh");
    eprintln!("  {}", artifact_name);
    eprintln!("Build with:  docker build -t modelc-serve {:?}", output_dir);
    eprintln!("Run with:    docker run -p 8080:8080 modelc-serve");

    Ok(())
}

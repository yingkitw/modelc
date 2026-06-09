use anyhow::Result;

#[test]
fn containerize_creates_files() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let artifact = temp_dir.path().join("test.modelc");
    std::fs::write(&artifact, b"FAKE_MODELC_DATA")?;

    let out_dir = temp_dir.path().join("docker");
    modelc::containerize::containerize(&artifact, &out_dir, "debian:bookworm-slim")?;

    assert!(out_dir.join("Dockerfile").exists());
    assert!(out_dir.join("entrypoint.sh").exists());
    assert!(out_dir.join("test.modelc").exists());

    let dockerfile = std::fs::read_to_string(out_dir.join("Dockerfile"))?;
    assert!(dockerfile.contains("FROM debian:bookworm-slim"));
    assert!(dockerfile.contains("COPY test.modelc"));

    let entrypoint = std::fs::read_to_string(out_dir.join("entrypoint.sh"))?;
    assert!(entrypoint.contains("modelc serve"));
    assert!(entrypoint.contains("8080"));

    Ok(())
}

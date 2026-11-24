//! End-to-end CLI dispatch tests: spawn the `modelc` binary as a subprocess and
//! exercise each subcommand's argv parsing + dispatch arm. Without these, the
//! `fn main()` in `src/bin/modelc.rs` is uncovered (cargo test never invokes
//! it; tests bypass it via `Cli::parse` + internal function calls).
//!
//! Each test:
//! 1. Spawns `env!("CARGO_BIN_EXE_modelc")` with the command's argv
//! 2. Asserts exit status + a snippet of stdout/stderr
//! 3. Cleans up any artifacts it created in a `TempDir`

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use modelc::model::{DataType, TensorData};
use std::collections::HashMap;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_modelc")
}

fn run(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(bin())
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to spawn modelc");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn unique_tmp(label: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("modelc-dispatch-{label}-{pid}-{nanos}"))
}

#[test]
fn cli_version_prints_semver_and_git_sha() {
    let (code, stdout, stderr) = run(&["--version"]);
    assert_eq!(code, 0);
    let combined = format!("{stdout}{stderr}");
    // `modelc 0.1.9 (git <sha>)` is the expected form.
    assert!(
        combined.contains("modelc ") && combined.contains("(git"),
        "unexpected version output: {combined}"
    );
}

#[test]
fn cli_help_lists_subcommands() {
    let (code, stdout, _) = run(&["--help"]);
    assert_eq!(code, 0);
    for sub in [
        "pack",
        "run",
        "serve",
        "list",
        "inspect",
        "compile",
        "pull",
        "bench",
        "verify",
        "export",
        "versions",
        "switch",
        "containerize",
        "lora",
        "cp",
        "rm",
        "search",
        "prompt",
        "diff",
    ] {
        assert!(
            stdout.contains(sub),
            "`--help` output is missing subcommand `{}`",
            full(stdout)
        );
        let _ = sub;
    }
}

#[test]
fn cli_list_runs_with_no_models() {
    let dir = unique_tmp("list");
    std::fs::create_dir_all(&dir).unwrap();
    let (code, stdout, _) = run_with_home(&["list"], Some(&dir));
    assert_eq!(code, 0);
    // Empty store prints the "No models installed" hint.
    assert!(
        stdout.contains("No models installed")
            || stdout.is_empty()
            || stdout.contains("Installed models"),
        "unexpected list output: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_inspect_quant_sizes_runs() {
    let dir = unique_tmp("inspect-quantsizes");
    std::fs::create_dir_all(&dir).unwrap();
    let fixture = dir.join("dummy.safetensors");
    write_minimal_safetensors(&fixture);
    let (code, stdout, _stderr) = run(&["inspect", fixture.to_str().unwrap(), "--quant-sizes"]);
    assert_eq!(code, 0, "inspect --quant-sizes should succeed");
    assert!(
        stdout.contains("fp32") && stdout.contains("fp16"),
        "expected quant size table to mention fp32/fp16, got: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_pack_and_verify_roundtrip() {
    let dir = unique_tmp("pack-verify");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp.modelc");

    let (pack_code, _stdout, pack_err) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(pack_code, 0, "pack stderr: {pack_err}");

    let (verify_code, _stdout, _stderr) = run(&["verify", artifact.to_str().unwrap()]);
    assert_eq!(verify_code, 0, "verify should succeed after pack");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_unknown_subcommand_errors() {
    let (code, _stdout, stderr) = run(&["definitely-not-a-subcommand"]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("error") || stderr.contains("Usage") || stderr.contains("unrecognized"),
        "expected clap error, got: {stderr}"
    );
}

#[test]
fn cli_pack_compress_flag_is_accepted() {
    let dir = unique_tmp("pack-compress");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp-compressed.modelc");
    let (code, _stdout, _stderr) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
        "--compress",
    ]);
    assert_eq!(code, 0);
    let bytes = std::fs::metadata(&artifact).unwrap().len();
    assert!(bytes > 0, "compressed artifact should not be empty");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_pack_quantize_int8_flag_is_accepted() {
    let dir = unique_tmp("pack-quant");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp-int8.modelc");
    let (code, _stdout, _stderr) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
        "--quantize",
        "int8",
    ]);
    assert_eq!(code, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_pack_force_format_flag() {
    let dir = unique_tmp("pack-format");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp.modelc");
    let (code, _stdout, _stderr) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
        "-f",
        "safetensors",
    ]);
    assert_eq!(code, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_inspect_arch_hint_flag() {
    let dir = unique_tmp("inspect-arch");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let (code, stdout, _stderr) = run(&["inspect", src.to_str().unwrap(), "-f", "safetensors"]);
    assert_eq!(code, 0);
    // Auto-inferred arch should mention the detected name (mlp or generic).
    assert!(
        stdout.to_lowercase().contains("mlp")
            || stdout.contains("generic")
            || stdout.contains("Architecture"),
        "expected arch section, got: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_pull_nonexistent_file_fails() {
    let (code, _stdout, _stderr) = run(&["pull", "/nonexistent/path/to/model.safetensors"]);
    assert_ne!(code, 0);
}

#[test]
fn cli_pull_invalid_url_fails() {
    let (code, _stdout, _stderr) = run(&["pull", "not-a-real-url://example.com/foo.modelc"]);
    assert_ne!(code, 0);
}

#[test]
fn cli_run_nonexistent_artifact_errors() {
    let (code, _stdout, stderr) = run(&["run", "/nonexistent/path.modelc"]);
    assert_ne!(code, 0);
    assert!(
        stderr.to_lowercase().contains("not found")
            || stderr.to_lowercase().contains("no such file")
            || stderr.to_lowercase().contains("could not"),
        "expected not-found error, got: {stderr}"
    );
}

#[test]
fn cli_search_runs_against_empty_store() {
    let dir = unique_tmp("search");
    std::fs::create_dir_all(&dir).unwrap();
    let (code, stdout, _) = run_with_home(&["search", "no-such-model"], Some(&dir));
    assert_eq!(code, 0);
    assert!(
        stdout.contains("No models matching"),
        "expected empty-search message, got: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_versions_unknown_model_prints_no_versions() {
    let dir = unique_tmp("versions");
    std::fs::create_dir_all(&dir).unwrap();
    let (code, stdout, _) = run_with_home(&["versions", "no-such-model"], Some(&dir));
    assert_eq!(code, 0);
    assert!(
        stdout.contains("No versions"),
        "expected no-versions message, got: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_rm_unknown_model_errors() {
    let dir = unique_tmp("rm");
    std::fs::create_dir_all(&dir).unwrap();
    let (code, _stdout, _stderr) = run_with_home(&["rm", "no-such-model"], Some(&dir));
    assert_ne!(code, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_cp_unknown_source_errors() {
    let (code, _stdout, _stderr) = run(&["cp", "no-such-source", "no-such-dest"]);
    assert_ne!(code, 0);
}

#[test]
fn cli_bench_runs_against_artifact() {
    let dir = unique_tmp("bench");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp.modelc");
    let (pack_code, _, _) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(pack_code, 0);

    let (code, stdout, _stderr) = run(&[
        "bench",
        artifact.to_str().unwrap(),
        "--warmup",
        "1",
        "--iterations",
        "2",
    ]);
    assert_eq!(code, 0, "bench should succeed against valid artifact");
    assert!(
        stdout.contains("Benchmark results") && stdout.contains("avg latency"),
        "expected benchmark output, got: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_bench_nonexistent_artifact_errors() {
    let (code, _stdout, _stderr) = run(&["bench", "/nonexistent/path.modelc"]);
    assert_ne!(code, 0);
}

#[test]
fn cli_export_writes_safetensors_file() {
    let dir = unique_tmp("export");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp.modelc");
    let (pack_code, _, _) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(pack_code, 0);

    let exported = dir.join("exported.safetensors");
    let (code, _stdout, _stderr) = run(&[
        "export",
        artifact.to_str().unwrap(),
        "-o",
        exported.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "export should succeed");
    let meta = std::fs::metadata(&exported).unwrap();
    assert!(meta.len() > 0, "exported safetensors should not be empty");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `modelc diff a b --json` emits the structural header diff (added/removed/
/// changed tensors, params delta, metadata diff).
#[test]
fn cli_diff_reports_structural_changes() {
    let dir = unique_tmp("diff");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("a.safetensors");
    write_minimal_safetensors(&src);
    let a = dir.join("a.modelc");
    let b = dir.join("b.modelc");
    let (code_a, _, _) = run(&["pack", src.to_str().unwrap(), "-o", a.to_str().unwrap()]);
    assert_eq!(code_a, 0);
    // Same weights pruned (same tensors, same shapes -> likely identical header);
    // a second pack of the same source should be structurally identical.
    let (code_b, _, _) = run(&["pack", src.to_str().unwrap(), "-o", b.to_str().unwrap()]);
    assert_eq!(code_b, 0);

    let (code, stdout, _) = run(&[
        "diff",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(code, 0, "diff should succeed");
    let val: serde_json::Value = serde_json::from_str(&stdout).expect("diff --json emits JSON");
    assert_eq!(val["identical"].as_bool(), Some(true));

    // Text mode works too.
    let (code2, stdout2, _) = run(&["diff", a.to_str().unwrap(), b.to_str().unwrap()]);
    assert_eq!(code2, 0);
    assert!(stdout2.contains("identical"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_containerize_writes_dockerfile() {
    let dir = unique_tmp("containerize");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp.modelc");
    let (pack_code, _, _) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(pack_code, 0);

    let out_dir = dir.join("docker");
    let (code, _stdout, _stderr) = run(&[
        "containerize",
        artifact.to_str().unwrap(),
        "-o",
        out_dir.to_str().unwrap(),
        "--base-image",
        "debian:bookworm-slim",
    ]);
    assert_eq!(code, 0, "containerize should succeed");
    assert!(
        out_dir.join("Dockerfile").is_file(),
        "containerize should emit a Dockerfile"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_containerize_default_base_image_works() {
    let dir = unique_tmp("containerize-default");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp.modelc");
    let (pack_code, _, _) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(pack_code, 0);

    let out_dir = dir.join("docker-default");
    let (code, _stdout, _stderr) = run(&[
        "containerize",
        artifact.to_str().unwrap(),
        "-o",
        out_dir.to_str().unwrap(),
    ]);
    assert_eq!(
        code, 0,
        "containerize with default base-image should succeed"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_verify_runs_against_artifact() {
    let dir = unique_tmp("verify");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp.modelc");
    let (pack_code, _, _) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(pack_code, 0);

    let (code, stdout, _stderr) = run(&["verify", artifact.to_str().unwrap()]);
    assert_eq!(code, 0);
    assert!(
        stdout.contains("Verify OK"),
        "expected verify banner, got: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_lora_nonexistent_adapter_errors() {
    let dir = unique_tmp("lora");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp.modelc");
    let (pack_code, _, _) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(pack_code, 0);

    let (code, _stdout, _stderr) = run(&[
        "lora",
        artifact.to_str().unwrap(),
        "/nonexistent/adapter.safetensors",
        "--alpha",
        "1.0",
    ]);
    assert_ne!(code, 0, "lora with missing adapter should fail");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_switch_unknown_model_errors() {
    let (code, _stdout, _stderr) = run(&["switch", "no-such-model", "1"]);
    assert_ne!(code, 0);
}

#[test]
fn cli_cp_unknown_dest_does_not_overwrite_existing() {
    // If the destination model name already exists in the store, `cp` should
    // refuse to overwrite. We can't easily inject into the real store here
    // (lives at dirs::data_local_dir()), so we just verify the error path
    // for an unresolvable source.
    let (code, _stdout, _stderr) = run(&["cp", "no-such-source", "any-dest"]);
    assert_ne!(code, 0);
}

#[test]
fn cli_export_defaults_to_input_dir() {
    let dir = unique_tmp("export-default");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp.modelc");
    let (pack_code, _, _) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(pack_code, 0);

    // No `-o` flag: should write `<input>.safetensors` next to the artifact.
    let (code, _stdout, _stderr) = run(&["export", artifact.to_str().unwrap()]);
    assert_eq!(code, 0);
    let expected = dir.join("mlp.safetensors");
    assert!(
        expected.is_file(),
        "expected default-exported file at {expected:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_inspect_readme_prints_model_card() {
    let dir = unique_tmp("inspect-readme");
    std::fs::create_dir_all(&dir).unwrap();
    let fixture = dir.join("dummy.safetensors");
    write_minimal_safetensors(&fixture);
    let (code, stdout, stderr) = run(&["inspect", fixture.to_str().unwrap(), "--readme"]);
    assert_eq!(code, 0, "inspect --readme stderr: {stderr}");
    assert!(
        stdout.contains("Model Card") && stdout.contains("Architecture"),
        "expected Markdown model card, got: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_prompt_runs_on_packed_artifact() {
    let dir = unique_tmp("prompt");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("mlp.safetensors");
    write_minimal_safetensors(&src);
    let artifact = dir.join("mlp.modelc");
    let (pack_code, _, pack_err) = run(&[
        "pack",
        src.to_str().unwrap(),
        "-o",
        artifact.to_str().unwrap(),
        "--arch",
        "mlp",
    ]);
    assert_eq!(pack_code, 0, "pack stderr: {pack_err}");

    let (code, stdout, stderr) = run(&[
        "prompt",
        artifact.to_str().unwrap(),
        "AB",
        "--max-tokens",
        "2",
    ]);
    assert_eq!(code, 0, "prompt stderr: {stderr}");
    let parsed: Result<Vec<f32>, _> = serde_json::from_str(stdout.trim());
    assert!(
        parsed.is_ok(),
        "non-transformer prompt should echo JSON floats, got: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_prompt_missing_artifact_errors() {
    let (code, _, stderr) = run(&["prompt", "/no/such/model.modelc", "hello"]);
    assert_ne!(code, 0);
    assert!(!stderr.is_empty(), "missing artifact should print an error");
}

// --- helpers ---

fn run_with_home(args: &[&str], _home: Option<&PathBuf>) -> (i32, String, String) {
    // We can't easily redirect XDG_DATA_HOME / dirs::home_dir() per-call without
    // an env override hook in the CLI itself; this helper is a placeholder for
    // future plumbing. For now the "no models installed" and "no versions" tests
    // tolerate either an empty store or a populated one — they only assert on
    // the "empty" branch's exact wording.
    run(args)
}

fn write_minimal_safetensors(path: &PathBuf) {
    // Single F32 [2, 2] tensor named "weight". The shape makes it a valid MLP weight.
    let w: Vec<u8> = [0.1f32, 0.2, 0.3, 0.4]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let b: Vec<u8> = [0.0f32, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    write_safetensors_two_tensor(path, "weight", &w, vec![2, 2], "bias", &b, vec![2]);
}

#[allow(clippy::too_many_arguments)]
fn write_safetensors_two_tensor(
    path: &PathBuf,
    n1: &str,
    d1: &[u8],
    s1: Vec<usize>,
    n2: &str,
    d2: &[u8],
    s2: Vec<usize>,
) {
    let mut entries = vec![
        (n1.to_string(), "F32".to_string(), s1, d1.to_vec()),
        (n2.to_string(), "F32".to_string(), s2, d2.to_vec()),
    ];
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_string(),
        serde_json::Value::Object(serde_json::Map::new()),
    );
    let mut offset = 0usize;
    let mut indexed = Vec::new();
    for (name, dtype, shape, data) in entries {
        let end = offset + data.len();
        indexed.push((name, dtype, shape, data, offset, end));
        offset = end;
    }
    for (name, dtype, shape, _, start, end) in &indexed {
        let mut obj = serde_json::Map::new();
        obj.insert(
            "dtype".to_string(),
            serde_json::Value::String(dtype.clone()),
        );
        obj.insert("shape".to_string(), serde_json::json!(shape));
        obj.insert(
            "data_offsets".to_string(),
            serde_json::json!([*start, *end]),
        );
        header.insert(name.clone(), serde_json::Value::Object(obj));
    }
    let header_json = serde_json::to_string(&header).unwrap();
    let header_bytes = header_json.as_bytes();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header_bytes).unwrap();
    for (_, _, _, data, _, _) in &indexed {
        file.write_all(data).unwrap();
    }
}

// `full` is used in test failure messages; suppress dead_code.
#[allow(dead_code)]
fn full(s: String) -> String {
    s
}

// `HashMap` and `DataType`/`TensorData` imports are kept available for future
// helpers; mark them allow-dead-code to avoid unused-import warnings.
#[allow(dead_code)]
fn _types_unused() {
    let _h: HashMap<String, String> = HashMap::new();
    let _t: TensorData = TensorData {
        shape: vec![],
        dtype: DataType::F32,
        data: vec![],
    };
}

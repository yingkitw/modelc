//! Doc-sync checks: make documentation drift a `cargo test` failure.
//!
//! When these tests fail, update the DOC (AGENTS.md structure / ARCHITECTURE.md
//! mentions / TODO.md test counts) — not this file. AGENTS.md Step 9 (Update
//! Documentation) is the workflow that keeps them green.

use std::fs;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    fs::read_to_string(root().join(rel)).unwrap_or_else(|e| panic!("cannot read {rel}: {e}"))
}

fn count_attr(dir: &Path, attr: &str) -> usize {
    let mut count = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let text = fs::read_to_string(&p).unwrap();
                for line in text.lines() {
                    let t = line.trim_start();
                    // Match `#[test]`, `#[tokio::test]` (with optional args);
                    // any trailing attributes / macros are ignored by the prefix check.
                    if t.starts_with(&format!("{attr}]"))
                        || t.starts_with(&format!("{attr}("))
                        || t.starts_with(&format!("{attr} #["))
                    {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

/// Extract the integer immediately before `marker` (e.g. "920 inline unit tests" → 920).
fn leading_int(text: &str, marker: &str) -> usize {
    let idx = text
        .find(marker)
        .unwrap_or_else(|| panic!("doc missing marker '{marker}'"));
    let digits: String = text[..idx]
        .chars()
        .rev()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits
        .chars()
        .rev()
        .collect::<String>()
        .parse()
        .unwrap_or_else(|_| panic!("no integer before '{marker}'"))
}

fn lib_modules() -> Vec<String> {
    let lib = read("src/lib.rs");
    lib.lines()
        .filter_map(|l| {
            let t = l.trim();
            t.strip_prefix("pub mod ")?
                .strip_suffix(';')
                .map(str::to_owned)
        })
        .collect()
}

#[test]
fn agents_md_structure_lists_every_module() {
    let agents = read("AGENTS.md");
    let missing: Vec<String> = lib_modules()
        .into_iter()
        .filter(|m| {
            !agents.contains(&format!("{m}.rs"))
                && !agents.contains(&format!("{m}/"))
                && !agents.contains(&format!("{m}::"))
                && !agents.contains(&format!("`{m}`"))
        })
        .collect();
    assert!(
        missing.is_empty(),
        "modules missing from the AGENTS.md project structure: {missing:?}"
    );
}

/// A module counts as "documented" in prose docs when its name appears in a
/// recognisable reference form: `name.rs`, `name/`, `name::`, or `name` in backticks.
fn documented_in(module: &str, doc: &str) -> bool {
    doc.contains(&format!("{module}.rs"))
        || doc.contains(&format!("{module}/"))
        || doc.contains(&format!("{module}::"))
        || doc.contains(&format!("`{module}`"))
}

#[test]
fn architecture_md_mentions_every_module() {
    let arch = read("ARCHITECTURE.md");
    let missing: Vec<String> = lib_modules()
        .into_iter()
        .filter(|m| !documented_in(m, &arch))
        .collect();
    assert!(
        missing.is_empty(),
        "modules missing from ARCHITECTURE.md: {missing:?}"
    );
}

/// Count `#[test]` + `#[tokio::test]` in the source tree (the latter expands
/// to a `#[test]` at runtime). Anything outside `src/` and `tests/` is
/// excluded — `examples/` aren't counted as part of the project's suite.
fn count_inline_tests() -> usize {
    count_attr(&root().join("src"), "#[test") + count_attr(&root().join("src"), "#[tokio::test")
}

fn count_integration_tests() -> usize {
    count_attr(&root().join("tests"), "#[test") + count_attr(&root().join("tests"), "#[tokio::test")
}

#[test]
fn todo_md_unit_test_count_matches_code() {
    let actual = count_inline_tests();
    let claimed = leading_int(&read("TODO.md"), "inline unit tests");
    assert_eq!(
        claimed, actual,
        "TODO.md unit-test count is stale (run `cargo test`, then update the Done section)"
    );
}

#[test]
fn todo_md_integration_test_count_matches_code() {
    let actual = count_integration_tests();
    let claimed = leading_int(&read("TODO.md"), "integration tests");
    assert_eq!(
        claimed, actual,
        "TODO.md integration-test count is stale (run `cargo test`, then update the Done section)"
    );
}

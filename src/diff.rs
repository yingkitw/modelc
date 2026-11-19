//! Structural diff between two `.modelc` artifacts.
//!
//! Compares artifact *headers* — tensor graph, architecture, parameter count,
//! size, metadata — without touching tensor data. Port of modelc-web's
//! `api/_diff.ts`. Use cases: "what changed between the fp32 and int8 pack?",
//! "did the fine-tune change the architecture?", "which layers did the new
//! version add?".

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::pack::ArtifactHeader;

/// Cap the per-list output so a diff never becomes a wall of text.
const MAX_LISTED: usize = 50;

#[derive(Debug, Serialize)]
pub struct TensorChange {
    pub name: String,
    pub changes: Vec<&'static str>,
    pub from: TensorSide,
    pub to: TensorSide,
}

#[derive(Debug, Serialize)]
pub struct TensorSide {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct MetadataChange {
    pub key: String,
    pub from: Option<String>,
    pub to: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    pub from: u64,
    pub to: u64,
    pub delta: i64,
}

#[derive(Debug, Serialize)]
pub struct DiffResult {
    pub source: String,
    pub target: String,
    pub identical: bool,
    pub architecture_from: String,
    pub architecture_to: String,
    pub architecture_changed: bool,
    pub parameters: Delta,
    pub size_bytes: Delta,
    pub tensor_count: Delta,
    pub tensors_added: Vec<String>,
    pub tensors_removed: Vec<String>,
    pub tensors_changed: Vec<TensorChange>,
    pub tensors_unchanged: usize,
    pub metadata_added: Vec<String>,
    pub metadata_removed: Vec<String>,
    pub metadata_changed: Vec<MetadataChange>,
    pub metadata_unchanged: usize,
}

/// Read the header of an artifact (path or store name resolved by caller).
pub fn load_header(path: &Path) -> Result<ArtifactHeader> {
    crate::pack::read_header(path)
        .with_context(|| format!("failed to read artifact header from {:?}", path))
}

fn parameter_count(header: &ArtifactHeader) -> u64 {
    header
        .tensors
        .iter()
        .map(|t| t.shape.iter().product::<usize>() as u64)
        .sum()
}

fn delta(from: u64, to: u64) -> Delta {
    Delta {
        from,
        to,
        delta: to as i64 - from as i64,
    }
}

/// Compare two artifact headers. The result describes what *target* changed
/// relative to *source*.
pub fn diff_headers(
    source: &ArtifactHeader,
    target: &ArtifactHeader,
    source_label: &str,
    target_label: &str,
    source_size: u64,
    target_size: u64,
) -> DiffResult {
    let source_tensors: HashMap<&str, _> = source
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    let target_tensors: HashMap<&str, _> = target
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    let mut unchanged = 0usize;

    for (name, t) in &target_tensors {
        match source_tensors.get(name) {
            None => added.push((*name).to_string()),
            Some(from) => {
                let mut changes = Vec::new();
                if from.shape != t.shape {
                    changes.push("shape");
                }
                if !from.dtype.eq_ignore_ascii_case(&t.dtype) {
                    changes.push("dtype");
                }
                if from.length != t.length {
                    changes.push("bytes");
                }
                if changes.is_empty() {
                    unchanged += 1;
                } else {
                    changed.push(TensorChange {
                        name: (*name).to_string(),
                        changes,
                        from: TensorSide {
                            shape: from.shape.clone(),
                            dtype: from.dtype.clone(),
                            bytes: from.length,
                        },
                        to: TensorSide {
                            shape: t.shape.clone(),
                            dtype: t.dtype.clone(),
                            bytes: t.length,
                        },
                    });
                }
            }
        }
    }
    for name in source_tensors.keys() {
        if !target_tensors.contains_key(name) {
            removed.push((*name).to_string());
        }
    }
    added.sort();
    removed.sort();
    changed.sort_by(|a, b| a.name.cmp(&b.name));

    // Metadata diff.
    let mut m_added = Vec::new();
    let mut m_removed = Vec::new();
    let mut m_changed = Vec::new();
    let mut m_unchanged = 0usize;
    for (k, v) in &target.metadata {
        match source.metadata.get(k) {
            None => m_added.push(k.clone()),
            Some(sv) if sv == v => m_unchanged += 1,
            Some(sv) => m_changed.push(MetadataChange {
                key: k.clone(),
                from: Some(sv.clone()),
                to: Some(v.clone()),
            }),
        }
    }
    for k in source.metadata.keys() {
        if !target.metadata.contains_key(k) {
            m_removed.push(k.clone());
        }
    }
    m_added.sort();
    m_removed.sort();
    m_changed.sort_by(|a, b| a.key.cmp(&b.key));

    let identical = added.is_empty()
        && removed.is_empty()
        && changed.is_empty()
        && m_added.is_empty()
        && m_removed.is_empty()
        && m_changed.is_empty()
        && source.architecture == target.architecture
        && source_size == target_size;

    DiffResult {
        source: source_label.to_string(),
        target: target_label.to_string(),
        identical,
        architecture_from: source.architecture.clone(),
        architecture_to: target.architecture.clone(),
        architecture_changed: source.architecture != target.architecture,
        parameters: delta(parameter_count(source), parameter_count(target)),
        size_bytes: delta(source_size, target_size),
        tensor_count: delta(source.tensors.len() as u64, target.tensors.len() as u64),
        tensors_added: added,
        tensors_removed: removed,
        tensors_changed: changed,
        tensors_unchanged: unchanged,
        metadata_added: m_added,
        metadata_removed: m_removed,
        metadata_changed: m_changed,
        metadata_unchanged: m_unchanged,
    }
}

/// CLI entry: resolve both inputs (path or store name), diff, print or emit JSON.
pub fn diff_command(source: &str, target: &str, json: bool) -> Result<()> {
    let src_path = crate::store::resolve_model_path(source)?;
    let dst_path = crate::store::resolve_model_path(target)?;
    let src_header = load_header(&src_path)?;
    let dst_header = load_header(&dst_path)?;
    let src_size = std::fs::metadata(&src_path).map(|m| m.len()).unwrap_or(0);
    let dst_size = std::fs::metadata(&dst_path).map(|m| m.len()).unwrap_or(0);

    let result = diff_headers(
        &src_header,
        &dst_header,
        source,
        target,
        src_size,
        dst_size,
    );

    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    print_diff(&result);
    Ok(())
}

fn signed(d: i64) -> String {
    if d >= 0 { format!("+{d}") } else { format!("{d}") }
}

fn list_block(title: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    println!("{title} ({})", items.len());
    for (i, item) in items.iter().enumerate() {
        if i >= MAX_LISTED {
            println!("  … (+{} omitted)", items.len() - MAX_LISTED);
            break;
        }
        println!("  {item}");
    }
}

fn print_diff(r: &DiffResult) {
    println!("diff: {} → {}", r.source, r.target);
    if r.identical {
        println!("  artifacts are structurally identical");
        return;
    }
    println!(
        "  architecture: {} → {}{}",
        r.architecture_from,
        r.architecture_to,
        if r.architecture_changed { " (changed)" } else { "" }
    );
    println!(
        "  parameters:   {} → {} ({})",
        r.parameters.from,
        r.parameters.to,
        signed(r.parameters.delta)
    );
    println!(
        "  size:         {} → {} bytes ({} bytes)",
        r.size_bytes.from,
        r.size_bytes.to,
        signed(r.size_bytes.delta)
    );
    println!(
        "  tensors:      {} → {} ({})",
        r.tensor_count.from,
        r.tensor_count.to,
        signed(r.tensor_count.delta)
    );

    list_block("added tensors", &r.tensors_added);
    list_block("removed tensors", &r.tensors_removed);
    if !r.tensors_changed.is_empty() {
        println!("changed tensors ({})", r.tensors_changed.len());
        for (i, c) in r.tensors_changed.iter().enumerate() {
            if i >= MAX_LISTED {
                println!("  … (+{} omitted)", r.tensors_changed.len() - MAX_LISTED);
                break;
            }
            println!(
                "  {}: {} [{:?} {:?} {}B] → [{:?} {:?} {}B]",
                c.name,
                c.changes.join(","),
                c.from.shape,
                c.from.dtype,
                c.from.bytes,
                c.to.shape,
                c.to.dtype,
                c.to.bytes
            );
        }
    }
    println!("unchanged tensors: {}", r.tensors_unchanged);

    list_block("added metadata keys", &r.metadata_added);
    list_block("removed metadata keys", &r.metadata_removed);
    if !r.metadata_changed.is_empty() {
        println!("changed metadata ({})", r.metadata_changed.len());
        for (i, c) in r.metadata_changed.iter().enumerate() {
            if i >= MAX_LISTED {
                println!("  … (+{} omitted)", r.metadata_changed.len() - MAX_LISTED);
                break;
            }
            println!(
                "  {}: {:?} → {:?}",
                c.key,
                c.from.clone().unwrap_or_default(),
                c.to.clone().unwrap_or_default()
            );
        }
    }
    println!("unchanged metadata keys: {}", r.metadata_unchanged);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::ArtifactTensor;

    fn header(tensors: Vec<ArtifactTensor>, metadata: &[(&str, &str)], arch: &str) -> ArtifactHeader {
        ArtifactHeader {
            name: "t".into(),
            architecture: arch.into(),
            metadata: metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            tensors,
        }
    }

    fn tensor(name: &str, shape: &[usize], dtype: &str, len: u64) -> ArtifactTensor {
        ArtifactTensor {
            name: name.into(),
            shape: shape.to_vec(),
            dtype: dtype.into(),
            offset: 0,
            length: len,
        }
    }

    #[test]
    fn identical_headers_diff_clean() {
        let h = header(
            vec![tensor("w", &[4, 4], "f32", 64)],
            &[("format", "safetensors")],
            "mlp",
        );
        let r = diff_headers(&h, &h, "a", "b", 100, 100);
        assert!(r.identical);
        assert_eq!(r.tensors_unchanged, 1);
        assert_eq!(r.metadata_unchanged, 1);
        assert_eq!(r.parameters.delta, 0);
    }

    #[test]
    fn added_removed_changed_tensors_reported() {
        let a = header(
            vec![
                tensor("keep", &[4], "f32", 16),
                tensor("chg", &[8], "f32", 32),
                tensor("gone", &[2], "f32", 8),
            ],
            &[],
            "mlp",
        );
        let b = header(
            vec![
                tensor("keep", &[4], "f32", 16),
                tensor("chg", &[8], "f16", 16),
                tensor("new", &[3], "f32", 12),
            ],
            &[],
            "mlp",
        );
        let r = diff_headers(&a, &b, "a", "b", 56, 44);
        assert!(!r.identical);
        assert_eq!(r.tensors_added, vec!["new"]);
        assert_eq!(r.tensors_removed, vec!["gone"]);
        assert_eq!(r.tensors_changed.len(), 1);
        assert_eq!(r.tensors_changed[0].name, "chg");
        assert_eq!(r.tensors_changed[0].changes, vec!["dtype", "bytes"]);
        assert_eq!(r.tensors_unchanged, 1);
    }

    #[test]
    fn shape_change_detected() {
        let a = header(vec![tensor("w", &[4, 4], "f32", 64)], &[], "mlp");
        let b = header(vec![tensor("w", &[2, 8], "f32", 64)], &[], "mlp");
        let r = diff_headers(&a, &b, "a", "b", 0, 0);
        assert_eq!(r.tensors_changed[0].changes, vec!["shape"]);
    }

    #[test]
    fn metadata_and_arch_changes_reported() {
        let a = header(vec![], &[("license", "MIT"), ("old", "x")], "mlp");
        let b = header(vec![], &[("license", "Apache"), ("new", "y")], "llama");
        let r = diff_headers(&a, &b, "a", "b", 0, 0);
        assert!(r.architecture_changed);
        assert_eq!(r.metadata_added, vec!["new"]);
        assert_eq!(r.metadata_removed, vec!["old"]);
        assert_eq!(r.metadata_changed.len(), 1);
        assert_eq!(r.metadata_changed[0].key, "license");
    }

    #[test]
    fn parameter_delta_computed() {
        let a = header(vec![tensor("w", &[10], "f32", 40)], &[], "mlp");
        let b = header(vec![tensor("w", &[20], "f32", 80)], &[], "mlp");
        let r = diff_headers(&a, &b, "a", "b", 40, 80);
        assert_eq!(r.parameters.from, 10);
        assert_eq!(r.parameters.to, 20);
        assert_eq!(r.parameters.delta, 10);
        assert_eq!(r.size_bytes.delta, 40);
    }
}

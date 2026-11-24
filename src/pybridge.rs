//! Compile a Python program into model weights.
//!
//! `modelc pack model.py` executes the program with the system Python via a small
//! embedded driver ([`DRIVER`]) that harvests tensors from the program's globals —
//! `state_dict()`-bearing objects (torch `nn.Module`), numpy arrays, or plain nested
//! lists of numbers — and writes a Safetensors file that the normal packing pipeline
//! then consumes. The driver is stdlib-only on the Python side (numpy is used as a
//! fast path when importable), and no interpreter is embedded on the Rust side.
//!
//! Naming normalization: when every `state_dict()` entry forms an alternating
//! 2-D `*.weight` / 1-D `*.bias` stack, parameters are re-emitted as
//! `layer{N}.weight` / `layer{N}.bias` so `Model::infer_architecture` detects
//! `mlp` and the runtime/codegen MLP forward paths can run the artifact as a
//! model. Bare globals named `layer{N}_weight` / `layer{N}_bias` are normalized
//! the same way; everything else keeps its original (prefixed) name.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};

/// Python driver executed as `python <driver> <program.py> <out.safetensors>`.
/// Deliberately dependency-free: it must run on any Python 3 with only the stdlib.
const DRIVER: &str = r#"
"""modelc driver: execute a Python program and dump its weights as Safetensors."""
import json
import os
import re
import runpy
import struct
import sys
import traceback

LAYER_RE = re.compile(r"^layer(\d+)[._](weight|bias)$")


class _NotNumeric(Exception):
    pass


class Ragged(Exception):
    pass


def fail(msg, code=3):
    sys.stderr.write("modelc: %s\n" % msg)
    sys.exit(code)


def measure_value(obj):
    """(shape, flat_values) when obj is tensor-like, else None. Raises Ragged."""
    fn = getattr(obj, "tolist", None)
    if callable(fn):
        try:
            values = fn()
        except Exception:
            return None
        if not isinstance(values, list):
            return None
    elif isinstance(obj, list):
        values = obj
    else:
        return None
    shape = []
    cur = values
    while isinstance(cur, list):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    flat = []
    try:
        _flatten(values, flat)
    except _NotNumeric:
        return None
    total = 1
    for dim in shape:
        total *= dim
    if len(flat) != total:
        raise Ragged("shape %r holds %d values" % (shape, len(flat)))
    return shape, flat


def _flatten(node, out):
    if isinstance(node, list):
        for item in node:
            _flatten(item, out)
    elif isinstance(node, (int, float)):  # bool is an int subclass
        out.append(node)
    else:
        raise _NotNumeric(repr(node))


def state_dict_of(obj):
    fn = getattr(obj, "state_dict", None)
    if not callable(fn):
        return None
    try:
        sd = fn()
    except Exception:
        return None
    if isinstance(sd, dict) and sd and all(isinstance(k, str) for k in sd):
        return sd
    return None


def affine_pairs(sd):
    """[(weight_key, bias_key), ...] iff entries alternate 2-D *.weight / 1-D *.bias."""
    keys = list(sd.keys())
    if len(keys) < 2 or len(keys) % 2:
        return None
    try:
        for i in range(0, len(keys), 2):
            wval = measure_value(sd[keys[i]])
            bval = measure_value(sd[keys[i + 1]])
            if wval is None or bval is None:
                return None
            if not keys[i].endswith("weight") or not keys[i + 1].endswith("bias"):
                return None
            if len(wval[0]) != 2 or len(bval[0]) != 1:
                return None
    except Ragged:
        return None
    return [(keys[i], keys[i + 1]) for i in range(0, len(keys), 2)]


def harvest(tensors, name, obj):
    """Record obj under `name` when tensor-like; a ragged nest is a hard error."""
    try:
        measured = measure_value(obj)
    except Ragged as exc:
        fail("%s: ragged nesting (%s)" % (name, exc))
    if measured is not None:
        tensors[name] = measured


def extract(globs):
    tensors = {}
    sd_items = []
    for key, value in globs.items():
        if key == "_" or key.startswith("__"):
            continue
        sd = state_dict_of(value)
        if sd is not None:
            sd_items.append((key, sd))

    all_affine = bool(sd_items) and all(affine_pairs(sd) for _, sd in sd_items)
    if all_affine:
        layer = 0
        for _, sd in sd_items:
            for wkey, bkey in affine_pairs(sd):
                harvest(tensors, "layer%d.weight" % layer, sd[wkey])
                harvest(tensors, "layer%d.bias" % layer, sd[bkey])
                layer += 1
    else:
        for key, sd in sd_items:
            prefix = key + "." if len(sd_items) > 1 else ""
            for pkey, pval in sd.items():
                harvest(tensors, prefix + pkey, pval)

    for key, value in globs.items():
        if key == "_" or key.startswith("__") or state_dict_of(value) is not None:
            continue
        if isinstance(value, dict) and value:
            for sub, subval in value.items():
                if isinstance(sub, str):
                    harvest(tensors, key + "." + sub, subval)
            continue
        match = LAYER_RE.match(key)
        name = "layer%s.%s" % (match.group(1), match.group(2)) if match else key
        harvest(tensors, name, value)
    return tensors


def encode(flat, name):
    if all(isinstance(v, int) for v in flat) and all(
        -2147483648 <= v < 2147483648 for v in flat
    ):
        dtype, fmt, np_dtype = "I32", "<i", "i4"
    else:
        dtype, fmt, np_dtype = "F32", "<f", "f4"
    try:
        import numpy as np

        data = np.asarray(flat, dtype=np_dtype).tobytes()
    except Exception:
        try:
            data = b"".join(struct.pack(fmt, v) for v in flat)
        except Exception as exc:
            fail("%s: cannot encode values (%s)" % (name, exc))
    return dtype, data


def save_safetensors(tensors, out_path):
    header = {}
    blob = bytearray()
    for name, (shape, flat) in tensors.items():
        dtype, data = encode(flat, name)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob.extend(data)
    header["__metadata__"] = {
        "format": "modelc-python",
        "source": os.path.basename(sys.argv[1]),
    }
    header_json = json.dumps(header).encode("utf-8")
    with open(out_path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_json)))
        fh.write(header_json)
        fh.write(bytes(blob))


def run_pass(program, run_name):
    try:
        globs = runpy.run_path(program, run_name=run_name)
    except SystemExit:
        fail("%s called sys.exit() before tensors could be collected" % program, 1)
    except BaseException:
        traceback.print_exc()
        fail("failed to execute %s" % program, 1)
    return extract(globs)


def main():
    if len(sys.argv) != 3:
        fail("driver usage: modelc_driver.py <program.py> <out.safetensors>")
    program, out_path = sys.argv[1], sys.argv[2]

    tensors = run_pass(program, "__modelc__")
    if not tensors:
        # Second chance: scripts that build their model inside a main block.
        tensors = run_pass(program, "__main__")
    if not tensors:
        fail(
            "no tensors found in %s; expose weights as module-level lists, numpy "
            "arrays, an object with a state_dict(), or a dict of tensors" % program
        )
    save_safetensors(tensors, out_path)
    sys.stderr.write("modelc: extracted %d tensor(s) from %s\n" % (len(tensors), program))


if __name__ == "__main__":
    main()
"#;

/// Locate a usable Python interpreter, trying `python3` before `python`.
/// `status.success()` is required so Windows Store `python3` stubs are rejected.
fn find_python() -> Option<&'static str> {
    for candidate in ["python3", "python"] {
        let ok = Command::new(candidate)
            .arg("--version")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false);
        if ok {
            return Some(candidate);
        }
    }
    None
}

/// Execute `program` and write its harvested weights to `out_path` as Safetensors.
///
/// The user's code runs twice at most: first as an import (`__name__ ==
/// "__modelc__"`, so `if __name__ == "__main__"` blocks stay dormant); only if
/// that yields no tensors is the program re-run as `__main__`. Script stdout is
/// echoed back after the run completes; on failure stderr is folded into the
/// returned error so tracebacks and driver guidance reach the CLI message.
pub fn python_to_safetensors(program: &Path, out_path: &Path) -> Result<()> {
    let python = find_python().context(
        "no Python interpreter found (tried `python3`, `python`); \
         compiling .py programs requires Python on PATH",
    )?;
    let dir = crate::tempdir::tempdir().context("failed to create temp dir")?;
    let driver = dir.path().join("modelc_pydriver.py");
    std::fs::write(&driver, DRIVER).context("failed to write python driver")?;

    let output = Command::new(python)
        .arg(&driver)
        .arg(program)
        .arg(out_path)
        .output()
        .with_context(|| format!("failed to launch {python}"))?;
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "python weight extraction failed ({}) while compiling {}:\n{}{}",
            output.status,
            program.display(),
            stdout,
            stderr
        );
    }
    // Echo the script's own output so it is not swallowed by the capture.
    if !output.stdout.is_empty() {
        print!("{}", String::from_utf8_lossy(&output.stdout));
    }
    if !output.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    if !out_path.exists() {
        anyhow::bail!("python driver did not write {}", out_path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Tests drive the real bridge end-to-end; they self-skip when no
    /// Python interpreter is on PATH so the suite stays green elsewhere.
    fn skip_without_python() -> bool {
        if find_python().is_none() {
            eprintln!("skipping: no python3/python on PATH");
            true
        } else {
            false
        }
    }

    fn run_bridge(script: &str) -> (crate::tempdir::TempDir, std::path::PathBuf) {
        let dir = crate::tempdir::tempdir().expect("tempdir");
        let program = dir.path().join("program.py");
        std::fs::write(&program, script).expect("write program");
        let out = dir.path().join("out.safetensors");
        python_to_safetensors(&program, &out).expect("bridge run");
        (dir, out)
    }

    /// name → (shape, dtype, decoded f32 values) for quick assertions.
    fn read_tensors(path: &Path) -> BTreeMap<String, (Vec<usize>, String, Vec<f32>)> {
        let data = std::fs::read(path).expect("read safetensors");
        let st = safetensors::SafeTensors::deserialize(&data).expect("valid safetensors");
        let mut out = BTreeMap::new();
        for (name, view) in st.tensors() {
            let values = match view.dtype() {
                safetensors::Dtype::F32 => view.data().chunks_exact(4).map(|c| {
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                }).collect(),
                safetensors::Dtype::I32 => view.data().chunks_exact(4).map(|c| {
                    i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32
                }).collect(),
                other => panic!("unexpected dtype {other:?}"),
            };
            out.insert(
                name,
                (view.shape().to_vec(), format!("{:?}", view.dtype()), values),
            );
        }
        out
    }

    #[test]
    fn bridge_pure_python_lists_become_safetensors() {
        if skip_without_python() {
            return;
        }
        let (_dir, out) = run_bridge(
            r#"
layer0_weight = [[1.0, -2.0], [3.5, 4.0], [0.0, 0.5]]
layer0_bias = [0.25, -0.25]
counts = [[1, 2], [3, 4]]
skip_me = "not a tensor"
lr = 0.1
"#,
        );
        let tensors = read_tensors(&out);
        assert_eq!(tensors.len(), 3, "scalars/strings must be skipped: {tensors:?}");
        let (shape, dtype, values) = &tensors["layer0.weight"];
        assert_eq!(shape, &vec![3usize, 2]);
        assert_eq!(dtype, "F32");
        assert_eq!(values, &vec![1.0, -2.0, 3.5, 4.0, 0.0, 0.5]);
        assert_eq!(tensors["layer0.bias"].0, vec![2usize]);
        let (shape, dtype, _) = &tensors["counts"];
        assert_eq!(shape, &vec![2usize, 2]);
        assert_eq!(dtype, "I32", "all-int lists should stay I32");

        let raw = std::fs::read(&out).unwrap();
        let (_, meta) = safetensors::SafeTensors::read_metadata(&raw).unwrap();
        let meta = meta.metadata().as_ref().expect("__metadata__ present");
        assert_eq!(meta.get("format").map(String::as_str), Some("modelc-python"));
        assert_eq!(meta.get("source").map(String::as_str), Some("program.py"));
    }

    #[test]
    fn bridge_module_state_dict_normalizes_to_layer_names() {
        if skip_without_python() {
            return;
        }
        let (_dir, out) = run_bridge(
            r#"
class FakeTensor:
    def __init__(self, data):
        self._data = data
    def tolist(self):
        return self._data

class Layer:
    def __init__(self, w, b):
        self.weight = FakeTensor(w)
        self.bias = FakeTensor(b)

class Net:
    def __init__(self):
        self.fc1 = Layer([[1.0, 2.0]], [0.5])
        self.fc2 = Layer([[3.0], [4.0]], [1.0, 2.0])
    def state_dict(self):
        return {
            "fc1.weight": self.fc1.weight,
            "fc1.bias": self.fc1.bias,
            "fc2.weight": self.fc2.weight,
            "fc2.bias": self.fc2.bias,
        }

model = Net()
"#,
        );
        let tensors = read_tensors(&out);
        let names: Vec<&str> = tensors.keys().map(String::as_str).collect();
        assert_eq!(
            names,
            vec!["layer0.bias", "layer0.weight", "layer1.bias", "layer1.weight"],
            "affine stacks normalize to layer{{N}} names for the MLP runtime"
        );
        assert_eq!(tensors["layer0.weight"].0, vec![1usize, 2]);
        assert_eq!(tensors["layer1.weight"].0, vec![2usize, 1]);
        assert_eq!(tensors["layer1.bias"].2, vec![1.0, 2.0]);
    }

    #[test]
    fn bridge_keeps_original_names_when_state_dict_is_not_an_affine_stack() {
        if skip_without_python() {
            return;
        }
        let (_dir, out) = run_bridge(
            r#"
class FakeTensor:
    def __init__(self, data):
        self._data = data
    def tolist(self):
        return self._data

class Linear:
    def __init__(self):
        self.weight = FakeTensor([[1.0, 0.0], [0.0, 1.0]])
        self.bias = FakeTensor([0.0, 0.0])
    def state_dict(self):
        return {"weight": self.weight, "bias": self.bias}

class Block:
    def __init__(self):
        self.w = FakeTensor([[1.0, 2.0, 3.0]])
    def state_dict(self):
        return {"w": self.w}

net = Linear()
block = Block()
"#,
        );
        let tensors = read_tensors(&out);
        let names: Vec<&str> = tensors.keys().map(String::as_str).collect();
        assert_eq!(
            names,
            vec!["block.w", "net.bias", "net.weight"],
            "multiple state-dict objects keep prefixed original names"
        );
        assert_eq!(tensors["net.weight"].0, vec![2usize, 2]);
    }

    #[test]
    fn bridge_no_tensors_error_mentions_guidance() {
        let dir = crate::tempdir::tempdir().expect("tempdir");
        let program = dir.path().join("empty.py");
        std::fs::write(&program, "x = 5\n").expect("write program");
        let out = dir.path().join("out.safetensors");
        let err = python_to_safetensors(&program, &out).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("no tensors found"), "got: {msg}");
        assert!(msg.contains("state_dict"), "got: {msg}");
    }

    #[test]
    fn bridge_reruns_as_main_when_module_level_yields_nothing() {
        if skip_without_python() {
            return;
        }
        let (_dir, out) = run_bridge(
            r#"
def build():
    global layer0_weight, layer0_bias
    layer0_weight = [[2.0, 0.0], [0.0, 2.0]]
    layer0_bias = [1.0, 1.0]

if __name__ == "__main__":
    build()
"#,
        );
        let tensors = read_tensors(&out);
        assert_eq!(tensors["layer0.weight"].2, vec![2.0, 0.0, 0.0, 2.0]);
        assert_eq!(tensors["layer0.bias"].2, vec![1.0, 1.0]);
    }

    #[test]
    fn bridge_ragged_nested_list_fails_with_name() {
        let dir = crate::tempdir::tempdir().expect("tempdir");
        let program = dir.path().join("ragged.py");
        std::fs::write(&program, "bad = [[1.0, 2.0], [3.0]]\n").expect("write program");
        let out = dir.path().join("out.safetensors");
        let err = python_to_safetensors(&program, &out).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("bad: ragged nesting"), "got: {msg}");
    }
}

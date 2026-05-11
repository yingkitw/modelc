# SPEC — modelc

## Purpose

**modelc** is a command-line tool and library that:

1. **Loads** neural network **weights** from supported on-disk formats.
2. **Normalizes** them into an internal **`Model`** (named tensors with shape, dtype, and raw bytes).
3. **Emits** a temporary Rust **crate** that embeds tensors in a **single contiguous blob** (`embedded_weights.bin`) plus static `TensorMeta` offsets, and builds to a **standalone executable** (`model-serve`) with an HTTP listener address fixed at compile time.

The primary user outcome: ship a **single binary** per model for serving or integration, without requiring Python or the original framework at runtime.

## Non-goals (current scope)

- Full graph capture and arbitrary framework-accurate inference for every ONNX or PyTorch export (parsers focus on **weights** into the IR; generated servers emit a real **FP32 MLP** forward when `--arch mlp` matches `layerN.(weight,bias)` or `weight`/`bias`, otherwise `/infer` still echoes input).
- Training or fine-tuning.
- Replacing general-purpose inference servers (e.g. full ONNX Runtime) unless explicitly extended later.

## User-facing commands

### `inspect`

**Input:** path to a weights file; optional `-f` / `--format`.

**Output:** stdout summary — model name, architecture string, format name, parameter count, total size, sorted tensor list, optional metadata (`__metadata__` from Safetensors is merged when present).

**Errors:** missing or undetectable format; parse failure.

### `compile`

**Input:** path to weights; optional `-f`, `-o`, `--arch`, `--bind`, `--port`, **`--listen ADDR:PORT`** (overrides bind+port), `--target`, **`--debug`** (use debug profile on the emitted crate instead of `--release`).

**Output:** path to the built executable (copied from the generated crate’s `target/{...}/model-serve`).

**Side effects:** temp directory, generated `modelc_build` project, `cargo build`, copy binary, chmod on Unix.

**Errors:** parse failure; invalid `--listen` / `--bind`; `cargo` failure.

**Version line:** each `compile` invocation prints `modelc <semver> (git <sha>)` to stderr (semver from Cargo, SHA from `build.rs`, or `unknown` without git).

## Supported weight formats (declared)

| Format       | Parser status |
|-------------|---------------|
| Safetensors | Implemented (`safetensors` crate). |
| GGUF        | Implemented for **F32/F16/BF16** and contiguous integer blobs; quantized GGUF types error until dequant tooling exists. |
| ONNX        | Implemented for inlined **initializer** tensors via [`onnx-rs`](https://crates.io/crates/onnx-rs); external/tensor-segment payloads rejected. |
| PyTorch     | Implemented for **Safetensors-in-ZIP** (and standalone Safetensors mislabeled `.pt`/`.pth`); pickle-only checkpoints need export outside `modelc`. |

## Format detection

1. **Extension** (`.safetensors`, `.gguf`, `.onnx`, `.pt`, `.pth`, heuristics on `.bin`).
2. **Sniffing** when needed: `GGUF` prefix, zip `PK\x03\x04` (common for torch archives), or full-file Safetensors parse for files **≤ 64 MiB** when the extension is ambiguous or missing.
3. Otherwise detection fails and the user must pass `-f`.

## Internal model contract

- **`Model`**: `name`, `architecture`, `tensors: HashMap<String, TensorData>`, `metadata: HashMap<String, String>`.
- **`TensorData`**: `shape`, `dtype` (`DataType` enum), `data: Vec<u8>`.

Serialization via `serde` for tests and tooling.

## Generated artifact (`model-serve`)

- Rust edition **2021** in the emitted crate; **axum**, **tokio**, **serde**, **serde_json**.
- Static bytes: `include_bytes!("../embedded_weights.bin")` built from the IR (sorted tensor names, streamed concatenation straight to disk) so `byte_offset` / `byte_len` match the blob **without retaining a second full in-memory blob copy** during codegen.
- **Listen address** parsed at runtime from a string literal produced from `modelc compile` (`--listen` or `--bind` + `--port`).

### HTTP API (stable enough to document)

- **`GET /info`** — JSON object: `name`, `architecture`, `total_params`, `total_bytes`, `tensors` (array of tensor name strings).
- **`POST /infer`** — request JSON `{ "input": number[] }` (`f32`), response `{ "output": number[] }`. When the manifest `architecture` is **`mlp`**, codegen lowers a **GEMV stack with bias (+ ReLU between hidden layers)** for strict `layerN.weight`/`layerN.bias` pairs (contiguous IDs) or a single `weight`/`bias`; all other architectures still **echo** the input until expanded.

## Library runtime

`modelc::runtime::ops` exposes tensor helpers (`matmul`, `linear`, `softmax`, …). `Runtime::from_raw` loads tensors from `TensorData` into `f32` buffers for **F32, F16, BF16, I64, I32, I16, I8, U8, Bool** (integer/bool values are cast to `f32` for the internal representation).

## Compatibility

- **Host:** `cargo` required for `compile`.
- **Targets:** optional `--target` triple for the generated build.

## Success criteria

- `cargo test` and CI (`fmt`, `clippy -D warnings`, `test`) pass.
- `inspect` / `compile` succeed on supported fixtures and examples where parsers are implemented.

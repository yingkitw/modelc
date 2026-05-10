# SPEC — modelc

## Purpose

**modelc** is a command-line tool and library that:

1. **Loads** neural network **weights** from supported on-disk formats.
2. **Normalizes** them into an internal **`Model`** (named tensors with shape, dtype, and raw bytes).
3. **Emits** a temporary Rust **crate** that embeds those weights and builds to a **standalone executable** (`model-serve`) that can run an HTTP server.

The primary user outcome: ship a **single binary** per model for serving or integration, without requiring Python or the original framework at runtime.

## Non-goals (current scope)

- Full graph capture and arbitrary framework-accurate inference for every ONNX or PyTorch export (parsers focus on bringing **weights** into the internal representation; execution coverage grows incrementally).
- Training or fine-tuning.
- Replacing general-purpose inference servers (e.g. full ONNX Runtime feature set) unless explicitly extended later.

## User-facing commands

### `inspect`

**Input:** path to a weights file; optional `-f` / `--format`.

**Output:** stdout summary — model name, architecture string, format name, parameter count, total size, sorted tensor list (name, shape, dtype, size), optional metadata.

**Errors:** missing or undetectable format; parse failure.

### `compile`

**Input:** path to weights; optional `-f`, `-o` output path, `--port` for generated server, `--target` for `cargo build --target`, `--release` (default true).

**Output:** path to the built executable (copied from the generated crate’s `target/{...}/model-serve`).

**Side effects:** creates a temporary directory, writes a generated project under `modelc_build`, runs `cargo build`, copies the binary, sets executable permissions on Unix.

**Errors:** same as inspect for parse; `cargo` missing or build failure.

## Supported weight formats (declared)

| Format       | Role in pipeline                          |
|-------------|---------------------------------------------|
| Safetensors | Primary; uses `safetensors` crate           |
| GGUF        | Parser module present                       |
| ONNX        | Parser module present                       |
| PyTorch     | Parser module present                       |

Exact coverage (which tensors / dtypes / layouts) is defined by each parser implementation and tests.

## Internal model contract

- **`Model`**: `name`, `architecture` (string), `tensors: HashMap<String, TensorData>`, `metadata: HashMap<String, String>`.
- **`TensorData`**: `shape`, `dtype` (`DataType` enum), `data: Vec<u8>`.

Serializable via `serde` for tests and tooling.

## Generated artifact (`model-serve`)

- Rust edition **2021**, dependencies include **axum**, **tokio**, **serde**, **serde_json**.
- Embeds copied weights file and generated `main.rs` that describes tensor metadata for loading.
- Listens on HTTP using the port supplied to `modelc compile`.

Specific HTTP routes and JSON schemas should be documented once considered stable; until then, treat as implementation-defined.

## Compatibility

- **Host:** requires `cargo` available when running `compile`.
- **Targets:** optional `--target` triple passed through to the generated build.

## Success criteria

- `cargo test` passes in a clean checkout.
- `inspect` and `compile` succeed on supported examples in `examples/` and test fixtures.
- Generated binary runs where the host (or cross) toolchain can build the generated crate.

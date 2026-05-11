# ARCHITECTURE — modelc

## High-level pipeline

```
weights file → WeightParser → Model → NativeCodegen → temp crate → cargo build → model-serve binary
```

The **compiler** orchestrates parsing, optional `--arch` override, codegen, and the external `cargo` invocation. The **CLI** (`clap`) dispatches `compile` and `inspect`.

## Crate layout

| Module / path | Responsibility |
|---------------|----------------|
| `src/main.rs` | Binary entry; resolves `--listen` / `--bind`+`--port`, prints `CLI_VERSION`, calls `compiler`. |
| `src/cli.rs` | `Cli`, `Commands`, `WeightFormat`, `ModelArch`, format detection + magic sniffing, `compile_listen`. |
| `src/lib.rs` | Re-exports modules; `CLI_VERSION` (`CARGO_PKG_VERSION` + `MODELC_GIT_SHA` from `build.rs`). |
| `src/model.rs` | Canonical `Model`, `TensorData`, `DataType`; size helpers. |
| `src/compiler.rs` | Parser selection, `apply_arch_hint`, `compile`, `inspect`; tempfile + `cargo` subprocess. |
| `src/parsers/` | Format parsers (`WeightParser` trait). Safetensors, **GGUF (non‑quant contiguous)**, **ONNX initializers**, and **Safetensors-in-ZIP / raw Safetensors** PyTorch paths are implemented (`onnx-rs`, `zip`; see each module for limits). |
| `src/codegen/` | `CodeGenerator`; `native.rs` **streams** `embedded_weights.bin` (sorted tensors), `Cargo.toml`, `main.rs` with `/info` + `/infer`, optional **MLP GEMV/ReLU forward** when `architecture == "mlp"` naming matches; embeds listen `SocketAddr`. |
| `src/runtime/` | Library tensor + ops scaffolding (`ops`, `serve`, `tensor`). |

## Key abstractions

### `WeightParser`

Loads a path into `Model`. The compiler selects an implementation from `WeightFormat` or fails detection.

### `CodeGenerator`

`NativeCodegen::generate(model, output_dir, listen)` writes:

1. `modelc_build/` with `src/main.rs`.
2. `embedded_weights.bin` — **authoritative** weight bytes for `include_bytes!` (contiguous offsets per tensor).
3. Embedded listen address string (Display of `SocketAddr`).

Original weight files are **not** copied into the emitted crate anymore; the blob is derived from parsed `TensorData`.

### `Model` as IR

Parsers normalize into one structure so codegen and tests stay format-agnostic at the boundary.

Safetensors: optional header `__metadata__` is merged into `Model.metadata`; `architecture` prefers `architecture` then `model_type` keys.

## Build and I/O boundaries

- **Parsing:** read-only on user paths; surfaced via `anyhow::Context`.
- **Codegen:** writes under caller-provided temp directory, then copies output binary only.
- **Subprocess:** `compiler` invokes `cargo`; no rustc API linkage.

## Testing strategy

- **Integration:** `tests/` (CLI sniffing, codegen layout, parsers, compiler smoke, runtime).
- **`#![allow(dead_code)]`** on `tests/common` so shared helpers do not trigger `-D dead_code` under clippy across split test crates.

## Extension points

- New formats: implement `WeightParser`, extend `WeightFormat` + detector.
- Alternative emitters: new `CodeGenerator` impl(s).
- Broader codegen: architectures beyond **`mlp`** stacked linear (+ library `runtime::ops` helpers).

## Dependencies (compiler crate)

clap, serde/serde_json, anyhow, safetensors, byteorder, tempfile, **half**, **onnx-rs**, **zip**, plus generated HTTP stack (axum/tokio) in emitted projects. **`half`** underpins FP16/BF16 → FP32 casts in `runtime::serve`.

## Design principles

- Keep **`Model`** small and stable.
- Embedding uses a **single blob** so tensor metadata offsets are truthful.
- **Explicit** `-f/--format` when sniffing hits the 64 MiB Safetensors cap or ambiguous huge files.

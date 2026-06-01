# ARCHITECTURE — modelc

## High-level pipeline

```
weights file → WeightParser → Model → pack (.modelc) → run → HTTP inference
              └─→ NativeCodegen → temp crate → cargo build → model-serve binary
```

The **packer** creates `.modelc` single-file artifacts with JSON header + compressed tensor blob. The **runner** loads artifacts and serves HTTP inference. The **compiler** (legacy) generates standalone binaries via `cargo build`. The **CLI** (`clap`) dispatches `pack`, `run`, `list`, `pull`, `inspect`, and `compile`. Apple Silicon acceleration is available where supported.

## Crate layout

| Module / path | Responsibility |
|---------------|----------------|
| `src/main.rs` | Binary entry; resolves `--listen` / `--bind`+`--port`, prints `CLI_VERSION`, calls compiler/packer/runner. |
| `src/cli.rs` | `Cli`, `Commands`, `WeightFormat`, `ModelArch`, format detection + magic sniffing, `compile_listen`. |
| `src/lib.rs` | Re-exports modules; `CLI_VERSION` (`CARGO_PKG_VERSION` + `MODELC_GIT_SHA` from `build.rs`). |
| `src/model.rs` | Canonical `Model`, `TensorData`, `DataType`; size helpers. |
| `src/compiler.rs` | Parser selection, `apply_arch_hint`, `compile`, `inspect`; tempfile + `cargo` subprocess. |
| `src/pack.rs` | `.modelc` artifact writer: JSON header + raw tensor blob, optional zstd compression. |
| `src/store.rs` | Local model store management; platform-specific paths via `dirs` crate. |
| `src/parsers/` | Format parsers (`WeightParser` trait). Safetensors, **GGUF** (dense + **Q4_0 / Q8_0 → F32** expansion), **ONNX initializers** (inline or **external_data** next to the model), and **Safetensors-in-ZIP / raw Safetensors** PyTorch paths are implemented (`onnx-rs`, `zip`; ONNX segment slices and many quant dtypes are still out of scope). |
| `src/codegen/` | `CodeGenerator`; `native.rs` **streams** `embedded_weights.bin` (sorted tensors), `Cargo.toml`, `main.rs` with `/info` + `/infer`, optional **MLP GEMV/ReLU forward** when `architecture == "mlp"` naming matches; embeds listen `SocketAddr`. |
| `src/runtime/` | Library tensor + ops scaffolding (`ops`, `serve`, `tensor`). |
| `src/metal.rs` | Apple Silicon Metal acceleration skeleton (macOS only). |
| `src/compute/` | Metal compute shaders (`shaders.metal`). |

## Key abstractions

### `WeightParser`

Loads a path into `Model`. The compiler selects an implementation from `WeightFormat` or fails detection.

### `CodeGenerator`

`NativeCodegen::generate(model, output_dir, listen)` writes:

1. `modelc_build/` with `src/main.rs`.
2. `embedded_weights.bin` — **authoritative** weight bytes for `include_bytes!` (contiguous offsets per tensor).
3. Embedded listen address string (Display of `SocketAddr`).

Original weight files are **not** copied into the emitted crate anymore; the blob is derived from parsed `TensorData`.

### `.modelc` artifact format

Single-file format created by `pack` command:
- **JSON header** — model metadata (name, architecture, tensor count, compression flag)
- **Raw tensor blob** — concatenated tensor data in sorted order
- **Optional compression** — zstd compression for version 2 format

The `run` command loads `.modelc` artifacts by:
1. Reading JSON header to parse metadata
2. Decompressing (if compressed version 2) or reading raw tensor data
3. Loading tensors into runtime for inference

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
- Platform backends: Metal for Apple Silicon (skeleton implemented, full kernels pending), CPU SIMD paths (AVX/NEON) for other targets.
- Remote model registry: implement `pull` command to fetch from model hubs.

## Dependencies (compiler crate)

clap, serde/serde_json, anyhow, safetensors, byteorder, tempfile, **half**, **onnx-rs**, **zip**, **dirs** (platform paths), **zstd** (compression), plus generated HTTP stack (axum/tokio) in emitted projects. **`half`** underpins FP16/BF16 → FP32 casts in `runtime::serve`. **`metal`** crate (macOS only) for Apple Silicon acceleration.

## Design principles

- Keep **`Model`** small and stable.
- Embedding uses a **single blob** so tensor metadata offsets are truthful.
- The packaged artifact is a **single file** optimized for size and fast loading.
- **Explicit** `-f/--format` when sniffing hits the 64 MiB Safetensors cap or ambiguous huge files.
- Cross-platform by default; acceleration is opt-in per platform backend.

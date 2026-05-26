# modelc

**modelc** is a Rust-based CLI that packages model files into a **single, self-contained artifact** optimized for size, footprint, and performance. It supports multiple model formats and provides an inference experience similar to **Ollama**, **vLLM**, and **SGLang** — run models locally with minimal setup.

## Why modelc

- **Single-file packaging.** One file contains all model data and can be loaded directly by the CLI. No external weight trees, no path coordination, no "wrong checkpoint" drift.
- **Size and performance optimized.** The packaged format is designed for minimal footprint and fast loading. Parsing and tensor layout happen at package time, so runtime overhead is low.
- **Cross-platform.** Runs on **macOS**, **Windows**, and **Linux** from the same toolchain.
- **Apple Silicon acceleration.** Native support for Apple Silicon **M-series** chips via Metal acceleration.
- **Multiple format support.** Compatible with **Safetensors**, **GGUF**, **ONNX**, and **PyTorch** checkpoints.
- **Simple operations.** `modelc run`, `modelc inspect`, `modelc compile` — a minimal command set for packaging, inspection, and serving.

**When weight files remain the better fit:** rapid A/B swaps without repackaging, very large checkpoints where embedding blows up artifacts, multitenant “one server, many paths,” or ecosystems that assume on-disk formats (mmap, GGUF loaders, ONNX Runtime with external weights).

See [SPEC.md](./SPEC.md) for scope and limits.

## Prerequisites

- [Rust](https://rustup.rs/) toolchain with `cargo` on your `PATH` (the compiler runs `cargo build` on a generated project).

## Build

Produce the **`modelc`** binary:

```bash
cargo build --release
```

Binary path: `./target/release/modelc` (or `target/debug/modelc` without `--release`). To put `modelc` on your PATH:

```bash
cargo install --path .
```

## Test

```bash
cargo test
```

## Examples

Runnable programs live under [`examples/`](./examples/); see [`examples/README.md`](./examples/README.md) for a table and quick flows. Typical invocations:

```bash
cargo run --example create_simple_model -- ./demo_mlp.safetensors
cargo run --example parse_weights -- ./demo_mlp.safetensors safetensors
cargo run --example runtime_inference
```

## Usage

Examples use the **`modelc`** command. Put it on your `PATH` with `cargo install --path .`, or after `cargo build --release` call the binary by path (see **Build**).

Inspect weights (tensor names, shapes, dtypes, sizes):

```bash
modelc inspect path/to/model.safetensors
modelc inspect path/to/file -f gguf
```

Compile to a standalone binary (default output: `<stem>_serve` next to the input):

```bash
modelc compile path/to/model.safetensors -o ./my-model-serve
```

From the built artifact in this repository (no install):

```bash
./target/release/modelc inspect path/to/model.safetensors
./target/release/modelc compile path/to/model.safetensors -o ./my-model-serve
```

`modelc --help` and `modelc --version` show subcommands and semver plus a short git revision (from `build.rs` when `.git` is present).

### `compile` networking

The generated `model-serve` binary binds to **`--bind`** (IP, default `0.0.0.0`) plus **`--port`** (default `8080`), unless **`--listen ADDR:PORT`** is set, which wins and is embedded verbatim (IPv6 literals such as `[::1]:8080` are supported).

### `compile` other flags

- **`--arch`** — optional hint (`llama`, `gpt2`, …) stored in the model and surfaced in `/info`.
- **`--format` / `-f`** — weight format when extensions or magic-byte sniffing are ambiguous.
- **`--target`** — passed through to `cargo build --target`.
- **`--debug`** — builds the generated crate with Cargo’s debug profile instead of `--release` (release is the default).

Supported input formats (`-f` when needed):

| Flag value    | Typical extensions                          |
|---------------|---------------------------------------------|
| `safetensors` | `.safetensors`                              |
| `gguf`        | `.gguf`, `.bin` with sniff / name heuristics |
| `onnx`        | `.onnx`                                     |
| `pytorch`     | `.pt`, `.pth`, name-heuristic `.bin`        |

Ambiguous files (e.g. extensionless or generic `.bin`): the CLI may **sniff** GGUF / zip (PyTorch-ish) / small Safetensors blobs (see [SPEC.md](./SPEC.md)).

## Generated server HTTP API (`model-serve`)

| Method | Path     | Body / response |
|--------|----------|-----------------|
| `GET`  | `/info`  | JSON: `name`, `architecture`, `total_params`, `total_bytes`, `tensors` (names). |
| `POST` | `/infer` | Request JSON: `{ "input": [f32, ...] }`. Response: `{ "output": [f32, ...] }`. If the embedded **`architecture`** is **`mlp`**, codegen emits a **stacked GEMV + bias (+ ReLU between hidden layers)** using `layerN.weight`/`layerN.bias` (strictly contiguous indices) or a single `weight`/`bias`; other architectures keep the lightweight input echo stub. |

Both responses are `application/json`.

## crates.io checklist

Before the first (`modelc`) publish:

1. Bump `Cargo.toml` `version`, tag `vX.Y.Z`, run `cargo publish --dry-run`.
2. Snapshot `modelc --help` / subcommand help after any CLI churn (crate README can embed the summary).
3. After the first publish, add a crates.io version badge ([shields.io versioning](https://shields.io/category/version); crate URL https://crates.io/crates/modelc once live).
4. Maintain a concise changelog (`CHANGELOG.md` optional) noting parser/format support changes.

## Format references (parsers / exports)

- Safetensors — [huggingface/safetensors](https://github.com/huggingface/safetensors)
- GGUF — [GGML GGUF notes](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- ONNX — [onnx.ai](https://onnx.ai/onnx/intro/)
- PyTorch checkpoints — accepts mislabeled standalone Safetensors bytes and Torch **ZIP** containers that nest `*.safetensors` payloads; pickled-only checkpoints should be exported to Safetensors, ONNX, or GGUF externally.

## Repository layout

- `src/` — CLI, parsers, `Model` IR, codegen, runtime helpers.
- `examples/` — runnable `cargo --example` programs ([`examples/README.md`](./examples/README.md)).
- `tests/` — integration tests.

See [SPEC.md](./SPEC.md), [ARCHITECTURE.md](./ARCHITECTURE.md), and [TODO.md](./TODO.md).

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](./LICENSE)).

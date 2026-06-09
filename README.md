# modelc

**modelc** is a Rust-based CLI that packages model files into **single, self-contained artifacts** optimized for size, footprint, and performance. It supports multiple model formats and provides an inference experience similar to **Ollama**, **vLLM**, and **SGLang** — run models locally with minimal setup.

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

### Core workflow

**Pack** models into `.modelc` single-file artifacts:

```bash
modelc pack path/to/model.safetensors -o my-model.modelc
modelc pack path/to/model.gguf --compress -o my-model.modelc
modelc pack path/to/model.safetensors --quantize int8 --prune 0.001 -o my-model.modelc
```

**Run** inference from local artifacts:

```bash
modelc run my-model.modelc
modelc run my-model.modelc --port 8080 --profile
```

**List** locally stored models:

```bash
modelc list
```

**Pull** models from remote sources:

```bash
modelc pull username/model-name
modelc pull username/model-name --version 2
```

**Chat / completion inference:**

```bash
# HTTP API: POST /chat with messages
# HTTP API: POST /complete with prompt
# HTTP API: POST /chat/stream for SSE streaming
```

### Legacy compile workflow

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

### Pack flags

- **`--compress`** — use zstd compression to reduce artifact size (produces version 2 format).
- **`--arch`** — optional hint (`llama`, `gpt2`, `mlp`, …) stored in the model.
- **`--format` / `-f`** — weight format when extensions or magic-byte sniffing are ambiguous.
- **`-o`** — output path for the `.modelc` artifact.

### Run flags

- **`--port`** — HTTP server port (default `8080`).
- **`--bind`** — HTTP server bind address (default `0.0.0.0`).

### Compile flags (legacy)

The generated `model-serve` binary binds to **`--bind`** (IP, default `0.0.0.0`) plus **`--port`** (default `8080`), unless **`--listen ADDR:PORT`** is set, which wins and is embedded verbatim (IPv6 literals such as `[::1]:8080` are supported).

Other compile flags:
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

## HTTP API (run + model-serve)

| Method | Path           | Body / response |
|--------|----------------|-----------------|
| `GET`  | `/info`        | JSON: `name`, `architecture`, `total_params`, `total_bytes`, `tensors` (names). |
| `POST` | `/infer`       | Request JSON: `{ "input": [f32, ...] }` or `{ "inputs": [[f32, ...], ...] }` for batch. Response: `{ "output": [f32, ...] }` or `{ "outputs": [[f32, ...], ...] }`. |
| `POST` | `/chat`        | Request JSON: `{ "messages": [{"role": "user", "content": "..."}] }`. Response: `{ "message": {"role": "assistant", "content": "..."} }`. |
| `POST` | `/chat/stream` | SSE stream of `{ "delta": "...", "done": bool }` chunks. |
| `POST` | `/complete`    | Request JSON: `{ "prompt": "..." }`. Response: `{ "completion": "..." }`. |

**Inference backends** (priority order):
1. **ONNX execution plan** — if the model metadata contains `onnx.execution_plan`, ops are executed via the runtime tensor engine (MatMul, Gemm, Add, Mul, Div, Sub, Relu, Softmax, LayerNorm, Transpose, Reshape, Sigmoid, Tanh, Identity, Cast).
2. **MLP GEMV** — when `architecture == "mlp"`, emits a stacked GEMV + bias (+ ReLU between hidden layers) using `layerN.weight`/`layerN.bias` or a single `weight`/`bias`.
3. **Echo fallback** — returns input unchanged when no execution plan matches.

All JSON responses are `application/json`; SSE uses `text/event-stream`.

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

## Features

- **ONNX graph execution** — parses ONNX graph nodes into an execution plan (MatMul, Gemm, Add, Mul, Div, Sub, Relu, Softmax, LayerNorm, Transpose, Reshape, Sigmoid, Tanh, Identity, Cast) and runs inference via the tensor runtime.
- **LoRA adapter support** — load and apply LoRA adapters on top of a base model at runtime (`src/lora.rs`).
- **INT4 quantization + weight pruning** — pack-time `--quantize int4` and `--prune <threshold>` for extreme size reduction.
- **Docker/OCI image generation** — `modelc containerize <artifact>` emits a minimal Dockerfile + entrypoint.
- **Model versioning** — store multiple versions and `modelc switch <name> <version>`.
- **Shell completions** — generate bash/zsh/fish completions for all subcommands.
- **Per-op profiling** — `--profile` flag on `run` prints timing per inference step.

## Repository layout

- `src/` — CLI, parsers, `Model` IR, codegen, runtime helpers, ONNX execution engine.
  - `src/parsers/` — format-specific parsers, modularized by format (`gguf/`, `onnx/` subdirectories).
  - `src/onnx_exec/` — ONNX graph execution plan builder and executor (`mod.rs`, `helpers.rs`).
  - `src/codegen/native/` — native code generator, modularized (`mod.rs`, `forward.rs`, `helpers.rs`).
  - `src/serve/` — HTTP inference server, modularized (`mod.rs`, `handlers.rs`, `infer.rs`).
- `examples/` — runnable `cargo --example` programs ([`examples/README.md`](./examples/README.md)).
- `tests/` — integration tests.

See [SPEC.md](./SPEC.md), [ARCHITECTURE.md](./ARCHITECTURE.md), and [TODO.md](./TODO.md).

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](./LICENSE)).


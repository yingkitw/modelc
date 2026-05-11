# modelc

**modelc** compiles model weight files into standalone executable binaries that embed a **deterministic weight blob** (`embedded_weights.bin`) and expose a small HTTP API ([Axum](https://github.com/tokio-rs/axum) / Tokio).

## Why modelc (vs. loading weights from disk)

A **weight-file** workflow keeps checkpoints on disk (or blob storage): a generic runtime parses the format, memory-maps tensors, then runs inference. That is flexible—swap files without rebuilding—but you always coordinate **runner + paths + versioning + mounts**.

**modelc** favors the opposite trade‑off:

- **Single deployable artifact.** One executable embeds that snapshot’s weights at compile time (`embedded_weights.bin` in the generated crate). Fewer broken paths and “wrong checkpoint on prod” drift.
- **Reproducible bundles.** Build once; the binary ties together weights, embedded listen address, and metadata (`/info`) for a frozen snapshot auditors and CI can pin.
- **Simpler operations.** Serve a minimal HTTP binary without shipping a separate weight tree next to every replica (you still pay image/binary size instead of mounting files).
- **Clear boundary.** Parsing and tensor layout happen at **compile** time; generated `model-serve` stays small (**stacked GEMV/ReLU when `architecture` is `mlp`**, echo stub otherwise).

**When weight files remain the better fit:** rapid A/B swaps without rebuilds, very large checkpoints where embedding blows up images, multitenant “one server, many paths,” or ecosystems that assume on-disk formats (mmap, GGUF loaders, ONNX Runtime with external weights).

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
- `examples/` — usage examples.
- `tests/` — integration tests.

See [SPEC.md](./SPEC.md), [ARCHITECTURE.md](./ARCHITECTURE.md), and [TODO.md](./TODO.md).

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](./LICENSE)).

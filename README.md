# modelc

**modelc** compiles model weight files into standalone executable binaries that embed the weights and expose a small HTTP serving surface ([Axum](https://github.com/tokio-rs/axum) / Tokio).

## Prerequisites

- [Rust](https://rustup.rs/) toolchain with `cargo` on your `PATH` (compilation invokes `cargo build` on a generated project).

## Build

```bash
cargo build --release
```

Run the CLI from the project root:

```bash
cargo run --release -- --help
```

## Test

```bash
cargo test
```

## Usage

Inspect weights (tensor names, shapes, dtypes, sizes):

```bash
cargo run --release -- inspect path/to/model.safetensors
# or specify format explicitly:
cargo run --release -- inspect path/to/file -f gguf
```

Compile to a standalone binary (default output: `<stem>_serve` next to the input):

```bash
cargo run --release -- compile path/to/model.safetensors -o ./my-model-serve
```

Supported input formats (use `-f` / `--format` when auto-detection is ambiguous):

| Flag value   | Typical extensions        |
|-------------|---------------------------|
| `safetensors` | `.safetensors`          |
| `gguf`      | `.gguf`, some `.bin`      |
| `onnx`      | `.onnx`                   |
| `pytorch`   | `.pt`, `.pth`, `.bin`     |

Options on `compile` include `--port` (default `8080` for the generated server), `--target` for cross-compilation, and `--release` (defaults to release builds).

## Repository layout

- `src/` — CLI, parsers, internal `Model` representation, codegen, and library runtime helpers used by tests and examples.
- `examples/` — small programs demonstrating model creation, inspection, and runtime usage.
- `tests/` — integration and unit tests.

For product intent and module structure, see [SPEC.md](./SPEC.md) and [ARCHITECTURE.md](./ARCHITECTURE.md). Planned work is tracked in [TODO.md](./TODO.md).

## License

Add a `LICENSE` file if you intend to publish this crate.

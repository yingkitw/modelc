# ARCHITECTURE — modelc

## High-level pipeline

```
weights file → WeightParser → Model → NativeCodegen → temp crate → cargo build → model-serve binary
```

The **compiler** orchestrates parsing, codegen, and the external `cargo` invocation. The **CLI** (`clap`) dispatches `compile` and `inspect`.

## Crate layout

| Module / path | Responsibility |
|---------------|----------------|
| `src/main.rs` | Binary entry; parses CLI and calls `compiler`. |
| `src/cli.rs` | `Cli`, `Commands`, `WeightFormat`, `ModelArch`; format detection by extension. |
| `src/lib.rs` | Re-exports `cli`, `codegen`, `compiler`, `model`, `parsers`, `runtime`. |
| `src/model.rs` | Canonical `Model`, `TensorData`, `DataType`; size and serialization helpers. |
| `src/compiler.rs` | Parser selection, `compile`, `inspect`; temp dir + `cargo` subprocess. |
| `src/parsers/` | Format-specific parsers implementing `WeightParser` (`mod.rs`, `safetensors`, `gguf`, `onnx`, `pytorch`). |
| `src/codegen/` | `CodeGenerator` trait; `native.rs` writes `Cargo.toml` + `main.rs` for `model-serve`. |
| `src/runtime/` | Library-side runtime: `tensor`, `ops`, `serve`; used by tests/examples and conceptually mirrors ideas in generated servers. |

## Key abstractions

### `WeightParser`

Each format parser loads a file path and returns `Model`. The compiler picks a parser from `WeightFormat` or fails if the format cannot be detected.

### `CodeGenerator`

`NativeCodegen::generate` takes `&Model`, original weights path, output directory, and port. It:

1. Creates `modelc_build/` with `src/`.
2. Copies the weights file beside the generated manifest.
3. Writes `Cargo.toml` and `main.rs` with embedded tensor metadata and server wiring.

### `Model` as IR

All parsers normalize into the same structure so codegen and tests can be format-agnostic at the IR boundary.

## Build and I/O boundaries

- **Parsing:** read-only access to user-provided paths; failures surface as `anyhow::Context`.
- **Codegen:** writes only under the provided temp/build directory (except final binary copy to user `-o` path).
- **Subprocess:** `compiler` shells out to `cargo`; no embedded rustc API.

## Testing strategy

- **Unit / integration:** `tests/` covers CLI, parsers, codegen, compiler e2e, runtime pieces.
- **Examples:** `examples/` for manual exploration and documentation by example.

## Extension points

- New formats: implement `WeightParser`, register in `parsers/mod.rs` and `WeightFormat` / `get_parser`.
- New backends: additional `CodeGenerator` impls (only `NativeCodegen` today).
- Richer inference: extend `runtime` and align generated `main.rs` templates in `codegen/native.rs`.

## Dependencies (compiler crate)

- **clap** — CLI.
- **serde / serde_json** — model serialization.
- **anyhow** — error context.
- **safetensors** — Safetensors parsing.
- **byteorder** — binary layout helpers where needed.
- **tempfile** — isolated codegen builds.

Generated `model-serve` crates add **axum** and **tokio** for the HTTP server.

## Design principles

- Keep the **IR** (`Model`) stable and small; push format quirks into parsers.
- Keep **codegen** dumb but auditable: readable generated Rust for debugging.
- Prefer **explicit** format flags when detection is ambiguous.

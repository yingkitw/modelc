# TODO

Actionable work for **modelc**, kept in sync with the codebase.

## Product / CLI

- [ ] Wire `--arch` / `ModelArch` from `compile` into parsing or codegen (flag exists in `cli.rs` but is not passed through `main` / `compiler::compile` yet).
- [ ] Document the generated binary’s HTTP API (routes, request/response JSON) in README or SPEC once stable.
- [ ] Add `--version` / build metadata output where useful for reproducible builds.

## Parsers

- [ ] Harden format auto-detection for ambiguous extensions (e.g. generic `.bin`).
- [ ] Align each parser’s `architecture` / `metadata` fields with real model cards where possible (today some paths default to `"unknown"`).

## Codegen

- [ ] Review generated `model-serve` tensor loading: offsets currently may not reflect real file layout for all formats; verify against large real checkpoints.
- [ ] Optional: configurable listen address (not only `--port`).

## Runtime

- [ ] Extend `Runtime::from_raw` beyond `F32` where needed for target models.
- [ ] Expose additional ops or inference paths as the compiler grows (see `src/runtime/ops.rs`).

## Quality

- [ ] Add CI (e.g. `cargo fmt`, `clippy`, `cargo test`) on push/PR.
- [ ] Add `LICENSE` and fill in crate `license` field in `Cargo.toml` if open-sourcing.

## Docs

- [ ] Keep README install/usage in sync when CLI or defaults change.
- [ ] Link to upstream format specs (Safetensors, GGUF, ONNX, PyTorch) from SPEC or README for parser maintainers.

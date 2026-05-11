# Examples

Run these with `cargo run --example <name> -- [args…]` from the crate root (`--` separates Cargo flags from the example’s CLI).

| Example | Purpose |
|---------|---------|
| [`parse_weights`](./parse_weights.rs) | Load a weights file via the library parsers (`safetensors`, `gguf`, `onnx`, `pytorch` ZIP bundles / Safetensors mislabeled `.pt`). Prints tensor summary. |
| [`create_simple_model`](./create_simple_model.rs) | Writes a tiny **`mlp`-shaped** Safetensors file (`layer*.weight` / `layer*.bias`) for `modelc inspect` / `compile`. |
| [`inspect_model`](./inspect_model.rs) | Thin wrapper around `modelc::compiler::inspect` (same output style as `modelc inspect`). |
| [`runtime_inference`](./runtime_inference.rs) | Builds tensors in memory and runs **`Runtime` + ops** (`linear`, `relu`, `softmax`) without parsing a file. |

### Quick flows

Generate a toy checkpoint and inspect it:

```bash
cargo run --example create_simple_model -- ./demo_mlp.st
cargo run --example parse_weights -- ./demo_mlp.st safetensors
# or after `cargo install --path .`:
modelc inspect ./demo_mlp.st --arch mlp
```

Exercise the ONNX external-weights path requires a separate `.bin` plus an `.onnx` that references it; unit tests under `src/parsers/onnx.rs` build minimal models programmatically—you can mirror that pattern when exporting real models.

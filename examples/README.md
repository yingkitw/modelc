# Examples

Run these with `cargo run --example <name> -- [args…]` from the crate root (`--` separates Cargo flags from the example’s CLI).

| Example | Purpose |
|---------|---------|
| [`parse_weights`](./parse_weights.rs) | Load a weights file via the library parsers (`safetensors`, `gguf`, `onnx`, `pytorch` ZIP bundles / Safetensors mislabeled `.pt`). Prints tensor summary. |
| [`create_simple_model`](./create_simple_model.rs) | Writes a tiny **`mlp`-shaped** Safetensors file (`layer*.weight` / `layer*.bias`) for `modelc inspect` / `compile`. |
| [`inspect_model`](./inspect_model.rs) | Thin wrapper around `modelc::compiler::inspect` (same output style as `modelc inspect`). |
| [`runtime_inference`](./runtime_inference.rs) | Builds tensors in memory and runs **`Runtime` + ops** (`linear`, `relu`, `softmax`) without parsing a file. |
| [`python_model`](./python_model.rs) | Compiles a **Python program** into a `.modelc` artifact via the python bridge (`modelc pack model.py` flow) and runs it as a model. |
| [`mcp_client`](./mcp_client.rs) | Spawns `modelc serve --stdio` and drives an **MCP session** (initialize → tools/list → tools/call chat) as a client, validating every response. |

### Python sample programs (`examples/python/`)

These are the test fixtures exercised by `tests/python_samples_test.rs` — read them as the cookbook for the python bridge, each with its expected compilation documented in the file:

| Sample | Demonstrates | Expected result |
|---|---|---|
| [`mlp_lists.py`](./python/mlp_lists.py) | Pure-stdlib nested lists, `layer{N}` naming | `mlp` artifact, input `[1, 2]` → `1.3` |
| [`numpy_mlp.py`](./python/numpy_mlp.py) | Numpy arrays with stdlib fallback | identical artifact either way |
| [`torch_state_dict.py`](./python/torch_state_dict.py) | `state_dict()` duck-typing (no torch import needed) | affine stack normalized to `layer{N}` |
| [`main_block.py`](./python/main_block.py) | Weights built inside `__main__` | second-pass extraction |
| [`mixed_globals.py`](./python/mixed_globals.py) | Scalars/strings skipped | exactly `weight` + `bias` harvested |
| [`int_matrix.py`](./python/int_matrix.py) | Integer weights | preserved as `I32` |
| [`invalid_ragged.py`](./python/invalid_ragged.py) | Ragged nest | fails naming the tensor |
| [`invalid_no_tensors.py`](./python/invalid_no_tensors.py) | No tensors | fails with guidance |

### Quick flows

Generate a toy checkpoint and inspect it:

```bash
cargo run --example create_simple_model -- ./demo_mlp.st
cargo run --example parse_weights -- ./demo_mlp.st safetensors
# or after `cargo install --path .`:
modelc inspect ./demo_mlp.st --arch mlp
modelc pack ./demo_mlp.st -o ./demo.modelc --arch mlp
modelc prompt ./demo.modelc "hello"          # one-shot, no HTTP server
```

Exercise the ONNX external-weights path requires a separate `.bin` plus an `.onnx` that references it; unit tests under `src/parsers/onnx.rs` build minimal models programmatically—you can mirror that pattern when exporting real models.

# TODO

Previously listed follow-ups are **done** as of this tree:

| Item | Notes |
|------|------|
| GGUF / ONNX / PyTorch parsers | `src/parsers/gguf.rs`, `onnx.rs`, `pytorch.rs` (+ `Cargo.toml`: `onnx-rs`, `zip`). |
| MLP codegen forward | [`src/codegen/native.rs`](./src/codegen/native.rs): FP32 GEMV chain when `architecture == "mlp"` and tensor naming matches; echo otherwise. |
| Compile peak RAM | Embedding writes stream via `BufWriter` (no auxiliary `Vec` blob doubling tensor RAM). |
| crates.io checklist | README “crates.io checklist” subsection. |

## Optional next steps

- GGUF: dequantize additional block types (e.g. Q4_K, Q5_0) or richer error recovery.
- ONNX: tensor-segment loaders; more dtypes without raw-only restriction.
- PyTorch: optional Python bridge or minimal pickle reconstruction (high effort).
- Broader codegen: `gpt2`, `llama`, etc., beyond stacked linear + ReLU.

# TODO

## Done

- [x] GGUF / ONNX / PyTorch parsers --- src/parsers/gguf.rs, onnx.rs, pytorch.rs (+ Cargo.toml: onnx-rs, zip).
- [x] MLP codegen forward --- src/codegen/native.rs: FP32 GEMV chain when architecture == "mlp" and tensor naming matches; echo otherwise.
- [x] Compile peak RAM --- Embedding writes stream via BufWriter (no auxiliary Vec blob doubling tensor RAM).
- [x] crates.io checklist --- README crates.io checklist subsection.
- [x] Single-file artifact format --- .modelc binary format with JSON header + raw tensor blob (src/pack.rs).
- [x] Cross-platform runtime --- dirs for platform-specific store paths; #[cfg] guards for Unix/Windows permissions.
- [x] Apple Silicon Metal acceleration --- Skeleton in src/metal.rs with metal crate (macOS only). Full GPU kernels to be implemented.
- [x] Ollama-like UX --- modelc run, modelc pull, modelc list with local model store (src/store.rs).
- [x] Size optimization --- --compress flag on pack uses zstd compression (version 2 artifact format).

## Active (aligned with PRODUCT.md)

- [ ] **High** --- Full Metal GPU kernels --- Implement compute shaders for matmul on Apple Silicon.
- [ ] **Medium** --- Quantization at pack time --- FP32 to FP16 / INT8 / Q4_0 to reduce artifact size.

## Optional next steps

- GGUF: dequantize additional block types (e.g. Q4_K, Q5_0) or richer error recovery.
- ONNX: tensor-segment loaders; more dtypes without raw-only restriction.
- PyTorch: optional Python bridge or minimal pickle reconstruction (high effort).
- Broader codegen: gpt2, llama, etc., beyond stacked linear + ReLU.
- CPU SIMD paths (AVX, NEON) for non-Metal targets.

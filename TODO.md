# TODO

## Done

- [x] GGUF / ONNX / PyTorch parsers --- src/parsers/gguf.rs, onnx.rs, pytorch.rs (+ Cargo.toml: onnx-rs, zip).
- [x] MLP codegen forward --- src/codegen/native.rs: FP32 GEMV chain when architecture == "mlp" and tensor naming matches; echo otherwise.
- [x] Compile peak RAM --- Embedding writes stream via BufWriter (no auxiliary Vec blob doubling tensor RAM).
- [x] crates.io checklist --- README crates.io checklist subsection.
- [x] Single-file artifact format --- .modelc binary format with JSON header + raw tensor blob (src/pack.rs).
- [x] Cross-platform runtime --- dirs for platform-specific store paths; #[cfg] guards for Unix/Windows permissions.
- [x] Apple Silicon Metal acceleration --- Skeleton in src/metal.rs with metal crate (macOS only). Compute shaders in src/compute/shaders.metal.
- [x] Ollama-like UX --- modelc pack, modelc run, modelc pull, modelc list with local model store (src/store.rs).
- [x] Size optimization --- --compress flag on pack uses zstd compression (version 2 artifact format).
- [x] Local model store management --- Cross-platform model directory with platform-specific paths (src/store.rs).

## Active (aligned with PRODUCT.md)

- [x] **High** --- Complete Metal GPU kernel bindings --- Bind existing shaders (softmax, layer_norm, relu, add, mul_scalar) in src/metal.rs and wire into runtime/ops.rs.
- [x] **High** --- Quantization at pack time --- Add --quantize flag (fp16, int8) to convert F32 tensors at pack time and reduce artifact size.
- [x] **Medium** --- CPU SIMD matmul --- Add AVX (x86_64) and NEON (aarch64) optimized paths for matmul_cpu fallback.
- [x] **Medium** --- Store metadata enrichment --- modelc list reads artifact headers to show architecture, parameter count, and compression status.
- [x] **Medium** --- Remote model pull --- modelc pull supports HTTP/HTTPS URLs for downloading .modelc artifacts.

## Optional next steps

- GGUF: dequantize additional block types (e.g. Q4_K, Q5_0) or richer error recovery.
- ONNX: tensor-segment loaders; more dtypes without raw-only restriction.
- PyTorch: optional Python bridge or minimal pickle reconstruction (high effort).
- Broader codegen: gpt2, llama, etc., beyond stacked linear + ReLU.

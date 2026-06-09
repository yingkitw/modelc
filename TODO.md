# TODO — Plan

## Implemented

- [x] **Parsers** — GGUF (F32/F16/BF16/I8/I16/I32/I64/F64, Q4_0/Q5_0/Q8_0/Q4_K/Q6_K dequant), ONNX (inline/external/segmented initializers), PyTorch (Safetensors-in-ZIP), Safetensors (`src/parsers/`).
- [x] **ONNX graph execution** — parses ops (MatMul, Gemm, Add, Mul, Div, Sub, Relu, Softmax, LayerNorm, Transpose, Reshape, Sigmoid, Tanh, Identity, Cast) into a JSON execution plan stored in metadata (`src/onnx_exec/`).
- [x] **MLP codegen forward** — FP32 GEMV chain when `architecture == "mlp"` (`src/codegen/native/`).
- [x] **GPT2 / LLaMA codegen skeletons** — simplified transformer forward pass stubs (`src/codegen/native/forward.rs`).
- [x] **Auto architecture inference** — detects `gpt2`, `llama`, `bert`, `mlp` from tensor naming patterns (`src/model.rs`).
- [x] **Compile peak RAM** — Embedding writes stream via `BufWriter` (no auxiliary Vec blob).
- [x] **crates.io checklist** — README subsection with publish steps.
- [x] **Single-file artifact format** — `.modelc` binary with JSON header + raw tensor blob (`src/pack.rs`).
- [x] **Cross-platform runtime** — Platform-specific store paths via `dirs` crate; `#[cfg]` Unix/Windows guards.
- [x] **Apple Silicon Metal acceleration** — Full kernels for relu, add, mul_scalar, softmax, layer_norm (`src/metal.rs`, `src/compute/shaders.metal`).
- [x] **Ollama-like UX** — `pack`, `run`, `pull`, `list` with local model store (`src/store.rs`).
- [x] **Size optimization** — `--compress` flag on `pack` uses zstd (version 2 format).
- [x] **Quantization at pack time** — `--quantize fp16/int8/int4` converts F32 tensors; INT4/INT8 auto-dequantized on `run`.
- [x] **Weight pruning** — `--prune <threshold>` zeroes small weights.
- [x] **CPU SIMD matmul** — AVX (x86_64) and NEON (aarch64) paths with runtime feature detection.
- [x] **Multi-threaded CPU ops** — `rayon` parallelization for large matmuls.
- [x] **GPU memory pressure handling** — pre-allocation checks before Metal buffer creation.
- [x] **Store metadata enrichment** — `modelc list` reads artifact headers to show architecture, params, and compression status.
- [x] **Remote model pull** — `modelc pull <url>` supports HTTP/HTTPS downloads with versioning.
- [x] **Benchmark command** — `modelc bench <artifact>` measures warm/cold latency and throughput.
- [x] **Artifact verification** — `modelc verify <artifact>` checks header integrity.
- [x] **Export to Safetensors** — `modelc export <artifact> -o out.safetensors`.
- [x] **Model search** — `modelc search <query>` filters store by architecture, size, or name.
- [x] **Config file support** — `~/.modelc/config.toml` for default bind, store path, compression.
- [x] **Shell completions** — bash/zsh/fish via `modelc completions <shell>`.
- [x] **Per-op profiling** — `--profile` flag on `run` prints per-step timing.
- [x] **Model versioning** — store multiple versions and `modelc switch <name> <version>`.
- [x] **Model card generation** — `modelc inspect --readme` generates Markdown model card.
- [x] **Docker/OCI image generation** — `modelc containerize <artifact>` emits Dockerfile + entrypoint.
- [x] **LoRA adapter support** — load and apply LoRA adapters at runtime (`src/lora.rs`).
- [x] **Chat/completion endpoints** — `/chat`, `/complete`, `/chat/stream` (SSE).
- [x] **Batch inference API** — `/infer` accepts `{"inputs": [...]}`.
- [x] **Code modularization** — split large files into focused submodules (`gguf/`, `onnx/`, `onnx_exec/`, `codegen/native/`, `serve/`).

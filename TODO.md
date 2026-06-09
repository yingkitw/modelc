# TODO — Plan

## Implemented

- [x] **Parsers** — GGUF, ONNX, PyTorch, Safetensors (`src/parsers/`).
- [x] **MLP codegen forward** — FP32 GEMV chain when `architecture == "mlp"` (`src/codegen/native.rs`).
- [x] **Compile peak RAM** — Embedding writes stream via `BufWriter` (no auxiliary Vec blob).
- [x] **crates.io checklist** — README subsection with publish steps.
- [x] **Single-file artifact format** — `.modelc` binary with JSON header + raw tensor blob (`src/pack.rs`).
- [x] **Cross-platform runtime** — Platform-specific store paths via `dirs` crate; `#[cfg]` Unix/Windows guards.
- [x] **Apple Silicon Metal acceleration** — Skeleton with `metal` crate + compute shaders (`src/metal.rs`, `src/compute/shaders.metal`).
- [x] **Ollama-like UX** — `pack`, `run`, `pull`, `list` with local model store (`src/store.rs`).
- [x] **Size optimization** — `--compress` flag on `pack` uses zstd (version 2 format).
- [x] **Complete Metal GPU kernel bindings** — `relu`, `add`, `mul_scalar`, `softmax`, `layer_norm` shaders bound and wired into `runtime/ops.rs`.
- [x] **Quantization at pack time** — `--quantize fp16/int8` converts F32 tensors to reduce artifact size; INT8 auto-dequantized on `run`.
- [x] **CPU SIMD matmul** — AVX (x86_64) and NEON (aarch64) paths with runtime feature detection.
- [x] **Store metadata enrichment** — `modelc list` reads artifact headers to show architecture, params, and compression status.
- [x] **Remote model pull** — `modelc pull <url>` supports HTTP/HTTPS downloads.

## Backlog (parser / format)

- [ ] **GGUF**: dequantize additional block types (Q4_K, Q5_0, Q6_K) or richer error recovery.
- [ ] **ONNX**: tensor-segment loaders; more dtypes without raw-only restriction.
- [ ] **PyTorch**: optional Python bridge or minimal pickle reconstruction (high effort).

## Backlog (codegen / architecture)

- [ ] **Broader codegen**: gpt2, llama, etc. — beyond stacked linear + ReLU.
- [ ] **ONNX graph execution**: parse ops and compile into runtime, not just weight extraction.
- [x] **Auto architecture inference**: detect `--arch` from tensor naming patterns (`transformer.h.*`, `model.layers.*`).

## Backlog (performance)

- [x] **INT4 quantization**: pack-time `--quantize int4` for extreme size reduction.
- [x] **Weight pruning**: `--prune <threshold>` to zero out small weights and store sparse tensors.
- [x] **Multi-threaded CPU ops**: `rayon` or thread pool for parallel CPU matmul on large matrices.
- [x] **GPU memory pressure handling**: large-model streaming or memory-mapped loading to avoid OOM on Metal.
- [x] **Q4_K / Q5_0 / Q6_K GGUF dequant**: expand parser to support more block types natively.

## Backlog (CLI / UX)

- [x] **Benchmark command**: `modelc bench <artifact>` measures warm/cold latency and throughput.
- [x] **Artifact verification**: `modelc verify <artifact>` checks header integrity and tensor blob checksums.
- [x] **Export to Safetensors**: `modelc export <artifact> -o out.safetensors` reverses packaging.
- [x] **Model search**: `modelc search <query>` filters store by architecture, size, or name pattern.
- [x] **Config file support**: `~/.modelc/config.toml` for default bind address, store path, compression level.
- [x] **Shell completions**: generate bash/zsh/fish completions for all subcommands.
- [x] **Per-op profiling**: `--profile` flag on `run` prints timing per inference step.
- [x] **Model versioning**: store multiple versions and `modelc switch <name> <version>`.
- [x] **Model card generation**: `modelc inspect --readme` generates Markdown model card from metadata.
- [x] **Docker/OCI image generation**: `modelc containerize <artifact>` emits a minimal Docker image.
- [x] **LoRA adapter support**: load and apply LoRA adapters on top of a base model at runtime.

## Backlog (HTTP API / serving)

- [x] **Chat/completion endpoints**: `/chat` and `/complete` for LLM-style streaming inference.
- [x] **Streaming inference (SSE)**: token-by-token Server-Sent Events for long-generation models.
- [x] **Batch inference API**: `/infer` accepts `{"inputs": [[...], [...]]}` and returns multiple outputs.

## Backlog (codegen / architecture)

- [x] **Auto architecture inference**: detect `--arch` from tensor naming patterns (`transformer.h.*`, `model.layers.*`).
- [x] **Broader codegen**: gpt2, llama, etc. — beyond stacked linear + ReLU.
- [x] **ONNX graph execution**: parse ops and compile into runtime, not just weight extraction.

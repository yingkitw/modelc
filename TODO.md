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
- [x] **Transformer runtime in `run`** — `modelc run` now executes GPT-2 / LLaMA forward passes directly from the in-memory runtime (`src/runtime/transformer.rs`) instead of echoing input. Structure inspection shared with codegen via `src/arch.rs`; numeric helpers mirror the emitted codegen so `run` and `compile` outputs match.

## Next / Competitive gaps

- [x] **`modelc rm <name>`** — Delete a model from the local store (and its versioned copies). Every model manager (Ollama, Docker) has this; we don't.
- [x] **OpenAI-compatible API** — `/v1/models`, `/v1/chat/completions` (non-streaming). Ollama, llama.cpp server, vLLM, and SGLang all expose this; it's the dominant integration pattern for client libraries.
- [x] **`/embeddings` endpoint** — Produce vector embeddings from the hidden-state pooler via `POST /embeddings`.
- [x] **KV cache in transformer runtime** — `KvCache` struct stores per-layer K/V vectors. `forward_gpt2_cached` and `forward_llama_cached` accept optional cache, append current K/V, and compute attention over all cached positions via `attention_kv`. Eliminates redundant K/V recomputation during autoregressive generation.
- [x] **Built-in tokenization** — Byte-level BPE tokenizer (`src/tokenizer.rs`) with encode/decode and greedy merge algorithm. Foundation for real text-in/text-out transformer inference.
- [x] **GGUF tokenizer metadata extraction** — `extract_tokenizer_metadata` reads vocab, merges, BOS/EOS IDs from GGUF KV metadata. `BpeTokenizer::from_gguf` constructs a working tokenizer from this data.
- [x] **`/health` endpoint** — Standard liveness probe for inference servers and container orchestration.
- [x] **Model deletion guard** — Optional `--force` flag on `rm`; refuse to delete the main artifact if versioned copies exist unless `--all` or `--force` is provided.

## Competitive gaps (research-driven)

- [x] **Autoregressive text generation** — `src/generate.rs` provides `generate()` with greedy and temperature sampling. Wires into `run_text_inference` for transformer models so `/chat`, `/complete`, and `/v1/chat/completions` return real generated text instead of logits JSON.
- [x] **Chat template support** — reads Jinja2 chat templates from GGUF metadata (`tokenizer.chat_template`) and applies them to format messages before tokenization via `minijinja`. Falls back to simple `role: content\n` concatenation when no template is present.
- [x] **GGUF quantization inference** — GGUF parser preserves Q4_0, Q5_0, Q8_0, Q4_K, Q6_K block-quantized bytes in IR instead of expanding to F32 at parse time. `Runtime::from_raw` dequantizes on-the-fly via `dequantize_gguf_tensor`. This reduces `.modelc` artifact size (~8x for Q4_0) while keeping inference functional. Codegen path calls `dequantize_in_place` before generating the server.
- [x] **Structured output / JSON mode** — `response_format: { type: "json_object" }` on `/v1/chat/completions` injects a system prompt instructing JSON-only output, then post-processes generated text with `extract_json_object` to extract the first well-formed JSON object or array. Falls back to raw text if no valid JSON is found.
- [x] **Function calling / tool use** — `POST /v1/chat/completions` accepts an OpenAI-compatible `tools` array. When tools are present, a system prompt describing available tools is injected. Generated output is parsed for a `tool_calls` JSON array; if found, the response returns `tool_calls` with `finish_reason: "tool_calls"`. Otherwise falls back to regular content.
- [x] **Continuous batching (MLP)** — `POST /infer` with multiple `inputs` and MLP architecture now routes to `run_mlp_forward_batched`, which computes all items in a single pass via `batched_gemv_bias`. Eliminates redundant weight traversal and improves cache locality. Transformer generation batching remains future work.
- [x] **Speculative decoding** — `generate()` supports `config.gamma > 0` to enable speculative decoding with an n-gram draft model (`src/generate.rs`). The draft model proposes `gamma` candidate tokens by looking up trigram continuations from the prefix; the target model verifies each candidate in a loop using the KV cache. Accepted tokens skip the sampling step. Establishes infrastructure for future faster draft models (e.g., smaller transformer or prompt lookup decoding).

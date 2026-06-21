# modelc

**modelc** is a Rust-based CLI that packages model files into **single, self-contained artifacts** optimized for size, footprint, and performance. It supports multiple model formats and provides an inference experience similar to **Ollama**, **vLLM**, and **SGLang** — run models locally with minimal setup.

## Why modelc

- **Single-file packaging.** One file contains all model data and can be loaded directly by the CLI. No external weight trees, no path coordination, no "wrong checkpoint" drift.
- **Size and performance optimized.** The packaged format is designed for minimal footprint and fast loading. Parsing and tensor layout happen at package time, so runtime overhead is low.
- **Cross-platform.** Runs on **macOS**, **Windows**, and **Linux** from the same toolchain.
- **Apple Silicon acceleration.** Native support for Apple Silicon **M-series** chips via Metal acceleration.
- **Multiple format support.** Compatible with **Safetensors**, **GGUF**, **ONNX**, and **PyTorch** checkpoints.
- **Simple operations.** `modelc run`, `modelc inspect`, `modelc compile` — a minimal command set for packaging, inspection, and serving.

**When weight files remain the better fit:** rapid A/B swaps without repackaging, very large checkpoints where embedding blows up artifacts, multitenant “one server, many paths,” or ecosystems that assume on-disk formats (mmap, GGUF loaders, ONNX Runtime with external weights).

See [SPEC.md](./SPEC.md) for scope and limits.

## Prerequisites

- [Rust](https://rustup.rs/) toolchain with `cargo` on your `PATH` (the compiler runs `cargo build` on a generated project).

## Build

Produce the **`modelc`** binary:

```bash
cargo build --release
```

Binary path: `./target/release/modelc` (or `target/debug/modelc` without `--release`). To put `modelc` on your PATH:

```bash
cargo install --path .
```

## Test

```bash
cargo test
```

## Examples

Runnable programs live under [`examples/`](./examples/); see [`examples/README.md`](./examples/README.md) for a table and quick flows. Typical invocations:

```bash
cargo run --example create_simple_model -- ./demo_mlp.safetensors
cargo run --example parse_weights -- ./demo_mlp.safetensors safetensors
cargo run --example runtime_inference
```

## Usage

Examples use the **`modelc`** command. Put it on your `PATH` with `cargo install --path .`, or after `cargo build --release` call the binary by path (see **Build**).

### Core workflow

**Pack** models into `.modelc` single-file artifacts:

```bash
modelc pack path/to/model.safetensors -o my-model.modelc
modelc pack path/to/model.gguf --compress -o my-model.modelc
modelc pack path/to/model.safetensors --quantize int8 --prune 0.001 -o my-model.modelc
```

**Run** inference from local artifacts:

```bash
modelc run my-model.modelc
modelc run my-model.modelc --port 8080 --profile
modelc run my-model.modelc --max-context 2048 --grammar "^\d+$"
modelc run my-model.modelc --temperature 0.7 --max-tokens 512
modelc run my-model.modelc --api-key sk-xxx --rate-limit 120
```

**List** locally stored models:

```bash
modelc list
```

**Remove** a model from the local store:

```bash
modelc rm my-model
modelc rm my-model --all          # also delete versioned copies
modelc rm my-model --force        # delete even if versioned copies exist
```

**Pull** models from remote sources:

```bash
modelc pull username/model-name
modelc pull username/model-name --version 2
```

**Chat / completion inference:**

```bash
# HTTP API: POST /chat with messages
# HTTP API: POST /complete with prompt
# HTTP API: POST /chat/stream for SSE streaming
```

### Legacy compile workflow

Inspect weights (tensor names, shapes, dtypes, sizes):

```bash
modelc inspect path/to/model.safetensors
modelc inspect path/to/file -f gguf
```

Compile to a standalone binary (default output: `<stem>_serve` next to the input):

```bash
modelc compile path/to/model.safetensors -o ./my-model-serve
```

From the built artifact in this repository (no install):

```bash
./target/release/modelc inspect path/to/model.safetensors
./target/release/modelc compile path/to/model.safetensors -o ./my-model-serve
```

`modelc --help` and `modelc --version` show subcommands and semver plus a short git revision (from `build.rs` when `.git` is present).

### Pack flags

- **`--compress`** — use zstd compression to reduce artifact size (produces version 2 format).
- **`--arch`** — optional hint (`llama`, `gpt2`, `mlp`, …) stored in the model.
- **`--format` / `-f`** — weight format when extensions or magic-byte sniffing are ambiguous.
- **`-o`** — output path for the `.modelc` artifact.

### Run flags

- **`--port`** — HTTP server port (default `8080`).
- **`--bind`** — HTTP server bind address (default `0.0.0.0`).
- **`--api-key <key>`** — Require `Authorization: Bearer <key>` on all endpoints except `/health` and `/info`.
- **`--rate-limit <N>`** — Max requests per minute per client IP (default: unlimited).
- **`--max-context <N>`** — Hard context limit; triggers sliding-window KV eviction when exceeded.
- **`--anchor-tokens <N>`** — Preserve the first N tokens during context shifting (StreamingLLM-style).
- **`--temperature <T>`** — Sampling temperature (default `1.0`).
- **`--max-tokens <N>`** — Max tokens to generate (default `256`).
- **`--top-p <P>`** — Nucleus sampling threshold (default `0.0`, disabled).
- **`--grammar <pattern>`** — Regex grammar constraint applied during sampling.
- **`--profile`** — Print per-step inference timing.

### Compile flags (legacy)

The generated `model-serve` binary binds to **`--bind`** (IP, default `0.0.0.0`) plus **`--port`** (default `8080`), unless **`--listen ADDR:PORT`** is set, which wins and is embedded verbatim (IPv6 literals such as `[::1]:8080` are supported).

Other compile flags:
- **`--arch`** — optional hint (`llama`, `gpt2`, …) stored in the model and surfaced in `/info`.
- **`--format` / `-f`** — weight format when extensions or magic-byte sniffing are ambiguous.
- **`--target`** — passed through to `cargo build --target`.
- **`--debug`** — builds the generated crate with Cargo’s debug profile instead of `--release` (release is the default).

Supported input formats (`-f` when needed):

| Flag value    | Typical extensions                          |
|---------------|---------------------------------------------|
| `safetensors` | `.safetensors`                              |
| `gguf`        | `.gguf`, `.bin` with sniff / name heuristics |
| `onnx`        | `.onnx`                                     |
| `pytorch`     | `.pt`, `.pth`, name-heuristic `.bin`        |

Ambiguous files (e.g. extensionless or generic `.bin`): the CLI may **sniff** GGUF / zip (PyTorch-ish) / small Safetensors blobs (see [SPEC.md](./SPEC.md)).

## HTTP API (run + model-serve)

| Method | Path                    | Body / response |
|--------|-------------------------|-----------------|
| `GET`  | `/info`                 | JSON: `name`, `architecture`, `total_params`, `total_bytes`, `tensors` (names). |
| `GET`  | `/health`               | JSON: `status`, `model`, `architecture` — liveness probe. |
| `POST` | `/infer`                | Request JSON: `{ "input": [f32, ...] }` or `{ "inputs": [[f32, ...], ...] }` for batch. Response: `{ "output": [f32, ...] }` or `{ "outputs": [[f32, ...], ...] }`. |
| `POST` | `/chat`                 | Request JSON: `{ "messages": [{"role": "user", "content": "..."}] }`. Response: `{ "message": {"role": "assistant", "content": "..."} }`. |
| `POST` | `/chat/stream`          | SSE stream of `{ "delta": "...", "done": bool }` chunks. |
| `POST` | `/complete`             | Request JSON: `{ "prompt": "..." }`. Response: `{ "completion": "..." }`. |
| `POST` | `/embeddings`           | Request JSON: `{ "input": "..." }`. Response: `{ "embedding": [f32, ...], "model": "..." }`. |
| `GET`  | `/v1/models`            | OpenAI-compatible model list. |
| `POST` | `/v1/chat/completions`  | OpenAI-compatible chat completion (streaming + non-streaming). |
| `POST` | `/v1/completions`       | OpenAI-compatible legacy text completion (streaming + non-streaming). |

**Inference backends** (priority order):
1. **ONNX execution plan** — if the model metadata contains `onnx.execution_plan`, ops are executed via the runtime tensor engine (MatMul, Gemm, Add, Mul, Div, Sub, Relu, Softmax, LayerNorm, Transpose, Reshape, Sigmoid, Tanh, Identity, Cast).
2. **Transformer forward** — when `architecture == "gpt2"` or `"llama"`, runs the full FP32 transformer forward (layer norm / RMS norm, GeLU / SwiGLU, RoPE, single-token causal attention, output projection). Inputs are resized to the model hidden size; numerics match the `compile`-emitted server.
3. **MLP GEMV** — when `architecture == "mlp"`, emits a stacked GEMV + bias (+ ReLU between hidden layers) using `layerN.weight`/`layerN.bias` or a single `weight`/`bias`.
4. **Echo fallback** — returns input unchanged when no execution plan matches.

All JSON responses are `application/json`; SSE uses `text/event-stream`.

## crates.io checklist

Before the first (`modelc`) publish:

1. Bump `Cargo.toml` `version`, tag `vX.Y.Z`, run `cargo publish --dry-run`.
2. Snapshot `modelc --help` / subcommand help after any CLI churn (crate README can embed the summary).
3. After the first publish, add a crates.io version badge ([shields.io versioning](https://shields.io/category/version); crate URL https://crates.io/crates/modelc once live).
4. Maintain a concise changelog (`CHANGELOG.md` optional) noting parser/format support changes.

## Format references (parsers / exports)

- Safetensors — [huggingface/safetensors](https://github.com/huggingface/safetensors)
- GGUF — [GGML GGUF notes](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- ONNX — [onnx.ai](https://onnx.ai/onnx/intro/)
- PyTorch checkpoints — accepts mislabeled standalone Safetensors bytes and Torch **ZIP** containers that nest `*.safetensors` payloads; pickled-only checkpoints should be exported to Safetensors, ONNX, or GGUF externally.

## Features

- **ONNX graph execution** — parses ONNX graph nodes into an execution plan (MatMul, Gemm, Add, Mul, Div, Sub, Relu, Softmax, LayerNorm, Transpose, Reshape, Sigmoid, Tanh, Identity, Cast) and runs inference via the tensor runtime.
- **LoRA adapter support** — load and apply LoRA adapters on top of a base model at runtime (`src/lora.rs`).
- **INT4 quantization + weight pruning** — pack-time `--quantize int4` and `--prune <threshold>` for extreme size reduction.
- **Docker/OCI image generation** — `modelc containerize <artifact>` emits a minimal Dockerfile + entrypoint.
- **Model versioning** — store multiple versions and `modelc switch <name> <version>`.
- **Shell completions** — generate bash/zsh/fish completions for all subcommands.
- **Per-op profiling** — `--profile` flag on `run` prints timing per inference step.
- **KV cache** — `KvCache` stores per-layer Key/Value vectors for autoregressive generation. `forward_gpt2_cached` / `forward_llama_cached` append current K/V and compute attention over all cached positions, eliminating redundant recomputation.
- **Autoregressive text generation** — `src/generate.rs` provides `generate()` with greedy, temperature, and top-p (nucleus) sampling, KV cache reuse, and token embedding lookup. Wires into the HTTP server so `/chat`, `/complete`, and `/v1/chat/completions` return real generated text for GPT-2 / LLaMA models. Accepts per-request `max_tokens`, `temperature`, and `top_p` overrides on all text generation endpoints.
- **Byte-level BPE tokenizer** — `src/tokenizer.rs` provides encode/decode with greedy merge algorithm. Foundation for real text-in/text-out transformer inference.
- **GGUF tokenizer metadata extraction** — reads vocab, merges, and BOS/EOS token IDs directly from GGUF KV metadata (`extract_tokenizer_metadata`). Constructs a `BpeTokenizer` from in-file tokenizer data without external files.
- **Chat template rendering** — `src/chat_template.rs` reads Jinja2 templates from GGUF metadata (`tokenizer.chat_template`) and formats messages before tokenization using `minijinja`. Falls back to simple concatenation when no template is present.
- **GGUF quantization inference** — Q4_0, Q5_0, Q8_0, Q4_K, Q6_K block-quantized tensors are preserved in IR and dequantized on-the-fly by `Runtime::from_raw` (`src/runtime/serve.rs`). Reduces `.modelc` artifact size ~8x for Q4_0 models while keeping the transformer forward pass functional. Codegen path (`compile`) calls `dequantize_in_place` before emitting the server so generated binaries still receive FP32 weights.
- **Structured output / JSON mode** — `POST /v1/chat/completions` accepts `response_format: { type: "json_object" }` to constrain output to valid JSON. Injects a system prompt requesting JSON-only responses and post-processes generated text with `extract_json_object` to extract the first well-formed JSON object or array. Falls back to raw text if no valid JSON is found.
- **Function calling / tool use** — `POST /v1/chat/completions` accepts an OpenAI-compatible `tools` array. When tools are present, a system prompt describing available tools is injected into the conversation. Generated output is parsed for a `tool_calls` JSON array; if found, the response returns `tool_calls` with `finish_reason: "tool_calls"`. Otherwise falls back to regular text content. Supports both inline JSON arguments and pre-serialized argument strings.
- **Continuous batching (MLP)** — `POST /infer` with multiple `inputs` and MLP architecture routes to `run_mlp_forward_batched` (`src/serve/infer.rs`), which computes the entire batch in a single pass via `batched_gemv_bias`. Eliminates redundant weight traversal and improves CPU cache locality compared to serial per-item inference. Transformer generation batching is future work.
- **Speculative decoding** — `generate()` supports `config.gamma > 0` to enable speculative decoding with an n-gram draft model (`src/generate.rs`). The draft model proposes `gamma` candidate tokens by looking up trigram continuations from the prefix; the target model verifies each candidate in a loop using the KV cache. Accepted tokens skip the sampling step. Now supports prefix caching (restores cached K/V before priming), context shifting (`max_context`), and grammar constraints (`sample_constrained`). Disabled by default (`gamma: 0`). Establishes infrastructure for future faster draft models (e.g., smaller transformer or prompt lookup decoding).
- **Streaming token generation** — `/chat/stream` and `/v1/chat/completions` with `stream: true` generate token IDs via `generate_token_ids()`, then decode and emit each token incrementally through SSE. After each new token, the cumulative sequence is decoded and only the text delta is sent. The OpenAI-compatible stream emits `chat.completion.chunk` objects with `delta.role` on the first chunk and `delta.content` per token, ending with `finish_reason: "stop"` and a `[DONE]` sentinel. Non-transformer models fall back to a single chunk.
- **Batched embeddings** — `POST /embeddings` accepts `inputs: ["...", "..."]` and returns `embeddings: [{"embedding": [...], "index": 0}, ...]`. Single-input `input: "..."` remains backward compatible and returns `embedding: [...]`.
- **Token-level logprobs** — `POST /v1/chat/completions` accepts OpenAI-compatible `logprobs: true` and `top_logprobs: N` (0–20). `generate_with_logprobs()` (`src/generate.rs`) records `ln(P(token))` from the raw pre-temperature softmax plus up to N top alternatives per position, so logprobs are deterministic regardless of sampling temperature. The response carries `choices[].logprobs.content` with per-token `token`, `logprob`, raw UTF-8 `bytes`, and `top_logprobs` entries. The field is `null` when not requested (matching OpenAI); non-transformer models return an empty `content` array.
- **Quantized KV cache** — `KvLayer::Int8` (`src/runtime/transformer.rs`) stores per-token K/V vectors as INT8 with per-token scales, giving ~4× memory reduction vs FP32. `KvCache::new_quantized(n_layers, hidden, use_int8)` selects INT8 or FP32 at creation; `GenerationConfig.use_int8_kv` wires the choice through `generate_token_ids`, `generate_with_logprobs`, and `speculative_generate`. Dequantization is transparent via `k_all`/`v_all`, so attention and generation logic is unchanged.
- **Prefix caching** — `PrefixCache` (`src/prefix_cache.rs`) stores `KvCache` snapshots keyed by token sequence. Longest-prefix matching lets later requests whose prompt starts with a cached sequence restore the saved K/V and only process the divergent suffix. Wired into `generate_core`, `generate_token_ids_with_cache`, and `generate_with_logprobs_with_cache`. The serve layer locks a per-state cache and passes it through. LRU eviction bounds memory to 32 entries.
- **LoRA loading at runtime via HTTP** — `POST /lora/load { path: "...", alpha: 1.0 }` clones the base model tensors, applies a Safetensors LoRA adapter (`src/lora.rs`), and atomically swaps the runtime via `RwLock<Runtime>`. `POST /lora/unload` restores the base model from the stored `base_tensors` without restarting the server.
- **Grammar-based constrained decoding** — `GenerationConfig.constraint` holds an optional `Arc<dyn Constraint>`. `RegexConstraint` (`src/constraint.rs`) wraps a regex pattern and masks invalid tokens to `-inf` before sampling. The `sample_constrained` helper in `generate.rs` decodes the partial output, builds a per-token validity mask, and applies it. HTTP endpoints (`/chat`, `/complete`, `/v1/chat/completions`) accept an optional `grammar` string (regex pattern) per request. The heuristic is permissive (tries common suffixes to avoid false negatives) rather than exact, which is acceptable for a minimal implementation.
- **Prometheus metrics** — `GET /metrics` renders request counts, inference latency histograms (buckets: 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, +Inf), active request gauge, and tokens-generated counter in Prometheus text format. Handlers use `ActiveRequestGuard` and `InferenceTimer` RAII structs so instrumentation is automatic and zero-overhead when not scraped.
- **JSON Schema constrained generation** — `POST /chat`, `/complete`, and `/v1/chat/completions` accept an optional `json_schema` object per request. Generated text is parsed as JSON and validated against the schema using the `jsonschema` crate. If invalid, generation is retried up to 3 times with increasing temperature to encourage diversity. More robust than the post-hoc `json_object` mode (`response_format: { type: "json_object" }`), which still works for backward compatibility.
- **Context shifting (sliding window KV cache)** — `GenerationConfig.max_context` sets a hard context limit. When total tokens exceed it during generation, `generate_core` shifts both the `KvCache` (`KvLayer::shift` removes oldest K/V vectors from every layer) and `token_ids` left, keeping roughly the newest 75%. `prompt_len` is adjusted so generation continues seamlessly. Enables effectively infinite context length at the cost of losing distant history.
- **Mixed-precision KV cache** — `KvLayer::Mixed` stores recent tokens in FP32 and automatically moves older tokens to INT8 cold storage when the hot window (default 64 tokens) overflows. Preserves accuracy for the actively attended context while reducing memory footprint vs pure FP32. Compatible with prefix caching, context shifting, and speculative decoding.
- **Anchor token preservation (StreamingLLM-style)** — `GenerationConfig.anchor_tokens` (default 0, set via `--anchor-tokens N`) preserves the first N initial tokens during context shifting. The first few tokens act as "attention sinks"; evicting them degrades generation quality. `KvLayer::shift_anchored` removes tokens from the middle (after anchors) rather than from the beginning. Works with all KV cache variants (FP32, INT8, Mixed).
- **Concurrent transformer generation** — The prefix cache was a `Mutex` that serialized all transformer requests. Refactored to `RwLock` with brief lock hold times: read-lock for lookup, release during generation, write-lock for insertion. Multiple transformer requests now run in parallel, limited only by CPU cores. Covered by `run_server_handles_concurrent_transformer_requests` test.
- **Attention allocation optimization** — `attention_kv` previously allocated ~3×n_heads buffers per attention call (e.g., ~432 allocations per token for a 12-layer, 12-head model). Rewrote to use a single pre-allocated scratch buffer reused across heads, plus `softmax_inplace` with zero allocations. Eliminates the dominant allocation hotspot in the transformer forward pass.
- **OpenAI `/v1/completions` endpoint** — Legacy (non-chat) completions API: `POST /v1/completions { model, prompt, max_tokens, temperature, top_p, stop, stream }`. Returns `text_completion` objects with `choices[].text`. Supports streaming SSE, `logprobs`, `top_logprobs`, `grammar`, and `json_schema`.
- **EAGLE3 / neural speculative decoding** — `src/draft.rs` introduces a `DraftModel` trait and an `MlpDraftModel`: a tiny 2-layer MLP (embedding → FC1 + ReLU → FC2 → logits) that is orders of magnitude faster than the full transformer. Falls back to n-gram draft when no neural draft tensors are present. Wired into all serve endpoints automatically.
- **API key authentication + rate limiting** — `modelc run` accepts `--api-key <key>` (requires `Authorization: Bearer <key>` on all endpoints except `/health` and `/info`) and `--rate-limit <N>` (max requests per minute per client IP). Returns standard HTTP status codes (`401 Unauthorized`, `429 Too Many Requests`).
- **Stop sequences** — `GenerationConfig.stop` holds a list of strings that halt generation when any appear in the decoded output. Checked after each token; output is truncated to just before the matched sequence. Wired through all text generation endpoints.

## Repository layout

- `src/` — CLI, parsers, `Model` IR, codegen, runtime helpers, ONNX execution engine.
  - `src/parsers/` — format-specific parsers, modularized by format (`gguf/`, `onnx/` subdirectories).
  - `src/onnx_exec/` — ONNX graph execution plan builder and executor (`mod.rs`, `helpers.rs`).
  - `src/codegen/native/` — native code generator, modularized (`mod.rs`, `forward.rs`, `helpers.rs`).
  - `src/serve/` — HTTP inference server, modularized (`mod.rs`, `handlers.rs`, `infer.rs`).
  - `src/tokenizer.rs` — byte-level BPE tokenizer (encode/decode with greedy merge algorithm).
  - `src/chat_template.rs` — Jinja2 chat template rendering for message formatting before tokenization.
- `examples/` — runnable `cargo --example` programs ([`examples/README.md`](./examples/README.md)).
- `tests/` — integration tests.

See [SPEC.md](./SPEC.md), [ARCHITECTURE.md](./ARCHITECTURE.md), and [TODO.md](./TODO.md).

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](./LICENSE)).


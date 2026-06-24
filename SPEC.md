# SPEC — modelc

## Purpose

**modelc** is a Rust-based command-line tool and library that:

1. **Loads** neural network **weights** from supported on-disk formats (Safetensors, GGUF, ONNX, PyTorch).
2. **Normalizes** them into an internal **`Model`** IR (named tensors with shape, dtype, and raw bytes).
3. **Packages** them into a **single optimized artifact** that contains all model data and can be loaded by the CLI for inference.
4. **Runs** inference locally with an HTTP API, optimized for size, footprint, and performance.

The primary user outcome: a **single file** per model that can be inspected, moved, and run without Python or the original framework at runtime — similar to the experience of Ollama, vLLM, or SGLang.

## Non-goals (current scope)

- Full graph capture and arbitrary framework-accurate inference for every ONNX or PyTorch export (parsers focus on **weights** into the IR; ONNX execution covers a core subset of ops; `run` emits real FP32 forwards for **MLP** and **GPT-2 / LLaMA** transformers — single-vector, single-token — otherwise falls back to the ONNX plan or echoes input).
- Training or fine-tuning.
- Replacing general-purpose inference servers (e.g. full ONNX Runtime) for models outside the supported op set.

## User-facing commands

### `pack`

**Input:** path to weights file; optional `-f`, `-o`, `--arch`, `--compress`, `--quantize fp16|int8|int4`, `--prune <threshold>`.

**Output:** `.modelc` single-file artifact containing JSON header + tensor blob (optionally compressed / quantized / pruned).

**Side effects:** writes artifact file to specified path.

**Errors:** parse failure; invalid output path; compression failure.

### `run`

**Input:** path to `.modelc` artifact (or uses default store); optional `--port`, `--bind`, `--profile`.

**Output:** HTTP server listening for inference requests.

**Side effects:** loads artifact into memory, starts HTTP server.

**Errors:** missing artifact; invalid artifact format; server bind failure.

### `list`

**Input:** none (uses default model store).

**Output:** stdout list of locally stored `.modelc` artifacts with metadata.

**Errors:** store access failure.

### `rm`

**Input:** model name; optional `--all` to also delete versioned copies (`<name>.v<N>.modelc`); optional `--force` to delete the main artifact even when versioned copies exist.

**Output:** confirmation of deleted files.

**Side effects:** removes files from the local store.

**Errors:** model not found in store; versioned copies exist (without `--all` or `--force`); filesystem failure.

### `pull`

**Input:** model identifier (e.g., `username/model-name`); optional `--version`.

**Output:** downloaded `.modelc` artifact stored locally.

**Errors:** network failure; invalid model identifier.

### `search`

**Input:** query string.

**Output:** filtered list of models matching name or architecture.

### `bench`

**Input:** path to `.modelc` artifact.

**Output:** warm/cold latency and throughput measurements.

### `containerize`

**Input:** path to `.modelc` artifact; optional `--base-image`.

**Output:** Dockerfile + entrypoint.sh in target directory.

### `switch`

**Input:** model name, version number.

**Output:** activates specified version in store.

### `inspect`

**Input:** path to a weights file; optional `-f` / `--format`.

**Output:** stdout summary — model name, architecture string, format name, parameter count, total size, sorted tensor list, optional metadata (`__metadata__` from Safetensors is merged when present).

**Errors:** missing or undetectable format; parse failure.

**`--quant-sizes`** — instead of the normal summary, print a quantization size preview: the total byte size the model would occupy under each format (fp32, fp16, int8, int4, q4_0) without actually quantizing. Sizes are computed from element counts (shape-based), so they are accurate regardless of the tensors' current dtype. INT4 packs two signed nibbles per byte; Q4_0 uses 18 bytes per block of 32 elements with per-tensor block rounding. Shows each format's size in MB and the percentage change vs the current stored size.

### `compile` (legacy)

**Input:** path to weights; optional `-f`, `-o`, `--arch`, `--bind`, `--port`, **`--listen ADDR:PORT`** (overrides bind+port), `--target`, **`--debug`** (use debug profile on the emitted crate instead of `--release`).

**Output:** path to the built executable (copied from the generated crate’s `target/{...}/model-serve`).

**Side effects:** temp directory, generated `modelc_build` project, `cargo build`, copy binary, chmod on Unix.

**Errors:** parse failure; invalid `--listen` / `--bind`; `cargo` failure.

**Version line:** each `compile` invocation prints `modelc <semver> (git <sha>)` to stderr (semver from Cargo, SHA from `build.rs`, or `unknown` without git).

## Supported weight formats (declared)

| Format       | Parser status |
|-------------|---------------|
| Safetensors | Implemented (`safetensors` crate). |
| GGUF        | Implemented for **F32/F16/BF16/F64/I8/I16/I32/I64** and contiguous integer blobs; **Q4_0, Q5_0, Q8_0, Q4_K, Q6_K** blocks are **dequantized to F32** in IR. Unsupported quant types still error with a named type hint. |
| ONNX        | Implemented for **initializer** tensors: inlined `raw_data` / typed fields, **external** payloads (`external_data` with `location`/`offset`/`length`), and **segmented** initializers. Also parses the graph into an `ExecutionPlan` stored as `onnx.execution_plan` metadata. |
| PyTorch     | Implemented for **Safetensors-in-ZIP** (and standalone Safetensors mislabeled `.pt`/`.pth`); pickle-only checkpoints need export outside `modelc`. |

## Format detection

1. **Extension** (`.safetensors`, `.gguf`, `.onnx`, `.pt`, `.pth`, heuristics on `.bin`).
2. **Sniffing** when needed: `GGUF` prefix, zip `PK\x03\x04` (common for torch archives), or full-file Safetensors parse for files **≤ 64 MiB** when the extension is ambiguous or missing.
3. Otherwise detection fails and the user must pass `-f`.

## Internal model contract

- **`Model`**: `name`, `architecture`, `tensors: HashMap<String, TensorData>`, `metadata: HashMap<String, String>`.
- **`TensorData`**: `shape`, `dtype` (`DataType` enum), `data: Vec<u8>`.

Serialization via `serde` for tests and tooling.

## Artifact formats

### `.modelc` (primary)

Single-file binary format with:
- **JSON header** — model metadata (name, architecture, version, compression flag)
- **Tensor blob** — concatenated tensor data in sorted order
- **Compression** — optional zstd compression (version 2 format)

Created by `pack`, consumed by `run`.

### `model-serve` binary (legacy)

- Rust edition **2021** in the emitted crate; **axum**, **tokio**, **serde**, **serde_json**.
- Static bytes: `include_bytes!("../embedded_weights.bin")` built from the IR (sorted tensor names, streamed concatenation straight to disk) so `byte_offset` / `byte_len` match the blob **without retaining a second full in-memory blob copy** during codegen.
- **Listen address** parsed at runtime from a string literal produced from `modelc compile` (`--listen` or `--bind` + `--port`).

### HTTP API (both `run` and `model-serve`)

- **`GET /info`** — JSON object: `name`, `architecture`, `total_params`, `total_bytes`, `tensors` (array of tensor name strings).
- **`GET /health`** — JSON object: `status` (`"ok"`), `model`, `architecture`. Liveness probe for load balancers and orchestration.
- **`POST /infer`** — request JSON `{ "input": number[] }` (`f32`) or `{ "inputs": [[...], [...]] }` for batch. Response `{ "output": number[] }` or `{ "outputs": [[...], [...]] }`.
  - **Batch optimization (MLP)** — when `inputs` has multiple items and the model architecture is `"mlp"`, the server routes to `run_mlp_forward_batched`, which computes the entire batch in a single pass via `batched_gemv_bias`. This eliminates redundant weight traversal and improves CPU cache locality compared to serial per-item inference. Non-MLP models fall back to serial execution.
- **`POST /chat`** — request JSON `{ "messages": [{"role": "...", "content": "..."}], "max_tokens": 128, "temperature": 0.0, "top_p": 0.0, "min_p": 0.0 }`, response `{ "message": {"role": "assistant", "content": "..."} }`. Messages are formatted through the model's chat template (if present) before tokenization; otherwise concatenated as `role: content\n`. `max_tokens`, `temperature`, `top_p`, and `min_p` are optional and override server defaults.
- **`POST /chat/stream`** — SSE stream of `{ "delta": "...", "done": bool }` chunks. Same template logic as `/chat`. For transformer models, each token is decoded and emitted as it is generated, so the client receives text incrementally. For non-transformer models, the full response is sent in a single chunk. Accepts optional `max_tokens`, `temperature`, `top_p`, and `min_p`. If the client disconnects, the server trips a cooperative cancellation flag (`CancelOnDrop`) that aborts the generation loop after the current token instead of running to completion.
- **`POST /complete`** — request JSON `{ "prompt": "...", "max_tokens": 128, "temperature": 0.0, "top_p": 0.0, "min_p": 0.0 }`, response `{ "completion": "..." }`. `max_tokens`, `temperature`, `top_p`, and `min_p` are optional.
- **`POST /embeddings`** — request JSON `{ "input": "..." }` (single) or `{ "inputs": ["...", "..."] }` (batch). Single response: `{ "embedding": [f32, ...], "model": "..." }`. Batch response: `{ "embeddings": [{"embedding": [f32, ...], "index": 0}, ...], "model": "..." }`. Uses the final hidden state (after layer norm / RMS norm, before the output head) for transformer models; echo fallback for others.
- **`POST /tokenize`** — encode text to token IDs using the model's byte-level BPE tokenizer (the same one `/chat` and `/complete` use). Request JSON `{ "input": "..." }` (single) or `{ "inputs": ["...", "..."] }` (batch). Single response: `{ "tokens": [id, ...], "count": N }`. Batch response: `{ "tokens_batch": [[id, ...], ...], "count": N }` where `count` is the total across all inputs. Useful for prompt budgeting and prompt management.
- **`GET /v1/system`** — best-effort system/hardware info for orchestration and debugging. JSON: `model`, `architecture`, `total_params`, `total_bytes`, `cpu_cores` (logical cores via `available_parallelism`), `os`, `cpu_arch`, `pointer_width`, `metal_available` (true on Apple Silicon), and `memory_total_bytes` (`null` if unavailable; reads `/proc/meminfo` on Linux and `sysctl hw.memsize` on macOS — no added dependencies).
- **`POST /lora/load`** — request JSON `{ "path": "/path/to/adapter.safetensors", "alpha": 1.0 }`. Clones the base model tensors, applies the Safetensors LoRA adapter (`src/lora.rs`), and atomically swaps the runtime via `RwLock<Runtime>`. Response `{ "applied": <usize>, "skipped": <usize>, "message": "..." }`. On failure (e.g., file not found), `applied` and `skipped` are 0 and `message` contains the error.
- **`POST /lora/unload`** — no body. Restores the base model from the stored `base_tensors` without restarting the server. Response `{ "message": "LoRA unloaded; base model restored" }`.
- **`GET /metrics`** — Prometheus text exposition format. Returns `modelc_requests_total` (counter), `modelc_inference_duration_seconds` (histogram with buckets 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, +Inf), `modelc_active_requests` (gauge), and `modelc_tokens_generated_total` (counter). Handlers use RAII `ActiveRequestGuard` and `InferenceTimer` structs for automatic tracking.
- **Authentication & rate limiting** — `modelc run` accepts `--api-key <key>` and `--rate-limit <N>` (requests per minute per client IP). When `--api-key` is set, every request except `/health` and `/info` must include an `Authorization: Bearer <key>` header; missing or incorrect tokens return `401 Unauthorized`. When `--rate-limit` is set, a per-IP token bucket (`src/serve/auth.rs`) enforces the cap; exceeded requests return `429 Too Many Requests`. Both protections can be active simultaneously. `/health` and `/info` are always exempt so load-balancer liveness probes continue to work.
- **`POST /chat`**, **`POST /complete`**, **`POST /v1/chat/completions`**, **`POST /v1/completions`** — all accept an optional `grammar` string (regex pattern) per request. When provided, `RegexConstraint` (`src/constraint.rs`) masks tokens whose decoded text would violate the regex to `-inf` before sampling. The heuristic tries common suffixes to avoid false negatives; it is permissive rather than exact. If the regex is invalid, the constraint is silently ignored.
- **`POST /chat`**, **`POST /complete`**, **`POST /v1/chat/completions`**, **`POST /v1/completions`** — also accept an optional `json_schema` object per request. Generated text is parsed as JSON and validated against the schema using the `jsonschema` crate. If invalid, generation is retried up to 3 times with increasing temperature to encourage diversity. More robust than the post-hoc `json_object` mode. The `json_schema` and `grammar` fields can be used together (schema validates the final output, grammar constrains individual tokens).
- **Sampling penalties** (`POST /chat`, `/complete`, `/v1/chat/completions`, `/v1/completions`) — three optional parameters discourage token repetition by adjusting logits before sampling:
  - **`repetition_penalty`** (float, default 1.0 = disabled) — multiplicative penalty (transformers-style): positive logits for seen tokens are divided by the penalty, negative logits multiplied. Values > 1.0 push repeated-token probabilities down. Each seen token is penalized once.
  - **`presence_penalty`** (float, default 0.0 = disabled) — OpenAI-style: subtracts once from the logit of any token already present in the generated text.
  - **`frequency_penalty`** (float, default 0.0 = disabled) — OpenAI-style: subtracts `count * penalty` from a seen token's logit, so frequently repeated tokens are penalized more strongly.
  - All three can be combined and are applied in `sample_constrained` (`src/generate.rs`). They share a fast path that skips the per-token allocation when all are at their defaults and no grammar constraint is set. Server defaults come from the CLI flags (`--repetition-penalty`, `--presence-penalty`, `--frequency-penalty` on `modelc run`); per-request values override them.
- **`GET /v1/models`** — OpenAI-compatible model list: `{ "object": "list", "data": [{ "id": "...", "object": "model", "owned_by": "modelc" }] }`.
- **`POST /v1/chat/completions`** — OpenAI-compatible chat completion: request JSON `{ "model": "...", "messages": [...], "stream": false, "response_format": { "type": "json_object" }, "tools": [...], "max_tokens": 128, "temperature": 0.0, "top_p": 0.0, "logprobs": false, "top_logprobs": 0 }`, response with `choices`, `usage`. Applies the chat template before generation, same as `/chat`. When `stream: true`, returns an SSE stream of `chat.completion.chunk` objects with incremental `delta.content`, ending with `finish_reason: "stop"` and `[DONE]`.
- **`POST /v1/completions`** — OpenAI-compatible legacy (non-chat) completion: request JSON `{ "model": "...", "prompt": "...", "max_tokens": 128, "temperature": 0.0, "top_p": 0.0, "logprobs": false, "top_logprobs": 0, "stop": [], "stream": false }`, response `{ "id": "...", "object": "text_completion", "created": 0, "model": "...", "choices": [{ "index": 0, "text": "...", "finish_reason": "stop", "logprobs": null }], "usage": { "prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M } }`. No chat template is applied; the prompt is used directly. Supports the same `grammar`, `json_schema`, `logprobs`, `top_logprobs`, `stop`, and `stream` parameters as `/v1/chat/completions`. When `stream: true`, emits `text_completion` SSE chunks with incremental `text` deltas, ending with `finish_reason: "stop"` and `[DONE]`.
  - **`response_format`** — optional. When `type` is `"json_object"`, a system prompt instructing JSON-only output is injected, and the generated text is post-processed to extract the first well-formed JSON object or array. Falls back to raw text if no valid JSON is found.
  - **`tools`** — optional array of tool definitions (`{ "type": "function", "function": { "name": "...", "description": "...", "parameters": {...} } }`). When present, a system prompt describing available tools is injected. Generated output is parsed for a `tool_calls` JSON array; if found, the response returns `message.tool_calls` with `finish_reason: "tool_calls"`. Otherwise falls back to regular text content.
  - **`max_tokens`** — optional, default 128. Maximum number of tokens to generate.
  - **`temperature`** — optional, default 0.0. Sampling temperature; 0.0 = greedy argmax.
  - **`top_p`** — optional, default 0.0. Nucleus (top-p) sampling threshold; 0.0 = disabled. When > 0 and < 1, keeps the smallest set of tokens whose cumulative probability exceeds `top_p`, renormalizes, and samples.
  - **`min_p`** — optional, default 0.0. Min-p sampling threshold; 0.0 = disabled. When > 0 and < 1, keeps only tokens whose probability is at least `min_p` fraction of the max probability, renormalizes, and samples. The threshold scales with the model's confidence per step; can combine with `top_p` (min-p runs after top-p).
  - **`logprobs`** — optional, default `false`. When `true`, the response includes `choices[].logprobs.content`: an array of per-token entries, each with `token` (decoded text), `logprob` (`ln(P(token))` from the raw pre-temperature softmax, so deterministic regardless of `temperature`), `bytes` (raw UTF-8 byte array), and `top_logprobs`. The field serializes to `null` when `logprobs` is omitted/false (matching OpenAI). Non-transformer models return an object with an empty `content` array.
  - **`top_logprobs`** — optional, default 0, range 0–20 (clamped). Requires `logprobs: true`. Number of top alternative tokens to include per position in each entry's `top_logprobs`, sorted by descending probability. Each alternative is `{ "token": "...", "logprob": <float>, "bytes": [...] }`.

**Inference pipeline** (in order of preference):
1. **ONNX execution plan** — if `onnx.execution_plan` metadata exists, ops are executed via the tensor runtime.
2. **Transformer forward** — when `architecture == "gpt2"` or `"llama"`, runs the full FP32 transformer forward (layer norm / RMS norm, GeLU / SwiGLU, RoPE, single-token causal attention, output projection) via `src/runtime/transformer.rs`. Inputs are resized to the model's hidden size. Mirrors the codegen forward so `run` and `compile` produce identical outputs. Supports an optional `KvCache` for autoregressive generation: cached K/V vectors are appended each step, and attention is computed over all cached positions via `attention_kv`. The `generate()` function in `src/generate.rs` drives autoregressive text generation: tokenizes the prompt, looks up embeddings, runs the forward pass in a loop with the KV cache, samples the next token (greedy or temperature), and decodes the result. This powers `/chat`, `/complete`, and `/v1/chat/completions` for real text-in/text-out.
   - **KV cache quantization** — `GenerationConfig.use_int8_kv` (default `false`) enables INT8 per-token quantization of the KV cache (`KvLayer::Int8` in `src/runtime/transformer.rs`). Each token's K and V vectors are quantized with a per-token scale (max-abs / 127) and stored as `i8`; dequantization to FP32 is transparent via `k_all()` / `v_all()`. This gives ~4× memory reduction vs FP32 and is compatible with all transformer generation paths (`generate_token_ids`, `generate_with_logprobs`, `speculative_generate`).
   - **Mixed-precision KV cache** — `GenerationConfig.use_mixed_kv` (default `false`) combines FP32 accuracy for recent tokens with INT8 memory savings for older ones (`KvLayer::Mixed`). A configurable hot window (default 64 tokens) keeps the most recent K/V vectors in full FP32 precision; when the window overflows, the oldest token is quantized to INT8 and moved to cold storage. This preserves accuracy for the actively attended context while reducing overall memory footprint vs pure FP32. `k_all()` / `v_all()` transparently concatenate cold (dequantized) + hot (FP32) in chronological order. Compatible with prefix caching, context shifting, and speculative decoding. Covered by `kv_layer_mixed_keeps_recent_in_fp32`, `kv_layer_mixed_matches_fp32_output`, and `kv_layer_mixed_shift_discards_oldest` tests.
   - **Prefix caching** — `PrefixCache` (`src/prefix_cache.rs`) stores `KvCache` snapshots keyed by token sequence. Before generation, `generate_core` looks up the longest cached prefix of the prompt; if found, the saved K/V state is cloned and only the divergent suffix tokens are processed. After priming, the full prompt's KV state is inserted back into the cache. This eliminates redundant computation for repeated or shared-prefix prompts (system prompts, tool descriptions, few-shot examples). LRU eviction bounds the cache to 32 entries per loaded model. Wired through the serve layer so all text generation endpoints (`/chat`, `/complete`, `/v1/chat/completions`) benefit automatically.
   - **Concurrent transformer generation** — The `PrefixCache` is an `RwLock` (was `Mutex`), and the serve layer (`src/serve/infer.rs`) holds it only briefly: read-lock for lookup, release during generation, write-lock for insertion. `generate_core` and `speculative_generate` (`src/generate.rs`) accept a pre-computed `CacheLookup` and return the cacheable `KvCache` state, so callers manage their own locking. This allows multiple transformer requests to run in parallel, limited only by CPU cores. The previous `Mutex`-based design serialized all generation through the prefix cache, which was the primary throughput bottleneck under load. Covered by `run_server_handles_concurrent_transformer_requests` test.
   - **Attention allocation optimization** — `attention_kv` (`src/runtime/transformer.rs`) previously allocated a fresh `scores` Vec and called `softmax` (which allocated twice more) for every head, on every layer, on every token — ~3×n_heads allocations per attention call. For a 12-layer, 12-head model that's ~432 heap allocations per generated token. Rewrote `attention_kv` to use a single pre-allocated scratch buffer reused across heads, and added `softmax_inplace` which operates in-place with zero allocations. The Q·K dot product also uses an explicit indexed loop instead of `zip().map().sum()`, which is more reliably optimized by the CPU backend. This eliminates the dominant allocation hotspot in the transformer forward pass. Covered by `softmax_inplace_matches_softmax` and `attention_kv_multi_head_no_alloc_bloat` tests.
   - **LoRA loading at runtime via HTTP** — `POST /lora/load` accepts a Safetensors LoRA adapter path and an optional `alpha` scaling factor. The server clones the base model tensors, applies the low-rank update (`src/lora.rs`), and atomically swaps the active `Runtime` via `RwLock<Runtime>`. `POST /lora/unload` restores the base model from the stored `base_tensors` without restarting. This lets operators switch model personalities (e.g., coding assistant, creative writer) on the fly.
   - **Grammar-based constrained decoding** — `GenerationConfig.constraint` holds an optional `Arc<dyn Constraint>`. `RegexConstraint` (`src/constraint.rs`) wraps a regex pattern and, before each sampling step, decodes the partial output, builds a per-token validity mask, and masks invalid tokens to `-inf`. The heuristic tries common suffixes to avoid false negatives, making it permissive rather than exact. HTTP endpoints accept an optional `grammar` regex per request. This is a minimal implementation; a full FSM-based approach (like Outlines) would provide exact constraints.
   - **Context shifting (sliding window KV cache)** — `GenerationConfig.max_context` sets a hard context limit. When total tokens exceed it during generation, `generate_core` shifts both the `KvCache` (`KvLayer::shift` removes oldest K/V vectors from every layer, keeping roughly the newest 75%) and `token_ids` left. `prompt_len` is adjusted so generation continues seamlessly. Enables effectively infinite context length at the cost of losing distant history. Covered by `kv_layer_fp32_shift_discards_oldest`, `kv_layer_int8_shift_discards_oldest`, and `kv_cache_shift_syncs_all_layers` tests.
   - **Anchor token preservation (StreamingLLM-style)** — `GenerationConfig.anchor_tokens` (default 0) sets how many initial tokens to preserve during context shifting. Inspired by StreamingLLM, the first few tokens act as "attention sinks" — they accumulate disproportionate attention scores and evicting them degrades generation quality. When `anchor_tokens > 0` and `max_context` is exceeded, `context_shift` calls `KvLayer::shift_anchored` and `KvCache::shift_anchored` to remove tokens from the middle (after the anchors) rather than from the beginning. This preserves the first N prompt tokens (e.g., system prompt, few-shot examples) while still evicting older non-anchor context to make room for new tokens. Works with all KV cache variants (FP32, INT8, Mixed). Covered by `kv_layer_fp32_shift_anchored_preserves_first`, `kv_layer_int8_shift_anchored_preserves_first`, `kv_layer_mixed_shift_anchored_preserves_first`, and `kv_layer_shift_anchored_falls_back_when_not_enough_tokens` tests. Set via `--anchor-tokens N` on `modelc run`.
   - **Speculative decoding** — `GenerationConfig.gamma > 0` enables speculative decoding with an n-gram draft model. The draft model builds a trigram index from the prompt and proposes `gamma` candidate tokens by looking up historical continuations. The target model verifies each candidate in a forward-pass loop using the KV cache; accepted tokens skip the sampling step. On mismatch, the model's sampled token is used and drafting restarts. Now supports prefix caching (restores cached K/V before priming), context shifting (`max_context`), and grammar constraints (`sample_constrained`). Disabled by default (`gamma: 0`).
3. **MLP GEMV** — when `architecture == "mlp"`, runs stacked GEMV + bias (+ ReLU between hidden layers) using `layerN.weight`/`layerN.bias` or a single `weight`/`bias`.
4. **Echo fallback** — returns input unchanged.

## Library runtime

`modelc::runtime::ops` exposes tensor helpers (`matmul`, `linear`, `softmax`, …). `Runtime::from_raw` loads tensors from `TensorData` into `f32` buffers for **F32, F16, BF16, I64, I32, I16, I8, U8, Bool** (integer/bool values are cast to `f32` for the internal representation).

## Tokenizer

`modelc::tokenizer::BpeTokenizer` implements byte-level BPE:
- **Vocabulary** — each token ID maps to a byte sequence (`Vec<u8>`). The base vocab contains 256 single-byte tokens.
- **Merges** — an ordered list of `(first_id, second_id, merged_id)` rules. During encoding, the algorithm greedily applies the highest-priority merge that appears in the current token sequence, repeating until no more merges are possible.
- **Encoding** — `encode(text: &str) -> Vec<u32>` maps input text to token IDs.
- **Decoding** — `decode(tokens: &[u32]) -> String` concatenates the byte sequences for each token ID (invalid IDs are skipped).
- **`byte_fallback()`** — creates a minimal 256-token vocab with no merges, useful as a fallback or for testing.

### GGUF tokenizer metadata extraction

`modelc::parsers::gguf::extract_tokenizer_metadata(path)` reads the KV section of a GGUF file and returns `GgufTokenizerMetadata` containing:
- `model` — tokenizer model type (e.g., `"gpt2"`, `"llama"`).
- `vocab` — all token strings from `tokenizer.ggml.tokens` (or `tokenizer.ggml.vocab`).
- `merges` — BPE merge rules from `tokenizer.ggml.merges`.
- `bos_token_id` / `eos_token_id` — special token IDs.
- `chat_template` — Jinja2 chat template from `tokenizer.chat_template` (if present).

`BpeTokenizer::from_gguf(&metadata)` constructs a working tokenizer from this data by converting each vocab entry to bytes, building a token→ID map, and parsing merge rules into `(first_id, second_id, merged_id)` tuples. This enables real text-in/text-out for GGUF models without external tokenizer files.

## Chat templates

`modelc::chat_template::apply_chat_template(template, messages)` renders a conversation into a single prompt string using a Jinja2 template.

- **Template source** — if the GGUF file contains `tokenizer.chat_template`, that string is loaded at server startup and stored in `AppState.chat_template`.
- **Rendering** — `minijinja` evaluates the template with `messages`, `bos_token`, `eos_token`, and `add_generation_prompt` in scope. The `messages` array contains objects with `role` and `content` fields.
- **Fallback** — when no template is present, messages are concatenated as `role: content\n` (sufficient for basic interaction).
- **Usage** — `/chat`, `/chat/stream`, and `/v1/chat/completions` all apply the template before tokenizing the prompt and invoking `generate()`.

## Quantization

### GGUF block-quantized inference

`modelc` supports Q4_0, Q5_0, Q8_0, Q4_K, and Q6_K GGML quantization layouts from GGUF files:

- **Parse time** — the GGUF parser (`src/parsers/gguf/mod.rs`) preserves the raw quantized bytes and stores them with `DataType::Q4_0` / `Q5_0` / `Q8_0` / `Q4_K` / `Q6_K` in the IR. This keeps `.modelc` artifacts compact (~4.5 bits/element for Q4_0 vs 32 for F32).
- **Runtime dequantization** — `Runtime::from_raw` detects quantized dtypes and transparently dequantizes via `dequantize_gguf_tensor()` before building the in-memory `Tensor` objects. The transformer forward pass (`src/runtime/transformer.rs`) therefore always sees FP32 weights without any code changes.
- **Codegen path** — `modelc compile` calls `Model::dequantize_in_place()` before generating the server, so the emitted binary embeds FP32 weights (matching the current codegen contract).
- **Export** — `modelc export` to Safetensors requires FP32 tensors; quantized tensors must be dequantized first (`dequantize_in_place`).

### Pack-time quantization (`--quantize`)

`modelc compile --quantize fp16/int8/int4` converts FP32 tensors at pack time:
- **FP16** — stored as `DataType::F16`.
- **INT8** — per-tensor symmetric quantization to `DataType::I8` with `quant_scale.<name>` metadata. `Runtime::from_raw` casts to FP32 directly (scale is not applied at runtime; this is a known limitation for `run` — `dequantize_in_place` handles it for `inspect`/`export`).
- **INT4** — packed two signed nibbles per byte, stored as `DataType::I8` with `quant_mode.<name> = "int4"` metadata.

## Platform support

- **macOS** — primary development and runtime target; Apple Silicon M-series acceleration via Metal.
- **Windows** — runtime target; CLI and generated server build and run.
- **Linux** — runtime target; CLI and generated server build and run.

## Acceleration

- **Apple Silicon (M-series):** Full Metal GPU kernels for relu, add, mul_scalar, softmax, layer_norm, matmul (`src/metal.rs`, `src/compute/shaders.metal`). GPU memory pressure handling via pre-allocation checks.
- **CPU:** AVX (x86_64) and NEON (aarch64) SIMD paths with runtime feature detection; `rayon` parallelization for large matmuls.

## Compatibility

- **Host:** `cargo` required for `compile`.
- **Targets:** optional `--target` triple for the generated build.

## Success criteria

- `cargo test` and CI (`fmt`, `clippy -D warnings`, `test`) pass.
- `inspect` / `pack` / `run` succeed on supported fixtures and examples where parsers are implemented.
- `.modelc` artifact is a **single file** containing all model data.
- `list` correctly enumerates locally stored artifacts.
- Artifact size and load time are optimized relative to raw weight directories.

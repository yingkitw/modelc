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

**Input:** path to weights file; optional `-f`, `-o`, `--arch`, `--compress`, `--quantize fp16|int8|int4|q4_0|q8_0`, `--prune <threshold>`.

**Output:** `.modelc` single-file artifact containing JSON header + tensor blob (optionally compressed / quantized / pruned).

**Side effects:** writes artifact file to specified path.

**Errors:** parse failure; invalid output path; compression failure.

**Python programs (`.py`):** when the input ends in `.py`, it is *executed* by the python bridge (`src/pybridge.rs`) — the system Python (`python3`, falling back to `python`) runs an embedded stdlib-only driver that harvests tensors from the program's globals and writes a temporary Safetensors file which then flows through the normal pack path. Harvested sources: any global with a callable `state_dict()` (torch `nn.Module` — no torch import needed by modelc), numpy arrays, dicts of tensors, and plain nested lists of numbers. When every `state_dict()` entry alternates a 2-D `*.weight` with a 1-D `*.bias`, parameters are re-emitted as `layer{N}.weight`/`layer{N}.bias` (bare globals named `layer{N}_weight`/`layer{N}_bias` likewise), which makes the artifact `mlp`-shaped and runnable through the MLP inference path. The program first runs as an import (`__name__ == "__modelc__"`) so `if __name__ == "__main__"` blocks stay dormant; only when that yields no tensors is it re-run as `__main__`. `-f` is rejected for `.py` inputs; the program may run twice in that second case, and script output is echoed after the run completes. `inspect` does **not** execute programs (undetectable without `-f`).

### `run`

**Input:** path to `.modelc` artifact (or uses default store); optional `--port`, `--bind`, `--profile`.

**Output:** HTTP server listening for inference requests.

**Side effects:** loads artifact into memory, starts HTTP server.

**Errors:** missing artifact; invalid artifact format; server bind failure.

### `prompt`

**Input:** path to `.modelc` artifact (or store name); prompt string; optional `--max-tokens`, `--temperature`, `--seed`, `--top-p`, `--min-p`.

**Output:** generated text on stdout (transformer artifacts) or a JSON array of byte-normalized floats (MLP / other), matching `POST /complete` semantics. Does **not** start an HTTP server.

**Errors:** missing artifact; invalid artifact format.

### `serve`

One command, two transports. `modelc serve <artifact> [flags]` shares `run`'s flag set (`--port`/`--bind`, `--api-key`/`--rate-limit`/`--max-concurrent`, and all generation-default flags) — without `--stdio` it is the HTTP server, byte-for-byte.

**HTTP mode (default):** identical to `run`; binds `--bind:--port` and serves the full router until `SIGINT`/`SIGTERM`.

**stdio mode (`--stdio`):** serves **newline-delimited JSON-RPC 2.0 on stdin/stdout**, wire-compatible with the Model Context Protocol (MCP) stdio transport — attachable from editors and agent runtimes. Design rules:

- stdout carries **only** protocol messages; all logs go to stderr. Each response is one line, flushed immediately.
- Methods: `initialize` (echoes a supported `protocolVersion` — `2024-11-05`, `2025-03-26`, `2025-06-18` — and reports `capabilities.tools` + `serverInfo: {name: "modelc", version}`; `capabilities.resources` is advertised only from `2025-03-26` on, because the `2024-11-05` spec predates resources), `ping` → `{}`, `tools/list` → `chat` / `complete` / `embeddings` / `rerank` tools with JSON-Schema `inputSchema`s mirroring the native request shapes, `tools/call` → inference, `resources/list` → the read-only model resources, `resources/read` → one resource's contents, `notifications/initialized` → silent.
- `tools/call` results use the MCP content envelope `{content: [{type: "text", text}], isError}`. Tool failures (bad arguments, unknown tool) are error **results** with `isError: true`; protocol violations use JSON-RPC errors: `-32601` unknown method, `-32600` missing method, `-32700` unparseable line (id `null`). Resource reads use `-32002` for an unknown `uri` and `-32602` when the `uri` parameter is missing.
- **Resources** (read-only model metadata, `resources/list` / `resources/read`): `modelc://model` (name, architecture, tensor count, parameter and byte totals, quantization label, and the artifact's own metadata), `modelc://model/tensors` (the tensor table — name, shape, dtype, element count, stored bytes, sorted by name), `modelc://model/tokenizer` (model type, BOS/EOS token ids, chat-template size, and the raw `tokenizer.*` entries the artifact carried). Contents come back in the MCP envelope `{contents: [{uri, mimeType: "application/json", text}]}` with the payload as JSON text; metadata values are capped at 1024 characters.
- **The `rerank` tool** takes `{query, documents, top_n?, return_documents?}` and returns `{results: [{index, relevance_score, document?}]}` — the same ranking, from the same shared core, that the `/v1/rerank` endpoint produces (scores clamped into the documented `[0, 1]` range; `document` only when `return_documents` is true).
- Inference goes through the same internals as the HTTP handlers (chat template, sampling params, JSON-Schema retry, prefix cache, metrics), so both transports agree on semantics.
- `--api-key` / `--rate-limit` apply to the HTTP mode only; stdio has no authentication layer.

**Example session** (`modelc serve my.modelc --stdio`):

```
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"modelc","version":"…"}}}
→ {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"complete","arguments":{"prompt":"hello"}}}
← {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"…"}],"isError":false}}
```

**Errors:** missing artifact; invalid artifact format; port already bound (HTTP mode).

### `list`

**Input:** none (uses default model store).

**Output:** stdout list of locally stored `.modelc` artifacts with metadata.

**Errors:** store access failure.

### `rm`

**Input:** model name; optional `--all` to also delete versioned copies (`<name>.v<N>.modelc`); optional `--force` to delete the main artifact even when versioned copies exist.

**Output:** confirmation of deleted files.

**Side effects:** removes files from the local store.

**Errors:** model not found in store; versioned copies exist (without `--all` or `--force`); filesystem failure.

### `cp`

**Input:** source model name or path; destination model name.

**Output:** confirmation of copied file path.

**Side effects:** copies `<source>.modelc` to `<dest>.modelc` in the local store.

**Errors:** source not found; destination already exists; filesystem failure.

### `pull`

**Input:** `http://`/`https://` URL (or local file path); optional `--name`, `--version`, `--token`.

**Output:** downloaded `.modelc` artifact stored locally.

- `https://` is delegated to the system `curl` (present on macOS/Linux/Windows 10+).
- `--token <t>` or env `MODELC_TOKEN` sends `Authorization: Bearer <t>` — required for token-gated endpoints (e.g. modelc-web).
- When the server sends `X-Modelc-Digest` (or `ETag`) as `sha256:<hex>`, the downloaded bytes are verified against it — a mismatch fails the pull.

**Errors:** network failure; digest mismatch; missing `curl` for `https://`.

### `diff`

**Input:** two artifact identifiers (path or store name); optional `--json`.

**Output:** structural diff of the artifact headers — tensors added/removed/changed (shape/dtype/bytes), architecture, parameter and size deltas, metadata diff. Compares headers only, never tensor data.

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

**`--quant-sizes`** — instead of the normal summary, print a quantization size preview: the total byte size the model would occupy under each format (fp32, fp16, int8, int4, q4_0, q8_0) without actually quantizing. Sizes are computed from element counts (shape-based), so they are accurate regardless of the tensors' current dtype. INT4 packs two signed nibbles per byte; Q4_0 uses 18 bytes per block of 32 elements and Q8_0 34 bytes, with per-tensor block rounding. Shows each format's size in MB and the percentage change vs the current stored size.

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
| Python program | Implemented via the **python bridge** (`src/pybridge.rs`): `.py` inputs are executed with the system Python (embedded stdlib-only driver) and their tensors are written as Safetensors — torch `state_dict()` objects, numpy arrays, and plain nested lists are all harvested. |

## Format detection

0. **Python programs** — a `.py` extension short-circuits detection entirely: the python bridge executes the program (see `pack`).

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

- **`GET /`** — built-in web UI. Returns a self-contained HTML page with a chat interface that connects to `/v1/chat/completions` (SSE streaming) and displays model info from `/info`. No external dependencies, no build step — just open the server URL in a browser. Covered by `run_server_serves_web_ui` test.
- **`GET /info`** — JSON object: `name`, `architecture`, `total_params`, `total_bytes`, `tensors` (array of tensor name strings).
- **`GET /health`** — JSON object: `status` (`"ok"`), `model`, `architecture`. Liveness probe for load balancers and orchestration.
- **`POST /infer`** — request JSON `{ "input": number[] }` (`f32`) or `{ "inputs": [[...], [...]] }` for batch. Response `{ "output": number[] }` or `{ "outputs": [[...], [...]] }`.
  - **Batch optimization (MLP)** — when `inputs` has multiple items and the model architecture is `"mlp"`, the server routes to `run_mlp_forward_batched`, which computes the entire batch in a single pass via `batched_gemv_bias`. This eliminates redundant weight traversal and improves CPU cache locality compared to serial per-item inference. Non-MLP models fall back to serial execution.
- **`POST /chat`** — request JSON `{ "messages": [{"role": "...", "content": "..."}], "max_tokens": 128, "temperature": 0.0, "top_p": 0.0, "min_p": 0.0 }`, response `{ "message": {"role": "assistant", "content": "..."} }`. Messages are formatted through the model's chat template (if present) before tokenization; otherwise concatenated as `role: content\n`. `max_tokens`, `temperature`, `top_p`, and `min_p` are optional and override server defaults.
- **`POST /chat/stream`** — SSE stream of `{ "delta": "...", "done": bool }` chunks. Same template logic as `/chat`. For transformer models, each token is decoded and emitted as it is generated, so the client receives text incrementally. For non-transformer models, the full response is sent in a single chunk. Accepts optional `max_tokens`, `temperature`, `top_p`, and `min_p`. If the client disconnects, the server trips a cooperative cancellation flag (`CancelOnDrop`) that aborts the generation loop after the current token instead of running to completion.
- **`POST /complete`** — request JSON `{ "prompt": "...", "max_tokens": 128, "temperature": 0.0, "top_p": 0.0, "min_p": 0.0 }`, response `{ "completion": "..." }`. `max_tokens`, `temperature`, `top_p`, and `min_p` are optional.
- **`POST /infill`** — fill-in-the-middle (FIM) code completion. Request JSON `{ "prefix": "...", "suffix": "...", "prompt": "...", "max_tokens": 128, "temperature": 0.0, "top_p": 0.0, "min_p": 0.0, "seed": null, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0 }`. If `prompt` is supplied, the model generates with `prompt + "\n" + prefix` as the input and `suffix`, `"\n```"`, and `prompt` as stop sequences; otherwise just `prefix` with `suffix` and `"\n```"` as stop sequences. Response `{ "completion": "..." }`. `max_tokens`, `temperature`, `top_p`, `min_p`, `seed`, `repetition_penalty`, `presence_penalty`, `frequency_penalty` are optional and override the server defaults.
- **`POST /embeddings`** — request JSON `{ "input": "..." }` (single) or `{ "inputs": ["...", "..."] }` (batch). Single response: `{ "embedding": [f32, ...], "model": "..." }`. Batch response: `{ "embeddings": [{"embedding": [f32, ...], "index": 0}, ...], "model": "..." }`. Uses the final hidden state (after layer norm / RMS norm, before the output head) for transformer models; echo fallback for others.
- **`POST /v1/embeddings`** — OpenAI-compatible embeddings endpoint. Request JSON: `{ "input": "..." }` or `{ "input": ["...", "..."] }`, optional `encoding_format: "float"` (default) or `"base64"`, optional `model`. Response: `{ "object": "list", "data": [{ "object": "embedding", "embedding": [f32, ...] | "base64...", "index": 0, "dimensions": N }, ...], "model": "...", "usage": { "prompt_tokens": N, "total_tokens": N } }`. Uses the same embedding logic as `/embeddings`.
- **`POST /reranking`** — llama.cpp-compatible reranking endpoint for RAG pipelines. Request JSON: `{ "query": "...", "documents": ["...", "..."], "top_n": N }` (`top_n` optional). Embeds the query and each document via the model's embedding path, computes cosine similarity between the query embedding and each document embedding, and returns results sorted by descending relevance score. Response: `{ "model": "...", "results": [{ "index": 0, "relevance_score": 0.95, "document": { "text": "..." } }, ...] }`. When `top_n` is provided, only the top N results are returned. Covered by `run_server_serves_reranking` test.
- **`POST /v1/rerank` · `/v2/rerank`** — Cohere-compatible reranking for RAG pipelines. Both API versions' paths share one handler: `cohere.Client` calls `/v1/rerank` and `cohere.ClientV2` (Cohere's current default) calls `/v2/rerank`, each appending its own path to the configured `base_url` — so either SDK generation can point at a modelc server. Request JSON: `{ "model": "...", "query": "...", "documents": ["...", { "text": "..." }], "top_n": N, "return_documents": false }`. `model` is accepted but ignored (one model per server). `documents` are plain strings or objects ranked on their `text` field (Cohere v1); an object without a string `text` returns `400 { "message": "..." }`. `top_n` limits the number of results (default: all). The query and each document go through the same embedding path as `/reranking`; results are ordered by descending cosine similarity and `relevance_score` is that similarity clamped into Cohere's documented `[0, 1]` range — the ordering is meaningful, the magnitude is not a calibrated relevance probability. `document` is echoed back only when `return_documents` is true (string documents as `{ "text": "..." }`, object documents with their extra fields preserved). Response: `{ "id": "rerank-<model>", "results": [{ "index": 0, "relevance_score": 0.95, "document": { "text": "..." } }, ...] }`. Not supported (accepted, no effect): `rank_fields` (always `text`), `max_chunks_per_doc` / `max_tokens_per_doc`, `relevant_spans`. Covered by `run_server_serves_cohere_rerank` and `run_server_serves_cohere_rerank_object_documents_and_v2_path`.
- **`POST /tokenize`** — encode text to token IDs using the model's byte-level BPE tokenizer (the same one `/chat` and `/complete` use). Request JSON `{ "input": "..." }` (single) or `{ "inputs": ["...", "..."] }` (batch). Single response: `{ "tokens": [id, ...], "count": N }`. Batch response: `{ "tokens_batch": [[id, ...], ...], "count": N }` where `count` is the total across all inputs. Useful for prompt budgeting and prompt management.
- **`POST /detokenize`** — decode token IDs back to text using the model's byte-level BPE tokenizer. Request JSON `{ "tokens": [id, ...] }`. Response `{ "text": "..." }`. Complements `/tokenize` for round-trip token management.
- **`GET /api/version`** — exposes CLI version and git SHA for orchestration/health checks. JSON: `{ "version": "<CARGO_PKG_VERSION>", "git_sha": "abc123" }`. `version` is captured at compile time via `env!("CARGO_PKG_VERSION")`; `git_sha` is captured at compile time by `build.rs` via `git rev-parse --short HEAD` (or `"unknown"` when git is unavailable).
- **`GET /api/tags`** — Ollama-ecosystem endpoint listing `.modelc` artifacts in the local store. Response: `{ "models": [{ "name": "...", "model": "...", "modified_at": "<RFC3339>", "size": N, "digest": "sha256:<hex>", "details": { "format": "...", "family": "...", "families": ["..."], "parameter_size": "8.0B", "quantization_level": "Q4_0" }, "architecture": "...", "tensor_count": N, "compressed": bool }, ...] }`. Field names match what real Ollama clients parse (`serve/ollama.rs`); `architecture`/`tensor_count`/`compressed` are modelc extras. Covered by `run_server_serves_api_tags` test.
- **`POST /api/show`** — Ollama-ecosystem endpoint for model details. Accepts `{ "model": "..." }` (Ollama) or `{ "name": "..." }` (legacy). Returns `{ "model": "...", "modified_at": "...", "size": N, "digest": "sha256:<hex>", "details": {...}, "model_info": { "general.architecture": "...", "general.parameter_count": N, "general.file_type": N }, "capabilities": ["completion","embedding","insert","tools"], "license": "...", "modelfile": "", "parameters": "", "template": "", "name": "...", "architecture": "...", "size_bytes": N, "parameter_size": "...", "quantization": "...", "tensor_count": N }`. Returns 404 for nonexistent models. Covered by `run_server_serves_api_show` test.
- **`POST /api/generate`** — Ollama-ecosystem text generation. Request `{ "model": "...", "prompt": "...", "system": "...", "raw": false, "stream": true, "format": "json" | {schema}, "options": { "num_predict": N, "temperature": f, "top_p": f, "min_p": f, "seed": N, "repeat_penalty": f, "presence_penalty": f, "frequency_penalty": f, "stop": "s" | ["s"], "num_ctx": N } }`. `raw: true` sends the prompt verbatim; otherwise `system` + prompt go through the chat template. `num_predict` ≤ 0 uses the server default. With `"stream": false` returns `{ "model", "created_at", "response", "done": true, "done_reason": "stop", "context": [ids], "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration" }`. Default `stream: true` emits **NDJSON** lines (`{"model","created_at","response": delta,"done":false}` then a final `done:true` line with the stats above) — Ollama's wire format, not SSE. `format` routes through the JSON-schema retry path. Covered by `run_server_serves_api_generate` and `run_server_serves_api_generate_streams_ndjson`.
- **`POST /api/chat`** — Ollama-ecosystem chat generation. Request `{ "model": "...", "messages": [{"role","content"}], "stream": true, "format": ..., "options": {...} }`; messages are templated like `/chat`. Non-streaming response mirrors `/api/generate` but carries `message: {"role":"assistant","content":"..."}` instead of `response`. Streaming emits NDJSON chunks with `message` deltas. Covered by `run_server_serves_api_chat`.
- **`POST /api/embed`** — Ollama-ecosystem embeddings. Request `{ "model": "...", "input": "..." | ["...", "..."] }` (`truncate`, `options`, `keep_alive` ignored). Response `{ "model": "...", "embeddings": [[f32, ...]], "total_duration": N, "load_duration": N, "prompt_eval_count": N }` — one embedding per input element, same hidden state as `/embeddings`. Covered by `run_server_serves_api_embed`.
- **`POST /api/embeddings`** — legacy Ollama embeddings endpoint. Request `{ "model": "...", "prompt": "..." }`; response `{ "embedding": [f32, ...] }`. Covered by `run_server_serves_api_embeddings_legacy`.
- **`GET /props`** — llama.cpp-compatible server properties. JSON: `{ "model": "...", "architecture": "...", "total_params": N, "total_bytes": N, "chat_template": "..." | null, "default_generation": { "max_tokens": N, "temperature": F, "top_p": F, "min_p": F, "repetition_penalty": F, "presence_penalty": F, "frequency_penalty": F, "gamma": N } }`. Exempt from auth and backpressure so clients can discover server capabilities without credentials.
- **`GET /v1/system`** — best-effort system/hardware info for orchestration and debugging. JSON: `model`, `architecture`, `total_params`, `total_bytes`, `cpu_cores` (logical cores via `available_parallelism`), `os`, `cpu_arch`, `pointer_width`, `metal_available` (true on Apple Silicon), and `memory_total_bytes` (`null` if unavailable; reads `/proc/meminfo` on Linux and `sysctl hw.memsize` on macOS — no added dependencies).
- **`POST /lora/load`** — request JSON `{ "path": "/path/to/adapter.safetensors", "alpha": 1.0 }`. Clones the base model tensors, applies the Safetensors LoRA adapter (`src/lora.rs`), and atomically swaps the runtime via `RwLock<Runtime>`. Response `{ "applied": <usize>, "skipped": <usize>, "message": "..." }`. On failure (e.g., file not found), `applied` and `skipped` are 0 and `message` contains the error.
- **`POST /lora/unload`** — no body. Restores the base model from the stored `base_tensors` without restarting the server. Response `{ "message": "LoRA unloaded; base model restored" }`.
- **`GET /api/cache/stats`** — prefix cache statistics for the running model. Response `{ "entries": N, "capacity": N, "total_tokens": N, "total_kv_bytes": N, "utilization": F }` where `utilization` is `entries / capacity` in `[0.0, 1.0]`. Useful for ops dashboards; exempt from auth and backpressure so probes keep working under load.
- **`POST /api/cache/clear`** — drop every cached prefix snapshot. Use after loading a new LoRA adapter, swapping runtime tensors, or recovering from a suspected cache-poisoning incident. Response `{ "cleared_entries": N, "message": "prefix cache cleared" }`.
- **`GET /metrics`** — Prometheus text exposition format. Returns `modelc_requests_total` (counter), `modelc_inference_duration_seconds` (histogram with buckets 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, +Inf), `modelc_active_requests` (gauge), and `modelc_tokens_generated_total` (counter). Handlers use RAII `ActiveRequestGuard` and `InferenceTimer` structs for automatic tracking.
- **Authentication, rate limiting, and backpressure** — `modelc run` accepts `--api-key <key>`, `--rate-limit <N>` (requests per minute per client IP), and `--max-concurrent <N>` (in-flight inference limit). When `--api-key` is set, every request except `/health`, `/info`, and `/props` must include an `Authorization: Bearer <key>` header; missing or incorrect tokens return `401 Unauthorized`. When `--rate-limit` is set, a per-IP token bucket (`src/serve/auth.rs`) enforces the cap; exceeded requests return `429 Too Many Requests`. When `--max-concurrent` is set, a `tokio::sync::Semaphore` limits concurrent inference requests; saturated requests receive `503 Service Unavailable`. `/health`, `/info`, `/props`, and `/metrics` are always exempt so load-balancer liveness probes and observability continue to work.
- **Graceful shutdown** — `modelc run` (and compiled binaries generated by `modelc compile`) catch `SIGINT`/`SIGTERM` via `tokio::signal`. `axum::serve` uses `with_graceful_shutdown` to stop accepting new connections and drain in-flight requests before exiting. SSE streams cooperate via `CancelOnDrop`, whose `Drop` trips a cancellation flag when the client disconnects.
- **`POST /chat`**, **`POST /complete`**, **`POST /v1/chat/completions`**, **`POST /v1/completions`** — all accept an optional `grammar` string (regex pattern) per request. When provided, `RegexConstraint` (`src/constraint.rs`) masks tokens whose decoded text would violate the regex to `-inf` before sampling. The heuristic tries common suffixes to avoid false negatives; it is permissive rather than exact. If the regex is invalid, the constraint is silently ignored.
- **`POST /chat`**, **`POST /complete`**, **`POST /v1/chat/completions`**, **`POST /v1/completions`** — also accept an optional `json_schema` object per request. Generated text is parsed as JSON and validated against the schema's top-level `type` field (a lightweight parser-only check; full schema validation is not performed to avoid a third-party dependency). If invalid, generation is retried up to 3 times with increasing temperature to encourage diversity. More robust than the post-hoc `json_object` mode. The `json_schema` and `grammar` fields can be used together (schema validates the final output, grammar constrains individual tokens).
- **Sampling penalties** (`POST /chat`, `/complete`, `/v1/chat/completions`, `/v1/completions`) — three optional parameters discourage token repetition by adjusting logits before sampling:
  - **`repetition_penalty`** (float, default 1.0 = disabled) — multiplicative penalty (transformers-style): positive logits for seen tokens are divided by the penalty, negative logits multiplied. Values > 1.0 push repeated-token probabilities down. Each seen token is penalized once.
  - **`presence_penalty`** (float, default 0.0 = disabled) — OpenAI-style: subtracts once from the logit of any token already present in the generated text.
  - **`frequency_penalty`** (float, default 0.0 = disabled) — OpenAI-style: subtracts `count * penalty` from a seen token's logit, so frequently repeated tokens are penalized more strongly.
  - All three can be combined and are applied in `sample_constrained` (`src/generate.rs`). They share a fast path that skips the per-token allocation when all are at their defaults, no `logit_bias` is set, and no grammar constraint is set. Server defaults come from the CLI flags (`--repetition-penalty`, `--presence-penalty`, `--frequency-penalty` on `modelc run`); per-request values override them.
- **`logit_bias`** (`POST /chat`, `/complete`, `/v1/chat/completions`, `/v1/completions`) — optional, default `{}`. A JSON object mapping token ID (integer) to additive bias (float, typically -100 to 100). Positive values boost a token's likelihood; negative values suppress it. Applied after penalties but before the grammar constraint mask. Empty map = fast path (no allocation).
- **`POST /v1/chat/completions`** — OpenAI-compatible chat completion: request JSON `{ "model": "...", "messages": [...], "stream": false, "response_format": { "type": "json_object" }, "tools": [...], "max_tokens": 128, "temperature": 0.0, "top_p": 0.0, "logprobs": false, "top_logprobs": 0 }`, response with `choices`, `usage`, `created` (real Unix timestamp in seconds), and `system_fingerprint` (`fp_modelc_<git-sha>`). Applies the chat template before generation, same as `/chat`. When `stream: true`, returns an SSE stream of `chat.completion.chunk` objects with incremental `delta.content`, ending with `finish_reason: "stop"` and `[DONE]`.
- **`POST /v1/completions`** — OpenAI-compatible legacy (non-chat) completion: request JSON `{ "model": "...", "prompt": "..." or ["...", "..."], "max_tokens": 128, "temperature": 0.0, "top_p": 0.0, "logprobs": false, "top_logprobs": 0, "stop": [], "stream": false }`, response `{ "id": "...", "object": "text_completion", "created": 0, "model": "...", "choices": [{ "index": 0, "text": "...", "finish_reason": "stop", "logprobs": null }], "usage": { "prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M } }`. No chat template is applied; the prompt is used directly. Supports the same `grammar`, `json_schema`, `logprobs`, `top_logprobs`, `stop`, and `stream` parameters as `/v1/chat/completions`. When `prompt` is an array of strings, completions are generated for each prompt and returned as separate `choices` with incrementing `index`. When `stream: true`, emits `text_completion` SSE chunks with incremental `text` deltas, ending with `finish_reason: "stop"` and `[DONE]`.
  - **`response_format`** — optional. When `type` is `"json_object"`, a system prompt instructing JSON-only output is injected, and the generated text is post-processed to extract the first well-formed JSON object or array. Falls back to raw text if no valid JSON is found. When `type` is `"json_schema"`, the `json_schema.schema` is injected as a system prompt and `generate_with_schema` validates/retries output (up to 3 retries with increasing temperature).
  - **`tools`** — optional array of tool definitions (`{ "type": "function", "function": { "name": "...", "description": "...", "parameters": {...} } }`). When present, a system prompt describing available tools is injected. Generated output is parsed for a `tool_calls` JSON array; if found, the response returns `message.tool_calls` with `finish_reason: "tool_calls"`. Otherwise falls back to regular text content.
  - **`tool_choice`** — optional (`/v1/chat/completions` only). Controls tool usage: `"auto"` (default, model decides), `"none"` (tools not offered, no `tool_calls` in response), `"required"` (model must call a tool), or `{"type": "function", "function": {"name": "..."}}` (model must call the named tool). When `"none"`, tool descriptions are not injected.
  - **`max_tokens`** — optional, default 128. Maximum number of tokens to generate.
  - **`max_completion_tokens`** — optional (`/v1/chat/completions` and `/v1/completions`). Newer OpenAI parameter name for `max_tokens`. When both are provided, `max_completion_tokens` takes precedence.
  - **`temperature`** — optional, default 0.0. Sampling temperature; 0.0 = greedy argmax.
  - **`top_p`** — optional, default 0.0. Nucleus (top-p) sampling threshold; 0.0 = disabled. When > 0 and < 1, keeps the smallest set of tokens whose cumulative probability exceeds `top_p`, renormalizes, and samples.
  - **`min_p`** — optional, default 0.0. Min-p sampling threshold; 0.0 = disabled. When > 0 and < 1, keeps only tokens whose probability is at least `min_p` fraction of the max probability, renormalizes, and samples. The threshold scales with the model's confidence per step; can combine with `top_p` (min-p runs after top-p).
  - **`logprobs`** — optional, default `false`. When `true`, the response includes `choices[].logprobs.content`: an array of per-token entries, each with `token` (decoded text), `logprob` (`ln(P(token))` from the raw pre-temperature softmax, so deterministic regardless of `temperature`), `bytes` (raw UTF-8 byte array), and `top_logprobs`. The field serializes to `null` when `logprobs` is omitted/false (matching OpenAI). Non-transformer models return an object with an empty `content` array.
  - **`top_logprobs`** — optional, default 0, range 0–20 (clamped). Requires `logprobs: true`. Number of top alternative tokens to include per position in each entry's `top_logprobs`, sorted by descending probability. Each alternative is `{ "token": "...", "logprob": <float>, "bytes": [...] }`.
  - **`echo`** — optional, default `false` (`/v1/completions` only). When `true`, the response `choices[].text` includes the prompt followed by the completion. In streaming mode, the prompt is emitted as the first SSE chunk.
  - **`n`** — optional, default 1, range 1–20 (clamped) (`/v1/completions` and `/v1/chat/completions`). Generates N independent completions. The response includes N `choices` entries with incrementing `index`. `completion_tokens` in `usage` is the sum across all choices.
  - **`suffix`** — optional, default `""` (`/v1/completions` only). Text appended after the generated completion in the response. When combined with `echo: true`, the output format is `prompt + completion + suffix`.
  - **`best_of`** — optional, default `n` (or 1), range 1–20 (clamped) (`/v1/completions` only, non-streaming). Generates `best_of` completions server-side, sorts by token count (proxy for highest logprob), and returns the top `n`. When `best_of > n`, the client receives only the best `n` choices.
  - **`stream_options`** — optional (`/v1/chat/completions` and `/v1/completions`, streaming mode). JSON object `{ "include_usage": true }`. When set, the SSE stream emits a final chunk with empty `choices` and a `usage` object (`prompt_tokens`, `completion_tokens`, `total_tokens`) before `[DONE]`. Matches the OpenAI streaming protocol.
  - **`user`** — optional, default `null` (`/v1/chat/completions` and `/v1/completions`). OpenAI-standard user identifier for abuse tracking. Accepted but ignored (no multi-tenant state).

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
   - **Speculative decoding** — `GenerationConfig.gamma > 0` enables speculative decoding with a draft model. The draft model proposes `gamma` candidate tokens; the target model verifies each candidate in a forward-pass loop using the KV cache; accepted tokens skip the sampling step. On mismatch, the model's sampled token is used and drafting restarts. Now supports prefix caching (restores cached K/V before priming), context shifting (`max_context`), and grammar constraints (`sample_constrained`). Disabled by default (`gamma: 0`). Two draft models are available:
     - **Neural draft (`MlpDraftModel`)** — a tiny 2-layer MLP loaded from `draft.*` tensors in the runtime. Orders of magnitude faster than the full transformer. Falls back to borrowing the target model's embedding weights when `draft.embed.weight` is absent.
     - **Prompt lookup draft (`PromptLookupDraftModel`)** — a weight-free draft model that searches the current token context for the most recent occurrence of the last N tokens and proposes the continuation that followed. No weights required; used as the default when no neural draft weights are present. Particularly effective for code completion and document Q&A where output tends to repeat prompt substrings.
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
- **Rendering** — the template argument is accepted for API stability but currently ignored. Messages are always rendered via the fallback format (one `role: content\n` per message) to keep the dependency footprint minimal. Render complex chat templates (Llama-3 special tokens, tool descriptors, etc.) client-side.
- **Fallback** — when no template is present, messages are concatenated as `role: content\n` (sufficient for basic interaction).
- **Usage** — `/chat`, `/chat/stream`, and `/v1/chat/completions` all apply the template before tokenizing the prompt and invoking `generate()`.

## Quantization

### GGUF block-quantized inference

`modelc` supports Q4_0, Q5_0, Q8_0, Q4_K, and Q6_K GGML quantization layouts from GGUF files:

- **Parse time** — the GGUF parser (`src/parsers/gguf/mod.rs`) preserves the raw quantized bytes and stores them with `DataType::Q4_0` / `Q5_0` / `Q8_0` / `Q4_K` / `Q6_K` in the IR. This keeps `.modelc` artifacts compact (~4.5 bits/element for Q4_0 vs 32 for F32).
- **Runtime dequantization** — `Runtime::from_raw` detects quantized dtypes and transparently dequantizes via `dequantize_gguf_tensor()` before building the in-memory `Tensor` objects. The transformer forward pass (`src/runtime/transformer.rs`) therefore always sees FP32 weights without any code changes.
- **Codegen path** — `modelc compile` calls `Model::dequantize_in_place()` before generating the server, so the emitted binary embeds FP32 weights (matching the current codegen contract).
- **Export** — `modelc export` dequantizes GGUF block-quant tensors to F32 automatically (`dequantize_in_place`) since Safetensors has no block-quant dtype.

### Pack-time quantization (`--quantize`)

`modelc compile --quantize fp16/int8/int4/q4_0/q8_0` converts FP32 tensors at pack time:
- **FP16** — stored as `DataType::F16`.
- **INT8** — per-tensor symmetric quantization to `DataType::I8` with `quant_scale.<name>` metadata. `Runtime::from_raw` casts to FP32 directly (scale is not applied at runtime; this is a known limitation for `run` — `dequantize_in_place` handles it for `inspect`/`export`).
- **INT4** — packed two signed nibbles per byte, stored as `DataType::I8` with `quant_mode.<name> = "int4"` metadata.
- **Q4_0 / Q8_0** — GGML block quantization, using the same layouts llama.cpp writes: Q4_0 stores 18 bytes per 32 elements (f16 scale + packed 4-bit weights), Q8_0 stores 34 bytes per 32 elements (f16 scale + int8 weights). The encoded bytes are written to the artifact as-is and the tensor keeps its `DataType::Q4_0` / `Q8_0` dtype, so `Runtime::from_raw` dequantizes them on the fly — no scale metadata is needed, and the block layout is byte-compatible with llama.cpp. A tensor whose element count is not a multiple of 32 is not a whole number of blocks and stays F32 (llama.cpp's own fallback); the pack log reports how many tensors were skipped.

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

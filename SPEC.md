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
- **`POST /chat`** — request JSON `{ "messages": [{"role": "...", "content": "..."}] }`, response `{ "message": {"role": "assistant", "content": "..."} }`. Messages are formatted through the model's chat template (if present) before tokenization; otherwise concatenated as `role: content\n`.
- **`POST /chat/stream`** — SSE stream of `{ "delta": "...", "done": bool }` chunks. Same template logic as `/chat`.
- **`POST /complete`** — request JSON `{ "prompt": "..." }`, response `{ "completion": "..." }`.
- **`POST /embeddings`** — request JSON `{ "input": "..." }`, response `{ "embedding": [f32, ...], "model": "..." }`. Uses the final hidden state (after layer norm / RMS norm, before the output head) for transformer models; echo fallback for others.
- **`GET /v1/models`** — OpenAI-compatible model list: `{ "object": "list", "data": [{ "id": "...", "object": "model", "owned_by": "modelc" }] }`.
- **`POST /v1/chat/completions`** — OpenAI-compatible chat completion (non-streaming): request JSON `{ "model": "...", "messages": [...], "stream": false, "response_format": { "type": "json_object" }, "tools": [...] }`, response with `choices`, `usage`. Applies the chat template before generation, same as `/chat`.
  - **`response_format`** — optional. When `type` is `"json_object"`, a system prompt instructing JSON-only output is injected, and the generated text is post-processed to extract the first well-formed JSON object or array. Falls back to raw text if no valid JSON is found.
  - **`tools`** — optional array of tool definitions (`{ "type": "function", "function": { "name": "...", "description": "...", "parameters": {...} } }`). When present, a system prompt describing available tools is injected. Generated output is parsed for a `tool_calls` JSON array; if found, the response returns `message.tool_calls` with `finish_reason: "tool_calls"`. Otherwise falls back to regular text content.

**Inference pipeline** (in order of preference):
1. **ONNX execution plan** — if `onnx.execution_plan` metadata exists, ops are executed via the tensor runtime.
2. **Transformer forward** — when `architecture == "gpt2"` or `"llama"`, runs the full FP32 transformer forward (layer norm / RMS norm, GeLU / SwiGLU, RoPE, single-token causal attention, output projection) via `src/runtime/transformer.rs`. Inputs are resized to the model's hidden size. Mirrors the codegen forward so `run` and `compile` produce identical outputs. Supports an optional `KvCache` for autoregressive generation: cached K/V vectors are appended each step, and attention is computed over all cached positions via `attention_kv`. The `generate()` function in `src/generate.rs` drives autoregressive text generation: tokenizes the prompt, looks up embeddings, runs the forward pass in a loop with the KV cache, samples the next token (greedy or temperature), and decodes the result. This powers `/chat`, `/complete`, and `/v1/chat/completions` for real text-in/text-out.
   - **Speculative decoding** — `GenerationConfig.gamma > 0` enables speculative decoding with an n-gram draft model. The draft model builds a trigram index from the prompt and proposes `gamma` candidate tokens by looking up historical continuations. The target model verifies each candidate in a forward-pass loop using the KV cache; accepted tokens skip the sampling step. On mismatch, the model's sampled token is used and drafting restarts. Disabled by default (`gamma: 0`).
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

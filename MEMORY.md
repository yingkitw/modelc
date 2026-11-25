# Memory

This file captures institutional knowledge, patterns, and best practices for the **modelc** development team. Always check this file before implementing new features.

**Cross-repo**: `../modelc-web` (Bun+TS portal + MCP server) contains a byte-compatible TypeScript port of this CLI (`src/modelc/`, `src/code2model/`). Its MEMORY.md documents upstream gaps found during the port — check it when implementing anything the web already ported (Ollama schema, diff, digest, safetensors write). Entries marked "(learned from modelc-web)" below were ported back from it.

## 1. Format Parsers (GGUF, Safetensors, ONNX, PyTorch)

### GGUF parser entry + metadata extraction (`src/parsers/gguf/mod.rs`, 2026-11)
- **Two entry points, two I/O modes**: `GgufParser::parse(path)`/`extract_tokenizer_metadata(path)` read **from disk**; `parse_gguf_bytes(data, path)` takes bytes. Tests that bypass the trait never cover lines 56-61 — write a fixture to a `tempfile()` and call the trait method.
- **KV tolerance contract**: mismatched-type tokenizer entries are consumed via `read_kv_value_formatted(value_type)`, never skipped byte-wise — the cursor stays aligned. `extract_tokenizer_metadata` matches on exact key strings; wrong-typed `tokenizer.ggml.model/tokens/merges/bos_token_id/eos_token_id` yield defaults, correct-typed eos/chat_template harvest.
- **`general.alignment` KV drives runtime `align_offset`** — an absurd alignment (1 MiB) pushes the tensor base past EOF ("past end of file"), while `alignment = 1` short-circuits padding entirely.
- **F64 widens to F32 in IR** — the on-disk payload is 8 B/elem but `TensorData.data` is the widened 4 B/elem blob (`f64_blob_to_f32`), so payload-length assertions must use the IR width.
- **Provably-unreachable defensive arms** (fine to leave uncovered): `unpack_ggml_payload`'s quant arms are guarded out by `is_gguf_quant_type` before the call; `dense_dtype`'s internal error only fires on a programming error; `quant_block_byte_len`'s `_` arm likewise; `ggml_type_hint` arms for always-parseable types are reachable only through error messages of *error-producing* types (Q4_1=3, Q5_1=7 hit them).
- **GGUF fixture builder**: assembling raw bytes (`GGUF` magic + version + counts + KV block + tensor infos + aligned blob) beats building real GGUF files — the `gguf_bytes()` helper in `gguf/tests.rs` parameterizes kv/tensor counts, dims, and ggml type ids for error-path matrices.

### Shared architecture inspection (`src/arch.rs`, 2026-11)
- Hidden-dim fallback **chains are ordered and type-driven**: GPT-2 first-block `c_attn.weight` uses **cols** ([3H, H]), `ln_1.weight` uses **rows** ([H]), then `wte`/`lm_head` cols; LLaMA `q_proj.weight` cols, `input_layernorm.weight` rows, then `model.embed_tokens`/`lm_head` cols. Tests should exercise every rung independently (omit the earlier rung's tensors).
- `llama_layers` merges `model.layers.` and `layers.` prefixes with contains-check dedup — a fixture needs tensors under **both** prefixes to cover the merge.
- `llama_head_count` metadata hint (`attention.head_count`) wins, clamps to ≥ 1, and an unparseable hint falls back to hidden/64.

### Magic-byte sniffing order (`src/cli.rs:431-457`)
The `sniff_path` helper applies these checks **in this order** — earlier checks win, even on ambiguous inputs:

1. **GGUF magic** — first 4 bytes are `b"GGUF"` → `WeightFormat::Gguf`
2. **Zip magic** — first 4 bytes are `b"PK\x03\x04"` → `WeightFormat::Pytorch` (covers both `.pt`/`.pth` Torch ZIPs and any ZIP archive)
3. **Safetensors full-parse** — only if `len() <= 64 MiB` (`MAX_FULL_SNIFF_BYTES`); tries `SafeTensors::deserialize` on the full file → `WeightFormat::Safetensors`
4. **None** — caller must pass `-f` explicitly

**Anti-pattern**: don't try to short-circuit on extension first when sniffing is required; `.bin` is intentionally ambiguous (could be GGUF or PyTorch) and the extension-based detector `WeightFormat::detect` already handles known suffixes before falling through to `sniff_path`.

### GGUF block-quant encoder (`src/parsers/gguf/quant.rs`, 2026-11-25)
The **write** side of the formats `dequant.rs` decodes, used by `modelc pack --quantize q4_0|q8_0`. It lives in the gguf package next to the decoder so the two halves cannot drift.
- **Q4_0 block (18 B / 32 elements)**: f16 scale `d` + 16 bytes of packed nibbles. llama.cpp's `quantize_row_q4_0_ref`: `amax` = largest magnitude in the block, `max` = the *signed* value that produced it, `d = max / -8`, `id = 1/d`; then for j in 0..16 `xi = MIN(15, (int8_t)(x·id + 8.5))` (the C cast truncates toward zero) — the low nibble of byte j holds element j, the high nibble holds element j + 16.
- **Q8_0 block (34 B / 32 elements)**: f16 scale `d = amax/127` (always positive) + 32 int8 weights `round(x/d)`. Rust's `f32::round` is round-half-away-from-zero, matching C's `roundf`.
- **The scale is stored as f16 but the weights are computed with the f32 scale** — exactly like llama.cpp, so the round-trip error includes the scale rounding (≈0.05 % of amax). Do not "fix" this by quantizing against the f16 scale: it would break byte-parity with llama.cpp.
- **Q4_0's range is asymmetric**: with `d = max/-8` the representable range is `[-8·d, 7·d]` — when the block's extreme is negative the scale is positive and both extremes land exactly; when it is positive the scale is negative and the most negative value saturates at `-7/8·amax` (nibble 15). Pinned by `q4_0_encoder_matches_llama_cpp_reference_block` (scale `0xB000`, payload `0x80, 0x8F, 0x88 × 14`).
- **Whole blocks only**: both encoders return `None` when `len % 32 != 0`; the packer keeps such tensors F32 (llama.cpp's fallback) and logs "skipped N tensors (element count not a multiple of 32)".
- **No scale metadata needed**: the block carries its own f16 scale, so the tensor keeps `DataType::Q4_0`/`Q8_0` and `Runtime::from_raw` dequantizes it via `dequantize_gguf_tensor` — unlike the INT8 path, whose `quant_scale.*` metadata is *not* applied at runtime (a documented limitation, see SPEC "Pack-time quantization").
- **Golden bytes, not self-consistency**: the encoder tests assert the exact bytes of a hand-derived llama.cpp reference block *and* round-trip through the existing decoder — a nibble-order bug cannot pass both.

### GGUF dtype coverage (`src/parsers/gguf/mod.rs`)
- **Direct dtypes**: F32, F16, BF16, I8, I16, I32, I64, F64 → stored as bytes, optionally cast to F32 at runtime
- **Block-quants**: Q4_0, Q5_0, Q8_0, Q4_K, Q6_K → preserved in IR with `DataType::*`; dequantized on-the-fly via `dequantize_gguf_tensor` in `Runtime::from_raw`
- **Unsupported quants** (e.g., Q4_1, Q8_1) → named error hint (`"unsupported GGUF quant type {GGML_TYPE_*:?}"`)
- **Tensor shapes**: GGUF stores a single `ne[..]` array (rank-major); the parser maps to a Rust `Vec<usize>` directly

### ONNX initializer loading (`src/parsers/onnx/initializers.rs`)
Three paths, all funneled through the same `TensorData` IR:
1. **Inline** — `raw_data` field on the initializer protobuf, or typed fields (`float_data`, `int32_data`, etc.)
2. **External** — `external_data` field with `location` + optional `offset` + `length`; reads from the path stored in `location`
3. **Segmented** — `data_location == 1` with multiple segments concatenated; offset/length per segment

**Pattern**: always `try_into::<[u8; N]>().unwrap()` on fixed-width chunks for little-endian conversions — never use a temporary buffer.

### Safetensors metadata (`src/parsers/safetensors.rs`)
- `__metadata__` key in the Safetensors header is merged into `Model.metadata`
- `architecture` (then `model_type`) is preferred for `Model.architecture`
- Other keys become opaque string → string metadata

### PyTorch ZIP (`src/parsers/pytorch.rs`)
- ZIP archives containing `.safetensors` entries are detected and routed to the Safetensors parser
- Pickle-only checkpoints (`data.pkl`) cannot be loaded — explicit error
- Nested ZIPs are **not** supported (one level only)

### Python programs via the bridge (`src/pybridge.rs`, 2026-11)
`pack`/`compile` on a `.py` file executes it with the system Python through an **embedded stdlib-only driver** (a Rust `const &str` written to a temp dir) that writes a Safetensors file consumed by the normal parse path.
- **Samples as golden fixtures** (`examples/python/` + `tests/python_samples_test.rs`): keep capability tests anchored to *user-visible* sample programs, each asserting (a) architecture/tensor names/shapes/dtypes and (b) a forward-pass value computed independently in the test — never just "it packs". Duck-typed torch samples (FakeTensor with `tolist`, class with `state_dict()`) prove the state-dict path without torch installed; the numpy sample uses a stdlib fallback so it stays deterministic on machines without numpy.
- **Safetensors writer contract (pure Python)**: `u64 LE header length` + header JSON + blob; tensor `data_offsets` must be **contiguous starting at 0** — the Rust `safetensors` crate *sorts header entries by offset before validating*, so insertion order is safe and names need not be sorted; `__metadata__` carries `format: modelc-python` + `source` basename (SafetensorsParser merges these into `Model.metadata`).
- **Duck-typing, no torch import**: any global with a callable `state_dict()` returning a non-empty str-keyed dict is treated as a module (its optimizer-like values — e.g. `param_groups` lists of dicts — are skipped, not fatal); anything with a working `tolist()` returning a list, or a plain nested list, is tensor-like.
- **Tolerance vs diagnostics**: non-tensor-like globals (scalars, strings, functions, imports) are skipped silently; a numeric nested list that is **ragged** (shape product ≠ leaf count) is a hard, named failure.
- **Naming normalization for runnability**: when *every* `state_dict()` entry alternates 2-D `*.weight` / 1-D `*.bias` (≥1 pair, full coverage — a trailing lone weight, 4-D conv weights, or 1-D BatchNorm tensors all break the rule), params are re-emitted as `layer{N}.weight/bias` with a running layer index across modules. This makes `Model::infer_architecture` → `mlp` and satisfies `layered_mlp_pairs` (contiguous layer ids, chained `cols == prev rows`). Bare globals matching `^layer(\d+)[._](weight|bias)$` normalize identically. Non-affine state dicts keep original names, prefixed with the variable name when multiple modules exist — which preserves HF `model.layers.*` / `transformer.*` names for llama/gpt2 inference.
- **Two-pass runpy**: `runpy.run_path(path, run_name="__modelc__")` first (main blocks dormant); re-run as `"__main__"` only if zero tensors — scripts that build their model inside a main block still compile, at the cost of executing the program twice.
- **Python detection**: try `python3` then `python`, requiring `status.success()` on `--version` — the Windows Store `python3` alias runs but exits 9009, so success-flag checks (not just spawn) are mandatory.
- **Subprocess capture**: `.output()` captures both streams; stderr is folded into the `anyhow` error (testable), stdout/stderr are echoed on success so script output isn't swallowed. `.output()` also makes child stdin reads EOF instead of hanging.
- **Encode fast path**: numpy `<f4`/`<i4` arrays when importable (one `asarray().tobytes()`), `struct.pack` fallback otherwise; all-int tensors (incl. bools) stay `I32`, everything else `F32` (the runtime is f32 anyway).
- **Perf note**: the stdlib writer materializes Python floats — fine for toy/medium models; for multi-million-param programs prefer exporting to a real Safetensors file.

### Limitation: 64 MiB Safetensors sniff cap
Larger Safetensors files fail sniffing and require explicit `-f safetensors`. Avoid raising this cap — full-file deserialization is O(file size) and slow.

## 2. `.modelc` Artifact Format

### Header schema (`src/pack.rs:1-100`)
```
[magic 8 bytes "MODELC\0\0"]
[u32 version (LE)]            — currently 2 (version 1 = uncompressed, version 2 = zstd-compressed)
[u32 flags (LE)]              — bit 0: compressed
[u64 header_len (LE)]         — JSON header byte length
[header_len bytes JSON]       — name, architecture, tensor_count, total_bytes, compressed flag
[tensor blob]                 — sorted by name, raw bytes (or zstd frame if compressed)
```

### Byte layout invariant
Tensors are stored in **sorted-by-name order** with no padding between them. `PackEntry { name, dtype, shape, byte_offset, byte_len }` is the authoritative index for offset math; do not recompute offsets from tensor sizes.

### Compression (`src/pack.rs`)
- **Version 2 only**: zstd-compressed blob. The JSON header stays uncompressed.
- **Detected at read time**: `flags & 1 != 0` → decompress with `zstd::decode::bulk`
- **No streaming write**: the entire blob is built in memory and compressed in one frame. For large artifacts this is a memory hot spot — prefer version 1 (uncompressed) when serving from disk anyway.

### Content digest convention (`src/digest.rs`)
- **`sha256:<hex>` over the FULL artifact file** (magic + version + flags + header + blob) — the bytes a client downloads. Matches the modelc-web store (`X-Modelc-Digest`, `ETag`, DB `digest`) and Ollama/Docker conventions, so `sha256sum` on a downloaded file reproduces the published digest.
- `Sha256` is hand-rolled streaming (update/finalize) — no crypto dep, used for integrity not keyed auth. **Gotcha**: `update()` must *append* at `buf_len` (a `buf_len = data.len()` assignment drops buffered bytes → wrong hash; a padding loop then hangs because `buf_len` never reaches 56).
- `rfc3339()` = epoch→`YYYY-MM-DDTHH:MM:SSZ` via civil-from-days (no `time`/`chrono` dep).
- Used by: `pull` digest verification, `/api/tags` + `/api/show` `digest` field, `verify` output.

### dequantize_in_place contract (`src/model.rs:dequantize_in_place`)
- Called by `modelc compile` and `modelc export` before codegen / safetensors write
- Operates on `&mut Model`, mutating each quantized `TensorData` to F32 in place
- Handles own INT8/INT4 (`quant_scale.*`/`quant_mode.*` metadata) AND GGUF block-quants (via `parsers::gguf::dequantize_gguf_tensor`)
- Idempotent — calling twice is a no-op on already-F32 tensors
- **Required**: codegen emits FP32-only helpers; quantized tensors would fail at runtime without this step
- **Export contract (learned from modelc-web)**: `export_to_safetensors` used to `panic!` on GGUF quants; now dequantizes to F32 first — a converted file that loads beats an error.

## 3. Runtime & Forward Passes

### Runtime::from_raw dequantization paths (`src/runtime/serve.rs`)
For each tensor:
- F32 / F16 / BF16 → cast to F32 (use `half::f16::from_le_bytes(...).to_f32()`)
- I8 / I16 / I32 / I64 → reinterpret as `as f32`
- U8 / Bool → cast to F32 (0.0 / 1.0)
- Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K → `dequantize_gguf_tensor(name, data)` returns FP32 bytes

### Transformer forward invariants (`src/runtime/transformer.rs`)
- `forward_gpt2_cached(input, runtime, cache) -> Vec<f32>` and `forward_llama_cached(...)` share the KV-cache protocol: `Option<&mut KvCache>` parameter
- **Zero tokens in → zero tokens out**: empty `input` returns `vec![]` (used by attention masks)
- Attention reads K/V via `KvLayer::k_all_cow` / `v_all_cow` — FP32 path is zero-copy (`Cow::Borrowed`), quantized paths materialize (`Cow::Owned`)
- `softmax_inplace` is allocation-free and is the **only** softmax the transformer uses

### KV cache layout (`src/runtime/transformer.rs`)
- `KvCache` holds `Vec<KvLayer>` — one per transformer layer (typically 12–32)
- Each `KvLayer` is one of three variants: `Fp32(Vec<Vec<f32>>)`, `Int8 { data, scales }`, `Mixed { cold_int8, hot_fp32 }`
- Append: `KvLayer::push(&mut self, k: &[f32], v: &[f32])`
- Concatenate across layers: `KvCache::k_all(&self, layer) -> Vec<f32>` (rarely used; `k_all_cow` preferred)
- Shift: `KvLayer::shift(&mut self, keep: usize)` removes oldest tokens; `KvCache::shift_anchored` keeps the first `anchor_tokens` and removes from the middle (StreamingLLM-style)

### Mixed-precision KV (`src/runtime/transformer.rs::KvLayer::Mixed`)
- Hot window = `hot_fp32: Vec<Vec<f32>>` (default 64 tokens)
- When `hot_fp32.len() > hot_window`, the oldest entry is quantized to INT8 and pushed to `cold_int8`
- `k_all()` / `v_all()` dequantize the cold prefix then concatenate with the hot suffix in chronological order
- Coverage: `kv_layer_mixed_keeps_recent_in_fp32`, `kv_layer_mixed_matches_fp32_output`, `kv_layer_mixed_shift_discards_oldest`

### SIMD GEMV kernel (`src/runtime/gemv.rs`)
- `gemv_dot(w: &[f32], x: &[f32]) -> f32` — K-dimension vectorized dot product
- NEON (aarch64): `f32x4` FMA
- AVX (x86_64): `f32x8` mul + horizontal add
- Scalar fallback (other targets / unstable features)
- **Fixed horizontal-reduction order** is required for bit-identical results across `run` and `compile`
- Shared source via `include_str!("gemv.rs")` in `src/codegen/native/helpers.rs` (`//!` doc lines stripped at emit time)
- **Anti-pattern**: do not call `.iter().zip().map().sum()` in the GEMV hot path — the compiler cannot autovectorize without fast-math; use the explicit indexed loop

### Runtime model_view memoization (`src/runtime/transformer.rs::Runtime::model_view`)
- `OnceLock<Model>` holds a synthetic `Model` with shape-only tensors for the arch inspector
- Built once per `Runtime`; free on subsequent calls
- Replaced wholesale only — LoRA load/unload via `*runtime = Runtime::from_raw(...)` swaps in a fresh lock with its own `OnceLock`
- Coverage: `model_view_is_memoized_and_exposes_shapes`, `model_view_refreshes_after_runtime_swap`

## 4. Serve Layer (HTTP)

### Endpoint-coverage test patterns (2026-11, `tests/transformer_runtime_test.rs`)
- **Streaming bodies are NOT de-chunked**: `common::http` reads the raw response — HTTP chunked-transfer framing (`"1A\r\n…\r\n"` + terminal `"0\r\n"`) interleaves with payload lines. Parse with an **object-lines filter** (`from_str(..).ok()` + `v.is_object()` drops framing numbers); SSE: strip `data: ` first, same filter. Existing SSE tests use substring asserts for this reason — prefer object filtering when asserting envelopes/`done` ordering.
- **4xx/5xx arrive as `Err(HttpError::StatusCode(s))`** — a status-assert test needs a `post_status` helper that matches the `Err` arm, not `.expect()`.
- **Previously-uncovered handler paths, now pinned**: batched `/infer` (per-item `run_inference`) with `profile: true` (the per-op timing eprintln), batched `/embeddings` (`embedding` field is `skip_serializing_if none` — absent by design in batch responses), `/complete` `json_schema` retry closure, `/chat` `grammar` constraint arm, `/chat/stream` full SSE loop (cancel flag + decode_deltas + final `done:true`), LoRA-load success (adapter safetensors tensor named after a base tensor, e.g. `weight`), `/reranking` zero-norm → `0.0` (non-transformer fixtures embed to empty vectors — cosine must return 0.0, never NaN).
- **Ollama options contract** (`options_config`): every field is assertable via `/api/generate` non-stream — `num_predict` (-1 = unbounded), temperature/top_p/min_p/penalties, `stop` as **string or array**, `num_ctx`. `format: "json"` → object schema; object schema → passthrough (both stream and non-stream).
- **Prompt assembly is precisely assertable**: `raw: true` bypasses the template **unless `system` is present** — with system, both messages render: `"system: SYS\n" + "user: hi\n"` = 21 bytes → 21 byte-fallback tokens → `prompt_eval_count` pins the assembly (not just presence).
- **`pack::pack` vs `compiler::pack` in fixtures**: `compiler::pack` runs `apply_arch_hint` (blank architecture gets inferred from tensor names!) — to test header-driven fallbacks use `pack::pack`, which preserves the header verbatim. `/api/show` family falls back to `"generic"` for empty `header.architecture`; an unknown quant label (header metadata `"quantization": "weird"`) falls through `ollama_file_type` to `0`.
- `quantization_level("int8")` → `"Q8_0"` is **intentional** Ollama-label mapping (INT8 reports as the nearest recognizable level), not a mislabel.
- **`cargo llvm-cov` wipes `target/debug`** — run it after (not between) normal builds; it executes the full suite, so stale docs_sync markers abort it.

### MCP-compatible stdio transport (`src/serve/stdio.rs`, 2026-11)
`modelc serve --stdio` = newline-delimited JSON-RPC 2.0 on stdin/stdout, wire-compatible with the MCP stdio transport.
- **stdout is a protocol channel**: never `println!`/`eprintln!` from the serving path — logs go to stderr, every response line is flushed immediately. A stray `println!` corrupts a JSON-RPC client silently.
- **Sync, not async**: the stdio loop is a blocking `stdin.lock().lines()` loop; the inference internals (`run_text_inference_with_config`, `run_embeddings`) are already sync, so no tokio is needed on this path. `AppState::for_test` is the *production* bootstrap too (build_router uses it) — stdio reuses it directly.
- **MCP error semantics**: tool failures are **results** with `isError: true` (unknown tool, invalid arguments); `-32601`/`-32600`/`-32700` are reserved for protocol violations. Notifications (no `id`) never get a response — including for unknown methods.
- **`initialize` protocol-version echo**: echo the client's requested version when it is in the supported list (`2024-11-05`, `2025-03-26`, `2025-06-18`), otherwise reply with the server's latest.
- **Reuse over duplication**: tool arguments deserialize into the same `types::ChatRequest`/`CompleteRequest`/`EmbeddingsRequest` DTOs the axum handlers use, and generation params go through `handlers::make_generation_config` — one code path per behavior, two transports.
- **Tool-schema DSL**: build JSON-Schema `inputSchema`s with `serde_json::json!` + a tiny `merge_object` helper (sampling properties shared between `chat` and `complete` tools).
- **Child-process test pattern**: `env!("CARGO_BIN_EXE_modelc")` + `Stdio::piped()` stdin/stdout, `stderr(Stdio::inherit())`, helper fns `send_msg`/`read_msg`, `child.kill()` + `wait()` **before** assertions that might panic — a panicked test otherwise leaks a live child holding its port. Also: `common::http` retries `TimedOut` 10× with a 30s read timeout (300s worst case) — when polling a starting server, rely on connection-refused being non-retryable, never on read timeouts.

### MCP Streamable HTTP transport (`POST /mcp`, `src/serve/mcp_http.rs`, 2026-11-25)
The MCP specification's *current* standard transport (it replaced HTTP+SSE in `2024-11-05`); stdio is the other one. It is what makes a **hosted** modelc reachable by MCP clients rather than only a local subprocess.
- **Shared core**: the JSON-RPC handling was extracted out of `stdio::handle_message` into `stdio::handle_value(state, &Value)` — the stdio transport parses a line and calls it, the HTTP transport parses a body and calls it, so the two transports cannot answer differently. New MCP methods belong in `handle_value`, never in a transport.
- **Spec requirements implemented**: `202 Accepted` with no body for notifications and client responses; `Mcp-Session-Id` issued at `initialize` and echoed on later requests (an id we did not issue → `400`); `MCP-Protocol-Version` validated on every request (unsupported → `400`); `Origin` validated (same-origin **or** both sides loopback → allowed, otherwise `403`); an unparseable body → `400` carrying a JSON-RPC `-32700` body; `GET`/`DELETE` → `405`, which the spec allows for a server with no server→client stream.
- **Origin validation is the DNS-rebinding defense**: compare the Origin's host with the `Host` header, allowing loopback↔loopback. Strip the scheme and the port first; IPv6 literals arrive as `[::1]`.
- **Session ids without a uuid crate**: 16 bytes from `/dev/urandom` (unix) shaped as a UUID v4 (version/variant bits set), with a clock+counter fallback elsewhere. The spec says "SHOULD be cryptographically secure"; the UUID shape is what clients expect.
- **Test seam**: the handlers take `State`, `HeaderMap` and `Bytes`, so unit tests call them directly and read the body with `axum::body::to_bytes` — the repo has no `tower::ServiceExt` dependency, and the integration test drives the endpoint against the real binary instead.

### Router assembly (`src/serve/mod.rs:251-291`)
33 routes mounted via `Router::new().route(...)` + `with_state(Arc<AppState>)` (counted 2026-11-25; `/v2/rerank` was the latest). Backpressure middleware (`backpressure_middleware`) exempts `/health`, `/info`, `/props`, `/metrics` so probes and observability work under load.

### State sharing
- `Arc<AppState>` cloned into every handler
- `runtime: RwLock<Runtime>` — write-locked only on LoRA load/unload
- `prefix_cache: RwLock<PrefixCache>` — read-locked for `lookup`, released during generation, write-locked for `insert`
- `metrics: Metrics` — atomic counters + RAII guards

### Cancellation (`src/serve/mod.rs::CancelOnDrop`)
- `GenerationConfig.cancel: Option<Arc<AtomicBool>>` checked once per token in `generate_core` / `speculative_generate`
- SSE response wrapped in `CancelOnDrop` whose `Drop` impl sets the flag (client disconnect trips it)
- Drip loops break when the channel closes
- Coverage: `cancel_flag_defaults_to_not_cancelled`, `cancel_flag_set_is_cancelled`, `cancel_on_drop_sets_flag`, `cancellation_flag_stops_generation`

### Auth + backpressure (`src/serve/auth.rs`)
- `--api-key`: per-request `Authorization: Bearer <key>` check (axum middleware). Exempts `/health`, `/info`, `/props`.
- `--rate-limit`: per-IP token bucket (`src/serve/auth.rs`), returns `429 Too Many Requests` over the cap
- `--max-concurrent`: `tokio::sync::Semaphore`, returns `503 Service Unavailable` when saturated
- **Anti-pattern**: don't wrap `/metrics` in auth or backpressure — Prometheus scrapers don't carry auth headers

### Prefix cache management (2026-09)
- `GET /api/cache/stats` → `PrefixCache::stats()` serialized as `CacheStatsResponse` (`src/serve/mod.rs`)
- `POST /api/cache/clear` → `PrefixCache::clear()`, reports cleared entries
- Both exempt from auth/backpressure (probes + dashboards must work under load)
- Testing: `AppState::for_test(model, profile, generation, auth, max_concurrent)` builds a synthetic `Arc<AppState>`; `modelc::serve::build_router_from_state(state)` wraps it in a router. This avoids the `run_server` bootstrap and lets tests bind an ephemeral port via `tokio::net::TcpListener::bind("127.0.0.1:0")` + `drop(listener)` then pass the same `SocketAddr` to `run_server`.
- **Pattern**: when adding a new endpoint, add the route in `build_router_from_state`, a handler in `handlers.rs`, the response struct in `mod.rs`, a SPEC.md entry, and one integration test in `tests/cache_endpoints_test.rs` (or `runtime_serve_test.rs`)

### Cohere-compatible rerank (`/v1/rerank` + `/v2/rerank`, `src/serve/openai.rs`, 2026-11-25)
- **One handler, two paths**: `cohere.Client` POSTs `{base_url}/v1/rerank`, `cohere.ClientV2` (Cohere's current default — its "Private Deployment" quickstart is `ClientV2(base_url=...)` + `co.rerank(...)`) POSTs `{base_url}/v2/rerank`. Both SDKs append their own path, so mount the same `openai::rerank` on both routes — a v2 client against a v1-only server gets a 404.
- **Wire contract (verified against Cohere's v1 + v2 reference, 2026-11-25)**: request `{model, query, documents, top_n?, return_documents?}`; v1 `documents` are "document objects or strings" (v2: strings only) ranked on the `text` field; response `{id, results: [{index, relevance_score, document?}]}`; `relevance_score` is documented as **normalized to `[0, 1]`**; `document` is present only when `return_documents` is true, with extra object fields preserved. `id` / `meta` are optional in both versions.
- **Score = cosine clamped, ordering by raw cosine** (`rank_results`): sort on the raw `[-1, 1]` similarity → truncate to `top_n` → `clamp(0.0, 1.0)` the reported score. Clamping *after* ordering keeps ≤ 0 documents ordered among themselves and keeps positive scores identical to the native `/reranking` endpoint. **Anti-pattern**: do not rescale `(cos + 1) / 2` — it maps unrelated documents (cos ≈ 0) to 0.5, which silently passes the score thresholds RAG pipelines apply. The magnitude is not a calibrated probability; only the ordering is meaningful (stated in SPEC).
- **Documents as an untagged enum**: `RerankDocument::{Text(String), Object(serde_json::Map)}` with `text()` → `Option<&str>` (`None` for an object with no *string* `text` → `400 {"message"}`) and `echoed()` → `Value` (`{"text": s}` for strings, the original object for objects). `model`, `rank_fields`, `max_chunks_per_doc`, `relevant_spans` are accepted and ignored.
- **Testable core over handler tests**: the GPT-2 fixture cannot be made to emit a negative cosine on demand, so the `[0, 1]` contract is pinned by inline tests on the pure `rank_results` (`rank_results_orders_by_raw_similarity_then_clamps` feeds `-0.5` / `1.4`) plus DTO deserialization tests; the integration tests pin `id` (`rerank-{name}` with the name read from `/info`), the range, object documents, the 400, and v1/v2 parity through the running server. `post_status` surfaces only the status code (the test client drops 4xx bodies) — assert bodies through the 200 path.
- **Resuming an interrupted loop**: this endpoint arrived as code + one test with the TODO box unchecked and no docs. Before implementing any TODO item, `grep` the tree for its keywords and check `git status` — an earlier session may have landed part of it. An unchecked box is not evidence the work is absent.

### MCP resources (`resources/list` + `resources/read`, `src/serve/stdio.rs`, 2026-11-25)
Read-only model metadata over the same stdio session, so an agent can inspect the artifact it is talking to.
- **URIs**: `modelc://model` (name, architecture, tensor count, parameter/byte totals, quantization label, artifact metadata), `modelc://model/tensors` (name, shape, dtype, element count, stored bytes — **sorted by name** so the output is deterministic; `base_tensors` is a `HashMap`, so an unsorted listing would flap between runs), `modelc://model/tokenizer` (model type, BOS/EOS ids, chat-template size, raw `tokenizer.*` entries).
- **Envelope**: contents use `{contents: [{uri, mimeType: "application/json", text}]}` — the payload is **JSON text inside `text`**, not a structured object; metadata values are capped at 1024 chars.
- **Error codes**: `-32002` unknown `uri`, `-32602` missing `uri` param — JSON-RPC *errors*, unlike tool failures which are `isError: true` results.
- **Gate the capability by protocol version**: resources entered MCP in `2025-03-26`, so `initialize` advertises `capabilities.resources` only when the negotiated version is ≥ that — a `2024-11-05` client must not be pointed at a method it does not know.
- **`AppState` now keeps `metadata`** (`HashMap<String, String>`): the parsed weights' metadata (GGUF KV pairs / Safetensors `__metadata__`) used to be dropped right after parsing, which made the tokenizer info unreachable from every server.
- **Adding an MCP method touches the error tests**: the "unknown method" cases had used `resources/list` as their example of an unknown method — a method that becomes known invalidates them, so re-point them (e.g. at `prompts/list`).
- **MCP tools** (`tools/list` + `tools/call`): `chat` / `complete` / `embeddings` / `rerank`. A tool result is **JSON text inside the content envelope** (`{content: [{type: "text", text}], isError}`) — parse `text` as JSON rather than expecting a structured object. Tool failures are `isError: true` results; only protocol violations are JSON-RPC errors.
- **One core per behavior**: the MCP `rerank` tool and the HTTP `/v1/rerank` endpoint both call `openai::rerank_documents` (embed → cosine → order → clamp). The HTTP handler used to inline its own embedding loop, which would have let the two surfaces drift. When adding a tool, put the shared logic in the module that owns the behavior and call it from both transports.

### Competitive surface (research pass 2026-11-25)
The bar we are measured against — MCP `2025-06-18` transports spec, the llama.cpp server README and vLLM's online-serving docs, all fetched 2026-11-25:
- **MCP's standard transport is Streamable HTTP**, not stdio: one endpoint taking POST (JSON or SSE response) and GET (SSE for server→client messages), the `Mcp-Session-Id` header for session state, `MCP-Protocol-Version` on every request, `202` for notifications/responses, and `Origin` validation against DNS rebinding. **We ship stdio only.**
- **Anthropic Messages API** (`/v1/messages`, `/v1/messages/count_tokens`) is shipped by both llama.cpp and vLLM — Claude Code and the Anthropic SDKs speak only that dialect.
- llama.cpp also has: a router server (`--models-dir`, LRU unload), `--reasoning-format` → `message.reasoning_content`, samplers we lack (top-k, typical-p, dynamic temperature, mirostat, DRY), SSE keep-alive pings, CORS handling, and direct Hugging Face downloads.
- vLLM also has: the OpenAI **Responses API** (`/v1/responses`), the **Batch API**, Cohere `/v2/embed`, pooling APIs (`/classify`, `/score`, `/pooling`), `/tokenizer_info`, and sleep mode.
The nine seeded gaps, each with a success criterion, live in TODO.md's "2026-11-25 research pass" section.

### Streaming protocol
- All streaming endpoints use SSE (`text/event-stream`)
- `ReceiverStream<Result<Event, Infallible>>` is the body
- For OpenAI streaming: `data: {json}\n\n` chunks; `data: [DONE]\n\n` terminator
- For `/chat/stream`: JSON-encoded `StreamChunk { delta, done }` per chunk (not OpenAI-shaped)

## 5. Codegen & Compile

### NativeCodegen::generate output layout (`src/codegen/native/mod.rs`)
```
<output_dir>/
├── modelc_build/                  # temp crate
│   ├── Cargo.toml                 # generated by generate_cargo_toml
│   ├── src/main.rs                # generated, ~400 lines
│   └── embedded_weights.bin       # authoritative weight bytes
└── <output>                       # final binary, copied from target/{...}/model-serve
```

### Embedded weights blob (`src/codegen/native/mod.rs::generate`)
- Concatenated tensor bytes in sorted-by-name order
- Streamed straight to disk via `BufWriter` — no in-memory mirror copy
- Indexed in the emitted `main.rs` via `static OFFSETS: &[(u32, u32)] = &[...]` (one entry per tensor)
- Reads in the emitted binary via `include_bytes!("../embedded_weights.bin")`

### Shared kernel embedding (`src/codegen/native/helpers.rs`)
- `include_str!("../../runtime/gemv.rs")` embeds the runtime GEMV kernel verbatim
- `//!` module-doc lines stripped at emit time (they would be invalid mid-file)
- **Anti-pattern**: never inline-edit the runtime GEMV; if you need a different reduction, add a new function alongside

### compile_listen resolution (`src/cli.rs:354-372`)
`--listen ADDR:PORT` overrides `--bind IP` + `--port N`. IPv6 supported via `[::1]:8080`. The resolved `SocketAddr` is embedded as a `Display`-formatted string literal in the emitted `main.rs`; the binary parses it back at startup.

## 6. OpenAI API Compatibility

### Field shapes (`src/serve/openai.rs`)
- `/v1/chat/completions`: `model`, `messages[]`, `stream`, `response_format`, `tools`, `tool_choice`, `max_tokens`, `temperature`, `top_p`, `min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs`, `top_logprobs`, `seed`, `stop`, `stream_options`, `user`, `max_completion_tokens`, `n`, `name` (per-message)
- `/v1/completions`: same minus `messages` / `tools` / `tool_choice` / `name`; plus `prompt` (string or string[]), `echo`, `best_of`, `suffix`
- `/v1/embeddings`: `input` (string or string[]), `encoding_format` (`"float"` or `"base64"`), `model`, `dimensions`
- `/v1/models`: `{ object: "list", data: [{ id, object: "model", created, owned_by }] }`
- `/v1/models/{id}`: 404 with `{ error: { type: "model_not_found", message } }` on miss

### Streaming chunk shape (OpenAI)
```
data: {"id":"…","object":"chat.completion.chunk","created":N,"choices":[{"index":0,"delta":{"role":"assistant"|"content":"…"},"finish_reason":null|"stop"}]}

data: {"id":"…","object":"chat.completion.chunk","created":N,"choices":[],"usage":{"prompt_tokens":N,"completion_tokens":N,"total_tokens":N}}

data: [DONE]
```
- `[DONE]` is the literal string — never omit it
- The final usage chunk is emitted only when `stream_options.include_usage = true`
- `finish_reason: "stop"` on the last chunk with content; `null` on intermediate chunks

### Sampling parameter precedence
Per-request value > server default (`GenerationConfig`) > CLI default. Resolution lives in `make_generation_config` (`src/serve/handlers.rs`).

### Sampling fast path
When all of `repetition_penalty == 1.0`, `presence_penalty == 0.0`, `frequency_penalty == 0.0`, `logit_bias.is_empty()`, and `constraint.is_none()`, the sampling path skips the per-token seen-token allocation and applies raw logits directly. `sample_constrained` checks `gen_cfg.fast_path()` once per call.

### Generation loop progress must be monotonic (2026-11 bug found by coverage tests)
- **Never** measure progress as `token_ids.len() - prompt_len` when `context_shift` is in play — `context_shift` shrinks *both* `token_ids` and `prompt_len`, so the difference can reset every shift → **infinite loop** when `max_context` < prompt+max_tokens (latent bug in `speculative_generate`, caught by `speculative_applies_context_shift` hanging 15 min).
- Correct pattern: `let mut generated = 0usize;` + `generated += 1` per pushed token + `while generated < config.max_tokens` — same as the non-speculative loop in `generate.rs`.
- Related: a draft-verify loop that pushes `gamma` tokens per round can overshoot `max_tokens` — guard `generated >= max_tokens` *inside* the verify loop, not just at the `while` bound.
- Coverage-test technique that found it: feed the *smallest* knobs extreme values (`max_context` shorter than the prompt, pre-tripped cancel flag) — cheap tests that exercise rarely-hit branches.

### Covering `serve/infer.rs` dispatch branches (2026-11)
- `state.onnx_plan` is populated from `model.metadata["onnx.execution_plan"]` (`ExecutionPlan::to_json()` round-trip) — inject a plan into a fixture `Model` to exercise `runtime_to_tensor_data` + `execute_plan` over real HTTP. An `Identity` plan covers the Ok path; an op referencing a missing tensor covers the Err→eprintln→fallback path (echo when no mlp/transformer fallback exists).
- `mlp_plan` comes from `infer_mlp_plan(&model)`: `create_large_test_model`'s `layer{0,1}.{weight,bias}` → `layered_mlp_pairs` → multi-layer plan → `relu_inplace` fires on intermediate layers. `/infer` `{inputs: [a]}` → `run_mlp_forward`; `{inputs: [a, b]}` → `run_mlp_forward_batched`. `profile: true` in `run_server` covers every timing `eprintln`.
- `gamma > 0` isn't a request field — it comes from the server's default `GenerationConfig` (4th arg to `run_server`). Boot the test server with `GenerationConfig { gamma: 2, .. }` to drive every `/chat` + `/complete` through `speculative_generate`.
- `run_text_inference_with_logprobs` returns `None` early on non-transformer arch — `/v1/completions` with `logprobs: true` on an `"mlp"` model covers that + the MLP text-inference fallback in one shot.
- llama serve branches (`forward_llama`, `embed_llama`, arch filters) need a llama `AppState` — `common::create_llama_test_model` + `run_server` covers them; there was no llama serve test before this pass.

### Covering Metal-gated CPU fallbacks (`src/runtime/ops.rs`, 2026-11)
- Every pub op is `#[cfg(macos)] Metal try → CPU fallback`. On Metal hardware the GPU path always succeeds → CPU fallbacks look dead to llvm-cov even though they're the production path elsewhere.
- **Fix = name the kernels** (the file's own `matmul_cpu` convention): `add_cpu`/`mul_scalar_cpu`/`softmax_cpu`/`relu_cpu`/`layer_norm_cpu` as private fns the pub wrappers delegate to. `#[cfg(test)] mod tests` in the same file calls them directly — no env-var seam, no prod behavioral change.
- Same for `matmul_row` — the scalar row kernel now shared by the rayon `par_chunks` arm and the small-matrix serial arm (dedup, not just coverage).
- `unsafe fn matmul_neon` is callable from tests under `unsafe {}` on aarch64 (runtime `is_aarch64_feature_detected!` gate stays in prod); `matmul_avx` is `cfg(x86_64)` — unreachable here by design.
- Fixture style: `t(&[f32], &[usize])` shorthand + hand-computed reference product (`[[1,2,3],[4,5,6]] @ 3×4` → `[74,80,86,92,173,188,203,218]`); `should_panic` for the incompatible-shapes arm.

### GGUF cursor coverage + DoS guard (2026-11)
- Private helpers (`read_kv_array_summary`, `consume_kv_value`) aren't callable from `gguf/tests.rs`, but both are driven via `read_kv_value_formatted(VAL_ARRAY)`: an array >32 elements truncates at `preview=32` and pushes the rest through `consume_kv_value`. One 33-elem array per scalar type covers every consume arm; a 33-elem array *of arrays* covers the nested VAL_ARRAY arm.
- **Parser-robustness rule**: never `Vec::with_capacity(n)` on a parsed length without a feasibility check — `read_string_array` now rejects `n > remaining()/8` (each string has an 8-byte header). Audit other `read_*` loops for the same pattern when adding parsers.
- Watch for **stub tests**: `#[test] fn x() { /* comment */ }` with no assertions counts toward the test tally but covers nothing — `docs_sync` counts them, coverage reports reveal them.

### n / best_of clamping
- `n` clamped to `1..=20` via `.clamp(1, 20)`
- `best_of` clamped similarly
- `top_logprobs` clamped to `0..=20`
- `top_p` / `min_p` clamped to `0.0..=1.0`

### `created` + `system_fingerprint` (OpenAI protocol, Sep 2026)
- Every `/v1/*` response and SSE chunk carries `created: unix_now()` — a real Unix timestamp in seconds. OpenAI clients (and log pipelines) treat epoch `0` as malformed; never hardcode it.
- `system_fingerprint` is `fp_modelc_<git-sha>` (`crate::GIT_SHA` from `build.rs`). Required field on non-streaming `ChatCompletionResponse` / `CompletionResponse`; `Option<String>` + `skip_serializing_if = "Option::is_none"` on chunks.
- **Anti-pattern**: adding a field to a response struct requires updating *every* initializer — there are ~11 construction sites in `src/serve/openai.rs`. Run `cargo build` and fix all `E0063` missing-field errors before committing.

### Ollama ecosystem schema (`/api/tags`, `/api/show` — learned from modelc-web)
- **Pattern**: implement the **ecosystem's** schema (what real clients parse), not just our own dialect: `name`, `model`, `modified_at` (RFC3339 mtime), `size`, `digest` (`sha256:<hex>`), `details.{format,family,families,parameter_size,quantization_level}` — plus `model_info` (`general.architecture/parameter_count/file_type`) and `capabilities` on `/api/show`. Keep modelc fields as extras — clients ignore unknown keys.
- **Anti-pattern**: a response that is only *superficially* Ollama-shaped (old `details.{architecture,parameter_size,quantization}`) breaks real clients silently.
- `/api/show` accepts `{"model": ...}` (Ollama) AND `{"name": ...}` (legacy) — both `Option<String>` in `ApiShowRequest`.
- Helpers live in `src/serve/ollama.rs` (port of modelc-web `api/_ollama.ts`): `format_parameter_size` ("8.0B"/"134.52M"/"1.5K"), `quantization_level` (modelc label → Ollama level), `ollama_file_type` (llama.cpp enum), `artifact_format` (`metadata.format` else "modelc").
- Quantization for tags/show comes from `pack::infer_quantization(header)` — checks `metadata["quantization"]` → `quant_mode.*=int4` → `quant_scale.*` → GGUF quant dtype → default `f32`.
- `InstalledModel` carries `format`/`quantization`/`tensor_count` (populated by `list_models` from the header it already reads — don't re-parse).

## 7. Testing Patterns

### Float comparisons
- **Bit-identical expected values**: use `assert_eq!` (transformer forward, GEMV)
- **Tolerated approximate**: `(actual - expected).abs() < 1e-5` for SIMD-rounded paths
- **Anthropic rule**: never compare `f32` from different rounding orders without acknowledging it

### Fixture tensors
- `tests/common/mod.rs::create_test_model` — 2-layer MLP (2×3 weight + 2 bias)
- `tests/common/mod.rs::create_large_test_model` — 64-dim MLP with ln params
- `tests/common/mod.rs::create_gpt2_test_model` — 1-layer GPT-2 with vocab=10
- `tests/common/mod.rs::create_llama_test_model` — 1-layer LLaMA with vocab=10

### Server-endpoint smoke tests
- Use `tests/common/http.rs::post_json` / `http::get` for binary-spawned server hits
- Or call the handler directly via `modelc::serve::build_router(state)` and use `axum::body::to_bytes` for assertions

### Network-dependent tests without external services
- `store::download` / `http_get` tested against a **local one-shot TCP server**: `std::net::TcpListener::bind("127.0.0.1:0")`, spawn a thread that accepts one connection, writes a canned `HTTP/1.1 200 OK\r\nContent-Length: N\r\n\r\n<payload>` response, closes. See `serve_one_http_response` in `tests/store_test.rs`. Covers success, error-status (404), and URL-scheme rejection without touching the network.
- Onnx external-initializer tests write `.bin` payloads next to a temp `.onnx` and encode a real protobuf via `onnx_rs` (`tests` in `src/parsers/onnx/mod.rs`).

### LoRA fixture gotcha
- `safetensors::TensorView` **borrows** its data buffer — buffers must outlive the views. Collect data `Vec<u8>`s into a separate `Vec` first, then build views in a second loop (`make_f16_lora_safetensors` in `tests/lora_test.rs`).
- `safetensors::serialize(&metadata, &None)` produces the file bytes; write to disk for `apply_lora` tests.

### Ollama inference endpoints (`src/serve/ollama.rs`)
- `/api/generate` + `/api/chat` are what Ollama **client libraries** actually drive; tags/show are discovery-only. Request `options` maps onto `GenerationConfig` (`num_predict`→max_tokens with ≤0 = server default, `repeat_penalty`→repetition_penalty, `num_ctx`→max_context; `stop` accepts string OR array — handle as `serde_json::Value`).
- **Ollama streams NDJSON, not SSE**: `Body::from_stream(ReceiverStream<String>)` + `Content-Type: application/x-ndjson`, one JSON object + `\n` per token, final line carries `done:true` + stats (`eval_count`, `prompt_eval_count`, `total_duration` ns, `context` token ids). Wrap the stream in `CancelOnDrop` like the SSE paths for disconnect cancellation.
- Test gotcha: streamed bodies arrive **chunked-transfer encoded** — chunk-size hex lines appear in the raw body. Filter `line.starts_with('{')` before parsing NDJSON (SSE tests use substring `contains` for the same reason).
- `ActiveRequestGuard` borrows `&state.metrics` — its `Drop` keeps the borrow alive, so `ndjson_stream(state.clone(), …)` (clone the `Arc`) instead of moving `state`.
- `/api/embed` (current) + `/api/embeddings` (legacy) complete the trio real clients need. `/api/embed` `input` accepts string OR array — reuse `stop_strings` for the shape-normalize trick; batch returns `embeddings: [[f32]]` + `prompt_eval_count` (tokenizer token count). Legacy `/api/embeddings` takes `prompt` and returns bare `{embedding: [f32]}`. Both reuse `run_embeddings` + the same `b as f32 / 255.0` byte encoding as `/embeddings` (`embed_input` helper).

### Module splitting patterns (2026-11 large-file modularization)
- `foo.rs` + `foo/bar.rs` coexist in edition 2021 — `mod bar;` inside `foo.rs` resolves to `foo/bar.rs`; no need to convert to `foo/mod.rs`. Used for `serve/openai/types.rs` and `generate/{sampling,speculative}.rs`.
- **Preserve public paths with re-exports**: moved `KvCache`/`KvLayer` to `runtime/kv.rs` and kept `crate::runtime::transformer::KvCache` working via `pub use crate::runtime::kv::{KvCache, KvLayer};`. Same for `pub(crate) use` of `generate::{argmax,softmax,multinomial,apply_top_p,speculative_generate}` (draft.rs, serve/infer.rs, transformer tests).
- **DTO visibility in sibling modules**: items in `serve/types.rs` used by `serve/handlers.rs` (a sibling) need `pub(super)` on struct AND fields; inside `serve/openai/types.rs` use `pub(in crate::serve)` to match the old `pub(super)`-in-openai visibility exactly. A private `use types::*` in `mod.rs` is visible to descendant modules — `handlers.rs`'s `use super::{ChatRequest}` keeps working unchanged.
- **Test-helper visibility**: private helpers used by `#[cfg(test)] mod tests` in the parent file need `pub(super)` in the moved module + `use submodule::*` in the parent; the parent's private `use` is visible to the test module via `use super::*`.
- **Unused-import choreography**: an item only used by the parent's tests wants `#[cfg(test)] use submodule::*`; a `pub(crate)` item with no crate-internal callers needs no re-export (rustc flags unused `pub(crate) use`).
- **Orphan gotcha**: start cuts at the `#[derive]`/`#[allow]`/doc-comment line, not the `struct`/`fn` line — attributes and doc comments left behind reattach to the next item (`#[derive(Clone)]` stranded on a `pub use` = E0774).
- Moved `mod tests { use super::*; ... }` bodies to `tests.rs` sibling files (gguf) work verbatim — `mod tests;` file IS the module body, just unindent one level.

### Streaming text deltas (`BpeTokenizer::decode_deltas`)
- **All streaming endpoints must use `tokenizer.decode_deltas(prefix_ids, token_ids)`** — never hand-roll the decode-cumulative/strip loop. Byte-level BPE merges cross the prompt boundary, so deltas come from `decode(prefix + gen[..i])` minus the previous decode.
- **Seed is `decode(prefix_ids)`, not `""` or the raw prompt**: with `""` the first "delta" is the entire prompt text (latent bug fixed 2026-11); the raw prompt can fail to strip under lossy decode. `decode(prefix_ids)` is a guaranteed prefix of every cumulative decode.
- Regression guard: stream tests should use a distinctive prompt marker and assert it never appears in chunk deltas.
- Shared OpenAI sampling params live in `serve/openai.rs::SamplingParams` (serde `#[serde(flatten)]` into both request structs — JSON surface unchanged) with `to_generation_config(&base)`; don't re-declare the 12 fields per endpoint.

### Self-improvement round-trip (`tests/roundtrip_alignment_test.rs`)
- Pattern for testing the full artifact loop under mutation: define a **code spec** (linear `y = X·W* + b*`), then each iteration `pack` → `verify` → `unpack` → `ops::linear` inference → closed-form gradient step. Asserts three things at once: artifact round-trip is lossless (F32 byte equality every cycle), weights actually evolve (`digest::digest_file` differs across cycles), and loss decreases monotonically to < 1e-4.
- Pick well-conditioned inputs (`(i % 5) * 0.5 - 1.0`) and conservative LR (0.3) — deterministic, no RNG, converges in <80 iters. Covers both `compress=false` and `compress=true` paths.

### KV cache regression cases (in `src/runtime/transformer.rs`)
- `kv_layer_fp32_push_extends_capacity`
- `kv_layer_int8_roundtrip`
- `kv_layer_mixed_keeps_recent_in_fp32`
- `kv_layer_fp32_shift_anchored_preserves_first`

## 8. CLI & Tooling

### WeightFormat sniffing precedence (`src/cli.rs:402-428`)
1. Extension match (`.safetensors`, `.gguf`, `.onnx`, `.pt`/`.pth`)
2. `.bin` substring heuristic (`.bin` + `ggml` → Gguf; `.bin` + `pytorch` → Pytorch; otherwise sniff)
3. Full sniff path (see § 1)

### Local model store layout (`src/store.rs`)
- `dirs::data_local_dir() / "modelc" / "models"` (platform-specific)
- Versioned copies are `<name>.v<N>.modelc`
- The "active" version is `<name>.modelc` (no `.vN`)
- `cp <src> <dest>` refuses to overwrite an existing destination
- `rm <name>` refuses to delete the main artifact if `.vN` copies exist unless `--all` or `--force`

### `modelc pull` — HTTPS + auth + digest (2026-11, learned from modelc-web)
- `download_auth(url, name, token)` → `http://` via hand-rolled `http_get` (unchanged), `https://` via the **system `curl`** (`-fsSL -D hdr -o out`) — no TLS stack in the binary. curl is preinstalled on macOS/Linux/Windows 10+.
- **Auth**: `--token` flag or `MODELC_TOKEN` env → `Authorization: Bearer` on both transports. Lets the CLI pull from token-gated endpoints like `https://www.modelc.net`.
- **Digest**: `X-Modelc-Digest` (or `ETag`) `sha256:<hex>` on the response is verified against the downloaded bytes — mismatch fails the pull. Verified end-to-end against the real store.
- **Anti-pattern**: don't add `reqwest`/`rustls` for https pulls — Footprint is a scored dimension and curl is already everywhere.

### `modelc diff` — structural header diff (`src/diff.rs`)
- Port of modelc-web `api/_diff.ts`: tensors added/removed/changed (shape|dtype|bytes), unchanged counts, metadata added/removed/changed, arch/params/size/tensor-count deltas, `identical` flag.
- Compares `ArtifactHeader`s only (via `store::resolve_model_path` + `pack::read_header`) — never loads tensor data. `MAX_LISTED = 50` caps per-list output.
- CLI: `modelc diff <a> <b> [--json]`; `diff_command` resolves store names OR paths for both inputs.

### `modelc prompt` one-shot generation (2026-09)
- `compiler::prompt_model` (`src/compiler.rs`) is the library entry; `Commands::Prompt` is the clap arm. **No HTTP server** — this is the Ollama `ollama run MODEL "…"` equivalent.
- Hidden size comes from `arch::transformer_hidden_dim` (`src/arch.rs`), the same helper `AppState` uses for `/complete`. Do not re-infer layers in the CLI arm.
- Transformer (`gpt2`/`llama`) → `generate()` decoded text. Anything else → JSON array of `byte as f32 / 255.0`, matching the non-transformer `/complete` fallback. Changing one path without the other is a wiring bug.
- Coverage: `prompt_model_mlp_echoes_byte_normalized_json`, `prompt_model_gpt2_returns_generated_text`, `cli_prompt_runs_on_packed_artifact`, `cli_prompt_missing_artifact_errors`.

### Shell completion (deprecated)
Previously `modelc completions <shell>` via `clap_complete`. Removed in v0.1.7 when `clap_complete` was dropped to slim dependencies. Users rely on shell completion loaders or `clap`-style argument completion.

### Bin-helper correctness (`src/bin/modelc.rs`, 2026-11)
- **`benchmark_inference` now mirrors the serving math**: it fed `[1, in] @ [out, in]` into `ops::matmul`, which (a) panics via `matmul_cpu`'s `assert_eq!(a.shape[1], b.shape[0])` on any rectangular layer (a `[1, 2]` second layer with input length 2 crashes `modelc bench`) and (b) measures a transposed matmul, not the real runtime work. Fixed to the shared `gemv::gemv_dot` row-major `[out, in]` kernel + zero-padding (mirroring `serve::infer::gemv_bias`) — regression tests pin serving-equal values (`bench_fixture` → 1.3).
- **Bin-target unit tests** (`#[cfg(test)] mod tests` in `src/bin/modelc.rs`) run under plain `cargo test` and count toward the docs_sync inline marker — they cover `generation_from_flags` (every override + invalid-grammar rejection), `load_artifact` (absolute paths skip the store), `run_benchmark` end-to-end, `benchmark_inference` branches (layer stack / singleton / zero-pad / 2-D fallback), and `print_model_list`. Dispatch arms stay covered by the child-process suite; llvm-cov **does** count spawned-binary coverage when the integration suite runs the instrumented bin (`--bin modelc` alone does not — scoped runs report ~54%, full-suite runs ~84%).
- **`cargo llvm-cov` wipes `target/debug`** and its instrumented target dir grows ~1 GB per full run — disk exhaustion mid-build surfaces as confusing `No such file or directory` rustc errors; clean `target/llvm-cov-target` and check `df` first. Scoped runs (`--bin modelc`) build much smaller but measure only that target's own tests.

### Rust edition 2021 — no let-chains (2026-11)
- `Cargo.toml` pins `edition = "2021"` + `rust-version = "1.90"` (verified via `cargo +1.90.0 check --all-targets`). **Let-chains (`if let A && let B`) are edition-2024 syntax — do not reintroduce them.** Desugar to nested `if`s, or collapse `Option` chains with `.and_then()` / `.filter()` / `.is_some_and()` when all bindings feed a single `Option` (e.g. `MetalBackend::new().and_then(|mb| mb.matmul_gpu(a, b))` in `src/runtime/ops.rs`, the `transformer_hidden` gates in `src/serve/infer.rs`).
- Desugaring touched ~16 files; patterns used: nested `if let` (ownership preserved), `(Some(a), Some(b))` tuples (`src/generate.rs` cache insert), `.filter()` for value predicates (`src/serve/auth.rs`).

### Exact-text edits in `serve/openai/types.rs` (2026-11-25, tooling)
- **Field convention**: items are declared with a space (`pub(in crate::serve) struct …` / `fn …`, 42 lines) but **fields have no space** — `pub(in crate::serve)model: Option<String>,` (132 lines). Match it in new DTOs; a multi-line `oldString` that "looks right" but differs by that one space is rejected.
- **Debug protocol for a rejected exact-text edit**: when the tool prints "closest lines" that look identical, the bytes differ — hexdump the range (`sed -n 'A,Bp' file | xxd`) before theorizing about line endings. The real causes here were (1) a hand-typed `crate:serve` (single colon) inside `crate::serve` and (2) the missing field space above. After editing this file run `grep -n 'crate:[a-z]' src/serve/openai/types.rs` (expect no output) and let `cargo build` catch the rest.
- **Reorganizing a block = move, don't retype**: to relocate a function (e.g. lifting a handler above `mod tests` for `clippy::items_after_test_module`), slice + splice the text with a few lines of Python rather than retyping it — then confirm with `git diff` that the change is a pure move.

## Maintenance

- New format parsers: implement `WeightParser`, extend `WeightFormat` + detector
- New HTTP endpoints: add `.route("/path", ...)` in `src/serve/mod.rs`, document in SPEC.md, add integration test
- New transformers: extend `forward_*_cached` in `src/runtime/transformer.rs` with KV cache support
- New sampling params: add to `GenerationConfig` + `make_generation_config`, document in SPEC.md
- New CLI commands: add a `Commands` variant, a dispatch arm in `src/bin/modelc.rs`, a SPEC.md subsection, a README usage line, and at least one `tests/cli_dispatch_test.rs` subprocess test
- **Update MEMORY.md after every non-trivial change** — capture the pattern, file:line, and rationale
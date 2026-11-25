# Agent Development Loop

This document defines the continuous improvement cycle for the **modelc** project — a Rust CLI that packages model weight files into a single `.modelc` artifact and serves an OpenAI-compatible HTTP inference API with Metal GPU and CPU SIMD acceleration.

## Project Structure

```
.
├── src/
│   ├── bin/modelc.rs        # CLI entry point (clap dispatch for pack/run/compile/list/…)
│   ├── lib.rs               # crate root, public module declarations, CLI_VERSION / GIT_SHA
│   ├── arch.rs              # shared architecture inspection (layers/hidden/heads) for runtime + codegen
│   ├── cli.rs               # Cli, Commands, WeightFormat, ModelArch, format detection + magic sniffing, compile_listen
│   ├── model.rs             # canonical Model / TensorData / DataType IR; size helpers; architecture inference; QuantPreview
│   ├── generate.rs          # autoregressive text generation (generate, generate_core); GenerationConfig;
│   │                        #   cooperative cancellation. `mod generate` holds submodules:
│   ├── generate/sampling.rs #   samplers (greedy/temperature/top-p/min-p), penalties (repetition/
│   │                        #   presence/frequency), logit bias, MiniRng seeded sampling
│   ├── generate/speculative.rs # speculative decoding (speculative_generate) + n-gram draft index
│   ├── tokenizer.rs         # byte-level BPE tokenizer (encode/decode, greedy merge, from_gguf, byte_fallback,
│   │                        #   decode_deltas for streaming endpoints)
│   ├── chat_template.rs     # chat message formatting (role: content\n fallback; Jinja2 templates accepted but ignored)
│   ├── constraint.rs        # grammar-based constrained decoding (Constraint trait, RegexConstraint token masking)
│   ├── json_schema.rs       # JSON-Schema-shaped generation validation (JSON parse + top-level type check, retry)
│   ├── prefix_cache.rs      # LRU PrefixCache of KV snapshots keyed by token sequence (longest-prefix reuse)
│   ├── draft.rs             # DraftModel trait + MlpDraftModel (EAGLE3-style) + PromptLookupDraftModel (weight-free)
│   ├── containerize.rs      # containerize command — Dockerfile + entrypoint for an artifact
│   ├── diff.rs              # structural diff between two artifact headers (modelc diff; port of modelc-web _diff.ts)
│   ├── digest.rs            # SHA-256 (streaming, stdlib-only) + RFC3339 timestamp — sha256:<hex> digest convention
│   ├── compiler.rs          # parser selection, apply_arch_hint, compile, inspect, pack, print_quant_sizes, prompt_model
│   ├── pack.rs              # .modelc artifact writer: JSON header + raw tensor blob, zstd compression, quantization, pruning
│   ├── pybridge.rs          # python bridge: executes a .py program with the system Python via an embedded stdlib-only
│   │                        #   driver; harvests weights (state_dicts / numpy / nested lists) into Safetensors
│   ├── store.rs             # local model store management; platform paths via dirs; search, versioning, pull, rm, cp
│   ├── lora.rs              # LoRA adapter loading and application on top of base model weights at runtime
│   ├── config.rs            # ~/.modelc/config.toml loading and saving (bind, port, store path, compression)
│   ├── kv_error.rs          # KvError + CacheValidationError + KvResult (rich error context for KV cache ops)
│   ├── tempdir.rs           # stdlib-only RAII TempDir / TempFile (replaces removed tempfile crate)
│   ├── parsers/             # format parsers (WeightParser trait), modularized by format
│   │   ├── mod.rs           # parser selection + dispatch
│   │   ├── gguf/            # GGUF parser (F32/F16/BF16/I8/I16/I32/I64/F64, Q4_0/Q5_0/Q8_0/Q4_K/Q6_K dequant) + Q4_0/Q8_0 block encoder
│   │   │   ├── mod.rs       # parser entry + tokenizer metadata extraction
│   │   │   ├── cursor.rs    # byte-level I/O
│   │   │   ├── dequant.rs   # block-quant → F32 dequantization (mod tests in gguf/tests.rs)
│   │   │   └── quant.rs     # F32 → Q4_0/Q8_0 block encoder (pack-time --quantize)
│   │   ├── onnx/            # ONNX parser
│   │   │   ├── mod.rs       # parser + execution plan building
│   │   │   └── initializers.rs  # inline / external / segmented tensor loading
│   │   ├── safetensors.rs   # Safetensors parser (safetensors crate)
│   │   └── pytorch.rs       # PyTorch ZIP / Safetensors-in-ZIP parser
│   ├── onnx_exec/           # ONNX graph execution engine
│   │   ├── mod.rs           # plan builder + executor
│   │   └── helpers.rs       # attribute getters, transpose, element-wise ops
│   ├── codegen/             # CodeGenerator trait; native/ module emits standalone Rust server
│   │   └── native/
│   │       ├── mod.rs       # codegen entry, generate_cargo_toml, generate_main_rs, MLP plan detection
│   │       ├── forward.rs   # MLP / GPT-2 / LLaMA forward generation
│   │       └── helpers.rs   # decode_f32, matmul_bias, relu_inplace + shared SIMD GEMV kernel
│   ├── runtime/             # library tensor + ops scaffolding for in-process inference
│   │   ├── tensor.rs        # Tensor type (Arc<Vec<f32>> + shape)
│   │   ├── ops.rs           # matmul, linear, softmax, layer_norm, rms_norm, …
│   │   ├── serve.rs         # Runtime::from_raw (load tensors, dequantize GGUF blocks, FP32 cast)
│   │   ├── transformer.rs   # GPT-2 / LLaMA forward passes + attention/MLP/norm helpers (re-exports kv::*)
│   │   ├── kv.rs            # KvCache / KvLayer (FP32, INT8, Mixed, shift, anchor preservation)
│   │   └── gemv.rs          # shared SIMD GEMV row dot-product (NEON/AVX with scalar fallback)
│   ├── serve/               # axum HTTP inference server
│   │   ├── mod.rs           # state, routing, CancelOnDrop cancellation wrapper
│   │   ├── handlers.rs      # Axum endpoints (/info, /infer, /chat, /chat/stream, /complete, /infill, /embeddings, /tokenize, /detokenize, /metrics, /health, /api/version, /v1/system, /reranking, /lora/load, /lora/unload, /api/cache/*)
│   │   ├── types.rs         # native modelc API request/response DTOs (chat/complete/infer/embeddings/…)
│   │   ├── web_ui.rs        # `GET /` static HTML page
│   │   ├── stdio.rs         # `modelc serve --stdio`: MCP-compatible newline-delimited JSON-RPC 2.0 server on
│   │   │                    #   stdin/stdout (initialize/ping/tools.list/tools.call; chat/complete/embeddings
│   │   │                    #   + resources.list/resources.read for model metadata)
│   │   ├── mcp_http.rs      # MCP Streamable HTTP transport (POST /mcp — the same JSON-RPC core as stdio)
│   │   ├── infer.rs         # ONNX / MLP / transformer inference + batched MLP
│   │   ├── openai.rs        # OpenAI-compatible endpoints (/v1/chat/completions, /v1/completions, /v1/embeddings, /v1/models)
│   │   ├── openai/types.rs  #   OpenAI wire-schema DTOs (requests, responses, chunks, SamplingParams)
│   │   ├── ollama.rs        # Ollama ecosystem: /api/tags + /api/show schema helpers (port of modelc-web
│   │   │                    #   _ollama.ts), /api/generate + /api/chat inference (NDJSON stream),
│   │   │                    #   /api/embed + /api/embeddings
│   │   ├── auth.rs          # API key + rate-limit middleware
│   │   └── metrics.rs       # Prometheus metrics + RAII ActiveRequestGuard / InferenceTimer
│   ├── metal.rs             # Apple Silicon Metal acceleration: matmul, relu, add, mul_scalar, softmax, layer_norm
│   └── compute/
│       └── shaders.metal    # Metal compute shader source (macOS only)
├── examples/                # runnable cargo examples (parse_weights, create_simple_model, inspect_model,
│                           #   runtime_inference, python_model, mcp_client) + python/ sample programs
│                           #   exercised by tests/python_samples_test.rs as golden-value validation cases
├── tests/                   # integration tests + docs_sync.rs (drift enforcement)
├── build.rs                 # captures MODELC_GIT_SHA via git rev-parse for CLI_VERSION
├── Cargo.toml               # package metadata + deps (clap, anyhow, axum, tokio, safetensors, half, onnx-rs, zip, dirs, toml, regex, zstd, rayon, metal on macOS)
└── Cargo.lock
```

## The Loop

### 0. Consult MEMORY.md Before Starting
**CRITICAL**: Always begin each task by reading relevant sections of `MEMORY.md`. This prevents:
- Reinventing solutions to already-solved problems
- Repeating known mistakes and anti-patterns
- Missing established conventions for similar features
- Ignoring domain-specific pitfalls (format-detection edge cases, KV-cache invariants, SIMD alignment, tokenizer round-trips)

**How to consult MEMORY.md**:
1. Search for keywords related to your task (e.g. "GGUF", "KV cache", "transformer", "completions API")
2. Read relevant pattern sections for context and proven approaches
3. Follow established conventions unless there's a clear reason to diverge
4. If MEMORY.md lacks relevant patterns, note this during the harvest step (Step 4)

### 1. Complete Remaining TODO Items
Pick the next highest-priority item from `TODO.md` (or `ARCHITECTURE.md` if the task is architectural). If no high-priority items remain, run the competitive intelligence step to seed new work. Implement with minimal, focused changes. Do not add speculative features.

### 2. Create Tests and Examples
For every new capability:
- Add inline `#[cfg(test)] mod tests` in the relevant source file — exercise the feature end-to-end
- Add unit tests for core logic where appropriate
- Provide a minimal usage example in `examples/` if the feature is client-facing
- Add a server-endpoint integration test to `tests/runtime_serve_test.rs` or `tests/transformer_runtime_test.rs` for HTTP behavior
- Bump the test counts in `TODO.md`'s `Done` section — `tests/docs_sync.rs` fails `cargo test` when they go stale

### 3. Ensure `cargo test` Passes
Run the full test suite:
```bash
cargo test                  # all inline unit tests + integration tests
cargo clippy --all-targets  # lint pass (warnings acceptable but noted; -D warnings before release)
```
Fix any failures before proceeding.

### 4. Harvest to MEMORY.md
After each completed feature, extract patterns and best practices:
- **Success patterns**: What worked well and should be repeated
- **Anti-patterns**: What to avoid in future implementations
- **Domain knowledge**: GGUF block-quant layouts, transformer forward invariants, OpenAI protocol gotchas
- **Rust patterns**: modelc-specific conventions for the IR, codegen, runtime, and serve dispatch
- **Testing patterns**: How to assert on float results, fixture tensors, regression cases for endpoints

Add these to `MEMORY.md` with clear categories and references to specific files/lines.

**Harvest Quality Checklist**:
- [ ] Added success patterns with code examples where useful
- [ ] Documented anti-patterns with what to avoid instead
- [ ] Included domain-specific gotchas and edge cases (parser sniffing limits, KV cache quantization, format-detection collisions)
- [ ] Referenced specific files/lines for future lookup
- [ ] Used existing MEMORY.md categories or created new ones if needed
- [ ] Made entries searchable with relevant keywords
- [ ] Noted any gaps found in MEMORY.md during Step 0 consultation

### 5. Loop Back to Step 1
Return to `TODO.md` and pick the next item. Repeat until the backlog is clear.

### 6. Audit and Optimize
After each batch of features, perform a quality pass:
- **Maintainability**: Are functions small and well-named? Is the module structure logical?
- **Docs alignment**: README/SPEC/ARCHITECTURE/TODO/MEMORY match the actual implementation (no drift). Enforced by `tests/docs_sync.rs`: the AGENTS.md module list and TODO.md test counts are checked by `cargo test` — if they fail, update the DOC, not the test
- **Leanness**: Remove dead code, unused imports, and speculative abstractions
- **Wiring**: Ensure all new features are properly integrated into `src/bin/modelc.rs`, the `clap` CLI dispatch, and HTTP routing in `src/serve/mod.rs`
- **Small footprint**: Avoid unnecessary dependencies; prefer standard library or lightweight crates
- **Consistency**: Match existing code style and patterns (Rust 2021 edition — no let-chains or other 2024-only syntax, `anyhow` for errors, `axum` handlers, `clap` derive)

### 7. Score the Codebase and Improve the Weakest Area
After the audit pass (Step 6) — or when the backlog runs dry — quantify overall health, then fix the single weakest dimension instead of guessing what to work on next.

**Scorecard** — score each dimension 0–5, evidence-based (never from vibes):

| Dimension | Evidence to check |
|---|---|
| Correctness & coverage | `cargo test` green; coverage % via `cargo-llvm-cov`/`cargo-tarpaulin`; no known failing/ignored cases |
| API fidelity | OpenAI-compatible endpoints match documented schema; GGUF/ONNX/Safetensors/PyTorch parsers round-trip |
| Maintainability | Dead code, clippy warning count, function sizes, cyclomatic complexity |
| Docs alignment | README/SPEC/ARCHITECTURE/TODO/MEMORY match the actual implementation (no drift) |
| Wiring & ergonomics | New features exported in `lib.rs`, CLI dispatch complete, examples compile, endpoints routed in `serve/mod.rs` |
| Footprint | Dependency count/weight, no speculative abstractions, release binary size |

Total is /30. Record the dated scorecard in `TODO.md` so progress is visible across iterations.

**Improve the weakness**:
1. Pick the single lowest-scoring dimension (tie-break: the one with the highest user impact)
2. Turn the fix into a concrete `TODO.md` item with a success criterion and test plan
3. Execute it through the normal loop (Steps 1–4) — tests and memory harvesting are not optional
4. Respect the Mature Module Policy: weak scores on mature modules are fixed by **adding tests/coverage first**, not rewrites
5. Re-score that dimension after the fix; a step that doesn't move the score didn't happen

### 8. Competitive Intelligence
Research similar open-source model servers (Ollama, llama.cpp server, vLLM, SGLang, TGI, mistral.rs, candle). Identify capabilities they have that this project lacks. Add the most valuable ones to the `TODO.md` brainstorming section. Prioritize features that provide clear competitive advantage.

### 9. Update Documentation
Keep all project docs aligned with the current implementation. Root docs (required):

- **`README.md`**: Quick start, CLI usage, feature list, HTTP API table, crate API summary
- **`ARCHITECTURE.md`**: Module relationships, data flow, design decisions
- **`TODO.md`**: Mark completed items, move them to Done, keep brainstorming current
- **`SPEC.md`**: User-facing commands, weight formats, HTTP endpoints, protocol semantics, error handling
- **`MEMORY.md`**: Harvested patterns, domain knowledge, technical conventions

Update **`AGENTS.md`** (this file) if the loop itself evolves.

## Memory System (MEMORY.md)

### Purpose
`MEMORY.md` is the institutional knowledge repository that accelerates development by:
- **Preventing wheel reinvention**: Reuse proven patterns instead of guessing
- **Domain knowledge preservation**: Capture GGUF/Safetensors/ONNX/PyTorch quirks that may be counter-intuitive
- **Onboarding acceleration**: New contributors (human or AI) can understand patterns quickly
- **Quality consistency**: Ensure all features follow established conventions

### Structure
Organize `MEMORY.md` into these sections:

#### 1. Format Parsers (GGUF, Safetensors, ONNX, PyTorch)
- Magic-byte sniffing order and ambiguity rules
- Tensor dtype handling per format (F32/F16/BF16/I8/I16/I32/I64/F64, GGUF block quants)
- External data and segmented initializer loading
- Tokenizer metadata extraction (GGUF vocab/merges/chat_template)
- Limitations (file size caps, pickle-only checkpoints)

#### 2. `.modelc` Artifact Format
- Header schema (JSON + version + flags)
- zstd compression (version 2)
- Byte layout (sorted tensor names → contiguous blob)
- `dequantize_in_place` contract for `compile` / `export`

#### 3. Runtime & Forward Passes
- `Runtime::from_raw` dequantization paths
- Transformer forward (`forward_gpt2_cached`, `forward_llama_cached`) invariants
- KV cache layout (FP32 / INT8 / Mixed) and shift semantics
- SIMD GEMV kernel (`src/runtime/gemv.rs`) — bit-identical between `run` and `compile`

#### 4. Serve Layer (HTTP)
- axum router assembly (`src/serve/mod.rs`)
- State sharing (`Arc<AppState>`, `RwLock<Runtime>`, `RwLock<PrefixCache>`)
- Cancellation (`CancelOnDrop`, `GenerationConfig.cancel`)
- Streaming (SSE, `ReceiverStream`)
- Auth/backpressure middleware (`api_key`, `rate_limit`, `max_concurrent`)

#### 5. Codegen & Compile
- `NativeCodegen::generate` output layout (`modelc_build/`, `embedded_weights.bin`)
- Shared kernel embedding via `include_str!`
- Cross-compilation (`--target`) and listen-address plumbing

#### 6. OpenAI API Compatibility
- `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` field shapes
- Streaming protocol (chunk format, `stream_options.include_usage`)
- Sampling parameters (`temperature`, `top_p`, `min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `seed`)
- Response shapes for non-streaming and streaming modes

#### 7. Testing Patterns
- Float comparisons (exact equality only when bit-identical)
- Fixture tensors for MLP / GPT-2 / LLaMA
- Server-endpoint smoke tests with `tests/common/http.rs`
- KV cache regression cases

#### 8. CLI & Tooling
- `clap` derive patterns and `compile_listen` semantics
- `WeightFormat` sniffing (extension → magic → safetensors parse)
- Local model store layout (versioned copies, `cp`/`rm`/`switch`)

### Maintaining MEMORY.md Quality

**When to Update MEMORY.md**:
- After completing any non-trivial feature or bug fix
- When discovering a new pattern or anti-pattern
- After resolving a tricky debugging session (especially transformer numerics or KV-cache invariants)
- When establishing new conventions
- When a known reference value or validation case is confirmed

**How to Write Good MEMORY.md Entries**:
1. **Be specific**: Reference actual files and line numbers where patterns occur
2. **Show examples**: Include minimal code snippets demonstrating the pattern
3. **Explain why**: Don't just say what—explain the reasoning behind the pattern
4. **Link related patterns**: Cross-reference related entries
5. **Keep it searchable**: Use keywords that future developers will search for
6. **Date entries**: Add dates so readers know how current the information is

**Signs MEMORY.md Needs Attention**:
- Same questions or patterns coming up repeatedly in development
- Developers (human or AI) solving problems that should be documented
- Frequent bugs in similar areas (indicates missing anti-pattern documentation)
- New contributors asking the same questions
- Numerical regressions in similar areas

**MEMORY.md Hygiene Routine** (run monthly):
- [ ] Review for outdated information (deprecated APIs, changed conventions)
- [ ] Consolidate redundant entries
- [ ] Add missing patterns from recent work
- [ ] Update cross-references if structure changed
- [ ] Verify all file references still exist
- [ ] Check for learnings that weren't captured

## Principles

- **Simplicity over flexibility**: Solve the problem at hand, not every hypothetical future problem
- **Surgical changes**: Touch only what you must; clean up only your own mess
- **Goal-driven**: Every change should have a verifiable success criterion
- **Test before ship**: No feature is complete until it has passing tests
- **Docs are code**: Documentation drift is a bug — `tests/docs_sync.rs` enforces AGENTS.md and TODO.md alignment at test time
- **Memory first**: Consult `MEMORY.md` before starting work — reuse proven patterns, avoid known pitfalls, follow established conventions. If `MEMORY.md` lacks relevant information, note the gap and fill it during harvest
- **Pattern harvesting**: After success, update `MEMORY.md` with patterns, anti-patterns, and learnings so others benefit from your experience
- **Memory hygiene**: Keep `MEMORY.md` current, searchable, and cross-referenced. It's only valuable if maintained

## Mature Module Policy

**Mature modules are effectively frozen.** Do not modify them as part of feature work, cleanup, or "drive-by" refactoring — even if the code looks improvable.

**A module is considered mature when ALL of the following hold:**
- It is stable and correct: no open bugs, no known numerical issues
- It is well-covered by tests that have not needed changes for several iterations
- Its API is depended on by many other modules (e.g. `model.rs`, `parsers/`, `runtime/transformer.rs`, `serve/openai.rs`)
- It has documented, validated reference behavior (`MEMORY.md` entries, known-good test values)

**Rules**:
1. **Extend, don't edit**: Add new capabilities in new modules or new functions; never reshape a mature module's public API to fit a new feature
2. **Wrap, don't rewrite**: Prefer adapters/new types over altering mature internals
3. **Bug fixes only**: A change to a mature module requires a concrete, reproducible bug or test failure — fix minimally and add a regression test; no opportunistic restructuring in the same commit
4. **Require explicit approval**: Any non-trivial change to a mature module must be justified in advance (risk, motivation, test plan) and confirmed by the user before starting
5. **Prefer coverage over rewriting**: If a mature module has high complexity/low coverage, respond by adding tests, not by rewriting — rewriting risk outweighs cosmetic gains
6. **Update docs, not code**: If behavior is confusing, document it in `MEMORY.md` rather than "fixing" it

## File Positioning and Value

### README.md
- **Value**: User-facing documentation and project overview
- **Audience**: Users, contributors, stakeholders
- **Position**: Entry point for anyone discovering the project
- **Focus**: Features, quick start, CLI usage, HTTP API table, crate API summary

### TODO.md
- **Value**: Feature roadmap and backlog management
- **Audience**: Development team (human and AI agents)
- **Position**: Development planning and prioritization
- **Focus**: What to build next, what's done, competitive intelligence

### ARCHITECTURE.md
- **Value**: Module relationships, data flow, and design decisions
- **Audience**: Contributors maintaining or extending the crate
- **Position**: Structural reference for the codebase
- **Focus**: Module boundaries, data flow, deployment topology

### SPEC.md
- **Value**: Interface specification for the CLI and HTTP API
- **Audience**: Users and contributors integrating with modelc
- **Position**: Contract definition for inputs and outputs
- **Focus**: User-facing commands, weight formats, HTTP endpoints, protocol semantics, error handling

### MEMORY.md
- **Value**: Institutional knowledge and pattern library
- **Audience**: Development team (accelerates onboarding and consistency)
- **Position**: Development acceleration and quality consistency
- **Focus**: Proven patterns, domain knowledge, technical conventions
- **Update**: Must be updated after each completed feature to capture patterns and lessons learned

### AGENTS.md (this file)
- **Value**: Development process and workflow definition
- **Audience**: AI agents and human developers following the development loop
- **Position**: Process automation and continuous improvement
- **Focus**: How we work, the loop, memory system, principles
- **Update**: This file should be updated when the development loop itself evolves or when new process patterns emerge

## How These Files Work Together

1. **`README.md`** tells stakeholders what the project is and how to use it
2. **`SPEC.md`** defines the CLI and HTTP API contract
3. **`ARCHITECTURE.md`** describes how the modules fit together
4. **`TODO.md`** tells developers what to build next (driven by competitive intelligence)
5. **`AGENTS.md`** tells agents how to work through the TODO items with quality and memory
6. **`MEMORY.md`** captures what we learned so we don't repeat mistakes and provides proven patterns to accelerate development

The loop reinforces these files:
Consult MEMORY.md → Complete TODO → Test → Harvest to MEMORY → Optimize → Score & improve weakest → Research → Update TODO

This creates a flywheel of continuous improvement with institutional knowledge preservation. **`MEMORY.md` is both the starting point (consult before work) and the destination (harvest after work), creating a virtuous cycle of learning and improvement.**
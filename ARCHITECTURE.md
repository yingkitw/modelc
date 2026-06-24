# ARCHITECTURE — modelc

## High-level pipeline

```
weights file → WeightParser → Model → pack (.modelc) → run → HTTP inference
              └─→ NativeCodegen → temp crate → cargo build → model-serve binary
```

The **packer** creates `.modelc` single-file artifacts with JSON header + compressed tensor blob. The **runner** loads artifacts and serves HTTP inference. The **compiler** (legacy) generates standalone binaries via `cargo build`. The **CLI** (`clap`) dispatches `pack`, `run`, `list`, `pull`, `inspect`, and `compile`. Apple Silicon acceleration is available where supported.

## Crate layout

| Module / path | Responsibility |
|---------------|----------------|
| `src/main.rs` | Binary entry; resolves `--listen` / `--bind`+`--port`, prints `CLI_VERSION`, calls compiler/packer/runner. |
| `src/cli.rs` | `Cli`, `Commands`, `WeightFormat`, `ModelArch`, format detection + magic sniffing, `compile_listen`, shell completions. |
| `src/lib.rs` | Re-exports modules; `CLI_VERSION` (`CARGO_PKG_VERSION` + `MODELC_GIT_SHA` from `build.rs`). |
| `src/model.rs` | Canonical `Model`, `TensorData`, `DataType`; size helpers; auto architecture inference from tensor names; `QuantPreview` for `--quant-sizes`. |
| `src/generate.rs` | Autoregressive text generation (`generate`, `generate_core`, `speculative_generate`), samplers (greedy/temperature/top-p/min-p), penalties (repetition/presence/frequency), grammar constraint application, and cooperative cancellation (`GenerationConfig.cancel`). |
| `src/tokenizer.rs` | Byte-level BPE tokenizer (encode/decode, greedy merge; `from_gguf`, `byte_fallback`). |
| `src/chat_template.rs` | Jinja2 (`minijinja`) chat-template rendering for message formatting before tokenization. |
| `src/constraint.rs` | Grammar-based constrained decoding (`Constraint` trait, `RegexConstraint` token masking). |
| `src/json_schema.rs` | JSON-Schema-constrained generation with retry (validates output against `jsonschema`). |
| `src/prefix_cache.rs` | LRU `PrefixCache` of KV snapshots keyed by token sequence (longest-prefix reuse across requests). |
| `src/draft.rs` | `DraftModel` trait + `MlpDraftModel` neural draft for EAGLE3-style speculative decoding (n-gram fallback). |
| `src/containerize.rs` | `containerize` command — emits Dockerfile + entrypoint for an artifact. |
| `src/compiler.rs` | Parser selection, `apply_arch_hint`, `compile`, `inspect`, `pack`; `print_quant_sizes` (`--quant-sizes`). |
| `src/pack.rs` | `.modelc` artifact writer: JSON header + raw tensor blob, optional zstd compression, quantization, pruning. |
| `src/store.rs` | Local model store management; platform-specific paths via `dirs` crate; search, versioning, pull, `rm`. |
| `src/lora.rs` | LoRA adapter loading and application on top of base model weights at runtime. |
| `src/config.rs` | `~/.modelc/config.toml` loading and saving (bind, port, store path, compression). |
| `src/parsers/` | Format parsers (`WeightParser` trait). Modularized by format. |
| `src/parsers/gguf/` | `mod.rs` (parser), `cursor.rs` (byte-level I/O), `dequant.rs` (Q4_0, Q5_0, Q8_0, Q4_K, Q6_K → F32). |
| `src/parsers/onnx/` | `mod.rs` (parser, execution plan building), `initializers.rs` (inline, external, segmented tensor loading). |
| `src/parsers/safetensors.rs` | Safetensors parser (`safetensors` crate). |
| `src/parsers/pytorch.rs` | PyTorch ZIP/Safetensors-in-ZIP parser. |
| `src/onnx_exec/` | ONNX graph execution engine. `mod.rs` (plan builder, executor), `helpers.rs` (attribute getters, transpose, element-wise ops). |
| `src/codegen/` | `CodeGenerator` trait; `native/` module emits standalone Rust server. |
| `src/codegen/native/` | `mod.rs` (codegen entry, `generate_cargo_toml`, `generate_main_rs`, MLP plan detection), `forward.rs` (MLP/GPT2/LLaMA forward generation), `helpers.rs` (decode_f32, matmul_bias, relu_inplace). |
| `src/runtime/` | Library tensor + ops scaffolding (`ops`, `serve`, `tensor`); `transformer.rs` runs GPT-2/LLaMA forward passes + KV cache (`KvCache`/`KvLayer` FP32/INT8/Mixed, context shifting, anchor preservation). |
| `src/serve/` | HTTP inference server. `mod.rs` (state, routing, `CancelOnDrop` cancellation wrapper), `handlers.rs` (Axum endpoints incl. `/tokenize`, `/v1/system`), `infer.rs` (ONNX/MLP/transformer inference + batched MLP), `openai.rs` (OpenAI-compatible endpoints), `auth.rs` (API key + rate-limit middleware), `metrics.rs` (Prometheus metrics). |
| `src/metal.rs` | Apple Silicon Metal acceleration: matmul, relu, add, mul_scalar, softmax, layer_norm kernels (macOS only). |
| `src/compute/` | Metal compute shaders (`shaders.metal`). |

## Key abstractions

### `WeightParser`

Loads a path into `Model`. The compiler selects an implementation from `WeightFormat` or fails detection.

### `CodeGenerator`

`NativeCodegen::generate(model, output_dir, listen)` writes:

1. `modelc_build/` with `src/main.rs`.
2. `embedded_weights.bin` — **authoritative** weight bytes for `include_bytes!` (contiguous offsets per tensor).
3. Embedded listen address string (Display of `SocketAddr`).

Original weight files are **not** copied into the emitted crate anymore; the blob is derived from parsed `TensorData`.

### `.modelc` artifact format

Single-file format created by `pack` command:
- **JSON header** — model metadata (name, architecture, tensor count, compression flag)
- **Raw tensor blob** — concatenated tensor data in sorted order
- **Optional compression** — zstd compression for version 2 format

The `run` command loads `.modelc` artifacts by:
1. Reading JSON header to parse metadata
2. Decompressing (if compressed version 2) or reading raw tensor data
3. Loading tensors into runtime for inference

### `Model` as IR

Parsers normalize into one structure so codegen and tests stay format-agnostic at the boundary.

Safetensors: optional header `__metadata__` is merged into `Model.metadata`; `architecture` prefers `architecture` then `model_type` keys.

## Build and I/O boundaries

- **Parsing:** read-only on user paths; surfaced via `anyhow::Context`.
- **Codegen:** writes under caller-provided temp directory, then copies output binary only.
- **Subprocess:** `compiler` invokes `cargo`; no rustc API linkage.

## Testing strategy

- **Integration:** `tests/` (CLI sniffing, codegen layout, parsers, compiler smoke, runtime).
- **`#![allow(dead_code)]`** on `tests/common` so shared helpers do not trigger `-D dead_code` under clippy across split test crates.

## Extension points

- New formats: implement `WeightParser`, extend `WeightFormat` + detector.
- Alternative emitters: new `CodeGenerator` impl(s).
- Broader codegen: architectures beyond **`mlp`**, `gpt2`, `llama` skeletons (expand forward.rs stubs).
- New ONNX ops: add handlers in `onnx_exec/mod.rs` and test in `tests/onnx_exec_test.rs`.
- Platform backends: Metal for Apple Silicon (full kernels implemented), CPU SIMD paths (AVX/NEON + rayon parallelization) active.
- Remote model registry: extend `pull` command to fetch from model hubs.

## Dependencies (compiler crate)

clap, serde/serde_json, anyhow, safetensors, byteorder, tempfile, **half**, **onnx-rs**, **zip**, **dirs** (platform paths), **zstd** (compression), **rayon** (parallel CPU ops), plus generated HTTP stack (axum/tokio) in emitted projects. **`half`** underpins FP16/BF16 → FP32 casts in `runtime::serve`. **`metal`** crate (macOS only) for Apple Silicon acceleration.

## Design principles

- Keep **`Model`** small and stable.
- Embedding uses a **single blob** so tensor metadata offsets are truthful.
- The packaged artifact is a **single file** optimized for size and fast loading.
- **Explicit** `-f/--format` when sniffing hits the 64 MiB Safetensors cap or ambiguous huge files.
- Cross-platform by default; acceleration is opt-in per platform backend.
- **Modularize large files**: files over ~400 lines are split into focused submodules (e.g., `gguf/` → `mod.rs` + `cursor.rs` + `dequant.rs`).

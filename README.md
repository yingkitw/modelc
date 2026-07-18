# modelc

**Package LLM weights into one portable file. Serve an OpenAI-compatible API. All from a single Rust CLI.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Rust edition](https://img.shields.io/badge/rust-edition%202024-orange.svg)](https://www.rust-lang.org/)
[![MSRV](https://img.shields.io/badge/MSRV-1.88+-orange.svg)](https://www.rust-lang.org/)
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](#cross-platform)
[![Built with Rust](https://img.shields.io/badge/built%20with-Rust-dea584.svg)](https://www.rust-lang.org/)
[![docs.rs](https://img.shields.io/badge/docs.rs-modelc-rust)](https://docs.rs/modelc)

> **modelc** is a Rust CLI that compiles model weight files — **GGUF, Safetensors, ONNX, and PyTorch** — into a single self-contained `.modelc` artifact and runs an **OpenAI-compatible inference server** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`). Think of it as a build tool plus a lightweight Ollama-style local runtime, in one binary, with native **Apple Silicon (Metal)** acceleration and CPU SIMD.

---

## What is modelc?

modelc is a command-line tool for working with local language-model weight files. It does two jobs:

1. **Package** weights into one `.modelc` file (JSON header + compressed tensor blob, optional INT4/INT8/FP16 quantization and pruning).
2. **Serve** that artifact as an HTTP inference API that speaks the OpenAI protocol, so existing clients and SDKs work without changes.

It supports **GPT-2 and LLaMA** transformer forward passes, **MLP** models, and **ONNX** graph execution directly from the in-memory runtime — no Python, no PyTorch, no large framework runtime required. Output binaries are pure Rust (axum + tokio).

### Cross-platform

Runs on **macOS**, **Linux**, and **Windows** from the same Rust toolchain. On Apple Silicon (M1–M4) it uses the **Metal** GPU for matmul/elementwise kernels; on x86_64 and aarch64 it uses **AVX / NEON** SIMD GEMV with a scalar fallback.

## How it compares

| Capability | modelc | Ollama | vLLM | llama.cpp |
|---|---|---|---|---|
| Implementation language | Rust | Go (wraps llama.cpp) | Python + CUDA | C++ |
| Model artifact | **single `.modelc` file** | layered blobs | HuggingFace weights | GGUF file |
| OpenAI-compatible API | ✅ `/v1/*` | ✅ | ✅ | ✅ |
| Native GPU | Metal (macOS) | via llama.cpp | CUDA / ROCm | CUDA, Metal, Vulkan |
| CPU SIMD | ✅ AVX / NEON GEMV | via llama.cpp | limited | ✅ |
| GGUF / Safetensors / ONNX / PyTorch | ✅ all four | GGUF | HF / Safetensors | GGUF |
| Quantization | INT4 / INT8 / FP16 at pack | GGUF quants | AWQ / GPTQ | GGUF quants |
| Speculative decoding, prefix cache, KV quant | ✅ | via llama.cpp | ✅ | ✅ |
| Standalone emitted binary (`compile`) | ✅ | ❌ | ❌ | ❌ |
| Best fit | packaging, portable single-file artifacts, lightweight serving | model zoo + chat app | high-throughput GPU serving | widest model support / edge |

> modelc's niche is **packaging and portability**: one tool that compiles, optimizes, and serves from a single file. It is not a high-throughput GPU serving engine like vLLM — see [When not to use modelc](#when-not-to-use-modelc).

## Quickstart

```bash
# 1. Build & install the CLI (Rust toolchain required)
cargo install --path .

# 2. Package a model into a single .modelc artifact
modelc pack model.gguf --compress -o my.modelc

# 3. Serve an OpenAI-compatible API on http://localhost:8080
modelc run my.modelc

# 4. Chat with it (OpenAI protocol)
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"my","messages":[{"role":"user","content":"hello"}]}'
```

Or run a prebuilt binary directly:

```bash
cargo build --release
./target/release/modelc --help
```

## Supported formats

| Format | Flag (`-f`) | Extensions | Notes |
|---|---|---|---|
| **GGUF** | `gguf` | `.gguf`, `.bin` (sniffed) | F32/F16/BF16/I8/I16/I32/I64/F64 + Q4_0/Q5_0/Q8_0/Q4_K/Q6_K (dequantized on-the-fly). Reads BPE vocab, merges, BOS/EOS, and Jinja2 chat template from KV metadata. |
| **Safetensors** | `safetensors` | `.safetensors` | Merges `__metadata__` (`architecture` / `model_type`) into the model IR. |
| **ONNX** | `onnx` | `.onnx` | Inline, external-data, and segmented initializers; graph ops executed via the tensor runtime. |
| **PyTorch** | `pytorch` | `.pt`, `.pth` | Torch **ZIP** containers incl. nested Safetensors payloads. Pickle-only checkpoints should be exported to Safetensors/ONNX/GGUF first. |

Ambiguous or extensionless files (e.g. generic `.bin`) are auto-sniffed against GGUF / zip / Safetensors magic bytes.

## Usage

### Core workflow

```bash
# Package
modelc pack model.safetensors -o my.modelc
modelc pack model.gguf --compress -o my.modelc
modelc pack model.safetensors --quantize int8 --prune 0.001 -o my.modelc

# Serve
modelc run my.modelc
modelc run my.modelc --port 8080 --profile
modelc run my.modelc --max-context 2048 --grammar "^\d+$"
modelc run my.modelc --temperature 0.7 --max-tokens 512
modelc run my.modelc --api-key sk-xxx --rate-limit 120
modelc run my.modelc --max-concurrent 8

# Local model store (Ollama-style)
modelc list
modelc rm my-model [--all|--force]
modelc cp my-model my-model-copy
modelc pull username/model-name --version 2
modelc switch my-model 2
```

### Inspect & export

```bash
modelc inspect model.safetensors
modelc inspect model.safetensors --quant-sizes   # preview fp32/fp16/int8/int4/q4_0 sizes
modelc inspect model.modelc --readme              # generate a Markdown model card
modelc export model.modelc -o out.safetensors
modelc verify model.modelc
modelc bench model.modelc --iterations 50
modelc search llama
```

### Compile to a standalone binary (legacy)

```bash
modelc compile model.safetensors -o ./my-model-serve   # emits a pure-Rust axum server
modelc containerize my.modelc                          # emits Dockerfile + entrypoint
```

The `compile` path and the `run` path produce **bit-identical logits** — they share one SIMD GEMV kernel (`src/runtime/gemv.rs`), so you can develop with `run` and ship `compile` and get the same numbers.

## Flags reference

### `pack` flags
- **`--compress`** — zstd-compress the artifact (version-2 format).
- **`--quantize fp16|int8|int4`** — quantize F32 tensors at pack time (INT4/INT8 auto-dequantize on `run`).
- **`--prune <threshold>`** — zero small weights below threshold.
- **`--arch llama|gpt2|mlp|…`** — optional architecture hint.
- **`--format` / `-f`** — force the input weight format.
- **`-o`** — output `.modelc` path.

### `run` flags
| Flag | Default | Purpose |
|---|---|---|
| `--bind` / `--port` | `0.0.0.0` / `8080` | HTTP bind address / port |
| `--api-key <key>` | — | Require `Authorization: Bearer <key>` (exempts `/health`, `/info`) |
| `--rate-limit <N>` | unlimited | Max requests/min per client IP |
| `--max-concurrent <N>` | unlimited | In-flight inference cap; over-limit gets `503` |
| `--max-context <N>` | — | Hard context limit → sliding-window KV eviction |
| `--anchor-tokens <N>` | `0` | Preserve first N tokens during shifts (StreamingLLM) |
| `--max-tokens <N>` | `256` | Max tokens to generate |
| `--temperature <T>` | `1.0` | Sampling temperature |
| `--top-p <P>` | `0.0` | Nucleus sampling |
| `--min-p <P>` | `0.0` | Min-p sampling |
| `--repetition-penalty` / `--presence-penalty` / `--frequency-penalty` | `1.0` / `0.0` / `0.0` | Token-reuse penalties |
| `--grammar <regex>` | — | Regex grammar constraint during sampling |
| `--seed <n>` | — | Deterministic sampling |
| `--profile` | — | Print per-step inference timing |

### `compile` flags
The emitted `model-serve` binary binds `--bind`+`--port` unless `--listen ADDR:PORT` is set (wins, embedded verbatim; IPv6 like `[::1]:8080` supported). Other flags: `--arch`, `--format`/`-f`, `--target` (passed to `cargo build --target`), `--debug` (build with debug profile instead of `--release`).

## HTTP API

All responses are `application/json`; streaming endpoints use SSE (`text/event-stream`).

| Method | Path | Description |
|---|---|---|
| `GET` | `/info` | Model name, architecture, params, bytes, tensor names |
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Prometheus text (request count, latency histogram, active gauge, tokens counter) |
| `GET` | `/props` | llama.cpp-compatible server properties |
| `GET` | `/api/version` | `{ version, git_sha }` |
| `GET` | `/v1/system` | CPU/GPU/memory/core info |
| `GET` | `/v1/models` · `/v1/models/:id` | OpenAI model list / retrieval |
| `POST` | `/infer` | Raw tensor inference (`input` or batch `inputs`) |
| `POST` | `/tokenize` · `/detokenize` | Token ID ↔ text round-trip |
| `POST` | `/chat` · `/complete` | Native chat / completion |
| `POST` | `/chat/stream` | SSE chat stream |
| `POST` | `/embeddings` | Native embeddings (single or batch) |
| `POST` | `/v1/chat/completions` | **OpenAI chat** (stream + non-stream, tools, logprobs, JSON schema, `n`, …) |
| `POST` | `/v1/completions` | **OpenAI legacy completion** (stream + non-stream, `echo`, `best_of`, `suffix`, `n`, …) |
| `POST` | `/v1/embeddings` | **OpenAI embeddings** (`encoding_format` float/base64, `dimensions`, `usage`) |
| `POST` | `/lora/load` · `/lora/unload` | Hot-swap LoRA adapters without restart |

**Inference backend priority:** ONNX execution plan → transformer forward (`gpt2`/`llama`) → MLP GEMV → echo fallback. Inputs longer than the hidden size are gracefully truncated.

## Features

### Inference & generation
- **GPT-2 / LLaMA transformer runtime** — full FP32 forward (layer norm / RMS norm, GeLU / SwiGLU, RoPE, single-token causal attention, output projection), numerically matched to the emitted `compile` server.
- **Autoregressive generation** — greedy, temperature, top-p (nucleus), min-p sampling; repetition / presence / frequency penalties; logit bias; stop sequences; deterministic `--seed`.
- **KV cache** — per-layer K/V with FP32, INT8 (~4× smaller), and Mixed (hot FP32 + cold INT8) variants; sliding-window context shifting + StreamingLLM anchor-token preservation.
- **Prefix caching** — LRU of KV snapshots keyed by token sequence; longest-prefix reuse across requests.
- **Speculative decoding** — EAGLE3-style `MlpDraftModel` (tiny 2-layer MLP) or weight-free `PromptLookupDraftModel`; verified via the KV cache.
- **Streaming** — SSE token streaming with request cancellation on client disconnect.
- **Concurrent generation** — `RwLock` prefix cache with short hold times; transformer requests run in parallel up to CPU core count.

### Quantization & optimization
- **Pack-time quantization** — `--quantize fp16|int8|int4`; preview with `inspect --quant-sizes`.
- **Weight pruning** — `--prune <threshold>`.
- **GGUF quantized inference** — Q4_0/Q5_0/Q8_0/Q4_K/Q6_K preserved in-IR and dequantized on-the-fly (~8× artifact size reduction for Q4_0).
- **zstd compression** — `--compress` (version-2 artifact format).
- **SIMD GEMV** — NEON (aarch64) / AVX (x86_64) K-dimension vectorized dot product, shared bit-for-bit between `run` and `compile` via `src/runtime/gemv.rs`.

### Serving & operations
- **OpenAI compatibility** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`; tools, tool_choice, JSON mode / JSON Schema, `n`, `best_of`, `echo`, `suffix`, `logprobs`, `stream_options.include_usage`, `user`, `max_completion_tokens`, `response_format`.
- **Auth & limits** — `--api-key` bearer auth, `--rate-limit` per-IP token bucket, `--max-concurrent` backpressure.
- **Observability** — Prometheus `/metrics`, `/v1/system` hardware info, `/api/version`, `/props`.
- **LoRA hot-swap** — load/unload adapters at runtime via HTTP, atomic runtime swap.
- **Graceful shutdown** — drains in-flight requests (incl. SSE) on `SIGINT`/`SIGTERM`.
- **Docker/OCI** — `modelc containerize` emits a minimal Dockerfile + entrypoint.

## Examples

Runnable programs live under [`examples/`](./examples/) (see [`examples/README.md`](./examples/README.md)):

```bash
cargo run --example create_simple_model -- ./demo_mlp.safetensors
cargo run --example parse_weights -- ./demo_mlp.safetensors safetensors
cargo run --example runtime_inference
```

## When not to use modelc

modelc trades flexibility for portability. Prefer raw weight files when you need:
- **Rapid A/B swaps** without repackaging.
- **Very large checkpoints** where embedding everything into one artifact is wasteful.
- **Multitenant "one server, many paths"** deployments that assume on-disk layouts.
- **Ecosystems that expect on-disk formats** (mmap loaders, GGUF loaders, ONNX Runtime with external weights, HuggingFace `transformers`).

See [SPEC.md](./SPEC.md) for full scope and limits.

## FAQ

**Is modelc an Ollama replacement?**
It overlaps Ollama's `pack`/`run`/`list` UX and exposes the same OpenAI API, but modelc focuses on single-file packaging and portable artifacts rather than a model zoo. Use Ollama for its catalog and chat app; use modelc when you want one optimized file you can ship anywhere.

**Does modelc need a GPU?**
No. It runs on CPU everywhere (AVX/NEON SIMD). On Apple Silicon it additionally uses Metal. There is no CUDA backend — for GPU high-throughput serving use vLLM.

**Which model architectures work end-to-end?**
GPT-2 and LLaMA run full transformer forward passes with text generation. MLP models run stacked GEMV. ONNX models execute their graph ops. Other architectures parse and inspect correctly but fall back to echo output for generation.

**Is the API really OpenAI-compatible?**
Yes — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and `/v1/models` follow the OpenAI schema (streaming, tools, logprobs, `n`, JSON schema, etc.), so drop-in clients work. Behavior is bounded by the loaded model's quality.

**How is `run` different from `compile`?**
`run` serves an existing `.modelc` artifact in-process. `compile` emits a standalone pure-Rust binary (`cargo build`). Both share the same SIMD GEMV kernel and produce identical logits, so `run` is for iteration and `compile` is for shipping.

## Configuration

A config file at `~/.modelc/config.toml` can set default `bind`, `port`, store path, and compression. Shell completions: `modelc completions bash|zsh|fish`. `modelc --help` and `modelc --version` list subcommands and the semver + git SHA (from `build.rs` when `.git` is present).

## Repository layout

```
src/
  bin/modelc.rs     CLI entrypoint
  cli.rs            clap commands, format detection, completions
  model.rs          canonical Model / TensorData / DataType IR
  parsers/          gguf/ onnx/ safetensors.rs pytorch.rs (WeightParser impls)
  onnx_exec/        ONNX graph execution engine
  codegen/native/   standalone-binary emitter (forward.rs, helpers.rs)
  runtime/          tensor/ops/serve + transformer.rs (KV cache) + gemv.rs (shared SIMD)
  serve/            axum server: handlers, infer, openai (v1/*), auth, metrics
  generate.rs       autoregressive generation, samplers, speculative decoding
  tokenizer.rs      byte-level BPE tokenizer
  lora.rs prefix_cache.rs constraint.rs draft.rs  ...
examples/           runnable cargo examples
tests/              integration tests
```

Companion docs: [SPEC.md](./SPEC.md) · [ARCHITECTURE.md](./ARCHITECTURE.md) · [TODO.md](./TODO.md) · [PRODUCT.md](./PRODUCT.md)

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](./LICENSE)).

# TODO

Most backlog items from the previous list are implemented (arch wiring, sniffing, embedded blob codegen, `--bind` / `--listen`, runtime dtypes beyond F32, CI, Apache `LICENSE`, documented HTTP surface, upstream format links).

## Optional follow-ups

- [ ] Implement real GGUF / ONNX / PyTorch parsers (beyond stubs with doc links).
- [ ] Teach codegen to emit a non-placeholder `forward` for selected `ModelArch` + operator coverage.
- [ ] Reduce peak memory during `compile` for huge models (avoid duplicate tensor buffers between IR + blob write when possible).
- [ ] crates.io publishing checklist (`README` badge, `--help` snapshot, changelog).

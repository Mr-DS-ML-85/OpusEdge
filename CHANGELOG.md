# Changelog

All notable changes to OpusEdge will be logged here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version
scheme is [SemVer](https://semver.org/).

## [Unreleased]

### Changed

- **Licence — MIT → Noncommercial.** The code now ships under
  [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)
  (SPDX: `PolyForm-Noncommercial-1.0.0`). The paper text, figures, and PDF/DOCX
  renderings are now under [CC BY-NC 4.0](paper/LICENSE). Both licences permit
  research, teaching, personal study, and use by charitable / educational /
  public-research / government organisations; commercial use requires a
  separate agreement. Contact via the maintainer's GitHub profile for a
  commercial licence.

### Fixed

- **Fake KV-reduction metric.** The chunked-SelKV bench previously reported
  `1 - budget/total_S`, which was a config-derived number, not a measurement.
  It now sums `K.numel() * K.element_size() + V.numel() * V.element_size()`
  across every cache layer before and after each trim, and reports the peak
  reduction against a no-eviction reference run.
- **Falcon-H1 `NaN` on SelKV.** transformers 5.x `DynamicCache.crop()` and
  `.update()` both fail on Falcon-H1's `LinearAttentionAndFullAttentionLayer`.
  Replaced with direct in-place `layer.keys = keys[..., -n_kept:, :].clone()`
  which works across DynamicCache, HybridChunkedCache, and the Falcon-H1
  hybrid layer, and (verified) actually frees the trimmed tensor storage.
- **65K on 8 GB VRAM** — dense Qwen2.5-0.5B and MoE Granite-3.1-1B-A400M
  stream 65 536 tokens through the model with 93.8 % measured KV-cache
  reduction; hybrid Falcon-H1 hits 16 K under the same budget.

## [1.0.0] — 2026-06-01

First public reference implementation of the OpusEdge paper.

### Added

- **C++20 engine** (`engine/`) — header-only, `Eigen 3.4+`.
  - 16 primitive headers implementing SelKV, SMSA, Delta-AR, ΔRank,
    HeadDeactivate, StateCompress, DenseEvic, GAKV/R-GAKV, Pareto Frontier,
    Router-IR, plus stabilizers MPSR/SACT, EB-AR/EB-DAR, SSR/CASP, NDPA,
    IPSS/CSA, and task controllers CAL, R-CAL.
  - 30 Python bindings exposed via `opusedge_cpp` package.
  - Family-scoped SDK at `engine/include/opusedge/sdk.h` with
    `DensePipeline`, `HybridPipeline`, `MoEPipeline` classes.
- **Python SDK** (`opusedge_cpp.sdk`) — same three-pipeline API mirrored in
  Python.
- **C++ benchmarks** — `bench_delta_ar`, `bench_scaling` (empirical
  complexity classifier via log-log regression), `bench_pipeline`
  (per-family end-to-end flow with CSV output).
- **Python benchmark harness** (`bench/`) — `uv`-managed, loads real
  HuggingFace models (Qwen2.5-0.5B, Falcon-H1-0.5B, Granite-3.1-1B-A400M),
  extracts native Δ / Proxy-Δ / Router-IR, runs SelKV + SMSA + Delta-AR
  against real hidden states, reports PPL + KV cache footprint in MiB and
  percent.
- **Three.js landing page** (`web/index.html`) — multi-scene visualisation
  with bloom postprocessing and scroll-driven scene swaps.
- **Aesthetic docs site** (`docs/`) — Getting Started, Primitives Reference,
  Architecture, Benchmarks (each linked from `docs/index.html`).
- **GitHub CI** — configure → build → ctest → run demo → run scaling bench →
  run pipeline bench.
- **Test suite** — 9 GTest suites covering SelKV, SMSA (analytical + real
  forward + adaptive width + SSD mask), Delta-AR, ΔRank, HeadGate,
  StateCompress, SignalExtractor, CAL/R-CAL, Pareto.

### Fixed (against pre-1.0 draft)

- Router entropy softmax denominator (was mathematically broken).
- `DeltaRank::ssr` out-of-bounds when `rank_fraction == 1.0`.
- Spearman with tied ranks now uses average-rank + Pearson-on-ranks.
- Delta-AR index padding uses `-1` sentinel instead of index 0 (fixed
  double-attention to token 0 for `t < K`).
- HeadGate weighted average now uses float division (was integer).
- HeadGate config auto-scales tier budgets when `n_heads != 32`.

### Documented

- `full_causal_attn` in `smsa.h` is deliberately Θ(S²·D). Any
  linear-attention approximation belongs in a separate primitive; the
  reference forward stays honest.
- The paper's contribution 2 (§1) — all 10 core primitives are applicable
  across all three architecture families — is now the SDK's design axiom.

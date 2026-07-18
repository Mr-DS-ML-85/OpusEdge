# OpusEdge C++ Core — Architecture

## Layered Design

```
wrapper.cpp         ← CPython C API (26 bindings, 1:1 with primitives)
       │
       ▼
primitives/         ← 16 independent policy primitives (single-purpose)
  cal.h             CAL/R-CAL task-tier classification
  composite.h       Meta-primitive: SMSA + SelKV + ΔAR + HeadGate
  delta_ar.h        Top-K attention routing + EBDAR
  delta_rank.h      Adaptive low-rank decomposition (SVD-based)
  ebar.h            Entropy-buffered compute modulation
  gakv.h            Gating-aware KV scoring (GAKV + R-GAKV)
  head_gate.h       Multi-head deactivation (4-tier gating)
  ipss.h            Variance-based head salience smoothing
  mpsr.h            Sigmoid-projection state recycling
  ndpa.h            Rank-1 Δ–attention rectifier
  pareto.h          Pareto-frontier config sweep
  rcal.h            Recurrent CAL with EMA confidence & freeze
  selkv.h           Δ-score KV cache eviction
  smsa.h            Sliding-window sparse attention mask
  ssr.h             Spectral soft-thresholding + CASP
  state_compress.h  Predictable state dimension compression
       │
       ▼
core/               ← Foundational types and utilities
  types.h           Float, VectorXf, TierPolicy, SignalExtraction, etc.
  signal.h          Proxy-Δ, Spearman, normalize, SACT transmute
  config.h          OpusEdgeConfig with kPolicy rule list
       │
       ▼
Eigen3              ← Linear algebra (vectors, matrices, SVD)
OpenMP              ← Parallel loops (implicit in Eigen + explicit)
```

## Data Flow

```
Model forward pass
    │
    ▼
Extractor (Python) → hidden states + attention weights
    │
    ▼
proxy_delta()   ───→ Δ scores [1 × seq_len]
normalize()     ───→ norm scores
spearman()      ───→ ρ correlation
    │
    ├──→ selkv_evict()       → retained/evicted indices
    ├──→ smsa_analyze()      → speedup, memory
    ├──→ head_active()       → active head count
    ├──→ delta_ar_indices()  → top-K indices
    ├──→ gakv_analyze()      → composite scores
    ├──→ ndpa_rectify()      → rectified Δ, γ
    ├──→ mpsr_project()      → compression ratio
    ├──→ ssr_analyze()       → thresholded scores
    ├──→ ipss_analyze()      → salience ordering
    ├──→ ebar_analyze()      → compute modulation
    ├──→ cal/rcal_classify() → tier, threshold
    └──→ pareto_sweep()      → Pareto frontier
```

## Memory Model

All primitives are **stateless** by design — they receive inputs, compute, and return results. No mutable global state. Exception: `RCAL` uses an EMA confidence accumulator (per-instance), which is why `rcal_classify` returns a confidence value that evolves over repeated calls (currently not persisted across benchmark rows).

## Adding a New Primitive

1. Create `include/opusedge/primitives/newprim.h` with a struct/class
2. Include in `wrapper.cpp`
3. Write `static PyObject* py_newprim(...)` using `PyArg_ParseTuple` + `Py_BuildValue`
4. Register in `OpusEdgeMethods[]`
5. Add to `bench_arxiv.py` `bms` dict + row field extraction
6. Add to `benchmark_results.md` latency + output description

## C++ → Python Type Mapping

| C++ (Eigen) | Python |
|---|---|
| `VectorXf` | `list[float]` |
| `MatrixXf` | `list[list[float]]` |
| `std::vector<int>` | `list[int]` |
| `std::string` | `str` |
| `int` | `int` |
| `Float` (= `double`) | `float` |
| `struct` | `tuple` via `Py_BuildValue` |
| `std::vector<struct>` | `list[tuple]` |

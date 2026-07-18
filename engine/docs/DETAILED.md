# OpusEdge C++ Core

**High-Performance Inference Primitives — 26 CPython bindings, 16 C++ headers**

Zero-cost dynamic compute allocation for SSM-hybrid, dense, and MoE transformer architectures. Implements all techniques from the OpusEdge paper: predictive KV eviction (SelKV), sparse attention (SMSA, Delta-AR), dynamic head gating, state compression, and 9 new policy techniques (GAKV, NDPA, MPSR, EB-AR, SSR, IPSS, R-CAL/CAL, Pareto).

```
pip install -e .    # builds _opusedge_cpp extension
```

> **Python 3.13+ / C++20 / Eigen3 / OpenMP.**  
> The `.so` is architecture-specific — rebuild if you move between machines.

---

## Quick Start

### From Python

```python
import sys; sys.path.insert(0, "cpp/")
import _opusedge_cpp as oe
import numpy as np

# Extract Δ signal from model hidden states
deltas = [0.3, 0.6, 0.8, 0.2, 0.1, 0.5, 0.7, 0.4]

# SelKV: evict 50% of KV cache entries
retained, evicted = oe.selkv_evict(deltas, 0.5)
print(f"Retained {len(retained)}/{len(deltas)} tokens")

# SMSA: sliding-window speedup analysis
speedup, memory_savings = oe.smsa_analyze(2048, 256)
print(f"SMSA speedup: {speedup:.1f}x at window=256")
```

### From C++ (standalone)

```cpp
#include <opusedge/primitives/selkv.h>
#include <opusedge/primitives/smsa.h>
#include <opusedge/primitives/head_gate.h>
#include <opusedge/primitives/cal.h>

opusedge::VectorXf delta(8);
delta << 0.3, 0.6, 0.8, 0.2, 0.1, 0.5, 0.7, 0.4;

// SelKV: evict 50%
auto r = opusedge::SelKV::evict(delta, 0.5, 8);
// r.retained_indices, r.evicted_indices, r.memory_savings

// SMSA
auto [sp, flop, mem] = opusedge::SMSA::analyze(2048, 256);

// Head gating
opusedge::HeadGate hg({32});
int active = hg.active_heads(0.5);

// CAL tier classification
auto tier = opusedge::CAL().classify("math");
// tier.name = "High", tier.rigidity = 0.85
```

### Build

```bash
# Python extension
cd cpp/
python setup.py build_ext --inplace

# CMake (examples + tests + benchmarks)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DOPUSEDGE_BUILD_EXAMPLES=ON \
    -DOPUSEDGE_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build

# Run the comprehensive demo (all 30+ primitives)
./build/examples/all_primitives_demo
./build/examples/delta_ar_demo
```

---

## Installation

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| C++20 compiler | GCC ≥ 11 / Clang ≥ 14 | Core language |
| Eigen3 | ≥ 3.4 | Linear algebra (vectors, matrices, SVD) |
| OpenMP | Included in GCC/Clang | Parallel loops |
| Python | ≥ 3.12 | CPython bindings |
| setuptools | ≥ 68 | Python extension build |
| CUDA | ≥ 12 (optional) | GPU benchmarks |

### From source

```bash
# Install system dependencies
sudo apt install build-essential cmake libeigen3-dev

# Build Python extension
cd cpp/
python3 -m venv .venv && source .venv/bin/activate
pip install setuptools numpy
python setup.py build_ext --inplace

# Verify
python -c "import sys; sys.path.insert(0,'.'); import _opusedge_cpp; print('OK:', dir(_opusedge_cpp))"
```

### As a CMake dependency (downstream C++ projects)

Install the library system-wide:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --install build
```

Then in your downstream `CMakeLists.txt`:

```cmake
find_package(opusedge 1.0 REQUIRED)
target_link_libraries(my_app PRIVATE opusedge::opusedge)
```

Or add as a subdirectory:

```cmake
add_subdirectory(third_party/opusedge)
target_link_libraries(my_app PRIVATE opusedge)
```

Your downstream C++ code:

```cpp
#include <opusedge/primitives/selkv.h>
#include <opusedge/primitives/gakv.h>
#include <opusedge/primitives/cal.h>
#include <opusedge/primitives/pareto.h>

opusedge::VectorXf delta(8);
delta << 0.3, 0.6, 0.8, 0.2, 0.1, 0.5, 0.7, 0.4;

// GAKV composite scores
opusedge::GAKV gakv({0.5, 0.5, 0.3, 0.3});
auto res = gakv.analyze(
    std::vector<double>(delta.data(), delta.data() + 8),
    std::vector<double>{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}
);

// Pareto sweep
auto points = opusedge::ParetoFrontier::sweep(delta, 2048);
auto frontier = opusedge::ParetoFrontier::pareto_optimal(points);
```

---

## Project Structure

```
cpp/
├── setup.py                        # Python extension build script
├── wrapper.cpp                     # CPython C API (26 bindings, 573 lines)
├── include/opusedge/
│   ├── core/
│   │   ├── types.h                 # Base types: Float, VectorXf, TierPolicy, etc.
│   │   ├── signal.h                # Proxy-Δ, Spearman, normalize, SACT
│   │   └── config.h                # OpusEdgeConfig with kPolicy rules
│   └── primitives/
│       ├── selkv.h                 # Δ-score KV cache eviction
│       ├── smsa.h                  # Sliding-window sparse attention mask
│       ├── delta_ar.h              # Top-K attention routing + EBDAR
│       ├── delta_rank.h            # Adaptive low-rank decomposition (SVD)
│       ├── head_gate.h             # Dynamic multi-head deactivation (4 tiers)
│       ├── state_compress.h        # State dimension compression
│       ├── composite.h             # Meta-primitive: SMSA+SelKV+ΔAR+HeadGate
│       ├── pareto.h                # Pareto-optimal config sweeps
│       ├── cal.h                   # CAL: task-rigidity tiering (dense/MoE)
│       ├── rcal.h                  # R-CAL: recurrent CAL w/SSM state gating
│       ├── gakv.h                  # GAKV + R-GAKV: gating-aware KV scoring
│       ├── ndpa.h                  # NDPA: Δ–attention alignment rectifier
│       ├── mpsr.h                  # MPSR: sigmoid-projection state recycling
│       ├── ebar.h                  # EB-AR: entropy-buffered compute modulation
│       ├── ssr.h                   # SSR + CASP: spectral thresholding
│       └── ipss.h                  # IPSS: variance-based head salience
├── tests/                          # C++ unit tests (CMake, 9 test files)
├── benchmarks/                     # C++ microbenchmarks (CMake)
├── examples/                       # C++ usage demos (CMake)
├── docs/
│   └── architecture.md             # Layered design, data flow, extension guide
└── build/                          # Build artifacts (gitignored)
```

---

## All 26 CPython Bindings

Every primitive is exposed as a flat function on the `_opusedge_cpp` module. Call them with Python lists of floats and get back tuples/lists.

### Original 14 (Sections 3–4)

| Python call | C++ source | Returns | Description |
|---|---|---|---|
| `proxy_delta(hidden_states)` | `signal.h` | `list[float]` | RMS hidden-state drift (Proxy-Δ) |
| `spearman(a, b)` | `signal.h` | `float` | Spearman ρ correlation |
| `normalize(v)` | `signal.h` | `list[float]` | L2 normalization |
| `sact_transmute(v, residual)` | `signal.h` | `list[float]` | SSM-assisted cache transmutation |
| `delta_ar_indices(v, k)` | `delta_ar.h` | `list[list[int]]` | Top-K indices per query |
| `delta_ar_flops(n, k)` | `delta_ar.h` | `float` | FLOP reduction fraction |
| `ebdar(v, mask, beta)` | `delta_ar.h` | `(float, float)` | Entropy-boosted Delta-AR |
| `selkv_evict(v, ratio)` | `selkv.h` | `(list[int], list[int])` | Retained + evicted indices |
| `selkv_quality_ratio(v, ratio)` | `selkv.h` | `float` | Quality ratio vs random |
| `smsa_analyze(seq_len, window)` | `smsa.h` | `(float, float)` | Speedup, memory savings |
| `head_active(dt, n_heads)` | `head_gate.h` | `int` | Active heads at Δ threshold |
| `head_flop_reduction(v)` | `head_gate.h` | `float` | FLOP reduction from head pruning |
| `state_keep_ratio(dt)` | `state_compress.h` | `float` | State compression keep ratio |
| `composite_analyze(seq_len, n_heads, head_dim, hidden)` | `composite.h` | `(float, float)` | Composite FLOP + memory |

### New 9 Paper Techniques (Section 5)

| Python call | C++ source | Equation | Returns | Description |
|---|---|---|---|---|
| `cal_classify(task, base)` | `cal.h` | Eq. 15 | `(str, int, float, float)` | CAL tier, rigidity, threshold |
| `cal_rigidity(task)` | `cal.h` | Eq. 15 | `float` | Task rigidity value |
| `rcal_classify(task, base)` | `rcal.h` | Eq. 15–17 | `(str, str, float, float)` | R-CAL tier, threshold, confidence |
| `rcal_modulate(task, base)` | `rcal.h` | Eq. 16 | `float` | Modulated threshold |
| `rcal_eviction_cap(task)` | `rcal.h` | Eq. 17 | `float` | Task eviction cap |
| `gakv_analyze(deltas, ir)` | `gakv.h` | Eq. 18–19 | `(list, list, list)` | Composite scores, retained, evicted |
| `rgakv_analyze(deltas, ir, cal_mod)` | `gakv.h` | Eq. 20 | `(list, list, list)` | CAL-modulated scores, indices |
| `ndpa_rectify(deltas, attn)` | `ndpa.h` | Eq. 21 | `(list, float, int)` | Rectified Δ, γ, active flag |
| `mpsr_project(scores, state_dim)` | `mpsr.h` | Eq. 22–23 | `(list, float, float)` | Projected state, compression, energy |
| `mpsr_sact(scores, state_dim, residual)` | `mpsr.h` | Eq. 22–23 | `(list, float, float)` | MPSR with SACT residual |
| `ebar_analyze(entropies)` | `ebar.h` | Eq. 24–25 | `(list, list, float)` | Compute per step, buffer, savings |
| `ebar_entropy(log_probs)` | `ebar.h` | — | `list[float]` | Shannon entropy from log-probs |
| `ssr_analyze(singular_values, entropy)` | `ssr.h` | Eq. 26–27 | `(list, int, float, int)` | Thresholded values, preserved count, fraction, compression |
| `ssr_casp(singular_values, curvature)` | `ssr.h` | — | `(list, int, float, int)` | CASP curvature variant |
| `ipss_analyze(head_variances)` | `ipss.h` | Eq. 28–29 | `(list, list, float, int)` | Active heads, salience, FLOP red, count |
| `pareto_sweep(deltas, seq_len)` | `pareto.h` | Sec. 5.6 | `list[tuple]` | Pareto frontier points |

---

## Architecture-Aware Dispatch

Not all primitives apply to all architectures. The benchmark (`bench_arxiv.py`) routes by `family`:

| Family | Models | Signal Source | Excluded |
|---|---|---|---|
| **hybrid** | Falcon-H1, Jamba | Native SSM Δ | None — all 23 run |
| **dense** | Qwen2.5, LLaMA | Proxy-Δ (RMS drift) | SACT, StateCompress (need SSM state) |
| **moe** | Huihui-MoE, OLMoE | I_r + Proxy-Δ | SACT, StateCompress (need SSM state) |

### CAL vs R-CAL

| Technique | Architecture | Mechanism | Key output |
|---|---|---|---|
| **CAL** | Dense, MoE | Task-rigidity tier map | Tier name, rigidity, threshold |
| **R-CAL** | Hybrid (SSM) | SSM state gating + EMA confidence | Tier, threshold, confidence |

From the benchmark (72 prompts across 3 models):

```python
# Hybrid → R-CAL
oe.rcal_classify("math", 0.6)    # → ("Math", "Math", 1.11, 1.0)

# Dense/MoE → CAL
oe.cal_classify("math", 0.6)     # → ("High", 1, 0.85, 0.925)
oe.cal_classify("summary", 0.6)  # → ("Balanced", 2, 0.30, 0.65)
```

---

## C++ API Example

Build and run the comprehensive demo (all 30+ primitives in a single binary):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOPUSEDGE_BUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
./build/examples/all_primitives_demo
```

Or write your own `demo.cpp`:

```cpp
#include <opusedge/primitives/selkv.h>
#include <opusedge/primitives/smsa.h>
#include <opusedge/primitives/head_gate.h>
#include <opusedge/primitives/gakv.h>
#include <opusedge/primitives/cal.h>
#include <opusedge/primitives/ndpa.h>
#include <iostream>

int main() {
    // Proxy-Δ scores from model
    opusedge::VectorXf delta(8);
    delta << 0.3, 0.6, 0.8, 0.2, 0.1, 0.5, 0.7, 0.4;

    // SelKV: evict 50%
    auto ev = opusedge::SelKV::evict(delta, 0.5, 8);
    std::cout << "SelKV: retained=" << ev.retained_indices.size()
              << " evicted=" << ev.evicted_indices.size() << "\n";

    // SMSA at 2048 tokens, window 256
    auto [sp, flop, mem] = opusedge::SMSA::analyze(2048, 256);
    std::cout << "SMSA: speedup=" << sp << "x  mem=" << mem << "\n";

    // Head gate
    opusedge::HeadGateConfig hcfg; hcfg.n_heads = 32;
    opusedge::HeadGate hg(hcfg);
    int active = hg.active_heads(0.5);
    std::cout << "Active heads at Δ=0.5: " << active << "/32\n";

    // GAKV composite scoring
    opusedge::VectorXf ir(8);
    ir << 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2;
    opusedge::GAKV gakv;
    auto gr = gakv.analyze(
        std::vector<double>(delta.data(), delta.data() + 8),
        std::vector<double>(ir.data(), ir.data() + 8)
    );
    std::cout << "GAKV evicted: " << gr.n_evicted << "\n";

    // CAL tier classification
    auto tier = opusedge::CAL().classify("code");
    std::cout << "CAL tier: " << tier.name << "\n";

    // NDPA rectification
    std::vector<double> attn = {0.2, 0.7, 0.9, 0.1, 0.15, 0.4, 0.6, 0.3};
    auto nr = opusedge::NDPA().analyze(
        std::vector<double>(delta.data(), delta.data() + 8), attn);
    std::cout << "NDPA rectified, γ=" << nr.gamma << "\n";

    return 0;
}
```

Compile with one of:

```bash
# Standalone (no CMake)
c++ -std=c++20 -O3 -march=native -fopenmp \
    -I include -I /usr/include/eigen3 \
    demo.cpp -o demo

# Via CMake (after install)
find_package(opusedge 1.0 REQUIRED)
target_link_libraries(demo PRIVATE opusedge::opusedge)
```

---

## Performance

All primitives complete in microseconds. Measured on RTX 4060 (bench_arxiv.py, 72 prompts × 3 models):

| Primitive | Latency (μs) |
|---|---|
| head_active / delta_ar_flops / smsa_analyze | 0.5–0.8 |
| spearman / state_keep / head_flop_red | 0.7–1.0 |
| normalize / ndpa_rectify / ebar_analyze | 1.1–1.4 |
| ssr_analyze / selkv_evict / sact_transmute | 1.5–1.7 |
| ipss_analyze / gakv_analyze | 2.0–2.3 |
| mpsr_project / rgakv_analyze / rcal_classify | 2.6–2.8 |
| composite | 4.7–5.1 |
| selkv_qr | 4.9–5.3 |
| delta_ar_idx | 16.0–16.8 |
| ebdar | 18.8–19.4 |
| pareto_sweep | 25.6–27.7 |
| **proxy_delta** (data marshaling dominated) | **5700–5870** |

**Total pipeline overhead: < 6 ms** (dominated by proxy_delta PyTorch → C++ marshaling).

---

## Running Tests + Demos

```bash
# Build everything
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DOPUSEDGE_BUILD_EXAMPLES=ON \
    -DOPUSEDGE_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# Run C++ unit tests
ctest --test-dir build

# Individual test
./build/tests/test_selkv
./build/tests/test_smsa
./build/tests/test_head_gate
./build/tests/test_pareto
./build/tests/test_delta_ar
./build/tests/test_delta_rank
./build/tests/test_state_compress
./build/tests/test_cal

# Run all-primitives demo (standalone C++, no Python)
./build/examples/all_primitives_demo

# Run original demo
./build/examples/delta_ar_demo

# Python integration benchmark
cd ../python
python bench_arxiv.py
```

---

## Adding a New Primitive

1. **Header:** Create `include/opusedge/primitives/myprim.h` — structs + class with methods
2. **Example:** Add to `examples/all_primitives_demo.cpp` to exercise the C++ API directly
3. **CMake:** If the example file is new, add it to `examples/CMakeLists.txt`
4. **Wrapper:** Include in `wrapper.cpp`, write `PyObject* py_myprim(...)` using `PyArg_ParseTuple` + `Py_BuildValue`
5. **Register:** Add to `OpusEdgeMethods[]` array at `wrapper.cpp:519`
6. **Rebuild:** `python setup.py build_ext --inplace`
7. **Benchmark:** Add to `bench_arxiv.py` `bms` dict + row field extraction
8. **Document:** Add to this README's tables

Type mapping for Python ↔ C++:

| Python | C++ |
|---|---|
| `list[float]` | `VectorXf` / `std::vector<Float>` |
| `list[list[float]]` | `MatrixXf` |
| `list[int]` | `std::vector<int>` |
| `str` | `const char*` |
| `int` | `int` |
| `float` | `Float` (= `double`) |
| `tuple` | `Py_BuildValue("(sidd)", ...)` |
| `list[tuple]` | `PyList_New` + `Py_BuildValue` per item |

---

## Architecture

```
wrapper.cpp              ← CPython C API (26 bindings)
     │
     ▼
primitives/              ← 16 independent policy primitives
  selkv.h  smsa.h  delta_ar.h  delta_rank.h  head_gate.h
  state_compress.h  composite.h  pareto.h  cal.h  rcal.h
  gakv.h  ndpa.h  mpsr.h  ebar.h  ssr.h  ipss.h
     │
     ▼
core/                    ← Foundational types and signal extraction
  types.h  signal.h  config.h
     │
     ▼
Eigen3 + OpenMP          ← Linear algebra + parallelism
```

## Data Flow

```
Model forward pass → hidden_states
    │
    ▼
proxy_delta()  ──→ Δ scores [1 × S]
normalize()    ──→ norm scores
spearman()     ──→ ρ(Δ, attention)
    │
    ├──→ selkv_evict()           → indices to retain / evict
    ├──→ smsa_analyze()          → speedup, memory savings
    ├──→ head_active()           → how many heads are needed
    ├──→ delta_ar_indices()      → top-K per query
    ├──→ gakv_analyze()          → composite importance scores
    ├──→ ndpa_rectify()          → rectify Δ by attention correlation
    ├──→ mpsr_project()          → compress to lower dimension
    ├──→ ssr_analyze()           → soft-threshold singular values
    ├──→ ipss_analyze()          → rank heads by salience
    ├──→ ebar_analyze()          → modulate compute by entropy
    ├──→ cal/rcal_classify()     → assign compute tier
    └──→ pareto_sweep()          → find Pareto-optimal configs
```

---

## Citing

If you use this code in academic work:

```bibtex
@misc{opusedge2025,
  title = {Delta-Signal Unification: A Single Importance Metric
           for Sparse Computation on Hybrid SSM-Attention Models},
  author = {Irfan Mahir},
  year = {2025}
}
```

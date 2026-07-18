# Getting started

A five-minute walkthrough: build the engine, run the demo, wire a primitive into your own code.

## 1. Install prerequisites

```bash
# Debian / Ubuntu
sudo apt-get install libeigen3-dev cmake g++

# macOS
brew install eigen cmake
```

You need a compiler that speaks `c++20` (GCC 12+, Clang 15+, MSVC 19.32+).

## 2. Build the engine

```bash
git clone https://github.com/Mr-DS-ML-85/OpusEdge && cd OpusEdge

cmake -S engine -B engine/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPUSEDGE_BUILD_EXAMPLES=ON \
  -DOPUSEDGE_BUILD_BENCHMARKS=ON
cmake --build engine/build -j
```

The engine is **header-only** — you can also just `#include <opusedge/...>` into your own project
and link against Eigen. CMake install:

```bash
sudo cmake --install engine/build
```

then in your own `CMakeLists.txt`:

```cmake
find_package(opusedge REQUIRED)
target_link_libraries(myapp PRIVATE opusedge::opusedge)
```

## 3. Run the demo

```bash
./engine/build/examples/all_primitives_demo
```

You should see the full primitive matrix execute in <1 s — 30 primitives, Δ-signals,
SelKV eviction, SMSA windowing, Delta-AR routing, MPSR projection, EB-AR compute, SSR
soft-SVD, IPSS head fallback, CAL / R-CAL classification, Pareto sweep.

## 4. Use a primitive

Pick whatever fits your model:

```cpp
#include <opusedge/primitives/selkv.h>
#include <opusedge/primitives/smsa.h>
#include <opusedge/primitives/delta_ar.h>
#include <opusedge/core/signal.h>

using namespace opusedge;

// (a) extract Δ from any transformer family
VectorXf delta = SignalExtractor::proxy_delta(layer_hidden_states);

// (b) evict 75% of the KV cache
auto evict = SelKV::evict(delta, /*ratio=*/0.75, delta.size());

// (c) or route each query to the top-K keys before attention runs
auto indices = build_delta_ar_indices(delta, /*top_k=*/64);
auto out = sparse_attention(Q, K, V, indices.cast<int>());
```

Every primitive is a pure function of Δ (and a config struct). No mutable global state,
no CUDA dependency at the primitive layer.

## 5. Python (optional)

```bash
pip install -e engine/
```

then

```python
import _opusedge_cpp as oe

# eviction indices
retained, evicted = oe.selkv_evict(scores, 0.875)

# sliding-window analysis
speedup, mem_savings, effective_window = oe.smsa_analyze(seq_len=2048, window=64)

# Pareto sweep
points = oe.pareto_sweep(scores, seq_len=2048)   # (ratio, w, rank, keep, ppl, flops, mem, dppl)
```

The Python wrapper mirrors the C++ API 1:1 — 30 bindings total.

## Next

- [Primitives reference](primitives.md) — one-page-per-primitive with formulas and defaults.
- [Architecture](architecture.md) — how the telemetry / decision / execution layers fit together.
- [Benchmarks](benchmarks.md) — the paper numbers, reproducible on RTX 4060.

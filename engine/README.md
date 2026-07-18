# OpusEdge C++

**A signal-driven compute allocation library for hybrid SSM-attention, dense, and MoE transformers.**

```bash
pip install -e .                  # Python bindings
cmake -B build && cmake --build build  # C++ standalone
```

**What it does:** Extracts importance signals (Δ) from model hidden states, then calls C++ primitives — **10 core + 4 stabilizers + 2 task controllers, 30 Python bindings total** — to evict KV entries, gate heads, compress state, or tier compute in microseconds.

## Quick Start

```python
# Python
import _opusedge_cpp as oe
scores = [0.3, 0.6, 0.8, 0.2, 0.1, 0.5, 0.7, 0.4]
retained, evicted = oe.selkv_evict(scores, 0.5)   # drop 50% low-Δ tokens
```

```cpp
// C++
#include <opusedge/primitives/selkv.h>
opusedge::VectorXf delta(8); delta << 0.3, 0.6, 0.8, 0.2, 0.1, 0.5, 0.7, 0.4;
auto r = opusedge::SelKV::evict(delta, 0.5, 8);   // same result
```

```bash
# See everything run
./build/examples/all_primitives_demo               # 30+ primitives, pure C++
```

## Primitives

| Area | Primitive | What it does |
|---|---|---|
| **KV eviction** | SelKV, GAKV, R-GAKV | Drop low-importance cache entries |
| **Sparse attention** | SMSA, Delta-AR, EB-DAR | Limit attention scope to important tokens |
| **Head gating** | HeadGate, IPSS | Deactivate low-salience heads |
| **State compression** | StateCompress, MPSR, SSR | Compress SSM state dimensionality |
| **Compute modulation** | EB-AR | Scale compute by token entropy |
| **Tiering** | CAL, R-CAL | Task-rigidity → compute tier map |
| **Config search** | Pareto | Sweep eviction/window/rank tradeoffs |

Build → [docs/DETAILED.md](docs/DETAILED.md) for the full reference (all 26 bindings, APIs, performance tables, extension guide).

## Requirements

C++20, Eigen3 ≥ 3.4, Python ≥ 3.12. Built with GCC 15, no GPU needed for primitives.

## Cite

```bibtex
@misc{opusedge2025,
  title = {Delta-Signal Unification: A Single Importance Metric
           for Sparse Computation on Hybrid SSM-Attention Models},
  author = {Irfan Mahir},
  year = {2025}
}
```

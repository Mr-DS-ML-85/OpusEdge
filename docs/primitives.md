# Primitives reference

> ⚠️ **DUE TO HARDWARE CONSTRAINT WE WERE UNABLE TO TEST LARGER MODELS.** Every real-model
> benchmark in this repo runs on the 0.5B – 1.3B envelope. C++ SDK / bench targets are
> model-agnostic and only bounded by RAM.

## SDK — start here

Use the family-scoped pipeline instead of hand-picking primitives. The SDK enforces the
paper's applicability rules — you can't accidentally call SMSA on a dense model or NDPA on
a hybrid one.

**Python (`opusedge_cpp.sdk`):**

```python
from opusedge_cpp.sdk import DensePipeline, HybridPipeline, MoEPipeline, ModelShape

shape = ModelShape(n_layers=36, n_heads=8, n_kv_heads=2,
                   head_dim=128, hidden_dim=1024, seq_len=2048)
hybrid = HybridPipeline(shape)
result = hybrid.selkv(native_delta, ratio=0.875)
smsa   = hybrid.smsa_analyze()              # → {'speedup': 32.0, 'memory_savings': 0.97, ...}
rcal   = hybrid.rcal_classify()             # → {'tier_name': 'General', 'effective_threshold': 0.75, ...}
kv_mib = hybrid.kv_after_pipeline_mib()     # combined SelKV∩SMSA footprint
```

**C++ (`#include <opusedge/sdk.h>`):**

```cpp
using namespace opusedge::sdk;
ModelShape shape{ .n_layers=36, .n_heads=8, .n_kv_heads=2,
                  .head_dim=128, .hidden_dim=1024, .seq_len=2048 };
HybridPipeline hybrid(shape);
auto r  = hybrid.selkv(native_delta);
auto O  = hybrid.smsa_forward(Q, K, V);
auto rc = hybrid.rcal_classify();
```

Try `DensePipeline` for Qwen/LLaMA, `HybridPipeline` for Falcon-H1/Jamba, `MoEPipeline` for
Mixtral/OLMoE/Granite MoE. See `engine/include/opusedge/sdk.h` for the full class API.



Every primitive is a **pure function of Δ** (or a signal derived from Δ). Configs are plain structs;
defaults match the paper. The library ships **10 core primitives, 4 stabilizers, and 2 task
controllers** across 16 C++ headers, exposed as **30 Python bindings**.

| Category | Count | Members |
|----------|------:|---------|
| Core primitives | 10 | SelKV · SMSA · Delta-AR · ΔRank · HeadDeactivate · StateCompress · DenseEvic · GAKV/R-GAKV · Pareto Frontier · Router-IR |
| Stabilizers     |  4 | MPSR/SACT · EB-AR/EB-DAR · SSR/CASP + NDPA · IPSS/CSA |
| Task controllers|  2 | CAL · R-CAL |

---

## Δ — the shared signal

Three ways to obtain Δ, all consumed by every downstream primitive:

| Family | Extractor | Cost | ρ vs attention |
|--------|-----------|------|---------------:|
| **Hybrid SSM-attention** | native selectivity `dt` from the SSM block | O(1) — free | **0.51** (Falcon-H1) |
| **Dense**                | `proxy_delta` = per-layer RMS drift of hidden state, dim-normalised | O(L) | **0.28** (Qwen2.5) |
| **MoE**                  | `router_entropy` = normalised H of expert softmax | O(E) | — (used as IR) |

```cpp
// hybrid
VectorXf delta = SignalExtractor::native_delta(dt_per_layer, n_tokens);

// dense
VectorXf delta = SignalExtractor::proxy_delta(hidden_states);   // list of [T, d_model]

// MoE
VectorXf ir = SignalExtractor::compute_router_entropy_per_token(router_logits);
```

The **NDPA** rectifier optionally warps a Proxy-Δ onto the internal attention manifold via a
single least-squares step, closing part of the correlation gap without retraining.

---

## SelKV — Δ-guided KV cache eviction

Drop the bottom-p% tokens by Δ **before** the attention kernel runs. On Falcon-H1 the eviction
decision itself is **500× cheaper** than attention-based baselines because Δ is a by-product of
the SSM forward pass.

```cpp
auto ev = SelKV::evict(delta, /*ratio=*/0.875, delta.size());
// → EvictionResult { retained_indices, evicted_indices, memory_savings }
```

Empirical: **100.5×** quality retention over random at 87.5% compression (paper Table 5).

Variants: `SelKV::attention_mask(...)` returns a ready-to-multiply causal mask;
`SelKV::gakv_score(delta, router_entropy, α, β)` fuses Δ with MoE router confidence.

---

## SMSA — SSM-Masked Sparse Attention

In parallel hybrid blocks the SSM head already provides long-range coverage — the attention head
is redundantly recomputing it at O(S²) cost. SMSA replaces full attention with a sliding window
of width w scaled by 1/Δ.

```cpp
SMSA smsa(/*window=*/64);
auto r = smsa.analyze(seq_len);            // r.speedup, r.memory_savings
MatrixXf mask = smsa.attention_mask(seq_len);
```

At w=64 and 2,048 tokens: **3.56–4.98× speedup, 96.9% VRAM reduction** (paper Table 1).

---

## Delta-AR — pre-attention routing

SelKV evicts *after* attention wastes FLOPs; SMSA uses a fixed window ignoring content.
Delta-AR routes each query to the top-K keys **before** the softmax runs.

```cpp
auto indices = build_delta_ar_indices(delta, /*top_k=*/64);  // [S, K], -1 padded
auto out     = sparse_attention(Q, K, V, indices.cast<int>()); // O(S·K·D)
```

Attention FLOPs drop from O(S²) to O(S·K). Under EB-DAR, tokens masked from routing feed a
temporal reservoir `E` and are re-admitted when their residual grows past μ + 0.1σ:

```cpp
MatrixXf mask = build_delta_ar_mask(delta, K);
auto [reservoir, boosted] = ebdar(scores, mask, /*β=*/0.85);
```

---

## ΔRank — selectivity-adaptive low-rank projections

High-Δ tokens need full-rank Q/K/V; low-Δ tokens sit in a predictable regime where a low-rank
approximation is exact enough. Four-tier map:

| Δ range | rank | FLOP reduction |
|--------:|-----:|---------------:|
| < 0.05 | 16 | **64×** |
| 0.05 – 0.20 | 32 | 16× |
| 0.20 – 0.50 | 64 | 4× |
| ≥ 0.50 | 128 | 1× (full) |

Paired with **SSR** (soft spectral relaxation) to avoid the perplexity cliff of hard SVD:

```cpp
MatrixXf W_soft = DeltaRank::ssr(W, /*rank_fraction=*/0.5, /*α=*/2.0);
```

---

## HeadDeactivate — adaptive multi-head gating

Gate heads by per-token informational entropy — 4/16/24/32 active tiers on a 32-head model.
Sub-threshold heads fall through to the **IPSS** linear-time path
(K̄ = time-averaged key vector, one V·mean matmul), avoiding hard cliffs.

```cpp
HeadGate hg;
int active = hg.active_heads(delta_t);
auto stats = hg.analyze(delta_vec);   // → mean active, FLOP reduction
```

---

## StateCompress — SSM state channel compression

When Δ ≪ τ_compress the SSM state update collapses to passive decay; high-frequency channels
become redundant. StateCompress zeros the bottom-(1−γ)% channels by magnitude.

```cpp
StateCompress sc;               // γ_min=0.25, γ_max=1.0, τ_c=0.05, τ_f=0.30
VectorXf s2 = sc.compress_channels(state, delta_t);
```

Bonus on mobile silicon: zeroed cells don't toggle in SRAM → measurable dynamic-power drop.

---

## DenseEvic — dense-architecture KV eviction

The dense counterpart to SelKV. Partitions the cache into a **Protected Boundary**
(system prompt + rolling recent tokens) and an **Eviction Candidate Pool**, then applies a
depth-weighted Proxy-Δ score. Empirical: **13.7×** on Qwen2.5-0.5B at 87.5% compression.

---

## GAKV / R-GAKV — gating-aware KV (semantic cache)

Composite retention score `S = α · Δ_norm + β · IR_norm`. A token is evicted only when
**both** signals fall below their thresholds. R-GAKV multiplies the thresholds by
`(1 + rigidity)` from CAL, so reasoning/code tasks keep more.

```cpp
GAKV g;
auto r = g.analyze(delta_scores, ir_scores);
// composite_scores, retained_indices, evicted_indices
```

---

## CAL / R-CAL — contextual accuracy locking

CAL classifies the prompt into a rigidity tier (`code=0.85`, `math=0.85`, `summarize=0.30`, …)
and modulates every primitive's threshold by `(1 + rigidity)`. R-CAL adds an EMA over token
log-prob confidence and can *freeze* recomputation of the classifier once the model has settled.

```cpp
CAL cal;
auto tier = cal.classify("math");           // → {"High", ComputeTier::High, 0.85, 0.00}
Float τ_eff = cal.effective_threshold("math");  // 0.5 * (1 + 0.85) = 0.925
```

---

## Pareto Frontier — control-plane

Given a Δ vector and a sequence length, sweep the discrete config grid
(ratio × window × rank × channel-keep), keep the Pareto-optimal points, expose the **knee**.
The runtime toggles primitives dynamically to sit on the knee for the current thermal /
battery envelope.

```cpp
auto points = ParetoFrontier::sweep(delta, seq_len);
auto frontier = ParetoFrontier::pareto_optimal(points);
auto knee = ParetoFrontier::knee_point(frontier);
```

---

## Stabilizers

Four small primitives that prevent quality cliffs under aggressive compression:

| Stabilizer | Fix | Where it lives |
|------------|-----|----------------|
| **MPSR / SACT** | Projects evicted KV → SSM state so context isn't lost | `mpsr.h` |
| **EB-AR**       | Scales compute per-step by output entropy | `ebar.h` |
| **SSR / CASP**  | Sigmoid-gated soft SVD (replaces hard truncation) | `ssr.h`, `delta_rank.h` |
| **IPSS / CSA**  | Linear-time head fallback for sub-salient heads | `ipss.h`, `head_gate.h` |

Every stabilizer is a plain function you can drop in wherever you're already doing the
corresponding hard operation.

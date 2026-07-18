# Architecture

OpusEdge is a **three-layer decision pipeline** that sits between the model and the runtime.
The model emits telemetry, the policy engine emits decisions, the execution layer applies them.
None of the layers touches the model weights.

```
┌─────────────────────────────────────────────────────────────────────┐
│                              Model                                  │
│      Falcon-H1  ·  Qwen2.5  ·  Jamba  ·  Mixtral  ·  OLMoE          │
└──────┬──────────────┬───────────────────┬───────────────────────────┘
       │              │                   │
       ▼              ▼                   ▼
 ┌──────────┐  ┌──────────────┐   ┌──────────────┐
 │  native  │  │  Proxy-Δ     │   │  Router-IR   │      Telemetry Layer
 │    Δ     │  │  RMS drift   │   │  softmax H   │      (forward hooks)
 └────┬─────┘  └──────┬───────┘   └──────┬───────┘
      └──────────┬────┴──────────────────┘
                 ▼
     ┌────────────────────────────┐
     │      Signal Extractor      │       unified Δ / IR vector
     └───────────────┬────────────┘
                     ▼
     ┌────────────────────────────┐
     │       Policy Engine        │       Decision Layer
     │                            │       - CAL classifier
     │  ┌─── Pareto Frontier ──┐  │       - R-CAL EMA freezer
     │  │ knee-point selector  │  │       - per-tier thresholds
     │  └──────────────────────┘  │
     └───────────────┬────────────┘
                     ▼
 ┌────────────┬──────┴──────┬───────────────┬────────────────┐
 │  masking   │  weight-op  │    hooks      │    cache-mgmt  │       Execution
 │  SelKV     │   ΔRank     │  HeadDeact.   │    DenseEvic   │       Layer
 │  SMSA      │   SSR/CASP  │  IPSS         │    MPSR/SACT   │
 │  Delta-AR  │   NDPA      │  StateCompr.  │                │
 └────────────┴─────────────┴───────────────┴────────────────┘
```

## Telemetry layer

Forward-hooks on every attention, SSM, and MLP module capture:

- `Δ` (native or Proxy-)
- attention weight matrices (for correlation + audit)
- router logits (for MoE IR)

Overhead measured at **≤ 1% of inference latency** in the paper's PyTorch eager harness.

## Decision layer

The policy engine is a configurable rule set. Inputs: Δ vector, task label,
CAL rigidity, current KV budget. Outputs: per-token, per-layer primitive configs.

Two things happen here:

1. **CAL tier classification** — the prompt is bucketed by task (code/math/reasoning/…);
   the rigidity multiplier bumps every downstream threshold. R-CAL adds an EMA over
   token log-prob confidence and freezes recomputation once the model has settled.

2. **Pareto knee selection** — the sweep grid (eviction × window × rank × channel-keep)
   is scored offline; the frontier is stored; at runtime the knee point is picked based
   on thermal / battery envelope.

## Execution layer

Decisions land in four channels, and each maps 1:1 to primitives:

| Channel | Primitives |
|---------|------------|
| **attention masking** | SelKV, SMSA, Delta-AR |
| **weight modification** | ΔRank + SSR, CASP, NDPA |
| **forward hooks**       | HeadDeactivate + IPSS, StateCompress |
| **cache management**    | DenseEvic, MPSR / SACT |

## Stabilizers — the four fixes

Aggressive compression breaks things unless you catch it. OpusEdge ships four small primitives
whose only job is to prevent quality cliffs:

- **MPSR / SACT** — project evicted KV → SSM state (context conserved, not lost).
- **EB-AR** — modulate compute per-step by output entropy (never underspend when the model is uncertain).
- **SSR / CASP** — sigmoid-gated soft SVD replaces hard rank truncation.
- **CSA / IPSS** — sub-salient heads fall through to a linear-time K̄ path instead of being zeroed.

## The audit trail

Every inference cycle emits a **Reasoning Snapshot**:

```json
{"step": 42, "primitive": "SelKV", "confidence": 0.93, "action": "Reduce"}
```

Post-hoc verification of quality-efficiency trade-offs, needed for production regulated deployments.

## Reference implementation

The C++ engine in [`engine/`](../engine) is header-only. Every primitive is a pure function
of Δ; there is no mutable global state and no CUDA dependency at the primitive layer.

The paper's full Infernix Engine (Python + Rust/CUDA) is the production stack — this repo is the
**reference implementation** you can port anywhere.

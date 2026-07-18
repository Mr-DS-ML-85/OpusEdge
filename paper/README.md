# OpusEdge

**Zero-Cost Dynamic Compute Allocation for LLM Inference**

A plug-and-play Python library that extracts **Δ** (SSM discretization timestep) signals from any HuggingFace causal LM and uses them to drive **15+ inference sparsity primitives** — automatically choosing the right combination for your model and task.

```
pip install opusedge
```

> **Google Colab ready.** Works with free T4 (15 GB VRAM) and 4-bit quantization.  
> Only FP/BF16 or bitsandbytes-quantized models are supported — GGUF/GPTQ/AWQ use fused kernels that break the hooks.

---

## Quick Start

```python
from opusedge import create_engine

engine = create_engine("Qwen/Qwen2.5-0.5B-Instruct").load()
result = engine.generate("The capital of France is")
print(result["text"])
engine.unload()
```

### Streaming

```python
from opusedge import OpusEdgePipeline, create_engine

engine = create_engine("Qwen/Qwen2.5-0.5B-Instruct").load()
pipe = OpusEdgePipeline(engine)

for token in pipe.generate_stream("Explain quantum computing in 3 sentences.",
                                   max_new_tokens=128):
    print(token["text"], end="", flush=True)
```

### With 4-bit Quantization (fits smaller GPUs)

```python
from opusedge import OpusEdgeConfig, OpusEdgeEngine

cfg = OpusEdgeConfig(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    quantize="4bit",
    max_memory_gb=6.0,
)
engine = OpusEdgeEngine(cfg).load()
result = engine.generate("What is the speed of light?")
print(result["text"])
engine.unload()
```

---

## Installation

### From PyPI (recommended)

```bash
pip install opusedge
```

### From source

```bash
git clone https://github.com/your-org/opusedge
cd OpusEdge/python
pip install -e .
```

### Dependencies

| Package       | Minimum | Why                     |
|---------------|---------|-------------------------|
| `torch`       | ≥ 2.1   | Core tensor ops         |
| `transformers`| ≥ 4.40  | HF model loading        |
| `accelerate`  | ≥ 0.30  | Device map + offload    |
| `scipy`       | ≥ 1.10  | Spearman correlation    |
| `bitsandbytes`| ≥ 0.43  | 4-bit / 8-bit quant     |
| `datasets`    | ≥ 2.18  | WikiText-103 eval       |
| `sentencepiece`| ≥ 0.2  | Tokenizer support       |
| `protobuf`    | ≥ 4.25  | Tokenizer support       |

> **Important for Colab:** `bitsandbytes` needs a version compiled for your CUDA. In Colab (CUDA 12.x):
> ```bash
> pip install bitsandbytes>=0.43.0
> ```

---

## Colab Quick-Start Notebook

Create a new notebook and run:

```
# Cell 1
!pip install opusedge

# Cell 2
from opusedge import create_engine
engine = create_engine(
    "Qwen/Qwen2.5-0.5B-Instruct",
    quantize="4bit",
    max_memory_gb=6.0,
).load()
print("Loaded on:", engine.device)

# Cell 3
result = engine.generate("Explain the transformer architecture in 3 sentences.",
                          max_new_tokens=128)
print(result["text"])
print(f"Generated {result['generated_len']} tokens in {result['total_ms']:.0f}ms")
engine.unload()
```

**Tip:** For Colab T4 (15 GB VRAM), use `max_memory_gb=10.0` for larger models like Llama-3.1-8B in 4-bit.

---

## Architecture Detection

OpusEdge automatically detects which architecture family your model belongs to:

| Family   | Detection                           | Example Models                     |
|----------|-------------------------------------|------------------------------------|
| **dense**   | Standard transformer (QKV attn)  | Qwen2.5-0.5B, Llama-3.1-8B       |
| **hybrid**  | SSM layers + attention layers    | Falcon-H1-0.5B, Jamba-3B          |
| **ssm**     | No attention layers (Mamba-only) | LFM2-1.2B (Liquid AI)             |
| **moe**     | Has router / expert gate modules | OLMoE-1B-7B, Huihui-MoE-1B        |

```python
from opusedge import Extractor

ext = Extractor("tiiuae/Falcon-H1-0.5B-Instruct").load()
print(ext.detected_family)     # "hybrid"
print(ext.n_ssm_layers)        # number of SSM layers detected
print(ext.n_attn_layers)       # number of attention layers
ext.unload()
```

---

## Signal Extraction

The core idea: extract importance signals and use them to guide sparse computation.

### For hybrid/SSM models (native Δ)

```python
ext = Extractor("tiiuae/Falcon-H1-0.5B-Instruct").load()
out = ext.forward("The capital of France is", max_len=512)

# Per-token Δ importance scores
delta_scores = Extractor.delta_scores(out["deltas"], len(out["tokens"]))
# Per-token attention importance scores
attn_scores = Extractor.attn_scores(out["attn_w"], len(out["tokens"]))
```

### For dense models (Proxy-Δ)

Dense transformers don't produce a native Δ signal. OpusEdge uses the **Proxy-Δ** approximation: per-layer hidden-state RMS drift.

```python
ext = Extractor("Qwen/Qwen2.5-0.5B-Instruct").load()
out = ext.forward("The capital of France is", max_len=512,
                  output_hidden_states=True)

# Proxy-Δ = RMS difference between consecutive hidden states
proxy_delta = Extractor.proxy_delta(out["hidden_states"])
```

### What's in the forward output

```python
out = ext.forward("Hello world", max_len=128)
# out keys:
#   "input_ids"      — torch.Tensor [1, S]
#   "tokens"         — list[str] decoded tokens
#   "logits"         — torch.Tensor [1, S, V]
#   "hidden_states"  — list[Tensor] | None (per-layer, only if requested)
#   "deltas"         — dict[str, Tensor] per-SSM-layer Δ
#   "attn_w"         — dict[str, Tensor] per-attention-layer weights
#   "router"         — dict[str, Tensor] per-MoE-router softmax
#   "fwd_ms"         — float (forward pass wall time)
```

---

## Primitives

All **15 modules** (16 engine primitives, ~25 individual techniques) exposed via `opusedge.primitives`.

### Applicability Matrix

Verified against paper v14.4 §4.1–4.10 + code (benchmark.py FAMILY_TESTS):

| Primitive | Dense | Pure SSM | Hybrid | MoE |
|-----------|-------|----------|--------|-----|
| SelKV (§4.1) | ✓ (Proxy-Δ) | n/a⁵ | ✓ (native Δ) | ✓ (native Δ) |
| SACT (§3.4) | n/a¹ | ✓ | ✓ | n/a¹ |
| SMSA (§4.2) | ✓ | n/a⁵ | ✓ | ✓ |
| Delta-AR (§4.3) | ✓ (Proxy-Δ) | n/a⁵ | ✓ (native Δ) | ✓ (native Δ) |
| ΔRank + SSR (§4.4) | ✓ | n/a⁴ | ✓ | ✓ |
| ΔHeadDeactivate / IPSS (§4.5) | ✓ | n/a⁵ | ✓ | ✓ |
| ΔStateCompress (§4.6) ★ | ✓² | ✓² | ✓² | ✓² |
| DenseEvic (§4.7) | ✓ | n/a | n/a | n/a |
| MoE I_R signal (§4.9) | n/a | n/a | n/a | ✓ |
| GAKV (§4.10) | ✓³ | n/a | ✓³ | ✓ |
| S-GAKV | ✓ | ✓ | ✓ | ✓ |
| CAL (§5) | ✓ | ✓ | ✓ | ✓ |
| R-CAL (SSM freeze) | n/a¹ | ✓ | ✓ | n/a¹ |
| MPSR (§1.1) | n/a¹ | ~ | ~ | n/a¹ |
| EB-AR (§1.2) | ✓ | ✓ | ✓ | ✓ |
| CASP + NDPA (§1.3) | ✓ | n/a⁴ | ✓ | ✓ |
| CSA (§1.4) | ✓ | n/a⁵ | ✓ | ✓ |

`✓` = runnable. `~` = surrogate (needs weight training for paper claim). `n/a` = no applicable internal state.
¹ No SSM recurrent state → MPSR/SACT/R-CAL inapplicable.  
² Targets MLP down-proj channels — universal to all transformer families (§4.6).  
³ Δ-only (ω₂=0) when no router signal available.  
⁴ Pure SSM (e.g. LFM2) has no attention Q/K/V projections.  
⁵ No KV cache / softmax attention → attention primitives inapplicable.

**Signal source per family:**
- **Native Δ** (SSM layers) → free, extracted via forward hooks → SelKV, Delta-AR, ΔStateCompress
- **Proxy-Δ** (dense hidden-state RMS drift) → O(L) cost → DenseKV, DenseEvic, NDPA
- **Router I_R** (MoE expert gates) → entropy of routing distribution → GAKV
- **Attention weights** (all attn-bearing layers) → SMSA, ΔRank, ΔHeadDeactivate, CSA

| # | Module | Techniques | Paper § |
|---|--------|------------|---------|
| 1 | `selkv` | SelKV, SACT (§3.4 residual) | §4.1 |
| 2 | `smsa` | SMSA (sliding-window attention) | §4.2 |
| 3 | `delta_ar` | Delta-AR, EB-DAR | §4.3 |
| 4 | `delta_rank` | ΔRank, SSR (§3.6) | §4.4 |
| 5 | `head_gate` | ΔHeadDeactivate, IPSS (§3.7) | §4.5 |
| 6 | `state_compress` | ΔStateCompress | §4.6 |
| 7 | `densekv` | DenseKV, DenseEvic | §4.7 |
| 8 | `densevic` | DenseEvic baselines (random/attn/proxy) | §4.7 |
| 9 | `gakv` | MoE I_R, GAKV Semantic Cache | §4.9-4.10 |
| 10 | `sgakv` | S-GAKV (SVD compression) | §10b |
| 11 | `composite` | Multi-primitive orchestration | §4 |
| 12 | `pareto` | Pareto frontier optimization | §4.8 |
| 13 | `cal` | CAL / R-CAL (R-CAL = SSM state-freeze extension) | §5 |
| 14 | `reasoning` | Reasoning snapshot audit log | §5.2 |
| 15 | `novel` | MPSR (§1.1), EB-AR (§1.2), CASP (§1.3), NDPA (§1.3), CSA (§1.4) | §1 |

### 1. SelKV — Δ-Guided KV Eviction

Evicts the least-important KV cache entries based on Δ scores.

```python
from opusedge.primitives import selkv as P_selkv

out = ext.forward("Long context passage...", max_len=1024)
delta = Extractor.delta_scores(out["deltas"], len(out["tokens"]))

# Sweep eviction ratios
sweep = P_selkv.sweep(ext.model, out["input_ids"], delta)
for r in sweep:
    print(f"Evict {r['ratio']*100:.0f}%: "
          f"SelKV PPL={r['selkv_ppl']:.2f}  Random PPL={r['random_ppl']:.2f}")
```

### 2. SMSA — Sliding-Window Attention

Standard sliding-window attention (Paper §4.2). Caps KV cache to `window` tokens regardless of sequence length.

```python
from opusedge.primitives import smsa as P_smsa

results = P_smsa.sweep(seq_lengths=[8192, 16384, 32768, 65536])
for r in results:
    print(f"Seq={r['seq_len']}: full={r['full_ms']:.1f}ms "
          f"SW={r['sw_ms']:.1f}ms speedup={r['speedup']:.1f}× "
          f"KV mem↓={r['kv_memory_reduction_pct']:.0f}%")
```

### 3. Delta-AR — Δ-Routed Sparse Attention

Builds a sparse causal mask based on top-Δ tokens.

```python
from opusedge.primitives import delta_ar as P_delta_ar

out = ext.forward("The theory of relativity states that", max_len=512)
delta = Extractor.delta_scores(out["deltas"], len(out["tokens"]))
result = P_delta_ar.measure(ext.model, out["input_ids"], delta, top_k=64)
print(f"Baseline PPL={result['baseline_ppl']:.3f} → "
      f"Delta-AR={result['delta_ar_ppl']:.3f} "
      f"(attended {result['tokens_attended']}/{result['total_tokens']})")
```

### 4. ΔRank — Singular-Value Retention

Low-rank projection of attention weight matrices guided by Δ importance.

```python
from opusedge.primitives import delta_rank as P_rank

out = ext.forward("Quantum entanglement is", max_len=512)
delta = Extractor.delta_scores(out["deltas"], len(out["tokens"]))
result = P_rank.measure(ext.model, out["input_ids"],
                         rank_fraction=0.70, blend=0.85)
```

### 5. ΔHeadDeactivate / IPSS — Head Salience Gating

Deactivates attention heads with low Δ salience.

```python
from opusedge.primitives import head_gate as P_head

out = ext.forward("The mitochondria is the", max_len=512)
delta = Extractor.delta_scores(out["deltas"], len(out["tokens"]))
result = P_head.measure(ext.model, out["input_ids"], delta, kappa=0.3)
```

### 6. ΔStateCompress — Channel Zeroing

Zeros out low-importance channels in SSM states. The **star primitive** — achieves highest compression with minimal PPL loss.

```python
from opusedge.primitives import state_compress as P_compress

out = ext.forward("DNA replication involves", max_len=512)
delta = Extractor.delta_scores(out["deltas"], len(out["tokens"]))
result = P_compress.measure(ext.model, out["input_ids"], delta)
print(f"Zerod {result['avg_channels_zeroed_pct']:.1f}% channels "
      f"at +{result['delta_ppl']:+.2f} PPL")
```

### 7. GAKV — Gating-Aware KV Eviction (MoE only)

For mixture-of-experts models: combines Δ scores with router entropy.

```python
from opusedge.primitives import gakv as P_gakv

out = ext.forward("Explain the training of a mixture of experts model.", max_len=512)
delta = Extractor.delta_scores(out["deltas"], len(out["tokens"]))
result = P_gakv.gakv_eviction(ext.model, out["input_ids"], delta,
                               router=out["router"], ratio=0.30)
```

### 8. DenseKV — Proxy-Δ Eviction Runtime (dense only)

For dense transformers: uses Proxy-Δ hidden-state drift as the eviction signal.

```python
from opusedge.primitives import densevic as P_densevic

out = ext.forward("The history of machine learning", max_len=512,
                  output_hidden_states=True)
proxy_delta = Extractor.proxy_delta(out["hidden_states"])
attn = Extractor.attn_scores(out["attn_w"], len(out["tokens"]))

comparison = P_densevic.compare(ext.model, out["input_ids"],
                                 proxy_delta, attn)
```

### 9. CAL — Contextual Accuracy Locking

Automatically selects the compute tier based on task difficulty.

```python
from opusedge.primitives import cal as P_cal

tier = P_cal.classify("Math: Solve this integral", tau_global=0.5)
print(tier.name)              # "High", "Balanced", or "Efficiency"
print(tier.rigidity)          # task difficulty score (0.0–1.0)
```

### 10. Composite — Orchestrate Multiple Primitives

Chain primitives together with a preset or custom config.

```python
from opusedge.primitives.composite import (
    measure_composite, CompositeConfig, PRESETS
)

out = ext.forward("Long document text...", max_len=1024)
delta = Extractor.delta_scores(out["deltas"], len(out["tokens"]))

# Use a preset
result = measure_composite(ext.model, out["input_ids"], delta,
                            PRESETS["opusedge_balanced"])
print(f"Composite PPL={result['composite_ppl']:.3f} "
      f"ΔPPL={result['delta_ppl']:+.3f}")

# Or build a custom config
cfg = CompositeConfig(
    name="my_config",
    eviction_ratio=0.50,
    window=256,
    top_k_ar=64,
    rank_fraction=0.70,
)
result = measure_composite(ext.model, out["input_ids"], delta, cfg)
```

### 11. Pareto Frontier — Multi-Objective Sweep

Finds the optimal set of configurations balancing PPL vs compute cost.

```python
from opusedge.primitives import pareto as P_pareto

out = ext.forward("Energy conservation law", max_len=512)
delta = Extractor.delta_scores(out["deltas"], len(out["tokens"]))
result = P_pareto.sweep(ext.model, out["input_ids"], delta)
print("Pareto front:", result["pareto_front"])
print("Knee point:", result["knee"])
```

### 15. Reasoning Snapshot — Audit Log

Records confidence and tier decisions at each generation step.

```python
from opusedge.primitives.reasoning import ReasoningLog, confidence_from_delta

log = ReasoningLog()
log.record(step=1, primitive="selkv", confidence=0.85, tier="Balanced",
           action="evict 50% of KV entries")
print(log.summary())
print(log.to_json())
```

---

## Engine API

The `OpusEdgeEngine` is the main inference interface with runtime primitive installation.

```python
from opusedge import OpusEdgeConfig, OpusEdgeEngine

# With auto tier selection via CAL
cfg = OpusEdgeConfig(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    tier="auto",
    cal_enabled=True,
    selkv=True,
    smsa=True,
    delta_ar=True,
    max_new_tokens=256,
)
engine = OpusEdgeEngine(cfg).load()

# Set task label for CAL-aware tier selection
engine.set_task("Math: Algebra")

# Extract signals manually
signals = engine.extract_signals("What is 2+2?", max_len=128)

# Generate with primitives
result = engine.generate("Solve for x: 2x + 5 = 15", max_new_tokens=128)
print(f"Tier used: {result['tier']}")
print(f"Primitives: {result['enabled_primitives']}")
print(f"Speed: {result['tokens_per_sec']:.1f} tok/s")

# Benchmark forward pass
stats = engine.benchmark_forward("The quick brown fox", repeats=5)

engine.unload()
```

### Engine generate output

```python
result = engine.generate("Hello", max_new_tokens=64)
# {
#   "text": str,               # Generated text
#   "prompt": str,             # Original prompt
#   "prompt_len": int,         # Input token count
#   "generated_len": int,      # Output token count
#   "total_len": int,          # Total tokens
#   "gen_ms": float,           # Generation time (ms)
#   "total_ms": float,         # Total time including prefill
#   "tokens_per_sec": float,   # Throughput
#   "tier": str,               # CAL tier used
#   "enabled_primitives": list,# Active primitives
#   "n_hooks_installed": int,  # Forward hooks count
# }
```

---

## Pipeline API

The `OpusEdgePipeline` adds streaming, adaptive tier switching, and advanced eviction strategies.

```python
from opusedge import OpusEdgePipeline, create_engine

engine = create_engine(
    "tiiuae/Falcon-H1-0.5B-Instruct",
    tier="auto",
    quantize="4bit",
).load()

pipe = OpusEdgePipeline(engine)

# Streaming with adaptive tier
for token in pipe.generate_stream(
    "Write a short story about AI.",
    task_label="Creative Writing",
    max_new_tokens=256,
):
    if token["done"]:
        print(f"\n[Done: {token['tokens_per_sec']:.1f} tok/s]")
    else:
        print(token["text"], end="", flush=True)

engine.unload()
```

### Adaptive Tier Switching

The pipeline monitors Δ entropy in real-time and switches tiers mid-generation:

- **High Δ entropy** → "High" tier (conservative, more compute)
- **Low Δ entropy** → "Efficiency" tier (aggressive sparsity)

```python
# Manual access to entropy monitoring
entropy = pipe._delta_entropy(pipe._delta_history[-10:])
print(f"Current Δ entropy: {entropy:.4f}")
```

### SelKV-2 / SelKV-3: Advanced Eviction

```python
# SelKV-2: delta-guided KV prefetching
kv = pipe.prefetch_kv(past_key_values, delta_scores, prefetch_ratio=0.3)

# SelKV-3: progressive prefill
result = pipe.progressive_prefill("Very long document...", max_len=4096)

# DenseKV generation (dense models only)
result = pipe.densekv_generate("Explain neural networks.", max_new_tokens=128)
```

---

## Configuration Reference

### `OpusEdgeConfig` Fields

| Parameter           | Type           | Default                          | Description                          |
|---------------------|----------------|----------------------------------|--------------------------------------|
| `model_id`          | `str`          | `Qwen/Qwen2.5-0.5B-Instruct`    | HF model ID or local path            |
| `device`            | `str`          | `auto`                           | `auto`, `cuda`, `cpu`                |
| `dtype`             | `str`          | `auto`                           | `auto`, `float16`, `bfloat16`        |
| `quantize`          | `str\|None`    | `None`                           | `None`, `4bit`, `8bit`               |
| `max_memory_gb`     | `float`        | `6.0`                            | VRAM ceiling for offload             |
| `tier`              | `str`          | `auto`                           | `High`, `Balanced`, `Efficiency`, `auto` |

### Tier Policies

| Tier         | Eviction Cap | Window | Rank Keep | Head Keep | Channel Keep | CSA  | EB-DAR | SACT |
|--------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **High**     | 30%  | 512 | 85% | 75% | 75% | ✓ | ✓ | ✓ |
| **Balanced** | 60%  | 256 | 70% | 50% | 62% | ✓ | ✓ | ✓ |
| **Efficiency**| 85% | 128 | 50% | 30% | 50% | ✗ | ✗ | ✗ |

### Per-Primitive Toggles

All primitives can be individually enabled/disabled in `OpusEdgeConfig`:

```python
cfg = OpusEdgeConfig(
    model_id="...",
    selkv=True,
    smsa=True,
    delta_ar=False,
    delta_rank=True,
    head_gate=False,
    state_compress=True,
    gakv=True,
    sgakv=False,
    sact=True,
    csa=False,
    ebdar=True,
    mpsr=False,
    ebar=True,
    casp=True,
    ndpa=False,
    densekv=True,
)
```

### Generation Parameters

| Parameter           | Default | Description                  |
|---------------------|---------|------------------------------|
| `max_new_tokens`    | 256     | Max tokens to generate       |
| `temperature`       | 0.7     | Sampling temperature         |
| `top_p`             | 0.9     | Nucleus sampling cutoff      |
| `repetition_penalty`| 1.0     | Repetition penalty           |
| `do_sample`         | True    | Whether to sample            |
| `cal_tau_global`    | 0.5     | CAL classification threshold |
| `head_kappa`        | 0.3     | Head gating sensitivity      |
| `sact_residual`     | 0.15    | SACT residual ratio          |
| `ebdar_beta`        | 0.85    | EB-DAR damping factor        |

---

## CLI Usage

```bash
# List available models
uv run opusedge --list-models

# Run all tests on a model
uv run opusedge bench --model Qwen/Qwen2.5-0.5B-Instruct --html report.html

# Run specific tests
uv run opusedge --model tiiuae/Falcon-H1-0.5B-Instruct --tests corr,selkv,smsa

# Inference mode
uv run opusedge inx --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hello" --stream

# Inference with interactive prompt loop
uv run opusedge inx --model Qwen/Qwen2.5-0.5B-Instruct --interactive

# Benchmark all models
uv run opusedge bench --all-models --html report.html

# Long-context benchmark
uv run opusedge-longctx --lengths 8192,16384,32768 --out results.json

# Export signals for C++ integration
uv run opusedge-export --out exports/ --models qwen2_5_0_5b,falcon_h1_0_5b

# With 4-bit quantization
uv run opusedge --model meta-llama/Llama-3.1-8B-Instruct --quantize 4bit
```

### Available Tests

| Test Key      | Description                                |
|---------------|--------------------------------------------|
| `corr`        | Spearman ρ(Δ, attention)                   |
| `corr_ppl`    | Spearman ρ(Δ, ΔPPL) — what abstract claims |
| `selkv`       | KV eviction sweep                          |
| `smsa`        | Sliding-window microbench                  |
| `delta_ar`    | Δ-routed sparse attention + EB-DAR         |
| `rank`        | SSR low-rank projection                    |
| `head`        | IPSS head salience gating                  |
| `compress`    | ΔStateCompress channel zeroing             |
| `gakv`        | Gating-Aware KV (MoE)                      |
| `memory`      | Memory footprint                           |
| `densevic`    | Dense eviction baselines (dense only)      |
| `sgakv`       | SVD compression (SSM)                      |
| `cal`         | CAL tier classification                    |
| `reasoning`   | Reasoning snapshot audit log               |
| `wikitext`    | WikiText-103 PPL baseline                  |
| `pareto`      | Pareto frontier optimization               |
| `all`         | Everything                                 |

---

## Evaluation

```python
from opusedge.eval import (
    spearman_delta_attention,
    spearman_delta_ppl_drop,
    wikitext_ppl,
    fmt_table,
)

# Correlation between Δ and attention importance
ext = Extractor("tiiuae/Falcon-H1-0.5B-Instruct").load()
out = ext.forward("The capital of France is", max_len=512)
n = len(out["tokens"])
d = Extractor.delta_scores(out["deltas"], n)
a = Extractor.attn_scores(out["attn_w"], n)

rho = spearman_delta_attention(d, a)
print(f"ρ = {rho['rho']:+.4f}, p = {rho['p']:.2e}, n = {rho['n']}")

# Per-layer breakdown
from opusedge.eval import per_layer_spearman
layers = per_layer_spearman(out["deltas"], out["attn_w"], n)
for layer_result in layers:
    print(f"Layer {layer_result['layer']}: ρ={layer_result['rho']:+.4f}")

# WikiText-103
ppl = wikitext_ppl(ext.model, ext.tok)
print(f"WikiText PPL = {ppl:.4f}")

ext.unload()
```

---

## Model Registry

Built-in model shortcuts in `opusedge.tasks.MODELS`:

| Key               | Model ID                                    | Family  | Default Quant |
|-------------------|---------------------------------------------|---------|:---:|
| `falcon_h1_0_5b` | `tiiuae/Falcon-H1-0.5B-Instruct`          | hybrid  | —    |
| `qwen2_5_0_5b`   | `Qwen/Qwen2.5-0.5B-Instruct`              | dense   | —    |
| `jamba_3b`       | `ai21labs/AI21-Jamba-Reasoning-3B`        | hybrid  | 4bit |
| `olmoe_1b_7b`    | `allenai/OLMoE-1B-7B-0125-Instruct`       | moe     | 4bit |
| `llama_3_1_8b`   | `meta-llama/Llama-3.1-8B-Instruct`        | dense   | 4bit |
| `mistral_7b`     | `mistralai/Mistral-7B-Instruct-v0.3`      | dense   | 4bit |
| `qwen2_5_7b`     | `Qwen/Qwen2.5-7B-Instruct`                | dense   | 4bit |
| `qwen3_8b`       | `Qwen/Qwen3-8B`                           | dense   | 4bit |
| `lfm2_1_2b`      | `LiquidAI/LFM2-1.2B`                      | ssm     | —    |

Use shortcuts with the CLI:

```bash
uv run opusedge --model falcon_h1_0_5b --tests all
uv run opusedge inx --model qwen2_5_0_5b --prompt "Hello"
uv run opusedge bench --model llama_3_1_8b --html report.html
```

Or in Python:

```python
from opusedge.tasks import MODELS

model_id, family, quant = MODELS["falcon_h1_0_5b"]
# model_id = "tiiuae/Falcon-H1-0.5B-Instruct"
# family   = "hybrid"
# quant    = None
```

---

## Signal Export (for C++/Rust integration)

Export per-model Δ and attention signals to `.pt` files:

```python
from opusedge.export import export_model, export_all

# Single model
export_model("my_model", "Qwen/Qwen2.5-0.5B-Instruct",
             family="dense", quantize=None,
             prompts=["Hello world", "Test prompt"],
             out_dir="exports/")

# All models in registry
export_all(out_dir="exports", max_prompts=8)
```

Each export creates a directory with `.pt` files containing `deltas`, `attn_w`, `hidden_states`, and metadata.

---

## Benchmark Suite

### Single model

```python
from opusedge.benchmarks import benchmark_model

results = benchmark_model(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    family="dense",
    max_memory_gb=6.0,
    html_path="report.html",
)
```

### All models

```python
from opusedge.benchmarks import benchmark_all

results = benchmark_all(
    max_prompts=3,
    html_path="opusedge_benchmark_all.html",
    json_path="opusedge_benchmark_all.json",
)
```

### Long-context benchmark

```python
from opusedge.benchmarks.long_context import benchmark

results = benchmark(
    seq_lengths=[8192, 16384, 32768, 65536],
    n_heads=8,
    head_dim=64,
    window=256,
    hidden=2048,
)
```

### Family-aware test routing

```python
from opusedge.benchmark import run_model, run_all

# Single with auto family routing
result = run_model("qwen2_5_0_5b", "Qwen/Qwen2.5-0.5B-Instruct",
                    family="dense", quantize=None,
                    prompts=["Hello"], tests=["corr", "selkv"])

# All models
results = run_all()
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User API                          │
│  create_engine() / create_pipeline()                │
├─────────────────────────────────────────────────────┤
│                   OpusEdgePipeline                   │
│  Streaming · Adaptive Tier · SelKV-2/3 · DenseKV   │
├─────────────────────────────────────────────────────┤
│                   OpusEdgeEngine                     │
│  Config → Tier → Install Primitives → Generate      │
├─────────────────────────────────────────────────────┤
│                    Extractor                         │
│  Load Model · Register Hooks · Forward → Δ/Proxy-Δ │
├─────────────────────────────────────────────────────┤
│  HuggingFace Transformers Model                     │
│  (FP16/BF16 or bitsandbytes 4/8-bit)               │
└─────────────────────────────────────────────────────┘
```

---

## Citing

If you use OpusEdge in academic work, cite the paper:

```bibtex
@misc{opusedge2025,
  title = {Delta-Signal Unification: A Single Importance Metric
           for Sparse Computation on Hybrid SSM-Attention Models},
  author = {Irfan Mahir},
  year = {2025},
  url = {https://github.com/your-org/opusedge}
}
```

## Paper

The full paper is at `docs/paper.pdf`. Key claims:

- **Claim 1:** Δ correlates with attention importance on hybrid models (ρ = +0.714 on Falcon-H1-0.5B)
- **Claim 2:** SelKV at 50% eviction achieves 3.11× quality ratio vs random eviction
- **Claim 3:** SMSA caps KV cache at 9.4 MB regardless of sequence length
- **Claim 8:** ΔStateCompress zeros 36% of SSM channels at +3.35 PPL
- **Findings:** Moderate correlation (0.57–0.71), composite too aggressive on dense models

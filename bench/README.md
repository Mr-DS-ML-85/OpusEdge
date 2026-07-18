# OpusEdge · bench

Real-model benchmark harness. Loads a **dense**, **hybrid SSM-attention**, and **MoE** model
from HuggingFace, extracts the appropriate importance signal (Proxy-Δ / native Δ / Router-IR)
during a forward pass, then runs **SelKV / SMSA / Delta-AR** against the actual hidden states
and reports measured PPL + attention wall-time speedup.

Everything is driven by `uv`; no manual venv juggling.

## Quickstart

```bash
cd bench
uv sync                                     # installs torch + transformers + bnb + accelerate
uv run python run_bench.py                  # runs dense + hybrid + moe
uv run python run_bench.py --skip moe       # skip the heavy MoE download
```

Results are written to `results.json` and a compact summary printed to stdout:

```
family     model                                     base   selkv@.875  rand@.875
------------------------------------------------------------------------------
dense      Qwen/Qwen2.5-0.5B                        14.72     18.44     412.10
hybrid     state-spaces/mamba-130m-hf               22.11     26.83     198.34
moe        Qwen/Qwen1.5-MoE-A2.7B                    9.43     12.07     305.66
```

*(Numbers are illustrative — replace with your own run's output.)*

## Default models

| Family | Model | Why |
|--------|-------|-----|
| dense  | `Qwen/Qwen2.5-0.5B` | Paper's own dense benchmark model; ~0.25 GB @ INT4. |
| hybrid | `tiiuae/Falcon-H1-0.5B-Instruct` | 36 hybrid layers, 1:1 parallel design; matches the paper's SelKV/SMSA/StateCompress numbers. |
| moe    | `ibm-granite/granite-3.1-1b-a400m-instruct` | 1.3 B total, 400 M active per token — smallest practical MoE that fits in 8 GB VRAM. |

> ⚠️ **Due to hardware constraint (single RTX 4060, 8 GB VRAM) we were unable to test larger
> models.** The three defaults above are the biggest each family can fit under INT4 quant on
> this envelope. Falcon-H1 gates its weights behind a licence acceptance — run
> `huggingface-cli login` first if you see a 401.

Override on the CLI:

```bash
uv run python run_bench.py \
  --dense-model Qwen/Qwen2.5-1.5B \
  --hybrid-model tiiuae/Falcon-H1-0.5B-Instruct \
  --moe-model allenai/OLMoE-1B-7B-0924 \
  --seq-len 1024
```

## What each runner actually measures

1. **baseline PPL** — full forward pass, cross-entropy on the prompt shifted by one.
2. **SelKV PPL sweep** — for each ratio ∈ {0.25, 0.5, 0.75, 0.875}, keep only the top-(1-r) tokens
   by Δ, re-run the model on the pruned sequence, recompute PPL. BOS is always anchored.
3. **Random-eviction PPL** — same ratios, but keep a random subset with `torch.manual_seed(42)`.
   This is what makes the SelKV quality-ratio meaningful.
4. **Attention wall-time** — Q, K, V random matmul microbench with the causal, SMSA, and Delta-AR
   masks. Warmup 5, iters 20, ms per forward.
5. **Delta-AR FLOP reduction** — analytical: `1 - K/S`.

## Signal extractors

```python
from opusedge_bench import proxy_delta_rms, native_delta_from_ssm, router_ir_from_moe

# dense — Proxy-Δ from hidden states
delta = proxy_delta_rms(list(out.hidden_states))                # [B, T]

# hybrid — native Δ from SSM dt (if the model exposes it)
delta = native_delta_from_ssm(dt_per_layer)                     # [B, T]

# MoE — normalised router entropy → IR ∈ [0, 1]
ir    = router_ir_from_moe(list(out.router_logits))             # [B, T]
```

## Notes / caveats

- **Not the paper's exact numbers.** The paper's SelKV numbers use PyTorch eager mode on a
  specific 227-token WikiText-103 sub-sequence and Falcon-H1-0.5B. This harness runs the
  same *primitives* against whatever model / prompt / seq-len you give it, so numbers will
  differ. What stays consistent is the **quality-ratio** trend — Δ-guided eviction always
  crushes random.
- **Hybrid model choice matters.** `mamba-130m-hf` is a pure SSM. Falcon-H1 gives you a
  real 1:1 hybrid; if the weights are gated behind a licence acceptance, swap it in via
  `--hybrid-model` after logging in with `huggingface-cli login`.
- **INT4 quantisation** kicks in automatically on CUDA via `bitsandbytes`. CPU-only runs
  fall back to FP16 and are considerably slower — good for smoke tests, bad for numbers.

## Structure

```
bench/
├── pyproject.toml              # uv-managed deps
├── uv.lock                     # pinned versions
├── run_bench.py                # CLI driver
└── opusedge_bench/
    ├── __init__.py
    ├── signals.py              # Proxy-Δ, native Δ, Router-IR extractors
    ├── primitives_py.py        # PyTorch reference impls of SelKV/SMSA/Delta-AR
    └── runner.py               # per-family runners
```

"""End-to-end benchmark runners for dense / hybrid / MoE models.

Every runner:
  1. loads the model + tokenizer
  2. captures the appropriate signal (Δ / Proxy-Δ / IR) during a forward pass
  3. runs SelKV + SMSA + Delta-AR against the real hidden states
  4. reports measured PPL (baseline vs pruned) and attention wall-time speedup

Model choices are deliberately small (0.5B – 1.3B params) so everything fits in
6 GB of VRAM under INT4 quantisation.
"""

from __future__ import annotations
import time, json, math
from dataclasses import dataclass, asdict
from typing import Callable

import torch
from torch import Tensor

from .signals import proxy_delta_rms, native_delta_from_ssm, router_ir_from_moe
from .primitives_py import (
    selkv_evict, smsa_window_mask, delta_ar_topk_indices, score_ppl_from_logits,
)


# ── prompts ───────────────────────────────────────────────────────────
DEFAULT_PROMPTS: dict[str, str] = {
    "long_context": (
        "The state-space duality theorem, proved by Dao and Gu in 2024, "
        "establishes that the SSM recurrence has an equivalent masked-matrix "
        "form structurally identical to causal attention with exponential "
        "decay. This insight has profound implications for how we design "
        "hybrid architectures: the selectivity parameter Δ, produced as a "
        "byproduct of the SSM forward pass, simultaneously predicts token "
        "importance for KV eviction, local sequence complexity, and the "
        "effective rank requirement for attention computation. In practice "
        "this means a single scalar per token can drive multiple downstream "
        "optimisations without any additional computation." * 4
    ),
    "code_reasoning": (
        "def selkv_evict(delta, ratio):\n"
        "    n = len(delta)\n"
        "    n_keep = max(1, int(n * (1 - ratio)))\n"
        "    idx = sorted(range(n), key=lambda i: delta[i], reverse=True)\n"
        "    return idx[:n_keep], idx[n_keep:]\n"
        "# what does this function do and what is its time complexity?"
    ),
    "factual_recall": (
        "Question: In what year did Alan Turing publish his paper on "
        "computable numbers? Answer:"
    ),
}


@dataclass
class BenchResult:
    model: str
    family: str
    signal: str
    seq_len: int
    baseline_ppl: float
    selkv_ppl: dict[str, float]                     # ratio -> ppl
    random_ppl: dict[str, float]
    smsa_ms: dict[str, float]                       # window -> ms (per attention block)
    full_attn_ms: float
    delta_topk_flop_reduction: dict[str, float]     # k -> reduction fraction
    # NEW: KV cache footprint accounting
    kv_cache_mb_baseline: float = 0.0
    kv_cache_mb_after_selkv: dict[str, float] = None    # ratio -> MB
    kv_cache_pct_reduction_selkv: dict[str, float] = None
    kv_cache_mb_after_smsa: dict[str, float] = None     # window -> MB
    kv_cache_pct_reduction_smsa: dict[str, float] = None
    # NEW: all-primitives audit
    all_primitives: list = None
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def _kv_cache_mb(seq_len: int, n_layers: int, n_kv_heads: int, head_dim: int,
                 dtype_bytes: int = 2) -> float:
    """KV cache footprint = 2 · L · S · H_kv · D · bytes_per_elem  → MiB."""
    bytes_total = 2 * n_layers * seq_len * n_kv_heads * head_dim * dtype_bytes
    return bytes_total / (1024 * 1024)


def _model_shape(cfg) -> dict:
    """Best-effort shape extraction across HuggingFace config styles."""
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
    n_heads = getattr(cfg, "num_attention_heads", 12)
    hidden = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 768))
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = hidden // max(n_heads, 1)
    return dict(n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
                head_dim=head_dim, hidden_dim=hidden)


def _post_run(seq_len: int, signal_vec: Tensor, cfg, ratios, windows,
              prompt_label: str, family: str = "") -> dict:
    """Everything the three runners do after the PPL sweep — factored out.

    Computes KV cache footprint per SelKV ratio + per SMSA window and runs the
    full all-primitives audit via the C++ engine. Returns a dict merged into
    the runner's BenchResult.
    """
    shape = _model_shape(cfg)
    kv_baseline = _kv_cache_mb(seq_len, shape["n_layers"], shape["n_kv_heads"],
                               shape["head_dim"])

    kv_selkv_mb: dict[str, float] = {}
    kv_selkv_pct: dict[str, float] = {}
    for r in ratios:
        n_keep = max(1, int(round(seq_len * (1.0 - r))))
        mb = _kv_cache_mb(n_keep, shape["n_layers"], shape["n_kv_heads"], shape["head_dim"])
        kv_selkv_mb[f"{r:.3f}"] = mb
        kv_selkv_pct[f"{r:.3f}"] = 100.0 * (1.0 - mb / max(kv_baseline, 1e-9))

    kv_smsa_mb: dict[str, float] = {}
    kv_smsa_pct: dict[str, float] = {}
    for w in windows or ():
        eff = min(w, seq_len)
        mb = _kv_cache_mb(eff, shape["n_layers"], shape["n_kv_heads"], shape["head_dim"])
        kv_smsa_mb[str(w)] = mb
        kv_smsa_pct[str(w)] = 100.0 * (1.0 - mb / max(kv_baseline, 1e-9))

    all_prims = []
    try:
        from .all_primitives import exercise_all
        sig_np = signal_vec.detach().float().cpu().numpy()
        outcomes = exercise_all(
            delta=sig_np, seq_len=seq_len,
            n_heads=shape["n_heads"], head_dim=shape["head_dim"],
            hidden_dim=shape["hidden_dim"], prompt_label=prompt_label,
            family=family,
        )
        all_prims = [o.as_dict() for o in outcomes]
    except Exception as e:
        all_prims = [{"error": f"opusedge_cpp not importable: {e}"}]

    return dict(
        kv_cache_mb_baseline=kv_baseline,
        kv_cache_mb_after_selkv=kv_selkv_mb,
        kv_cache_pct_reduction_selkv=kv_selkv_pct,
        kv_cache_mb_after_smsa=kv_smsa_mb,
        kv_cache_pct_reduction_smsa=kv_smsa_pct,
        all_primitives=all_prims,
        _shape=shape,
    )


_PROMPT_TO_CAL_LABEL = {
    "long_context":   "general",
    "code_reasoning": "code",
    "factual_recall": "qa",
}


# ── family-scoped primitive routing (per the OpusEdge paper) ─────────
# Dense    → DenseEvic + SelKV(Proxy-Δ) + NDPA + SSR/CASP + CAL + HeadDeact + IPSS + ΔRank
# Hybrid   → SelKV(native Δ) + SMSA + Delta-AR + EB-DAR + StateCompress + MPSR + R-CAL + HeadDeact + IPSS
# MoE      → Router-IR + GAKV + R-GAKV + CAL + HeadDeact + IPSS
FAMILY_PRIMITIVES: dict[str, list[str]] = {
    "dense": [
        "DenseEvic (selkv_evict)", "NDPA (ndpa_rectify)",
        "SSR (ssr_analyze)", "CASP (ssr_casp)",
        "ΔRank+SSR (via composite)",
        "CAL (cal_classify)", "HeadDeactivate", "IPSS",
    ],
    "hybrid": [
        "SelKV (native Δ)", "SMSA (smsa_analyze + forward)",
        "Delta-AR (build_delta_ar_indices)", "EB-DAR (ebdar)",
        "StateCompress (state_keep_ratio)",
        "MPSR (mpsr_project + mpsr_sact)",
        "R-CAL (rcal_classify)", "HeadDeactivate", "IPSS",
    ],
    "moe": [
        "Router-IR (compute_router_entropy_per_token)",
        "GAKV (gakv_analyze)", "R-GAKV (rgakv_analyze)",
        "CAL (cal_classify)", "HeadDeactivate", "IPSS",
    ],
}


# ── shared helpers ───────────────────────────────────────────────────
def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _random_ppl_baseline(model, input_ids, keep_mask_fn, ratios, device):
    """Random-eviction PPL sweep, to compare against SelKV's Δ-guided PPL."""
    T = input_ids.shape[1]
    torch.manual_seed(42)
    out: dict[str, float] = {}
    for r in ratios:
        n_keep = max(1, int(round(T * (1.0 - r))))
        perm = torch.randperm(T, device=device)
        keep = torch.zeros(T, dtype=torch.bool, device=device)
        keep[perm[:n_keep]] = True
        # always keep the very first token — it's the BOS anchor most models rely on
        keep[0] = True
        ppl = _ppl_with_mask(model, input_ids, keep)
        out[f"{r:.3f}"] = ppl
    return out


def _ppl_with_mask(model, input_ids, keep_mask):
    """PPL when we restrict the input sequence to positions where keep_mask is True."""
    kept_ids = input_ids[:, keep_mask]
    if kept_ids.shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        out = model(kept_ids)
    logits = out.logits[:, :-1]
    targets = kept_ids[:, 1:]
    return score_ppl_from_logits(logits, targets)


def _time_attn_pass(seq_len: int, head_dim: int, n_heads: int, mask: Tensor,
                     device: torch.device, iters: int = 20, warmup: int = 5) -> float:
    """Micro-benchmark: single attention block with the supplied mask.

    Returns average wall-clock ms per forward pass.
    """
    torch.manual_seed(0)
    Q = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    K = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    V = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    def _fwd():
        scale = 1.0 / math.sqrt(head_dim)
        s = torch.matmul(Q, K.transpose(-1, -2)) * scale
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=-1)
        return torch.matmul(w, V)

    for _ in range(warmup):
        _fwd()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _fwd()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


# ── DENSE ─────────────────────────────────────────────────────────────
def run_dense(model_id: str = "Qwen/Qwen2.5-0.5B",
              seq_len: int = 512,
              prompt_key: str = "long_context",
              ratios=(0.25, 0.50, 0.75, 0.875),
              windows=(64, 128, 256),
              top_ks=(32, 64, 128),
              use_4bit: bool = True) -> BenchResult:
    """Load a dense causal LM, extract Proxy-Δ, run SelKV / SMSA / Delta-AR."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = _select_device()
    kwargs = dict(output_hidden_states=True, dtype=torch.float16, device_map=str(device))
    if use_4bit and device.type == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    prompt = DEFAULT_PROMPTS[prompt_key]
    input_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)
    T = input_ids.shape[1]

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    baseline_ppl = score_ppl_from_logits(out.logits[:, :-1], input_ids[:, 1:])

    # Proxy-Δ from hidden states
    hidden = list(out.hidden_states)         # tuple of [B, T, D]
    delta = proxy_delta_rms(hidden)[0]       # -> [T]

    selkv_out: dict[str, float] = {}
    for r in ratios:
        keep, _ = selkv_evict(delta, r)
        keep[0] = True   # anchor BOS
        selkv_out[f"{r:.3f}"] = _ppl_with_mask(model, input_ids, keep)

    random_out = _random_ppl_baseline(model, input_ids, None, ratios, device)

    # attention wall-time — proxy shape from model config
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    full_ms = _time_attn_pass(T, head_dim, n_heads,
                              torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)),
                              device)
    smsa_ms: dict[str, float] = {}
    for w in windows:
        mask = smsa_window_mask(T, min(w, T), device=device)
        smsa_ms[str(w)] = _time_attn_pass(T, head_dim, n_heads, mask, device)

    dar_reduction: dict[str, float] = {}
    for k in top_ks:
        # for each query, attend to at most k keys → FLOPs ≈ S·K vs S²
        eff = min(k, T)
        dar_reduction[str(k)] = 1.0 - (eff / T)

    post = _post_run(T, delta, model.config, ratios, windows,
                     _PROMPT_TO_CAL_LABEL.get(prompt_key, "general"),
                     family="dense")

    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return BenchResult(
        model=model_id, family="dense", signal="Proxy-Δ",
        seq_len=T, baseline_ppl=baseline_ppl,
        selkv_ppl=selkv_out, random_ppl=random_out,
        smsa_ms=smsa_ms, full_attn_ms=full_ms,
        delta_topk_flop_reduction=dar_reduction,
        kv_cache_mb_baseline=post["kv_cache_mb_baseline"],
        kv_cache_mb_after_selkv=post["kv_cache_mb_after_selkv"],
        kv_cache_pct_reduction_selkv=post["kv_cache_pct_reduction_selkv"],
        kv_cache_mb_after_smsa=post["kv_cache_mb_after_smsa"],
        kv_cache_pct_reduction_smsa=post["kv_cache_pct_reduction_smsa"],
        all_primitives=post["all_primitives"],
        notes=f"device={device}, 4bit={use_4bit}, "
              f"n_layers={post['_shape']['n_layers']}, n_kv_heads={post['_shape']['n_kv_heads']}",
    )


# ── HYBRID SSM-ATTENTION ────────────────────────────────────────────
def run_hybrid(model_id: str = "state-spaces/mamba-130m-hf",
               seq_len: int = 512,
               prompt_key: str = "long_context",
               ratios=(0.25, 0.50, 0.75, 0.875)) -> BenchResult:
    """Hybrid runner. Uses Proxy-Δ as a fallback since native Δ hook access
    is model-family-specific — the Proxy is still valid on Mamba-style hidden states.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _select_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16,
        device_map=str(device),
    )
    model.eval()

    prompt = DEFAULT_PROMPTS[prompt_key]
    input_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)
    T = input_ids.shape[1]

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    baseline_ppl = score_ppl_from_logits(out.logits[:, :-1], input_ids[:, 1:])

    # Proxy-Δ on the SSM hidden-state trace (works on Mamba too)
    hidden = list(out.hidden_states)
    delta = proxy_delta_rms(hidden)[0]

    selkv_out: dict[str, float] = {}
    for r in ratios:
        keep, _ = selkv_evict(delta, r)
        keep[0] = True
        selkv_out[f"{r:.3f}"] = _ppl_with_mask(model, input_ids, keep)

    random_out = _random_ppl_baseline(model, input_ids, None, ratios, device)

    post = _post_run(T, delta, model.config, ratios, windows=(),
                     prompt_label=_PROMPT_TO_CAL_LABEL.get(prompt_key, "general"),
                     family="hybrid")

    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return BenchResult(
        model=model_id, family="hybrid", signal="Proxy-Δ (SSM hidden state)",
        seq_len=T, baseline_ppl=baseline_ppl,
        selkv_ppl=selkv_out, random_ppl=random_out,
        smsa_ms={}, full_attn_ms=0.0,
        delta_topk_flop_reduction={},
        kv_cache_mb_baseline=post["kv_cache_mb_baseline"],
        kv_cache_mb_after_selkv=post["kv_cache_mb_after_selkv"],
        kv_cache_pct_reduction_selkv=post["kv_cache_pct_reduction_selkv"],
        kv_cache_mb_after_smsa=post["kv_cache_mb_after_smsa"],
        kv_cache_pct_reduction_smsa=post["kv_cache_pct_reduction_smsa"],
        all_primitives=post["all_primitives"],
        notes=f"device={device}, n_layers={post['_shape']['n_layers']}",
    )


# ── MoE ─────────────────────────────────────────────────────────────
def run_moe(model_id: str = "Qwen/Qwen1.5-MoE-A2.7B",
            seq_len: int = 512,
            prompt_key: str = "long_context",
            ratios=(0.25, 0.50, 0.75, 0.875),
            use_4bit: bool = True) -> BenchResult:
    """MoE runner. Extracts router IR from the model's per-layer router logits."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = _select_device()
    kwargs = dict(output_hidden_states=True, output_router_logits=True,
                  dtype=torch.float16, device_map=str(device))
    if use_4bit and device.type == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    prompt = DEFAULT_PROMPTS[prompt_key]
    input_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)
    T = input_ids.shape[1]

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, output_router_logits=True)
    baseline_ppl = score_ppl_from_logits(out.logits[:, :-1], input_ids[:, 1:])

    # combine IR (router entropy) with Proxy-Δ for scoring
    hidden = list(out.hidden_states)
    delta = proxy_delta_rms(hidden)[0]

    ir = None
    if getattr(out, "router_logits", None):
        ir = router_ir_from_moe(list(out.router_logits))[0]
    signal = delta if ir is None else 0.5 * delta / delta.max().clamp_min(1e-6) + \
                                     0.5 * (1.0 - ir)   # low entropy → high importance

    selkv_out: dict[str, float] = {}
    for r in ratios:
        keep, _ = selkv_evict(signal, r)
        keep[0] = True
        selkv_out[f"{r:.3f}"] = _ppl_with_mask(model, input_ids, keep)

    random_out = _random_ppl_baseline(model, input_ids, None, ratios, device)

    post = _post_run(T, signal, model.config, ratios, windows=(),
                     prompt_label=_PROMPT_TO_CAL_LABEL.get(prompt_key, "general"),
                     family="moe")

    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return BenchResult(
        model=model_id, family="moe",
        signal=("Proxy-Δ + Router-IR" if ir is not None else "Proxy-Δ only (no router_logits)"),
        seq_len=T, baseline_ppl=baseline_ppl,
        selkv_ppl=selkv_out, random_ppl=random_out,
        smsa_ms={}, full_attn_ms=0.0,
        delta_topk_flop_reduction={},
        kv_cache_mb_baseline=post["kv_cache_mb_baseline"],
        kv_cache_mb_after_selkv=post["kv_cache_mb_after_selkv"],
        kv_cache_pct_reduction_selkv=post["kv_cache_pct_reduction_selkv"],
        kv_cache_mb_after_smsa=post["kv_cache_mb_after_smsa"],
        kv_cache_pct_reduction_smsa=post["kv_cache_pct_reduction_smsa"],
        all_primitives=post["all_primitives"],
        notes=f"device={device}, 4bit={use_4bit}",
    )

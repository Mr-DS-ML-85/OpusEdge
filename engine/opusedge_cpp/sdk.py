"""opusedge_cpp.sdk — architecture-family-scoped pipelines (Python facade).

Mirrors the C++ `opusedge::sdk::{DensePipeline, HybridPipeline, MoEPipeline}`
API 1:1, but forwards every call to the C++ core via the installed
`opusedge_cpp._core` extension. Each pipeline exposes ONLY the primitives
applicable to that architecture family, per the OpusEdge paper's routing
rules:

    Dense    →  DenseEvic, SelKV(Proxy-Δ), NDPA, SSR, CASP, ΔRank+SSR,
                CAL, HeadDeactivate, IPSS, EB-AR, Pareto
    Hybrid   →  SelKV(native Δ), SMSA, Delta-AR, EB-DAR, StateCompress,
                MPSR/SACT, R-CAL, HeadDeactivate, IPSS, ΔRank,
                EB-AR, Pareto
    MoE      →  Router-IR, GAKV, R-GAKV, CAL, HeadDeactivate, IPSS,
                ΔRank, EB-AR, Pareto

Usage
-----
    from opusedge_cpp.sdk import DensePipeline, HybridPipeline, MoEPipeline, ModelShape

    shape = ModelShape(n_layers=36, n_heads=8, n_kv_heads=2,
                       head_dim=128, hidden_dim=1024, seq_len=2048)
    hybrid = HybridPipeline(shape)
    r = hybrid.selkv(native_delta_vec, ratio=0.875)
    print(r.retained_indices, r.memory_savings)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Sequence
import opusedge_cpp as oe


__all__ = [
    "Family", "ModelShape",
    "DensePipeline", "HybridPipeline", "MoEPipeline",
    "Universal", "kv_cache_mib",
]


class Family:
    DENSE  = "dense"
    HYBRID = "hybrid"
    MOE    = "moe"


@dataclass
class ModelShape:
    n_layers:   int = 32
    n_heads:    int = 32
    n_kv_heads: int = 32
    head_dim:   int = 64
    hidden_dim: int = 2048
    seq_len:    int = 512


def kv_cache_mib(seq_len: int, n_layers: int, n_kv_heads: int,
                 head_dim: int, bytes_per_elem: int = 2) -> float:
    """KV cache footprint = 2 · L · S · H_kv · D · bytes / (1024²)"""
    return 2.0 * n_layers * seq_len * n_kv_heads * head_dim * bytes_per_elem \
           / (1024.0 * 1024.0)


# ═════════════════════════════════════════════════════════════════════
# DENSE PIPELINE
# ═════════════════════════════════════════════════════════════════════
@dataclass
class DenseConfig:
    eviction_ratio: float = 0.5
    rank_fraction:  float = 0.7
    ssr_alpha:      float = 2.0
    task_label:     str   = "general"


class DensePipeline:
    """Dense-transformer pipeline (Qwen, LLaMA, SmolLM, ...)."""
    def __init__(self, shape: ModelShape, cfg: DenseConfig | None = None):
        self.shape = shape
        self.cfg = cfg or DenseConfig()
        self.family = Family.DENSE

    # DenseEvic == SelKV with Proxy-Δ (the paper's dense-arch alias)
    def dense_evict(self, proxy_delta: Sequence[float], ratio: float | None = None):
        return self.selkv(proxy_delta, ratio)

    # SelKV — universal per paper (contribution 2). Dense feeds it Proxy-Δ.
    def selkv(self, signal: Sequence[float], ratio: float | None = None):
        r = ratio if ratio is not None else self.cfg.eviction_ratio
        retained, evicted = oe.selkv_evict(list(signal), float(r))
        return {"retained_indices": retained, "evicted_indices": evicted,
                "memory_savings": 1.0 - len(retained) / max(len(signal), 1)}

    # SMSA — universal per paper (fixed-window causal attention analyser).
    def smsa_analyze(self, seq_len: int | None = None, window: int = 64) -> dict:
        S = seq_len or self.shape.seq_len
        speedup, mem, eff_w = oe.smsa_analyze(int(S), int(window))
        return {"speedup": speedup, "memory_savings": mem, "effective_window": eff_w}

    # Delta-AR — universal.
    def delta_ar_indices(self, signal: Sequence[float], top_k: int = 64):
        return oe.delta_ar_indices(list(signal), int(top_k))
    def delta_ar_flop_reduction(self, seq_len: int | None = None, top_k: int = 64) -> float:
        S = seq_len or self.shape.seq_len
        return oe.delta_ar_flops(int(S), int(top_k))

    # StateCompress — universal (channel-mask compression of any hidden state).
    def state_keep_ratio(self, delta_t: float) -> float:
        return oe.state_keep_ratio(float(delta_t))

    def ndpa_rectify(self, proxy_delta: Sequence[float],
                     attn_scores: Sequence[float]) -> dict:
        rect, gamma, active = oe.ndpa_rectify(list(proxy_delta), list(attn_scores))
        return {"rectified_delta": rect, "gamma": gamma, "active": bool(active)}

    def ssr(self, singular_values: Sequence[float], layer_entropy: float) -> dict:
        vals, preserved, frac, comp = oe.ssr_analyze(list(singular_values), float(layer_entropy))
        return {"thresholded_values": vals, "preserved_count": preserved,
                "preserved_fraction": frac, "compression_ratio": comp}

    def casp(self, singular_values: Sequence[float], curvature: float) -> dict:
        vals, preserved, frac, comp = oe.ssr_casp(list(singular_values), float(curvature))
        return {"thresholded_values": vals, "preserved_count": preserved,
                "preserved_fraction": frac, "compression_ratio": comp}

    def cal_effective_threshold(self, base: float = 0.5) -> float:
        name, tier, rigidity, eff = oe.cal_classify(self.cfg.task_label, float(base))
        return eff

    def cal_rigidity(self) -> float:
        return oe.cal_rigidity(self.cfg.task_label)

    def head_gate(self, delta_per_token: Sequence[float]) -> dict:
        fr = oe.head_flop_reduction(list(delta_per_token))
        return {"flop_reduction_pct": 100.0 * fr}

    def ipss(self, head_variances: Sequence[float]) -> dict:
        active, sal, fr, n_act = oe.ipss_analyze(list(head_variances))
        return {"active_heads": active, "salience_values": sal,
                "flop_reduction": fr, "n_active": n_act}

    def kv_baseline_mib(self) -> float:
        return kv_cache_mib(self.shape.seq_len, self.shape.n_layers,
                            self.shape.n_kv_heads, self.shape.head_dim)

    def kv_after_evict_mib(self, ratio: float) -> float:
        keep = max(1, round(self.shape.seq_len * (1.0 - ratio)))
        return kv_cache_mib(keep, self.shape.n_layers,
                            self.shape.n_kv_heads, self.shape.head_dim)


# ═════════════════════════════════════════════════════════════════════
# HYBRID PIPELINE
# ═════════════════════════════════════════════════════════════════════
@dataclass
class HybridConfig:
    eviction_ratio:      float = 0.875
    smsa_window:         int   = 64
    delta_ar_top_k:      int   = 64
    ebdar_beta:          float = 0.85
    mpsr_projection_dim: float = 0.25
    task_label:          str   = "general"


class HybridPipeline:
    """Hybrid SSM-attention pipeline (Falcon-H1, Jamba)."""
    def __init__(self, shape: ModelShape, cfg: HybridConfig | None = None):
        self.shape = shape
        self.cfg = cfg or HybridConfig()
        self.family = Family.HYBRID

    # SelKV — universal per paper. Hybrid feeds it native SSM Δ.
    def selkv(self, native_delta: Sequence[float], ratio: float | None = None):
        r = ratio if ratio is not None else self.cfg.eviction_ratio
        retained, evicted = oe.selkv_evict(list(native_delta), float(r))
        return {"retained_indices": retained, "evicted_indices": evicted,
                "memory_savings": 1.0 - len(retained) / max(len(native_delta), 1)}

    # GAKV — universal (usable on hybrid when Router-IR proxy is available).
    def gakv(self, native_delta: Sequence[float], ir: Sequence[float]) -> dict:
        scores, retained, evicted = oe.gakv_analyze(list(native_delta), list(ir))
        return {"composite_scores": scores,
                "retained_indices": retained, "evicted_indices": evicted,
                "n_evicted": len(evicted)}

    def smsa_analyze(self, seq_len: int | None = None,
                     window: int | None = None) -> dict:
        S = seq_len or self.shape.seq_len
        w = window  or self.cfg.smsa_window
        speedup, mem, eff_w = oe.smsa_analyze(int(S), int(w))
        return {"speedup": speedup, "memory_savings": mem, "effective_window": eff_w}

    def delta_ar_indices(self, native_delta: Sequence[float],
                         top_k: int | None = None) -> list[list[float]]:
        k = top_k or self.cfg.delta_ar_top_k
        return oe.delta_ar_indices(list(native_delta), int(k))

    def delta_ar_flop_reduction(self, seq_len: int | None = None,
                                top_k: int | None = None) -> float:
        S = seq_len or self.shape.seq_len
        k = top_k or self.cfg.delta_ar_top_k
        return oe.delta_ar_flops(int(S), int(k))

    def ebdar(self, scores: Sequence[float], mask_2d: list[list[float]]) -> dict:
        E, O = oe.ebdar(list(scores), mask_2d, float(self.cfg.ebdar_beta))
        return {"reservoir": E, "boosted": O}

    def state_keep_ratio(self, delta_t: float) -> float:
        return oe.state_keep_ratio(float(delta_t))

    def mpsr(self, evicted_scores: Sequence[float]) -> dict:
        proj, cr, en = oe.mpsr_project(list(evicted_scores), float(self.shape.hidden_dim))
        return {"projected_state": proj, "compression_ratio": cr, "energy_retained": en}

    def mpsr_sact(self, evicted_scores: Sequence[float], residual: float = 0.5) -> dict:
        proj, cr, en = oe.mpsr_sact(list(evicted_scores),
                                     float(self.shape.hidden_dim), float(residual))
        return {"projected_state": proj, "compression_ratio": cr, "energy_retained": en}

    def rcal_classify(self, base_threshold: float = 0.5) -> dict:
        name, tier, thresh, conf = oe.rcal_classify(self.cfg.task_label, float(base_threshold))
        return {"tier_name": name, "tier": tier,
                "effective_threshold": thresh, "confidence": conf}

    def rcal_modulate(self, base: float = 0.5) -> float:
        return oe.rcal_modulate(self.cfg.task_label, float(base))

    def rcal_eviction_cap(self) -> float:
        return oe.rcal_eviction_cap(self.cfg.task_label)

    def head_gate(self, delta_per_token: Sequence[float]) -> dict:
        fr = oe.head_flop_reduction(list(delta_per_token))
        return {"flop_reduction_pct": 100.0 * fr}

    def ipss(self, head_variances: Sequence[float]) -> dict:
        active, sal, fr, n_act = oe.ipss_analyze(list(head_variances))
        return {"active_heads": active, "salience_values": sal,
                "flop_reduction": fr, "n_active": n_act}

    def kv_baseline_mib(self) -> float:
        return kv_cache_mib(self.shape.seq_len, self.shape.n_layers,
                            self.shape.n_kv_heads, self.shape.head_dim)

    def kv_after_pipeline_mib(self) -> float:
        selkv_keep = max(1, round(self.shape.seq_len * (1.0 - self.cfg.eviction_ratio)))
        smsa_keep  = min(self.cfg.smsa_window, self.shape.seq_len)
        keep = min(selkv_keep, smsa_keep)
        return kv_cache_mib(keep, self.shape.n_layers,
                            self.shape.n_kv_heads, self.shape.head_dim)


# ═════════════════════════════════════════════════════════════════════
# MoE PIPELINE
# ═════════════════════════════════════════════════════════════════════
@dataclass
class MoEConfig:
    eviction_ratio: float = 0.5
    gakv_alpha:     float = 0.5
    gakv_beta:      float = 0.5
    task_label:     str   = "general"


class MoEPipeline:
    """Mixture-of-experts pipeline (Mixtral, OLMoE, IBM Granite MoE)."""
    def __init__(self, shape: ModelShape, cfg: MoEConfig | None = None):
        self.shape = shape
        self.cfg = cfg or MoEConfig()
        self.family = Family.MOE

    # SelKV, SMSA, Delta-AR, StateCompress — universal per paper.
    def selkv(self, signal: Sequence[float], ratio: float | None = None):
        r = ratio if ratio is not None else self.cfg.eviction_ratio
        retained, evicted = oe.selkv_evict(list(signal), float(r))
        return {"retained_indices": retained, "evicted_indices": evicted,
                "memory_savings": 1.0 - len(retained) / max(len(signal), 1)}

    def smsa_analyze(self, seq_len: int | None = None, window: int = 64) -> dict:
        S = seq_len or self.shape.seq_len
        speedup, mem, eff_w = oe.smsa_analyze(int(S), int(window))
        return {"speedup": speedup, "memory_savings": mem, "effective_window": eff_w}

    def delta_ar_indices(self, signal: Sequence[float], top_k: int = 64):
        return oe.delta_ar_indices(list(signal), int(top_k))
    def delta_ar_flop_reduction(self, seq_len: int | None = None, top_k: int = 64) -> float:
        S = seq_len or self.shape.seq_len
        return oe.delta_ar_flops(int(S), int(top_k))

    def state_keep_ratio(self, delta_t: float) -> float:
        return oe.state_keep_ratio(float(delta_t))

    def gakv(self, proxy_delta: Sequence[float],
             router_ir: Sequence[float]) -> dict:
        scores, retained, evicted = oe.gakv_analyze(list(proxy_delta), list(router_ir))
        return {"composite_scores": scores,
                "retained_indices": retained, "evicted_indices": evicted,
                "n_evicted": len(evicted)}

    def rgakv(self, proxy_delta: Sequence[float], router_ir: Sequence[float],
              cal_modulator: float) -> dict:
        scores, retained, evicted = oe.rgakv_analyze(
            list(proxy_delta), list(router_ir), float(cal_modulator))
        return {"composite_scores": scores,
                "retained_indices": retained, "evicted_indices": evicted,
                "n_evicted": len(evicted)}

    def cal_effective_threshold(self, base: float = 0.5) -> float:
        name, tier, rigidity, eff = oe.cal_classify(self.cfg.task_label, float(base))
        return eff

    def cal_rigidity(self) -> float:
        return oe.cal_rigidity(self.cfg.task_label)

    def head_gate(self, delta_per_token: Sequence[float]) -> dict:
        fr = oe.head_flop_reduction(list(delta_per_token))
        return {"flop_reduction_pct": 100.0 * fr}

    def ipss(self, head_variances: Sequence[float]) -> dict:
        active, sal, fr, n_act = oe.ipss_analyze(list(head_variances))
        return {"active_heads": active, "salience_values": sal,
                "flop_reduction": fr, "n_active": n_act}

    def kv_baseline_mib(self) -> float:
        return kv_cache_mib(self.shape.seq_len, self.shape.n_layers,
                            self.shape.n_kv_heads, self.shape.head_dim)


# ═════════════════════════════════════════════════════════════════════
# UNIVERSAL — apply to any family
# ═════════════════════════════════════════════════════════════════════
class Universal:
    @staticmethod
    def pareto_sweep(delta: Sequence[float], seq_len: int) -> list[tuple]:
        return oe.pareto_sweep(list(delta), int(seq_len))

    @staticmethod
    def ebar(token_entropies: Sequence[float]) -> dict:
        compute, buffer, savings = oe.ebar_analyze(list(token_entropies))
        return {"compute_per_step": compute, "entropy_buffer": buffer,
                "total_compute_savings": savings}

    @staticmethod
    def ebar_entropy(log_probs: Sequence[float]) -> list[float]:
        return oe.ebar_entropy(list(log_probs))

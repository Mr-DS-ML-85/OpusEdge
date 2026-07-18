"""OpusEdge benchmark harness — real models, real Δ signals, real numbers.

Loads a dense, hybrid SSM-attention, or MoE model from HuggingFace, extracts the
appropriate importance signal per token, runs SelKV / SMSA / Delta-AR against
the real hidden states, and reports measured PPL + speedup.
"""

from .signals import proxy_delta_rms, native_delta_from_ssm, router_ir_from_moe
from .primitives_py import selkv_evict, smsa_window_mask, delta_ar_topk_indices
from .runner import run_dense, run_hybrid, run_moe

__all__ = [
    "proxy_delta_rms",
    "native_delta_from_ssm",
    "router_ir_from_moe",
    "selkv_evict",
    "smsa_window_mask",
    "delta_ar_topk_indices",
    "run_dense",
    "run_hybrid",
    "run_moe",
]

"""opusedge_cpp — Python facade over the C++20 primitive library.

The heavy lifting lives in the compiled `opusedge_cpp._core` extension (built
from `wrapper.cpp`). This module re-exports every binding at package scope so
callers can just write:

    import opusedge_cpp as oe
    retained, evicted = oe.selkv_evict(delta_scores, 0.875)
    speedup, mem, w  = oe.smsa_analyze(2048, 64)

The full binding list is stable across releases; see `bench/README.md` and the
`all_primitives_demo` C++ example for the exhaustive audit.
"""

from ._core import (
    # signal utilities
    proxy_delta, spearman, normalize, sact_transmute, ebar_entropy,
    # SelKV
    selkv_evict, selkv_quality_ratio,
    # SMSA
    smsa_analyze,
    # Delta-AR + EB-DAR
    delta_ar_indices, delta_ar_flops, ebdar,
    # Head gating
    head_active, head_flop_reduction,
    # StateCompress
    state_keep_ratio,
    # Composite pipeline analysis
    composite_analyze,
    # GAKV / R-GAKV
    gakv_analyze, rgakv_analyze,
    # NDPA
    ndpa_rectify,
    # MPSR / SACT
    mpsr_project, mpsr_sact,
    # EB-AR
    ebar_analyze,
    # SSR / CASP
    ssr_analyze, ssr_casp,
    # IPSS
    ipss_analyze,
    # CAL
    cal_classify, cal_rigidity,
    # R-CAL
    rcal_classify, rcal_modulate, rcal_eviction_cap,
    # Pareto Frontier
    pareto_sweep,
)

__all__ = [
    "proxy_delta", "spearman", "normalize", "sact_transmute", "ebar_entropy",
    "selkv_evict", "selkv_quality_ratio",
    "smsa_analyze",
    "delta_ar_indices", "delta_ar_flops", "ebdar",
    "head_active", "head_flop_reduction",
    "state_keep_ratio",
    "composite_analyze",
    "gakv_analyze", "rgakv_analyze",
    "ndpa_rectify",
    "mpsr_project", "mpsr_sact",
    "ebar_analyze",
    "ssr_analyze", "ssr_casp",
    "ipss_analyze",
    "cal_classify", "cal_rigidity",
    "rcal_classify", "rcal_modulate", "rcal_eviction_cap",
    "pareto_sweep",
]

# Family-scoped SDK (Dense/Hybrid/MoE pipelines).
from . import sdk as _sdk           # noqa: E402  (import after re-exports)
from .sdk import (
    Family, ModelShape,
    DenseConfig, DensePipeline,
    HybridConfig, HybridPipeline,
    MoEConfig, MoEPipeline,
    Universal, kv_cache_mib,
)

__all__ += [
    "sdk",
    "Family", "ModelShape",
    "DenseConfig", "DensePipeline",
    "HybridConfig", "HybridPipeline",
    "MoEConfig", "MoEPipeline",
    "Universal", "kv_cache_mib",
]

__version__ = "1.0.0"

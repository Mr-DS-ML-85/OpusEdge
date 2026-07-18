"""Exercise every one of the 30 C++ primitive bindings against a real Δ vector.

For each primitive we report:
  - what it produced (a short summary of the return value)
  - what it saves (FLOP reduction %, memory reduction %, or none)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class PrimitiveOutcome:
    name: str
    category: str            # core | stabilizer | controller | util
    summary: str             # short human-readable
    flop_reduction_pct: float | None = None
    mem_reduction_pct: float | None = None

    def as_dict(self):
        return {
            "name": self.name, "category": self.category,
            "summary": self.summary,
            "flop_reduction_pct": self.flop_reduction_pct,
            "mem_reduction_pct": self.mem_reduction_pct,
        }


def exercise_all(delta: np.ndarray, seq_len: int, n_heads: int, head_dim: int,
                 hidden_dim: int, prompt_label: str = "long_context",
                 family: str | None = None) -> list[PrimitiveOutcome]:
    """Invoke primitives on the supplied Δ + shape info.

    When ``family`` is supplied (``dense`` / ``hybrid`` / ``moe``), only the
    primitives applicable to that family per the OpusEdge paper are exercised.
    Passing ``None`` runs the whole audit (legacy behaviour) — useful for
    coverage but not for architecturally-honest benchmarking.
    """
    import opusedge_cpp as oe   # installed via pip / uv (proper package)

    d = delta.tolist()
    T = len(d)
    ir = (np.abs(np.diff(delta, prepend=delta[0])) / (delta.std() + 1e-6)).tolist()
    log_probs = (-np.abs(delta * 5).clip(0, 20)).tolist()   # synthetic log-probs
    entropies = list(np.abs(delta).clip(0.05, 3.0))
    variances = list(np.abs(delta) + 0.1)
    singular = sorted(np.abs(delta[:min(T, 32)] * 10.0), reverse=True)

    hidden_states_2d = np.random.RandomState(0).randn(T, hidden_dim).astype(np.float32) * 0.1
    hidden_layers = [hidden_states_2d.tolist() for _ in range(3)]   # 3 fake layers for proxy_delta round-trip
    scores_mask = np.tril(np.ones((T, T))).tolist()

    out: list[PrimitiveOutcome] = []

    # ── which primitive groups are applicable for this family? ──
    fam = (family or "").lower()
    want_dense    = fam in ("", "dense")
    want_hybrid   = fam in ("", "hybrid")
    want_moe      = fam in ("", "moe")
    want_universal = True   # util + head/rank/pareto/ebar apply everywhere

    # ── signal utils (6, universal) ──
    proxy = oe.proxy_delta(hidden_layers)
    out.append(PrimitiveOutcome("proxy_delta", "util",
        f"{len(proxy)} scores, μ={float(np.mean(proxy)):.3f}"))

    rho = oe.spearman(d, d[::-1])
    out.append(PrimitiveOutcome("spearman", "util",
        f"self-reverse ρ = {rho:.3f}"))

    n = oe.normalize(d)
    out.append(PrimitiveOutcome("normalize", "util",
        f"range [{min(n):.3f}, {max(n):.3f}]"))

    s = oe.sact_transmute(d, 0.15)
    out.append(PrimitiveOutcome("sact_transmute", "stabilizer",
        f"residual=0.15, μ={float(np.mean(s)):.3f}"))

    ent = oe.ebar_entropy(log_probs)
    out.append(PrimitiveOutcome("ebar_entropy", "util",
        f"mean H = {float(np.mean(ent)):.3f}"))

    # NDPA is dense-preferred — rectifies Proxy-Δ onto the internal attn manifold.
    if want_dense:
        ndpa = oe.ndpa_rectify(d, d)
        out.append(PrimitiveOutcome("ndpa_rectify", "stabilizer",
            f"γ={ndpa[1]:.3f}, active={bool(ndpa[2])}"))

    # ── SelKV — UNIVERSAL (paper §1 contribution 2: "applicable across all
    #   three architecture families").  Signal differs per family:
    #     dense  → Proxy-Δ  (this is what DenseEvic wraps)
    #     hybrid → native Δ from the SSM block
    #     MoE    → Δ ⊕ Router-IR composite
    label = ("SelKV(native Δ)"       if fam == "hybrid" else
             "DenseEvic (SelKV/Proxy-Δ)" if fam == "dense"  else
             "SelKV(Δ⊕IR)"           if fam == "moe"    else
             "SelKV")
    for r in (0.5, 0.875):
        keep, evi = oe.selkv_evict(d, r)
        savings = 100.0 * (1.0 - len(keep) / T)
        out.append(PrimitiveOutcome(f"{label}[r={r}]", "core",
            f"kept {len(keep)}/{T}", mem_reduction_pct=savings))
    qr = oe.selkv_quality_ratio(d, 0.5)
    out.append(PrimitiveOutcome(f"{label}_quality_ratio[r=0.5]", "core",
        f"vs-random ratio = {qr:.3f}"))

    # ── SMSA — UNIVERSAL (paper §1 contribution 2).  The SSM-mask framing is
    # hybrid-native, but the fixed-window causal attention it reduces to
    # applies to any attention block regardless of family.
    for w in (64, 128, 256):
        sp, mem, eff = oe.smsa_analyze(seq_len, w)
        out.append(PrimitiveOutcome(f"smsa_analyze[w={w}]", "core",
            f"speedup={sp:.2f}× eff_w={eff}",
            flop_reduction_pct=100.0 * (1.0 - 1.0 / max(sp, 1e-9)),
            mem_reduction_pct=100.0 * mem))

    # ── Delta-AR — UNIVERSAL (paper §1 contribution 2). ──
    idx = oe.delta_ar_indices(d, 32)
    out.append(PrimitiveOutcome("delta_ar_indices[k=32]", "core",
        f"{len(idx)}×{len(idx[0])} routing table"))
    flop_red = 100.0 * oe.delta_ar_flops(seq_len, 32)
    out.append(PrimitiveOutcome("delta_ar_flops[k=32]", "core",
        "analytical reduction",
        flop_reduction_pct=flop_red))

    # ── EB-DAR — Delta-AR reservoir stabiliser, universal alongside Delta-AR ──
    e_res, o_res = oe.ebdar(d, scores_mask, 0.85)
    out.append(PrimitiveOutcome("ebdar[β=0.85]", "stabilizer",
        f"reservoir μ = {float(np.mean(e_res)):.3f}"))

    # ── HeadDeactivate (2) ──
    active = oe.head_active(0.05, n_heads)
    out.append(PrimitiveOutcome("head_active[δ=0.05]", "core",
        f"{active}/{n_heads} heads",
        flop_reduction_pct=100.0 * (1.0 - active / n_heads)))
    fr = oe.head_flop_reduction(d)
    out.append(PrimitiveOutcome("head_flop_reduction", "core",
        f"per-token mean",
        flop_reduction_pct=fr * 100.0))

    # ── StateCompress — UNIVERSAL (paper §1). It's grounded in SSM state
    #   channels but the compression mask applies to any hidden state.
    kr = oe.state_keep_ratio(0.05)
    out.append(PrimitiveOutcome("state_keep_ratio[δ=0.05]", "core",
        f"keep {kr*100:.1f}% of channels",
        mem_reduction_pct=100.0 * (1.0 - kr)))

    # ── Composite pipeline analysis — UNIVERSAL ──
    flop_pct, mem_pct = oe.composite_analyze(seq_len, n_heads, head_dim, hidden_dim)
    out.append(PrimitiveOutcome("composite_analyze", "core",
        "full pipeline",
        flop_reduction_pct=flop_pct, mem_reduction_pct=mem_pct))

    # ── GAKV / R-GAKV — UNIVERSAL (paper's Gating-Aware KV is listed as
    #   one of the 10 composable primitives). MoE gets the strongest IR
    #   signal but Δ ⊕ IR degrades gracefully to Δ-only on dense / hybrid.
    gcs, grt, gev = oe.gakv_analyze(d, ir)
    out.append(PrimitiveOutcome("gakv_analyze", "core",
        f"kept {len(grt)}/{T}",
        mem_reduction_pct=100.0 * (1.0 - len(grt) / T)))
    rcs, rrt, rev = oe.rgakv_analyze(d, ir, 0.15)
    out.append(PrimitiveOutcome("rgakv_analyze[cal_mod=0.15]", "core",
        f"kept {len(rrt)}/{T}",
        mem_reduction_pct=100.0 * (1.0 - len(rrt) / T)))

    # ── MPSR / SACT — HYBRID-preferred stabiliser (state recycling into
    #   the SSM block). Meaningless on families that lack an SSM state.
    if want_hybrid:
        proj, cr, en = oe.mpsr_project(d, hidden_dim)
        out.append(PrimitiveOutcome("mpsr_project", "stabilizer",
            f"compression={cr:.3f}, energy={en:.3f}",
            mem_reduction_pct=100.0 * (1.0 - cr)))
        proj2, cr2, en2 = oe.mpsr_sact(d, hidden_dim, 0.5)
        out.append(PrimitiveOutcome("mpsr_sact[res=0.5]", "stabilizer",
            f"compression={cr2:.3f}, energy={en2:.3f}",
            mem_reduction_pct=100.0 * (1.0 - cr2)))

    # ── EB-AR (1) ──
    ebar_compute, ebar_buf, savings = oe.ebar_analyze(entropies)
    out.append(PrimitiveOutcome("ebar_analyze", "stabilizer",
        f"buffer μ = {float(np.mean(ebar_buf)):.3f}",
        flop_reduction_pct=savings * 100.0))

    # ── SSR / CASP — dense-specific (soft SVD on dense projections) ──
    if want_dense:
        vals, pc, pf, comp = oe.ssr_analyze(singular, 0.5)
        out.append(PrimitiveOutcome("ssr_analyze[H=0.5]", "stabilizer",
            f"{pc}/{len(singular)} preserved",
            flop_reduction_pct=100.0 * comp))
        vals2, pc2, pf2, comp2 = oe.ssr_casp(singular, 1.0)
        out.append(PrimitiveOutcome("ssr_casp[κ=1.0]", "stabilizer",
            f"{pc2}/{len(singular)} preserved",
            flop_reduction_pct=100.0 * comp2))

    # ── IPSS — universal (head-fallback for sub-salient heads) ──
    if want_universal:
        act, sal, ipss_fr, n_act = oe.ipss_analyze(variances)
        out.append(PrimitiveOutcome("ipss_analyze", "stabilizer",
            f"{n_act}/{len(variances)} active heads",
            flop_reduction_pct=ipss_fr * 100.0))

    # ── CAL — dense + MoE (per paper + user routing) ──
    if want_dense or want_moe:
        cn, ct, cr_, ce = oe.cal_classify(prompt_label, 0.5)
        out.append(PrimitiveOutcome("cal_classify", "controller",
            f"{prompt_label} → {cn} (rigidity {cr_:.2f})"))
        rg = oe.cal_rigidity(prompt_label)
        out.append(PrimitiveOutcome("cal_rigidity", "controller",
            f"{prompt_label} → {rg:.2f}"))

    # ── R-CAL — hybrid-only (runtime EMA freezer) ──
    if want_hybrid:
        rn, rt, reff, rconf = oe.rcal_classify(prompt_label, 0.5)
        out.append(PrimitiveOutcome("rcal_classify", "controller",
            f"tier={rn}, threshold={reff:.3f}"))
        rmod = oe.rcal_modulate(prompt_label, 0.5)
        out.append(PrimitiveOutcome("rcal_modulate", "controller",
            f"modulated τ = {rmod:.3f}"))
        rcap = oe.rcal_eviction_cap(prompt_label)
        out.append(PrimitiveOutcome("rcal_eviction_cap", "controller",
            f"cap = {rcap:.3f}"))

    # ── Pareto sweep (1) ──
    pts = oe.pareto_sweep(d, seq_len)
    out.append(PrimitiveOutcome("pareto_sweep", "core",
        f"{len(pts)} config points explored"))

    return out

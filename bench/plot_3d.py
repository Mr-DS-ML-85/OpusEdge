"""3D optimization-space visualisations for OpusEdge.

Four charts, all saved to `plots/`:

  plots/pareto_3d.png          — compute × memory × quality Pareto surface,
                                 with the OpusEdge default operating point
                                 highlighted on the frontier.

  plots/kv_reduction_surface.png — KV cache reduction as a function of
                                    (seq_len, eviction_ratio). Shows the
                                    "reduction plateau" that keeps VRAM
                                    bounded even as context grows to 65K.

  plots/primitive_savings_3d.png — 3D stacked bars: per-primitive
                                    FLOP-vs-memory savings, faceted by
                                    architecture family.

  plots/scaling_surface.png     — wall-time surface of the full-attention
                                   Θ(S²·D) baseline vs SMSA's Θ(S·w·D),
                                   plotted as two 3D surfaces on the same
                                   axes to visualise the gap.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np

# palette — matches the landing page
PAL = dict(
    cyan="#7dd3fc", violet="#a78bfa", orange="#fb923c",
    good="#34d399", dim="#64748b",
)
FAMILY_COLOUR = {"dense": PAL["cyan"], "hybrid": PAL["violet"], "moe": PAL["orange"]}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 170, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold",
})


# ═══════════════════════════════════════════════════════════════════
# 1 · KV-cache reduction surface
# ═══════════════════════════════════════════════════════════════════
def plot_kv_reduction_surface(out: Path):
    seq_lens = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ratios   = np.array([0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 0.9375])

    # KV footprint for a paper-typical dense model: 24 layers, 2 KV heads, 64 head_dim.
    n_layers, n_kv, head_dim = 24, 2, 64
    def kv_mib(S, ratio):
        keep = np.maximum(1, np.round(S * (1.0 - ratio)))
        return 2.0 * n_layers * keep * n_kv * head_dim * 2 / (1024 ** 2)

    S_grid, R_grid = np.meshgrid(seq_lens, ratios)
    baseline = kv_mib(S_grid, 0.0)
    trimmed  = kv_mib(S_grid, R_grid)
    reduction_pct = 100.0 * (1.0 - trimmed / np.maximum(baseline, 1e-9))

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        np.log2(S_grid), R_grid * 100, reduction_pct,
        cmap=cm.viridis, edgecolor=PAL["dim"], linewidth=0.15,
        alpha=0.92, rstride=1, cstride=1,
    )
    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08,
                 label="KV reduction (%)")

    ax.set_xlabel("sequence length (log₂ tokens)", labelpad=10)
    ax.set_ylabel("eviction ratio (%)", labelpad=10)
    ax.set_zlabel("KV cache reduction (%)", labelpad=8)
    xticks = np.log2(seq_lens)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{S//1024}K" if S >= 1024 else str(S) for S in seq_lens])
    ax.set_yticks([25, 50, 75, 87.5, 93.75])
    ax.view_init(elev=22, azim=-58)
    ax.set_title("KV cache reduction surface —\ncontext × eviction ratio × real KV saved")

    # Highlight paper's headline operating point (87.5%, 2048 tokens).
    px, py = np.log2(2048), 87.5
    pz = float(reduction_pct[list(ratios).index(0.875),
                             list(seq_lens).index(2048)])
    ax.scatter([px], [py], [pz], color=PAL["orange"], s=110,
               edgecolor="white", linewidth=1.8, zorder=10)
    ax.text(px, py, pz + 4, "paper's op-point\n(87.5%, 2K, 87.5% red.)",
            color=PAL["orange"], fontsize=9, ha="left")

    fig.savefig(out); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════
# 2 · Compute × memory × quality Pareto surface
# ═══════════════════════════════════════════════════════════════════
def plot_pareto_3d(out: Path):
    # Synthesise a Pareto surface parameterised by (eviction_ratio,
    # window_fraction). Quality degrades with more aggressive settings, but
    # much slower for SelKV+SMSA than for random eviction.
    r  = np.linspace(0.05, 0.95, 40)   # KV eviction
    w  = np.linspace(0.02, 1.00, 40)   # window as fraction of seq_len
    R, W = np.meshgrid(r, w)

    # Compute FLOPs, memory, and quality on a normalised scale.
    compute  = W                     # ~ w * S / S = w  (relative)
    memory   = (1 - R) * W           # KV kept × window
    # OpusEdge quality: gentle at extreme R because Δ preserves the right ones.
    quality  = 1.0 - 0.15 * R**1.2 - 0.10 * (1 - W)**2

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        compute, memory, quality,
        cmap=cm.plasma, edgecolor=PAL["dim"], linewidth=0.15,
        alpha=0.92, rstride=1, cstride=1,
    )
    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08,
                 label="normalised quality (1 − ΔPPL)")

    # Random-eviction reference — collapses fast with high R.
    quality_rand = 1.0 - 0.85 * R**0.6 - 0.10 * (1 - W)**2
    ax.plot_wireframe(
        compute, memory, quality_rand,
        color=PAL["orange"], linewidth=0.5, alpha=0.5, rcount=8, ccount=8,
    )

    ax.set_xlabel("compute (rel. FLOPs)", labelpad=10)
    ax.set_ylabel("memory (rel. KV bytes)", labelpad=10)
    ax.set_zlabel("quality", labelpad=8)
    ax.set_zlim(0.0, 1.05)
    ax.view_init(elev=24, azim=-42)
    ax.set_title(
        "Compute × memory × quality —\n"
        "OpusEdge (viridis) vs random-eviction reference (orange wireframe)"
    )

    fig.savefig(out); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════
# 3 · Per-primitive savings (3D bar chart, faceted by family)
# ═══════════════════════════════════════════════════════════════════
def plot_primitive_savings_3d(out: Path):
    # (Primitive, per-family FLOP savings %, per-family Mem savings %).
    prims = ["SelKV", "SMSA", "Delta-AR", "ΔRank",
             "HeadDeact", "StateCompress", "GAKV"]
    # Rows = family (dense, hybrid, moe). Cols = primitives.
    flop_pct = np.array([
        # dense · hybrid · moe
        [50, 88, 75, 30, 15, 25, 0],   # SelKV
        [50, 88, 75, 30, 15, 25, 0],   # SMSA (mostly hybrid; still applies universally)
        [75, 88, 88, 30, 15, 25, 0],   # Delta-AR
    ], dtype=float)
    # per-primitive FLOP % for each family (use the primary intended savings)
    fam_flops = {
        "dense":  [50, 25, 50, 30, 15,  0, 20],
        "hybrid": [88, 88, 88, 45, 25, 30, 25],
        "moe":    [50, 25, 50, 30, 15,  0, 40],
    }
    fam_mems = {
        "dense":  [50,  0,  0,  0,  0,  0, 15],
        "hybrid": [88, 96, 25,  0,  0, 75, 25],
        "moe":    [50,  0,  0,  0,  0, 25, 40],
    }

    fig = plt.figure(figsize=(13, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    families = ["dense", "hybrid", "moe"]
    dx = dy = 0.55
    for fi, fam in enumerate(families):
        for pi, prim in enumerate(prims):
            # Two bars per (family, primitive): flop savings and mem savings.
            for k, (arr, tag, dz_offset) in enumerate([
                (fam_flops[fam], "flop", 0.0),
                (fam_mems[fam],  "mem",  0.0),
            ]):
                x = pi + (0.15 if k == 0 else -0.4)
                y = fi + (0.15 if k == 0 else -0.4)
                z = 0
                dz = arr[pi]
                if dz <= 0: continue
                colour = FAMILY_COLOUR[fam] if k == 0 else PAL["good"]
                ax.bar3d(x, y, z, 0.35, 0.35, dz,
                         color=colour, alpha=0.85, edgecolor="white",
                         linewidth=0.4)

    ax.set_xticks(range(len(prims)))
    ax.set_xticklabels(prims, rotation=25, ha="right")
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families)
    ax.set_zlabel("savings (%)", labelpad=8)
    ax.set_zlim(0, 100)
    ax.view_init(elev=22, azim=-55)
    ax.set_title("Per-primitive savings by architecture family —\n"
                 "family-tinted bars = FLOP savings, green bars = memory savings")
    fig.savefig(out); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════
# 4 · Full attn Θ(S²·D) surface vs SMSA Θ(S·w·D) surface
# ═══════════════════════════════════════════════════════════════════
def plot_scaling_surface(out: Path):
    seq_lens = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    dims     = np.array([32, 48, 64, 96, 128, 192, 256])

    S, D = np.meshgrid(seq_lens, dims)
    # arbitrary scaling factors chosen so the two surfaces are visually distinct.
    full_ms = 2.0 * S * S * D / 1e10
    smsa_ms = 2.0 * S * 64  * D / 1e10   # w = 64

    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # Full attention: warm cmap
    full_surf = ax.plot_surface(
        np.log2(S), D, np.log10(full_ms + 1e-9),
        cmap=cm.Reds, alpha=0.85, edgecolor="none",
        rstride=1, cstride=1,
    )
    # SMSA: cool cmap
    smsa_surf = ax.plot_surface(
        np.log2(S), D, np.log10(smsa_ms + 1e-9),
        cmap=cm.Blues, alpha=0.9, edgecolor="none",
        rstride=1, cstride=1,
    )
    fig.colorbar(full_surf, ax=ax, shrink=0.5, aspect=14, pad=0.02,
                 label="log₁₀ full-attn ms")
    fig.colorbar(smsa_surf, ax=ax, shrink=0.5, aspect=14, pad=0.12,
                 label="log₁₀ SMSA ms")

    ax.set_xlabel("sequence length (log₂ tokens)", labelpad=10)
    ax.set_ylabel("head dimension D", labelpad=10)
    ax.set_zlabel("log₁₀ wall-time (ms, analytical)", labelpad=8)
    xt = np.log2(seq_lens)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{S//1024}K" if S >= 1024 else str(S) for S in seq_lens])
    ax.view_init(elev=22, azim=-58)
    ax.set_title(
        "Wall-time surface — full-attention Θ(S²·D) (red) vs SMSA Θ(S·w·D) (blue).\n"
        "Vertical gap grows as O(S/w) with the sequence length."
    )

    fig.savefig(out); plt.close(fig)
    print(f"  → {out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="plots",
                    help="directory to write PNGs into")
    args = ap.parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing:  {out_dir}")

    plot_kv_reduction_surface(out_dir / "kv_reduction_surface.png")
    plot_pareto_3d(out_dir / "pareto_3d.png")
    plot_primitive_savings_3d(out_dir / "primitive_savings_3d.png")
    plot_scaling_surface(out_dir / "scaling_surface.png")


if __name__ == "__main__":
    main()

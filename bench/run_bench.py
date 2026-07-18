"""OpusEdge — real-model benchmark driver.

Usage
-----
    uv run python run_bench.py                        # runs dense + hybrid + moe with defaults
    uv run python run_bench.py --skip moe             # skip MoE (heavy download)
    uv run python run_bench.py --dense-model Qwen/Qwen2.5-0.5B --seq-len 512

Results are written to `results.json` and a summary is printed to stdout.
"""

from __future__ import annotations
import argparse, json, time, sys, traceback
from pathlib import Path
from opusedge_bench import run_dense, run_hybrid, run_moe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── Defaults chosen to fit inside an 8 GB RTX 4060 with INT4 quant ──
    ap.add_argument("--dense-model",  default="Qwen/Qwen2.5-0.5B",
                    help="HuggingFace id for the dense benchmark (Qwen2.5-0.5B: 0.5B params, "
                         "~0.25 GB @ INT4 — matches the paper's dense baseline).")
    ap.add_argument("--hybrid-model", default="tiiuae/Falcon-H1-0.5B-Instruct",
                    help="HuggingFace id for the hybrid SSM-attention benchmark "
                         "(Falcon-H1-0.5B: 36 hybrid layers, 1:1 parallel design, "
                         "matches the paper's SelKV/SMSA/StateCompress numbers).")
    ap.add_argument("--moe-model",    default="ibm-granite/granite-3.1-1b-a400m-instruct",
                    help="HuggingFace id for the MoE benchmark "
                         "(IBM Granite-3.1-1B-A400M: 1.3B total params, 400M active per token — "
                         "the smallest practical MoE that fits in 8 GB VRAM).")
    ap.add_argument("--seq-len",   type=int, default=512)
    ap.add_argument("--prompt",    default="long_context",
                    choices=["long_context", "code_reasoning", "factual_recall"])
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["dense", "hybrid", "moe"],
                    help="Skip one or more families")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    results: dict = {"config": vars(args), "runs": []}

    families = [
        ("dense",  run_dense,  args.dense_model),
        ("hybrid", run_hybrid, args.hybrid_model),
        ("moe",    run_moe,    args.moe_model),
    ]

    for name, fn, mid in families:
        if name in args.skip:
            print(f"\n── skipping {name} ────────────────────────────────")
            continue
        print(f"\n── {name} · {mid} ─────────────────────────────────────")
        t0 = time.perf_counter()
        try:
            r = fn(model_id=mid, seq_len=args.seq_len, prompt_key=args.prompt)
            elapsed = time.perf_counter() - t0
            print(f"  ✓ done in {elapsed:.1f}s   "
                  f"baseline_ppl={r.baseline_ppl:.3f}")
            print(f"  selkv @ 87.5% = {r.selkv_ppl.get('0.875', 'n/a')}   "
                  f"random @ 87.5% = {r.random_ppl.get('0.875', 'n/a')}")
            results["runs"].append(r.to_dict())
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc(limit=2)
            results["runs"].append({"family": name, "model": mid, "error": str(e)})

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n→ wrote {out.resolve()}")

    # Compact stdout summary
    print("\n" + "=" * 78)
    print(f"{'family':7s} {'model':32s} {'base':>7s} {'selkv@.875':>10s} {'rand@.875':>10s} {'kv-red@.875':>11s}")
    print("-" * 78)
    for r in results["runs"]:
        if "error" in r:
            print(f"{r['family']:7s} {r['model']:32s}  FAILED")
            continue
        s = r["selkv_ppl"].get("0.875", float("nan"))
        rp = r["random_ppl"].get("0.875", float("nan"))
        kvred = (r.get("kv_cache_pct_reduction_selkv") or {}).get("0.875", float("nan"))
        print(f"{r['family']:7s} {r['model'][:32]:32s} "
              f"{r['baseline_ppl']:7.3f} {s:10.3f} {rp:10.3f} {kvred:10.2f}%")

    # KV cache reduction table
    print("\n── KV cache footprint (SelKV) ──")
    print(f"{'model':32s} {'baseline':>10s} {'@25%':>12s} {'@50%':>12s} {'@75%':>12s} {'@87.5%':>12s}")
    for r in results["runs"]:
        if "error" in r or not r.get("kv_cache_mb_after_selkv"): continue
        base = r["kv_cache_mb_baseline"]
        row = f"{r['model'][:32]:32s} {base:8.2f} MB"
        for k in ("0.250", "0.500", "0.750", "0.875"):
            mb = r["kv_cache_mb_after_selkv"].get(k, 0)
            pct = r["kv_cache_pct_reduction_selkv"].get(k, 0)
            row += f"  {mb:5.2f}MB/{pct:4.1f}%"
        print(row)

    # SMSA KV cache reduction
    print("\n── KV cache footprint (SMSA) ──")
    print(f"{'model':32s} {'window=64':>16s} {'window=128':>16s} {'window=256':>16s}")
    for r in results["runs"]:
        if "error" in r or not r.get("kv_cache_mb_after_smsa"): continue
        row = f"{r['model'][:32]:32s}"
        for w in ("64", "128", "256"):
            mb = r["kv_cache_mb_after_smsa"].get(w, 0)
            pct = r["kv_cache_pct_reduction_smsa"].get(w, 0)
            row += f"  {mb:5.2f}MB/{pct:5.1f}%"
        print(row)

    # All-primitives audit — count how many got exercised
    print("\n── All-primitives audit ──")
    for r in results["runs"]:
        if "error" in r: continue
        prims = r.get("all_primitives") or []
        if prims and "error" in prims[0]:
            print(f"  {r['model'][:40]:40s}  {prims[0]['error']}")
            continue
        by_cat: dict[str, int] = {}
        for p in prims:
            by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
        summary = " · ".join(f"{k}={v}" for k, v in sorted(by_cat.items()))
        print(f"  {r['model'][:40]:40s}  {len(prims)} bindings exercised   [{summary}]")


if __name__ == "__main__":
    sys.exit(main())

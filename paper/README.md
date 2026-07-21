# OpusEdge — Paper

> The full paper for the OpusEdge project. This directory holds the LaTeX
> source, the built PDF, the DOCX submission source, and the arXiv-ready
> tarball.

**Title.** OpusEdge: Telemetry-Guided Dynamic Compute Allocation for Dense,
Mixture-of-Experts, and Hybrid SSM–Attention Architectures.

**Author.** Irfan Mahir — Furylogic Labs / Independent Researcher, Bangladesh
— research conducted under the Infernix Inference Engine Project.
[`irfan@furylogic.com`](mailto:irfan@furylogic.com)

**Date.** June 2026 · v1.0.0 · 18 pages.

---

## Files

| File | Purpose |
|------|---------|
| [`OpusEdge.pdf`](OpusEdge.pdf) | Compiled PDF — the reference copy linked from the README badge and all docs. |
| [`opusedge.pdf`](opusedge.pdf) | Identical to `OpusEdge.pdf`, produced by `make`. |
| [`opusedge.tex`](opusedge.tex) | LaTeX source. Single-column `article`, 11 pt. Everything (24 equations, 10 data tables, all stabilizers, all limitations) is in here. |
| [`opusedge.bib`](opusedge.bib) | 17 references — Mamba, Mamba-2 (SSD), Falcon-H1, Jamba, H₂O, SAGE-KV, Ada-KV, vLLM, Mixtral, OLMoE, Qwen2.5, LLaMA-3, DeepSeek-V2, CALM, ToMe, Speculative Decoding. |
| [`Makefile`](Makefile) | `make` builds the PDF; `make arxiv` packages `opusedge-arxiv.tar.gz`; `make watch` uses `latexmk -pvc`. |
| [`opusedge-arxiv.tar.gz`](opusedge-arxiv.tar.gz) | Ready-to-upload arXiv submission — `.tex` + `.bbl` only, no clutter. |
| [`OpusEdge.docx`](OpusEdge.docx) | Original DOCX source (kept for editing convenience; the `.tex` is the source of truth). |
| [`LICENSE`](LICENSE) | **AGPL-3.0** — the paper and its figures / DOCX / PDF renderings. The code lives under a separate PolyForm-Noncommercial licence at the repo root. |

## Build

```bash
# from paper/
make                # pdflatex → bibtex → pdflatex × 2 → opusedge.pdf
make watch          # latexmk -pvc for live rebuild while editing
make arxiv          # produce opusedge-arxiv.tar.gz (tex + bbl only)
make clean          # remove aux/log/bbl/blg/toc etc.
make distclean      # also remove the PDF and arxiv tarball
```

Requires `pdflatex`, `bibtex`, and `latexmk` from TeX Live (Debian:
`sudo apt install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended latexmk`).

## What the paper covers

**Central claim.** All three modern transformer architecture families —
dense, mixture-of-experts, and hybrid SSM–attention — emit cheap per-token
importance signals that we can harvest to drive a single set of composable
inference primitives, with no retraining, on consumer hardware.

**Three signals, one framework.**

| Family | Signal | Cost | Correlation ρ with attention importance |
|--------|--------|------|----:|
| Hybrid (Falcon-H1, Jamba, Mamba-2) | native SSM Δ (selectivity) | O(1) — free | **0.51** |
| Dense (Qwen2.5, LLaMA-3, SmolLM) | Proxy-Δ (RMS hidden-state drift) | O(L) per token | **0.28** |
| MoE (Mixtral, OLMoE, IBM Granite MoE) | Router-Gated IR (max-min or entropy) | O(E) | — |

**Ten composable primitives** — SelKV, SMSA, Delta-AR, ΔRank, HeadDeactivate,
StateCompress, DenseEvic, Pareto Frontier, MoE Router-IR, GAKV/R-GAKV.

**Four stabilizers** — MPSR/SACT, EB-AR/EB-DAR, CASP+NDPA, IPSS/CSA.

**Two task controllers** — CAL (dense + MoE), R-CAL (hybrid).

## Headline empirical results

| Claim | Number | Model | Location in paper |
|-------|--------|-------|------|
| SelKV vs random @ 87.5 % KV eviction | **100.5× quality retention** | Falcon-H1-0.5B, WT-103 227-token | Table §7.2 |
| DenseEvic vs random @ 87.5 % (dense) | **13.7× quality retention** | Qwen2.5-0.5B | Table §7.9 |
| SMSA measured speedup @ 2K tokens | **3.56–4.98×** | Falcon-H1 & Qwen2.5 | Table §7.3 |
| StateCompress hidden-state compression | **37.5 %**, near-zero ΔPPL | Falcon-H1 | §7.7 |
| Integrated ablation @ 2K tokens | **98.9 % VRAM reduction** | Falcon-H1 | Table §7.10 |
| Δ ↔ attention correlation | ρ = **0.51** (native), **0.28** (Proxy-Δ) | 24 diverse prompts | §7.1 |
| Reference impl, chunked-SelKV | **93.8 % MEASURED KV reduction** @ 65 536 tokens on 1.8 GB VRAM | Qwen2.5-0.5B | §8 |

All experiments reproducible on a **single NVIDIA RTX 4060 (8 GB VRAM)**
using publicly available models.

## Reproducibility

Every experiment in the paper is backed by the reference implementation at
**<https://github.com/Mr-DS-ML-85/OpusEdge>**:

- C++20 engine (header-only) with 30 primitive bindings
- Python SDK (`opusedge_cpp`) mirroring the C++ SDK 1:1
- `bench/bench_llm_scaling.py` — real HF-model prefill/decode benchmarks
- `bench/bench_llm_scaling.py --effective-context 65536` — chunked-SelKV
  streaming benchmark on 8 GB VRAM
- `engine/benchmarks/bench_scaling` — C++ empirical-complexity classifier

Seeds pinned: `torch.manual_seed(0)`, `numpy.random.default_rng(42)`,
`std::mt19937(0)`.

## Cite

```bibtex
@techreport{opusedge2026,
  author      = {Irfan Mahir},
  title       = {OpusEdge: Telemetry-Guided Dynamic Compute Allocation for
                 Dense, Mixture-of-Experts, and Hybrid SSM--Attention Architectures},
  institution = {Furylogic Labs / Infernix Inference Engine Project},
  year        = {2026},
  month       = jun,
  note        = {\url{https://github.com/Mr-DS-ML-85/OpusEdge}},
}
```

Or use the machine-readable [`CITATION.cff`](../CITATION.cff) at repo root —
GitHub's "Cite this repository" button reads it directly.

## Licence

**Paper (this directory).** [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).
Share, adapt, and distribute with attribution; derivatives must use the same license.

**Code (repository root).** [PolyForm Noncommercial 1.0.0](../LICENSE).
Research, teaching, personal, and charitable / educational / government use
are permitted. Commercial use requires a separate licence — contact the
author.

## Contact

Irfan Mahir · [`irfan@furylogic.com`](mailto:irfan@furylogic.com) ·
[github.com/Mr-DS-ML-85](https://github.com/Mr-DS-ML-85) ·
Furylogic Labs, Bangladesh.

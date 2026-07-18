# Contributing to OpusEdge

Thanks for looking! OpusEdge is a research reference implementation of the paper
*"OpusEdge: Telemetry-Guided Dynamic Compute Allocation for Dense, MoE, and
Hybrid SSM-Attention Architectures"* — contributions that keep the code
faithful to the paper are the most useful.

## What we welcome

- **Bug fixes** — especially numerical correctness in `engine/include/opusedge/*`.
- **New model backends** — right now the Python bench exercises Qwen (dense),
  Falcon-H1 (hybrid), and Granite MoE. Any HF causal LM that exposes hidden
  states works with the harness.
- **New tests** in `engine/tests/`.
- **Documentation** — especially clearer derivations connecting the C++ code
  back to specific paper equations.
- **Landing/docs improvements** in `web/` and `docs/`.

## What we're not looking for (right now)

- Renames / stylistic rewrites of the C++ headers without a numerical reason.
  Every primitive matches a paper section; drift there breaks reproducibility.
- Removing the family-scoped SDK abstractions. Per the paper (§1 contribution
  2) all 10 core primitives are universal — the SDK signals that intent.

## Dev loop

```bash
# 1. C++ engine
cmake -S engine -B engine/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPUSEDGE_BUILD_TESTS=ON \
  -DOPUSEDGE_BUILD_BENCHMARKS=ON \
  -DOPUSEDGE_BUILD_EXAMPLES=ON
cmake --build engine/build -j
ctest --test-dir engine/build --output-on-failure

# 2. Python bench + SDK
cd bench
uv sync
uv run python run_bench.py --skip moe hybrid --seq-len 128    # dense smoke test
```

Everything is header-only C++20 + `Eigen 3.4`. No CUDA, no PyTorch on the
engine side.

## Style

- **C++**: match the surrounding file. `snake_case` free functions, `PascalCase`
  types, 4-space indent, `Float = double` throughout.
- **Python**: `black`-ish (line length ~ 100). Docstrings on public callables
  that cite the paper equation they implement whenever possible.
- **CMake**: 3.20+, no vendored deps except optional FetchContent for GTest.

## Pull request checklist

- [ ] `ctest` passes (`ctest --test-dir engine/build`)
- [ ] `bench_scaling` still classifies primitives correctly
- [ ] Commit message references the paper section or eq. you touched
- [ ] Docs updated if you changed a public API

## Signing off

You must have the right to license your contribution under the PolyForm
Noncommercial 1.0.0 License (for code) and CC BY-NC 4.0 (for any paper /
docs contributions). By opening a PR you assert that you do.

## Questions

Open an issue or ping [@Mr-DS-ML-85](https://github.com/Mr-DS-ML-85).
Bangladesh-time, mostly async.

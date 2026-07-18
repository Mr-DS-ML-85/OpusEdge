#include "opusedge/primitives/delta_ar.h"
#include <iostream>
#include <chrono>

using namespace opusedge;

double benchmark_sparse_vs_dense(int S, int K, int D, int iters = 100) {
    MatrixXf Q = MatrixXf::Random(S, D);
    MatrixXf Km = MatrixXf::Random(S, D);
    MatrixXf V = MatrixXf::Random(S, D);
    VectorXf delta = VectorXf::Random(S).cwiseAbs();

    MatrixXf idx = build_delta_ar_indices(delta, K);
    MatrixXi idx_int = idx.cast<int>();

    // Warmup
    for (int i = 0; i < 5; ++i)
        sparse_attention(Q, Km, V, idx_int);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        sparse_attention(Q, Km, V, idx_int);
    auto end = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return ms / iters;
}

int main() {
    std::cout << "=== Delta-AR Sparse Attention Benchmarks ===\n\n";

    struct BenchCase { int S, K; const char* label; };
    BenchCase cases[] = {
        {512, 64,  "S=512,  K=64  (12.5%)"},
        {1024, 128, "S=1024, K=128 (12.5%)"},
        {2048, 256, "S=2048, K=256 (12.5%)"},
        {2048, 128, "S=2048, K=128 (6.25%)"},
        {4096, 256, "S=4096, K=256 (6.25%)"},
    };

    std::cout << std::fixed << std::setprecision(3);
    for (auto& c : cases) {
        double t = benchmark_sparse_vs_dense(c.S, c.K, 64, 50);
        auto flops = delta_ar_flops(c.S, c.K, 64);
        std::cout << "  " << c.label
                  << "  avg=" << t << "ms  "
                  << "FLOPs: " << flops.routed / 1e6 << "M"
                  << "  (" << (flops.reduction * 100) << "% reduction)\n";
    }

    std::cout << "\n  Theoretical baseline (dense):\n";
    for (auto& c : cases) {
        auto flops = delta_ar_flops(c.S, c.S, 64);
        std::cout << "  S=" << c.S << "  dense FLOPs: "
                  << flops.baseline / 1e6 << "M\n";
    }
    return 0;
}

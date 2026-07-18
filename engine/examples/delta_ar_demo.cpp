#include "opusedge/primitives/delta_ar.h"
#include "opusedge/primitives/selkv.h"
#include "opusedge/primitives/smsa.h"
#include "opusedge/primitives/head_gate.h"
#include "opusedge/primitives/state_compress.h"
#include "opusedge/primitives/pareto.h"
#include "opusedge/primitives/composite.h"
#include "opusedge/core/signal.h"
#include <iostream>
#include <iomanip>

using namespace opusedge;

void demo_delta_ar() {
    std::cout << "\n=== Delta-AR + EB-DAR Demo ===\n";

    int S = 16, D = 8;
    VectorXf delta(S);
    delta << 0.1, 0.9, 0.05, 0.7, 0.3, 0.01, 0.6, 0.4,
             0.2, 0.8, 0.15, 0.5, 0.25, 0.35, 0.45, 0.55;

    auto res = measure_delta_ar(delta, 4);
    std::cout << "  Sequence length: " << res.total_tokens << "\n";
    std::cout << "  Top-K: " << res.top_k << "\n";
    std::cout << "  Tokens attended: " << res.tokens_attended << "\n";
    std::cout << "  FLOP reduction: " << (res.flop_reduction * 100) << "%\n";
    std::cout << "  EB-DAR mean reservoir: " << res.ebdar_mean_reservoir << "\n";

    auto flops = delta_ar_flops(S, res.top_k, D);
    std::cout << "  Baseline FLOPs: " << flops.baseline << "\n";
    std::cout << "  Routed FLOPs:   " << flops.routed << "\n";
}

void demo_selkv() {
    std::cout << "\n=== SelKV Demo ===\n";
    VectorXf delta(8);
    delta << 0.1, 0.9, 0.05, 0.7, 0.3, 0.01, 0.6, 0.4;
    auto ev = SelKV::evict(delta, 0.5, 8);
    std::cout << "  Retained " << ev.retained_indices.size()
              << " tokens, evicted " << ev.evicted_indices.size() << "\n";
    std::cout << "  Memory savings: " << (ev.memory_savings * 100) << "%\n";
    std::cout << "  Quality ratio vs random: "
              << SelKV::quality_ratio(delta, 0.5) << "\n";
}

void demo_smsa() {
    std::cout << "\n=== SMSA Demo ===\n";
    SMSA smsa(256);
    auto res = smsa.analyze(2048);
    std::cout << "  Window: " << res.effective_window << "\n";
    std::cout << "  Speedup: " << res.speedup << "x\n";
    std::cout << "  KV memory savings: " << (res.memory_savings * 100) << "%\n";
}

void demo_head_gate() {
    std::cout << "\n=== HeadDeactivate Demo ===\n";
    HeadGateConfig cfg;
    cfg.n_heads = 32;
    HeadGate hg(cfg);

    VectorXf deltas(6);
    deltas << 0.01, 0.10, 0.20, 0.50, 0.03, 0.25;
    auto res = hg.analyze(deltas);
    std::cout << "  Avg active heads: " << res.active_heads << "/32\n";
    std::cout << "  FLOP reduction: " << res.flop_reduction_pct << "%\n";
    std::cout << "  Distribution: L=" << res.low_count << " M=" << res.mid_count
              << " H=" << res.high_count << " C=" << res.critical_count << "\n";
}

void demo_composite() {
    std::cout << "\n=== Composite Demo ===\n";
    CompositeConfig cfg;
    Composite comp(cfg);
    auto res = comp.analyze(2048, 32, 64, 512);
    std::cout << "  FLOP reduction: " << res.flop_reduction_pct << "%\n";
    std::cout << "  Memory savings: " << res.memory_savings_pct << "%\n";
    std::cout << "  Primitives: ";
    for (auto& p : res.applied_primitives) std::cout << p << ", ";
    std::cout << "\n";
}

int main() {
    demo_delta_ar();
    demo_selkv();
    demo_smsa();
    demo_head_gate();
    demo_composite();
    return 0;
}

// bench_pipeline.cpp — end-to-end per-family pipeline benchmark.
//
// Runs the SDK's DensePipeline, HybridPipeline, MoEPipeline with a
// realistic sequence of primitives (per the OpusEdge paper's routing
// rules), reports per-primitive latency, cumulative KV footprint, and
// aggregate pipeline stats. Writes CSV for downstream plotting.
//
// Usage
//   bench_pipeline                                 # all three families, defaults
//   bench_pipeline --family dense --seq 2048
//   bench_pipeline --iters 30 --warmup 5 --out pipeline.csv

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <numeric>

#include "opusedge/sdk.h"

using namespace opusedge;
using namespace opusedge::sdk;
using clk = std::chrono::high_resolution_clock;

// ── measured event ───────────────────────────────────────────────────
struct Event {
    std::string family;
    std::string primitive;
    int seq_len;
    double mean_ms;
    double std_ms;
    double mem_saved_pct;
    double flop_saved_pct;
    std::string notes;
};

// ── time a lambda ────────────────────────────────────────────────────
template <class F>
static std::pair<double,double> time_it(F&& f, int iters, int warmup) {
    for (int i = 0; i < warmup; ++i) f();
    std::vector<double> ms; ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now(); f();
        auto t1 = clk::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double sum = 0; for (double x : ms) sum += x;
    double mean = sum / ms.size();
    double var = 0; for (double x : ms) var += (x - mean) * (x - mean);
    var /= std::max<size_t>(ms.size() - 1, 1);
    return {mean, std::sqrt(var)};
}

// ── random helpers ──────────────────────────────────────────────────
static VectorXf random_delta(int T, uint32_t seed = 0) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<Float> u(0.01, 1.0);
    VectorXf v(T);
    for (int i = 0; i < T; ++i) v(i) = u(g);
    return v;
}
static MatrixXf random_mat(int R, int C, uint32_t seed = 0) {
    std::mt19937 g(seed);
    std::normal_distribution<Float> nd(0, 1);
    MatrixXf m(R, C);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j) m(i, j) = nd(g);
    return m;
}

// ═══════════════════════════════════════════════════════════════════
// DENSE PIPELINE — DenseEvic → NDPA → SSR → ΔRank → HeadDeact → CAL
// (per paper §Dense + user routing: CAL for dense+MoE, R-CAL for hybrid)
// ═══════════════════════════════════════════════════════════════════
static std::vector<Event> bench_dense(const ModelShape& shape, int iters, int warmup) {
    std::vector<Event> out;

    DensePipeline::Config cfg;
    cfg.eviction_ratio = 0.5;
    cfg.rank_fraction  = 0.7;
    cfg.task_label     = "general";
    DensePipeline dense(shape, cfg);

    const int T = shape.seq_len;
    auto proxy_delta = random_delta(T, 1);

    // 1 · DenseEvic (SelKV with Proxy-Δ)
    EvictionResult evres;
    {
        auto [m, s] = time_it([&]{ evres = dense.dense_evict(proxy_delta); }, iters, warmup);
        double mem_pct = 100.0 * evres.memory_savings;
        out.push_back({"dense", "DenseEvic", T, m, s, mem_pct, 0.0,
                       "kept " + std::to_string(evres.retained_indices.size()) + "/" + std::to_string(T)});
    }

    // 2 · NDPA — rectify Proxy-Δ to internal attn manifold
    {
        auto delta_v  = std::vector<double>(proxy_delta.data(), proxy_delta.data() + T);
        auto attn_v   = std::vector<double>(proxy_delta.data(), proxy_delta.data() + T);
        std::reverse(attn_v.begin(), attn_v.end());
        std::vector<double> rectified;
        auto [m, s] = time_it([&]{ rectified = dense.ndpa_rectify(delta_v, attn_v); }, iters, warmup);
        out.push_back({"dense", "NDPA", T, m, s, 0.0, 0.0,
                       "rectified " + std::to_string(rectified.size()) + " scores"});
    }

    // 3 · SSR (soft SVD) on hidden-projection singular values
    {
        std::vector<double> sv(shape.hidden_dim);
        for (int i = 0; i < shape.hidden_dim; ++i) sv[i] = 1.0 / (i + 1);
        SSRResult r;
        auto [m, s] = time_it([&]{ r = dense.ssr(sv, 0.5); }, iters, warmup);
        out.push_back({"dense", "SSR", T, m, s, 0.0, 100.0 * r.compression_ratio,
                       std::to_string(r.preserved_count) + "/" + std::to_string(sv.size()) + " preserved"});
    }

    // 4 · CASP — curvature-adaptive variant of SSR
    {
        std::vector<double> sv(shape.hidden_dim);
        for (int i = 0; i < shape.hidden_dim; ++i) sv[i] = 1.0 / (i + 1);
        SSRResult r;
        auto [m, s] = time_it([&]{ r = dense.casp(sv, 1.0); }, iters, warmup);
        out.push_back({"dense", "CASP", T, m, s, 0.0, 100.0 * r.compression_ratio, ""});
    }

    // 5 · ΔRank projection (SSR-softened SVD on a Q/K/V block)
    {
        auto W = random_mat(shape.head_dim, shape.head_dim, 2);
        MatrixXf Wp;
        auto [m, s] = time_it([&]{ Wp = dense.delta_rank_project(W); }, iters, warmup);
        double flop_saved = 100.0 * (1.0 - cfg.rank_fraction);
        out.push_back({"dense", "DeltaRank+SSR", T, m, s, 0.0, flop_saved, ""});
    }

    // 6 · HeadDeactivate + IPSS
    {
        HeadGate::GateResult r;
        auto [m, s] = time_it([&]{ r = dense.head_gate(proxy_delta); }, iters, warmup);
        out.push_back({"dense", "HeadDeactivate", T, m, s, 0.0, r.flop_reduction_pct, ""});
    }
    {
        std::vector<double> hv(shape.n_heads, 0.3);
        IPSSResult r;
        auto [m, s] = time_it([&]{ r = dense.ipss(hv); }, iters, warmup);
        out.push_back({"dense", "IPSS", T, m, s, 0.0, 100.0 * r.flop_reduction,
                       std::to_string(r.n_active) + "/" + std::to_string(hv.size()) + " active"});
    }

    // 7 · CAL — task rigidity modulator
    {
        Float thresh;
        auto [m, s] = time_it([&]{ thresh = dense.cal_effective_threshold(0.5); }, iters, warmup);
        std::ostringstream oss; oss << "τ_eff=" << std::fixed << std::setprecision(3) << thresh;
        out.push_back({"dense", "CAL", T, m, s, 0.0, 0.0, oss.str()});
    }

    // KV footprint summary
    {
        double kv_b = dense.kv_baseline_mib();
        double kv_a = dense.kv_after_evict_mib(cfg.eviction_ratio);
        double pct  = 100.0 * (1.0 - kv_a / std::max(kv_b, 1e-9));
        std::ostringstream oss; oss << std::fixed << std::setprecision(2)
            << kv_b << " → " << kv_a << " MiB";
        out.push_back({"dense", "kv_footprint", T, 0, 0, pct, 0.0, oss.str()});
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════
// HYBRID PIPELINE — SelKV → SMSA → Delta-AR → EB-DAR → StateCompress
//                   → MPSR → R-CAL → HeadDeact → IPSS
// (per paper §Hybrid + user: R-CAL for hybrid)
// ═══════════════════════════════════════════════════════════════════
static std::vector<Event> bench_hybrid(const ModelShape& shape, int iters, int warmup) {
    std::vector<Event> out;

    HybridPipeline::Config cfg;
    cfg.eviction_ratio  = 0.875;
    cfg.smsa_window     = 64;
    cfg.delta_ar_top_k  = 64;
    cfg.task_label      = "general";
    HybridPipeline hybrid(shape, cfg);

    const int T = shape.seq_len;
    auto native_delta = random_delta(T, 3);

    // 1 · SelKV (native Δ) — hybrid's fast path
    EvictionResult evres;
    {
        auto [m, s] = time_it([&]{ evres = hybrid.selkv(native_delta); }, iters, warmup);
        out.push_back({"hybrid", "SelKV(native Δ)", T, m, s,
                       100.0 * evres.memory_savings, 0.0,
                       "kept " + std::to_string(evres.retained_indices.size()) + "/" + std::to_string(T)});
    }

    // 2 · SMSA — SSD-mask + sliding-window forward
    {
        MatrixXf ssd;
        auto [m, s] = time_it([&]{ ssd = hybrid.smsa_ssd_mask(native_delta); }, iters, warmup);
        double density = 100.0 * ssd.sum() / (ssd.rows() * ssd.cols());
        std::ostringstream oss; oss << "density=" << std::fixed << std::setprecision(1) << density << "%";
        out.push_back({"hybrid", "SMSA(SSD mask)", T, m, s, 100.0 - density, 100.0 - density, oss.str()});
    }
    {
        auto Q = random_mat(T, shape.head_dim, 4);
        auto K = random_mat(T, shape.head_dim, 5);
        auto V = random_mat(T, shape.head_dim, 6);
        MatrixXf O;
        auto [m, s] = time_it([&]{ O = hybrid.smsa_forward(Q, K, V); }, iters, warmup);
        double flop_saved = 100.0 * (1.0 - static_cast<double>(cfg.smsa_window) / T);
        out.push_back({"hybrid", "SMSA(forward)", T, m, s, flop_saved, flop_saved, ""});
    }

    // 3 · Delta-AR — pre-attention top-K routing
    MatrixXf dar_idx;
    {
        auto [m, s] = time_it([&]{ dar_idx = hybrid.delta_ar_indices(native_delta); }, iters, warmup);
        out.push_back({"hybrid", "Delta-AR(indices)", T, m, s, 0.0,
                       100.0 * (1.0 - static_cast<double>(cfg.delta_ar_top_k) / T), ""});
    }
    {
        auto Q = random_mat(T, shape.head_dim, 4);
        auto K = random_mat(T, shape.head_dim, 5);
        auto V = random_mat(T, shape.head_dim, 6);
        MatrixXi idx = dar_idx.cast<int>();
        MatrixXf O;
        auto [m, s] = time_it([&]{ O = hybrid.delta_ar_sparse_attn(Q, K, V, idx); }, iters, warmup);
        double flop_saved = 100.0 * (1.0 - static_cast<double>(cfg.delta_ar_top_k) / T);
        out.push_back({"hybrid", "Delta-AR(sparse attn)", T, m, s, flop_saved, flop_saved, ""});
    }

    // 4 · EB-DAR reservoir for masked tokens
    {
        auto mask = build_delta_ar_mask(native_delta, cfg.delta_ar_top_k);
        std::pair<VectorXf, VectorXf> r;
        auto [m, s] = time_it([&]{ r = hybrid.ebdar(native_delta, mask); }, iters, warmup);
        std::ostringstream oss; oss << "reservoir μ=" << std::fixed << std::setprecision(3) << r.first.mean();
        out.push_back({"hybrid", "EB-DAR", T, m, s, 0.0, 0.0, oss.str()});
    }

    // 5 · StateCompress — zero low-magnitude SSM channels
    {
        // Materialise the Eigen expression to a VectorXf before capturing in
        // the lambda — otherwise `auto` deduces a reference to a temporary.
        VectorXf state = random_mat(1, shape.hidden_dim, 7).row(0).transpose();
        VectorXf compressed;
        auto [m, s] = time_it([&]{ compressed = hybrid.state_compress(state, 0.05); }, iters, warmup);
        int zeroed = (compressed.array() == 0).count();
        double mem = 100.0 * zeroed / state.size();
        out.push_back({"hybrid", "StateCompress", T, m, s, mem, 0.0,
                       std::to_string(zeroed) + "/" + std::to_string(state.size()) + " zeroed"});
    }

    // 6 · MPSR — project evicted KV into SSM state
    {
        std::vector<double> evicted(native_delta.data(), native_delta.data() + T);
        MPSRResult r;
        auto [m, s] = time_it([&]{ r = hybrid.mpsr(evicted); }, iters, warmup);
        std::ostringstream oss; oss << "cr=" << std::fixed << std::setprecision(3) << r.compression_ratio
                                    << " en=" << r.energy_retained;
        out.push_back({"hybrid", "MPSR", T, m, s, 100.0 * (1.0 - r.compression_ratio), 0.0, oss.str()});
    }

    // 7 · R-CAL — runtime EMA classifier
    {
        RCALResult r;
        auto [m, s] = time_it([&]{ r = hybrid.rcal_classify(0.5); }, iters, warmup);
        std::ostringstream oss; oss << r.tier.name << " τ=" << std::fixed << std::setprecision(3) << r.effective_threshold;
        out.push_back({"hybrid", "R-CAL", T, m, s, 0.0, 0.0, oss.str()});
    }

    // 8 · HeadDeactivate + IPSS
    {
        HeadGate::GateResult r;
        auto [m, s] = time_it([&]{ r = hybrid.head_gate(native_delta); }, iters, warmup);
        out.push_back({"hybrid", "HeadDeactivate", T, m, s, 0.0, r.flop_reduction_pct, ""});
    }
    {
        std::vector<double> hv(shape.n_heads, 0.3);
        IPSSResult r;
        auto [m, s] = time_it([&]{ r = hybrid.ipss(hv); }, iters, warmup);
        out.push_back({"hybrid", "IPSS", T, m, s, 0.0, 100.0 * r.flop_reduction, ""});
    }

    // KV footprint (combined SelKV+SMSA)
    {
        double kv_b = hybrid.kv_baseline_mib();
        double kv_a = hybrid.kv_after_pipeline_mib();
        double pct  = 100.0 * (1.0 - kv_a / std::max(kv_b, 1e-9));
        std::ostringstream oss; oss << std::fixed << std::setprecision(2)
            << kv_b << " → " << kv_a << " MiB (SelKV∩SMSA)";
        out.push_back({"hybrid", "kv_footprint", T, 0, 0, pct, 0.0, oss.str()});
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════
// MoE PIPELINE — Router-IR → GAKV → R-GAKV → CAL → HeadDeact → IPSS
// (per paper §MoE + user: CAL for dense+MoE)
// ═══════════════════════════════════════════════════════════════════
static std::vector<Event> bench_moe(const ModelShape& shape, int n_experts,
                                     int iters, int warmup) {
    std::vector<Event> out;

    MoEPipeline::Config cfg;
    cfg.eviction_ratio = 0.5;
    cfg.gakv_alpha     = 0.5;
    cfg.gakv_beta      = 0.5;
    cfg.task_label     = "general";
    MoEPipeline moe(shape, cfg);

    const int T = shape.seq_len;

    // Synthetic router logits (a few MoE layers): [T, n_experts] each.
    std::vector<MatrixXf> router_logits;
    for (int l = 0; l < 4; ++l) router_logits.push_back(random_mat(T, n_experts, 10 + l));

    // 1 · Router-IR
    VectorXf ir;
    {
        auto [m, s] = time_it([&]{ ir = moe.router_ir(router_logits); }, iters, warmup);
        std::ostringstream oss; oss << "H̄=" << std::fixed << std::setprecision(3) << ir.mean();
        out.push_back({"moe", "Router-IR", T, m, s, 0.0, 0.0, oss.str()});
    }

    // 2 · GAKV — composite Δ + IR eviction
    auto proxy_delta = random_delta(T, 11);
    std::vector<double> delta_v(proxy_delta.data(), proxy_delta.data() + T);
    std::vector<double> ir_v(ir.data(), ir.data() + T);
    {
        GAKVResult r;
        auto [m, s] = time_it([&]{ r = moe.gakv(delta_v, ir_v); }, iters, warmup);
        double mem = 100.0 * r.evicted_indices.size() / T;
        out.push_back({"moe", "GAKV", T, m, s, mem, 0.0,
                       "evicted " + std::to_string(r.n_evicted) + "/" + std::to_string(T)});
    }

    // 3 · R-GAKV — CAL-modulated GAKV
    {
        GAKVResult r;
        Float cal_mod = moe.cal_effective_threshold(0.5);
        auto [m, s] = time_it([&]{ r = moe.rgakv(delta_v, ir_v, cal_mod); }, iters, warmup);
        double mem = 100.0 * r.evicted_indices.size() / T;
        out.push_back({"moe", "R-GAKV", T, m, s, mem, 0.0,
                       "cal_mod=" + std::to_string(cal_mod).substr(0, 5)});
    }

    // 4 · CAL — task modulator (dense+MoE per user)
    {
        Float thresh;
        auto [m, s] = time_it([&]{ thresh = moe.cal_effective_threshold(0.5); }, iters, warmup);
        std::ostringstream oss; oss << "τ_eff=" << std::fixed << std::setprecision(3) << thresh;
        out.push_back({"moe", "CAL", T, m, s, 0.0, 0.0, oss.str()});
    }

    // 5 · HeadDeactivate + IPSS
    {
        HeadGate::GateResult r;
        auto [m, s] = time_it([&]{ r = moe.head_gate(proxy_delta); }, iters, warmup);
        out.push_back({"moe", "HeadDeactivate", T, m, s, 0.0, r.flop_reduction_pct, ""});
    }
    {
        std::vector<double> hv(shape.n_heads, 0.3);
        IPSSResult r;
        auto [m, s] = time_it([&]{ r = moe.ipss(hv); }, iters, warmup);
        out.push_back({"moe", "IPSS", T, m, s, 0.0, 100.0 * r.flop_reduction, ""});
    }
    return out;
}

// ── CSV / pretty output ──────────────────────────────────────────────
static void emit_pretty(std::ostream& os, const std::vector<Event>& evs) {
    os << std::left
       << std::setw(9)  << "family"
       << std::setw(28) << "primitive"
       << std::right
       << std::setw(7)  << "S"
       << std::setw(12) << "ms"
       << std::setw(12) << "±"
       << std::setw(10) << "mem_%"
       << std::setw(10) << "flop_%"
       << "   notes\n";
    os << std::string(110, '-') << "\n";
    os << std::fixed << std::setprecision(3);
    std::string last_family;
    for (const auto& e : evs) {
        if (last_family != e.family) {
            if (!last_family.empty()) os << "\n";
            last_family = e.family;
        }
        os << std::left  << std::setw(9)  << e.family
                         << std::setw(28) << e.primitive
           << std::right << std::setw(7)  << e.seq_len
                         << std::setw(12) << e.mean_ms
                         << std::setw(12) << e.std_ms
                         << std::setw(10) << e.mem_saved_pct
                         << std::setw(10) << e.flop_saved_pct
           << "   " << e.notes << "\n";
    }
}

static void emit_csv(std::ostream& os, const std::vector<Event>& evs) {
    os << "family,primitive,seq_len,mean_ms,std_ms,mem_saved_pct,flop_saved_pct,notes\n";
    os << std::fixed << std::setprecision(6);
    for (const auto& e : evs) {
        os << e.family << ",\"" << e.primitive << "\"," << e.seq_len << ","
           << e.mean_ms << "," << e.std_ms << ","
           << e.mem_saved_pct << "," << e.flop_saved_pct << ","
           << "\"" << e.notes << "\"\n";
    }
}

// ── main ─────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // Falcon-H1-0.5B shape (hybrid): 36 hybrid layers, 32 heads, d=64
    // Qwen2.5-0.5B (dense): 48 layers, 14 heads, d=64
    // IBM Granite-3.1-1B-A400M (MoE): shape derived from HF config
    int seq_len = 512;
    int iters = 20, warmup = 5;
    std::string only_family = "";
    std::string csv_out = "";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing arg for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if      (a == "--seq")    seq_len = std::stoi(next());
        else if (a == "--iters")  iters = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--family") only_family = next();
        else if (a == "--out")    csv_out = next();
        else if (a == "--help" || a == "-h") {
            std::cout << "usage: bench_pipeline [--family dense|hybrid|moe] "
                      << "[--seq 512] [--iters 20] [--warmup 5] [--out file.csv]\n"
                      << "  runs the per-family SDK pipeline benchmark.\n";
            return 0;
        } else {
            std::cerr << "unknown arg: " << a << "\n"; return 1;
        }
    }

    std::vector<Event> all;

    // Dense shape — Qwen2.5-0.5B-ish
    ModelShape dense_shape{
        .n_layers = 24, .n_heads = 14, .n_kv_heads = 2,
        .head_dim = 64, .hidden_dim = 896, .seq_len = seq_len,
    };
    // Hybrid shape — Falcon-H1-0.5B-ish (parallel 1:1)
    ModelShape hybrid_shape{
        .n_layers = 36, .n_heads = 8, .n_kv_heads = 2,
        .head_dim = 128, .hidden_dim = 1024, .seq_len = seq_len,
    };
    // MoE shape — IBM Granite-3.1-1B-A400M-ish
    ModelShape moe_shape{
        .n_layers = 40, .n_heads = 24, .n_kv_heads = 8,
        .head_dim = 64, .hidden_dim = 1536, .seq_len = seq_len,
    };
    const int moe_n_experts = 32;

    if (only_family.empty() || only_family == "dense") {
        auto ev = bench_dense(dense_shape, iters, warmup);
        all.insert(all.end(), ev.begin(), ev.end());
    }
    if (only_family.empty() || only_family == "hybrid") {
        auto ev = bench_hybrid(hybrid_shape, iters, warmup);
        all.insert(all.end(), ev.begin(), ev.end());
    }
    if (only_family.empty() || only_family == "moe") {
        auto ev = bench_moe(moe_shape, moe_n_experts, iters, warmup);
        all.insert(all.end(), ev.begin(), ev.end());
    }

    emit_pretty(std::cout, all);
    if (!csv_out.empty()) {
        std::ofstream ofs(csv_out);
        if (!ofs) { std::cerr << "cannot open " << csv_out << "\n"; return 2; }
        emit_csv(ofs, all);
        std::cout << "\nwrote csv → " << csv_out << " (" << all.size() << " events)\n";
    }
    return 0;
}

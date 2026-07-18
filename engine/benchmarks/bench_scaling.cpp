// bench_scaling.cpp — measure every OpusEdge primitive across a
// sequence-length grid, report ms/iteration and inferred scaling.
//
// Usage
//   bench_scaling                        # default grid, csv on stdout
//   bench_scaling --seq 128,512,2048,8192
//   bench_scaling --iters 30 --warmup 5
//   bench_scaling --out /tmp/scaling.csv
//
// Each row of the CSV: primitive,seq_len,mean_ms,std_ms,ns_per_token,
//                      theoretical_complexity,notes

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <iomanip>

#include "opusedge/core/signal.h"
#include "opusedge/primitives/selkv.h"
#include "opusedge/primitives/smsa.h"
#include "opusedge/primitives/delta_ar.h"
#include "opusedge/primitives/delta_rank.h"
#include "opusedge/primitives/head_gate.h"
#include "opusedge/primitives/state_compress.h"
#include "opusedge/primitives/composite.h"
#include "opusedge/primitives/gakv.h"
#include "opusedge/primitives/ndpa.h"
#include "opusedge/primitives/mpsr.h"
#include "opusedge/primitives/ebar.h"
#include "opusedge/primitives/ssr.h"
#include "opusedge/primitives/ipss.h"
#include "opusedge/primitives/cal.h"
#include "opusedge/primitives/rcal.h"
#include "opusedge/primitives/pareto.h"

using namespace opusedge;
using clk = std::chrono::high_resolution_clock;

// ── Row ──────────────────────────────────────────────────────────────
struct Row {
    std::string primitive;
    int seq_len;
    double mean_ms;
    double std_ms;
    double ns_per_token;
    std::string complexity;   // "O(S)" / "O(S log S)" / "O(S²)" / …
    std::string notes;
};

// forward declaration for the += concatenation operator
static std::vector<Row>& operator+=(std::vector<Row>& a, std::vector<Row>&& b);

// ── time a nullary lambda across iters, return (mean_ms, std_ms) ────
template <class F>
std::pair<double, double> time_it(F&& f, int iters, int warmup) {
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

// ── random Δ / matrices ─────────────────────────────────────────────
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

// ── theoretical complexity classifier via log-log slope ─────────────
static std::string classify_scaling(const std::vector<int>& S,
                                     const std::vector<double>& t) {
    if (S.size() < 3) return "n/a";
    // linear regression of log(t) on log(S)
    double n = S.size();
    double sx = 0, sy = 0, sxy = 0, sxx = 0;
    for (size_t i = 0; i < S.size(); ++i) {
        double x = std::log(static_cast<double>(S[i]));
        double y = std::log(std::max(t[i], 1e-9));
        sx += x; sy += y; sxy += x * y; sxx += x * x;
    }
    double slope = (n * sxy - sx * sy) / std::max(n * sxx - sx * sx, 1e-12);
    if (slope < 0.4) return "O(1)";
    if (slope < 1.3) return "O(S)";
    if (slope < 1.6) return "O(S log S)";
    if (slope < 2.4) return "O(S^2)";
    return "O(>S^2)";
}

// ── benchmark one primitive across the seq_len grid ──────────────────
using PrimFn = std::function<void(int seq_len)>;
static std::vector<Row> bench_primitive(const std::string& name,
                                        const std::string& theory,
                                        const PrimFn& fn,
                                        const std::vector<int>& seq_lens,
                                        int iters, int warmup) {
    std::vector<Row> rows;
    std::vector<double> means;
    for (int S : seq_lens) {
        auto [mean, sd] = time_it([&] { fn(S); }, iters, warmup);
        double npt = (mean * 1e6) / std::max(S, 1);
        rows.push_back({name, S, mean, sd, npt, theory, ""});
        means.push_back(mean);
    }
    std::string measured = classify_scaling(seq_lens, means);
    for (auto& r : rows) r.notes = "measured=" + measured;
    return rows;
}

// ── CSV emit ─────────────────────────────────────────────────────────
static void emit_csv(std::ostream& os, const std::vector<Row>& rows) {
    os << "primitive,seq_len,mean_ms,std_ms,ns_per_token,theoretical,notes\n";
    os << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        // Quote every string field — primitive names contain commas like "[w=64,D=64]".
        os << "\"" << r.primitive << "\","
           << r.seq_len << ","
           << r.mean_ms << "," << r.std_ms << ","
           << r.ns_per_token << ","
           << "\"" << r.complexity << "\","
           << "\"" << r.notes << "\"\n";
    }
}

static void emit_pretty(std::ostream& os, const std::vector<Row>& rows) {
    os << std::left << std::setw(28) << "primitive"
       << std::right << std::setw(8) << "seq"
       << std::setw(12) << "ms"
       << std::setw(12) << "±"
       << std::setw(14) << "ns/token"
       << "   theory   measured\n";
    os << std::string(90, '-') << "\n";
    os << std::fixed << std::setprecision(3);
    for (const auto& r : rows) {
        os << std::left << std::setw(28) << r.primitive
           << std::right << std::setw(8) << r.seq_len
           << std::setw(12) << r.mean_ms
           << std::setw(12) << r.std_ms
           << std::setw(14) << r.ns_per_token
           << "   " << std::setw(8) << r.complexity
           << "  " << r.notes << "\n";
    }
}

// ── main ─────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::vector<int> seqs = {128, 256, 512, 1024, 2048, 4096};
    int iters = 20, warmup = 5;
    bool skip_quadratic = false;   // skip full_causal_attn + ebdar past small S
    int quadratic_cutoff = 8192;   // if skip_quadratic, drop O(S²) primitives above this
    std::string csv_out;
    std::string only = "";         // substring filter on primitive name

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (a == "--seq") {
            seqs.clear();
            std::stringstream ss(next()); std::string tok;
            while (std::getline(ss, tok, ',')) seqs.push_back(std::stoi(tok));
        } else if (a == "--iters")  iters  = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--out")    csv_out = next();
        else if (a == "--only")   only = next();
        else if (a == "--skip-quadratic") skip_quadratic = true;
        else if (a == "--quadratic-cutoff") quadratic_cutoff = std::stoi(next());
        else if (a == "--help" || a == "-h") {
            std::cout << "usage: bench_scaling [--seq 128,512,...] "
                      << "[--iters 20] [--warmup 5] [--out file.csv]\n"
                      << "                     [--skip-quadratic] [--quadratic-cutoff 8192]\n"
                      << "                     [--only <substring>]\n\n"
                      << "  --skip-quadratic     drop primitives whose theoretical is O(S²)+\n"
                      << "                       from grid points above --quadratic-cutoff\n"
                      << "                       (needed for 8K–65K sweeps, else full_causal_attn\n"
                      << "                       and ebdar burn hours per iteration)\n"
                      << "  --only pattern       only benchmark primitives whose name matches\n";
            return 0;
        } else {
            std::cerr << "unknown arg: " << a << "\n"; return 1;
        }
    }

    // A helper: cull grid points from a rows vector based on primitive complexity.
    auto cull_quadratic = [&](std::vector<Row>& rows) {
        if (!skip_quadratic) return;
        rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const Row& r) {
            bool quad = (r.complexity.find("S^2") != std::string::npos)
                     || (r.complexity.find("S²") != std::string::npos);
            return quad && r.seq_len > quadratic_cutoff;
        }), rows.end());
    };
    auto cull_only = [&](std::vector<Row>& rows) {
        if (only.empty()) return;
        rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const Row& r) {
            return r.primitive.find(only) == std::string::npos;
        }), rows.end());
    };
    // For quadratic primitives, when skip_quadratic is on, only run them at
    // grid points ≤ quadratic_cutoff. Achieved by clipping the seq vector
    // when calling bench_primitive for a quadratic-labeled primitive.
    auto quad_seqs = [&]() {
        std::vector<int> s;
        for (int S : seqs) if (S <= quadratic_cutoff) s.push_back(S);
        return s;
    };

    const int n_heads = 32, head_dim = 64, hidden = n_heads * head_dim;

    std::vector<Row> all;

    // ── SelKV ── O(S log S) — sort by δ
    all += bench_primitive("SelKV::evict[r=0.875]", "O(S log S)",
        [&](int S) {
            auto d = random_delta(S);
            auto r = SelKV::evict(d, 0.875, S);
            (void)r;
        }, seqs, iters, warmup);

    // ── SMSA analytical ── O(1) — cheap
    all += bench_primitive("SMSA::analyze", "O(1)",
        [&](int S) {
            SMSA smsa(64);
            auto r = smsa.analyze(S);
            (void)r;
        }, seqs, iters, warmup);

    // ── SMSA mask build ── O(S · w)
    all += bench_primitive("SMSA::attention_mask[w=64]", "O(S·w)",
        [&](int S) {
            SMSA smsa(64);
            auto m = smsa.attention_mask(S);
            (void)m;
        }, seqs, iters, warmup);

    // ── SMSA forward with real Q/K/V ── O(S · w · D)
    all += bench_primitive("SMSA::forward[w=64,D=64]", "O(S·w·D)",
        [&](int S) {
            auto Q = random_mat(S, head_dim, 1);
            auto K = random_mat(S, head_dim, 2);
            auto V = random_mat(S, head_dim, 3);
            SMSA smsa(64);
            auto O = smsa.forward(Q, K, V);
            (void)O;
        }, seqs, iters, warmup);

    // ── Full causal attention baseline ── O(S² · D)
    //   Skipped past --quadratic-cutoff when --skip-quadratic is on, because
    //   at S=65K a single iteration is ~9 minutes; the label is preserved so
    //   the docs still show it's Θ(S²·D) — the label lies below the grid.
    {
        auto quad_grid = skip_quadratic ? quad_seqs() : seqs;
        if (!quad_grid.empty()) {
            all += bench_primitive("full_causal_attn[D=64]", "O(S^2·D)",
                [&](int S) {
                    auto Q = random_mat(S, head_dim, 1);
                    auto K = random_mat(S, head_dim, 2);
                    auto V = random_mat(S, head_dim, 3);
                    auto O = SMSA::forward_full_causal(Q, K, V);
                    (void)O;
                }, quad_grid, iters, warmup);
        }
    }

    // ── SSD-mask build ── O(S² worst case, but early-stops)
    all += bench_primitive("SMSA::ssd_mask[τ=1e-3]", "O(S·w_eff)",
        [&](int S) {
            auto d = random_delta(S);
            SMSAConfig cfg; cfg.ssd_threshold = 1e-3;
            SMSA smsa(cfg);
            auto m = smsa.ssd_mask(d);
            (void)m;
        }, seqs, iters, warmup);

    // ── Delta-AR indices ── O(S log K)
    all += bench_primitive("delta_ar_indices[k=64]", "O(S log K)",
        [&](int S) {
            auto d = random_delta(S);
            auto idx = build_delta_ar_indices(d, 64);
            (void)idx;
        }, seqs, iters, warmup);

    // ── Delta-AR mask build ── O(S · K)
    all += bench_primitive("delta_ar_mask[k=64]", "O(S·K)",
        [&](int S) {
            auto d = random_delta(S);
            auto m = build_delta_ar_mask(d, 64);
            (void)m;
        }, seqs, iters, warmup);

    // ── Delta-AR sparse attention forward ── O(S · K · D)
    all += bench_primitive("delta_ar_sparse_attn[k=64,D=64]", "O(S·K·D)",
        [&](int S) {
            auto d = random_delta(S);
            auto Q = random_mat(S, head_dim, 1);
            auto K = random_mat(S, head_dim, 2);
            auto V = random_mat(S, head_dim, 3);
            auto idx_f = build_delta_ar_indices(d, 64);
            MatrixXi idx = idx_f.cast<int>();
            auto out = sparse_attention(Q, K, V, idx);
            (void)out;
        }, seqs, iters, warmup);

    // ── HeadGate::analyze ── O(S)
    all += bench_primitive("HeadGate::analyze", "O(S)",
        [&](int S) {
            HeadGateConfig cfg; cfg.n_heads = n_heads;
            HeadGate hg(cfg);
            auto d = random_delta(S);
            auto r = hg.analyze(d);
            (void)r;
        }, seqs, iters, warmup);

    // ── StateCompress::compress_channels ── O(D log D)
    all += bench_primitive("StateCompress::compress[D=hidden]", "O(D log D)",
        [&](int S) {
            (void)S;
            auto state = random_mat(1, hidden, 4).row(0).transpose();
            StateCompress sc;
            auto out = sc.compress_channels(state, 0.05);
            (void)out;
        }, seqs, iters, warmup);

    // ── GAKV::analyze ── O(S)
    all += bench_primitive("GAKV::analyze", "O(S)",
        [&](int S) {
            auto d = random_delta(S);
            std::vector<double> dv(d.data(), d.data() + S);
            std::vector<double> ir(S, 0.5);
            GAKV g;
            auto r = g.analyze(dv, ir);
            (void)r;
        }, seqs, iters, warmup);

    // ── Composite::analyze ── O(1)
    all += bench_primitive("Composite::analyze", "O(1)",
        [&](int S) {
            CompositeConfig cfg;
            Composite comp(cfg);
            auto r = comp.analyze(S, n_heads, head_dim, hidden);
            (void)r;
        }, seqs, iters, warmup);

    // ── MPSR::project ── O(S log S)
    all += bench_primitive("MPSR::project", "O(S log S)",
        [&](int S) {
            auto d = random_delta(S);
            std::vector<double> ev(d.data(), d.data() + S);
            MPSR m;
            auto r = m.project(ev, hidden);
            (void)r;
        }, seqs, iters, warmup);

    // ── EBAR::analyze ── O(S)
    all += bench_primitive("EBAR::analyze", "O(S)",
        [&](int S) {
            std::vector<double> h(S, 0.5);
            EBAR e;
            auto r = e.analyze(h);
            (void)r;
        }, seqs, iters, warmup);

    // ── SSR::analyze ── O(D log D) on singular value list
    all += bench_primitive("SSR::analyze[D=hidden]", "O(D log D)",
        [&](int S) {
            (void)S;
            std::vector<double> sv(hidden);
            for (int i = 0; i < hidden; ++i) sv[i] = 1.0 / (i + 1);
            SSR s;
            auto r = s.analyze(sv, 0.5);
            (void)r;
        }, seqs, iters, warmup);

    // ── IPSS::analyze_simple ── O(H)
    all += bench_primitive("IPSS::analyze[H=32]", "O(H)",
        [&](int S) {
            (void)S;
            std::vector<double> variances(n_heads, 0.3);
            IPSS ipss;
            auto r = ipss.analyze_simple(variances);
            (void)r;
        }, seqs, iters, warmup);

    // ── Pareto::sweep ── O(|grid|)
    all += bench_primitive("Pareto::sweep", "O(|grid|)",
        [&](int S) {
            auto d = random_delta(S);
            auto pts = ParetoFrontier::sweep(d, S);
            (void)pts;
        }, seqs, iters, warmup);

    // ── SignalExtractor::spearman ── O(S log S)
    all += bench_primitive("SignalExtractor::spearman", "O(S log S)",
        [&](int S) {
            auto a = random_delta(S, 1);
            auto b = random_delta(S, 2);
            double r = SignalExtractor::compute_spearman(a, b);
            (void)r;
        }, seqs, iters, warmup);

    // ── Signal utilities: normalize, sact_transmute ─────────────────
    all += bench_primitive("SignalExtractor::normalize", "O(S)",
        [&](int S) {
            auto d = random_delta(S);
            auto r = SignalExtractor::normalize_importance(d);
            (void)r;
        }, seqs, iters, warmup);
    all += bench_primitive("SignalExtractor::sact_transmute", "O(S)",
        [&](int S) {
            auto d = random_delta(S);
            auto r = SignalExtractor::sact_transmute(d, 0.15);
            (void)r;
        }, seqs, iters, warmup);

    // ── CAL / R-CAL ── O(1)
    all += bench_primitive("CAL::classify", "O(1)",
        [&](int S) { (void)S; CAL c; auto r = c.classify("math"); (void)r; },
        seqs, iters, warmup);
    all += bench_primitive("RCAL::classify", "O(1)",
        [&](int S) { (void)S; RCAL c; auto r = c.classify("math", 0.5); (void)r; },
        seqs, iters, warmup);

    // ── NDPA::analyze ── O(S)
    all += bench_primitive("NDPA::analyze", "O(S)",
        [&](int S) {
            auto d = random_delta(S, 1);
            auto a = random_delta(S, 2);
            std::vector<double> dv(d.data(), d.data() + S);
            std::vector<double> av(a.data(), a.data() + S);
            NDPA ndpa;
            auto r = ndpa.analyze(dv, av);
            (void)r;
        }, seqs, iters, warmup);

    // ── DeltaRank::ssr ── O(D^3) (SVD dominated); use fixed D
    all += bench_primitive("DeltaRank::ssr[D=64]", "O(D^3)",
        [&](int S) {
            (void)S;
            auto W = random_mat(head_dim, head_dim, 5);
            auto out = DeltaRank::ssr(W, 0.5, 2.0);
            (void)out;
        }, seqs, iters, warmup);

    // ── ebdar ── O(S²) — same cutoff logic as full_causal_attn.
    {
        auto quad_grid = skip_quadratic ? quad_seqs() : seqs;
        if (!quad_grid.empty()) {
            all += bench_primitive("ebdar[β=0.85]", "O(S^2)",
                [&](int S) {
                    auto d = random_delta(S);
                    auto mask = build_delta_ar_mask(d, 64);
                    auto [E, O] = ebdar(d, mask, 0.85);
                    (void)E; (void)O;
                }, quad_grid, iters, warmup);
        }
    }

    cull_quadratic(all);
    cull_only(all);

    // ── output ────────────────────────────────────────────────────────
    emit_pretty(std::cout, all);
    if (!csv_out.empty()) {
        std::ofstream ofs(csv_out);
        if (!ofs) { std::cerr << "cannot open " << csv_out << "\n"; return 2; }
        emit_csv(ofs, all);
        std::cout << "\nwrote csv → " << csv_out << " (" << all.size() << " rows)\n";
    }
    return 0;
}

// ── vector concat helper (rows += rows) ──────────────────────────────
static std::vector<Row>& operator+=(std::vector<Row>& a, std::vector<Row>&& b) {
    a.insert(a.end(),
             std::make_move_iterator(b.begin()),
             std::make_move_iterator(b.end()));
    return a;
}

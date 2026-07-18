#include "opusedge/core/signal.h"
#include "opusedge/primitives/selkv.h"
#include "opusedge/primitives/smsa.h"
#include "opusedge/primitives/delta_ar.h"
#include "opusedge/primitives/delta_rank.h"
#include "opusedge/primitives/head_gate.h"
#include "opusedge/primitives/state_compress.h"
#include "opusedge/primitives/composite.h"
#include "opusedge/primitives/pareto.h"
#include "opusedge/primitives/cal.h"
#include "opusedge/primitives/rcal.h"
#include "opusedge/primitives/gakv.h"
#include "opusedge/primitives/ndpa.h"
#include "opusedge/primitives/mpsr.h"
#include "opusedge/primitives/ebar.h"
#include "opusedge/primitives/ssr.h"
#include "opusedge/primitives/ipss.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>

using namespace opusedge;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "═══ OpusEdge C++  All 26 Primitives Demo ═══\n\n";

    // ── Shared test data ────────────────────────────────────────
    std::vector<Float> delta_vec  = {0.3, 0.6, 0.8, 0.2, 0.1, 0.5, 0.7, 0.4, 0.9, 0.05, 0.45, 0.65};
    std::vector<Float> ir_scores  = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.85, 0.55, 0.35};
    std::vector<Float> attn_vec   = {0.2, 0.7, 0.9, 0.1, 0.15, 0.4, 0.6, 0.3, 0.8, 0.05, 0.5, 0.45};
    std::vector<Float> log_probs  = {-0.5, -1.2, -0.3, -2.1, -0.8, -1.5};
    std::vector<Float> singular_values = {5.0, 3.0, 1.5, 0.8, 0.3, 0.1};
    std::vector<Float> head_vars  = {0.9, 0.15, 0.6, 0.05, 0.8, 0.2, 0.7, 0.1};
    std::vector<Float> state_dim  = {1.2, 0.8, 0.5, 0.2, 0.05, 0.01};
    int seq_len = 2048;
    int n_heads = 32;

    // ── 1. proxy_delta ──────────────────────────────────────────
    int n_tokens = 12, d_model = 8, n_layers = 4;
    std::vector<MatrixXf> hidden_states(n_layers, MatrixXf(n_tokens, d_model));
    for (int l = 0; l < n_layers; ++l)
        for (int t = 0; t < n_tokens; ++t)
            for (int d = 0; d < d_model; ++d)
                hidden_states[l](t, d) = std::sin(Float(t) * 0.5 + l) * (1.0 + d * 0.1);
    VectorXf proxy_delta_out = SignalExtractor::proxy_delta(hidden_states);
    std::cout << "1. proxy_delta        :";
    for (int i = 0; i < proxy_delta_out.size(); ++i)
        std::cout << " " << proxy_delta_out(i);
    std::cout << "\n";

    // ── 2. spearman ─────────────────────────────────────────────
    VectorXf a = VectorXf::Map(delta_vec.data(), delta_vec.size());
    VectorXf b = VectorXf::Map(attn_vec.data(), attn_vec.size());
    Float rho = SignalExtractor::compute_spearman(a, b);
    std::cout << "2. spearman           : p = " << rho << "\n";

    // ── 3. normalize ────────────────────────────────────────────
    VectorXf normed = SignalExtractor::normalize_importance(a);
    std::cout << "3. normalize          :";
    for (int i = 0; i < normed.size(); ++i)
        std::cout << " " << normed(i);
    std::cout << "\n";

    // ── 4. sact_transmute ───────────────────────────────────────
    VectorXf sact = SignalExtractor::sact_transmute(a, 0.3);
    std::cout << "4. sact_transmute     :";
    for (int i = 0; i < sact.size(); ++i)
        std::cout << " " << sact(i);
    std::cout << "\n";

    // ── 5. delta_ar_indices ─────────────────────────────────────
    auto delta_ar_res = measure_delta_ar(a, 4);
    std::cout << "5. delta_ar_indices   : K=" << delta_ar_res.top_k
              << " attended=" << delta_ar_res.tokens_attended
              << " flop_red=" << (delta_ar_res.flop_reduction * 100) << "%\n";

    // ── 6. delta_ar_flops ───────────────────────────────────────
    auto flops = delta_ar_flops(seq_len, delta_ar_res.top_k, 64);
    std::cout << "6. delta_ar_flops     : baseline=" << flops.baseline
              << " routed=" << flops.routed << "\n";

    // ── 7. ebdar ────────────────────────────────────────────────
    Float ebdar_mean = delta_ar_res.ebdar_mean_reservoir;
    std::cout << "7. ebdar              : mean_reservoir=" << ebdar_mean << "\n";

    // ── 8. selkv_evict ──────────────────────────────────────────
    auto selkv_r = SelKV::evict(a, 0.5f, 12);
    std::cout << "8. selkv_evict        : retained=" << selkv_r.retained_indices.size()
              << " evicted=" << selkv_r.evicted_indices.size()
              << " mem_savings=" << (selkv_r.memory_savings * 100) << "%\n";

    // ── 9. selkv_quality_ratio ──────────────────────────────────
    Float qr = SelKV::quality_ratio(a, 0.5f);
    std::cout << "9. selkv_quality_ratio: " << qr << "\n";

    // ── 10. smsa_analyze ────────────────────────────────────────
    SMSA smsa(256);
    auto smsa_r = smsa.analyze(seq_len);
    std::cout << "10. smsa_analyze      : speedup=" << smsa_r.speedup << "x"
              << " mem=" << (smsa_r.memory_savings * 100) << "%\n";

    // ── 11. head_active ─────────────────────────────────────────
    HeadGateConfig hcfg;
    hcfg.n_heads = n_heads;
    HeadGate hg(hcfg);
    auto hg_r = hg.analyze(a);
    std::cout << "11. head_active       : active=" << hg_r.active_heads
              << "/" << n_heads << " flop_red=" << hg_r.flop_reduction_pct << "%\n";

    // ── 12. head_flop_reduction ─────────────────────────────────
    std::cout << "12. head_flop_reduction: " << hg_r.flop_reduction_pct << "%\n";

    // ── 13. state_keep_ratio ────────────────────────────────────
    StateCompress sc;
    Float keep = sc.keep_ratio(proxy_delta_out(0));
    std::cout << "13. state_keep_ratio  : " << keep << "\n";

    // ── 14. composite_analyze ───────────────────────────────────
    CompositeConfig cfg;
    Composite comp(cfg);
    auto comp_r = comp.analyze(seq_len, n_heads, 64, 512);
    std::cout << "14. composite_analyze : flop_red=" << comp_r.flop_reduction_pct
              << "% mem=" << comp_r.memory_savings_pct << "%\n";

    // ── 15. cal_classify ────────────────────────────────────────
    CAL cal;
    auto cal_r = cal.classify("math");
    std::cout << "15. cal_classify      : tier=" << cal_r.name
              << " rigidity=" << cal_r.rigidity
              << " evict_cap=" << cal_r.eviction_cap << "\n";

    // ── 16. cal_rigidity ────────────────────────────────────────
    Float rigidity = cal.rigidity_of("summarize");
    std::cout << "16. cal_rigidity      : summarize=" << rigidity << "\n";

    // ── 17. rcal_classify ───────────────────────────────────────
    RCAL rcal(0.1, 0.9);
    auto rcal_r = rcal.classify("math", 0.6);
    std::cout << "17. rcal_classify     : tier=" << rcal_r.tier.name
              << " thresh=" << rcal_r.effective_threshold
              << " conf=" << rcal_r.confidence << "\n";

    // ── 18. rcal_modulate ───────────────────────────────────────
    Float mod_thresh = rcal.modulate_threshold("math", 0.6);
    std::cout << "18. rcal_modulate     : " << mod_thresh << "\n";

    // ── 19. rcal_eviction_cap ───────────────────────────────────
    Float ev_cap = rcal.modulated_eviction_cap("math");
    std::cout << "19. rcal_eviction_cap : " << ev_cap << "\n";

    // ── 20. gakv_analyze ────────────────────────────────────────
    GAKV gakv({0.5, 0.5, 0.3, 0.3});
    auto gakv_r = gakv.analyze(delta_vec, ir_scores);
    std::cout << "20. gakv_analyze      : n_evicted=" << gakv_r.n_evicted
              << " composite=[";
    for (size_t i = 0; i < gakv_r.composite_scores.size() && i < 6; ++i)
        std::cout << gakv_r.composite_scores[i] << " ";
    std::cout << "...]\n";

    // ── 21. rgakv_analyze ───────────────────────────────────────
    auto rgakv_r = gakv.rgakv_analyze(delta_vec, ir_scores, 0.2);
    std::cout << "21. rgakv_analyze     : n_evicted=" << rgakv_r.n_evicted << "\n";

    // ── 22. ndpa_rectify ────────────────────────────────────────
    NDPA ndpa;
    auto ndpa_r = ndpa.analyze(delta_vec, attn_vec);
    std::cout << "22. ndpa_rectify      : gamma=" << ndpa_r.gamma
              << " rho_raw=" << ndpa_r.rho_raw
              << " active=" << (ndpa_r.active ? "yes" : "no") << "\n";

    // ── 23. mpsr_project ────────────────────────────────────────
    MPSR mpsr({1.0, 0.25});
    auto mpsr_r = mpsr.project(state_dim, 6.0);
    std::cout << "23. mpsr_project      : compression=" << mpsr_r.compression_ratio
              << " energy=" << mpsr_r.energy_retained << "\n";

    // ── 24. mpsr_sact ───────────────────────────────────────────
    auto mpsr_s_r = mpsr.project_sact(state_dim, 6.0, 0.3);
    std::cout << "24. mpsr_sact         : compression=" << mpsr_s_r.compression_ratio
              << " energy=" << mpsr_s_r.energy_retained << "\n";

    // ── 25. ebar_analyze ────────────────────────────────────────
    EBAR ebar({0.3, 1.0, 0.85, 0.1});
    auto ebar_r = ebar.analyze(log_probs);
    std::cout << "25. ebar_analyze      : savings=" << (ebar_r.total_compute_savings * 100) << "%\n";

    // ── 26. ebar_entropy ────────────────────────────────────────
    auto entropies = EBAR::compute_shannon_entropy(log_probs);
    std::cout << "26. ebar_entropy      :";
    for (auto h : entropies) std::cout << " " << h;
    std::cout << "\n";

    // ── 27. ssr_analyze ─────────────────────────────────────────
    SSR ssr({2.0, 1.5, 0.05});
    auto ssr_r = ssr.analyze(singular_values, 0.6);
    std::cout << "27. ssr_analyze       : preserved=" << ssr_r.preserved_count
              << "/" << singular_values.size()
              << " fraction=" << ssr_r.preserved_fraction
              << " compression=" << ssr_r.compression_ratio << "\n";

    // ── 28. ssr_casp ────────────────────────────────────────────
    auto casp_r = ssr.casp_analyze(singular_values, 0.8);
    std::cout << "28. ssr_casp          : preserved=" << casp_r.preserved_count
              << " compression=" << casp_r.compression_ratio << "\n";

    // ── 29. ipss_analyze ────────────────────────────────────────
    IPSS ipss({0.3, 0.15, 4});
    auto ipss_r = ipss.analyze_simple(head_vars);
    std::cout << "29. ipss_analyze      : n_active=" << ipss_r.n_active
              << "/" << head_vars.size()
              << " flop_red=" << ipss_r.flop_reduction << "\n";

    // ── 30. pareto_sweep ───────────────────────────────────────
    auto pareto_pts = ParetoFrontier::sweep(a, seq_len);
    auto frontier = ParetoFrontier::pareto_optimal(pareto_pts);
    auto knee = ParetoFrontier::knee_point(frontier);
    std::cout << "30. pareto_sweep      : " << pareto_pts.size() << " points, "
              << frontier.size() << " frontier, knee @ flops="
              << knee.flops << " ppl=" << knee.ppl << "\n";

    // ── delta_rank (bonus) ──────────────────────────────────────
    DeltaRank dr;
    int r = dr.get_rank_for_delta(0.3);
    MatrixXf W_ssr = DeltaRank::ssr(MatrixXf::Random(12, 64), 0.5);
    std::cout << "B1. delta_rank        : rank_for_delta=" << r
              << " ssr_frobenius=" << W_ssr.norm() << "\n";

    // ── router_entropy (bonus) ──────────────────────────────────
    VectorXf gates(4);
    gates << 0.1, 0.4, 0.3, 0.2;
    Float H = SignalExtractor::router_entropy(gates);
    std::cout << "B2. router_entropy    : H=" << H << "\n";

    std::cout << "\n All primitives executed successfully\n";
    return 0;
}

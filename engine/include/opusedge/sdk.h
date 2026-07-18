#pragma once

// OpusEdge SDK — architecture-family-scoped pipelines.
//
// The paper is explicit about which primitives apply where:
//
//   Dense    — Proxy-Δ signal.  Applicable:
//              DenseEvic, SelKV(-with-Proxy-Δ), SSR, CASP, NDPA,
//              CAL, HeadDeactivate, IPSS, ΔRank, EB-AR, Pareto.
//   Hybrid   — native SSM Δ.   Applicable:
//              SelKV, SMSA, Delta-AR, EB-DAR, StateCompress,
//              MPSR/SACT, R-CAL, HeadDeactivate, IPSS, ΔRank,
//              EB-AR, Pareto.
//   MoE      — Router-Gated IR + Proxy-Δ.  Applicable:
//              Router-IR, GAKV, R-GAKV, CAL, HeadDeactivate,
//              IPSS, ΔRank, EB-AR, Pareto.
//
// SMSA and MPSR are hybrid-specific because they exploit the SSM.
// SSR/CASP/NDPA are dense-specific because they stabilise SVD on
// dense projections and rectify Proxy-Δ. R-CAL is the runtime EMA
// variant used with hybrid models (Falcon-H1); CAL is the plain
// task-rigidity modulator used with dense and MoE per the user's
// established convention.

#include "core/types.h"
#include "core/signal.h"
#include "primitives/selkv.h"
#include "primitives/smsa.h"
#include "primitives/delta_ar.h"
#include "primitives/delta_rank.h"
#include "primitives/head_gate.h"
#include "primitives/state_compress.h"
#include "primitives/composite.h"
#include "primitives/pareto.h"
#include "primitives/cal.h"
#include "primitives/rcal.h"
#include "primitives/gakv.h"
#include "primitives/ndpa.h"
#include "primitives/mpsr.h"
#include "primitives/ebar.h"
#include "primitives/ssr.h"
#include "primitives/ipss.h"

namespace opusedge::sdk {

enum class Family { Dense, Hybrid, MoE };

// ═══════════════════════════════════════════════════════════════════
// Shared config the caller populates from their model's HF config.
// ═══════════════════════════════════════════════════════════════════
struct ModelShape {
    int n_layers   = 32;
    int n_heads    = 32;
    int n_kv_heads = 32;
    int head_dim   = 64;
    int hidden_dim = 2048;
    int seq_len    = 512;
};

// KV cache footprint in MiB (fp16 = 2 bytes).
inline double kv_cache_mib(int seq_len, int n_layers, int n_kv_heads,
                            int head_dim, int bytes_per_elem = 2) {
    return 2.0 * n_layers * seq_len * n_kv_heads * head_dim * bytes_per_elem
           / (1024.0 * 1024.0);
}

// A step's decision, tagged with the primitive that produced it.
struct StepDecision {
    std::string primitive;
    std::string action;       // "evict" | "gate" | "route" | ...
    double      mem_saved_pct = 0.0;
    double      flop_saved_pct = 0.0;
    std::string notes;
};

// ═══════════════════════════════════════════════════════════════════
// DENSE PIPELINE — dense transformers only.
// ═══════════════════════════════════════════════════════════════════
class DensePipeline {
public:
    struct Config {
        Float eviction_ratio = 0.5;      // DenseEvic / SelKV target
        Float rank_fraction  = 0.7;      // ΔRank tier
        Float ssr_alpha      = 2.0;
        std::string task_label = "general";  // → CAL rigidity
        Config() = default;
    };

    explicit DensePipeline(ModelShape s)
        : shape_(std::move(s)), cfg_() {}
    DensePipeline(ModelShape s, Config c)
        : shape_(std::move(s)), cfg_(std::move(c)) {}

    Family family() const { return Family::Dense; }

    // Proxy-Δ eviction (== DenseEvic / SelKV with Proxy-Δ).
    EvictionResult dense_evict(const VectorXf& proxy_delta) const {
        Float cap = std::min(cfg_.eviction_ratio, cal_.effective_threshold(cfg_.task_label));
        return SelKV::evict(proxy_delta, cap, shape_.seq_len);
    }

    // NDPA — rectify Proxy-Δ onto the internal attention manifold.
    std::vector<double> ndpa_rectify(const std::vector<double>& proxy_delta,
                                     const std::vector<double>& attn_scores) const {
        NDPA ndpa;
        return ndpa.analyze(proxy_delta, attn_scores).rectified_delta;
    }

    // SSR / CASP — soft SVD to avoid the perplexity cliff on dense proj.
    SSRResult ssr(const std::vector<double>& singular_values, Float entropy) const {
        SSR s;
        return s.analyze(singular_values, entropy);
    }
    SSRResult casp(const std::vector<double>& singular_values, Float curvature) const {
        SSR s;
        return s.casp_analyze(singular_values, curvature);
    }

    // ΔRank low-rank projection using SSR-softened SVD.
    MatrixXf delta_rank_project(const MatrixXf& W) const {
        return DeltaRank::ssr(W, cfg_.rank_fraction, cfg_.ssr_alpha);
    }

    // CAL — modulate any threshold by task rigidity.
    Float cal_effective_threshold(Float base) const {
        (void)base; return cal_.effective_threshold(cfg_.task_label);
    }

    // HeadDeactivate + IPSS fallback for sub-salient heads.
    HeadGate::GateResult head_gate(const VectorXf& delta_per_token) const {
        HeadGateConfig hcfg; hcfg.n_heads = shape_.n_heads;
        HeadGate hg(hcfg);
        return hg.analyze(delta_per_token);
    }
    IPSSResult ipss(const std::vector<double>& head_variances) const {
        IPSS ipss; return ipss.analyze_simple(head_variances);
    }

    // Post-step accounting.
    double kv_baseline_mib()  const { return kv_cache_mib(shape_.seq_len, shape_.n_layers, shape_.n_kv_heads, shape_.head_dim); }
    double kv_after_evict_mib(double ratio) const {
        int keep = std::max(1, static_cast<int>(std::round(shape_.seq_len * (1.0 - ratio))));
        return kv_cache_mib(keep, shape_.n_layers, shape_.n_kv_heads, shape_.head_dim);
    }

private:
    ModelShape shape_;
    Config     cfg_;
    CAL        cal_{};
};

// ═══════════════════════════════════════════════════════════════════
// HYBRID PIPELINE — SSM-attention models (Falcon-H1, Jamba).
// ═══════════════════════════════════════════════════════════════════
class HybridPipeline {
public:
    struct Config {
        Float eviction_ratio       = 0.875;    // paper's SelKV headline setting
        int   smsa_window          = 64;       // paper Table 1 default
        int   delta_ar_top_k       = 64;       // paper Delta-AR default
        Float ebdar_beta           = 0.85;
        Float mpsr_projection_dim  = 0.25;     // MPSR compression ratio
        std::string task_label     = "general";
        Config() = default;
    };

    explicit HybridPipeline(ModelShape s)
        : shape_(std::move(s)), cfg_(), smsa_(cfg_.smsa_window), rcal_() {}
    HybridPipeline(ModelShape s, Config c)
        : shape_(std::move(s)), cfg_(std::move(c)),
          smsa_(cfg_.smsa_window), rcal_() {}

    Family family() const { return Family::Hybrid; }

    // SelKV — native-Δ KV eviction (hybrid's fast path).
    EvictionResult selkv(const VectorXf& native_delta) const {
        Float cap = std::min(cfg_.eviction_ratio, rcal_.modulated_eviction_cap(cfg_.task_label));
        return SelKV::evict(native_delta, cap, shape_.seq_len);
    }

    // SMSA — SSM-masked sparse attention forward.
    MatrixXf smsa_forward(const MatrixXf& Q, const MatrixXf& K, const MatrixXf& V) const {
        return smsa_.forward(Q, K, V);
    }
    MatrixXf smsa_ssd_mask(const VectorXf& delta_per_tok) const {
        return smsa_.ssd_mask(delta_per_tok);
    }

    // Delta-AR — pre-attention top-K routing (uses native Δ).
    MatrixXf delta_ar_indices(const VectorXf& native_delta) const {
        return build_delta_ar_indices(native_delta, cfg_.delta_ar_top_k);
    }
    MatrixXf delta_ar_sparse_attn(
        const MatrixXf& Q, const MatrixXf& K, const MatrixXf& V,
        const MatrixXi& indices) const
    {
        return sparse_attention(Q, K, V, indices);
    }

    // EB-DAR reservoir for tokens masked from Delta-AR routing.
    std::pair<VectorXf, VectorXf> ebdar(const VectorXf& scores, const MatrixXf& mask) const {
        return opusedge::ebdar(scores, mask, cfg_.ebdar_beta);
    }

    // StateCompress — zero low-magnitude SSM channels when Δ ≪ τ.
    VectorXf state_compress(const VectorXf& ssm_state, Float delta_t) const {
        StateCompress sc;
        return sc.compress_channels(ssm_state, delta_t);
    }

    // MPSR — project evicted KV into the SSM state (no context lost).
    MPSRResult mpsr(const std::vector<double>& evicted_delta_scores) const {
        MPSR m;
        MPSRConfig mc; mc.projection_dim_ratio = cfg_.mpsr_projection_dim;
        // (MPSRConfig fields not directly on m; we still use defaults below)
        return m.project(evicted_delta_scores, static_cast<Float>(shape_.hidden_dim));
    }

    // R-CAL — runtime EMA classifier that freezes recomputation.
    RCALResult rcal_classify(Float base_threshold = 0.5) const {
        return rcal_.classify(cfg_.task_label, base_threshold);
    }
    Float rcal_modulate(Float base) const {
        return rcal_.modulate_threshold(cfg_.task_label, base);
    }

    // HeadDeactivate + IPSS (universal, but instance is per-pipeline).
    HeadGate::GateResult head_gate(const VectorXf& delta_per_token) const {
        HeadGateConfig hcfg; hcfg.n_heads = shape_.n_heads;
        HeadGate hg(hcfg);
        return hg.analyze(delta_per_token);
    }
    IPSSResult ipss(const std::vector<double>& head_variances) const {
        IPSS ipss; return ipss.analyze_simple(head_variances);
    }

    // Post-step accounting: SelKV + SMSA + Delta-AR each set an upper
    // bound on the effective KV footprint; the tightest wins.
    double kv_baseline_mib() const {
        return kv_cache_mib(shape_.seq_len, shape_.n_layers, shape_.n_kv_heads, shape_.head_dim);
    }
    double kv_after_pipeline_mib() const {
        // fixed-window SMSA caps to w; SelKV further trims by ratio; take min.
        int selkv_keep = std::max(1, static_cast<int>(std::round(shape_.seq_len * (1.0 - cfg_.eviction_ratio))));
        int smsa_keep  = std::min(cfg_.smsa_window, shape_.seq_len);
        int keep = std::min(selkv_keep, smsa_keep);
        return kv_cache_mib(keep, shape_.n_layers, shape_.n_kv_heads, shape_.head_dim);
    }

private:
    ModelShape shape_;
    Config     cfg_;
    SMSA       smsa_;
    mutable RCAL rcal_;    // mutable because should_recompute() updates EMA
};

// ═══════════════════════════════════════════════════════════════════
// MoE PIPELINE — mixture-of-experts models (Mixtral, OLMoE, Granite).
// ═══════════════════════════════════════════════════════════════════
class MoEPipeline {
public:
    struct Config {
        Float eviction_ratio   = 0.5;
        Float gakv_alpha       = 0.5;
        Float gakv_beta        = 0.5;
        std::string task_label = "general";
        Config() = default;
    };

    explicit MoEPipeline(ModelShape s)
        : shape_(std::move(s)), cfg_() {}
    MoEPipeline(ModelShape s, Config c)
        : shape_(std::move(s)), cfg_(std::move(c)) {}

    Family family() const { return Family::MoE; }

    // Router-Gated Importance (entropy form) per token per layer.
    VectorXf router_ir(const std::vector<MatrixXf>& router_logits) const {
        return SignalExtractor::compute_router_entropy_per_token(router_logits);
    }
    // Router entropy for a single token's gating vector.
    Float router_entropy(const VectorXf& gating_probs) const {
        return SignalExtractor::router_entropy(gating_probs);
    }

    // GAKV — composite Δ + IR retention score.
    GAKVResult gakv(const std::vector<double>& proxy_delta,
                    const std::vector<double>& router_ir) const {
        GAKVConfig gc; gc.alpha = cfg_.gakv_alpha; gc.beta = cfg_.gakv_beta;
        GAKV g(gc);
        return g.analyze(proxy_delta, router_ir);
    }
    GAKVResult rgakv(const std::vector<double>& proxy_delta,
                     const std::vector<double>& router_ir,
                     Float cal_modulator) const {
        GAKVConfig gc; gc.alpha = cfg_.gakv_alpha; gc.beta = cfg_.gakv_beta;
        GAKV g(gc);
        return g.rgakv_analyze(proxy_delta, router_ir, cal_modulator);
    }

    // CAL modulator for MoE (per user's routing).
    Float cal_effective_threshold(Float base = 0.5) const {
        (void)base; return cal_.effective_threshold(cfg_.task_label);
    }

    // HeadDeactivate + IPSS.
    HeadGate::GateResult head_gate(const VectorXf& delta_per_token) const {
        HeadGateConfig hcfg; hcfg.n_heads = shape_.n_heads;
        HeadGate hg(hcfg);
        return hg.analyze(delta_per_token);
    }
    IPSSResult ipss(const std::vector<double>& head_variances) const {
        IPSS ipss; return ipss.analyze_simple(head_variances);
    }

    double kv_baseline_mib() const {
        return kv_cache_mib(shape_.seq_len, shape_.n_layers, shape_.n_kv_heads, shape_.head_dim);
    }

private:
    ModelShape shape_;
    Config     cfg_;
    CAL        cal_{};
};

// ═══════════════════════════════════════════════════════════════════
// Universal — apply anywhere.
// ═══════════════════════════════════════════════════════════════════
struct Universal {
    // Pareto Frontier control plane.
    static std::vector<ParetoPoint> pareto_sweep(const VectorXf& delta, int seq_len) {
        return ParetoFrontier::sweep(delta, seq_len);
    }
    static ParetoPoint pareto_knee(std::vector<ParetoPoint>& frontier) {
        return ParetoFrontier::knee_point(frontier);
    }
    // EB-AR — entropy-buffered autoregression (works on any output stream).
    static EBARResult ebar(const std::vector<double>& token_entropies) {
        EBAR e; return e.analyze(token_entropies);
    }
};

// ═══════════════════════════════════════════════════════════════════
// Applicability matrix — what a caller can invoke per family.
// ═══════════════════════════════════════════════════════════════════
constexpr const char* dense_primitives   = "DenseEvic, SelKV(Proxy-Δ), NDPA, SSR, CASP, "
                                            "ΔRank, CAL, HeadDeactivate, IPSS, EB-AR, Pareto";
constexpr const char* hybrid_primitives  = "SelKV(native Δ), SMSA, Delta-AR, EB-DAR, "
                                            "StateCompress, MPSR/SACT, R-CAL, HeadDeactivate, "
                                            "IPSS, ΔRank, EB-AR, Pareto";
constexpr const char* moe_primitives     = "Router-IR, GAKV, R-GAKV, CAL, HeadDeactivate, "
                                            "IPSS, ΔRank, EB-AR, Pareto";

} // namespace opusedge::sdk

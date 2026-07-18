#pragma once

#include "../core/types.h"
#include <cmath>
#include <algorithm>

namespace opusedge {

struct HeadGateConfig {
    int n_heads = 32;
    int k_low = 4, k_mid = 16, k_high = 24;
    Float theta_low = 0.05, theta_mid = 0.15, theta_high = 0.30;
    Float kappa = 0.3;
};

struct HeadGate {
    HeadGateConfig cfg;

    explicit HeadGate(const HeadGateConfig& c = {}) : cfg(c) {
        // Auto-scale tier budgets when n_heads differs from the paper's
        // reference 32-head model. Keeps the 12.5% / 50% / 75% ratios so
        // flop_reduction never goes negative for smaller models.
        if (cfg.n_heads != 32) {
            cfg.k_low  = std::max(1, cfg.n_heads / 8);            // ≈ 12.5%
            cfg.k_mid  = std::max(cfg.k_low + 1, cfg.n_heads / 2);      // 50%
            cfg.k_high = std::max(cfg.k_mid + 1, (cfg.n_heads * 3) / 4);// 75%
            if (cfg.k_high >= cfg.n_heads) cfg.k_high = cfg.n_heads - 1;
        }
    }

    int active_heads(Float delta_t) const {
        if (delta_t < cfg.theta_low) return cfg.k_low;
        if (delta_t < cfg.theta_mid) return cfg.k_mid;
        if (delta_t < cfg.theta_high) return cfg.k_high;
        return cfg.n_heads;
    }

    Float flop_reduction(Float delta_t) const {
        int active = active_heads(delta_t);
        return 1.0 - static_cast<Float>(active) / cfg.n_heads;
    }

    struct GateResult {
        int active_heads;
        Float flop_reduction_pct;
        int low_count, mid_count, high_count, critical_count;
    };

    GateResult analyze(const VectorXf& deltas) const {
        GateResult r{};
        for (int i = 0; i < deltas.size(); ++i) {
            int k = active_heads(deltas(i));
            if (k <= cfg.k_low) r.low_count++;
            else if (k <= cfg.k_mid) r.mid_count++;
            else if (k <= cfg.k_high) r.high_count++;
            else r.critical_count++;
        }
        const int n = deltas.size();
        const Float weighted = static_cast<Float>(cfg.k_low * r.low_count
            + cfg.k_mid * r.mid_count + cfg.k_high * r.high_count
            + cfg.n_heads * r.critical_count);
        Float mean_active = (n > 0) ? weighted / static_cast<Float>(n) : 0.0;
        r.active_heads = static_cast<int>(std::round(mean_active));
        r.flop_reduction_pct = 100.0 * (1.0 - mean_active / cfg.n_heads);
        return r;
    }

    static Float head_salience(const MatrixXf& activations) {
        int T = activations.rows();
        if (T < 2) return 1.0;
        VectorXf mean = activations.colwise().mean();
        VectorXf var = (activations.rowwise() - mean.transpose())
                       .colwise().squaredNorm() / (T - 1);
        Float s = var.mean();
        return std::min(1.0, std::max(0.0, s));
    }

    static MatrixXf ipss_fallback(const MatrixXf& Q, const MatrixXf& K,
                                  const MatrixXf& V, const VectorXf& K_mean,
                                  Float scale) {
        VectorXf QKbar = Q * K_mean / scale;
        VectorXf V_mean = V.colwise().mean();
        return QKbar * V_mean.transpose();
    }
};

} // namespace opusedge

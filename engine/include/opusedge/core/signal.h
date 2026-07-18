#pragma once

#include "types.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace opusedge {

struct SignalExtractor {
    static VectorXf native_delta(const std::vector<VectorXf>& dt_tokens, int n_tokens) {
        int n_layers = dt_tokens.size();
        VectorXf delta = VectorXf::Zero(n_tokens);
        for (int t = 0; t < n_tokens; ++t) {
            Float sum = 0.0;
            for (int l = 0; l < n_layers; ++l) {
                sum += std::abs(dt_tokens[l](t));
            }
            delta(t) = sum / n_layers;
        }
        return delta;
    }

    static VectorXf proxy_delta(const std::vector<MatrixXf>& hidden_states) {
        int n_layers = hidden_states.size();
        if (n_layers == 0) return {};
        int n_tokens = hidden_states[0].rows();
        int d_model = hidden_states[0].cols();
        VectorXf delta = VectorXf::Zero(n_tokens);
        for (int t = 1; t < n_tokens; ++t) {
            Float sum = 0.0;
            for (int l = 0; l < n_layers; ++l) {
                VectorXf diff = hidden_states[l].row(t) - hidden_states[l].row(t - 1);
                sum += diff.norm();
            }
            delta(t) = sum / (n_layers * std::sqrt(static_cast<Float>(d_model)));
        }
        delta(0) = delta(1);
        return delta;
    }

    static VectorXf normalize_importance(const VectorXf& signal) {
        Float max_val = signal.maxCoeff();
        if (max_val < 1e-12) return VectorXf::Zero(signal.size());
        VectorXf normalized = signal.array() / max_val;
        return normalized;
    }

    static VectorXf compute_attn_scores(const std::vector<MatrixXf>& attn_weights, int n_tokens) {
        VectorXf scores = VectorXf::Zero(n_tokens);
        for (const auto& W : attn_weights) {
            int n_heads = W.rows();
            for (int h = 0; h < n_heads; ++h) {
                VectorXf head_attn = W.row(h);
                scores += head_attn.segment(0, n_tokens);
            }
        }
        scores /= static_cast<Float>(attn_weights.size());
        return scores;
    }

    static Float router_entropy(const VectorXf& gating_probs) {
        Float H = 0.0;
        for (int i = 0; i < gating_probs.size(); ++i) {
            Float p = gating_probs(i);
            if (p > 1e-12)
                H -= p * std::log(p);
        }
        return H;
    }

    // Router-Gated Importance (Eq. router entropy form). For each token, average
    // the per-layer softmax(router_logits) into a single distribution over experts,
    // then compute normalized Shannon entropy H / log(n_experts) ∈ [0, 1].
    // Low entropy ↔ specialized routing (high importance).
    static VectorXf compute_router_entropy_per_token(
        const std::vector<MatrixXf>& router_logits) {
        if (router_logits.empty()) return {};
        const int n_tokens = router_logits[0].rows();
        const int n_experts = router_logits[0].cols();
        const int n_layers = router_logits.size();
        VectorXf entropies = VectorXf::Zero(n_tokens);
        const Float log_E = std::log(static_cast<Float>(std::max(2, n_experts)));

        for (int t = 0; t < n_tokens; ++t) {
            VectorXf p_avg = VectorXf::Zero(n_experts);
            for (int l = 0; l < n_layers; ++l) {
                RowVectorXf logits = router_logits[l].row(t);
                Float mx = logits.maxCoeff();
                VectorXf ex = (logits.array() - mx).exp();
                Float Z = ex.sum();
                if (Z > 1e-30) p_avg += ex / Z;
            }
            p_avg /= static_cast<Float>(n_layers);
            Float H = 0.0;
            for (int e = 0; e < n_experts; ++e) {
                Float p = p_avg(e);
                if (p > 1e-12) H -= p * std::log(p);
            }
            entropies(t) = H / log_E;
        }
        return entropies;
    }

    static VectorXf sact_transmute(const VectorXf& scores, Float residual) {
        int n = scores.size();
        VectorXf result(n);
        Float S_prev = 0.0;
        for (int t = 0; t < n; ++t) {
            S_prev = scores(t) * (1.0 - residual) + residual * S_prev;
            result(t) = S_prev;
        }
        return result;
    }

    // Spearman rank correlation with average-rank tie handling and
    // Pearson-on-ranks fallback (needed for tie correction).
    static Float compute_spearman(const VectorXf& x, const VectorXf& y) {
        const int n = x.size();
        if (n < 3) return 0.0;

        auto avg_ranks = [n](const VectorXf& v) {
            std::vector<int> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b) { return v(a) < v(b); });
            VectorXf r(n);
            int i = 0;
            while (i < n) {
                int j = i;
                while (j + 1 < n && v(idx[j + 1]) == v(idx[i])) ++j;
                Float avg = 0.5 * (i + j) + 1.0;  // 1-based average rank
                for (int k = i; k <= j; ++k) r(idx[k]) = avg;
                i = j + 1;
            }
            return r;
        };

        VectorXf rx = avg_ranks(x);
        VectorXf ry = avg_ranks(y);
        Float mx = rx.mean(), my = ry.mean();
        VectorXf dx = rx.array() - mx;
        VectorXf dy = ry.array() - my;
        Float num = dx.dot(dy);
        Float den = std::sqrt(dx.squaredNorm() * dy.squaredNorm());
        return (den > 1e-12) ? num / den : 0.0;
    }

    static Float ndpa_rectify(Float rho, const VectorXf& proxy_delta,
                              const VectorXf& attn_scores) {
        int n = std::min(proxy_delta.size(), attn_scores.size());
        if (n < 3) return rho;
        Float spearman_raw = compute_spearman(proxy_delta, attn_scores);
        return rho * (1.0 - std::abs(spearman_raw)) + spearman_raw * std::abs(spearman_raw);
    }
};

} // namespace opusedge

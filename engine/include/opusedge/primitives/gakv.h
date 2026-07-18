#pragma once

#include "../core/types.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace opusedge {

struct GAKVConfig {
    Float alpha = 0.5;
    Float beta = 0.5;
    Float theta_delta = 0.3;
    Float theta_ir = 0.3;
};

struct GAKVResult {
    std::vector<Float> composite_scores;
    std::vector<int> evicted_indices;
    std::vector<int> retained_indices;
    int n_evicted;
};

class GAKV {
    GAKVConfig cfg;
public:
    explicit GAKV(const GAKVConfig& c = {}) : cfg(c) {}

    GAKVResult analyze(const std::vector<Float>& delta_scores,
                       const std::vector<Float>& ir_scores) const {
        int n = std::min(delta_scores.size(), ir_scores.size());
        GAKVResult r;
        r.composite_scores.resize(n);

        Float d_max = *std::max_element(delta_scores.begin(), delta_scores.begin() + n);
        Float ir_max = *std::max_element(ir_scores.begin(), ir_scores.begin() + n);
        d_max = std::max(d_max, Float(1e-10));
        ir_max = std::max(ir_max, Float(1e-10));

        for (int i = 0; i < n; ++i) {
            Float d_norm = delta_scores[i] / d_max;
            Float ir_norm = ir_scores[i] / ir_max;
            r.composite_scores[i] = cfg.alpha * d_norm + cfg.beta * ir_norm;
        }

        for (int i = 0; i < n; ++i) {
            Float d_norm = delta_scores[i] / d_max;
            Float ir_norm = ir_scores[i] / ir_max;
            if (d_norm < cfg.theta_delta && ir_norm < cfg.theta_ir)
                r.evicted_indices.push_back(i);
            else
                r.retained_indices.push_back(i);
        }
        r.n_evicted = r.evicted_indices.size();
        return r;
    }

    GAKVResult rgakv_analyze(const std::vector<Float>& delta_scores,
                             const std::vector<Float>& ir_scores,
                             Float cal_modulator) const {
        GAKVConfig adj = cfg;
        adj.theta_delta = cfg.theta_delta * (1.0 + cal_modulator);
        adj.theta_ir = cfg.theta_ir * (1.0 + cal_modulator);
        GAKV relaxed(adj);
        return relaxed.analyze(delta_scores, ir_scores);
    }
};

} // namespace opusedge

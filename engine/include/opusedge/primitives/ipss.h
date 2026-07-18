#pragma once

#include "../core/types.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace opusedge {

struct IPSSConfig {
    Float salience_threshold = 0.3;
    Float ema_alpha = 0.15;
    int warmup_steps = 4;
};

struct IPSSResult {
    std::vector<int> active_heads;
    std::vector<int> smoothed_heads;
    std::vector<Float> salience_values;
    Float flop_reduction;
    int n_active;
};

class IPSS {
    IPSSConfig cfg;
public:
    explicit IPSS(const IPSSConfig& c = {}) : cfg(c) {}

    IPSSResult analyze(const std::vector<std::vector<Float>>& head_activations) const {
        int n_heads = head_activations.size();
        IPSSResult r;
        r.salience_values.resize(n_heads);
        r.active_heads.clear();
        r.smoothed_heads.clear();

        for (int h = 0; h < n_heads; ++h) {
            const auto& acts = head_activations[h];
            if (acts.size() < 2) {
                r.salience_values[h] = 1.0;
                r.active_heads.push_back(h);
                continue;
            }

            Float mean = std::accumulate(acts.begin(), acts.end(), Float(0.0)) / acts.size();
            Float var = 0.0;
            for (Float a : acts) var += (a - mean) * (a - mean);
            var /= acts.size();
            Float salience = std::sqrt(var) / std::max(Float(1e-10), std::abs(mean));
            r.salience_values[h] = salience;

            if (h < cfg.warmup_steps) {
                r.active_heads.push_back(h);
            } else if (salience >= cfg.salience_threshold) {
                r.active_heads.push_back(h);
            } else {
                r.smoothed_heads.push_back(h);
            }
        }

        for (int h : r.smoothed_heads) {
            if (r.active_heads.size() < 1)
                r.active_heads.push_back(h);
        }

        r.n_active = r.active_heads.size();
        r.flop_reduction = 1.0 - static_cast<Float>(r.n_active) / n_heads;
        return r;
    }

    IPSSResult analyze_simple(const std::vector<Float>& head_variances) const {
        std::vector<std::vector<Float>> acts;
        for (Float v : head_variances) acts.push_back({v});
        auto r = analyze(acts);
        for (int h = 0; h < static_cast<int>(head_variances.size()); ++h)
            r.salience_values[h] = head_variances[h];
        return r;
    }
};

} // namespace opusedge

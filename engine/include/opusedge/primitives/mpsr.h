#pragma once

#include "../core/types.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace opusedge {

struct MPSRConfig {
    Float sigmoid_scale = 1.0;
    Float projection_dim_ratio = 0.25;
};

struct MPSRResult {
    std::vector<Float> projected_state;
    Float compression_ratio;
    Float energy_retained;
};

class MPSR {
    MPSRConfig cfg;
public:
    explicit MPSR(const MPSRConfig& c = {}) : cfg(c) {}

    MPSRResult project(const std::vector<Float>& evicted_scores,
                       Float state_dim) const {
        int n = evicted_scores.size();
        int proj_dim = std::max(1, static_cast<int>(state_dim * cfg.projection_dim_ratio));

        MPSRResult r;
        r.compression_ratio = static_cast<Float>(proj_dim) / std::max(Float(1.0), state_dim);

        r.projected_state.resize(std::min(proj_dim, n));
        Float total_energy = 0.0;
        Float retained_energy = 0.0;

        // Sort by importance
        std::vector<std::pair<Float,int>> idx;
        for (int i = 0; i < n; ++i) idx.push_back({evicted_scores[i], i});
        std::sort(idx.begin(), idx.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        for (auto& v : idx) total_energy += std::abs(v.first);

        int k = std::min(static_cast<int>(r.projected_state.size()), n);
        for (int j = 0; j < k; ++j) {
            Float s = evicted_scores[idx[j].second];
            r.projected_state[j] = 1.0 / (1.0 + std::exp(-cfg.sigmoid_scale * s));
            retained_energy += std::abs(s);
        }

        r.energy_retained = total_energy > 0 ? retained_energy / total_energy : 0.0;
        return r;
    }

    MPSRResult project_sact(const std::vector<Float>& evicted_scores,
                            Float state_dim, Float residual) const {
        auto r = project(evicted_scores, state_dim);
        for (auto& v : r.projected_state) v = v * residual;
        return r;
    }
};

} // namespace opusedge

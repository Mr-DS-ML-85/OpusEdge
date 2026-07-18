#pragma once

#include "../core/types.h"
#include <vector>
#include <cmath>

namespace opusedge {

struct EBARConfig {
    Float eta = 0.3;
    Float base_compute = 1.0;
    Float momentum = 0.85;
    Float min_compute = 0.1;
};

struct EBARResult {
    std::vector<Float> compute_per_step;
    std::vector<Float> entropy_buffer;
    Float total_compute_savings;
};

class EBAR {
    EBARConfig cfg;
public:
    explicit EBAR(const EBARConfig& c = {}) : cfg(c) {}

    EBARResult analyze(const std::vector<Float>& token_entropies) const {
        int n = token_entropies.size();
        EBARResult r;
        r.compute_per_step.resize(n);
        r.entropy_buffer.resize(n);

        Float epsilon = 0.0;
        Float total_saved = 0.0;

        for (int t = 0; t < n; ++t) {
            Float H = token_entropies[t];
            Float C_t = cfg.base_compute * (1.0 - cfg.eta * H);

            epsilon = cfg.momentum * epsilon + (1.0 - cfg.momentum) * C_t;
            r.entropy_buffer[t] = epsilon;

            C_t = std::max(cfg.min_compute, std::min(cfg.base_compute, C_t + epsilon));
            r.compute_per_step[t] = C_t;

            total_saved += cfg.base_compute - C_t;
        }

        r.total_compute_savings = total_saved / (n * cfg.base_compute);
        return r;
    }

    static std::vector<Float> compute_shannon_entropy(const std::vector<Float>& log_probs) {
        std::vector<Float> entropies;
        entropies.reserve(log_probs.size());
        for (Float lp : log_probs) {
            Float p = std::exp(std::max(Float(-20.0), std::min(Float(0.0), lp)));
            Float H = (p > 1e-10) ? -p * std::log(p) : 0.0;
            entropies.push_back(H);
        }
        return entropies;
    }
};

} // namespace opusedge

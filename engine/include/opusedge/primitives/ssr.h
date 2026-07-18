#pragma once

#include "../core/types.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace opusedge {

struct SSRConfig {
    Float elasticity_base = 2.0;
    Float entropy_sensitivity = 1.5;
    Float min_threshold = 0.05;
};

struct SSRResult {
    std::vector<Float> thresholded_values;
    int preserved_count;
    Float preserved_fraction;
    Float compression_ratio;
};

class SSR {
    SSRConfig cfg;
public:
    explicit SSR(const SSRConfig& c = {}) : cfg(c) {}

    SSRResult analyze(const std::vector<Float>& singular_values,
                      Float layer_entropy) const {
        int n = singular_values.size();
        SSRResult r;

        Float kappa = cfg.elasticity_base / std::max(Float(0.01), layer_entropy * cfg.entropy_sensitivity + 0.01);
        Float tau = cfg.min_threshold;

        Float max_sv = *std::max_element(singular_values.begin(), singular_values.end());
        Float threshold = tau * max_sv;

        r.thresholded_values.resize(n);
        int preserved = 0;

        for (int i = 0; i < n; ++i) {
            Float s = singular_values[i] / std::max(Float(1e-10), max_sv);
            Float sigmoid = 1.0 / (1.0 + std::exp(-kappa * (s - threshold / std::max(Float(1e-10), max_sv))));
            r.thresholded_values[i] = singular_values[i] * sigmoid;
            if (r.thresholded_values[i] > 0.01 * max_sv) preserved++;
        }

        r.preserved_count = preserved;
        r.preserved_fraction = static_cast<Float>(preserved) / n;
        r.compression_ratio = 1.0 - r.preserved_fraction;
        return r;
    }

    SSRResult casp_analyze(const std::vector<Float>& singular_values,
                           Float curvature) const {
        Float effective_entropy = 1.0 / std::max(Float(0.01), curvature);
        return analyze(singular_values, effective_entropy);
    }
};

} // namespace opusedge

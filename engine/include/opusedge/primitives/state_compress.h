#pragma once

#include "../core/types.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace opusedge {

struct StateCompress {
    Float gamma_min = 0.25;
    Float gamma_max = 1.0;
    Float tau_compress = 0.05;
    Float tau_full = 0.30;

    Float keep_ratio(Float delta_t) const {
        if (delta_t < tau_compress) return gamma_min;
        if (delta_t >= tau_full) return gamma_max;
        Float t = (delta_t - tau_compress) / (tau_full - tau_compress);
        return gamma_min + (gamma_max - gamma_min) * t;
    }

    VectorXf compress_channels(const VectorXf& state, Float delta_t) const {
        Float gamma = keep_ratio(delta_t);
        int d = state.size();
        int n_keep = std::max(1, static_cast<int>(d * gamma));

        std::vector<int> idx(d);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return std::abs(state(a)) > std::abs(state(b)); });

        VectorXf result = state;
        for (int i = n_keep; i < d; ++i)
            result(idx[i]) = 0.0;
        return result;
    }

    struct CompressResult {
        int channels_zeroed;
        Float keep_ratio;
        Float state_memory_savings;
    };

    CompressResult analyze(Float delta_t, int d_state) const {
        Float gamma = keep_ratio(delta_t);
        int n_zero = static_cast<int>(d_state * (1.0 - gamma));
        return {n_zero, gamma, 1.0 - gamma};
    }
};

} // namespace opusedge

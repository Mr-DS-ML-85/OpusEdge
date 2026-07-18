#pragma once

#include "../core/types.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace opusedge {

struct NDPAResult {
    std::vector<Float> rectified_delta;
    Float gamma;
    Float rho_raw;
    bool active;
};

class NDPA {
public:
    NDPAResult analyze(const std::vector<Float>& proxy_delta,
                       const std::vector<Float>& attn_scores) const {
        int n = std::min(proxy_delta.size(), attn_scores.size());
        NDPAResult r;
        r.active = false;

        if (n < 5) {
            r.rectified_delta = proxy_delta;
            r.gamma = 0.0;
            r.rho_raw = 0.0;
            return r;
        }

        auto mean = [](const std::vector<Float>& v, int sz) {
            return std::accumulate(v.begin(), v.begin() + sz, Float(0.0)) / sz;
        };
        auto stddev = [&](const std::vector<Float>& v, int sz, Float m) {
            Float accum = 0.0;
            for (int i = 0; i < sz; ++i) accum += (v[i] - m) * (v[i] - m);
            return std::sqrt(accum / sz);
        };

        Float md = mean(proxy_delta, n);
        Float ma = mean(attn_scores, n);
        Float sd = stddev(proxy_delta, n, md);
        Float sa = stddev(attn_scores, n, ma);

        if (sd < 1e-10 || sa < 1e-10) {
            r.rectified_delta = std::vector<Float>(proxy_delta.begin(), proxy_delta.begin() + n);
            r.gamma = 0.0;
            r.rho_raw = 0.0;
            return r;
        }

        Float cov = 0.0;
        for (int i = 0; i < n; ++i)
            cov += (proxy_delta[i] - md) * (attn_scores[i] - ma);
        cov /= n;

        r.rho_raw = cov / (sd * sa);
        r.gamma = std::max(Float(0.0), r.rho_raw);

        if (r.gamma < 1e-6) {
            r.rectified_delta = std::vector<Float>(proxy_delta.begin(), proxy_delta.begin() + n);
            return r;
        }

        r.rectified_delta.resize(n);
        for (int i = 0; i < n; ++i) {
            Float ds_norm = (proxy_delta[i] - md) / sd;
            Float ds_rank1 = ds_norm * sa + ma;
            r.rectified_delta[i] = (1.0 - r.gamma) * proxy_delta[i] + r.gamma * ds_rank1;
        }
        r.active = r.gamma > 0.1;
        return r;
    }
};

} // namespace opusedge

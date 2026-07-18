#pragma once

#include "../core/types.h"
#include "../core/signal.h"
#include <algorithm>
#include <numeric>
#include <random>

namespace opusedge {

struct SelKV {
    static EvictionResult evict(const VectorXf& delta, Float ratio, int seq_len) {
        int n = delta.size();
        int n_keep = std::max(1, static_cast<int>(seq_len * (1.0 - ratio)));

        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return delta(a) > delta(b); });

        EvictionResult r;
        r.retained_indices.assign(indices.begin(), indices.begin() + n_keep);
        r.evicted_indices.assign(indices.begin() + n_keep, indices.end());
        r.memory_savings = 1.0 - static_cast<Float>(n_keep) / seq_len;
        return r;
    }

    static MatrixXf attention_mask(const VectorXf& delta, Float ratio) {
        int n = delta.size();
        auto ev = evict(delta, ratio, n);
        MatrixXf mask = MatrixXf::Ones(n, n);
        for (int idx : ev.evicted_indices)
            mask.col(idx).setZero();
        return mask;
    }

    static Float quality_ratio(const VectorXf& delta, Float ratio, int seed = 42) {
        int n = delta.size();
        auto ev = evict(delta, ratio, n);
        int n_keep = ev.retained_indices.size();

        std::vector<int> rnd(n);
        std::iota(rnd.begin(), rnd.end(), 0);
        std::mt19937 g(seed);
        std::shuffle(rnd.begin(), rnd.end(), g);

        Float selkv_sum = 0, rnd_sum = 0;
        for (int i : ev.retained_indices) selkv_sum += delta(i);
        for (int i = 0; i < n_keep; ++i) rnd_sum += delta(rnd[i]);

        return (rnd_sum > 1e-12) ? selkv_sum / rnd_sum : 0.0;
    }

    static VectorXf gakv_score(const VectorXf& delta,
                               const VectorXf& router_entropy,
                               Float w1 = 0.8, Float w2 = 0.2) {
        auto normalize = [](VectorXf v) {
            Float mn = v.minCoeff(), mx = v.maxCoeff();
            if (mx - mn > 1e-12) v = (v.array() - mn) / (mx - mn);
            return v;
        };
        VectorXf d_n = normalize(delta);
        if (router_entropy.size() == 0) return w1 * d_n;
        VectorXf r_n = normalize(router_entropy);
        return w1 * d_n + w2 * r_n;
    }
};

} // namespace opusedge

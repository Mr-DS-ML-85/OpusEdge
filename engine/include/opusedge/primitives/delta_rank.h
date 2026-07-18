#pragma once

#include "../core/types.h"
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace opusedge {

struct DeltaRank {
    Float delta_threshold;
    int rank_low, rank_mid, rank_high, rank_full;

    DeltaRank() : delta_threshold(0.05), rank_low(16), rank_mid(32),
                  rank_high(64), rank_full(128) {}

    int get_rank_for_delta(Float delta_val) const {
        if (delta_val < 0.05) return rank_low;
        if (delta_val < 0.20) return rank_mid;
        if (delta_val < 0.50) return rank_high;
        if (delta_val < 1.00) return rank_full;
        return rank_full;
    }

    // Soft Spectral Relaxation (SSR): replace hard rank-k truncation with a
    // sigmoid-gated soft threshold at the k-th singular value. Guards against
    // out-of-bounds when rank_fraction == 1.0.
    static MatrixXf ssr(const MatrixXf& W, Float rank_fraction, Float alpha = 2.0) {
        const int d = std::min<int>(W.rows(), W.cols());
        const int k = std::clamp(static_cast<int>(d * rank_fraction), 1, d);

        Eigen::BDCSVD<MatrixXf> svd(W, Eigen::ComputeThinU | Eigen::ComputeThinV);
        VectorXf S = svd.singularValues();

        // τ sits at the boundary between kept and decayed singular values.
        const Float tau = S(std::min(k, static_cast<int>(S.size()) - 1));
        VectorXf S_new(S.size());
        for (int i = 0; i < S.size(); ++i) {
            Float gate = 1.0 / (1.0 + std::exp(-alpha * (S(i) - tau)));
            S_new(i) = S(i) * gate;
        }
        return svd.matrixU() * S_new.asDiagonal() * svd.matrixV().transpose();
    }
};

} // namespace opusedge

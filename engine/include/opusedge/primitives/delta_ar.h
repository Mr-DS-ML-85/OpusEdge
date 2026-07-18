#pragma once

#include "../core/types.h"
#include "../core/signal.h"
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>

namespace opusedge {

//
// Build per-query top-K indices  (O[S·log K] min-heap)
// Returns [S, K] matrix where row i = sorted (desc Δ) key indices ≤ i
//
// Build per-query top-K causal routing indices (O(S · log K) min-heap).
// Returns [S, K] where row i lists the top-K key positions j ≤ i by δ (descending).
// Slots that don't yet have K causal candidates are padded with -1 (sentinel),
// so downstream consumers can skip them instead of double-attending to token 0.
inline MatrixXf build_delta_ar_indices(const VectorXf& delta_scores, int top_k) {
    const int S = delta_scores.size();
    const int K = std::max(1, std::min(top_k, S));
    MatrixXf result = MatrixXf::Constant(S, K, -1.0);

    using pair = std::pair<Float, int>;
    auto cmp = [](pair a, pair b) { return a.first > b.first; };  // min-heap
    std::priority_queue<pair, std::vector<pair>, decltype(cmp)> heap(cmp);

    for (int i = 0; i < S; ++i) {
        heap.push({delta_scores(i), i});
        while ((int)heap.size() > K) heap.pop();

        std::vector<pair> items;
        items.reserve(heap.size());
        while (!heap.empty()) { items.push_back(heap.top()); heap.pop(); }
        std::sort(items.begin(), items.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });
        for (int j = 0; j < (int)items.size(); ++j)
            result(i, j) = static_cast<Float>(items[j].second);
        for (auto& p : items) heap.push(p);
    }
    return result;
}

//
// Build per-query [S,S] boolean mask from indices
//
inline MatrixXf build_delta_ar_mask(const VectorXf& delta_scores, int top_k) {
    const int S = delta_scores.size();
    MatrixXf idx = build_delta_ar_indices(delta_scores, top_k);
    MatrixXf mask = MatrixXf::Zero(S, S);
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < idx.cols(); ++j) {
            int col = static_cast<int>(idx(i, j));
            if (col >= 0 && col < S) mask(i, col) = 1;
        }
        // Guarantee self-attention: queries always see themselves.
        mask(i, i) = 1;
    }
    return mask;
}

//
// EB-DAR reservoir + output injection  (Eq 10a/10b)
//
//   Eq 10b:  E_l = (1-M_l) ⊙ score_l + (1-β)·E_{l-1}
//   Eq 10a:  O_l =  M_l ⊙ score_l +     β ·E_{l-1}
//
// Returns (reservoir, boosted_output) — both [S].
//
inline std::pair<VectorXf, VectorXf> ebdar(
    const VectorXf& scores, const MatrixXf& mask_2d, Float beta = 0.85)
{
    int S = scores.size();
    VectorXf E = VectorXf::Zero(S);
    VectorXf O = VectorXf::Zero(S);

    for (int t = 0; t < S; ++t) {
        Float prev = (t > 0) ? E(t - 1) : 0.0;
        Float attended = 0.0, skipped = 0.0;
        for (int j = 0; j <= t; ++j) {
            Float m = mask_2d(t, j);
            Float s = scores(j);
            attended += m * s;
            skipped += (1.0 - m) * s;
        }
        E(t) = skipped + (1.0 - beta) * prev;
        O(t) = attended + beta * prev;
    }
    return {E, O};
}

//
// True O(S·K·d) sparse attention  — single head
// Q, K, V : [S, D]
// indices : [S, K]  — per-query selected key positions
//
inline MatrixXf sparse_attention(
    const MatrixXf& Q,
    const MatrixXf& K,
    const MatrixXf& V,
    const MatrixXi& indices)
{
    int S = Q.rows();
    int D = Q.cols();
    int Kmax = indices.cols();
    MatrixXf output = MatrixXf::Zero(S, D);
    Float scale = 1.0 / std::sqrt(static_cast<Float>(D));

    for (int i = 0; i < S; ++i) {
        // Collect valid (non-sentinel) key positions for this query.
        std::vector<int> valid;
        valid.reserve(Kmax);
        for (int j = 0; j < Kmax; ++j) {
            int kidx = indices(i, j);
            if (kidx >= 0 && kidx < S) valid.push_back(kidx);
        }
        if (valid.empty()) valid.push_back(i);  // fall back to self

        const int Kv = valid.size();
        VectorXf scores(Kv);
        for (int j = 0; j < Kv; ++j) {
            Float dot = 0.0;
            for (int d = 0; d < D; ++d)
                dot += Q(i, d) * K(valid[j], d);
            scores(j) = dot * scale;
        }

        Float max_s = scores.maxCoeff();
        VectorXf expv = (scores.array() - max_s).exp();
        Float sum_exp = expv.sum();
        if (sum_exp > 1e-30) expv /= sum_exp;

        for (int d = 0; d < D; ++d) {
            Float val = 0.0;
            for (int j = 0; j < Kv; ++j)
                val += expv(j) * V(valid[j], d);
            output(i, d) = val;
        }
    }
    return output;
}

//
// Multi-head sparse attention wrapper
// Q, K, V : [S, H, D]  (stored column-major: H*D per row, or flat)
// indices : [S, K]
//
inline MatrixXf sparse_causal_attention(
    const MatrixXf& Q,
    const MatrixXf& K,
    const MatrixXf& V,
    const MatrixXi& indices,
    int n_heads, int head_dim)
{
    int S = Q.rows();
    MatrixXf output = MatrixXf::Zero(S, n_heads * head_dim);
    for (int h = 0; h < n_heads; ++h) {
        int col_off = h * head_dim;
        MatrixXf Qh = Q.middleCols(col_off, head_dim);
        MatrixXf Kh = K.middleCols(col_off, head_dim);
        MatrixXf Vh = V.middleCols(col_off, head_dim);
        MatrixXf Oh = sparse_attention(Qh, Kh, Vh, indices);
        output.middleCols(col_off, head_dim) = Oh;
    }
    return output;
}

//
// FLOP cost helpers
//
struct DeltaARFlops {
    Float baseline;      // 2·S²·D per layer
    Float routed;        // 2·S·K·D per layer
    Float reduction;     // 1 - routed/baseline
};

inline DeltaARFlops delta_ar_flops(int S, int K, int D = 1) {
    Float bl = 2.0 * static_cast<Float>(S) * S * D;
    Float rt = 2.0 * static_cast<Float>(S) * K * D;
    return {bl, rt, (bl > 0) ? 1.0 - rt / bl : 0.0};
}

//
// Full Delta-AR + EB-DAR measure
//
struct DeltaARResult {
    Float baseline_ppl;
    Float routed_ppl;
    Float delta_ppl;
    int tokens_attended;
    int total_tokens;
    Float flop_reduction;
    Float ebdar_mean_reservoir;
    int top_k;
};

inline DeltaARResult measure_delta_ar(
    const VectorXf& delta_scores,
    int top_k = -1,
    Float beta = 0.85,
    Float ebdar_boost = 0.10)
{
    int S = delta_scores.size();
    int K = (top_k > 0) ? std::min(top_k, S) : std::max(16, S / 2);
    K = std::max(1, K);

    MatrixXf mask_2d = build_delta_ar_mask(delta_scores, K);
    auto [E, O] = ebdar(delta_scores, mask_2d, beta);

    // Re-admit tokens with high reservoir energy
    Float e_mean = E.mean();
    Float e_std = std::sqrt((E.array() - e_mean).square().mean());
    Float e_thresh = e_mean + ebdar_boost * e_std;
    if (e_thresh > 0) {
        for (int j = 0; j < S; ++j) {
            if (E(j) > e_thresh) {
                for (int i = j; i < S; ++i)
                    mask_2d(i, j) = 1;
            }
        }
    }

    int tokens_attended = static_cast<int>(mask_2d.sum());
    Float flop_red = delta_ar_flops(S, K).reduction;

    return DeltaARResult{
        .baseline_ppl = 0.0,  // caller provides actual PPL
        .routed_ppl = 0.0,
        .delta_ppl = 0.0,
        .tokens_attended = tokens_attended,
        .total_tokens = S,
        .flop_reduction = flop_red,
        .ebdar_mean_reservoir = E.mean(),
        .top_k = K,
    };
}

} // namespace opusedge

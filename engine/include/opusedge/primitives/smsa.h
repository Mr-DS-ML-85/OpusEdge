#pragma once

// SMSA — SSM-Masked Sparse Attention.
// The paper's empirical implementation uses a **fixed** causal sliding window
// (Table 1, w=64 across all sequence lengths). This header also exposes the
// theoretical adaptive-width mapping w(δ) referenced in Section 4.2 and a
// canonical SSD-mask sparse attention using the exponential-decay mask
// M[i,j] = exp(-∑ᵏ_{j+1..i} δ_k) from the SSD framework (Dao & Gu, 2024).

#include "../core/types.h"
#include <cmath>
#include <algorithm>

namespace opusedge {

struct SMSAConfig {
    int  window_size    = 64;    // fixed window w (paper Table 1 default)
    int  w_min          = 32;    // adaptive lower bound
    int  w_max          = 512;   // adaptive upper bound
    Float ssd_threshold = 1e-3;  // keep positions where M[i,j] > τ
};

struct SMSA {
    SMSAConfig cfg;
    int window_size;             // convenience mirror of cfg.window_size

    explicit SMSA(int w) : cfg(), window_size(w) { cfg.window_size = w; }
    explicit SMSA(const SMSAConfig& c) : cfg(c), window_size(c.window_size) {}

    // ── fixed-window causal mask [T, T] ────────────────────────────────
    MatrixXf attention_mask(int seq_len) const {
        MatrixXf mask = MatrixXf::Zero(seq_len, seq_len);
        for (int i = 0; i < seq_len; ++i) {
            int start = std::max(0, i - window_size + 1);
            for (int j = start; j <= i; ++j) mask(i, j) = 1;
        }
        return mask;
    }

    // ── theoretical adaptive width w(δ) from paper §4.2 ───────────────
    // Monotone increasing in δ: high-δ tokens get a wider window (SSM's
    // effective range is short), low-δ tokens get a narrow window (SSM
    // already covers the long range). Linear interpolation between w_min
    // and w_max on a δ-normalised scale, clamped.
    int window_for_delta(Float delta_t) const {
        Float t = std::clamp(delta_t, Float(0.0), Float(1.0));
        return cfg.w_min + static_cast<int>(std::round(t * (cfg.w_max - cfg.w_min)));
    }

    // ── SSD-mask sparse attention mask (canonical paper form) ─────────
    // M[i,j] = exp(-Σ_{k=j+1..i} δ_k). Keep positions where M[i,j] > τ
    // AND j <= i (causal). Reduces to a per-token adaptive window whose
    // effective width is set by the local δ-decay rate.
    MatrixXf ssd_mask(const VectorXf& delta_per_tok) const {
        const int T = delta_per_tok.size();
        MatrixXf mask = MatrixXf::Zero(T, T);
        const Float log_tau = std::log(std::max(cfg.ssd_threshold, Float(1e-30)));
        for (int i = 0; i < T; ++i) {
            Float cum = 0.0;
            mask(i, i) = 1.0;                       // always keep self
            for (int j = i - 1; j >= 0; --j) {
                cum += std::abs(delta_per_tok(j + 1));
                if (-cum < log_tau) break;          // M[i,j] < τ — stop
                mask(i, j) = 1.0;
            }
        }
        return mask;
    }

    // ── actual forward pass: causal sliding-window softmax attention ──
    // Q, K, V : [T, D] single-head slices.  Returns O : [T, D].
    static MatrixXf forward_single_head(
        const MatrixXf& Q, const MatrixXf& K, const MatrixXf& V, int window)
    {
        const int T = Q.rows(), D = Q.cols();
        MatrixXf out = MatrixXf::Zero(T, D);
        const Float scale = 1.0 / std::sqrt(static_cast<Float>(D));
        for (int i = 0; i < T; ++i) {
            int start = std::max(0, i - window + 1);
            int W = i - start + 1;
            VectorXf scores(W);
            for (int j = 0; j < W; ++j) {
                Float dot = 0.0;
                for (int d = 0; d < D; ++d) dot += Q(i, d) * K(start + j, d);
                scores(j) = dot * scale;
            }
            Float mx = scores.maxCoeff();
            VectorXf ex = (scores.array() - mx).exp();
            Float Z = ex.sum();
            if (Z > 1e-30) ex /= Z;
            for (int d = 0; d < D; ++d) {
                Float v = 0.0;
                for (int j = 0; j < W; ++j) v += ex(j) * V(start + j, d);
                out(i, d) = v;
            }
        }
        return out;
    }

    // Full causal attention baseline.  **THIS IS DELIBERATELY O(S² · D)** —
    // it exists to be the honest O(S²) reference the sparse variants (SMSA,
    // Delta-AR) beat by construction. Do NOT "optimise" this into a linear
    // form: standard exact self-attention with per-token softmax is provably
    // Θ(S²·D) in FLOPs (S queries × S causal keys × D dot-product). Any
    // implementation that measures O(S) either
    //   (a) is a linear-attention *approximation* (Linformer / Performer),
    //   (b) benchmarks a single decode step ignoring the S cached keys, or
    //   (c) is broken.
    // The scaling bench (bench_scaling.cpp) empirically classifies this as
    // O(S²) via log-log regression, and it must keep doing so.
    //
    // Loop nesting:  ∀ i ∈ [0,T)  |  ∀ j ∈ [0, i]  |  ∀ d ∈ [0, D)  →  Θ(T²·D)
    static MatrixXf forward_full_causal(
        const MatrixXf& Q, const MatrixXf& K, const MatrixXf& V)
    {
        const int T = Q.rows(), D = Q.cols();
        MatrixXf out = MatrixXf::Zero(T, D);
        const Float scale = 1.0 / std::sqrt(static_cast<Float>(D));
        for (int i = 0; i < T; ++i) {                    // O(T) — every query
            const int W = i + 1;                          // grows to O(T)
            VectorXf scores(W);
            for (int j = 0; j < W; ++j) {                 // O(T) causal keys
                Float dot = 0.0;
                for (int d = 0; d < D; ++d)               // O(D) per dot product
                    dot += Q(i, d) * K(j, d);
                scores(j) = dot * scale;
            }
            Float mx = scores.maxCoeff();
            VectorXf ex = (scores.array() - mx).exp();
            Float Z = ex.sum(); if (Z > 1e-30) ex /= Z;
            for (int d = 0; d < D; ++d) {                 // O(D) output write
                Float v = 0.0;
                for (int j = 0; j < W; ++j)               // O(T) weighted sum
                    v += ex(j) * V(j, d);
                out(i, d) = v;
            }
        }
        return out;                                      // total: Θ(T²·D) FLOPs
    }

    // Instance method wrapping the fixed-window forward.
    MatrixXf forward(const MatrixXf& Q, const MatrixXf& K, const MatrixXf& V) const {
        return forward_single_head(Q, K, V, window_size);
    }

    // ── analytical FLOP / memory savings ───────────────────────────────
    Float kv_memory_reduction(int seq_len) const {
        Float full = static_cast<Float>(seq_len);
        Float window = static_cast<Float>(std::min(window_size, seq_len));
        return 1.0 - window / full;
    }

    Float speedup(int seq_len) const {
        Float full = static_cast<Float>(seq_len) * seq_len;
        int w = std::min(window_size, seq_len);
        Float sparse = static_cast<Float>(seq_len) * w;
        return (sparse > 0) ? full / sparse : 1.0;
    }

    struct SMSAResult {
        Float speedup;
        Float memory_savings;
        int effective_window;
    };

    SMSAResult analyze(int seq_len) const {
        int w = std::min(window_size, seq_len);
        return { .speedup = speedup(seq_len),
                 .memory_savings = kv_memory_reduction(seq_len),
                 .effective_window = w };
    }
};

} // namespace opusedge

#include <gtest/gtest.h>
#include "opusedge/primitives/smsa.h"
#include <random>

using namespace opusedge;

// ── existing analytical tests ────────────────────────────────────────
TEST(SMSATest, WindowMask) {
    SMSA smsa(3);
    auto mask = smsa.attention_mask(5);
    ASSERT_EQ(mask.rows(), 5);
    ASSERT_EQ(mask.cols(), 5);
    EXPECT_EQ(mask(0, 0), 1);
    EXPECT_EQ(mask(2, 0), 1);
    EXPECT_EQ(mask(3, 0), 0);
    EXPECT_EQ(mask(4, 2), 1);
    EXPECT_EQ(mask(4, 1), 0);
}

TEST(SMSATest, Speedup) {
    SMSA smsa(256);
    auto res = smsa.analyze(2048);
    EXPECT_NEAR(res.speedup, 8.0, 0.01);
    EXPECT_NEAR(res.memory_savings, 0.875, 0.01);
}

// ── correctness: window == seq_len reduces to full causal attention ──
TEST(SMSATest, FullWindowEqualsFullCausal) {
    const int T = 32, D = 8;
    std::mt19937 g(0);
    std::normal_distribution<Float> nd(0, 1);
    MatrixXf Q(T, D), K(T, D), V(T, D);
    for (int i = 0; i < T; ++i)
        for (int d = 0; d < D; ++d) {
            Q(i, d) = nd(g); K(i, d) = nd(g); V(i, d) = nd(g);
        }
    SMSA smsa(T);   // window covers everything
    MatrixXf sw = smsa.forward(Q, K, V);
    MatrixXf full = SMSA::forward_full_causal(Q, K, V);
    Float err = (sw - full).norm() / std::max(full.norm(), Float(1e-9));
    EXPECT_LT(err, 1e-6);
}

// ── correctness: window == 1 is identity through the value at position i ──
TEST(SMSATest, Window1IsSelfAttention) {
    const int T = 16, D = 4;
    MatrixXf Q = MatrixXf::Random(T, D);
    MatrixXf K = MatrixXf::Random(T, D);
    MatrixXf V = MatrixXf::Random(T, D);
    SMSA smsa(1);
    MatrixXf out = smsa.forward(Q, K, V);
    // with window=1 each row of out should equal the corresponding row of V
    for (int i = 0; i < T; ++i)
        for (int d = 0; d < D; ++d)
            EXPECT_NEAR(out(i, d), V(i, d), 1e-9);
}

// ── correctness: sliding window drops distant tokens ─────────────────
TEST(SMSATest, WindowExcludesDistantTokens) {
    const int T = 64, D = 8, W = 8;
    MatrixXf Q = MatrixXf::Random(T, D);
    MatrixXf K = MatrixXf::Random(T, D);
    MatrixXf V = MatrixXf::Random(T, D);
    SMSA smsa(W);
    MatrixXf out = smsa.forward(Q, K, V);

    // Perturb K/V at an early position outside the window at t=T-1.
    MatrixXf K2 = K, V2 = V;
    for (int d = 0; d < D; ++d) { K2(0, d) += 100.0; V2(0, d) += 100.0; }
    MatrixXf out2 = smsa.forward(Q, K2, V2);
    // Position T-1's window is [T-W..T-1] — must NOT depend on position 0.
    for (int d = 0; d < D; ++d) EXPECT_NEAR(out(T-1, d), out2(T-1, d), 1e-9);
    // Position 0 IS inside its own trivial window and MUST change.
    EXPECT_GT((out.row(0) - out2.row(0)).norm(), 1.0);
}

// ── adaptive width: monotone increasing in δ ─────────────────────────
TEST(SMSATest, AdaptiveWindowMonotone) {
    SMSA smsa(64);
    int w_lo = smsa.window_for_delta(0.05);
    int w_mid = smsa.window_for_delta(0.5);
    int w_hi = smsa.window_for_delta(0.95);
    EXPECT_LT(w_lo, w_mid);
    EXPECT_LT(w_mid, w_hi);
    EXPECT_GE(w_lo, smsa.cfg.w_min);
    EXPECT_LE(w_hi, smsa.cfg.w_max);
}

// ── SSD mask is causal + always keeps self + decays with δ ──────────
TEST(SMSATest, SSDMaskCausalAndDecays) {
    const int T = 32;
    VectorXf delta(T);
    for (int i = 0; i < T; ++i) delta(i) = 0.1;    // uniform decay
    SMSAConfig cfg; cfg.ssd_threshold = 1e-3;
    SMSA smsa(cfg);
    MatrixXf mask = smsa.ssd_mask(delta);
    // causal: no upper triangle
    for (int i = 0; i < T; ++i)
        for (int j = i + 1; j < T; ++j)
            EXPECT_EQ(mask(i, j), 0.0);
    // always self-attending
    for (int i = 0; i < T; ++i) EXPECT_EQ(mask(i, i), 1.0);
    // effective window: log(τ) / -δ = log(1e-3) / -0.1 ≈ 69, capped by i
    // → row T-1 should keep ~min(69, T) positions
    int kept = static_cast<int>(mask.row(T - 1).sum());
    EXPECT_GT(kept, 20);
    EXPECT_LE(kept, T);
}

#include <gtest/gtest.h>
#include "opusedge/primitives/delta_ar.h"
#include "opusedge/core/signal.h"

using namespace opusedge;

TEST(DeltaARTest, BuildIndices) {
    VectorXf delta(5);
    delta << 0.1, 0.5, 0.05, 0.8, 0.3;
    auto idx = build_delta_ar_indices(delta, 2);

    ASSERT_EQ(idx.rows(), 5);
    ASSERT_EQ(idx.cols(), 2);

    // Query 0 (i=0): only token 0 available → {0}
    // Query 3 (i=3): tokens {0,1,2,3}, top-2 by Δ = {3(0.8), 1(0.5)}
    // Actual order depends on heap sort stability; check presence
    auto check_presence = [&](int i, std::vector<int> expected) {
        for (int e : expected) {
            bool found = false;
            for (int j = 0; j < idx.cols(); ++j) {
                if (static_cast<int>(idx(i, j)) == e) { found = true; break; }
            }
            EXPECT_TRUE(found) << "index[" << i << "] missing " << e;
        }
    };
    check_presence(3, {3, 1});  // Δ=0.8 and Δ=0.5
    check_presence(4, {3, 1});  // Δ=0.8 and Δ=0.5 (token 4 Δ=0.3 < 0.5)
}

TEST(DeltaARTest, SparseAttentionObeysKComplexity) {
    int S = 10, D = 4, K = 3;
    MatrixXf Q = MatrixXf::Random(S, D);
    MatrixXf Kmat = MatrixXf::Random(S, D);
    MatrixXf V = MatrixXf::Random(S, D);

    VectorXf delta(S);
    delta.setRandom();
    delta = delta.cwiseAbs();

    MatrixXf idx = build_delta_ar_indices(delta, K);
    MatrixXi idx_int = idx.cast<int>();
    MatrixXf out = sparse_attention(Q, Kmat, V, idx_int);

    ASSERT_EQ(out.rows(), S);
    ASSERT_EQ(out.cols(), D);
    EXPECT_TRUE(out.array().isFinite().all());
}

TEST(DeltaARTest, EBDAR_OutputEq10a) {
    VectorXf scores(4);
    scores << 1.0, 0.5, 0.2, 0.8;

    MatrixXf mask(4, 4);
    mask.setZero();
    mask(0, 0) = 1;
    mask(1, 0) = 1; mask(1, 1) = 1;
    mask(2, 0) = 1; mask(2, 2) = 1;
    mask(3, 0) = 1; mask(3, 3) = 1;

    Float beta = 0.85;
    auto [E, O] = ebdar(scores, mask, beta);

    // Step 0: attended=1.0*1=1.0, skipped=0, E[0]=0+(1-0.85)*0=0, O[0]=1+0.85*0=1
    EXPECT_NEAR(O(0), 1.0, 1e-6);
    EXPECT_NEAR(E(0), 0.0, 1e-6);

    // Step 1: mask=[1,1], attended=1.0+0.5=1.5, skipped=0
    //   E[1] = 0 + (1-0.85)*0 = 0
    //   O[1] = 1.5 + 0.85*0 = 1.5
    EXPECT_NEAR(O(1), 1.5, 1e-6);
    EXPECT_NEAR(E(1), 0.0, 1e-6);

    // Step 2: mask[2] = [1, 0, 1]
    //   attended = 1.0 + 0.2 = 1.2
    //   skipped = 0.5
    //   E[2] = 0.5 + (1-0.85)*0 = 0.5
    //   O[2] = 1.2 + 0.85*0 = 1.2
    EXPECT_NEAR(O(2), 1.2, 1e-6);
    EXPECT_NEAR(E(2), 0.5, 1e-6);

    // Step 3: mask[3] = [1, 0, 0, 1]
    //   attended = 1.0 + 0.8 = 1.8
    //   skipped = 0.5 + 0.2 = 0.7
    //   E[3] = 0.7 + (1-0.85)*0.5 = 0.7 + 0.075 = 0.775
    //   O[3] = 1.8 + 0.85*0.5 = 1.8 + 0.425 = 2.225
    EXPECT_NEAR(O(3), 2.225, 1e-6);
    EXPECT_NEAR(E(3), 0.775, 1e-6);
}

TEST(DeltaARTest, FLOPsReduction) {
    auto flops = delta_ar_flops(2048, 256, 64);
    EXPECT_NEAR(flops.reduction, 1.0 - 256.0 / 2048.0, 1e-6);
    EXPECT_EQ(flops.baseline, 2.0 * 2048 * 2048 * 64);
    EXPECT_EQ(flops.routed, 2.0 * 2048 * 256 * 64);
}

TEST(DeltaARTest, Measure) {
    VectorXf delta(8);
    delta << 0.1, 0.9, 0.05, 0.7, 0.3, 0.01, 0.6, 0.4;
    auto res = measure_delta_ar(delta, 3, 0.85, 0.10);

    EXPECT_EQ(res.total_tokens, 8);
    EXPECT_EQ(res.top_k, 3);
    EXPECT_LE(res.tokens_attended, res.total_tokens * res.top_k);
    EXPECT_GT(res.flop_reduction, 0.5);
    EXPECT_TRUE(std::isfinite(res.ebdar_mean_reservoir));
}

TEST(DeltaARTest, ProxyDeltaExtraction) {
    int n_layers = 3, n_tokens = 5, d_model = 8;
    std::vector<MatrixXf> hidden_states;
    for (int l = 0; l < n_layers; ++l)
        hidden_states.push_back(MatrixXf::Random(n_tokens, d_model));

    VectorXf pd = SignalExtractor::proxy_delta(hidden_states);
    ASSERT_EQ(pd.size(), n_tokens);
    EXPECT_TRUE(pd.array().isFinite().all());
    EXPECT_GE(pd(0), 0);  // proxy_delta(0) = proxy_delta(1) ≥ 0
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

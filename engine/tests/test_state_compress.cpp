#include <gtest/gtest.h>
#include "opusedge/primitives/state_compress.h"

using namespace opusedge;

TEST(StateCompressTest, KeepRatioMonotonic) {
    StateCompress sc;
    EXPECT_EQ(sc.gamma_min, 0.25);
    EXPECT_EQ(sc.gamma_max, 1.0);
    // Below τ_compress (0.05): saturate at γ_min.
    EXPECT_NEAR(sc.keep_ratio(0.01), 0.25, 1e-6);
    // Linear interp on [τ_compress=0.05, τ_full=0.30]:
    //   δ=0.20  →  t = (0.20-0.05)/(0.30-0.05) = 0.6
    //           →  0.25 + 0.75 * 0.6 = 0.70
    EXPECT_NEAR(sc.keep_ratio(0.20), 0.70, 1e-6);
    // Above τ_full: saturate at γ_max.
    EXPECT_NEAR(sc.keep_ratio(0.50), 1.00, 1e-6);
    // Monotonicity spot-check.
    EXPECT_LT(sc.keep_ratio(0.05), sc.keep_ratio(0.15));
    EXPECT_LT(sc.keep_ratio(0.15), sc.keep_ratio(0.30));
}

TEST(StateCompressTest, CompressChannels) {
    StateCompress sc;
    VectorXf state = VectorXf::Random(8);
    VectorXf compressed = sc.compress_channels(state, 0.01);
    int n_zero = (compressed.array() == 0).count();
    EXPECT_GE(n_zero, 4);
}

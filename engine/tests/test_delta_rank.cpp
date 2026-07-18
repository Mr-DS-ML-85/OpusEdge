#include <gtest/gtest.h>
#include "opusedge/primitives/delta_rank.h"

using namespace opusedge;

TEST(DeltaRankTest, MapDeltaToRank) {
    DeltaRank dr;
    EXPECT_EQ(dr.get_rank_for_delta(0.01), 16);
    EXPECT_EQ(dr.get_rank_for_delta(0.10), 32);
    EXPECT_EQ(dr.get_rank_for_delta(0.35), 64);
    EXPECT_EQ(dr.get_rank_for_delta(0.75), 128);
}

TEST(DeltaRankTest, SSRPreservesShape) {
    MatrixXf W = MatrixXf::Random(16, 16);
    MatrixXf W_ssr = DeltaRank::ssr(W, 0.5);
    EXPECT_EQ(W_ssr.rows(), 16);
    EXPECT_EQ(W_ssr.cols(), 16);
    EXPECT_TRUE(W_ssr.array().isFinite().all());
}

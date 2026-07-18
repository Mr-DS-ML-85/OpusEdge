#include <gtest/gtest.h>
#include "opusedge/core/signal.h"

using namespace opusedge;

TEST(SignalTest, NativeDelta) {
    std::vector<VectorXf> dt = {
        VectorXf::Random(5),
        VectorXf::Random(5),
    };
    auto delta = SignalExtractor::native_delta(dt, 5);
    EXPECT_EQ(delta.size(), 5);
    EXPECT_TRUE(delta.array().isFinite().all());
}

TEST(SignalTest, SpearmanCorrelation) {
    VectorXf x(5); x << 1.0, 2.0, 3.0, 4.0, 5.0;
    VectorXf y(5); y << 5.0, 4.0, 3.0, 2.0, 1.0;
    Float rho = SignalExtractor::compute_spearman(x, y);
    EXPECT_NEAR(rho, -1.0, 1e-6);
}

TEST(SignalTest, NormalizeImportance) {
    VectorXf s(4); s << 0.0, 2.0, 4.0, 6.0;
    VectorXf n = SignalExtractor::normalize_importance(s);
    EXPECT_NEAR(n.maxCoeff(), 1.0, 1e-6);
    EXPECT_NEAR(n.minCoeff(), 0.0, 1e-6);
}

TEST(SignalTest, SACTTransmute) {
    VectorXf scores(4); scores << 0.5, 0.1, 0.9, 0.3;
    VectorXf sact = SignalExtractor::sact_transmute(scores, 0.15);
    EXPECT_EQ(sact.size(), 4);
    EXPECT_TRUE(sact.array().isFinite().all());
}

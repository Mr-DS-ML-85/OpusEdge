#include <gtest/gtest.h>
#include "opusedge/primitives/head_gate.h"

using namespace opusedge;

TEST(HeadGateTest, ActiveHeadsByDelta) {
    HeadGateConfig cfg;
    cfg.n_heads = 32;
    HeadGate hg(cfg);
    EXPECT_EQ(hg.active_heads(0.01), 4);
    EXPECT_EQ(hg.active_heads(0.10), 16);
    EXPECT_EQ(hg.active_heads(0.20), 24);
    EXPECT_EQ(hg.active_heads(0.50), 32);
}

TEST(HeadGateTest, HeadSalience) {
    MatrixXf acts = MatrixXf::Random(10, 4);
    Float s = HeadGate::head_salience(acts);
    EXPECT_GE(s, 0.0);
    EXPECT_LE(s, 1.0);
}

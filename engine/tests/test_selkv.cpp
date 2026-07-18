#include <gtest/gtest.h>
#include "opusedge/primitives/selkv.h"

using namespace opusedge;

TEST(SelKVTest, EvictPreservesTopTokens) {
    VectorXf delta(6);
    delta << 0.1, 0.9, 0.05, 0.7, 0.3, 0.01;
    auto ev = SelKV::evict(delta, 0.5, 6);
    EXPECT_EQ(ev.retained_indices.size(), 3);
    EXPECT_EQ(ev.evicted_indices.size(), 3);
    for (int idx : ev.retained_indices)
        EXPECT_GE(delta(idx), delta(ev.evicted_indices[0]));
}

TEST(SelKVTest, QualityRatio) {
    VectorXf delta(6);
    delta << 0.1, 0.9, 0.05, 0.7, 0.3, 0.01;
    Float qr = SelKV::quality_ratio(delta, 0.5, 42);
    EXPECT_GE(qr, 1.0);
}

TEST(SelKVTest, GAKVScore) {
    VectorXf delta(4);
    delta << 0.1, 0.5, 0.05, 0.8;
    VectorXf entropy(4);
    entropy << 0.2, 0.6, 0.1, 0.7;
    VectorXf score = SelKV::gakv_score(delta, entropy);
    EXPECT_EQ(score.size(), 4);
    EXPECT_TRUE(score.array().isFinite().all());
}

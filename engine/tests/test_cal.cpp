#include <gtest/gtest.h>
#include "opusedge/primitives/cal.h"
#include "opusedge/primitives/rcal.h"

using namespace opusedge;

TEST(CALTest, ClassifyTask) {
    CAL cal;
    EXPECT_EQ(cal.classify("math").tier, ComputeTier::High);
    EXPECT_EQ(cal.classify("summarize").tier, ComputeTier::Efficiency);
}

TEST(CALTest, RCALFreezeAndRecompute) {
    // RCAL's confidence EMA starts at 1.0 (freeze-favouring). A high log-prob
    // keeps it frozen; a long run of low log-probs eventually drops the EMA
    // below the freeze threshold and forces recompute.
    RCAL rcal(0.5, 0.9);   // heavy ema α=0.5 so a few steps flip the state
    // Frozen initially (ema=1.0 > 0.9):
    EXPECT_FALSE(rcal.should_recompute(0.95));
    EXPECT_GE(rcal.frozen_step_count(), 1);
    // Push several low-confidence steps to drag ema below 0.9:
    for (int i = 0; i < 5; ++i) rcal.should_recompute(0.1);
    EXPECT_TRUE(rcal.should_recompute(0.1));
}

TEST(CALTest, RCALTierLookup) {
    RCAL rcal;
    auto r = rcal.classify("math", 0.5);
    EXPECT_EQ(r.tier.name, "Math");
    EXPECT_NEAR(r.tier.rigidity, 0.85, 1e-6);
}

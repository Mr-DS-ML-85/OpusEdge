#include <gtest/gtest.h>
#include "opusedge/primitives/pareto.h"

using namespace opusedge;

TEST(ParetoTest, ParetoOptimal) {
    std::vector<ParetoPoint> pts = {
        {0.50, 256, 0.5, 0.5, 20.0, 1e9, 1e6, 10.0},
        {0.25, 128, 0.7, 0.6, 15.0, 2e9, 2e6,  5.0},
        {0.00, 512, 1.0, 1.0, 10.0, 4e9, 4e6,  0.0},
    };
    auto frontier = ParetoFrontier::pareto_optimal(pts);
    ASSERT_FALSE(frontier.empty());
    EXPECT_LE(frontier.size(), pts.size());
}

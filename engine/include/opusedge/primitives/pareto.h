#pragma once

#include "../core/types.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace opusedge {

struct ParetoConfig {
    Float eviction_ratio;
    int window_size;
    Float rank_fraction;
    Float channel_keep;
};

class ParetoFrontier {
public:
    static std::vector<ParetoPoint> pareto_optimal(std::vector<ParetoPoint>& points) {
        std::sort(points.begin(), points.end(),
            [](auto& a, auto& b) { return a.flops < b.flops; });

        std::vector<ParetoPoint> frontier;
        Float best_ppl = std::numeric_limits<Float>::max();
        for (auto& p : points) {
            if (p.ppl < best_ppl) {
                best_ppl = p.ppl;
                frontier.push_back(p);
            }
        }
        return frontier;
    }

    static ParetoPoint knee_point(const std::vector<ParetoPoint>& frontier) {
        if (frontier.empty()) return {};
        Float max_dist = -1;
        int knee_idx = 0;

        auto& first = frontier.front();
        auto& last = frontier.back();
        Float dx = last.flops - first.flops;
        Float dy = last.ppl - first.ppl;
        Float norm = std::sqrt(dx * dx + dy * dy);
        if (norm < 1e-12) return frontier[frontier.size() / 2];

        for (size_t i = 0; i < frontier.size(); ++i) {
            Float dist = std::abs(dy * frontier[i].flops - dx * frontier[i].ppl
                                  + last.flops * first.ppl - last.ppl * first.flops) / norm;
            if (dist > max_dist) {
                max_dist = dist;
                knee_idx = i;
            }
        }
        return frontier[knee_idx];
    }

    static std::vector<ParetoPoint> sweep(const VectorXf& delta, int seq_len) {
        std::vector<ParetoPoint> points;
        std::vector<Float> ratios = {0.0, 0.25, 0.50, 0.625, 0.75, 0.875};
        std::vector<int> windows = {64, 128, 256, 512};
        std::vector<Float> ranks = {0.25, 0.50, 0.70, 1.0};

        for (Float r : ratios) {
            for (int w : windows) {
                for (Float rf : ranks) {
                    Float flops = 2.0 * seq_len * (seq_len * (1 - r) +
                                  static_cast<Float>(w) * r);
                    points.push_back({r, w, rf, 0.5, 0.0, flops, 0.0, 0.0});
                }
            }
        }
        return points;
    }
};

} // namespace opusedge

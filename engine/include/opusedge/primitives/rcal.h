#pragma once

#include "../core/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace opusedge {

struct RCALTier {
    std::string name;
    Float rigidity;
    Float eviction_cap;
    Float rank_keep;
    Float head_keep;
    Float channel_keep;
};

struct RCALResult {
    RCALTier tier;
    Float effective_threshold;
    Float confidence;
    bool frozen;
};

class RCAL {
    Float ema_alpha;
    Float freeze_threshold;
    Float confidence_ema;
    int frozen_steps;

    static const std::unordered_map<std::string, RCALTier>& tiers() {
        static const std::unordered_map<std::string, RCALTier> t = {
            {"code",     {"Code",     0.85, 0.00, 1.00, 1.00, 0.90}},
            {"math",     {"Math",     0.85, 0.00, 1.00, 1.00, 0.90}},
            {"reasoning",{"Reasoning",0.80, 0.10, 0.90, 0.90, 0.85}},
            {"logic",    {"Logic",    0.75, 0.15, 0.85, 0.85, 0.80}},
            {"physics",  {"Physics",  0.70, 0.20, 0.80, 0.80, 0.75}},
            {"general",  {"General",  0.50, 0.35, 0.70, 0.75, 0.60}},
            {"summary",  {"Summary",  0.30, 0.50, 0.50, 0.50, 0.50}},
        };
        return t;
    }

public:
    explicit RCAL(Float alpha = 0.1, Float freeze = 0.9)
        : ema_alpha(alpha), freeze_threshold(freeze),
          confidence_ema(1.0), frozen_steps(0) {}

    RCALResult classify(const std::string& task_label, Float base_threshold) const {
        auto it = tiers().find(task_label);
        const RCALTier& t = (it != tiers().end()) ? it->second : tiers().at("general");

        Float eff = base_threshold * (1.0 + t.rigidity);
        return {t, eff, confidence_ema, confidence_ema > freeze_threshold};
    }

    bool should_recompute(Float log_prob) {
        confidence_ema = ema_alpha * log_prob + (1.0 - ema_alpha) * confidence_ema;
        if (confidence_ema > freeze_threshold) {
            frozen_steps++;
            return false;
        }
        frozen_steps = 0;
        return true;
    }

    Float modulate_threshold(const std::string& task_label, Float base) const {
        auto t = classify(task_label, base);
        return t.effective_threshold;
    }

    Float modulated_eviction_cap(const std::string& task_label) const {
        auto it = tiers().find(task_label);
        return (it != tiers().end()) ? it->second.eviction_cap : 0.35;
    }

    int frozen_step_count() const { return frozen_steps; }
    Float current_confidence() const { return confidence_ema; }
};

} // namespace opusedge

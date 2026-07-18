#pragma once

#include "../core/types.h"
#include <string>
#include <unordered_map>
#include <cmath>

namespace opusedge {

struct CALConfig {
    Float tau_global = 0.5;
};

struct CALTier {
    std::string name;
    ComputeTier tier;
    Float rigidity;
    Float eviction_cap;
};

class CAL {
    CALConfig cfg;
    static const std::unordered_map<std::string, Float>& task_rigidity_map() {
        static const std::unordered_map<std::string, Float> m = {
            {"math", 0.85}, {"code", 0.75}, {"reasoning", 0.80},
            {"logic", 0.70}, {"physics", 0.65}, {"translate", 0.35},
            {"summarize", 0.30}, {"qa", 0.50}, {"general", 0.50},
        };
        return m;
    }
public:
    explicit CAL(const CALConfig& c = {}) : cfg(c) {}

    Float rigidity_of(const std::string& task_label) const {
        auto it = task_rigidity_map().find(task_label);
        return (it != task_rigidity_map().end()) ? it->second : 0.5;
    }

    CALTier classify(const std::string& task_label) const {
        Float s = rigidity_of(task_label);
        if (s >= 0.70) return {"High", ComputeTier::High, s, 0.0};
        if (s >= 0.40) return {"Balanced", ComputeTier::Balanced, s, 0.25};
        return {"Efficiency", ComputeTier::Efficiency, s, 0.50};
    }

    Float effective_threshold(const std::string& task_label) const {
        return cfg.tau_global * (1.0 + rigidity_of(task_label));
    }
};

} // namespace opusedge

#pragma once

#include "types.h"
#include <unordered_map>

namespace opusedge {

struct OpusEdgeConfig {
    Float tau_global = 0.5;
    Float default_eviction_ratio = 0.25;
    Float default_window_ratio = 0.125;
    Float default_rank_fraction = 0.7;
    Float default_head_keep = 0.75;
    Float default_channel_keep = 0.6;
    Float sact_residual = 0.15;
    Float ebdar_beta = 0.85;
    Float ndpa_rho = 0.5;
    Float mpsr_lambda = 0.01;
    Float cal_rigidity_low = 0.2;
    Float cal_rigidity_high = 0.9;
    Float rcal_ema_alpha = 0.1;
    Float rcal_freeze_threshold = 0.9;
    bool rcal_enabled = true;

    static OpusEdgeConfig defaults() { return OpusEdgeConfig{}; }

    struct TaskRigidity {
        Float rigidity;
        ComputeTier tier;
    };

    TaskRigidity classify(const std::string& task_label) const;
    ComputeTier resolve_tier(const std::string& task_label) const;
    TierPolicy resolve_policy(ComputeTier tier) const;
    bool primitive_enabled(PrimitiveType prim, ComputeTier tier) const;
};

} // namespace opusedge

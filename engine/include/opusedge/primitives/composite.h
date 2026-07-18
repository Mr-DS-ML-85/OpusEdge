#pragma once

#include "../core/types.h"
#include "selkv.h"
#include "smsa.h"
#include "delta_ar.h"
#include "delta_rank.h"
#include "head_gate.h"
#include "state_compress.h"
#include <vector>
#include <string>

namespace opusedge {

struct CompositeConfig {
    Float eviction_ratio = 0.25;
    int window_size = 256;
    int top_k = 64;
    Float rank_fraction = 0.70;
    Float channel_keep = 0.60;
    Float head_keep = 0.75;
    bool use_ebdar = true;
    bool use_sact = true;
};

struct CompositeResult {
    Float flop_reduction_pct;
    Float memory_savings_pct;
    Float estimated_delta_ppl;
    int total_active_heads;
    int seq_len;
    std::vector<std::string> applied_primitives;
};

class Composite {
    CompositeConfig cfg;

public:
    explicit Composite(const CompositeConfig& c = {}) : cfg(c) {}

    CompositeResult analyze(int seq_len, int n_heads, int head_dim, int hidden_dim) const {
        auto dar_flops = delta_ar_flops(seq_len, cfg.top_k, head_dim);
        SMSA smsa(cfg.window_size);
        auto smsa_res = smsa.analyze(seq_len);

        Float attn_savings = dar_flops.reduction;
        Float retained_smsa = 1.0 - smsa_res.memory_savings;
        Float kv_savings = 1.0 - (1.0 - cfg.eviction_ratio) * retained_smsa;
        HeadGateConfig hcfg;
        hcfg.n_heads = n_heads;
        int active_h = static_cast<int>(n_heads * cfg.head_keep);
        Float head_savings = 1.0 - static_cast<Float>(active_h) / n_heads;

        CompositeResult r;
        r.flop_reduction_pct = 100.0 * (1.0 - (1.0 - attn_savings) * (1.0 - head_savings));
        r.memory_savings_pct = 100.0 * kv_savings;
        r.estimated_delta_ppl = (1.0 - r.flop_reduction_pct / 100.0) * 5.0;
        r.total_active_heads = active_h;
        r.seq_len = seq_len;
        r.applied_primitives = {"SelKV", "SMSA", "DeltaAR", "EB-DAR"};

        if (cfg.rank_fraction < 1.0)
            r.applied_primitives.push_back("SSR/DeltaRank");
        if (cfg.head_keep < 1.0)
            r.applied_primitives.push_back("HeadDeactivate");
        if (cfg.channel_keep < 1.0)
            r.applied_primitives.push_back("StateCompress");

        return r;
    }
};

} // namespace opusedge

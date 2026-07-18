#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <variant>

namespace opusedge {

using Float = double;
using VectorXf = Eigen::Matrix<Float, Eigen::Dynamic, 1>;
using MatrixXf = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using RowVectorXf = Eigen::Matrix<Float, 1, Eigen::Dynamic>;

enum class ArchitectureFamily {
    Dense,
    Hybrid,
    MoE,
    SSM
};

enum class ComputeTier {
    High,
    Balanced,
    Efficiency
};

enum class PrimitiveType {
    SelKV,
    SMSA,
    DeltaAR,
    EBDAR,
    DeltaRank,
    HeadDeactivate,
    StateCompress,
    DenseEvic,
    GAKV,
    SGAKV,
    CAL,
    RCAL,
    MPSR,
    EBAR,
    CASP,
    NDPA,
    CSA
};

struct SignalExtraction {
    VectorXf delta_scores;
    VectorXf proxy_delta;
    VectorXf attn_scores;
    VectorXf router_entropy;
    std::vector<MatrixXf> hidden_states;
    std::vector<MatrixXf> attn_weights;
    int n_tokens;
    int n_layers;
    ArchitectureFamily family;

    Float importance_at(int t) const {
        if (delta_scores.size() > 0)
            return delta_scores(t);
        if (proxy_delta.size() > 0)
            return proxy_delta(t);
        return 0.0;
    }
};

struct EvictionResult {
    std::vector<int> retained_indices;
    std::vector<int> evicted_indices;
    Float memory_savings;
};

struct TierPolicy {
    std::string name;
    ComputeTier tier;
    Float eviction_cap;
    int window_min;
    Float rank_keep;
    Float head_keep;
    Float channel_keep;
    bool use_csa;
    bool use_ebdar;
    bool use_sact;
};

struct PrimitiveDecision {
    int step;
    PrimitiveType primitive;
    Float confidence;
    bool applied;
    ComputeTier tier;
    std::string action;
    std::chrono::steady_clock::time_point ts;

    std::string prim_name() const;
};

struct ReasoningSnapshot {
    std::vector<PrimitiveDecision> decisions;
    void record(int step, PrimitiveType prim, Float conf, ComputeTier tier,
                bool applied, const std::string& action = "");
    std::string summary() const;
};

struct ParetoPoint {
    Float eviction_ratio;
    int window_size;
    Float rank_fraction;
    Float channel_keep;
    Float ppl;
    Float flops;
    Float mem_bytes;
    Float delta_ppl;
};

} // namespace opusedge

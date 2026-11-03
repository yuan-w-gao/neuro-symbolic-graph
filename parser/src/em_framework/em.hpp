//
// Created by Yuan Gao on 04/06/2024.
//
#pragma once

#include <optional>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <memory>
#include "../manager.hpp"
#include "../forest_cache.hpp"
#include "em_base.hpp"
#include "em_types.hpp"
#include "em_utils.hpp"

namespace shrg {
namespace em {

// Metrics for profiling per-graph performance
struct GraphMetrics {
    std::string sentence_id;
    int node_count = 0;
    int edge_count = 0;
    size_t forest_size = 0;
    size_t max_chain_length = 0;
    size_t max_children = 0;
    size_t max_parents = 0;
    double parse_time_ms = 0.0;
    double deep_copy_time_ms = 0.0;
    double reset_flags_time_ms = 0.0;
    double inside_time_ms = 0.0;
    double outside_time_ms = 0.0;
    double expected_count_time_ms = 0.0;
    double total_em_time_ms = 0.0;
};

class EM : public EMBase {
  public:
    EM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold);
    EM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold, std::string dir);
    EM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold, std::string dir, int timeout_seconds);
    EM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold, std::string dir, int timeout_seconds, const std::unordered_set<std::string>& skip_graphs);

    // Enable forest caching
    void enableCaching(const std::string& cache_dir);

    // Get cache statistics
    size_t getCacheHits() const;
    size_t getCacheMisses() const;

    double computeInside(ChartItem *root);
    void computeOutsideNode(ChartItem *root, NodeLevelPQ &pq);
    void computeOutside(ChartItem *root);
    void initializeWeights();  // Set uniform weights for all rules

    void run() override;
    void run_safe();  // Like run(), but forks child to test each parse before committing
    void run_from_saved();

  protected:
    double prev_ll;
    double max_change;
    int max_change_ind;
    std::vector<ChartItem*> forests;

    bool converged() const override;
    void computeExpectedCount(ChartItem *root, double pw) override;
    void updateEM() override;

    float FindBestDerivationWeight(Generator *generator, ChartItem *root_ptr);
    void writeTreeScoresToDir(const std::string& filename,
                                  const std::vector<double>& scores);

    LabelToRule rule_dict;
    std::unordered_set<std::string> skip_graphs_;

    // Persistent memory pool for storing deep-copied derivation forests
    utils::MemoryPool<ChartItem> persistent_pool_;
    void run_1iter();

  private:
    LabelToRule getRuleDict();

    // Deep copy functions for persistent derivation forests
    ChartItem* deepCopyChartItem(ChartItem* original,
                                std::unordered_map<ChartItem*, ChartItem*>& copied_map,
                                utils::MemoryPool<ChartItem>& persistent_pool);

    void deepCopyDerivationRelations(ChartItem* original,
                                    ChartItem* copy,
                                    std::unordered_map<ChartItem*, ChartItem*>& copied_map,
                                    utils::MemoryPool<ChartItem>& persistent_pool);

    void collectAllReachableItems(ChartItem* root, std::unordered_set<ChartItem*>& all_items);

    void resetVisitedFlagsRecursive(ChartItem* item, std::unordered_set<ChartItem*>& visited);
    bool parentsOutsideReady(ChartItem* node) const;
    void computeOutsideChainOptimized(ChartItem* root);
    void computeOutside_optimized(ChartItem *root);

  public:
    // Main function to deep copy an entire derivation forest
    ChartItem* deepCopyDerivationForest(ChartItem* root, utils::MemoryPool<ChartItem>& persistent_pool);

    // Reset visited flags for all items in a derivation forest before each EM iteration
    void resetVisitedFlags(ChartItem* root);

    // Profiling support
    bool profiling_enabled_ = false;
    std::vector<GraphMetrics> graph_metrics_;

    void enableProfiling(bool enable = true) { profiling_enabled_ = enable; }
    size_t countForestSize(ChartItem* root);
    void computeForestMetrics(ChartItem* root, GraphMetrics& metrics);
    void writeMetricsToCSV(const std::string& filepath);
    void printMetricsSummary();

    // Optimized outside computation (O(n) instead of O(chain^depth))
    void computeOutsideFixed(ChartItem* root);

    // Validation support - compare original vs fixed implementations
    bool validateOutsideImplementations(ChartItem* root, double tolerance = 1e-10);
    bool validateFullEMCycle(double tolerance = 1e-10);
    void runValidation();

    // Results accessors (populated after run())
    const std::vector<double>& getLogLikelihoodHistory() const { return ll_history_; }
    const std::vector<double>& getIterationTimes() const { return iteration_times_; }
    double getFinalLogLikelihood() const { return ll; }
    int getNumIterations() const { return num_iterations_; }
    bool hasConverged() const { return converged_; }
    size_t getNumCachedForests() const { return num_cached_forests_; }

    // Verbose control
    void setVerbose(bool verbose) { verbose_ = verbose; }
    bool isVerbose() const { return verbose_; }

private:
    // Results storage
    std::vector<double> ll_history_;
    std::vector<double> iteration_times_;
    int num_iterations_ = 0;
    bool converged_ = false;
    size_t num_cached_forests_ = 0;
    bool verbose_ = true;

    // Forest caching support
    std::unique_ptr<forest_cache::ForestCache> cache_;
    bool caching_enabled_ = false;
    uint32_t grammar_hash_ = 0;

    // Helper to compute grammar hash for cache validation
    uint32_t computeGrammarHash() const;

    // Helper to compute graph hash for cache validation
    uint32_t computeGraphHash(const EdsGraph& graph) const;
};

} // namespace em
} // namespace shrg


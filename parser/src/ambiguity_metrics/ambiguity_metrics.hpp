#ifndef LEXCXG_AMBIGUITY_METRICS_HPP
#define LEXCXG_AMBIGUITY_METRICS_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace shrg {
    class ChartItem;
    class Generator;
}

namespace lexcxg {

struct ForestStats {
    int num_nodes = 0;
    int num_edges = 0;
    int max_depth = 0;
    double avg_branching = 0.0;
    double complexity = 0.0;
};

struct AmbiguityMetrics {
    double entropy = 0.0;
    double expected_count = 0.0;
    ForestStats forest_stats;

    int num_derivation_alternatives = 0;
    bool has_valid_probabilities = false;
};


double ComputeDerivationEntropy(::shrg::ChartItem* root, double log_partition);
double ComputeDerivationEntropy(::shrg::ChartItem* root, double log_partition, bool debug);

double ComputeDerivationEntropyDP(::shrg::ChartItem* root);
double ComputeDerivationEntropyDP(::shrg::ChartItem* root, bool debug);

void ComputePartitionAndEntropyDP(
    ::shrg::ChartItem* root,
    double& out_log_Z,
    double& out_entropy
);


double ComputeExpectedDerivationCount(::shrg::ChartItem* root);
double ComputeLogDerivationCount(::shrg::ChartItem* root);
ForestStats ComputeForestComplexity(::shrg::ChartItem* root);
AmbiguityMetrics ComputeAllMetrics(::shrg::ChartItem* root, double log_partition);
void PopulateChildren(::shrg::ChartItem* root, ::shrg::Generator* generator);
double CountDerivations(::shrg::ChartItem* root, ::shrg::Generator* generator);
double CountDerivationsLog(::shrg::ChartItem* root, ::shrg::Generator* generator);

::shrg::ChartItem* GetCanonicalNode(::shrg::ChartItem* node);

inline double LogAdd(double a, double b) {
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    if (a > b) {
        return a + std::log1p(std::exp(b - a));
    } else {
        return b + std::log1p(std::exp(a - b));
    }
}

inline bool IsValidProb(double log_prob) {
    return std::isfinite(log_prob);
}

}

#endif // LEXCXG_AMBIGUITY_METRICS_HPP

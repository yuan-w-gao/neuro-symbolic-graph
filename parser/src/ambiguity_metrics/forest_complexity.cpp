
#include "ambiguity_metrics/ambiguity_metrics.hpp"
#include "graph_parser/parser_chart_item.hpp"

#include <unordered_set>
#include <algorithm>
#include <functional>

namespace lexcxg {

namespace {


void ComputeStatsRecursive(
    shrg::ChartItem* node,
    std::unordered_set<shrg::ChartItem*>& visited,
    ForestStats& stats,
    int depth
) {
    if (!node) {
        return;
    }

    shrg::ChartItem* ptr = node;
    shrg::ChartItem* root = node;

    do {
        if (visited.count(ptr)) {
            ptr = ptr->next_ptr;
            continue;
        }

        visited.insert(ptr);
        stats.num_nodes++;
        stats.num_edges += static_cast<int>(ptr->children.size());
        stats.max_depth = std::max(stats.max_depth, depth);

        for (shrg::ChartItem* child : ptr->children) {
            ComputeStatsRecursive(child, visited, stats, depth + 1);
        }

        ptr = ptr->next_ptr;
    } while (ptr && ptr != root);
}

}

ForestStats ComputeForestComplexity(shrg::ChartItem* root) {
    ForestStats stats;

    if (!root) {
        return stats;
    }

    std::unordered_set<shrg::ChartItem*> visited;
    ComputeStatsRecursive(root, visited, stats, 0);

    if (stats.num_nodes > 0) {
        stats.avg_branching = static_cast<double>(stats.num_edges) / stats.num_nodes;
    }

    stats.complexity = stats.num_nodes * stats.avg_branching * stats.max_depth;

    return stats;
}


AmbiguityMetrics ComputeAllMetrics(shrg::ChartItem* root, double log_partition) {
    AmbiguityMetrics metrics;

    if (!root) {
        return metrics;
    }

    metrics.expected_count = ComputeExpectedDerivationCount(root);

    if (IsValidProb(log_partition) || IsValidProb(root->log_inside_prob)) {
        metrics.entropy = ComputeDerivationEntropy(root, log_partition, false);
        metrics.has_valid_probabilities = true;
    }

    metrics.forest_stats = ComputeForestComplexity(root);

    std::unordered_set<shrg::ChartItem*> visited;
    std::function<void(shrg::ChartItem*)> countAlternatives = [&](shrg::ChartItem* node) {
        if (!node) return;

        shrg::ChartItem* ptr = node;
        shrg::ChartItem* nodeRoot = node;
        int alt_count = 0;

        do {
            if (visited.count(ptr)) {
                ptr = ptr->next_ptr;
                continue;
            }
            visited.insert(ptr);
            alt_count++;

            for (shrg::ChartItem* child : ptr->children) {
                countAlternatives(child);
            }

            ptr = ptr->next_ptr;
        } while (ptr && ptr != nodeRoot);

        if (alt_count > 1) {
            metrics.num_derivation_alternatives += alt_count;
        }
    };

    visited.clear();
    countAlternatives(root);

    return metrics;
}

}

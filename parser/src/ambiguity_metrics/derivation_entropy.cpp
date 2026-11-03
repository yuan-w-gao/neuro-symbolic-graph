#include "ambiguity_metrics/ambiguity_metrics.hpp"
#include "graph_parser/parser_chart_item.hpp"

#include <unordered_set>
#include <cmath>
#include <iostream>

namespace lexcxg {

namespace {

void ComputeGammaLogWeightSum(
    shrg::ChartItem* node,
    double log_Z,
    double& sum_gamma_log_w,
    std::unordered_set<shrg::ChartItem*>& visited
) {
    if (!node || visited.count(node)) {
        return;
    }

    shrg::ChartItem* ptr = node;
    shrg::ChartItem* start = node;

    do {
        if (visited.count(ptr)) {
            ptr = ptr->next_ptr;
            continue;
        }
        visited.insert(ptr);

        double log_w = 0.0;
        if (ptr->rule_ptr) {
            log_w = ptr->rule_ptr->log_rule_weight;
        }
        double log_alpha_item = log_w;
        for (shrg::ChartItem* child : ptr->children) {
            if (IsValidProb(child->log_inside_prob)) {
                log_alpha_item += child->log_inside_prob;
            }
        }
        double log_outside = ptr->log_outside_prob;

        if (IsValidProb(log_alpha_item) && IsValidProb(log_outside)) {
            double log_gamma = log_alpha_item + log_outside - log_Z;
            double gamma = std::exp(log_gamma);

            if (gamma > 0 && std::isfinite(gamma) && std::isfinite(log_w)) {
                sum_gamma_log_w += gamma * log_w;
            }
        }

        for (shrg::ChartItem* child : ptr->children) {
            ComputeGammaLogWeightSum(child, log_Z, sum_gamma_log_w, visited);
        }

        ptr = ptr->next_ptr;
    } while (ptr && ptr != start);
}

}

double ComputeDerivationEntropy(shrg::ChartItem* root, double log_partition, bool debug) {
    if (!root) {
        return 0.0;
    }

    double log_Z = log_partition;
    if (!IsValidProb(log_Z)) {
        log_Z = root->log_inside_prob;
    }

    if (debug) {
        std::cerr << "[DEBUG] Derivation Entropy Computation\n";
        std::cerr << "[DEBUG] log_partition=" << log_partition
                  << " root->log_inside=" << root->log_inside_prob
                  << " log_Z=" << log_Z << "\n";
    }

    if (!IsValidProb(log_Z)) {
        if (debug) std::cerr << "[DEBUG] Invalid log_Z, returning 0\n";
        return 0.0;
    }

    double sum_gamma_log_w = 0.0;
    std::unordered_set<shrg::ChartItem*> visited;

    ComputeGammaLogWeightSum(root, log_Z, sum_gamma_log_w, visited);

    double entropy = log_Z - sum_gamma_log_w;

    if (debug) {
        std::cerr << "[DEBUG] log_Z=" << log_Z
                  << " sum_gamma_log_w=" << sum_gamma_log_w
                  << " entropy=" << entropy
                  << " visited=" << visited.size() << "\n";
    }

    return std::max(0.0, entropy);
}

double ComputeDerivationEntropy(shrg::ChartItem* root, double log_partition) {
    return ComputeDerivationEntropy(root, log_partition, false);
}

}

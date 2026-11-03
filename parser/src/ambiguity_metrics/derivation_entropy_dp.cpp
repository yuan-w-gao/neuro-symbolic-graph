#include "ambiguity_metrics/ambiguity_metrics.hpp"
#include "graph_parser/parser_chart_item.hpp"

#include <unordered_map>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

namespace lexcxg {

namespace {


struct OrNodeResult {
    double log_Z;
    double entropy;
};

shrg::ChartItem* GetCycleCanonical(shrg::ChartItem* node) {
    if (!node) return nullptr;

    shrg::ChartItem* canonical = node;
    shrg::ChartItem* ptr = node->next_ptr;

    while (ptr && ptr != node) {
        if (ptr < canonical) {
            canonical = ptr;
        }
        ptr = ptr->next_ptr;
    }

    return canonical;
}


OrNodeResult ComputeEntropyDP(
    shrg::ChartItem* node,
    std::unordered_map<shrg::ChartItem*, OrNodeResult>& memo,
    bool debug
) {
    if (!node) {
        return {-std::numeric_limits<double>::infinity(), 0.0};
    }

    shrg::ChartItem* canonical = GetCycleCanonical(node);

    auto it = memo.find(canonical);
    if (it != memo.end()) {
        return it->second;
    }

    struct Alternative {
        double log_w;
        double sum_child_entropy;
        std::vector<shrg::ChartItem*> children;
    };
    std::vector<Alternative> alternatives;

    shrg::ChartItem* ptr = node;
    shrg::ChartItem* start = node;

    do {
        Alternative alt;
        alt.sum_child_entropy = 0.0;

        double log_rule_weight = 0.0;
        if (ptr->rule_ptr) {
            log_rule_weight = ptr->rule_ptr->log_rule_weight;
        }

        // Compute log_w(a) = log p(rule) + sum_c log Z(c)
        // and accumulate child entropies
        alt.log_w = log_rule_weight;

        for (shrg::ChartItem* child : ptr->children) {
            OrNodeResult child_result = ComputeEntropyDP(child, memo, debug);
            alt.log_w += child_result.log_Z;
            alt.sum_child_entropy += child_result.entropy;
            alt.children.push_back(child);
        }

        if (std::isfinite(alt.log_w)) {
            alternatives.push_back(alt);
        }

        ptr = ptr->next_ptr;
    } while (ptr && ptr != start);

    if (alternatives.empty()) {
        OrNodeResult result = {-std::numeric_limits<double>::infinity(), 0.0};
        memo[canonical] = result;
        return result;
    }

    if (alternatives.size() == 1) {
        OrNodeResult result = {alternatives[0].log_w, alternatives[0].sum_child_entropy};
        memo[canonical] = result;
        return result;
    }

    // Compute log Z(v) = logsumexp of log_w(a)
    double max_log_w = -std::numeric_limits<double>::infinity();
    for (const auto& alt : alternatives) {
        max_log_w = std::max(max_log_w, alt.log_w);
    }

    double sum_exp = 0.0;
    for (const auto& alt : alternatives) {
        sum_exp += std::exp(alt.log_w - max_log_w);
    }
    double log_Z = max_log_w + std::log(sum_exp);

    // Compute responsibilities r(a|v) and entropy terms
    // H(v) = log Z(v) - sum_a r(a|v) log_w(a) + sum_a r(a|v) sum_c H(c)
    double sum_r_log_w = 0.0;
    double sum_r_child_entropy = 0.0;

    for (const auto& alt : alternatives) {
        double r = std::exp(alt.log_w - log_Z);  // responsibility
        sum_r_log_w += r * alt.log_w;
        sum_r_child_entropy += r * alt.sum_child_entropy;
    }

    double entropy = log_Z - sum_r_log_w + sum_r_child_entropy;

    entropy = std::max(0.0, entropy);

    if (debug) {
        std::cerr << "[DP] OR-node " << canonical
                  << ": #alts=" << alternatives.size()
                  << " log_Z=" << log_Z
                  << " entropy=" << entropy << "\n";
    }

    OrNodeResult result = {log_Z, entropy};
    memo[canonical] = result;
    return result;
}

}

double ComputeDerivationEntropyDP(shrg::ChartItem* root, bool debug) {
    if (!root) {
        return 0.0;
    }

    std::unordered_map<shrg::ChartItem*, OrNodeResult> memo;
    OrNodeResult result = ComputeEntropyDP(root, memo, debug);

    if (debug) {
        std::cerr << "[DP] Final: log_Z=" << result.log_Z
                  << " entropy=" << result.entropy
                  << " #OR-nodes=" << memo.size() << "\n";
    }

    return result.entropy;
}

double ComputeDerivationEntropyDP(shrg::ChartItem* root) {
    return ComputeDerivationEntropyDP(root, false);
}

void ComputePartitionAndEntropyDP(
    shrg::ChartItem* root,
    double& out_log_Z,
    double& out_entropy
) {
    if (!root) {
        out_log_Z = -std::numeric_limits<double>::infinity();
        out_entropy = 0.0;
        return;
    }

    std::unordered_map<shrg::ChartItem*, OrNodeResult> memo;
    OrNodeResult result = ComputeEntropyDP(root, memo, false);

    out_log_Z = result.log_Z;
    out_entropy = result.entropy;
}

}

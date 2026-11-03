/**
 * @file extra_ambiguity_metrics.cpp
 * @brief Python bindings for ambiguity metrics (entropy, derivation count, etc.)
 */

#include "extra_ambiguity_metrics.hpp"
#include "inside_outside.hpp"
#include <cmath>

namespace shrg {

// Forward declaration to match the actual implementation
void addRulePointer(ChartItem *root, std::vector<SHRG *> &shrg_rules);

/**
 * Initialize rule weights to uniform distribution if not already initialized.
 * Rules with log_rule_weight == 0 (weight = 1) are considered uninitialized.
 */
static void initializeUniformWeights(std::vector<SHRG*>& shrg_rules) {
    if (shrg_rules.empty()) return;

    // Check if weights are already initialized (any rule has non-zero log weight)
    bool already_initialized = false;
    for (const auto* rule : shrg_rules) {
        if (rule && rule->log_rule_weight < -1e-10) {  // Negative = initialized
            already_initialized = true;
            break;
        }
    }

    if (!already_initialized) {
        // Set uniform weights: log(1/N) = -log(N)
        double uniform_log_weight = -std::log(static_cast<double>(shrg_rules.size()));
        for (auto* rule : shrg_rules) {
            if (rule) {
                rule->log_rule_weight = uniform_log_weight;
            }
        }
    }
}

double Context_ComputeInsideOutside(Context& context) {
    ChartItem* root = const_cast<ChartItem*>(context.Result());
    if (!root) {
        return -std::numeric_limits<double>::infinity();
    }

    // Get the generator from the parser
    Generator* generator = context.parser->GetGenerator();
    if (!generator) {
        return -std::numeric_limits<double>::infinity();
    }

    // Get shrg_rules from the manager
    const Manager* manager = context.manager_ptr;
    if (!manager) {
        return -std::numeric_limits<double>::infinity();
    }

    // Need to cast away const since we may need to modify weights
    std::vector<SHRG*>& shrg_rules = const_cast<std::vector<SHRG*>&>(manager->shrg_rules);

    // Initialize weights to uniform if not already done
    initializeUniformWeights(shrg_rules);

    // Set up parent/children pointers (required for outside computation)
    addParentPointer(root, generator, 0);

    // Set up rule pointers (required for inside computation to get log weights)
    addRulePointer(root, shrg_rules);

    // Compute inside values (populates log_inside_prob)
    double log_partition = computeInside(root);

    // Compute outside values (populates log_outside_prob)
    computeOutside(root);

    return log_partition;
}

} // namespace shrg

//
// Created by Yuan Gao on 07/02/2025.
//
#pragma once

#include <optional>
#include "../manager.hpp"
#include "em_base.hpp"
#include "em_types.hpp"
#include "em_utils.hpp"

namespace shrg::em {
class OnlineEM : public EMBase {
public:
    OnlineEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs,
            Context *context, double threshold);
    OnlineEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs,
            Context *context, double threshold,  std::string dir, int timeout_seconds);
    void run() override;
    bool converged() const override;
protected:
    double prev_ll;
    LabelToRule rule_dict;
    size_t total_examples;
    size_t examples_seen;
    std::unordered_map<SHRG*, double> prev_weights;
    const double smoothing_factor = 1e-10;
    void computeExpectedCount(ChartItem *root, double pw) override;
    void updateEM() override;
    void verifyNormalization();
    LabelToRule getRuleDict();

};
}

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
class BatchEM : public EMBase {
public:
    BatchEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs,
            Context *context, double threshold, int batch_size);
    BatchEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs,
            Context *context, double threshold, int batch_size, std::string dir, int timeout_seconds);
    void run() override;
    bool converged() const override;
protected:
    double prev_ll;
    LabelToRule rule_dict;
    int batch_size_;
    int total_examples;
    int curr_batch_size;
    const double smoothing_factor = 1e-10;
    std::unordered_map<SHRG*, double> prev_weights;
    void computeExpectedCount(ChartItem *root, double pw) override;
    void updateEM() override;
    void verifyNormalization();
    LabelToRule getRuleDict();
};
}

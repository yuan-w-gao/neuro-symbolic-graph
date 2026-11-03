#pragma once
#include "em_base.hpp"
#include <cmath>
#include <iostream>
#include <utility>
#include "em_utils.hpp"

namespace shrg::em {

class ViterbiEM : public EMBase {
public:
    ViterbiEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs,
              Context *context, double threshold);
    ViterbiEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold, std::string outDir, int timeout_seconds);

    void run() override;

protected:
    double prev_ll;
    LabelToRule rule_dict;
    bool converged() const override;
    void computeExpectedCount(ChartItem *root, double pw) override;
    // Reorganizes parse forest to put best parse first at each node
    void reorganizeBestParse(ChartItem* root);

    // Build parent-child relationships for best parse path
    void buildBestParseRelationships(ChartItem* root, int level);
    ChartItem* findBestChild(ChartItem* parent, const SHRG* grammar, int edge_index);

    // Compute probabilities along best path only
    double computeViterbiInside(ChartItem* root);
    void computeViterbiOutside(ChartItem* root);
    // void computeViterbiExpectedCount(ChartItem* root, double pw);
    void updateEM();
    bool validateProbabilities();
private:

    LabelToRule getRuleDict();
};

} // namespace shrg::em
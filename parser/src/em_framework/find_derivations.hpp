//
// Created by Yuan Gao on 27/01/2025.
//

#ifndef FIND_DERIVATIONS_HPP
#define FIND_DERIVATIONS_HPP

#endif //FIND_DERIVATIONS_HPP

#include "../graph_parser/parser_chart_item.hpp"
#include "em_utils.hpp"
#include "../graph_parser/parser_utils.hpp"
#include <optional>
#include <random>
#include <set>

namespace shrg {
    using namespace std;
struct DerivationInfo {
    std::vector<int> rule_indices;
    std::vector<EdgeSet> edge_sets;
};

    float FindBestScoreWeight(ChartItem *root_ptr);
    Derivation FindBestDerivation_EMGreedy(ChartItem *root);
    Derivation FindBestDerivation_EMInside(ChartItem *root_ptr);
    Derivation FindBestDerivation_CountGreedy(ChartItem* root_ptr);
    Derivation FindBestDerivation_CountInside(ChartItem* root_ptr);

std::vector<int> ExtractRuleIndices_CountInside(ChartItem *root_ptr);
std::vector<int> ExtractRuleIndices_CountGreedy(ChartItem *root_ptr);
std::vector<int> ExtractRuleIndices_EMInside(ChartItem *root_ptr);
std::vector<int> ExtractRuleIndices_EMGreedy(ChartItem *root_ptr);
Derivation FindBestDerivation_sample(ChartItem *root_ptr);
std::vector<int> ExtractRuleIndices_sampled(ChartItem *root_ptr);
DerivationInfo ExtractRuleIndicesAndEdges_EMGreedy(ChartItem *root_ptr);
DerivationInfo ExtractRuleIndicesAndEdges_CountGreedy(ChartItem *root_ptr);
std::optional<DerivationInfo> ExtractGoldDerivationTree(ChartItem* root_ptr,
                                        const std::vector<int>& gold_indices);
std::optional<DerivationInfo> ExtractGoldDerivation(ChartItem* node,
                                        std::multiset<int>& gold_indices);
std::optional<DerivationInfo> ExtractGoldDerivation(ChartItem* root,
                                        const std::vector<int>& gold_indices_vec);
DerivationInfo ExtractDerivation_sampled(ChartItem *root_ptr);
DerivationInfo ExtractDerivation_uniform(ChartItem *root_ptr);
bool IndexExistsInSubtree(ChartItem* root_ptr, int target_index);
DerivationInfo ExtractRuleIndicesAndEdges_EMInside(ChartItem *root_ptr);
DerivationInfo ExtractRuleIndicesAndEdges_ScoreGreedy(ChartItem *root_ptr);
}
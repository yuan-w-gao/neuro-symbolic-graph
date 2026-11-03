#include "parser_tree_base.hpp"

namespace shrg {
namespace tree {

ChartItem empty_item;

std::vector<NodeMapping> masks_for_pred_edges{{}};
std::vector<NodeMapping> masks_for_structural_edges{{}, {.m8 = {0xff, 0}}, {.m8 = {0xff00, 0}}};

ChartItem *FindEdgeInTree(ChartItem *chart_item_ptr, const SHRG::Edge *edge_ptr) {
    if (!chart_item_ptr)
        return nullptr;

    TreeNodeBase *node_ptr = static_cast<TreeNodeBase *>(chart_item_ptr->attrs_ptr);
    if (!node_ptr)
        return nullptr;
    // NOTE: for unary node left_graph_ptr points to a subgraph of PassiveItem
    ChartItem *result_ptr = node_ptr->covered_edge_ptr == edge_ptr // unary node
                                ? chart_item_ptr->left_ptr
                                : FindEdgeInTree(chart_item_ptr->right_ptr, edge_ptr);

    if (!result_ptr)
        result_ptr = FindEdgeInTree(chart_item_ptr->left_ptr, edge_ptr);

    return result_ptr;
}

ChartItem *TreeGenerator::FindChartItemByEdge(ChartItem *chart_item_ptr,
                                              const SHRG::Edge *shrg_edge_ptr) {
    ChartItem *result_ptr = FindEdgeInTree(chart_item_ptr, shrg_edge_ptr);
    assert(result_ptr && result_ptr->attrs_ptr->grammar_ptr->label == shrg_edge_ptr->label);
    return result_ptr;
}

} // namespace tree
} // namespace shrg

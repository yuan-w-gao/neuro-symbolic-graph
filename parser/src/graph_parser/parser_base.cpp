#include <iostream>

#include "parser_base.hpp"

namespace shrg {

using std::size_t;

static ChartItem invalid_chart_item(nullptr, {}, {.m8 = {UINT64_MAX, UINT64_MAX}});
const Ref<ChartItem> ChartItemSet::empty_key_(invalid_chart_item);

ParserError SHRGParserBase::BeforeParse(const EdsGraph &graph) {
    size_t edge_count = graph.edges.size();
    if (edge_count > MAX_GRAPH_EDGE_COUNT)
        return ParserError::kTooLarge;

    // set edge mask
    graph_ptr_ = &graph;
    all_edges_in_graph_ = 0;
    all_edges_in_graph_.flip();
    all_edges_in_graph_ >>= all_edges_in_graph_.size() - edge_count;
    return ParserError::kNone;
}

void SHRGParserBase::ClearChart() {
    SHRG_DEBUG_RESET(num_grammars_available_);
    SHRG_DEBUG_RESET(num_terminal_subgraphs_);
    SHRG_DEBUG_RESET(num_passive_items_);
    SHRG_DEBUG_RESET(num_active_items_);
    SHRG_DEBUG_RESET(num_succ_merge_operations_);
    SHRG_DEBUG_RESET(num_total_merge_operations_);
    SHRG_DEBUG_RESET(num_indexing_keys_);

    matched_item_ptr_ = nullptr;
    items_pool_.Clear();
}

void PrecomputeBoundaryNodesForHRG(NodeMapping &boundary_nodes_of_hrg, const SHRG &grammar,
                                   const EdgeSet &matched_edges) {
    NodeMapping matched_nodes{}; // all macthed nodes
    for (auto &edge : grammar.fragment.edges)
        if (matched_edges[edge.index])
            for (auto node_ptr : edge.linked_nodes)
                matched_nodes[node_ptr->index] = 1;

    boundary_nodes_of_hrg.m8.fill(0);
    size_t node_count = grammar.fragment.nodes.size();
    for (size_t i = 0; i < node_count; ++i) {
        if (!matched_nodes[i])
            continue;
        const SHRG::Node &node = grammar.fragment.nodes[i];
        if (node.is_external) {
            boundary_nodes_of_hrg[i] = 1; // boundary node of SHRG
            continue;
        }
        boundary_nodes_of_hrg[i] = 0;
        for (const SHRG::Edge *edge_ptr : node.linked_edges)
            if (!matched_edges[edge_ptr->index]) {
                boundary_nodes_of_hrg[i] = 1; // there is an unmatched edge
                break;
            }
    }
}

} // namespace shrg

#include <iostream>

#include "parser_linear_base.hpp"

namespace shrg {
namespace linear {

using std::size_t;

void AttributesBase::Initialize(const SHRG &grammar) {
    grammar_ptr = &grammar;

    uint num_nonterminals = grammar.nonterminal_edges.size();
    if (num_nonterminals == 0)
        return;

    EdgeSet matched_edges = 0;
    for (auto edge_ptr : grammar.terminal_edges)
        matched_edges[edge_ptr->index] = true;

    boundary_nodes_of_steps.resize(num_nonterminals);
    for (uint i = 0; i < num_nonterminals; ++i) {
        const SHRG::Edge *edge_ptr = grammar.nonterminal_edges[i];

        matched_edges[edge_ptr->index] = true;
        // set boundary nodes of SHRG grammar for each step
        PrecomputeBoundaryNodesForHRG(boundary_nodes_of_steps[i], grammar, matched_edges);
    }
}

void LinearSHRGParserBase::CheckTerminalItems(AttributesBase *attrs_ptr, //
                                              const NodeMapping &original_node_mapping,
                                              const EdgeSet &edge_set) {
    const SHRG *grammar_ptr = attrs_ptr->grammar_ptr;
    const std::vector<SHRG::Node> &shrg_nodes = grammar_ptr->fragment.nodes;

    size_t shrg_node_count = shrg_nodes.size();
    int boundary_node_count = 0;
    NodeMapping node_mapping = original_node_mapping; // copy
    // all nodes linked with a terminal edge will be mapped now, but we should record those
    // boundary nodes only
    for (size_t i = 0; i < shrg_node_count; ++i) {
        uint8_t &index = node_mapping[i];
        // index == 0 means that, the node has only non-terminal edges
        if (index > 0) {
            const SHRG::Node &node = shrg_nodes[i];
            if (IsBoundaryNode(graph_ptr_->nodes[index - 1], edge_set)) {
                boundary_node_count++;
                if (!node.is_external && node.type == NodeType::kFixed)
                    return; // internal nodes shouldn't be mapped
            } else {
                if (node.is_external || node.type != NodeType::kFixed)
                    return; // external/boundary nodes should be mapped
                index = 0;
            }
        }
    }

    ChartItem *chart_item_ptr = items_pool_.Push(attrs_ptr, edge_set, node_mapping); //yg:BUG
    if (grammar_ptr->nonterminal_edges.empty())
        chart_item_ptr->status = boundary_node_count;

    attrs_ptr->terminal_items.push_back(chart_item_ptr);
    SHRG_DEBUG_INC(num_terminal_subgraphs_);
}

void LinearSHRGParserBase::MatchTerminalEdges(AttributesBase *attrs_ptr, //
                                              NodeMapping &node_mapping, EdgeSet &edge_set,
                                              NodeSet &node_set, uint index) {
    size_t edge_count = attrs_ptr->grammar_ptr->terminal_edges.size();
    if (index == edge_count) {
        CheckTerminalItems(attrs_ptr, node_mapping, edge_set); //yg:BUG
        return;
    }

    const SHRG::Edge *shrg_edge_ptr = attrs_ptr->grammar_ptr->terminal_edges[index];
    EdgeHash edge_hash = shrg_edge_ptr->Hash();

    assert(shrg_edge_ptr->is_terminal);
    size_t node_count = shrg_edge_ptr->linked_nodes.size();
    // get mapped nodes of current edges
    uint from = node_mapping[shrg_edge_ptr->linked_nodes[0]->index];
    uint to = 0;
    if (node_count > 1)
        to = node_mapping[shrg_edge_ptr->linked_nodes[1]->index];

    // all nodes of current edge are mapped
    if (from > 0 && (to > 0 || node_count == 1)) { // find the only one edge that may be matched
        auto it = terminal_complete_map_.find(MakeTerminalHash(edge_hash, from, to));
        if (it != terminal_complete_map_.end()) {
            int edge_index = it->second->index;
            if (!edge_set[edge_index]) {
                edge_set[edge_index] = true;
                MatchTerminalEdges(attrs_ptr, node_mapping, edge_set, node_set, index + 1);
                edge_set[edge_index] = false;
            }
        }
        return;
    }

    TerminalEdges *maybe_matched_edges_ptr = nullptr;
    if (from > 0 || to > 0) { // only one side of the edge is mapped
        auto it = terminal_partial_map_.find(MakeTerminalHash(edge_hash, from, to));
        if (it != terminal_partial_map_.end())
            maybe_matched_edges_ptr = &it->second;
    } else { // both sides of the edge are not mapped
        auto it = terminal_map_.find(edge_hash);
        if (it != terminal_map_.end())
            maybe_matched_edges_ptr = &it->second;
    }

    if (!maybe_matched_edges_ptr)
        return;

    for (const EdsGraph::Edge *edge_ptr : *maybe_matched_edges_ptr) {
        int edge_index = edge_ptr->index;
        if (edge_set[edge_index])
            continue;

        assert(edge_ptr->linked_nodes.size() == node_count);
        if (node_count == 1) {
            int shrg_index = shrg_edge_ptr->linked_nodes[0]->index;
            int eds_index = edge_ptr->linked_nodes[0]->index;
            if (node_set[eds_index]) // the node is already used
                continue;
            node_mapping[shrg_index] = eds_index + 1;
            node_set[eds_index] = true;

            edge_set[edge_index] = true;
            MatchTerminalEdges(attrs_ptr, node_mapping, edge_set, node_set, index + 1);
            edge_set[edge_index] = false;

            // restore. here we have from == 0, because the edge is not complete
            node_mapping[shrg_index] = from;
            node_set[eds_index] = false;
        } else { // node_count == 2
            int shrg_from_index = shrg_edge_ptr->linked_nodes[0]->index;
            int eds_from_index = edge_ptr->linked_nodes[0]->index;
            int shrg_to_index = shrg_edge_ptr->linked_nodes[1]->index;
            int eds_to_index = edge_ptr->linked_nodes[1]->index;
            if ((from == 0 && node_set[eds_from_index]) || (to == 0 && node_set[eds_to_index]))
                continue;
            node_mapping[shrg_from_index] = eds_from_index + 1;
            node_mapping[shrg_to_index] = eds_to_index + 1;
            node_set[eds_from_index] = true;
            node_set[eds_to_index] = true;

            edge_set[edge_index] = true;
            MatchTerminalEdges(attrs_ptr, node_mapping, edge_set, node_set, index + 1); //yg:BUG 2nd iter
            edge_set[edge_index] = false;

            node_mapping[shrg_from_index] = from;
            node_set[eds_from_index] = from > 0;
            node_mapping[shrg_to_index] = to;
            node_set[eds_to_index] = to > 0;
        }
    }
}

void LinearSHRGParserBase::InitializeChart() {
    // all terminal edges of current graph;
    for (const EdsGraph::Edge &edge : graph_ptr_->edges)
        if (edge.is_terminal) {
            size_t node_count = edge.linked_nodes.size();
            assert(node_count <= 2 && node_count >= 1);

            EdgeHash edge_hash = edge.Hash();
            TerminalHash complete_hash;

            int from = edge.linked_nodes[0]->index + 1;

            terminal_map_[edge_hash].push_back(&edge);
            if (node_count == 2) {
                int to = edge.linked_nodes[1]->index + 1;
                // record terminal edge with one of its node
                terminal_partial_map_[MakeTerminalHash(edge_hash, from, 0)].push_back(&edge);
                terminal_partial_map_[MakeTerminalHash(edge_hash, 0, to)].push_back(&edge);
                // record terminal edge with both of its nodes
                complete_hash = MakeTerminalHash(edge_hash, from, to);
            } else
                complete_hash = MakeTerminalHash(edge_hash, from, 0);

            assert(terminal_complete_map_.find(complete_hash) == terminal_complete_map_.end());
            terminal_complete_map_[complete_hash] = &edge;
        }
}

void LinearSHRGParserBase::ClearChart() {
    SHRGParserBase::ClearChart();

    terminal_map_.clear();
    terminal_complete_map_.clear();
    terminal_partial_map_.clear();
}

ChartItem *LinearGenerator::FindChartItemByEdge(ChartItem *chart_item_ptr,
                                                const SHRG::Edge *shrg_edge_ptr) {
    const SHRG *grammar_ptr = chart_item_ptr->attrs_ptr->grammar_ptr;
    uint num_nonterminals = grammar_ptr->nonterminal_edges.size();

    assert(num_nonterminals > 0);

    ChartItem *result_ptr = nullptr;
    for (int i = num_nonterminals - 1; i >= 0; --i) {
        const SHRG::Edge *edge_ptr = grammar_ptr->nonterminal_edges[i];
        if (edge_ptr == shrg_edge_ptr) {
            result_ptr = chart_item_ptr->right_ptr;
            break;
        }
        chart_item_ptr = chart_item_ptr->left_ptr;
    }

    assert(result_ptr && result_ptr->attrs_ptr->grammar_ptr->label == shrg_edge_ptr->label);
    return result_ptr;
}

} // namespace linear
} // namespace shrg

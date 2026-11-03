#include <iostream>
#include <random>

#include "parser_utils.hpp"

namespace shrg {

int RandomRange(int start, int end) {
    thread_local static std::random_device rd;
    thread_local static std::mt19937 gen(rd());
    return std::uniform_int_distribution<>(start, end - 1)(gen);
}

bool CheckAndChangeMappingFinally(const SHRG *grammar_ptr, uint boundary_node_count,
                                  NodeMapping &merged_mapping) {
    const std::vector<SHRG::Node *> &external_nodes = grammar_ptr->external_nodes;
    if (boundary_node_count != external_nodes.size()) // check boundary nodes
        return false;

    bool is_compatible = true;
    NodeMapping node_mapping{};
    for (size_t j = 0; j < external_nodes.size(); ++j) {
        // get correct index of current external node
        uint8_t eds_node_index = merged_mapping[external_nodes[j]->index];
        if (eds_node_index == 0) { // TODO: necessary ???
            is_compatible = false;
            break;
        }
        assert(eds_node_index > 0); // External node maps to strange position
        // We should not use `node_mapping[external_nodes[j]->index] = eds_node_index`. Because the
        // passive_item will be use in other grammars, so the index of the this external node is
        // not the same as the other ones. Here we only record the j-th external node maps to the
        // `eds_node_index`-th eds node.
        node_mapping[j] = eds_node_index;
    }
    if (!is_compatible)
        return false;

    merged_mapping = node_mapping;

    return true;
}

inline int MergeMappings(const EdsGraph *graph_ptr, uint shrg_node_count, //
                         const NodeMapping &right_mapping,                //
                         NodeMapping &merged_mapping, const EdgeSet &merged_edge_set,
                         const NodeMapping &boundary_nodes_of_hrg) {
    // TODO: optimize for empty graph
    NodeSet node_set = 0;
    int boundary_node_count = 0;
    for (uint node_index = 0; node_index < shrg_node_count; ++node_index) {
        uint8_t right_eds_index = right_mapping[node_index];    // index of node in EdsGraph
        uint8_t &merged_eds_index = merged_mapping[node_index]; // index of node in EdsGraph (ref)
        if (merged_eds_index > 0) {
            if (right_eds_index > 0) {
                // case #1: merged_eds_index > 0 && right_eds_index > 0
                if (merged_eds_index != right_eds_index) // the two bijection is not compatible
                    return -1;
                // check whether current node is a boundary node of merged subgraph
                if (!IsBoundaryNode(graph_ptr->nodes[merged_eds_index - 1], merged_edge_set)) {
                    // current node is a internal node of merged subgraph, remove it from
                    // node_mapping
                    if (boundary_nodes_of_hrg[node_index])
                        return -1; // but this node is a boundary node in SHRG Grammar
                    merged_eds_index = 0;
                }
            }
            // case #2: merged_eds_index > 0 && right_eds_index is 0
            // current eds node is a boundary node of left subgraph but is not a boundary node
            // of right one, so current eds node is a boundary node of merged subgraph (in EDS).
            // for the same reason, the corresponding shrg node is a boundary node (in SHRG).
        } else if (right_eds_index > 0) {
            // case #3: merged_eds_index = 0 && right_eds_index > 0
            // current node is a boundary node of right subgraph but is not a boundary node of
            // left one, so current node is a boundary node of merged subgraph. for the same
            // reason, the corresponding shrg node is a boundary node (in SHRG).
            merged_eds_index = right_eds_index;
        }

        if (merged_eds_index > 0) {
            if (!boundary_nodes_of_hrg[node_index]) // this node should not be a boundary anymore
                return -1;
            if (node_set[merged_eds_index - 1]) // two different SHRG nodes map to a same Eds node
                return -1;
            node_set[merged_eds_index - 1] = true;
            ++boundary_node_count;
        }
    }
    return boundary_node_count;
}

int MergeTwoChartItems(const EdsGraph *graph_ptr,                                       //
                       const ChartItem *left_item_ptr, const ChartItem *right_item_ptr, //
                       uint shrg_node_count, EdgeSet &merged_edge_set, NodeMapping &merged_mapping,
                       const NodeMapping &boundary_nodes_of_hrg) {
    // *IMPORTANT*
    // boundary_node_mapping of left/right subgraph is index-of-SHRG-node => index-of-EDS-node
    if ((left_item_ptr->edge_set & right_item_ptr->edge_set).any())
        return -1; // these two subgraphs are not disjoint

    // merge subgraphs in a grammar, so the index of SHRG node is the same
    merged_mapping = left_item_ptr->boundary_node_mapping;
    merged_edge_set = left_item_ptr->edge_set | right_item_ptr->edge_set;

    return MergeMappings(graph_ptr, shrg_node_count, //
                         right_item_ptr->boundary_node_mapping, merged_mapping, merged_edge_set,
                         boundary_nodes_of_hrg);
}

int MergeTwoChartItems(const EdsGraph *graph_ptr,      //
                       const ChartItem *left_item_ptr, //
                       const ChartItem *right_item_ptr, const SHRG::Edge *left_edge_ptr,
                       uint shrg_node_count, EdgeSet &merged_edge_set, NodeMapping &merged_mapping,
                       const NodeMapping &boundary_nodes_of_hrg) {
    // *IMPORTANT*
    // boundary_node_mapping of left subgraph is index-of-external-node => index-of-EDS-node
    // boundary_node_mapping of right subgraph is index-of-SHRG-node => index-of-EDS-node
    if ((left_item_ptr->edge_set & right_item_ptr->edge_set).any())
        return -1; // these two subgraphs are not disjoint

    // left subgraph is a passive item, its node mapping should be computed specially. the i-th
    // slot of `left_mapping` is aligned to i-th node of `left_edge_ptr->linked_nodes`
    assert(left_edge_ptr); // left subgraph is a passive item, but its edge is nullptr
    int external_node_index = 0;
    const NodeMapping &left_mapping = left_item_ptr->boundary_node_mapping;
    for (const SHRG::Node *external_node_ptr : left_edge_ptr->linked_nodes)
        merged_mapping[external_node_ptr->index] = left_mapping[external_node_index++];

    merged_edge_set = left_item_ptr->edge_set | right_item_ptr->edge_set;

    return MergeMappings(graph_ptr, shrg_node_count, //
                         right_item_ptr->boundary_node_mapping, merged_mapping, merged_edge_set,
                         boundary_nodes_of_hrg);
}

} // namespace shrg

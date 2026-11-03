#pragma once

#include "parser_chart_item.hpp"
#include "synchronous_hyperedge_replacement_grammar.hpp"

namespace shrg {

using TerminalHash = std::uint64_t;
using TerminalEdges = std::vector<const EdsGraph::Edge *>;
using SmallKey = EdgeHash;
using MediumKey = NodeMapping;
struct LargeKey {
    EdgeHash edge_hash;
    NodeMapping node_mapping;
};

struct DerivationNode {
    static const int Empty = -1;
    static const int Error = -2;

    const SHRG *grammar_ptr;
    const SHRG::CFGRule *cfg_ptr;
    const ChartItem *item_ptr;

    std::vector<int> children;
};
using Derivation = std::vector<DerivationNode>;

inline bool operator==(const LargeKey &key1, const LargeKey &key2) {
    return key1.edge_hash == key2.edge_hash && key1.node_mapping == key2.node_mapping;
}

inline TerminalHash MakeTerminalHash(EdgeHash edge_hash, int from, int to) {
    return (static_cast<TerminalHash>(edge_hash) << 32) | (from << 16) | to;
}

inline NodeMapping GetMapping(const std::vector<SHRG::Node *> &linked_nodes,
                              const NodeMapping &input) {
    NodeMapping output{};
    std::size_t size = linked_nodes.size();
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = input[linked_nodes[i]->index];
        assert(output[i] > 0);
    }
    return output;
}

inline NodeMapping GetMapping(const SHRG::Edge *edge_ptr, const NodeMapping &input) {
    NodeMapping output{};
    auto &nodes = edge_ptr->linked_nodes;
    std::size_t size = nodes.size();
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = input[nodes[i]->index];
        assert(output[i] > 0);
    }
    return output;
}

inline NodeMapping MaskedNodeMapping(const SHRG::Edge *edge_ptr, const NodeMapping &node_mapping,
                                     const NodeMapping &mask) {
    NodeMapping output = mask;
    auto &nodes = edge_ptr->linked_nodes;
    std::size_t size = nodes.size();

    for (std::size_t i = 0; i < size; ++i)
        output[i] &= node_mapping[nodes[i]->index];
    return output;
}

inline NodeMapping MaskedNodeMapping(const NodeMapping &node_mapping, const NodeMapping &mask) {
    return {.m8 = {node_mapping.m8[0] & mask.m8[0], // node mapping
                   node_mapping.m8[1] & mask.m8[1]}};
}

template <typename NodeType>
inline bool IsBoundaryNode(const NodeType &node, const EdgeSet &edge_set) {
    for (auto *eds_edge_ptr : node.linked_edges)
        if (!edge_set[eds_edge_ptr->index])
            return true;
    return false;
}

template <typename Set>
inline bool IsGrammarCompatiable(const SHRG &grammar, const Set &terminal_edges_set) {
    // terminal_edges_set of grammar is much smaller than the one of input graph
    for (EdgeHash edge_hash : grammar.terminal_edges_set)
        if (terminal_edges_set.find(edge_hash) == terminal_edges_set.end()) {
            // there is a terminal edge in grammar but not in edsgraph
            return false; // this grammar won't be used in this graph
        }
    return true;
}

int RandomRange(int start, int end);

// check whther a merged_mapping (the boundary_node_mapping of a chart_item) is valid for
// grammar_ptr (the boundary nodes of merged_mapping are exactly the external nodes of
// grammar_ptr).  Then change merged_mapping to standard format
bool CheckAndChangeMappingFinally(const SHRG *grammar_ptr, uint boundary_node_count,
                                  NodeMapping &merged_mapping);

// *IMPORTANT*
// boundary_node_mapping of left/right chart_item is index-of-SHRG-node => index-of-EDS-node
int MergeTwoChartItems(const EdsGraph *graph_ptr,                                       //
                       const ChartItem *left_item_ptr, const ChartItem *right_item_ptr, //
                       uint shrg_node_count, EdgeSet &merged_edge_set, NodeMapping &merged_mapping,
                       const NodeMapping &boundary_nodes_of_hrg);

// *IMPORTANT*
// boundary_node_mapping of left chart_item is index-of-external-node => index-of-EDS-node
// boundary_node_mapping of right chart_item is index-of-SHRG-node => index-of-EDS-node
// return the count of boundary nodes or -1 if failed
int MergeTwoChartItems(const EdsGraph *graph_ptr, //
                       const ChartItem *left_item_ptr,
                       const ChartItem *right_item_ptr, //
                       const SHRG::Edge *left_edge_ptr, //
                       uint shrg_node_count, EdgeSet &merged_edge_set, NodeMapping &merged_mapping,
                       const NodeMapping &boundary_nodes_of_hrg);

template <typename ValueType> class ChartItemMap {
    static_assert(sizeof(EdgeHash) == 8, "EdgeHash should be 64-bit long");

  public:
    void Clear() {
        small_map_.clear();
        medium_map_.clear();
        large_map_.clear();
    }

    ValueType &At(const SHRG::Edge *edge_ptr, const NodeMapping &node_mapping) {
        return At(edge_ptr->Hash(), node_mapping, edge_ptr->linked_nodes.size());
    }

    ValueType &At(EdgeHash edge_hash, const NodeMapping &node_mapping, uint boundary_node_count) {
        if (boundary_node_count <= 4)
            return small_map_[(edge_hash << 32) | node_mapping.m4[0]];

        if (boundary_node_count <= 12) {
            NodeMapping key = node_mapping; // copy
            key.m4[3] = edge_hash;
            return medium_map_[key];
        }

        return large_map_[{edge_hash, node_mapping}];
    }

    const std::unordered_map<SmallKey, ValueType> &Small() const { return small_map_; };
    const std::unordered_map<MediumKey, ValueType> &Medium() const { return medium_map_; };
    const std::unordered_map<LargeKey, ValueType> &Large() const { return large_map_; };

  private:
    std::unordered_map<SmallKey, ValueType> small_map_;
    std::unordered_map<MediumKey, ValueType> medium_map_;
    std::unordered_map<LargeKey, ValueType> large_map_;
};

} // namespace shrg

namespace std {

using shrg::LargeKey;
using shrg::MediumKey;
template <> struct hash<LargeKey> {
    // !!! the hash function must be mark as const
    std::size_t operator()(const LargeKey &key) const {
        auto hash_value = key.edge_hash;
        boost::hash_combine(hash_value, key.node_mapping);
        return hash_value;
    }
};

} // namespace std

// Local Variables:
// mode: c++
// End:

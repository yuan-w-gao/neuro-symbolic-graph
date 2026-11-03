#pragma once

#include <unordered_set>

#include "../include/basic.hpp"

#include "hyper_graph.hpp"

namespace shrg {

[[maybe_unused]] const int MAX_SHRG_EDGE_COUNT = 32;
[[maybe_unused]] const int MAX_SHRG_NODE_COUNT = 16;

class SHRG {
  public:
    static const int kNone = -1;
    static const int kStrong = -2;
    static const int kWeak = 0;

    using Fragment = HyperGraph<>;
    using Edge = Fragment::Edge;
    using Node = Fragment::Node;
    using LabelHash = std::int64_t;

    struct CFGItem {
        Label label = EMPTY_LABEL;
        // In synchronous grammar, each CFG term should be aligned to a HRG hyperedge
        Edge *aligned_edge_ptr = nullptr;
    };

    struct CFGRule {
        Label label;                // label of CFG rule
        std::vector<CFGItem> items; // right hand side of CFG rule
        int shrg_index;             // index of this rule
        float score = 1e-20;

        bool operator<(const CFGRule &other) const { return score < other.score; }
    };

    Label label = EMPTY_LABEL;          // label of grammar
    LabelHash label_hash;               // cached hash of grammar label
    Fragment fragment;                  // hyper graph for HRG part
    std::vector<Node *> external_nodes; // external node for `fragment`

    std::unordered_set<EdgeHash> terminal_edges_set; // all terminal edge labels
    std::vector<Edge *> terminal_edges;              // all terminal edges in `fragment`
    std::vector<Edge *> nonterminal_edges;           // all nonterminal edges in `fragment`

    std::vector<CFGRule> cfg_rules; // A HRG rule can be aligned to multiple CFG rules

    // below two fields are used in selection of the best SHRG rule in a weighted model
    const CFGRule *best_cfg_ptr;
    int num_occurences;

    double log_rule_weight;
    double log_count = -std::numeric_limits<double>::infinity();
//    double log_count = 0;
    double prev_rule_weight = 0.0;
    int used = -1;

    // When the `fragment` is an empty graph, the grammar is marked as filtered
    bool IsEmpty() const { return fragment.nodes.empty(); }

    // load grammars from `input_file` into vector `grammars`
    static int Load(const std::string &input_file, std::vector<SHRG> &grammars,
                    TokenSet &token_set);

    // mark all disconnected grammars as filtered
    static void FilterDisconneted(std::vector<SHRG> &grammars);
};

} // namespace shrg

// Local Variables:
// mode: c++
//  End:

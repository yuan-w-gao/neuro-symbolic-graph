#pragma once

#include "generator.hpp"
#include "parser_base.hpp"

namespace shrg {
namespace linear {

struct AttributesBase : public GrammarAttributes {
    ChartItemList terminal_items; // items only have terminal edges

    std::vector<NodeMapping> boundary_nodes_of_steps;

    void Initialize(const SHRG &grammar);
};

class LinearGenerator : public Generator {
  public:
    using Generator::Generator;

    ChartItem *FindChartItemByEdge(ChartItem *subgraph_ptr,
                                   const SHRG::Edge *shrg_edge_ptr) override;
};

class LinearSHRGParserBase : public SHRGParserBase {
  protected:
    LinearGenerator generator_;

    // quick lookup for terminal edges
    std::unordered_map<EdgeHash, TerminalEdges> terminal_map_;
    std::unordered_map<TerminalHash, TerminalEdges> terminal_partial_map_;
    std::unordered_map<TerminalHash, const EdsGraph::Edge *> terminal_complete_map_;

    void CheckTerminalItems(AttributesBase *agenda_ptr, //
                            const NodeMapping &node_mapping, const EdgeSet &edge_set);

    void MatchTerminalEdges(AttributesBase *agenda_ptr, //
                            NodeMapping &node_mapping, EdgeSet &edge_set, NodeSet &node_set,
                            uint index);

    void InitializeChart();

    void ClearChart();

  public:
    LinearSHRGParserBase(const char *parser_type, const std::vector<SHRG> &grammars,
                         const TokenSet &label_set)
        : SHRGParserBase(parser_type, grammars, label_set), generator_(this) {}

    Generator *GetGenerator() override { return &generator_; }
};

} // namespace linear
} // namespace shrg

// Local Variables:
// mode: c++
// End:

#pragma once

#include "parser_base.hpp"

namespace shrg {

class Generator {
  protected:
    const SHRGParserBase *parser_ = nullptr;

    virtual float GetScoreOfChilren(ChartItem *current_ptr);

    bool MatchTerminalEdges(NodeMapping &merged_mapping, const SHRG *grammar_ptr, EdgeSet &edge_set,
                            uint index);

  public:
    // Find best chart_item in a cycle list recursively
    float FindBestChartItem(ChartItem *chart_item_ptr);

    Generator(const SHRGParserBase *parser) : parser_(parser) {}

    virtual ~Generator() {}

    virtual ChartItem *FindChartItemByEdge(ChartItem *chart_item_ptr,
                                           const SHRG::Edge *shrg_edge_ptr) = 0;

    int Generate(ChartItem *chart_item_ptr, Derivation &derivation, std::string &sentence);

    std::size_t CountChartItems(ChartItem *chart_item_ptr);

    ChartItem *BestResult();

    const EdsGraph *Graph() const { return parser_->graph_ptr_; }
};

} // namespace shrg

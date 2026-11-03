#pragma once

#include <queue>
#include <unordered_map>

#include "parser_linear_base.hpp"

namespace shrg {
namespace linear {

struct Agenda;
struct Attributes : public AttributesBase {
    const std::vector<NodeMapping> *required_masks = nullptr;

    std::vector<NodeMapping> edge_masks;

    void Clear() { terminal_items.clear(); }
};

struct Agenda {
    bool in_queue = false;

    struct ActiveItem {
        ChartItem *chart_item_ptr;
        uint index;
    };

    uint num_visited_passive_items = 0;
    uint num_visited_active_items = 0;

    ChartItemSet passive_items;
    std::vector<ActiveItem> active_items;

    void Clear() {
        in_queue = false;

        num_visited_passive_items = 0;
        num_visited_active_items = 0;

        passive_items.Clear();
        active_items.clear();
    }
};

class LinearSHRGParser : public LinearSHRGParserBase {
  private:
    std::unordered_map<EdgeHash, std::vector<NodeMapping>> all_required_masks_;

    // each SHRG grammar has a corresponding Attributes
    std::vector<Attributes> attributes_;
    ChartItemMap<Agenda> agendas_;
    std::queue<Agenda *> updated_agendas_; // chart agenda

    void ClearChart();

    void InitializeChart();

    void EmitSubGraph(ChartItem *chart_item_ptr, uint boundary_node_count, Attributes *attrs_ptr);

    bool MatchTerminalEdges(Attributes *attrs_ptr);

    void EnableGrammar(Attributes *attrs_ptr);

    void MergeItems(Agenda::ActiveItem &item, ChartItem *chart_item_ptr);

    void UpdateAgenda(Agenda *agenda_ptr);

  public:
    LinearSHRGParser(const std::vector<SHRG> &grammars, const TokenSet &label_set);

    ParserError Parse(const EdsGraph &graph) override;
};

} // namespace linear
} // namespace shrg

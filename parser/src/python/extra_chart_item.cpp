#include <sstream>

#include "../manager.hpp"
#include "extra_chart_item.hpp"

namespace shrg {

Partition ChartItem_ToList(const ChartItem &self, const EdsGraph &graph) {
    Partition result;
    EdgeSet_ToNodeList(self.edge_set, self.boundary_node_mapping, graph, result);
    return result;
}

void ChartItem_ToList(const ChartItem &self, const EdsGraph &graph, Partition &result) {
    EdgeSet_ToNodeList(self.edge_set, self.boundary_node_mapping, graph, result);
}

std::string ChartItem_ToString(const ChartItem &self, const EdsGraph &graph) {
    std::ostringstream os;
    if (self.attrs_ptr) {
        debug::Printer printer{&graph, Manager::manager.label_set};
        printer.Ln(*self.attrs_ptr->grammar_ptr, 0, false /* print_cfg_rules */, os);
    } else
        os << "No grammar:\n";
    return os.str();
}

std::string ChartItem_ToDot(const ChartItem &self, const EdsGraph &graph) {
    std::ostringstream os;
    debug::AttributesMap node_attrs_map;
    debug::AttributesMap edge_attrs_map;
    auto &edge_set = self.edge_set;
    auto &mapping = self.boundary_node_mapping;
    for (uint i = 0; i < edge_set.size(); ++i) {
        if (!edge_set[i])
            continue;
        edge_attrs_map[i]["color"] = debug::COLOR_CURRNENT;
    }
    for (auto v : mapping.m1)
        if (v > 0)
            node_attrs_map[v - 1]["shape"] = "rectangle";
    Label label = EMPTY_LABEL;
    if (self.attrs_ptr)
        label = self.attrs_ptr->grammar_ptr->label;

    auto &label_set = Manager::manager.label_set;
    os << "digraph { label=\"" << label_set[label] << "\" ";
    debug::Printer{nullptr, label_set}.DrawGraph(graph, "n", node_attrs_map, edge_attrs_map, os);
    os << "}";
    return os.str();
}

int ChartItem_GrammarIndex(const ChartItem &self) {
    return self.attrs_ptr ? self.attrs_ptr->grammar_ptr - Manager::manager.grammars.data() : -1;
}

} // namespace shrg

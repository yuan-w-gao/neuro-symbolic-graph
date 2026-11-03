#pragma once

#include <chrono>

#include "parser_utils.hpp"

#define SHRG_DEBUG_INC(var) (++(var))
#define SHRG_DEBUG_DEC(var) (--(var))
#define SHRG_DEBUG_RESET(var) ((var) = 0)

#define SHRG_DEBUG_START_TIMER() auto _start_time = std::chrono::system_clock::now();
#define SHRG_DEBUG_REPORT_TIMER(msg)                                                               \
    {                                                                                              \
        std::chrono::duration<double> elapsed_seconds =                                            \
            std::chrono::system_clock::now() - _start_time;                                        \
        LOG_INFO(msg " time cost: " << elapsed_seconds.count() << 's');                            \
        _start_time = std::chrono::system_clock::now();                                            \
    }

namespace shrg {
namespace tree {

class TreeNodeBase;
using Tree = std::vector<TreeNodeBase *>;

} // namespace tree

namespace debug {

const std::string COLOR_CURRNENT = "red";
const std::string COLOR_LEFT = "blue";
const std::string COLOR_RIGHT = "green";
const std::string COLOR_MATCHED = "cyan";

using Attributes = std::unordered_map<std::string, std::string>;
using AttributesMap = std::unordered_map<int, Attributes>;

std::ostream &Indent(int indent, std::ostream &os, bool fill_last_column = true);

struct Printer {
    const EdsGraph *graph_ptr;
    const TokenSet &label_set;

    std::string ToString(const MediumKey &key);
    std::string ToString(const LargeKey &key);
    std::string ToString(const SmallKey &key);

    void Ln(const ChartItem &chart_item, const SHRG::Edge *shrg_edge_ptr = nullptr, int indent = 0,
            std::ostream &os = std::cout);
    void Ln(const SHRG &grammar, int indent = 0, bool print_cfg_rules = false,
            std::ostream &os = std::cout);
    void Ln(const Derivation &derivation, const DerivationNode &node, int indent = 0,
            std::ostream &os = std::cout);
    void Ln(const tree::Tree &tree, std::ostream &os = std::cout);

    void operator()(const SHRG::CFGRule &cfg_rule, uint external_node_count,
                    std::ostream &os = std::cout);
    void operator()(const NodeMapping &node_mapping, std::ostream &os = std::cout);
    void operator()(const EdgeSet &edge_set, std::ostream &os = std::cout);
    void operator()(const EdsGraph::Node *node_ptr, std::ostream &os = std::cout);
    void operator()(const SHRG::Edge *ptr, std::ostream &os = std::cout);
    void operator()(const EdsGraph::Edge *ptr, std::ostream &os = std::cout);

    template <typename Graph>
    void DrawGraph(const Graph &graph, const char *prefix = "n",
                   const AttributesMap &node_attrs_map = {}, //
                   const AttributesMap &edge_attrs_map = {}, std::ostream &os = std::cout);
};

} // namespace debug
} // namespace shrg

// Local Variables:
// mode: c++
// End:

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <sys/stat.h>

#include "parser_base.hpp"
#include "parser_debug.hpp"
#include "tree_decomposer.hpp"

namespace shrg {
namespace debug {

const char *INDENT_STRING1 = "│   ";
const char *INDENT_STRING2 = "├──";
const char *INDENT_STRING3 = "└──";

std::ostream &Indent(int indent, std::ostream &os, bool fill_last_column) {
    for (int i = 1; i < indent; ++i)
        os << INDENT_STRING1;
    if (indent > 0)
        os << (fill_last_column ? INDENT_STRING2 : INDENT_STRING1);
    return os;
}

template <typename Graph>
void Printer::DrawGraph(const Graph &graph, const char *prefix,                                   //
                        const AttributesMap &node_attrs_map, const AttributesMap &edge_attrs_map, //
                        std::ostream &os) {
    static const Attributes empty_attrs;
    static const std::unordered_map<int, std::string> g_shapes{{3, "invtriangle"}, {4, "diamond"},
                                                               {5, "star"},        {6, "hexagon"},
                                                               {7, "polygon"},     {8, "octagon"}};

    for (auto &node : graph.nodes) {
        std::string node_name = prefix + std::to_string(node.index);
        auto it_attrs = node_attrs_map.find(node.index);
        const Attributes &node_attrs =
            (it_attrs == node_attrs_map.end() ? empty_attrs : it_attrs->second);
        os << node_name << "[width=\"0.35\" height=\"0.35\" fixedsize=true label=" << node.index
           << " ";
        for (auto &attr : node_attrs)
            os << attr.first << "=\"" << attr.second << "\" ";
        os << "] ";
    }

    for (auto &edge : graph.edges) {
        auto it_attrs = edge_attrs_map.find(edge.index);
        const Attributes &edge_attrs =
            (it_attrs == edge_attrs_map.end() ? empty_attrs : it_attrs->second);
        switch (edge.linked_nodes.size()) {
        case 1: {
            os << prefix << edge.linked_nodes[0]->index << "->" //
               << prefix << edge.linked_nodes[0]->index << "_end[label=\"" << label_set[edge.label]
               << "\" ";
            for (auto &attr : edge_attrs)
                os << attr.first << "=\"" << attr.second << "\" ";
            os << "] ";
            os << prefix << edge.linked_nodes[0]->index
               << "_end[width=\"0.005\" height=\"0.005\" fixedsize=true color=white label=\"\"] ";
            break;
        }
        case 2: {
            os << prefix << edge.linked_nodes[0]->index << "->" //
               << prefix << edge.linked_nodes[1]->index << "[label=\"" << label_set[edge.label]
               << "\" ";
            for (auto &attr : edge_attrs)
                os << attr.first << "=\"" << attr.second << "\" ";
            os << "] ";
            break;
        }
        default:
            std::size_t size = edge.linked_nodes.size();
            auto it_shape = g_shapes.find(size);
            os << "_" << &edge
               << "[shape=" << (it_shape == g_shapes.end() ? "doublecircle" : it_shape->second)
               << "] ";
            for (std::size_t i = 0; i < size; ++i) {
                os << "_" << &edge << "->" << prefix << edge.linked_nodes[i]->index;
                if (!edge_attrs.empty()) {
                    os << "[";
                    for (auto &attr : edge_attrs)
                        os << attr.first << "=\"" << attr.second << "\" ";
                    os << "] ";
                } else
                    os << " ";
            }
            break;
        }
    }
}

template void Printer::DrawGraph<EdsGraph>(const EdsGraph &graph, const char *,
                                           const AttributesMap &, const AttributesMap &,
                                           std::ostream &);
template void Printer::DrawGraph<SHRG::Fragment>(const SHRG::Fragment &graph, const char *,
                                                 const AttributesMap &, const AttributesMap &,
                                                 std::ostream &);

std::string Printer::ToString(const SmallKey &key) {
    std::ostringstream sin;
    sin << '[';
    if (key >> 32) {
        sin << label_set[key >> 40] << '#' << ((key >> 34) & 0x3f);
        union {
            std::int8_t mapping[4];
            std::uint32_t value;
        } key_;
        key_.value = key & 0xffffffff;
        for (int i = 0; i < 4; ++i)
            sin << ' ' << (int)key_.mapping[i];
    } else {
        sin << label_set[key >> 8] << '#' << ((key >> 2) & 0x3f);
    }
    sin << ']';
    return sin.str();
}

std::string Printer::ToString(const MediumKey &key) {
    std::ostringstream sin;
    sin << '[';
    int edge_hash = key.m4[3];
    sin << label_set[edge_hash >> 8] << '#' << ((edge_hash >> 2) & 0x3f);
    for (int i = 0; i < 12; ++i)
        sin << ' ' << (int)key[i];
    sin << ']';
    return sin.str();
}

std::string Printer::ToString(const LargeKey &key) {
    std::ostringstream sin;
    sin << '[';
    sin << label_set[key.edge_hash >> 8] << '#' << ((key.edge_hash >> 2) & 0x3f);
    for (int8_t v : key.node_mapping.m1)
        sin << ' ' << (int)v;
    sin << ']';
    return sin.str();
}

void Printer::operator()(const NodeMapping &node_mapping, std::ostream &os) {
    os << '[';
    for (auto index : node_mapping.m1)
        os << (int)index << ' ';
    os << ']';
}

void Printer::operator()(const EdgeSet &edge_set, std::ostream &os) {
    os << '{';
    auto size = edge_set.size();
 //   auto i = edge_set._Find_first();
//    while (i < size) {
    for (auto i = 0; i < size; ++i) {
        if (!edge_set[i])
            continue;
        os << i;
        //i = edge_set._Find_next(i);
        if (i < size)
            os << ',';
    }
    os << '}';
}

void Printer::operator()(const SHRG::Edge *edge_ptr, std::ostream &os) {
    if (!edge_ptr)
        os << "#?<...>";
    else {
        os << '<' << label_set[edge_ptr->label] << ": ";
        auto begin = edge_ptr->linked_nodes.begin();
        auto end = edge_ptr->linked_nodes.end();
        for (auto it = begin; it != end; ++it) {
            os << (*it)->index;
            if (it != end - 1)
                os << " -- ";
        }
        os << (edge_ptr->is_terminal ? " Y" : " N") << '@' << edge_ptr->index << '>';
    }
}

void Printer::operator()(const EdsGraph::Edge *edge_ptr, std::ostream &os) {
    if (!edge_ptr)
        os << "#?<...>";
    else {
        os << '<' << label_set[edge_ptr->label] << ": ";
        auto begin = edge_ptr->linked_nodes.begin();
        auto end = edge_ptr->linked_nodes.end();
        for (auto it = begin; it != end; ++it) {
            os << (*it)->id << '/' << (*it)->index;
            if (it != end - 1)
                os << " -- ";
        }
        os << (edge_ptr->is_terminal ? " Y" : " N") << '@' << edge_ptr->index << '>';
    }
}

void Printer::operator()(const EdsGraph::Node *node_ptr, std::ostream &os) {
    os << "<GraphNode: " << label_set[node_ptr->label] << '(' << node_ptr->carg << ")>";
}

void Printer::operator()(const SHRG::CFGRule &cfg_rule, uint external_node_count,
                         std::ostream &os) {
    os << label_set[cfg_rule.label] << '#' << external_node_count << " => ";
    auto begin = cfg_rule.items.begin();
    auto end = cfg_rule.items.end();
    for (auto it = begin; it != end; ++it) {
        const SHRG::Edge *edge_ptr = it->aligned_edge_ptr;
        os << label_set[it->label];
        if (edge_ptr && !edge_ptr->is_terminal)
            os << '#' << edge_ptr->linked_nodes.size();
        if (it != end - 1)
            os << " + ";
    }
}

void Printer::Ln(const SHRG &grammar, int indent, bool print_cfg_rules, std::ostream &os) {
    if (grammar.IsEmpty())
        Indent(indent, os) << "No semantic part\n";
    else {
        const SHRG::Fragment &fragment = grammar.fragment;
        Indent(indent, os) << "HRG-RHS: NodeCount=" << fragment.nodes.size()
                           << " EdgeCount=" << fragment.edges.size()
                           << " Label=" << label_set[grammar.label] << '\n';

        int index = 0;
        Indent(indent, os);
        for (const SHRG::Edge &edge : fragment.edges) {
            os << '#' << index++;
            this->operator()(&edge);
            os << ' ';
        }
        Indent(indent, os << '\n') << "EPs: ";
        for (const SHRG::Node *node_ptr : grammar.external_nodes)
            os << ' ' << node_ptr->index;
        os << '\n';
    }

    Indent(indent, os) << "CFG parts: " << grammar.cfg_rules.size() << '\n';
    if (!print_cfg_rules)
        return;

    for (const SHRG::CFGRule &cfg_rule : grammar.cfg_rules) {
        Indent(indent, os << cfg_rule.shrg_index << " ");
        this->operator()(cfg_rule, grammar.external_nodes.size());
        os << '\n';
    }
}

void PrintDerivation(Printer &printer, //
                     const Derivation &derivation, const DerivationNode &node, int indent,
                     std::stringstream &ss, //
                     std::vector<std::string> &lines, std::vector<std::string> &words) {
    static std::set<std::string> lexical_labels = {"pron",     "named",   "card",       "yofc",
                                                   "fraction", "season",  "year_range", "mofy",
                                                   "dofm",     "named_n", "much-many_a"};

    ss.str("");
    printer(*node.cfg_ptr, node.grammar_ptr->external_nodes.size(), Indent(indent, ss));
    lines.emplace_back(ss.str());

    if (std::all_of(node.children.begin(), node.children.end(),
                    [](int index) { return index == DerivationNode::Empty; })) {
        words.emplace_back();
        auto &word = words.back();
        for (auto &item : node.cfg_ptr->items) {
            word.append(printer.label_set[item.label]);
            word.push_back(' ');
        }
        return;
    }

    auto &edges = node.grammar_ptr->terminal_edges;
    const SHRG::Edge *edge_ptrs[MAX_SHRG_EDGE_COUNT];
    int found_count = 0;
    for (auto edge_ptr : edges)
        if (edge_ptr->linked_nodes.size() == 1) {
            auto &label = printer.label_set[edge_ptr->label];
            if ((!label.empty() && label[0] == '_') || lexical_labels.count(label))
                edge_ptrs[found_count++] = edge_ptr;
        }
    if (found_count > 0) {
        ss.str("");
        ss << " [ ";
        for (int i = 0; i < found_count; ++i) {
            printer(edge_ptrs[i], ss);
            ss << ' ';
        }
        ss << ']';
        lines.back().append(ss.str());
    }

    words.emplace_back("");
    for (auto index : node.children)
        if (index != DerivationNode::Empty)
            PrintDerivation(printer, derivation, derivation[index], indent + 1, ss, lines, words);
}

void Printer::Ln(const Derivation &derivation, const DerivationNode &node, int indent,
                 std::ostream &os) {
    std::stringstream ss;
    std::vector<std::string> lines, words;
    PrintDerivation(*this, derivation, node, indent, ss, lines, words);
    auto iter = std::max_element(words.begin(), words.end(),
                                 [](auto &w1, auto &w2) { return w1.size() < w2.size(); });
    for (uint i = 0; i < words.size(); ++i)
        os << std::left << std::setw(iter->size()) << words[i] << lines[i] << '\n';
}

void Printer::Ln(const ChartItem &chart_item, const SHRG::Edge *shrg_edge_ptr, int indent,
                 std::ostream &os) {
    if (!chart_item.edge_set.any())
        Indent(indent, os) << "Empty graph\n";
    else {
        const EdgeSet &edge_set = chart_item.edge_set;
        const NodeMapping &node_mapping = chart_item.boundary_node_mapping;
        Indent(indent, os) << "Edge: ";
        this->operator()(shrg_edge_ptr, os);
        os << "\n";
        Indent(indent, os) << "Matched edges:";

        int width = 0;
        for (size_t i = 0; i < edge_set.size(); ++i) {
            if (width++ == 0)
                Indent(indent, os << '\n') << "   ";
            this->operator()(&graph_ptr->edges[i], os);
            os << ' ';
            if (width == 5)
                width = 0;
        }

        Indent(indent, os << '\n') << "Indices: ";
        for (size_t i = 0; i < edge_set.size(); ++i) {
            if (!edge_set[i])
              continue;
            auto &nodes = graph_ptr->edges[i].linked_nodes;
            if (nodes.size() == 1)
                os << nodes[0]->id << '/' << nodes[0]->index << ' ';
        }
        Indent(indent, os << '\n') << "HRG <> EDS: ";
        for (size_t i = 0; i < node_mapping.size(); ++i) {
            if (node_mapping[i] > 0) {
                os << '[';
                if (!shrg_edge_ptr)
                    os << i;
                else {
                    if (i >= shrg_edge_ptr->linked_nodes.size()) {
                        LOG_DEBUG("Strange ???");
                    }
                    os << shrg_edge_ptr->linked_nodes[i]->index;
                }
                os << " <> " << node_mapping[i] - 1 << "] ";
            }
        }
        os << '\n';
    }
    os.flush();
}

void Printer::Ln(const tree::Tree &tree, std::ostream &os) {
    std::tuple<tree::TreeNodeBase *, int> stack[MAX_GRAPH_EDGE_COUNT];
    int stack_size = 0;

    stack[stack_size++] = {tree[0], 0};

    while (stack_size != 0) {
        tree::TreeNodeBase *node_ptr;
        int indent;

        std::tie(node_ptr, indent) = stack[--stack_size];

        int index = std::find(tree.begin(), tree.end(), node_ptr) - tree.begin();
        Indent(indent, os) << '#' << index << ' ';
        this->operator()(node_ptr->covered_edge_ptr, os);
        this->operator()(node_ptr->boundary_nodes, os << ' ');
        os << '\n';

        if (node_ptr->Right())
            stack[stack_size++] = {node_ptr->Right(), indent + 1};
        if (node_ptr->Left())
            stack[stack_size++] = {node_ptr->Left(), indent + 1};
    }
}

} // namespace debug
} // namespace shrg

#include <algorithm>
#include <bitset>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

#include "synchronous_hyperedge_replacement_grammar.hpp"

namespace shrg {

using std::size_t;

inline int CountFreeNodes(SHRG::Edge &edge) {
    return std::count_if(edge.linked_nodes.begin(), edge.linked_nodes.end(),
                         [](auto node_ptr) { return node_ptr->type == NodeType::kFree; });
}

// sort terminal edges in order of visiting time of DFS.
void SortTerminalEdges(std::vector<SHRG::Edge> &all_edges,
                       std::vector<SHRG::Edge *> &terminal_edges) {
    assert(terminal_edges.empty());

    std::vector<SHRG::Edge *> edge_stack;
    std::bitset<MAX_SHRG_EDGE_COUNT> visited_edges = 0;

    for (SHRG::Edge &edge : all_edges) {
        if (!edge.is_terminal || visited_edges[edge.index])
            continue;
        edge_stack.push_back(&edge);
        visited_edges[edge.index] = true;

        while (!edge_stack.empty()) {
            SHRG::Edge *edge_ptr = edge_stack.back();
            edge_stack.pop_back();
            terminal_edges.push_back(edge_ptr);

            for (SHRG::Node *node_ptr : edge_ptr->linked_nodes)
                for (SHRG::Edge *next_edge_ptr : node_ptr->linked_edges) {
                    int j = next_edge_ptr->index;
                    if (next_edge_ptr->is_terminal && !visited_edges[j]) {
                        visited_edges[j] = true;
                        edge_stack.push_back(next_edge_ptr);
                    }
                }
        }
    }
}

void SortNonterminalEdges(std::vector<SHRG::Edge> &all_edges,
                          std::vector<SHRG::Edge *> &nonterminal_edges) {
    for (SHRG::Edge &edge : all_edges)
        if (!edge.is_terminal)
            nonterminal_edges.push_back(&edge);

    std::stable_sort(nonterminal_edges.begin(), nonterminal_edges.end(),
                     [](auto edge_ptr1, auto edge_ptr2) {
                         return CountFreeNodes(*edge_ptr1) < CountFreeNodes(*edge_ptr2);
                     });
}

void Summary(const std::vector<SHRG> &grammars) {
    uint num_nonsemtantic_grammars = 0;
    uint num_nonlexical_grammars = 0;

    for (const SHRG &grammar : grammars) {
        if (grammar.IsEmpty()) {
            num_nonsemtantic_grammars++;
            continue;
        }

        if (!grammar.nonterminal_edges.empty())
            num_nonlexical_grammars++;
    }

    PRINT_EXPR(num_nonsemtantic_grammars);
    PRINT_EXPR(num_nonlexical_grammars);
}

int SHRG::Load(const std::string &input_file, std::vector<SHRG> &grammars, TokenSet &label_set) {
    OPEN_IFSTREAM(is, input_file, return 0);

    LOG_INFO("Loading SHRG Rules ... < " << input_file);

    int max_shrg_index = 0;
    std::string token;
    int rule_count;
    is >> rule_count;
    grammars.resize(rule_count);
    int counter = 0;
    for (SHRG &grammar : grammars) {
        int has_semantic_part;
        is >> has_semantic_part;

        SHRG::Fragment &fragment = grammar.fragment;
        if (has_semantic_part == 1) {
            int node_count, edge_count;
            is >> node_count >> edge_count;
            if (edge_count > MAX_SHRG_EDGE_COUNT || node_count > MAX_SHRG_NODE_COUNT) {
                LOG_ERROR("Grammar is too large");
                return 0;
            }
            fragment.nodes.resize(node_count);
            fragment.edges.resize(edge_count);

            for (int i = 0; i < node_count; ++i)
                fragment.nodes[i].index = i;

            int edge_index = 0;
            for (SHRG::Edge &edge : fragment.edges) {
                is >> token >> node_count;
                edge.label = label_set.Index(token);
                edge.index = edge_index++;
                for (int i = 0; i < node_count; ++i) {
                    int node_index;
                    is >> node_index;
                    SHRG::Node &node = fragment.nodes[node_index];
                    node.linked_edges.push_back(&edge);
                    edge.linked_nodes.push_back(&node);
                }
                char is_terminal;
                is >> is_terminal;
                edge.is_terminal = (is_terminal == 'Y');
                if (edge.is_terminal)
                    grammar.terminal_edges_set.insert(edge.Hash());
            }

            int external_node_count;
            is >> external_node_count;
            for (int i = 0; i < external_node_count; ++i) {
                int external_index;
                is >> external_index;
                SHRG::Node &external_node = fragment.nodes[external_index];
                grammar.external_nodes.push_back(&external_node);
                external_node.is_external = true;
            }
            for (SHRG::Node &node : fragment.nodes) {
                int nonterminal_count = 0;
                int terminal_count = 0;
                for (const SHRG::Edge *edge_ptr : node.linked_edges)
                    if (edge_ptr->is_terminal)
                        terminal_count++;
                    else
                        nonterminal_count++;
                node.type = terminal_count
                                ? (nonterminal_count ? NodeType::kSemiFixed : NodeType::kFixed)
                                : NodeType::kFree;
            }
        }

        int cfg_rule_count, item_count, current_count, total_count;
        is >> cfg_rule_count;
        assert(cfg_rule_count > 0);
        grammar.cfg_rules.resize(cfg_rule_count);
        grammar.best_cfg_ptr = grammar.cfg_rules.data();
        grammar.num_occurences = 0;

        float max_score = -std::numeric_limits<float>::infinity();
        for (SHRG::CFGRule &cfg_rule : grammar.cfg_rules) {
            is >> cfg_rule.shrg_index >> current_count >> total_count;
            is >> token >> item_count;

            grammar.num_occurences += current_count;
            max_shrg_index = std::max(max_shrg_index, cfg_rule.shrg_index);

            cfg_rule.label = label_set.Index(token);
            cfg_rule.score = log(current_count) - log(total_count);
            if (cfg_rule.score == -std::numeric_limits<float>::infinity()) {
                LOG_WARN("score of cfg_rule is -inf @" << cfg_rule.shrg_index);
                return 0;
            }
            if (cfg_rule.score > max_score) {
                max_score = cfg_rule.score;
                grammar.best_cfg_ptr = &cfg_rule;
            }

            if (grammar.label != EMPTY_LABEL && grammar.label != cfg_rule.label) {
                LOG_ERROR("Grammar has two label");
                return 0;
            }

            grammar.label = cfg_rule.label;
            cfg_rule.items.resize(item_count);
            for (SHRG::CFGItem &item : cfg_rule.items) {
                int edge_index;
                is >> token >> edge_index;
                item.label = label_set.Index(token);
                item.aligned_edge_ptr = (edge_index != -1 ? &fragment.edges[edge_index] : nullptr);
            }
        }
        grammar.label_hash = MakeLabelHash(grammar.label, grammar.external_nodes.size(), false);

        if (has_semantic_part) { // !!! below code should be run after grammar is initialized
            SortTerminalEdges(fragment.edges, grammar.terminal_edges);
            SortNonterminalEdges(fragment.edges, grammar.nonterminal_edges);

            for (auto edge_ptr : grammar.terminal_edges) {
                if (edge_ptr->is_terminal && edge_ptr->linked_nodes.size() == 2 &&
                    edge_ptr->linked_nodes[0] == edge_ptr->linked_nodes[1]) {
                    std::cout << "Self-loop detected in rule " << counter << ", edge label: " << edge_ptr->label << std::endl;
                    grammar.fragment.nodes.clear();
                    LOG_WARN("self-loop !!!");
                    break;
                }
            }
        }
        // if (counter == 126) {
        //     std::cout << "here";
        // }else if (counter == 127) {
        //     std::cout << "check null?";
        // }
        counter += 1;
    }

    is.close();

    LOG_INFO("Loaded " << rule_count << " rules");
    Summary(grammars);

    return max_shrg_index + 1;
}

void SHRG::FilterDisconneted(std::vector<SHRG> &grammars) {
    uint num_filtered = 0;

    LOG_INFO("Filter disconnected grammars ...");
    for (SHRG &grammar : grammars) {
        if (grammar.fragment.edges.empty() || grammar.nonterminal_edges.empty())
            continue;

        std::vector<SHRG::Edge *> edge_stack;
        std::bitset<MAX_SHRG_EDGE_COUNT> visited_edges;

        edge_stack.push_back(grammar.nonterminal_edges[0]);
        visited_edges[grammar.nonterminal_edges[0]->index] = true;

        while (!edge_stack.empty()) {
            SHRG::Edge *edge_ptr = edge_stack.back();
            edge_stack.pop_back();

            for (SHRG::Node *node_ptr : edge_ptr->linked_nodes)
                for (SHRG::Edge *next_edge_ptr : node_ptr->linked_edges) {
                    int next_index = next_edge_ptr->index;
                    if (!visited_edges[next_index]) {
                         visited_edges[next_index] = true;
                        edge_stack.push_back(next_edge_ptr);
                    }
                }
        }

        for (auto edge_ptr : grammar.terminal_edges) // exclude all terminal edges
            visited_edges[edge_ptr->index] = true;

        if (visited_edges.count() != grammar.fragment.edges.size()) { // disconnected
            grammar.fragment.nodes.clear();
            num_filtered++;
        }
    }
    LOG_INFO("Filter disconnected " << num_filtered << " grammars");
}

} // namespace shrg

#include <iostream>
#include <limits>

#include "generator.hpp"

namespace shrg {

inline void CopyMapping(const NodeMapping &input, NodeMapping &output,
                        const std::vector<SHRG::Node *> &linked_nodes) {
    int external_node_index = 0;
    for (const SHRG::Node *external_node_ptr : linked_nodes) {
        auto &output_index = output[external_node_ptr->index];
        auto input_index = input[external_node_index++];
        assert(output_index == 0 || output_index == input_index);
        output_index = input_index;
    }
}

inline void CopyMapping(const NodeMapping &input, NodeMapping &output,
                        const SHRG::Edge *edge_ptr = nullptr) {
    if (edge_ptr)
        CopyMapping(input, output, edge_ptr->linked_nodes);
    else
        for (uint i = 0; i < input.size(); ++i) {
            assert(output[i] == 0 || output[i] == input[i]);
            output[i] = input[i];
        }
}

inline void GenerateToken(std::string &sentence, const std::string &token,
                          const EdsGraph::Node &node) {
    if (node.carg != "#")
        sentence.append(node.carg);
    else if (token.size() > 1U && token[1] != '$')
        sentence.append(token); // <pron> ...
    else {
        assert(token[1] == '$' && token[2] == 'X'); // <$X...>
        sentence.append(node.lemma);
        sentence.append("$s$");
        sentence.append("￨");
        if (token.size() > 4U && token[4] == 'n') // <$X-neg>
            sentence.append("neg");
        else
            sentence.push_back('#');
        sentence.append("￨");
        sentence.push_back(node.pos_tag);
        sentence.append("￨");
        sentence.append(node.sense);
        for (auto &p : node.properties) {
            sentence.append("￨");
            sentence.append(p);
        }
        sentence.append("$/s$");
    }
}

bool Generator::MatchTerminalEdges(NodeMapping &merged_mapping, const SHRG *grammar_ptr,
                                   EdgeSet &edge_set, uint index) {
    size_t edge_count = grammar_ptr->terminal_edges.size();
    if (index == edge_count)
        return edge_set.none();

    const SHRG::Edge *shrg_edge_ptr = grammar_ptr->terminal_edges[index];
    assert(shrg_edge_ptr->is_terminal);
    size_t node_count = shrg_edge_ptr->linked_nodes.size();

    // get mapped nodes of current edges
    int shrg_from_index = shrg_edge_ptr->linked_nodes[0]->index;
    int shrg_to_index = (node_count > 1) ? shrg_edge_ptr->linked_nodes[1]->index : -1;

    uint from = merged_mapping[shrg_edge_ptr->linked_nodes[0]->index];
    uint to = (node_count > 1) ? merged_mapping[shrg_edge_ptr->linked_nodes[1]->index] : 0;

    const EdsGraph *graph_ptr = Graph();
    for (uint i = 0; i < edge_set.size(); i++) {
        const EdsGraph::Edge &eds_edge = graph_ptr->edges[i];
        if (eds_edge.linked_nodes.size() != node_count || eds_edge.label != shrg_edge_ptr->label)
            continue;
        uint current_from = eds_edge.linked_nodes[0]->index + 1;
        uint current_to = (node_count > 1) ? eds_edge.linked_nodes[1]->index + 1 : 0;
        if ((from > 0 && current_from != from) || (to > 0 && current_to != to))
            continue;

        merged_mapping[shrg_from_index] = current_from;
        if (shrg_to_index >= 0)
            merged_mapping[shrg_to_index] = current_to;

        edge_set[i] = false; // try match this edge
        if (MatchTerminalEdges(merged_mapping, grammar_ptr, edge_set, index + 1))
            return true;
        edge_set[i] = true;

        merged_mapping[shrg_from_index] = from;
        if (shrg_to_index >= 0)
            merged_mapping[shrg_to_index] = to;
    }
    return false;
}

std::size_t Generator::CountChartItems(ChartItem *chart_item_ptr) {
    std::size_t count = 0;
    ChartItem *current_ptr = chart_item_ptr;
    do {
        if (current_ptr->status != ChartItem::kVisited) {
            ++count;
            current_ptr->status = ChartItem::kVisited;

            const SHRG *grammar_ptr = current_ptr->attrs_ptr->grammar_ptr;
            for (auto edge_ptr : grammar_ptr->nonterminal_edges)
                count += CountChartItems(FindChartItemByEdge(current_ptr, edge_ptr));
        }
        current_ptr = current_ptr->next_ptr;
    } while (current_ptr != chart_item_ptr);

    return count;
}

float Generator::GetScoreOfChilren(ChartItem *current_ptr) {
    float score = 0.0f;
    for (auto edge_ptr : current_ptr->attrs_ptr->grammar_ptr->nonterminal_edges)
        score += FindBestChartItem(FindChartItemByEdge(current_ptr, edge_ptr));
    return score;
}

float Generator::FindBestChartItem(ChartItem *chart_item_ptr) {
    // When the first instantiation of a chart_item is found during parsing, we use this
    // instantiation to represent the whole chart_item. We use pointer of this instantiation to
    // make larger chart_items.  Other instantiations of the given chart_item will inserted to a
    // linked cycle list (the entry point of the list is the first-found instantiation). So in the
    // final derivation forest, all chart_items will only point (left_ptr & right_ptr) to a
    // first-found chart_item. In order to Find best derivation, we swap the best chart_item in a
    // cycle list with the head chart_item of the linked list.
    if (chart_item_ptr->score <= 0) // the score has been computed
        return chart_item_ptr->score;

    float max_score = -std::numeric_limits<float>::infinity();
    ChartItem *max_score_ptr = nullptr;
    ChartItem *current_ptr = chart_item_ptr;
    do {
        auto &cfg_rules = current_ptr->attrs_ptr->grammar_ptr->cfg_rules;
        auto cfg_ptr = std::max_element(cfg_rules.begin(), cfg_rules.end());

        float current_score = cfg_ptr->score + GetScoreOfChilren(current_ptr);

        if (current_score > max_score) {
            max_score = current_score;
            max_score_ptr = current_ptr;
        }

        current_ptr->score = current_score;
        current_ptr->status = cfg_ptr - cfg_rules.begin();

        current_ptr = current_ptr->next_ptr;
    } while (current_ptr != chart_item_ptr);

    if (max_score_ptr != chart_item_ptr)
        chart_item_ptr->Swap(*max_score_ptr);

    assert(chart_item_ptr->score <= 0);
    return chart_item_ptr->score;
}

int Generator::Generate(ChartItem *chart_item_ptr, Derivation &derivation, std::string &sentence) {
    if (!chart_item_ptr) {
        std::cerr << "ERROR: Generate called with null chart_item_ptr" << std::endl;
        return DerivationNode::Error;
    }
    GrammarAttributes *attrs_ptr = chart_item_ptr->attrs_ptr;
    if (!attrs_ptr) {
        std::cerr << "ERROR: chart_item_ptr has null attrs_ptr" << std::endl;
        return DerivationNode::Error;
    }

    const SHRG *grammar_ptr = attrs_ptr->grammar_ptr;
    if (!grammar_ptr) {
        std::cerr << "ERROR: attrs_ptr has null grammar_ptr" << std::endl;
        return DerivationNode::Error;
    }

    const SHRG::CFGRule *selected_rule = attrs_ptr->SelectRule(chart_item_ptr);
    if (!selected_rule) {
        std::cerr << "ERROR: SelectRule returned null" << std::endl;
        return DerivationNode::Error;
    }

    int parent_index = derivation.size();
    derivation.push_back({grammar_ptr, selected_rule, chart_item_ptr});

    NodeMapping full_mapping{};
    bool is_mapping_computed = false;

    const EdsGraph *graph_ptr = Graph();
    // !!! do not cache parent_index, because derivation may change during iteration
    for (auto &item : derivation[parent_index].cfg_ptr->items) {
        if (!item.aligned_edge_ptr || item.aligned_edge_ptr->is_terminal) { // string literals
            auto &token = parser_->label_set_[item.label];
            if (!token.empty() && token[0] == '<' && item.aligned_edge_ptr) {
                // only terminal graphs
                // NOTE: for tree & tree_general, below check will fail
                // assert(!chart_item_ptr->left_ptr && !chart_item_ptr->right_ptr);
                if (!is_mapping_computed) {
                    // recover full node mappings, since chart_items only record boundary nodes
                    is_mapping_computed = true;
                    CopyMapping(chart_item_ptr->boundary_node_mapping, full_mapping,
                                grammar_ptr->external_nodes);
                    EdgeSet edge_set = chart_item_ptr->edge_set;

                    [[maybe_unused]] bool success =
                        MatchTerminalEdges(full_mapping, grammar_ptr, edge_set, 0);
//                    assert(success);
                }
                auto &nodes = item.aligned_edge_ptr->linked_nodes;
                assert(nodes.size() == 1); // edge should be a pred edge
                int eds_index = full_mapping[nodes[0]->index] - 1;
                if (parser_->verbose_) {
                    sentence.append(token);
                    sentence.push_back('(');
                }
                if (eds_index >= 0 && eds_index < (int)graph_ptr->nodes.size())
                    GenerateToken(sentence, token, graph_ptr->nodes[eds_index]);
                else
                    sentence.append("???");
                if (parser_->verbose_)
                    sentence.push_back(')');
            } else // normal token
                sentence.append(token);
            sentence.push_back(' ');
            continue;
        }

        ChartItem *subpart = FindChartItemByEdge(chart_item_ptr, item.aligned_edge_ptr);
        if (!subpart || !subpart->attrs_ptr) {
            std::cerr << "ERROR: FindChartItemByEdge returned invalid subpart" << std::endl;
            return DerivationNode::Error;
        }
        int child_index = Generate(subpart, derivation, sentence);
        // !!! it's important to use index instead of pointer or reference
        // !!! it's important to compute &parent after child_index
        if (child_index == DerivationNode::Error)
            return child_index;

        auto &parent = derivation[parent_index];
        parent.children.push_back(child_index);
    }

    return parent_index;
}

ChartItem *Generator::BestResult() {
    ChartItem *ptr = parser_->matched_item_ptr_;
    if (ptr)
        FindBestChartItem(ptr);
    return ptr;
}

} // namespace shrg

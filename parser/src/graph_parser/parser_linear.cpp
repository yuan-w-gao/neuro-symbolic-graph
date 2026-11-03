#include <algorithm>
#include <iostream>

#include "parser_linear.hpp"

#define PUSH_AGENDA(ptr)                                                                           \
    if (!(ptr)->in_queue) {                                                                        \
        (ptr)->in_queue = true;                                                                    \
        updated_agendas_.push(ptr);                                                                \
    }

namespace shrg {
namespace linear {

using std::size_t;

LinearSHRGParser::LinearSHRGParser(const std::vector<SHRG> &grammars, const TokenSet &label_set)
    : LinearSHRGParserBase("linear", grammars, label_set), //
      attributes_(grammars.size()) {

    std::unordered_map<EdgeHash, std::unordered_set<NodeMapping>> activated_masks;
    for (size_t i = 0; i < grammars.size(); ++i) {
        const SHRG &grammar = grammars[i];
        Attributes &attrs = attributes_[i];

        attrs.Initialize(grammar);
        attrs.required_masks = &all_required_masks_[grammar.label_hash];

        uint num_nonterminals = grammar.nonterminal_edges.size();
        if (grammar.IsEmpty() || num_nonterminals == 0)
            continue;

        NodeMapping matched_nodes{};
        for (auto terminal_edge_ptr : grammar.terminal_edges)
            for (auto node_ptr : terminal_edge_ptr->linked_nodes)
                matched_nodes[node_ptr->index] = UINT8_MAX; // node is matched

        attrs.edge_masks.resize(num_nonterminals);
        for (uint i = 0; i < num_nonterminals; ++i) {
            SHRG::Edge *edge_ptr = grammar.nonterminal_edges[i];

            NodeMapping &mask = attrs.edge_masks[i];

            mask.m8.fill(0);

            int index = 0;
            bool insert_to_set = false;
            for (auto node_ptr : edge_ptr->linked_nodes) {
                mask[index] = matched_nodes[node_ptr->index];
                if (mask[index] == 0) // some position is masked
                    insert_to_set = true;
                // after current edge is matched, the node is also matched
                matched_nodes[node_ptr->index] = UINT8_MAX;
                index++;
            }

            // the full mask always be a member of active_masks
            if (insert_to_set)
                activated_masks[edge_ptr->Hash()].insert(mask);
        }
    }

    for (auto &item : activated_masks) {
        auto &required_masks = all_required_masks_[item.first];
        for (auto &mask : item.second)
            required_masks.push_back(mask);
    }
}

void LinearSHRGParser::EmitSubGraph(ChartItem *chart_item_ptr, uint boundary_node_count,
                                    Attributes *attrs_ptr) {
    LabelHash label_hash = attrs_ptr->grammar_ptr->label_hash;
    const NodeMapping &node_mapping = chart_item_ptr->boundary_node_mapping;

    assert(attrs_ptr->required_masks);

    Agenda *agenda_ptr = &agendas_.At(label_hash, node_mapping, boundary_node_count);
    // NOTE: it is important to insert full mask of chart_item into agendas_ first the next_ptr of
    // chart_item will always be set here, this ensure the derivation forest is correct
    if (!agenda_ptr->passive_items.TryInsert(chart_item_ptr))
        return;

    SHRG_DEBUG_INC(num_passive_items_);
    PUSH_AGENDA(agenda_ptr);

    for (auto &mask : *attrs_ptr->required_masks) {
        agenda_ptr = &agendas_.At(label_hash,                            //
                                  MaskedNodeMapping(node_mapping, mask), //
                                  boundary_node_count);

        if (agenda_ptr->passive_items.TryInsert(chart_item_ptr))
            PUSH_AGENDA(agenda_ptr);
    }
}

bool LinearSHRGParser::MatchTerminalEdges(Attributes *attrs_ptr) {
    const SHRG *grammar_ptr = attrs_ptr->grammar_ptr;
    if (grammar_ptr->IsEmpty()) // grammar without semantic part
        return false;

    if (grammar_ptr->terminal_edges.empty()) { // grammar without terminal edges
        attrs_ptr->terminal_items.push_back(items_pool_.Push(attrs_ptr));
        SHRG_DEBUG_INC(num_terminal_subgraphs_);
    } else {
        NodeMapping node_mapping{}; // initalization is important and necessary
        NodeSet node_set;
        EdgeSet edge_set;
        LinearSHRGParserBase::MatchTerminalEdges(attrs_ptr, node_mapping, edge_set, node_set, 0);//yg: BUG
    }

    auto &terminal_items = attrs_ptr->terminal_items;
    if (grammar_ptr->nonterminal_edges.empty()) {
        for (auto chart_item_ptr : terminal_items) {
            NodeMapping &node_mapping = chart_item_ptr->boundary_node_mapping;
            uint boundary_node_count = chart_item_ptr->status;
            chart_item_ptr->status = ChartItem::kEmpty;

            if (!CheckAndChangeMappingFinally(grammar_ptr, boundary_node_count, node_mapping))
                continue;
            // current grammar has no non-terminals, so it won't be updated any more directly add
            // the subgraph to passive_items_map_
            EmitSubGraph(chart_item_ptr, boundary_node_count, attrs_ptr);
        }
        terminal_items.clear();

        return false;
    }

    return !terminal_items.empty();
}

void LinearSHRGParser::ClearChart() {
    SHRG_DEBUG_START_TIMER();
    assert(updated_agendas_.empty());

    agendas_.Clear();

    LinearSHRGParserBase::ClearChart();
    if (verbose_)
        SHRG_DEBUG_REPORT_TIMER("Clear");
}

void LinearSHRGParser::EnableGrammar(Attributes *attrs_ptr) {
    const SHRG *grammar_ptr = attrs_ptr->grammar_ptr;

    assert(!grammar_ptr->nonterminal_edges.empty());

    const SHRG::Edge *first_edge_ptr = grammar_ptr->nonterminal_edges[0];
    const NodeMapping &first_boundary_nodes = attrs_ptr->edge_masks[0];
    for (auto terminal_item_ptr : attrs_ptr->terminal_items) {
        const NodeMapping &node_mapping = terminal_item_ptr->boundary_node_mapping;
        Agenda *agenda_ptr = //
            &agendas_.At(first_edge_ptr,
                         MaskedNodeMapping(first_edge_ptr, node_mapping, first_boundary_nodes));

        agenda_ptr->active_items.push_back({
            terminal_item_ptr, /* chart_item_ptr */
            0                  /* index */
        });
    }
}

void LinearSHRGParser::InitializeChart() {
    SHRG_DEBUG_START_TIMER();

    LinearSHRGParserBase::InitializeChart();
    for (size_t i = 0; i < grammars_.size(); ++i) {
        const SHRG &grammar = grammars_[i];

        if (!IsGrammarCompatiable(grammar, terminal_map_))
            continue;

        Attributes &attrs = attributes_[i];
        attrs.Clear();
        // TODO: check correctness of the result
        if (MatchTerminalEdges(&attrs)) { //yg: BUG
            assert(!grammar.nonterminal_edges.empty()); // Strange condition ???
            // add attrs to corresponding available_items
            EnableGrammar(&attrs);
            SHRG_DEBUG_INC(num_grammars_available_);
        }
    }

    if (verbose_) {
        SHRG_DEBUG_REPORT_TIMER("Initalization");
        PRINT_EXPR(num_grammars_available_);
        PRINT_EXPR(num_terminal_subgraphs_);
    }
}

void LinearSHRGParser::MergeItems(Agenda::ActiveItem &item, ChartItem *external_item_ptr) {
    SHRG_DEBUG_INC(num_total_merge_operations_);

    ChartItem *internal_item_ptr = item.chart_item_ptr;
    Attributes *attrs_ptr = static_cast<Attributes *>(internal_item_ptr->attrs_ptr);

    EdgeSet merged_edge_set;
    NodeMapping merged_mapping{}; // initialization is very important

    uint index = item.index;
    const SHRG *grammar_ptr = attrs_ptr->grammar_ptr;
    const SHRG::Edge *edge_ptr = grammar_ptr->nonterminal_edges[index];

    int boundary_node_count = MergeTwoChartItems(graph_ptr_,                                     //
                                                 external_item_ptr, internal_item_ptr, edge_ptr, //
                                                 grammar_ptr->fragment.nodes.size(),             //
                                                 merged_edge_set, merged_mapping,
                                                 attrs_ptr->boundary_nodes_of_steps[index]);

    if (boundary_node_count == -1)
        return;

    bool is_complete = (index + 1 == grammar_ptr->nonterminal_edges.size());
    if (is_complete &&
        !CheckAndChangeMappingFinally(grammar_ptr, boundary_node_count, merged_mapping))
        return;

    ChartItem *chart_item_ptr = items_pool_.Push(attrs_ptr, merged_edge_set, merged_mapping);
    chart_item_ptr->left_ptr = internal_item_ptr;
    chart_item_ptr->right_ptr = external_item_ptr;
    SHRG_DEBUG_INC(num_succ_merge_operations_);

    if (is_complete)
        EmitSubGraph(chart_item_ptr, boundary_node_count, attrs_ptr);
    else {
        const SHRG::Edge *next_edge_ptr = grammar_ptr->nonterminal_edges[index + 1];
        Agenda *next_item_ptr = //
            &agendas_.At(next_edge_ptr,
                         MaskedNodeMapping(next_edge_ptr, //
                                           chart_item_ptr->boundary_node_mapping,
                                           attrs_ptr->edge_masks[index + 1]));

        next_item_ptr->active_items.push_back({
            chart_item_ptr, /* chart_item_ptr */
            index + 1       /* type */
        });
        PUSH_AGENDA(next_item_ptr);
        SHRG_DEBUG_INC(num_active_items_);
    }
}

void LinearSHRGParser::UpdateAgenda(Agenda *agenda_ptr) {
    auto &active_items = agenda_ptr->active_items;
    auto &passive_items = agenda_ptr->passive_items;
    size_t active_item_size = active_items.size();
    size_t passive_item_size = passive_items.Size();

    if (passive_item_size == 0 || active_item_size == 0)
        return;
    // new active_items <=> new+old passive_items
    for (size_t i = agenda_ptr->num_visited_active_items; i < active_item_size; ++i)
        for (size_t j = 0; j < passive_item_size; ++j)
            MergeItems(active_items[i], passive_items[j]);
    // old active_items <=> new passive_items
    for (size_t i = 0; i < agenda_ptr->num_visited_active_items; ++i)
        for (size_t j = agenda_ptr->num_visited_passive_items; j < passive_item_size; ++j)
            MergeItems(active_items[i], passive_items[j]);

    agenda_ptr->num_visited_active_items = active_item_size;
    agenda_ptr->num_visited_passive_items = passive_item_size;
}

ParserError LinearSHRGParser::Parse(const EdsGraph &graph) {
    auto code = SHRGParserBase::BeforeParse(graph);
    if (code != ParserError::kNone)
        return code;

    ClearChart();
    InitializeChart();

    SHRG_DEBUG_START_TIMER();
    while (!updated_agendas_.empty()) {
        if (items_pool_.PoolSize() > max_pool_size_) {
            std::queue<Agenda *> empty;
            updated_agendas_.swap(empty);
            return ParserError::kOutOfMemory;
        }
        Agenda *agenda_ptr = updated_agendas_.front();
        updated_agendas_.pop();
        agenda_ptr->in_queue = false;
        UpdateAgenda(agenda_ptr);
    }

    SetCompleteItem(agendas_.Small(), 40 /* label_offset */);

    num_indexing_keys_ = agendas_.Small().size() +  //
                         agendas_.Medium().size() + //
                         agendas_.Large().size();
    if (verbose_) {
        SHRG_DEBUG_REPORT_TIMER("Parsing");
        PRINT_EXPR(num_active_items_);
        PRINT_EXPR(num_passive_items_);
        PRINT_EXPR(num_indexing_keys_);
        PRINT_EXPR(num_succ_merge_operations_);
        PRINT_EXPR(num_total_merge_operations_);
    }

    return matched_item_ptr_ ? ParserError::kNone : ParserError::kNoResult;
}

} // namespace linear
} // namespace shrg

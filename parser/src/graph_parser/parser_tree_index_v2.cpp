#include <iostream>

#include "parser_tree_index_v2.hpp"

#define PUSH_AGENDA(ptr)                                                                           \
    if (!(ptr)->in_queue) {                                                                        \
        (ptr)->in_queue = true;                                                                    \
        updated_agendas_.push(ptr);                                                                \
    }

namespace shrg {

namespace tree {
extern ChartItem empty_item;
extern std::vector<NodeMapping> masks_for_pred_edges;
extern std::vector<NodeMapping> masks_for_structural_edges;
} // namespace tree

namespace tree_v2 {
void ExpandActiveItem(ChartItem *chart_item_ptr, utils::MemoryPool<ChartItem> &items_pool);
}

namespace tree_index_v1 {
NodeMapping ComputeMasks(TreeNode *tree_node, tree::MaskMap &activated_masks);
} // namespace tree_index_v1

namespace tree_index_v2 {

using std::size_t;
using tree::empty_item;
using tree::MaskMap;
using tree::masks_for_pred_edges;
using tree::masks_for_structural_edges;
using tree_index_v1::ComputeMasks;
using tree_v2::ExpandActiveItem;

void SummaryBinaryAgendas(const std::vector<Tree> &trees, const TokenSet &label_set,
                          uint64_t stats[6]) {
    for (auto &tree : trees) {
        for (auto node_ptr_ : tree) {
            if (!node_ptr_->Right())
                continue; // binary node
            for (auto &agenda : static_cast<TreeNode *>(node_ptr_)->agendas) {
                uint64_t num_active1 = agenda.second.num_left_visited_items;
                uint64_t num_active2 = agenda.second.num_right_visited_items;
                uint64_t num_ops = num_active1 * num_active2;

                stats[1] = std::max(num_active2, std::max(num_active1, stats[1]));
                stats[2] = std::max(num_ops, stats[2]);
                stats[3] += num_ops;
                stats[4]++;
                if (num_ops == 0)
                    continue;

                debug::Printer{nullptr, label_set}(agenda.first, std::cout);
                std::cout << "  " << num_active1 << " * " << num_active2 << " = " << num_ops
                          << '\n';
            }
        }
    }
}

template <typename T>
void SummaryUnaryAgendas(const T &agendas, const TokenSet &label_set, uint64_t stats[6]) {
    for (auto &item : agendas) {
        uint64_t num_passive = item.second.passive_items.Size();
        uint64_t num_active = item.second.active_items.size();
        uint64_t num_ops = num_passive * num_active;
        if (stats[0] < num_passive) {
            stats[0] = num_passive;
            stats[5] = item.second.passive_items[0]->attrs_ptr->grammar_ptr->label_hash;
        }
        stats[1] = std::max(num_active, stats[1]);
        stats[2] = std::max(num_ops, stats[2]);
        stats[3] += num_ops;
        stats[4]++;
        if (num_ops == 0)
            continue;

        std::cout << "  " << debug::Printer{nullptr, label_set}.ToString(item.first) << ": "
                  << num_passive << " * " << num_active << " = " << num_ops << '\n';
    }
}

void TreeSHRGParser::InitializeTree() {
    MaskMap activated_masks;

    for (size_t i = 0; i < grammars_.size(); ++i) {
        const SHRG &grammar = grammars_[i];
        Tree &tree = tree_decompositions_[i];

        if (grammar.IsEmpty())
            continue;

        assert(!tree[0]->Parent()); // tree root;
        TreeNode *tree_root = static_cast<TreeNode *>(tree[0]);
        ComputeMasks(tree_root, activated_masks);

        tree_root->required_masks = &all_required_masks_[grammar.label_hash];
    }

    for (auto &item : activated_masks) {
        auto &required_masks = all_required_masks_[item.first];
        for (auto &mask : item.second)
            required_masks.push_back(mask);
    }
}

void TreeSHRGParser::EmitPartialSubgraph(ChartItem *chart_item_ptr, TreeNodeBase *node_ptr,
                                         bool submit) {
    if (!static_cast<TreeNode *>(node_ptr)->corresponding_items.TryInsert(chart_item_ptr))
        return;

    // TODO: optimize for empty subgraph
    TreeNode *parent_ptr = static_cast<TreeNode *>(node_ptr->Parent());
    const NodeMapping &covered_mask = parent_ptr->covered_mask;
    const NodeMapping &node_mapping = chart_item_ptr->boundary_node_mapping;

    Agenda *agenda_ptr;
    if (parent_ptr->Right()) { // parent is a binary node
        BinaryAgenda *binary_ptr =
            &parent_ptr->agendas[MaskedNodeMapping(node_mapping, covered_mask)];
        agenda_ptr = binary_ptr;
        binary_ptr->node_ptr = parent_ptr;

        ((parent_ptr->Left() == node_ptr) //
             ? binary_ptr->left_items
             : binary_ptr->right_items)
            .push_back(chart_item_ptr);
    } else { // parent is a unary node
        const SHRG::Edge *edge_ptr = parent_ptr->covered_edge_ptr;
        UnaryAgenda *unary_ptr = //
            &unary_agendas_.At(edge_ptr, MaskedNodeMapping(edge_ptr, node_mapping, covered_mask));
        agenda_ptr = unary_ptr;

        unary_ptr->active_items.push_back({
            chart_item_ptr, /* chart_item_ptr */
            parent_ptr      /* node_ptr */
        });
    }

    if (submit) {
        SHRG_DEBUG_INC(num_active_items_);
        PUSH_AGENDA(agenda_ptr);
    }
}

void TreeSHRGParser::EmitCompleteSubGraph(ChartItem *chart_item_ptr, uint boundary_node_count,
                                          LabelHash label_hash,
                                          const std::vector<NodeMapping> *required_masks) {
    const NodeMapping &node_mapping = chart_item_ptr->boundary_node_mapping;

    UnaryAgenda *unary_ptr = &unary_agendas_.At(label_hash, node_mapping, boundary_node_count);

    // NOTE: it is important to insert full mask of subgraph into unary_agendas_ first. The
    // next_ptr of chart_item will always be set here, this ensure the derivation forest is correct
    if (!unary_ptr->passive_items.TryInsert(chart_item_ptr))
        return;
    SHRG_DEBUG_INC(num_passive_items_);
    PUSH_AGENDA(unary_ptr);

    for (auto &mask : *required_masks) {
        // !!! here `node_mapping` is computed by `MergeFinal`, so the `node_mapping[i]` is
        // index of corresponding eds node of the i-th external node

        unary_ptr = &unary_agendas_.At(label_hash,                            //
                                       MaskedNodeMapping(node_mapping, mask), //
                                       boundary_node_count);

        bool success = unary_ptr->passive_items.TryInsert(chart_item_ptr);
        if (success)
            PUSH_AGENDA(unary_ptr);
    }
}

void TreeSHRGParser::MatchTerminalEdges() {
    // every edge in edsgraph can be view as a passive item
    for (const EdsGraph::Edge &edge : graph_ptr_->edges) {
        EdgeHash edge_hash = edge.Hash();

        assert(edge.linked_nodes.size() < 3);      // Strange edsgraph edge
        assert(edge.index < MAX_GRAPH_EDGE_COUNT); // Edge index out of range

        ChartItem *chart_item_ptr = items_pool_.Push();
        chart_item_ptr->edge_set[edge.index] = true;

        uint num_nodes = edge.linked_nodes.size();
        for (uint i = 0; i < num_nodes; ++i) { // should be less than 2
            // index of node in edsgraph, index starts from 1
            const EdsGraph::Node *eds_node_ptr = edge.linked_nodes[i];
            if (eds_node_ptr->linked_edges.size() > 1) // only collect boundary node
                chart_item_ptr->boundary_node_mapping[i] = edge.linked_nodes[i]->index + 1;
        }

        SHRG_DEBUG_INC(num_terminal_subgraphs_);
        EmitCompleteSubGraph(
            chart_item_ptr, num_nodes, edge_hash,
            (num_nodes == 1 ? &masks_for_pred_edges : &masks_for_structural_edges));
    }
}

void TreeSHRGParser::MergeItems(ChartItem *left_item_ptr, ChartItem *right_item_ptr,
                                TreeNodeBase *node_ptr, bool is_unary_node) {
    SHRG_DEBUG_INC(num_total_merge_operations_);
    const SHRG *grammar_ptr = node_ptr->grammar_ptr;
    EdgeSet merged_edge_set;
    NodeMapping merged_mapping{}; // initialization is very important

    int boundary_node_count =
        is_unary_node
            ? MergeTwoChartItems(graph_ptr_, //
                                 left_item_ptr, right_item_ptr, node_ptr->covered_edge_ptr,
                                 grammar_ptr->fragment.nodes.size(), //
                                 merged_edge_set, merged_mapping, node_ptr->boundary_nodes)
            : MergeTwoChartItems(graph_ptr_, //
                                 left_item_ptr, right_item_ptr,
                                 grammar_ptr->fragment.nodes.size(), //
                                 merged_edge_set, merged_mapping, node_ptr->boundary_nodes);

    if (boundary_node_count == -1)
        return;

    if (!node_ptr->Parent() &&
        !CheckAndChangeMappingFinally(grammar_ptr, boundary_node_count, merged_mapping))
        return;

    ChartItem *chart_item_ptr = items_pool_.Push(node_ptr, merged_edge_set, merged_mapping);
    chart_item_ptr->left_ptr = left_item_ptr;
    chart_item_ptr->right_ptr = right_item_ptr;
    SHRG_DEBUG_INC(num_succ_merge_operations_);

    if (node_ptr->Parent())
        EmitPartialSubgraph(chart_item_ptr, node_ptr, true /* submit */);
    else // node_ptr is the root of the tree decomposition, so the item is completed
        EmitCompleteSubGraph(chart_item_ptr, boundary_node_count, node_ptr);
}

void TreeSHRGParser::ClearChart() {
    SHRG_DEBUG_START_TIMER();
    assert(updated_agendas_.empty());

    unary_agendas_.Clear();

    SHRGParserBase::ClearChart();
    if (verbose_)
        SHRG_DEBUG_REPORT_TIMER("Clear");
}

void TreeSHRGParser::InitializeChart() {
    SHRG_DEBUG_START_TIMER();

    // all terminal edges of current graph;
    std::unordered_set<EdgeHash> terminal_edges_set;
    for (const EdsGraph::Edge &edge : graph_ptr_->edges)
        if (edge.is_terminal)
            terminal_edges_set.insert(edge.Hash());

    for (size_t i = 0; i < grammars_.size(); ++i) {
        const SHRG &grammar = grammars_[i];

        if (grammar.IsEmpty() || !IsGrammarCompatiable(grammar, terminal_edges_set))
            continue;

        SHRG_DEBUG_INC(num_grammars_available_);

        Tree &tree = tree_decompositions_[i];
        for (TreeNodeBase *node_ptr : tree) {
            node_ptr->Clear();

            if (node_ptr->Left())
                continue;

            EmitPartialSubgraph(&empty_item, node_ptr, false /* submit */);
        }
    }

    MatchTerminalEdges();

    if (verbose_) {
        SHRG_DEBUG_REPORT_TIMER("Initalization");
        PRINT_EXPR(num_grammars_available_);
        PRINT_EXPR(num_terminal_subgraphs_);
    }
}

void TreeSHRGParser::UpdateUnaryNode(UnaryAgenda *agenda_ptr) {
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

void TreeSHRGParser::UpdateBinaryNode(BinaryAgenda *agenda_ptr) {
    TreeNodeBase *node_ptr = agenda_ptr->node_ptr;
    assert(node_ptr && node_ptr->Right()); // node_ptr is a binary node

    auto &left_items = agenda_ptr->left_items;
    auto &right_items = agenda_ptr->right_items;
    size_t left_size = left_items.size();
    size_t right_size = right_items.size();

    if (left_size == 0 || right_size == 0)
        return;

    // new chart_items of left item <=> new+old chart_items of right item
    for (size_t i = agenda_ptr->num_left_visited_items; i < left_size; ++i)
        for (size_t j = 0; j < right_size; ++j)
            MergeItems(left_items[i], right_items[j], node_ptr);

    // new chart_items of right item <=> old chart_items of left item
    for (size_t i = agenda_ptr->num_right_visited_items; i < right_size; ++i)
        for (size_t j = 0; j < agenda_ptr->num_left_visited_items; ++j)
            MergeItems(left_items[j], right_items[i], node_ptr);

    agenda_ptr->num_left_visited_items = left_size;   // set all items as visited
    agenda_ptr->num_right_visited_items = right_size; // set all items as visited
}

ParserError TreeSHRGParser::Parse(const EdsGraph &graph) {
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

        if (agenda_ptr->is_binary)
            UpdateBinaryNode(static_cast<BinaryAgenda *>(agenda_ptr));
        else
            UpdateUnaryNode(static_cast<UnaryAgenda *>(agenda_ptr));
    }

    SetCompleteItem(unary_agendas_.Small(), 40 /* label_offset */);

    num_indexing_keys_ = unary_agendas_.Small().size() +  //
                         unary_agendas_.Medium().size() + //
                         unary_agendas_.Large().size();
    if (verbose_) {
        SHRG_DEBUG_REPORT_TIMER("Parsing");
        PRINT_EXPR(num_active_items_);
        PRINT_EXPR(num_passive_items_);
        PRINT_EXPR(num_indexing_keys_);
        PRINT_EXPR(num_succ_merge_operations_);
        PRINT_EXPR(num_total_merge_operations_);
    }

    if (matched_item_ptr_) {
        ExpandActiveItem(matched_item_ptr_, items_pool_);
        return ParserError::kNone;
    }

    return ParserError::kNoResult;
}

} // namespace tree_index_v2
} // namespace shrg

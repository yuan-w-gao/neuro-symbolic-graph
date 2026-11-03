#include "parser_tree_v2.hpp"

#define PUSH_AGENDA(ptr)                                                                           \
    if (!(ptr)->in_queue) {                                                                        \
        (ptr)->in_queue = true;                                                                    \
        updated_agendas_.push(ptr);                                                                \
    }

namespace shrg {

namespace tree {
extern ChartItem empty_item;
} // namespace tree

namespace tree_v2 {

using std::size_t;
using tree::empty_item;
using tree::Tree;

inline bool IsPassiveItem(ChartItem *item_ptr) {
    return item_ptr->attrs_ptr && !static_cast<TreeNode *>(item_ptr->attrs_ptr)->Parent();
}

void ExpandActiveItem(ChartItem *chart_item_ptr, utils::MemoryPool<ChartItem> &items_pool) {
    if (chart_item_ptr->status == ChartItem::kExpanded || !chart_item_ptr->attrs_ptr)
        return; // visited or terminal items

    assert(chart_item_ptr->next_ptr);

    ChartItem *current_ptr = chart_item_ptr;
    do {
        ChartItem *next_ptr = current_ptr->next_ptr;

        assert(current_ptr->status != ChartItem::kExpanded);
        current_ptr->status = ChartItem::kExpanded;

        auto left_ptr = current_ptr->left_ptr;
        auto right_ptr = current_ptr->right_ptr;

        ExpandActiveItem(left_ptr, items_pool);
        ExpandActiveItem(right_ptr, items_pool);

        bool is_left_passive = IsPassiveItem(left_ptr);
        bool is_right_passive = IsPassiveItem(right_ptr);

        if (is_left_passive && is_right_passive) {
            current_ptr = next_ptr;
            continue;
        }

        if (!is_left_passive && !is_right_passive) {
            ChartItem *ptr_l = left_ptr->next_ptr;
            while (ptr_l != left_ptr) {
                assert(ptr_l);
                ChartItem *ptr_r = right_ptr->next_ptr;
                while (ptr_r != right_ptr) {
                    assert(ptr_r);
                    ChartItem *new_item_ptr = items_pool.Push(*current_ptr);
                    new_item_ptr->left_ptr = ptr_l;
                    new_item_ptr->right_ptr = ptr_r;
                    current_ptr->Push(new_item_ptr);
                    ptr_r = ptr_r->next_ptr;
                }
                ptr_l = ptr_l->next_ptr;
            }
        }

        if (!is_right_passive) {
            ChartItem *ptr = right_ptr->next_ptr;
            while (ptr != right_ptr) {
                assert(ptr);
                ChartItem *new_item_ptr = items_pool.Push(*current_ptr);
                new_item_ptr->right_ptr = ptr;
                current_ptr->Push(new_item_ptr);
                ptr = ptr->next_ptr;
                // ptr = ptr->Pop(); // expand active item
            }
        }

        if (!is_left_passive) {
            ChartItem *ptr = left_ptr->next_ptr;
            while (ptr != left_ptr) {
                assert(ptr);
                ChartItem *new_item_ptr = items_pool.Push(*current_ptr);
                new_item_ptr->left_ptr = ptr;
                current_ptr->Push(new_item_ptr);
                ptr = ptr->next_ptr;
                // ptr = ptr->Pop(); // expand active item
            }
        }

        current_ptr = next_ptr;
    } while (current_ptr != chart_item_ptr);
}

void TreeSHRGParser::InitializeTree() {
    for (int i = grammars_.size() - 1; i >= 0; --i) {
        const SHRG &grammar = grammars_[i];
        if (grammar.IsEmpty())
            continue;

        Tree &tree = tree_decompositions_[i];
        assert(!tree[0]->Parent()); // tree root;
        for (auto node_ptr : tree) {
            if (!node_ptr->Left()) // skip leaf
                continue;

            if (node_ptr->Right()) { // binary Node
                TreeNode *node_ext_ptr = static_cast<TreeNode *>(node_ptr);
                BinaryAgenda *agenda_ptr = binary_agendas_pool_.Push();

                agenda_ptr->node_ptr = node_ext_ptr;
                node_ext_ptr->agenda_ptr = agenda_ptr;
            }
        }
    }
}

void TreeSHRGParser::EmitPartialSubgraph(ChartItem *chart_item_ptr, TreeNodeBase *node_ptr,
                                         bool submit) {
    if (!static_cast<TreeNode *>(node_ptr)->corresponding_items.TryInsert(chart_item_ptr))
        return;

    // TODO: optimize for empty subgraph
    TreeNode *parent_ptr = static_cast<TreeNode *>(node_ptr->Parent());
    if (!parent_ptr->Right()) { // parent is a unary node
        UnaryAgenda *agenda_ptr = static_cast<UnaryAgenda *>(parent_ptr->agenda_ptr);
        agenda_ptr->active_items.push_back({
            chart_item_ptr, /* chart_item_ptr */
            parent_ptr      /* node_ptr */
        });
    }

    if (submit) {
        SHRG_DEBUG_INC(num_active_items_);
        PUSH_AGENDA(parent_ptr->agenda_ptr);
    }
}

void TreeSHRGParser::EmitCompleteSubGraph(ChartItem *chart_item_ptr, EdgeHash edge_hash) {
    UnaryAgenda *agenda_ptr = &unary_agendas_[edge_hash];
    if (agenda_ptr->passive_items.TryInsert(chart_item_ptr)) {
        SHRG_DEBUG_INC(num_passive_items_);
        PUSH_AGENDA(agenda_ptr);
    }
}

void TreeSHRGParser::ClearChart() {
    SHRG_DEBUG_START_TIMER();
    assert(updated_agendas_.empty());

    unary_agendas_.clear();

    SHRGParserBase::ClearChart();
    if (verbose_)
        SHRG_DEBUG_REPORT_TIMER("Clear");
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
        EmitCompleteSubGraph(chart_item_ptr, edge_hash);
    }
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
            node_ptr->Clear(); // clear nodes

            if (node_ptr->Left() && !node_ptr->Right()) { // unary nodes
                const SHRG::Edge *edge_ptr = node_ptr->covered_edge_ptr;
                assert(edge_ptr);
                // precompute hash
                static_cast<TreeNode *>(node_ptr)->agenda_ptr = &unary_agendas_[edge_ptr->Hash()];
            }
        }

        // NOTE: all agenda_ptr should initialized before EmitCompleteSubGraph
        // the order of nodes in tree is unlimited
        for (TreeNodeBase *node_ptr : tree)
            if (!node_ptr->Left()) // leaf nodes
                EmitPartialSubgraph(&empty_item, node_ptr, false /* submit */);
    }

    MatchTerminalEdges();

    if (verbose_) {
        SHRG_DEBUG_REPORT_TIMER("Initalization");
        PRINT_EXPR(num_grammars_available_);
        PRINT_EXPR(num_terminal_subgraphs_);
    }
}

void TreeSHRGParser::MergeItems(ChartItem *left_item_ptr, ChartItem *right_item_ptr,
                                TreeNodeBase *node_ptr, bool is_unary_node) {
    SHRG_DEBUG_INC(num_total_merge_operations_);
    const SHRG *grammar_ptr = node_ptr->grammar_ptr;
    EdgeSet merged_edge_set;
    NodeMapping merged_mapping{}; // initialization is very important
    // when is_unary_node is true, external_graph is actually a subgraph of
    // node_ptr->covered_edge

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
    else // node_ptr is the root of the tree decomposition, so the recognization is completed
        EmitCompleteSubGraph(chart_item_ptr, grammar_ptr->label_hash);
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

    auto &left_items = static_cast<TreeNode *>(node_ptr->Left())->corresponding_items;
    auto &right_items = static_cast<TreeNode *>(node_ptr->Right())->corresponding_items;
    size_t left_size = left_items.Size();
    size_t right_size = right_items.Size();

    if (left_size == 0 || right_size == 0)
        return;

    // new items of left child <=> new+old items of right child
    for (size_t i = agenda_ptr->num_left_visited_items; i < left_size; ++i)
        for (size_t j = 0; j < right_size; ++j)
            MergeItems(left_items[i], right_items[j], node_ptr);

    // new items of right child <=> old items of left child
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
            decltype(updated_agendas_) empty;
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

    SetCompleteItem(unary_agendas_, 8 /* label_offset */);

    num_indexing_keys_ = unary_agendas_.size();
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

} // namespace tree_v2
} // namespace shrg

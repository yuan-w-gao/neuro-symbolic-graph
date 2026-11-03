#pragma once

#include <queue>
#include <unordered_map>

#include "parser_tree_base.hpp"

namespace shrg {
namespace tree_v1 {

using tree::Agenda;
using tree::TreeNode;
using tree::TreeNodeBase;
using tree::UnaryAgenda;
using ParserBase = tree::TreeSHRGParserBase<TreeNode>;

struct BinaryAgenda : public tree::BinaryAgenda {
    ChartItemList left_items;
    ChartItemList right_items;

    using tree::BinaryAgenda::BinaryAgenda;

    void Clear() override {
        tree::BinaryAgenda::Clear();

        left_items.clear();
        right_items.clear();
    }
};

class TreeSHRGParser : public ParserBase {
  protected:
    std::unordered_map<EdgeHash, UnaryAgenda> unary_agendas_;

    utils::MemoryPool<BinaryAgenda> binary_agendas_pool_;

    void ClearChart();

    void InitializeTree();

    void InitializeChart();

    void MatchTerminalEdges();

    void MergeItems(ChartItem *left_item_ptr, ChartItem *right_item_ptr, TreeNodeBase *node_ptr,
                    bool is_unary_node = false);

    void MergeItems(UnaryAgenda::ActiveItem &item, ChartItem *external_item_ptr) {
        return MergeItems(external_item_ptr, item.chart_item_ptr, item.node_ptr, true);
    }

    void UpdateUnaryNode(UnaryAgenda *agenda_ptr);
    void UpdateBinaryNode(BinaryAgenda *agenda_ptr);

    void EmitCompleteSubGraph(ChartItem *chart_item_ptr, EdgeHash edge_hash);
    void EmitPartialSubgraph(ChartItem *chart_item_ptr, TreeNodeBase *node_ptr, bool sumbit);

  public:
    template <typename DecomposerType>
    TreeSHRGParser(const std::vector<SHRG> &grammars, TreeDecomposer<DecomposerType> &decomposer,
                   const TokenSet &label_set)
        : ParserBase("tree_v1", grammars, decomposer, label_set) {
        InitializeTree();
    }

    template <typename DecomposerType>
    TreeSHRGParser(const std::vector<SHRG> &grammars, TreeDecomposer<DecomposerType> &&decomposer,
                   const TokenSet &label_set)
        : TreeSHRGParser(grammars, decomposer, label_set) {}

    ParserError Parse(const EdsGraph &graph) override;

    const ChartItemList *GetItemsByLabelHash(LabelHash label_hash) override {
        auto it = unary_agendas_.find(label_hash);
        if (it != unary_agendas_.end())
            return &it->second.passive_items.AsList();
        return nullptr;
    }
};

} // namespace tree_v1
} // namespace shrg

REGISTER_TYPE_NAME(shrg::tree_v1::TreeNode);

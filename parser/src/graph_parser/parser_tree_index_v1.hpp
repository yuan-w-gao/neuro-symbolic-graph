#pragma once

#include <queue>
#include <unordered_map>

#include "parser_tree_v1.hpp"

namespace shrg {
namespace tree_index_v1 {

using tree::Agenda;
using tree::Tree;
using tree::TreeGenerator;
using tree::TreeNodeBase;
using tree::UnaryAgenda;
using tree_v1::BinaryAgenda;

class TreeNode : public TreeNodeBase {
  public:
    using TreeNodeBase::TreeNodeBase;

    std::unordered_map<NodeMapping, BinaryAgenda> agendas;

    NodeMapping covered_mask{};
    const std::vector<NodeMapping> *required_masks = nullptr;

    void Clear() override {
        if (Right()) // binary node
            agendas.clear();
    }
};

using ParserBase = tree::TreeSHRGParserBase<TreeNode>;

class TreeSHRGParser : public ParserBase {
  protected:
    std::unordered_map<EdgeHash, std::vector<NodeMapping>> all_required_masks_;
    // passive items grouped by edges and boundary_nodes
    ChartItemMap<UnaryAgenda> unary_agendas_;

    void InitializeTree();

    void ClearChart();

    void InitializeChart();

    void MatchTerminalEdges();

    void MergeItems(ChartItem *left_item_ptr, ChartItem *right_item_ptr, TreeNodeBase *node_ptr,
                    bool is_unary_node = false);

    void MergeItems(UnaryAgenda::ActiveItem &item, ChartItem *external_subgraph_ptr) {
        return MergeItems(external_subgraph_ptr, item.chart_item_ptr, item.node_ptr, true);
    }

    void UpdateUnaryNode(UnaryAgenda *agenda_ptr);
    void UpdateBinaryNode(BinaryAgenda *agenda_ptr);

    void EmitCompleteSubGraph(ChartItem *chart_item_ptr, uint boundary_node_count,
                              LabelHash label_hash, //
                              const std::vector<NodeMapping> *required_masks);

    void EmitCompleteSubGraph(ChartItem *chart_item_ptr, uint boundary_node_count,
                              TreeNodeBase *node_ptr) {
        TreeNode *node_ext_ptr = static_cast<TreeNode *>(node_ptr);
        assert(node_ext_ptr->required_masks);

        EmitCompleteSubGraph(chart_item_ptr, boundary_node_count,   //
                             node_ext_ptr->grammar_ptr->label_hash, //
                             node_ext_ptr->required_masks);
    }

    void EmitPartialSubgraph(ChartItem *chart_item_ptr, TreeNodeBase *node_ptr, bool sumbit);

  public:
    template <typename DecomposerType>
    TreeSHRGParser(const std::vector<SHRG> &grammars, TreeDecomposer<DecomposerType> &decomposer,
                   const TokenSet &label_set)
        : ParserBase("tree_index_v1", grammars, decomposer, label_set) {
        InitializeTree();
    }

    template <typename DecomposerType>
    TreeSHRGParser(const std::vector<SHRG> &grammars, TreeDecomposer<DecomposerType> &&decomposer,
                   const TokenSet &label_set)
        : TreeSHRGParser(grammars, decomposer, label_set) {}

    ParserError Parse(const EdsGraph &graph) override;
};

} // namespace tree_index_v1
} // namespace shrg

REGISTER_TYPE_NAME(shrg::tree_index_v1::TreeNode);

// Local Variables:
// mode: c++
// End:

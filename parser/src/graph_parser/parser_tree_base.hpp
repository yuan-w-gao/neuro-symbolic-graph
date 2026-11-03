#pragma once

#include <queue>
#include <unordered_map>

#include "generator.hpp"
#include "tree_decomposer.hpp"

namespace shrg {
namespace tree {

using MaskMap = std::unordered_map<EdgeHash, std::unordered_set<NodeMapping>>;

struct Agenda {
    bool in_queue;
    bool is_binary;

    Agenda(bool _in_queue, bool _is_binary) : in_queue(_in_queue), is_binary(_is_binary) {}

    virtual void Clear() = 0;
};

struct UnaryAgenda : public Agenda {
    struct ActiveItem {
        ChartItem *chart_item_ptr;
        TreeNodeBase *node_ptr;
    };
    uint num_visited_passive_items = 0;
    uint num_visited_active_items = 0;

    ChartItemSet passive_items;
    std::vector<ActiveItem> active_items;

    UnaryAgenda() : Agenda(false, false) {}

    void Clear() override {
        in_queue = false;

        num_visited_passive_items = 0;
        num_visited_active_items = 0;

        passive_items.Clear();
        active_items.clear();
    }
};

struct BinaryAgenda : public Agenda {
    TreeNodeBase *node_ptr = nullptr;

    uint num_left_visited_items = 0;
    uint num_right_visited_items = 0;

    BinaryAgenda() : Agenda(false, true) {}

    void Clear() override {
        in_queue = false;

        num_left_visited_items = 0;
        num_right_visited_items = 0;
    }
};

class TreeNode : public tree::TreeNodeBase {
  public:
    using tree::TreeNodeBase::TreeNodeBase;

    Agenda *agenda_ptr = nullptr;

    void Clear() override {
        if (Right() && agenda_ptr) // binary node
            agenda_ptr->Clear();
    }
};

class TreeGenerator : public Generator {
  public:
    using Generator::Generator;

    ChartItem *FindChartItemByEdge(ChartItem *chart_item_ptr,
                                   const SHRG::Edge *shrg_edge_ptr) override;
};

template <typename NodeType> class TreeSHRGParserBase : public SHRGParserBase {
  public:
    template <typename DecomposerType>
    using TreeDecomposer = tree::TreeDecomposerTpl<NodeType, DecomposerType>;
    using TreeNode = NodeType;

  protected:
    // tree decomposition of all SHRG grammars
    std::vector<Tree> tree_decompositions_;
    std::queue<Agenda *> updated_agendas_; // chart agenda

    utils::MemoryPool<NodeType> tree_nodes_pool_;

    TreeGenerator generator_;

  public:
    template <typename DecomposerType>
    TreeSHRGParserBase(const char *parser_type, const std::vector<SHRG> &grammars,
                       TreeDecomposer<DecomposerType> &decomposer, const TokenSet &label_set)
        : SHRGParserBase(parser_type, grammars, label_set), //
          tree_decompositions_(grammars.size()),            //
          generator_(this) {
        decomposer.SetPool(&tree_nodes_pool_);
        for (int i = grammars_.size() - 1; i >= 0; --i) {
            const SHRG &grammar = grammars_[i];
            if (!grammar.IsEmpty())
                decomposer.Decompose(tree_decompositions_[i], grammar);
        }
    }

    // template <typename DecomposerType>
    // TreeSHRGParserBase(const char *parser_type, const std::vector<SHRG> &grammars,
    //                    TreeDecomposer<DecomposerType> &&decomposer, const TokenSet &label_set)
    //     : TreeSHRGParserBase(parser_type, grammars, decomposer, label_set){};

    const std::vector<Tree> &TreeDecompositions() const { return tree_decompositions_; }

    Generator *GetGenerator() override { return &generator_; }
};

} // namespace tree
} // namespace shrg

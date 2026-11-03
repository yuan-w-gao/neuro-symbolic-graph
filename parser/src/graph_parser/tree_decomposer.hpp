#pragma once

#include "parser_base.hpp"

namespace shrg {
namespace tree {

class TreeNodeBase;

using DegreeArray = std::array<std::uint8_t, MAX_SHRG_NODE_COUNT>;
using Tree = std::vector<TreeNodeBase *>;

class TreeNodeBase : public GrammarAttributes {
  public:
    // Tree node information //////////////////////////////////////////////////
    const SHRG::Edge *covered_edge_ptr;

    NodeMapping boundary_nodes;

  public:
    TreeNodeBase(const SHRG::Edge *edge = nullptr) : covered_edge_ptr(edge) {}

    virtual ~TreeNodeBase() {}

    virtual void Clear(){};

    TreeNodeBase *Parent() { return parent_; }
    TreeNodeBase *Left() { return left_; }
    TreeNodeBase *Right() { return right_; }
    const TreeNodeBase *Parent() const { return parent_; }
    const TreeNodeBase *Left() const { return left_; }
    const TreeNodeBase *Right() const { return right_; }

    int Width() const;

    void SetLeft(TreeNodeBase *left) {
        assert(left != nullptr); // Set null left child

        left_ = left;
        left->parent_ = this;
    }
    void SetRight(TreeNodeBase *right) {
        assert(right != nullptr); // Set null right child

        right_ = right;
        right->parent_ = this;
    }

    TreeNodeBase *RemoveLeft() {
        if (left_) {
            TreeNodeBase *result = left_;
            left_ = result->parent_ = nullptr;
            return result;
        }
        return nullptr;
    }

  private:
    // tree decomposition of grammar
    TreeNodeBase *parent_ = nullptr; // parent item
    TreeNodeBase *left_ = nullptr;   // left item
    TreeNodeBase *right_ = nullptr;  // right item
};

class TreeDecomposerBase {
  protected:
    void ConstructTree(Tree &tree_nodes, const SHRG &grammar);

  public:
    virtual ~TreeDecomposerBase() {}
    virtual TreeNodeBase *Create(const SHRG::Edge *edge = nullptr) = 0;

    virtual void Decompose(Tree &tree_nodes, const SHRG &grammar) = 0;
};

class NaiveDecomposer : public TreeDecomposerBase {
  protected:
    TreeNodeBase *DepthFirstSearch(Tree &tree_nodes, const SHRG::Edge *current_edge_ptr,
                                   EdgeSet &visited_edges);

    void AttachTreeNode(Tree &tree_nodes, TreeNodeBase *parent, TreeNodeBase *child);

  public:
    void Decompose(Tree &tree_nodes, const SHRG &grammar) override;
};

class TerminalFirstDecomposer : public TreeDecomposerBase {
  protected:
    TreeNodeBase *AttachToList(Tree &tree_nodes, TreeNodeBase *tail_node, TreeNodeBase *new_node);

  public:
    void Decompose(Tree &tree_nodes, const SHRG &grammar) override;
};

class MinimumWidthDecomposer : public NaiveDecomposer {
  private:
    struct Bag {
        int width;
        DegreeArray degrees;
        int parent_index;

        bool operator<(const Bag &other) const { return width < other.width; }
    };

    int min_bag_size_;
    int min_num_bags_;
    Bag min_bags_[MAX_SHRG_EDGE_COUNT * 2];

    void BruteForceSearch(const DegreeArray &all_degrees, Bag *all_bags, int num_bags);

    void ConstructTree(Tree &tree_nodes, const SHRG &grammar);

  public:
    int Treewidth() const { return min_bag_size_ - 1; }

    void Decompose(Tree &tree_nodes, const SHRG &grammar) override;
};

template <typename NodeType, typename DecomposerType>
class TreeDecomposerTpl : public DecomposerType {

  public:
    using Pool = utils::MemoryPool<NodeType>;

  private:
    Pool *pool_ = nullptr;

  protected:
    TreeNodeBase *Create(const SHRG::Edge *edge = nullptr) override { return pool_->Push(edge); }

  public:
    void SetPool(Pool *pool) { pool_ = pool; };
};

} // namespace tree
} // namespace shrg

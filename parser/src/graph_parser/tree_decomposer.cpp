#include "tree_decomposer.hpp"

namespace shrg {
namespace tree {

EdgeSet SetBoundaryNodes(TreeNodeBase *node_ptr) {
    if (!node_ptr->Left()) // leaf node
        return {};
    NodeMapping &boundary_nodes = node_ptr->boundary_nodes;
    const SHRG &grammar = *node_ptr->grammar_ptr;

    boundary_nodes.m8.fill(0);
    // traverse child left node first
    EdgeSet matched_edges = SetBoundaryNodes(node_ptr->Left());
    if (node_ptr->Right()) { // binary node, traverse right node
        EdgeSet right_matched_edges = SetBoundaryNodes(node_ptr->Right());
        matched_edges |= right_matched_edges;
    } else if (node_ptr->Left()) // unary node, set covered_edge
        matched_edges[node_ptr->covered_edge_ptr->index] = true;

    PrecomputeBoundaryNodesForHRG(boundary_nodes, grammar, matched_edges);
    return matched_edges;
}

int MergeDegreeArray(const DegreeArray &all_degrees, //
                     const DegreeArray &degrees1, const DegreeArray &degrees2,
                     DegreeArray &output) {
    int width = 0;
    for (uint k = 0; k < output.size(); ++k) {
        output[k] = degrees1[k] + degrees2[k];
        // active nodes of `bag` is either a boundary node of `bag1` or a boundary node of `bag2`
        if ((degrees1[k] && degrees1[k] < all_degrees[k]) ||
            (degrees2[k] && degrees2[k] < all_degrees[k]))
            ++width;
        assert(output[k] <= all_degrees[k]);
    }
    return width;
}

void ComputeAllDegrees(DegreeArray &all_degrees, const SHRG &grammar) {
    all_degrees.fill(0);
    for (auto &edge : grammar.fragment.edges)
        for (auto node_ptr : edge.linked_nodes)
            ++all_degrees[node_ptr->index];

    // external nodes of the rule should always be external nodes
    for (auto node_ptr : grammar.external_nodes)
        all_degrees[node_ptr->index] += 100;
}

int ComputeTreeWidth(const TreeNodeBase *node_ptr, const DegreeArray &all_degrees,
                     DegreeArray &output) {
    output.fill(0);

    if (!node_ptr->Left()) // leaf_node
        return 0;
    int width;

    DegreeArray left_degrees, right_degrees;
    if (node_ptr->Right()) { // binary node
        width = std::max(ComputeTreeWidth(node_ptr->Left(), all_degrees, left_degrees),
                         ComputeTreeWidth(node_ptr->Right(), all_degrees, right_degrees));
    } else {
        assert(node_ptr->covered_edge_ptr); // unary node

        width = ComputeTreeWidth(node_ptr->Left(), all_degrees, left_degrees);
        right_degrees.fill(0);
        for (auto shrg_node_ptr : node_ptr->covered_edge_ptr->linked_nodes)
            ++right_degrees[shrg_node_ptr->index];
    }

    return std::max(width, MergeDegreeArray(all_degrees, left_degrees, right_degrees, output));
}

int TreeNodeBase::Width() const {
    DegreeArray all_degrees, output;

    ComputeAllDegrees(all_degrees, *grammar_ptr);

    return ComputeTreeWidth(this, all_degrees, output) - 1;
}

void TreeDecomposerBase::ConstructTree(Tree &tree_nodes, const SHRG &grammar) {
    for (TreeNodeBase *tree_node : tree_nodes)
        tree_node->grammar_ptr = &grammar;

    SetBoundaryNodes(tree_nodes[0]);
}

///////////////////////////////////////////////////////////////////////////////
//                              NaiveDecomposer                               //
///////////////////////////////////////////////////////////////////////////////

void NaiveDecomposer::AttachTreeNode(Tree &tree_nodes, TreeNodeBase *parent, TreeNodeBase *child) {
    if (!parent->Left())
        parent->SetLeft(child);
    else {
        TreeNodeBase *binary_node = Create();
        tree_nodes.push_back(binary_node);

        binary_node->SetLeft(parent->Left());
        binary_node->SetRight(child);
        parent->SetLeft(binary_node);
    }
}

TreeNodeBase *NaiveDecomposer::DepthFirstSearch(Tree &tree_nodes,
                                                const SHRG::Edge *current_edge_ptr,
                                                EdgeSet &visited_edges) {
    visited_edges[current_edge_ptr->index] = true;
    TreeNodeBase *tree_node = Create(current_edge_ptr);
    tree_nodes.push_back(tree_node);

    for (const SHRG::Node *node_ptr : current_edge_ptr->linked_nodes) {
        for (const SHRG::Edge *edge_ptr : node_ptr->linked_edges) {
            if (visited_edges[edge_ptr->index])
                continue;
            AttachTreeNode(tree_nodes, tree_node,
                           DepthFirstSearch(tree_nodes, edge_ptr, visited_edges));
        }
    }
    if (!tree_node->Left()) {
        TreeNodeBase *leaf_node = Create();
        tree_node->SetLeft(leaf_node);
        tree_nodes.push_back(leaf_node);
    }
    return tree_node;
}

void NaiveDecomposer::Decompose(Tree &tree_nodes, const SHRG &grammar) {
    EdgeSet visited_edges = 0;
    assert(tree_nodes.empty()); // Re-decompose SHRG rule !!

    const std::vector<SHRG::Node *> &external_nodes = grammar.external_nodes;
    TreeNodeBase tree_root;

    if (!external_nodes.empty())
        for (const SHRG::Edge *edge_ptr : external_nodes[0]->linked_edges) {
            if (visited_edges[edge_ptr->index])
                continue;
            AttachTreeNode(tree_nodes, &tree_root,
                           DepthFirstSearch(tree_nodes, edge_ptr, visited_edges));
        }
    for (const SHRG::Edge &edge : grammar.fragment.edges) {
        if (visited_edges[edge.index])
            continue;
        AttachTreeNode(tree_nodes, &tree_root, //
                       DepthFirstSearch(tree_nodes, &edge, visited_edges));
    }

    assert(tree_root.Left()); // TreeRoot has no child !!
    assert(!tree_root.Right());

    TreeNodeBase *root_ptr = tree_root.RemoveLeft();
    for (auto &node_ptr : tree_nodes) // make sure the index of root pointer is zero
        if (node_ptr == root_ptr) {
            std::swap(tree_nodes[0], node_ptr);
            break;
        }

    ConstructTree(tree_nodes, grammar);
}

///////////////////////////////////////////////////////////////////////////////
//                           MinimumWidthDecomposer                          //
///////////////////////////////////////////////////////////////////////////////

void MinimumWidthDecomposer::Decompose(Tree &tree_nodes, const SHRG &grammar) {
    assert(tree_nodes.empty()); // Re-decompose SHRG rule !!

    if (grammar.fragment.edges.size() > 16) {
        LOG_WARN("rule contains more than 15 edges. !!! fallback to naive decomposer");
        NaiveDecomposer::Decompose(tree_nodes, grammar);
        min_bag_size_ = tree_nodes[0]->Width() + 1;
        return;
    }

    min_bag_size_ = grammar.fragment.nodes.size() + 1;
    min_num_bags_ = 0;

    DegreeArray all_degrees;
    Bag all_bags[MAX_SHRG_EDGE_COUNT * 2];

    int num_bags = 0;

    ComputeAllDegrees(all_degrees, grammar);

    for (auto &edge : grammar.fragment.edges) {
        Bag &bag = all_bags[num_bags++];
        bag.parent_index = -1;
        bag.degrees.fill(0);
        for (auto node_ptr : edge.linked_nodes)
            ++bag.degrees[node_ptr->index];

        int boundary_node_count = 0;
        for (uint i = 0; i < bag.degrees.size(); ++i)
            if (bag.degrees[i] && bag.degrees[i] < all_degrees[i])
                ++boundary_node_count;

        bag.width = boundary_node_count;
    }

    BruteForceSearch(all_degrees, all_bags, num_bags);
    ConstructTree(tree_nodes, grammar);
    TreeDecomposerBase::ConstructTree(tree_nodes, grammar);
}

void MinimumWidthDecomposer::BruteForceSearch(const DegreeArray &all_degrees, Bag *all_bags,
                                              int num_bags) {
    bool has_branch = false;
    for (int i = 0; i < num_bags; ++i) {
        if (all_bags[i].parent_index != -1) // bag is already merged
            continue;
        for (int j = i + 1; j < num_bags; ++j) {
            if (all_bags[j].parent_index != -1) // bag is already merged
                continue;
            has_branch = true;

            Bag &bag1 = all_bags[i];
            Bag &bag2 = all_bags[j];

            Bag &bag = all_bags[num_bags];
            bag.parent_index = -1;
            // the edges introduceed by bag1 and bag2 are disjoint
            bag.width = MergeDegreeArray(all_degrees, bag1.degrees, bag2.degrees, bag.degrees);

            if (bag.width >= min_bag_size_) // pruning
                continue;

            bag1.parent_index = bag2.parent_index = num_bags; // modify
            BruteForceSearch(all_degrees, all_bags, num_bags + 1);
            bag1.parent_index = bag2.parent_index = -1; // restore
        }
    }

    if (!has_branch) {
        Bag *bag_ptr = std::max_element(all_bags, all_bags + num_bags);
        if (bag_ptr->width < min_bag_size_) {
            min_bag_size_ = bag_ptr->width;
            min_num_bags_ = num_bags;
            std::copy_n(all_bags, num_bags, min_bags_);
        }
    }
}

void MinimumWidthDecomposer::ConstructTree(Tree &tree_nodes, const SHRG &grammar) {
    auto &edges = grammar.fragment.edges;
    int num_edges = edges.size();

    assert(min_num_bags_ >= num_edges);

    // remove redundant binary nodes
    for (int i = 0; i < num_edges; ++i) {
        Bag &bag = min_bags_[i];
        if (bag.parent_index == -1) {
            assert(min_num_bags_ == 1);
            continue;
        }

        int sibling_index = -1;
        for (int j = 0; j < min_num_bags_; ++j)
            if (min_bags_[j].parent_index == bag.parent_index && j != i) {
                sibling_index = j;
                break;
            }

        if (sibling_index == -1) { // unary node
            assert(bag.parent_index < num_edges);
            continue;
        }

        // the parent of current bag a binary bag
        Bag &sibling_bag = min_bags_[sibling_index];
        Bag &parent_bag = min_bags_[bag.parent_index];

        // replace its parent with itself
        sibling_bag.parent_index = i;
        bag.parent_index = parent_bag.parent_index;
        parent_bag.parent_index = -1000 - i; // mark as removed
    }

    int root_index = -1;

    tree_nodes.resize(min_num_bags_);
    for (int i = min_num_bags_ - 1; i >= 0; --i) {
        Bag &bag = min_bags_[i];
        if (bag.parent_index < -1) { // the node is removed
            tree_nodes[i] = nullptr;
            continue;
        }

        tree_nodes[i] = Create(i < num_edges ? &edges[i] : nullptr);
        if (bag.parent_index == -1) {
            assert(root_index == -1); // there should be only one root
            root_index = i;
            continue;
        }
    }

    for (int i = min_num_bags_ - 1; i >= 0; --i) {
        int parent_index = min_bags_[i].parent_index;
        if (parent_index == -1) {
            assert(i == root_index);
            continue;
        }
        if (parent_index < -1)
            continue;

        TreeNodeBase *parent_ptr = tree_nodes[parent_index];
        assert(parent_ptr);
        if (!parent_ptr->Left())
            parent_ptr->SetLeft(tree_nodes[i]);
        else {
            assert(!parent_ptr->Right());
            parent_ptr->SetRight(tree_nodes[i]);
        }
    }

    assert(root_index != -1);
    // root of the tree should be stored in the first place
    std::swap(tree_nodes[0], tree_nodes[root_index]);
    // remove redundant nullptr
    tree_nodes.erase(std::remove(tree_nodes.begin(), tree_nodes.end(), nullptr), tree_nodes.end());

    for (int i = tree_nodes.size() - 1; i >= 0; --i) {
        TreeNodeBase *node_ptr = tree_nodes[i];
        assert((!node_ptr->Right() && node_ptr->covered_edge_ptr) ||
               (node_ptr->Right() && !node_ptr->covered_edge_ptr));

        if (!node_ptr->Left()) { // attach leaf node;
            assert(!node_ptr->Right());
            TreeNodeBase *leaf_node = Create();
            node_ptr->SetLeft(leaf_node);
            tree_nodes.push_back(leaf_node);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
//                          TerminalFirstDecomposer                          //
///////////////////////////////////////////////////////////////////////////////

TreeNodeBase *TerminalFirstDecomposer::AttachToList(Tree &tree_nodes, TreeNodeBase *tail_node,
                                                    TreeNodeBase *new_node) {
    tree_nodes.push_back(new_node);
    new_node->SetLeft(tail_node);
    return new_node;
}

void TerminalFirstDecomposer::Decompose(Tree &tree_nodes, const SHRG &grammar) {
    assert(tree_nodes.empty()); // Re-decompose SHRG rule !!

    TreeNodeBase *last_node = Create();
    tree_nodes.push_back(last_node);

    std::vector<SHRG::Edge *> edges;

    edges.insert(edges.end(), grammar.terminal_edges.begin(), grammar.terminal_edges.end());
    edges.insert(edges.end(), grammar.nonterminal_edges.begin(), grammar.nonterminal_edges.end());

    for (const SHRG::Edge *edge_ptr : edges) {
        TreeNodeBase *new_node = Create(edge_ptr);
        tree_nodes.push_back(new_node);
        new_node->SetLeft(last_node);
        last_node = new_node;
    }

    std::reverse(tree_nodes.begin(), tree_nodes.end());

    ConstructTree(tree_nodes, grammar);
}

} // namespace tree
} // namespace shrg

#pragma once

#include <vector>

#include "tokens.hpp"

namespace shrg {

using Label = int;
using LabelHash = std::size_t;
using EdgeHash = std::size_t;
const Label EMPTY_LABEL = -1;

struct Empty {};

// [0-2)       [2-8)            [8-32)
// is_terminal linked_node_size label
inline LabelHash MakeLabelHash(Label label, int linked_node_size, bool is_terminal) {
    return (((label) << 8) | ((linked_node_size) << 2) | ((is_terminal) ? 1 : 0));
}

enum class NodeType : std::uint8_t {
    kFixed,     /* only has terminal edges */
    kSemiFixed, /* has both terminal and non-temrinal edges */
    kFree       /* only has non-terminal edges */
};

template <typename NodeBase = Empty, typename EdgeBase = Empty> class HyperGraph {
  public:
    struct Edge;
    struct Node : public NodeBase {
        int index = -1;
        bool is_external = false;
        NodeType type;
        std::vector<Edge *> linked_edges;
    };

    struct Edge : public EdgeBase {
        int index = -1;
        Label label = EMPTY_LABEL;
        bool is_terminal;

        // the order of nodes is very important
        std::vector<Node *> linked_nodes;

        bool IsEdgeConnected(const Edge &other) const {
            // TODO: find more quick way to check whether two edges have at least one common node
            for (const Node *node_ptr : linked_nodes)
                for (const Node *other_node_ptr : other.linked_nodes)
                    if (node_ptr == other_node_ptr)
                        return true;
            return false;
        }

        EdgeHash Hash() const { return MakeLabelHash(label, linked_nodes.size(), is_terminal); }
    };

    std::vector<Edge> edges;
    std::vector<Node> nodes;
};

} // namespace shrg

// Local Variables:
// mode: c++
//  End:

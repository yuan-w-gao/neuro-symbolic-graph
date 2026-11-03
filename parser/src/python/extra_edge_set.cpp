#include "extra_edge_set.hpp"

namespace shrg {

IntVec EdgeSet_ToIndices(const EdgeSet &self) {
    IntVec edges;
    for (uint i = 0; i < self.size(); ++i)
        if (self[i])
            edges.push_back(i);
    return edges;
}

void EdgeSet_ToNodeList(const EdgeSet &edge_set, const EdsGraph &graph, IntVec &result) {
    bool covered_nodes[MAX_GRAPH_NODE_COUNT];
    std::fill_n(covered_nodes, graph.nodes.size(), false);
    for (uint i = 0; i < edge_set.size(); ++ i) {
        if (!edge_set[i])
            continue;
        for (auto node_ptr : graph.edges[i].linked_nodes) {
            int index = node_ptr->index;
            if (!covered_nodes[index]) {
                covered_nodes[index] = true;
                result.push_back(index);
            };
        }
    }
}

void EdgeSet_ToNodeList(const EdgeSet &edge_set, const EdsGraph &graph, IntVec &result,
                        int *covered_nodes) {
    int num_nodes = graph.nodes.size();
    std::fill_n(covered_nodes, num_nodes, -1);
    for (uint i = 0; i < edge_set.size(); ++ i) {
        if (!edge_set[i])
            continue;
        for (auto node_ptr : graph.edges[i].linked_nodes)
            covered_nodes[node_ptr->index] = 0;
    }
    for (int i = 0; i < num_nodes; ++i) {
        if (covered_nodes[i] < 0)
            continue;
        auto &node = graph.nodes[i];
        size_t type = 0;
        for (auto edge_ptr : node.linked_edges) {
            if (!edge_set[edge_ptr->index])
                // type == 1 => boundary node and pred edge is matched
                // type == 2 => boundary node and pred edge is not matched_edges
                type = std::max(type, 3 - edge_ptr->linked_nodes.size());
        }
        result.push_back(i);
        covered_nodes[i] = type;
    }
}

void EdgeSet_ToNodeList(const EdgeSet &edge_set, const EdsGraph &graph, Partition &result) {
    int covered_nodes[MAX_GRAPH_NODE_COUNT];
    EdgeSet_ToNodeList(edge_set, graph, std::get<0>(result), covered_nodes);
    int num_nodes = graph.nodes.size();
    for (int i = 0; i < num_nodes; ++i)
        if (covered_nodes[i] > 0) {
            std::get<1>(result).push_back(i);
            std::get<2>(result).push_back(covered_nodes[i]);
            std::get<3>(result).push_back(-1);
        }
}

void EdgeSet_ToNodeList(const EdgeSet &edge_set, const NodeMapping &mapping, const EdsGraph &graph,
                        Partition &result) {
    int covered_nodes[MAX_GRAPH_NODE_COUNT];
    EdgeSet_ToNodeList(edge_set, graph, std::get<0>(result), covered_nodes);
    for (uint i = 0; i < mapping.size(); ++i) {
        auto v = mapping[i] - 1;
        if (v < 0)
            break; // for complete subgraph
        int type = covered_nodes[v];
        if (type > 0) {
            std::get<1>(result).push_back(v);
            std::get<2>(result).push_back(type);
            std::get<3>(result).push_back(i);
        }
    }
}

Partition EdgeSet_ToNodeList(const EdgeSet &edge_set, const EdsGraph &graph) {
    Partition result;
    EdgeSet_ToNodeList(edge_set, graph, result);
    return result;
}

} // namespace shrg

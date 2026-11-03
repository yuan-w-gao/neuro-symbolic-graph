#pragma once

#include "../graph_parser/parser_chart_item.hpp"

namespace shrg {

using IntVec = std::vector<int>;
using Partition = std::tuple<IntVec, IntVec, IntVec, IntVec>;

IntVec EdgeSet_ToIndices(const EdgeSet &self);

void EdgeSet_ToNodeList(const EdgeSet &edge_set, const EdsGraph &graph, Partition &result);

void EdgeSet_ToNodeList(const EdgeSet &edge_set, const NodeMapping &mapping, const EdsGraph &graph,
                        Partition &result);

void EdgeSet_ToNodeList(const EdgeSet &edge_set, const EdsGraph &graph, IntVec &result);

Partition EdgeSet_ToNodeList(const EdgeSet &edge_set, const EdsGraph &graph);

} // namespace shrg

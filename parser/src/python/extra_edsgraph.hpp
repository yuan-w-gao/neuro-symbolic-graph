#pragma once

#include <pybind11/pybind11.h>

#include "../graph_parser/edsgraph.hpp"

namespace shrg {

pybind11::object EdsGraphEdge_ToTuple(const EdsGraph::Edge &self, bool use_index);

pybind11::dict EdsGraph_EdgesMap(const EdsGraph &self);

} // namespace shrg

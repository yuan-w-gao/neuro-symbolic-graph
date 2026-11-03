#include "extra_edsgraph.hpp"
#include "../manager.hpp"

namespace shrg {

namespace py = pybind11;

py::object EdsGraphEdge_ToTuple(const EdsGraph::Edge &self, bool use_index) {
    auto &label_set = Manager::manager.label_set;
    auto &nodes = self.linked_nodes;
    switch (nodes.size()) {
    case 1:
        return use_index ? py::make_tuple(nodes[0]->index, self.label)
                         : py::make_tuple(nodes[0]->id, label_set[self.label]);
    case 2:
        return use_index ? py::make_tuple(nodes[0]->index, nodes[1]->index, self.label)
                         : py::make_tuple(nodes[0]->id, nodes[1]->id, label_set[self.label]);
    }
    return py::object();
}

py::dict EdsGraph_EdgesMap(const EdsGraph &self) {
    py::dict result;
    for (const EdsGraph::Edge &edge : self.edges) {
        if (edge.linked_nodes.size() > 2U)
            throw std::runtime_error(self.sentence_id + " has invalid edges ");
        result[EdsGraphEdge_ToTuple(edge, false)] = edge.index;
    }
    return result;
}

} // namespace shrg

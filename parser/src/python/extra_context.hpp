#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../manager.hpp"

namespace shrg {

inline const SHRG::Edge *Left(const SHRG *ptr) {
    return ptr->nonterminal_edges.size() < 1 ? nullptr : ptr->nonterminal_edges[0];
}

inline const SHRG::Edge *Right(const SHRG *ptr) {
    return ptr->nonterminal_edges.size() < 2 ? nullptr : ptr->nonterminal_edges[1];
}

pybind11::list Context_ExportDerivation(const Context &self, //
                                        pybind11::list &gold_derivation,
                                        pybind11::list &gold_subgraphs, uint num_negative_samples);

pybind11::list Context_FindBestDerivation(const Context &self, ChartItem &root);

pybind11::tuple Context_SplitItem(const Context &context, ChartItem &chart_item);
pybind11::tuple Context_SplitItem(const Context &context, ChartItem &chart_item,
                                  const EdsGraph &graph);

} // namespace shrg

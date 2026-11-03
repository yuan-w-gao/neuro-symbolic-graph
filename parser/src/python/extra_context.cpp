 #include "extra_context.hpp"
#include "extra_edge_set.hpp"

namespace shrg {

namespace py = pybind11;

py::tuple Context_SplitItem(const Context &context, ChartItem &chart_item) {
    if (!chart_item.attrs_ptr || !context.Check())
        throw std::runtime_error("empty chart_item or empty context");

    const SHRG *grammar_ptr = chart_item.attrs_ptr->grammar_ptr;
    const SHRG::Edge *left_edge = Left(grammar_ptr);
    const SHRG::Edge *right_edge = Right(grammar_ptr);
    if (!left_edge && !right_edge)
        return py::make_tuple(nullptr, nullptr);

    Generator *generator = context.parser->GetGenerator();
    const ChartItem *left_ptr = generator->FindChartItemByEdge(&chart_item, left_edge);
    assert(left_ptr);
    if (!right_edge)
        return py::make_tuple(py::cast(left_ptr, py::return_value_policy::reference), nullptr);

    const ChartItem *right_ptr = generator->FindChartItemByEdge(&chart_item, right_edge);
    assert(right_ptr);
    return py::make_tuple(py::cast(left_ptr, py::return_value_policy::reference), //
                          py::cast(right_ptr, py::return_value_policy::reference));
}

py::tuple Context_SplitItem(const Context &context, ChartItem &chart_item, const EdsGraph &graph) {
    if (!chart_item.attrs_ptr || !context.Check())
        throw std::runtime_error("empty chart_item or empty context");

    IntVec center;
    const SHRG *grammar_ptr = chart_item.attrs_ptr->grammar_ptr;
    const SHRG::Edge *left_edge = Left(grammar_ptr);
    const SHRG::Edge *right_edge = Right(grammar_ptr);
    if (!left_edge && !right_edge) {
        EdgeSet_ToNodeList(chart_item.edge_set, graph, center);
        return py::make_tuple(std::move(center), nullptr, nullptr);
    }

    Generator *generator = context.parser->GetGenerator();
    const ChartItem *left_ptr = generator->FindChartItemByEdge(&chart_item, left_edge);
    assert(left_ptr);
    if (!right_edge) {
        EdgeSet_ToNodeList(chart_item.edge_set & ~left_ptr->edge_set, graph, center);
        return py::make_tuple(std::move(center),
                              py::cast(left_ptr, py::return_value_policy::reference), //
                              nullptr);
    }

    const ChartItem *right_ptr = generator->FindChartItemByEdge(&chart_item, right_edge);
    assert(right_ptr);
    EdgeSet_ToNodeList(chart_item.edge_set & ~left_ptr->edge_set & ~right_ptr->edge_set, graph,
                       center);
    return py::make_tuple(std::move(center),
                          py::cast(left_ptr, py::return_value_policy::reference), //
                          py::cast(right_ptr, py::return_value_policy::reference));
}

} // namespace shrg

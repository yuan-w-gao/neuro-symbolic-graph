#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

#include "extra_context.hpp"
#include "../manager.hpp"
namespace shrg {

namespace py = pybind11;

static const int VISITED_FLAG = -2000;

float FindBestDerivation(Generator *generator, ChartItem *root_ptr) {
    if (root_ptr->status == VISITED_FLAG)
        return root_ptr->score;

    float log_sum_scores = 0;
    ChartItem *ptr = root_ptr;
    do {
        log_sum_scores += std::exp(ptr->score);
        ptr = ptr->next_ptr;
    } while (ptr != root_ptr);
    log_sum_scores = std::log(log_sum_scores);

    float max_score = -std::numeric_limits<float>::infinity();
    ChartItem *max_subgraph_ptr = root_ptr;

    ptr = root_ptr;
    do {
        float current_score = ptr->score - log_sum_scores;

        const SHRG *grammar_ptr = ptr->attrs_ptr->grammar_ptr;
        for (auto edge_ptr : grammar_ptr->nonterminal_edges)
            current_score +=
                FindBestDerivation(generator, generator->FindChartItemByEdge(ptr, edge_ptr));

        if (max_score < current_score) {
            max_score = current_score;
            max_subgraph_ptr = ptr;
        }

        ptr = ptr->next_ptr;
    } while (ptr != root_ptr);

    if (max_subgraph_ptr != root_ptr)
        root_ptr->Swap(*max_subgraph_ptr);

    root_ptr->status = VISITED_FLAG;
    root_ptr->score = max_score;
    return root_ptr->score;
}

py::list Context_FindBestDerivation(const Context &self, ChartItem &root) {
    if (!root.attrs_ptr || !self.Check())
        throw std::runtime_error("empty chart_item or empty context");

    ChartItem *item_stack[MAX_GRAPH_NODE_COUNT * 2];
    int size = 0;

    py::list derivation;

    Generator *generator = self.parser->GetGenerator();
    FindBestDerivation(generator, &root);

    item_stack[size++] = &root;
    while (size > 0) {
        auto ptr = item_stack[--size];

        const SHRG *grammar_ptr = ptr->attrs_ptr->grammar_ptr;
        for (auto edge_ptr : grammar_ptr->nonterminal_edges)
            item_stack[size++] = generator->FindChartItemByEdge(ptr, edge_ptr);

        if (grammar_ptr->cfg_rules.size() > 1U)
            derivation.append(py::cast(ptr, py::return_value_policy::reference));
    }

    return derivation;
}

} // namespace shrg

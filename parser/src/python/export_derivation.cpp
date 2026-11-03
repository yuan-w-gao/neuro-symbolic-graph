#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>

#include "extra_chart_item.hpp"
#include "extra_context.hpp"

namespace shrg {

namespace py = pybind11;

void SplitItem(Generator *generator,                         //
               ChartItem &chart_item, const EdsGraph &graph, //
               IntVec &center, Partition &left, Partition &right) {
    const SHRG *grammar_ptr = chart_item.attrs_ptr->grammar_ptr;
    assert(grammar_ptr->nonterminal_edges.size() <= 2);
    const SHRG::Edge *left_edge = Left(grammar_ptr);
    const SHRG::Edge *right_edge = Right(grammar_ptr);

    center.clear();
    std::get<0>(left).clear();
    std::get<1>(left).clear();
    std::get<2>(left).clear();
    std::get<3>(left).clear();
    std::get<0>(right).clear();
    std::get<1>(right).clear();
    std::get<2>(right).clear();
    std::get<3>(right).clear();

    EdgeSet center_edge_set = chart_item.edge_set;
    if (left_edge) {
        ChartItem *left_ptr = generator->FindChartItemByEdge(&chart_item, left_edge);
        assert(left_ptr);
        center_edge_set &= ~left_ptr->edge_set;
        ChartItem_ToList(*left_ptr, graph, left);
    }
    if (right_edge) {
        const ChartItem *right_ptr = generator->FindChartItemByEdge(&chart_item, right_edge);
        assert(right_ptr);
        center_edge_set &= ~right_ptr->edge_set;
        ChartItem_ToList(*right_ptr, graph, right);
    }
    EdgeSet_ToNodeList(center_edge_set, graph, center);
}

inline bool IsSubGraphMatched(ChartItem *chart_item_ptr, const EdgeSet &gold_edge_set,
                              const NodeMapping &gold_mapping) {
    if (!chart_item_ptr)
        return gold_edge_set.none();
    return chart_item_ptr->edge_set == gold_edge_set &&
           chart_item_ptr->boundary_node_mapping == gold_mapping;
}

bool IsSubGraphMatched(Generator *generator, ChartItem &chart_item,
                       const EdgeSet &gold_center,            //
                       const EdgeSet &gold_left,              //
                       const EdgeSet &gold_right,             //
                       const NodeMapping &gold_left_mapping,  //
                       const NodeMapping &gold_right_mapping, //
                       ChartItem *&left_item_ptr, ChartItem *&right_item_ptr) {
    if (!chart_item.attrs_ptr)
        return false;

    const SHRG *grammar_ptr = chart_item.attrs_ptr->grammar_ptr;
    const SHRG::Edge *left_edge = Left(grammar_ptr);
    const SHRG::Edge *right_edge = Right(grammar_ptr);

    bool is_left_none = gold_left.none();
    bool is_right_none = gold_right.none();

    if (!left_edge && !right_edge)
        return is_left_none && is_right_none && chart_item.edge_set == gold_center;

    auto ptr1 = generator->FindChartItemByEdge(&chart_item, left_edge);
    auto ptr2 = right_edge ? generator->FindChartItemByEdge(&chart_item, right_edge) : nullptr;
    EdgeSet center_edge_set = chart_item.edge_set & ~ptr1->edge_set;

    assert(ptr1);
    if (right_edge) {
        assert(ptr2);
        center_edge_set &= ~ptr2->edge_set;
    }

    if (center_edge_set != gold_center)
        return false;

    if (IsSubGraphMatched(ptr1, gold_left, gold_left_mapping) &&
        IsSubGraphMatched(ptr2, gold_right, gold_right_mapping)) {
        left_item_ptr = ptr1;
        right_item_ptr = ptr2;
        return true;
    }

    if (IsSubGraphMatched(ptr2, gold_left, gold_left_mapping) &&
        IsSubGraphMatched(ptr1, gold_right, gold_right_mapping)) {
        left_item_ptr = ptr2;
        right_item_ptr = ptr1;
        return true;
    }

    return false;
}

inline void SetEdgeSet(EdgeSet &edge_set, py::handle &&t) {
    if (!t.is_none())
        for (auto v : t)
            edge_set[v.cast<int>()] = true;
}

inline void SetMapping(NodeMapping &mapping, py::handle &&t) {
    if (!t.is_none()) {
        int i = 0;
        for (auto v : t)
            mapping[i++] = v.cast<int>() + 1;
    }
}

bool DepthFirstSearch(Generator *generator,                            //
                      py::list &gold_derivation, py::list &partitions, //
                      ChartItemList &item_ptrs, int step) {
    if (step == -1)
        return true;

    auto chart_item_ptr = item_ptrs[step];
    if (!chart_item_ptr)
        return DepthFirstSearch(generator, gold_derivation, partitions, item_ptrs, step - 1);

    py::tuple t1 = gold_derivation[step];
    py::tuple t2 = partitions[step];

    // gold partitions is set of edges
    EdgeSet gold_center = 0, gold_left = 0, gold_right = 0;
    NodeMapping left_mapping{}, right_mapping{};

    SetEdgeSet(gold_center, t2[0]);
    SetEdgeSet(gold_left, t2[1]);
    SetEdgeSet(gold_right, t2[2]);

    int shrg_index = t1[0].cast<int>();
    int left_index = t1[1].cast<int>();
    int right_index = t1[2].cast<int>();

    if (left_index >= 0) {
        py::tuple t = partitions[left_index];
        if (!t.is_none())
            SetMapping(left_mapping, t[3]);
    }
    if (right_index >= 0) {
        py::tuple t = partitions[right_index];
        if (!t.is_none())
            SetMapping(right_mapping, t[3]);
    }

    // std::cout << "Step: " << step;
    // debug::Print(gold_center, std::cout << "\n> C: ");
    // debug::Print(gold_left, std::cout << "\n> L: ");
    // debug::Print(gold_right, std::cout << "\n> R: ");
    // std::cout << std::endl;

    ChartItem *left_item_ptr = nullptr;
    ChartItem *right_item_ptr = nullptr;

    auto current_ptr = chart_item_ptr;
    do {
        auto item_ptr = current_ptr->attrs_ptr;
        if (item_ptr)
            for (auto &cfg_rule : item_ptr->grammar_ptr->cfg_rules)
                if (cfg_rule.shrg_index == shrg_index) {
                    if (IsSubGraphMatched(generator, *current_ptr,            //
                                          gold_center, gold_left, gold_right, //
                                          left_mapping, right_mapping,        //
                                          left_item_ptr, right_item_ptr)) {
                        // branch
                        if (left_index >= 0)
                            item_ptrs[left_index] = left_item_ptr;
                        if (right_index >= 0)
                            item_ptrs[right_index] = right_item_ptr;
                        // !!! NOTE: cfg_index is the index in cfg_rules of a given grammar. But
                        // here we use to store shrg_index (index of in all cfg_rules)
                        current_ptr->status = shrg_index; // set shrg_index
                        item_ptrs[step] = current_ptr;    // select correct chart_item

                        if (DepthFirstSearch(generator, gold_derivation, partitions, item_ptrs,
                                             step - 1))
                            return true;
                    }
                }
        current_ptr = current_ptr->next_ptr;
    } while (current_ptr != chart_item_ptr);

    return false;
}

py::list Context_ExportDerivation(const Context &self,                             //
                                  py::list &gold_derivation, py::list &partitions, //
                                  uint num_negative_samples) {
    if (!self.Check() || !self.parser->Graph())
        throw std::runtime_error("empty context");

    py::list result;
    ChartItemList item_ptrs(gold_derivation.size(), nullptr);
    ChartItemList sampled_ptrs;

    const EdsGraph &graph = *self.parser->Graph();
    auto grammar_start_addr = self.manager_ptr->grammars.data();
    Generator *generator = self.parser->GetGenerator();

    auto root_ptr = self.parser->Result();
    if (!root_ptr)
        return result;

    item_ptrs.back() = root_ptr;
    if (DepthFirstSearch(generator, gold_derivation, partitions, item_ptrs,
                         partitions.size() - 1)) {
        IntVec center;
        Partition left, right;

        for (auto chart_item_ptr : item_ptrs) {
            if (!chart_item_ptr) {
                result.append(nullptr);
                continue;
            }
            sampled_ptrs.clear();
            auto current_ptr = chart_item_ptr;
            do {
                sampled_ptrs.push_back(current_ptr);
                if (sampled_ptrs.size() > (num_negative_samples << 4)) {
                    LOG_WARN("Too many negative samples");
                    break;
                }
                current_ptr = current_ptr->next_ptr;
            } while (current_ptr != chart_item_ptr);

            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(sampled_ptrs.begin() + 1, sampled_ptrs.end(), g);
            if (sampled_ptrs.size() > num_negative_samples + 1)
                sampled_ptrs.erase(sampled_ptrs.begin() + num_negative_samples + 1,
                                   sampled_ptrs.end());

            py::list chart_items;
            for (auto current_ptr : sampled_ptrs) {
                SplitItem(generator, *current_ptr, graph, center, left, right);
                int grammar_index = current_ptr->attrs_ptr->grammar_ptr - grammar_start_addr;
                if (std::get<0>(left).empty() && std::get<0>(right).empty())
                    chart_items.append(py::make_tuple(grammar_index, center, nullptr, nullptr));
                else if (std::get<0>(right).empty())
                    chart_items.append(py::make_tuple(grammar_index, center, left, nullptr));
                else
                    chart_items.append(py::make_tuple(grammar_index, center, left, right));
            }
            assert(chart_item_ptr->status != ChartItem::kEmpty);
            result.append(py::make_tuple(chart_item_ptr->status, chart_items));
        }
        return result;
    }

    return result;
}

} // namespace shrg

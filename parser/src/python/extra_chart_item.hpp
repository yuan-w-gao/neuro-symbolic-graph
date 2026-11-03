#pragma once

#include "extra_edge_set.hpp"

namespace shrg {

std::string ChartItem_ToString(const ChartItem &self, const EdsGraph &graph);

std::string ChartItem_ToDot(const ChartItem &self, const EdsGraph &graph);

int ChartItem_GrammarIndex(const ChartItem &self);

void ChartItem_ToList(const ChartItem &self, const EdsGraph &graph, Partition &result);
Partition ChartItem_ToList(const ChartItem &self, const EdsGraph &graph);

struct ChartItemIterator : public std::iterator<std::input_iterator_tag, ChartItem> {
    ChartItem *current_ptr;
    bool started;

    ChartItemIterator(ChartItem *ptr, bool _started) : current_ptr(ptr), started(_started) {}

    ChartItemIterator &operator++() {
        started = true;
        current_ptr = current_ptr->next_ptr;
        return *this;
    }
    ChartItemIterator operator++(int) {
        auto tmp = *this;
        operator++();
        return tmp;
    }
    bool operator==(const ChartItemIterator &rhs) const {
        return current_ptr == rhs.current_ptr && started == rhs.started;
    }
    bool operator!=(const ChartItemIterator &rhs) const {
        return current_ptr != rhs.current_ptr || started != rhs.started;
    }
    ChartItem &operator*() { return *current_ptr; }
};

} // namespace shrg

namespace std {

using shrg::ChartItemIterator;
inline ChartItemIterator begin(ChartItem &item) { return ChartItemIterator(&item, false); }
inline ChartItemIterator end(ChartItem &item) { return ChartItemIterator(&item, true); }

} // namespace std

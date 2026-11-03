//
// Created by Yuan Gao on 03/06/2024.
//

#ifndef SHRG_GRAPH_PARSER_TYPES_H
#define SHRG_GRAPH_PARSER_TYPES_H

#include "../graph_parser/parser_chart_item.hpp"
#include <queue>

namespace shrg {

using ParentTup = std::tuple<ChartItem *, std::vector<ChartItem *>>;
using RuleVector = std::vector<SHRG *>;
using LabelToRule = std::map<LabelHash, RuleVector>;
using LabelCount = std::map<LabelHash, double>;

const int VISITED = -2000;

struct LessThanByLevel {
    bool operator()(ChartItem *lhs, const ChartItem *rhs) const { return lhs->level > rhs->level; }
};

typedef std::priority_queue<ChartItem *, std::vector<ChartItem *>, LessThanByLevel> NodeLevelPQ;

}
#endif // SHRG_GRAPH_PARSER_TYPES_H

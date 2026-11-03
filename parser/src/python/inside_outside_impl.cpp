//
// Standalone implementations of inside-outside functions for Python bindings
// These mirror the EMBase methods but as free functions
//
#include "../manager.hpp"
#include "../em_framework/em_utils.hpp"
#include "../em_framework/em_types.hpp"
#include <queue>

namespace shrg {

// Use VISITED and LessThanByLevel from em_types.hpp (via namespace)
using namespace shrg;

// Standalone addParentPointer (uses generator parameter)
void addParentPointer(ChartItem *root, Generator *generator, int level) {
    ChartItem *ptr = root;

    do {
        if (level > ptr->level) {
            ptr->level = level;
        }

        const SHRG *rule = ptr->attrs_ptr->grammar_ptr;
        if (ptr->child_visited_status != VISITED) {
            for (auto edge_ptr : rule->nonterminal_edges) {
                ChartItem *child = generator->FindChartItemByEdge(ptr, edge_ptr);
                ptr->children.push_back(child);
                addParentPointer(child, generator, ptr->level + 1);
            }
            for (size_t i = 0; i < ptr->children.size(); i++) {
                std::vector<ChartItem *> sib;
                for (size_t j = 0; j < ptr->children.size(); j++) {
                    if (j != i) {
                        sib.push_back(ptr->children[j]);
                    }
                }
                auto res = std::make_tuple(ptr, sib);
                ptr->children[i]->parents_sib.push_back(res);
            }
            ptr->child_visited_status = VISITED;
        } else {
            for (auto child : ptr->children) {
                addParentPointer(child, generator, ptr->level + 1);
            }
        }

        ptr = ptr->next_ptr;
    } while (ptr != root);
}

// Standalone addRulePointer (uses shrg_rules parameter)
void addRulePointer(ChartItem *root, std::vector<SHRG *> &shrg_rules) {
    if (root->rule_visited == VISITED) {
        return;
    }

    ChartItem *ptr = root;
    do {
        auto grammar_index = ptr->attrs_ptr->grammar_ptr->best_cfg_ptr->shrg_index;
        ptr->rule_ptr = shrg_rules[grammar_index];

        ptr->rule_visited = VISITED;
        for (ChartItem *child : ptr->children) {
            addRulePointer(child, shrg_rules);
        }
        ptr = ptr->next_ptr;
    } while (ptr != root);
}

// Standalone computeInside
double computeInside(ChartItem *root) {
    if (root->inside_visited_status == VISITED) {
        return root->log_inside_prob;
    }

    ChartItem *ptr = root;
    double log_inside = ChartItem::log_zero;

    do {
        double curr_log_inside = ptr->rule_ptr->log_rule_weight;
        curr_log_inside = sanitizeLogProb(curr_log_inside);

        double log_children = 0.0;
        for (ChartItem *child : ptr->children) {
            log_children += computeInside(child);
        }
        log_children = sanitizeLogProb(log_children);

        curr_log_inside += log_children;
        curr_log_inside = sanitizeLogProb(curr_log_inside);

        log_inside = addLogs(log_inside, curr_log_inside);
        log_inside = sanitizeLogProb(log_inside);

        ptr = ptr->next_ptr;
    } while (ptr != root);

    do {
        ptr->log_inside_prob = log_inside;
        ptr->inside_visited_status = VISITED;
        ptr = ptr->next_ptr;
    } while (ptr != root);

    return log_inside;
}

static void computeOutsideNode(ChartItem *root, NodeLevelPQ &pq) {
    ChartItem *ptr = root;

    do {
        if (ptr->parents_sib.empty()) {
            ptr = ptr->next_ptr;
            continue;
        }

        double log_outside = ChartItem::log_zero;

        for (auto &parent_sib : ptr->parents_sib) {
            ChartItem *parent = getParent(parent_sib);
            std::vector<ChartItem*> siblings = getSiblings(parent_sib);

            double curr_log_outside = parent->rule_ptr->log_rule_weight;
            curr_log_outside += parent->log_outside_prob;

            for (auto sib : siblings) {
                curr_log_outside += sib->log_inside_prob;
            }

            log_outside = addLogs(log_outside, curr_log_outside);
        }
        ptr->log_outside_prob = log_outside;
        ptr->outside_visited_status = VISITED;

        ptr = ptr->next_ptr;
    } while (ptr != root);

    double root_log_outside = root->log_outside_prob;

    do {
        if (ptr->outside_visited_status != VISITED) {
            ptr->log_outside_prob = root_log_outside;
            ptr->outside_visited_status = VISITED;
        }

        for (ChartItem *child : ptr->children) {
            pq.push(child);
        }

        ptr = ptr->next_ptr;
    } while (ptr != root);
}

// Standalone computeOutside
void computeOutside(ChartItem *root) {
    NodeLevelPQ pq;
    ChartItem *ptr = root;

    do {
        ptr->log_outside_prob = 0.0;
        ptr = ptr->next_ptr;
    } while (ptr != root);

    pq.push(ptr);

    do {
        ChartItem *node = pq.top();
        computeOutsideNode(node, pq);
        pq.pop();
    } while (!pq.empty());
}

} // namespace shrg

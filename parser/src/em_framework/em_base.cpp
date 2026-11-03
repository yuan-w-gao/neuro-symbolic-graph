//
// Created by Yuan Gao on 03/06/2024.
//
#include "em_utils.hpp"
#include "em_base.hpp"

namespace shrg {
namespace em{
const int EMBase::VISITED = -2000;

void EMBase::addParentPointer(ChartItem *root, int level){
//    std::cout << "addParentPointer" << std::endl;
        ChartItem *ptr = root;

        do {
            if (level > ptr->level) {
                ptr->level = level;
            }

            const SHRG *rule = ptr->attrs_ptr->grammar_ptr;
            if (ptr->child_visited_status != EMBase::VISITED) {
                for (auto edge_ptr : rule->nonterminal_edges) {
                    ChartItem *child = generator->FindChartItemByEdge(ptr, edge_ptr);
                    ptr->children.push_back(child);
                    addParentPointer(child, ptr->level + 1);
                }
                for (int i = 0; i < ptr->children.size(); i++) {
                    std::vector<ChartItem *> sib;
                    std::tuple<ChartItem *, std::vector<ChartItem *>> res;
                    for (int j = 0; j < ptr->children.size(); j++) {
                        if (j == i) {
                            continue;
                        }
                        sib.push_back(ptr->children[j]);
                    }

                    if (ptr) {
                        res = std::make_tuple(ptr, sib);
                        ptr->children[i]->parents_sib.push_back(res);
                    }
                }
                ptr->child_visited_status = VISITED;
            } else {
                for (auto child : ptr->children) {
                    addParentPointer(child, ptr->level + 1);
                }
            }
            assert(ptr->children.size() == rule->nonterminal_edges.size());

            ptr = ptr->next_ptr;
        } while (ptr != root);
}

void EMBase::addChildren(ChartItem* root) {
        if(root->child_visited_status == EMBase::VISITED){
            return;
        }
        ChartItem* start = root;
        ChartItem* ptr = start;

        do {
            const SHRG* rule = ptr->attrs_ptr->grammar_ptr;

            // Only compute children if not already done
            if (ptr->child_visited_status != EMBase::VISITED) {
                // Add children based on nonterminal edges in the rule
                for (auto edge_ptr : rule->nonterminal_edges) {
                    ChartItem* child = generator->FindChartItemByEdge(ptr, edge_ptr);
                    ptr->children.push_back(child);
                    addChildren(child);  // Recursively process children
                }

                ptr->child_visited_status = VISITED;
                assert(ptr->children.size() == rule->nonterminal_edges.size());
            } else {
                // If already visited, still need to ensure children are processed
                for (auto child : ptr->children) {
                    addChildren(child);
                }
            }

            ptr = ptr->next_ptr;
        } while (ptr != start);
}

void EMBase::addParentPointerOptimized(ChartItem *root, int level) {
//        std::cout << "addParentPointerOptimized" << std::endl;
    if (!root) return;

    std::queue<std::pair<ChartItem*, int>> queue;
    queue.push({root, level});

    while (!queue.empty()) {
        auto [ptr1, level] = queue.front();
        queue.pop();
        auto ptr = ptr1;
        do{
                if (level > ptr->level) {
                    ptr->level = level;
                }

                const SHRG *rule = ptr->attrs_ptr->grammar_ptr;

                if (ptr->child_visited_status != EMBase::VISITED) {
                    ptr->children.reserve(rule->nonterminal_edges.size());

                    for (auto edge_ptr : rule->nonterminal_edges) {
                        ChartItem *child = generator->FindChartItemByEdge(ptr, edge_ptr);
                        ptr->children.push_back(child);
                        queue.push({child, ptr->level + 1});
                    }

                    size_t childCount = ptr->children.size();
                    std::vector<std::vector<ChartItem*>> siblings(childCount);

                    for (size_t i = 0; i < childCount; ++i) {
                        for (size_t j = 0; j < childCount; ++j) {
                            if (i != j) {
                                siblings[i].push_back(ptr->children[j]);
                            }
                        }
                    }

                    // Assign parent and sibling information
                    for (size_t i = 0; i < childCount; ++i) {
                        ptr->children[i]->parents_sib.push_back({ptr, std::move(siblings[i])});
                    }

                    ptr->child_visited_status = VISITED;
                } else {
                    for (auto child : ptr->children) {
                        queue.push({child, ptr->level + 1});
                    }
                }

                assert(ptr->children.size() == rule->nonterminal_edges.size());
                ptr = ptr->next_ptr;
        }while(ptr1 != ptr);
    }
}

double EMBase::computeInside(ChartItem *root){
    if(root->inside_visited_status == VISITED){
        return root->log_inside_prob;
    }

    ChartItem *ptr = root;
    double log_inside = ChartItem::log_zero;

    do{
        double curr_log_inside = ptr->rule_ptr->log_rule_weight;
//        assert(is_negative(curr_log_inside));
        curr_log_inside = sanitizeLogProb(curr_log_inside);

        double log_children = 0.0;
        for(ChartItem *child:ptr->children){
            log_children += computeInside(child);
        }
        log_children = sanitizeLogProb(log_children);

//        assert(is_negative(log_children));

        curr_log_inside += log_children;
        curr_log_inside = sanitizeLogProb(curr_log_inside);
//        assert(is_negative(curr_log_inside));

        log_inside = addLogs(log_inside, curr_log_inside);
        log_inside = sanitizeLogProb(log_inside);
        //        assert(is_negative(log_inside));
//        if(!is_negative(log_inside)){
//            std::cout << "?";
//        }

        ptr = ptr->next_ptr;
    }while(ptr != root);

    do{
        is_negative(log_inside);
        ptr->log_inside_prob = log_inside;
        ptr->inside_visited_status = VISITED;
        ptr = ptr->next_ptr;
    }while(ptr != root);

    return log_inside;
}

void EMBase::computeOutsideNode(ChartItem *root, NodeLevelPQ &pq){
    ChartItem *ptr = root;

    do{
        if(ptr->parents_sib.empty()){
            ptr = ptr->next_ptr;
            continue;
        }

        double log_outside = ChartItem::log_zero;

        for(ParentTup &parent_sib : ptr->parents_sib){
            ChartItem *parent = getParent(parent_sib);
            std::vector<ChartItem*> siblings = getSiblings(parent_sib);

            double curr_log_outside = parent->rule_ptr->log_rule_weight;
            assert(is_negative(curr_log_outside));
            curr_log_outside += parent->log_outside_prob;
            assert(is_negative(curr_log_outside));

            for(auto sib:siblings){
                curr_log_outside += sib->log_inside_prob;
                assert(is_negative(curr_log_outside));
            }

            log_outside = addLogs(log_outside, curr_log_outside);
            assert(is_negative(log_outside));
        }
        ptr->log_outside_prob = log_outside;
        ptr->outside_visited_status = VISITED;


        ptr = ptr->next_ptr;
    }while(ptr != root);

    double root_log_outside = root->log_outside_prob;

    do{
        if(ptr->outside_visited_status != VISITED){
            ptr->log_outside_prob = root_log_outside;
            ptr->outside_visited_status = VISITED;
        }

        for(ChartItem *child:ptr->children){
            pq.push(child);
        }

        ptr = ptr->next_ptr;
    }while(ptr != root);
}

void EMBase::computeOutside(ChartItem *root){
    NodeLevelPQ pq;
    ChartItem *ptr = root;

    do{
        ptr->log_outside_prob = 0.0;
        ptr = ptr->next_ptr;
    }while(ptr != root);

    pq.push(ptr);

    do{
        ChartItem *node = pq.top();
        computeOutsideNode(node, pq);
        pq.pop();
    }while(!pq.empty());
}


void EMBase::addRulePointer(ChartItem *root) {
    if (root->rule_visited == VISITED) {
        return;
    }

    ChartItem *ptr = root;
    do {
        // Use the grammar pointer directly instead of indexing by shrg_index.
        // This ensures consistent indexing with grammar objects (0 to hrg_size-1)
        // rather than shrg_indices (0 to shrg_size-1) which may be larger.
        ptr->rule_ptr = const_cast<SHRG*>(ptr->attrs_ptr->grammar_ptr);
        ptr->shrg_index = ptr->attrs_ptr->grammar_ptr->best_cfg_ptr->shrg_index;

        ptr->rule_visited = VISITED;
        for (ChartItem *child : ptr->children) {
            addRulePointer(child);
        }
        ptr = ptr->next_ptr;
    } while (ptr != root);
}

void EMBase::clearRuleCount(){
    for(auto rule:shrg_rules){
        rule->log_count = ChartItem::log_zero;
    }
}

}
}
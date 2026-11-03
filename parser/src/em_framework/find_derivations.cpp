//
// Created by Yuan Gao on 27/01/2025.
//
#include "find_derivations.hpp"
#include <random>
#include <unordered_map>
#include <set>

namespace shrg {
using namespace std;

// Thread-local random generator for thread safety
thread_local std::mt19937 g_rng(std::random_device{}());
float FindBestScoreWeight(ChartItem *root_ptr) {
    if (root_ptr->em_greedy_score == VISITED)
        return root_ptr->score;

    ChartItem *ptr = root_ptr;

    double max_weight = -std::numeric_limits<double>::infinity();
    ChartItem *max_subgraph_ptr = root_ptr;

    do {
        double current_weight = ptr->rule_ptr->log_rule_weight;
        for (auto child:ptr->children) {
            current_weight += FindBestScoreWeight(child);
        }


        if (max_weight < current_weight) {
            max_weight = current_weight;
            max_subgraph_ptr = ptr;
        }

        ptr = ptr->next_ptr;
    } while (ptr != root_ptr);

    if (max_subgraph_ptr != root_ptr) {
        root_ptr->Swap(*max_subgraph_ptr);
    }


    root_ptr->em_greedy_score = VISITED;
    root_ptr->score = max_weight;
    return root_ptr->score;
}



int addDerivationNode(Derivation &derivation, ChartItem *item, const SHRG::CFGRule *best_cfg) {
    DerivationNode node;
    node.grammar_ptr = item->rule_ptr;
    node.cfg_ptr = best_cfg ? best_cfg : (item->rule_ptr ? &item->rule_ptr->cfg_rules[0] : nullptr);
    node.item_ptr = item;

    int node_index = derivation.size();
    derivation.push_back(node);
    return node_index;
}

Derivation FindBestDerivation_EMGreedy(ChartItem *root_ptr) {
    Derivation derivation;
    if (!root_ptr || root_ptr->em_greedy_deriv == VISITED) return derivation;

    // Find best root alternative using rule weights
    ChartItem *best_root = root_ptr;
    double best_root_weight = root_ptr->rule_ptr ? root_ptr->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();

    // Mark all alternatives as visited
    ChartItem *curr_root = root_ptr;
    do {
        curr_root->em_greedy_deriv = VISITED;
        if (curr_root->rule_ptr && curr_root->rule_ptr->log_rule_weight > best_root_weight) {
            best_root_weight = curr_root->rule_ptr->log_rule_weight;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    int root_index = addDerivationNode(derivation, best_root, nullptr);

    for(auto child_ptr : best_root->children) {
        ChartItem *best_child = child_ptr;
        double best_weight = child_ptr->rule_ptr ? child_ptr->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();

        ChartItem *curr_ptr = child_ptr->next_ptr;
        while(curr_ptr != child_ptr) {
            if(curr_ptr->rule_ptr && curr_ptr->rule_ptr->log_rule_weight > best_weight) {
                best_weight = curr_ptr->rule_ptr->log_rule_weight;
                best_child = curr_ptr;
            }
            curr_ptr = curr_ptr->next_ptr;
        }

        Derivation child_derivation = FindBestDerivation_EMGreedy(best_child);
        if (!child_derivation.empty()) {
            derivation[root_index].children.push_back(derivation.size());
            derivation.insert(derivation.end(), child_derivation.begin(), child_derivation.end());
        }
    }
    return derivation;
}

template <typename T>
T sampleWeighted(const std::vector<std::pair<T, double>>& alternatives) {
    double total_weight = 0.0;
    for (const auto& alternative : alternatives) {
        total_weight += std::exp(alternative.second);
    }

    std::uniform_real_distribution<double> dis(0.0, total_weight);
    double random_value = dis(g_rng);
    double cumulative_weight = 0.0;
    for (const auto& alternative : alternatives) {
        cumulative_weight += std::exp(alternative.second);
        if (random_value <= cumulative_weight) {
            return alternative.first;
        }
    }

    return alternatives.back().first;
}


Derivation FindBestDerivation_sample(ChartItem *root_ptr) {
    Derivation derivation;
    if (!root_ptr || root_ptr->em_greedy_deriv == VISITED)
        return derivation;

    // Collect all root alternatives and their weights
    std::vector<std::pair<ChartItem*, double>> root_alternatives;
    ChartItem *curr_root = root_ptr;
    do {
        curr_root->em_greedy_deriv = VISITED;
        if (curr_root->rule_ptr) {
            root_alternatives.push_back(std::make_pair(curr_root, curr_root->rule_ptr->log_rule_weight));
        }
        curr_root = curr_root->next_ptr;
    } while (curr_root != root_ptr);

    // Sample a root alternative based on weights
    ChartItem *sampled_root = sampleWeighted(root_alternatives);

    int root_index = addDerivationNode(derivation, sampled_root, nullptr);
    for (auto child_ptr : sampled_root->children) {
        // Collect all child alternatives and their weights
        std::vector<std::pair<ChartItem*, double>> child_alternatives;
        ChartItem *curr_ptr = child_ptr;
        do {
            if (curr_ptr->rule_ptr) {
                child_alternatives.push_back(std::make_pair(curr_ptr, curr_ptr->rule_ptr->log_rule_weight));
            }
            curr_ptr = curr_ptr->next_ptr;
        } while (curr_ptr != child_ptr);

        // Sample a child alternative based on weights
        ChartItem *sampled_child = sampleWeighted(child_alternatives);

        Derivation child_derivation = FindBestDerivation_EMGreedy(sampled_child);
        if (!child_derivation.empty()) {
            derivation[root_index].children.push_back(derivation.size());
            derivation.insert(derivation.end(), child_derivation.begin(), child_derivation.end());
        }
    }

    return derivation;
}



Derivation FindBestDerivation_EMInside(ChartItem *root_ptr) {
    Derivation derivation;
    if (!root_ptr || root_ptr->em_inside_deriv == VISITED) return derivation;

    // Find best root alternative using inside scores
    ChartItem *best_root = root_ptr;
    double best_root_score = root_ptr->log_inside_prob;

    // Mark all alternatives as visited
    ChartItem *curr_root = root_ptr;
    do {
        curr_root->em_inside_deriv = VISITED;
        if(curr_root->log_inside_prob > best_root_score) {
            best_root_score = curr_root->log_inside_prob;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    int root_index = addDerivationNode(derivation, best_root, nullptr);

    for(auto child_ptr : best_root->children) {
        ChartItem *best_child = child_ptr;
        double best_score = child_ptr->log_inside_prob;

        ChartItem *curr_ptr = child_ptr->next_ptr;
        while(curr_ptr != child_ptr) {
            if(curr_ptr->log_inside_prob > best_score) {
                best_score = curr_ptr->log_inside_prob;
                best_child = curr_ptr;
            }
            curr_ptr = curr_ptr->next_ptr;
        }

        Derivation child_derivation = FindBestDerivation_EMInside(best_child);
        if (!child_derivation.empty()) {
            derivation[root_index].children.push_back(derivation.size());
            derivation.insert(derivation.end(), child_derivation.begin(), child_derivation.end());
        }
    }
    return derivation;
}

// Helper function for uniform random sampling
template<typename T>
T* uniformRandomSample(const std::vector<T*>& items) {
    if (items.empty()) return nullptr;
    std::uniform_int_distribution<size_t> dis(0, items.size() - 1);
    size_t idx = dis(g_rng);
    return items[idx];
}

DerivationInfo ExtractDerivation_uniform(ChartItem *root_ptr) {
    DerivationInfo result;
    if (!root_ptr || root_ptr->em_greedy_deriv == VISITED)
        return result;

    // Collect all root alternatives
    std::vector<ChartItem*> root_alternatives;
    ChartItem *curr_root = root_ptr;
    do {
        curr_root->em_greedy_deriv = VISITED;
        if (curr_root->rule_ptr) {
            root_alternatives.push_back(curr_root);
        }
        curr_root = curr_root->next_ptr;
    } while (curr_root != root_ptr);

    // Uniform random sampling of root
    ChartItem *sampled_root = uniformRandomSample(root_alternatives);
    if (!sampled_root) return result;

    result.rule_indices.push_back(sampled_root->shrg_index);
    result.edge_sets.push_back(sampled_root->edge_set);

    for (auto child_ptr : sampled_root->children) {
        // // Collect all child alternatives
        // std::vector<ChartItem*> child_alternatives;
        // ChartItem *curr_ptr = child_ptr;
        // do {
        //     if (curr_ptr->rule_ptr) {
        //         child_alternatives.push_back(curr_ptr);
        //     }
        //     curr_ptr = curr_ptr->next_ptr;
        // } while (curr_ptr != child_ptr);
        //
        // // Uniform random sampling of child
        // ChartItem *sampled_child = uniformRandomSample(child_alternatives);
        // if (!sampled_child) continue;

        auto child_info = ExtractDerivation_uniform(child_ptr);

        // Combine child information with current results
        result.rule_indices.insert(
            result.rule_indices.end(),
            child_info.rule_indices.begin(),
            child_info.rule_indices.end()
        );
        result.edge_sets.insert(
            result.edge_sets.end(),
            child_info.edge_sets.begin(),
            child_info.edge_sets.end()
        );
    }

    return result;
}

Derivation FindBestDerivation_CountGreedy(ChartItem *root_ptr) {
    Derivation derivation;
    if (!root_ptr || root_ptr->count_greedy_deriv == VISITED) return derivation;

    // Find best root alternative using counts
    ChartItem *best_root = root_ptr;
    double best_root_score = root_ptr->score;

    // Mark all alternatives as visited
    ChartItem *curr_root = root_ptr;
    do {
        curr_root->count_greedy_deriv = VISITED;
        if(curr_root->score > best_root_score) {
            best_root_score = curr_root->score;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    int root_index = addDerivationNode(derivation, best_root, nullptr);

    for(auto child_ptr : best_root->children) {
        ChartItem *best_child = child_ptr;
        double best_score = child_ptr->score;

        ChartItem *curr_ptr = child_ptr->next_ptr;
        while(curr_ptr != child_ptr) {
            if(curr_ptr->score > best_score) {
                best_score = curr_ptr->score;
                best_child = curr_ptr;
            }
            curr_ptr = curr_ptr->next_ptr;
        }

        Derivation child_derivation = FindBestDerivation_CountGreedy(best_child);
        if (!child_derivation.empty()) {
            derivation[root_index].children.push_back(derivation.size());
            derivation.insert(derivation.end(), child_derivation.begin(), child_derivation.end());
        }
    }
    return derivation;
}

Derivation FindBestDerivation_CountInside(ChartItem *root_ptr) {
    Derivation derivation;
    if (!root_ptr || root_ptr->count_inside_deriv == VISITED) return derivation;

    // Find best root alternative using inside counts
    ChartItem *best_root = root_ptr;
    double best_root_score = root_ptr->log_inside_count;

    // Mark all alternatives as visited
    ChartItem *curr_root = root_ptr;
    do {
        curr_root->count_inside_deriv = VISITED;
        if(curr_root->log_inside_count > best_root_score) {
            best_root_score = curr_root->log_inside_count;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    int root_index = addDerivationNode(derivation, best_root, nullptr);

    for(auto child_ptr : best_root->children) {
        ChartItem *best_child = child_ptr;
        double best_score = child_ptr->log_inside_count;

        ChartItem *curr_ptr = child_ptr->next_ptr;
        while(curr_ptr != child_ptr) {
            if(curr_ptr->log_inside_count > best_score) {
                best_score = curr_ptr->log_inside_count;
                best_child = curr_ptr;
            }
            curr_ptr = curr_ptr->next_ptr;
        }

        Derivation child_derivation = FindBestDerivation_CountInside(best_child);
        if (!child_derivation.empty()) {
            derivation[root_index].children.push_back(derivation.size());
            derivation.insert(derivation.end(), child_derivation.begin(), child_derivation.end());
        }
    }
    return derivation;
}

std::vector<int> ExtractRuleIndices_EMGreedy(ChartItem *root_ptr) {
    std::vector<int> indices;
    if (!root_ptr || root_ptr->em_greedy_deriv == VISITED) return indices;

    ChartItem *best_root = root_ptr;
    double best_root_weight = root_ptr->rule_ptr ? root_ptr->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();

    ChartItem *curr_root = root_ptr;
    do {
        curr_root->em_greedy_deriv = VISITED;
        if (curr_root->rule_ptr && curr_root->rule_ptr->log_rule_weight > best_root_weight) {
            best_root_weight = curr_root->rule_ptr->log_rule_weight;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    indices.push_back(best_root->shrg_index);

    for(auto child_ptr : best_root->children) {
        ChartItem *best_child = child_ptr;
        double best_weight = child_ptr->rule_ptr ? child_ptr->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();

        ChartItem *curr_ptr = child_ptr->next_ptr;
        while(curr_ptr != child_ptr) {
            if(curr_ptr->rule_ptr && curr_ptr->rule_ptr->log_rule_weight > best_weight) {
                best_weight = curr_ptr->rule_ptr->log_rule_weight;
                best_child = curr_ptr;
            }
            curr_ptr = curr_ptr->next_ptr;
        }

        auto child_indices = ExtractRuleIndices_EMGreedy(best_child);
        indices.insert(indices.end(), child_indices.begin(), child_indices.end());
    }
    return indices;
}


double computeInside_score(ChartItem *root){
    if(root->inside_visited_status == VISITED){
        return root->log_inside_prob;
    }

    ChartItem *ptr = root;
    double log_inside = ChartItem::log_zero;

    do{
        double curr_log_inside = ptr->score;
        //        assert(is_negative(curr_log_inside));
        curr_log_inside = sanitizeLogProb(curr_log_inside);

        double log_children = 0.0;
        for(ChartItem *child:ptr->children){
            log_children += computeInside_score(child);
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

std::vector<int> ExtractRuleIndices_sampled(ChartItem *root_ptr) {
    std::vector<int> indices;
    if (!root_ptr || root_ptr->em_greedy_deriv == VISITED)
        return indices;

    // Collect all root alternatives and their weights
    std::vector<std::pair<ChartItem*, double>> root_alternatives;
    ChartItem *curr_root = root_ptr;
    do {
        curr_root->em_greedy_deriv = VISITED;
        if (curr_root->rule_ptr) {
            root_alternatives.push_back(std::make_pair(curr_root, 0));
        }
        curr_root = curr_root->next_ptr;
    } while (curr_root != root_ptr);

    // Sample a root alternative based on weights
    ChartItem *sampled_root = sampleWeighted(root_alternatives);
    indices.push_back(sampled_root->shrg_index);

    for (auto child_ptr : sampled_root->children) {
        // Collect all child alternatives and their weights
        std::vector<std::pair<ChartItem*, double>> child_alternatives;
        ChartItem *curr_ptr = child_ptr;
        do {
            if (curr_ptr->rule_ptr) {
                child_alternatives.push_back(std::make_pair(curr_ptr, curr_ptr->rule_ptr->log_rule_weight));
            }
            curr_ptr = curr_ptr->next_ptr;
        } while (curr_ptr != child_ptr);

        // Sample a child alternative based on weights
        ChartItem *sampled_child = sampleWeighted(child_alternatives);

        auto child_indices = ExtractRuleIndices_EMGreedy(sampled_child);
        indices.insert(indices.end(), child_indices.begin(), child_indices.end());
    }

    return indices;
}

DerivationInfo ExtractDerivation_sampled(ChartItem *root_ptr) {
    DerivationInfo result;
    if (!root_ptr || root_ptr->em_greedy_deriv == VISITED)
        return result;

    // Collect all root alternatives and their weights
    std::vector<std::pair<ChartItem*, double>> root_alternatives;
    ChartItem *curr_root = root_ptr;
    do {
        curr_root->em_greedy_deriv = VISITED;
        if (curr_root->rule_ptr) {
            root_alternatives.push_back(std::make_pair(curr_root, curr_root->rule_ptr->log_rule_weight));
        }
        curr_root = curr_root->next_ptr;
    } while (curr_root != root_ptr);

    // Sample a root alternative based on weights
    ChartItem *sampled_root = sampleWeighted(root_alternatives);
    result.rule_indices.push_back(sampled_root->shrg_index);
    result.edge_sets.push_back(sampled_root->edge_set);

    for (auto child_ptr : sampled_root->children) {
        // Collect all child alternatives and their weights
        std::vector<std::pair<ChartItem*, double>> child_alternatives;
        ChartItem *curr_ptr = child_ptr;
        do {
            if (curr_ptr->rule_ptr) {
                child_alternatives.push_back(std::make_pair(curr_ptr, curr_ptr->rule_ptr->log_rule_weight));
            }
            curr_ptr = curr_ptr->next_ptr;
        } while (curr_ptr != child_ptr);

        // Sample a child alternative based on weights
        ChartItem *sampled_child = sampleWeighted(child_alternatives);

        auto child_info = ExtractDerivation_sampled(sampled_child);

        // Combine child information with current results
        result.rule_indices.insert(
            result.rule_indices.end(),
            child_info.rule_indices.begin(),
            child_info.rule_indices.end()
        );
        result.edge_sets.insert(
            result.edge_sets.end(),
            child_info.edge_sets.begin(),
            child_info.edge_sets.end()
        );
    }

    return result;
}


std::vector<int> ExtractRuleIndices_EMInside(ChartItem *root_ptr) {
    std::vector<int> indices;
    if (!root_ptr || root_ptr->em_inside_deriv == VISITED) return indices;

    ChartItem *best_root = root_ptr;
    double best_root_score = root_ptr->log_inside_prob;

    ChartItem *curr_root = root_ptr;
    do {
        curr_root->em_inside_deriv = VISITED;
        if(curr_root->log_inside_prob > best_root_score) {
            best_root_score = curr_root->log_inside_prob;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    indices.push_back(best_root->shrg_index);

    for(auto child_ptr : best_root->children) {
        ChartItem *best_child = child_ptr;
        double best_score = child_ptr->log_inside_prob;

        ChartItem *curr_ptr = child_ptr->next_ptr;
        while(curr_ptr != child_ptr) {
            if(curr_ptr->log_inside_prob > best_score) {
                best_score = curr_ptr->log_inside_prob;
                best_child = curr_ptr;
            }
            curr_ptr = curr_ptr->next_ptr;
        }

        auto child_indices = ExtractRuleIndices_EMInside(best_child);
        indices.insert(indices.end(), child_indices.begin(), child_indices.end());
    }
    return indices;
}

std::vector<int> ExtractRuleIndices_CountGreedy(ChartItem *root_ptr) {
    std::vector<int> indices;
    if (!root_ptr || root_ptr->count_greedy_deriv == VISITED) return indices;

    ChartItem *best_root = root_ptr;
    double best_root_score = root_ptr->score;

    ChartItem *curr_root = root_ptr;
    do {
        curr_root->count_greedy_deriv = VISITED;
        if(curr_root->score > best_root_score) {
            best_root_score = curr_root->score;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    indices.push_back(best_root->shrg_index);

    for(auto child_ptr : best_root->children) {
        ChartItem *best_child = child_ptr;
        double best_score = child_ptr->score;

        ChartItem *curr_ptr = child_ptr->next_ptr;
        while(curr_ptr != child_ptr) {
            if(curr_ptr->score > best_score) {
                best_score = curr_ptr->score;
                best_child = curr_ptr;
            }
            curr_ptr = curr_ptr->next_ptr;
        }

        auto child_indices = ExtractRuleIndices_CountGreedy(best_child);
        indices.insert(indices.end(), child_indices.begin(), child_indices.end());
    }
    return indices;
}

std::vector<int> ExtractRuleIndices_CountInside(ChartItem *root_ptr) {
    std::vector<int> indices;
    if (!root_ptr || root_ptr->count_inside_deriv == VISITED) return indices;

    ChartItem *best_root = root_ptr;
    double best_root_score = root_ptr->log_inside_count;

    ChartItem *curr_root = root_ptr;
    do {
        curr_root->count_inside_deriv = VISITED;
        if(curr_root->log_inside_count > best_root_score) {
            best_root_score = curr_root->log_inside_count;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    indices.push_back(best_root->shrg_index);

    for(auto child_ptr : best_root->children) {
        ChartItem *best_child = child_ptr;
        double best_score = child_ptr->log_inside_count;

        ChartItem *curr_ptr = child_ptr->next_ptr;
        while(curr_ptr != child_ptr) {
            if(curr_ptr->log_inside_count > best_score) {
                best_score = curr_ptr->log_inside_count;
                best_child = curr_ptr;
            }
            curr_ptr = curr_ptr->next_ptr;
        }

        auto child_indices = ExtractRuleIndices_CountInside(best_child);
        indices.insert(indices.end(), child_indices.begin(), child_indices.end());
    }
    return indices;
}


DerivationInfo ExtractRuleIndicesAndEdges_EMGreedy(ChartItem *root_ptr) {
    DerivationInfo result;
    if (!root_ptr || root_ptr->em_greedy_deriv == VISITED) return result;

    ChartItem *best_root = root_ptr;
    double best_root_weight = root_ptr->rule_ptr ? root_ptr->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();

    ChartItem *curr_root = root_ptr;
    do {
        curr_root->em_greedy_deriv = VISITED;
        if (curr_root->rule_ptr && curr_root->rule_ptr->log_rule_weight > best_root_weight) {
            best_root_weight = curr_root->rule_ptr->log_rule_weight;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    // Add both the rule index and edge set from the best root
    result.rule_indices.push_back(best_root->shrg_index);
    result.edge_sets.push_back(best_root->edge_set);

    for(auto child_ptr : best_root->children) {
        // ChartItem *best_child = child_ptr;
        // double best_weight = child_ptr->rule_ptr ? child_ptr->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();
        //
        // ChartItem *curr_ptr = child_ptr->next_ptr;
        // while(curr_ptr != child_ptr) {
        //     if(curr_ptr->rule_ptr && curr_ptr->rule_ptr->log_rule_weight > best_weight) {
        //         best_weight = curr_ptr->rule_ptr->log_rule_weight;
        //         best_child = curr_ptr;
        //     }
        //     curr_ptr = curr_ptr->next_ptr;
        // }

        // Recursively get information from children
        auto child_info = ExtractRuleIndicesAndEdges_EMGreedy(child_ptr);

        // Combine the child information with our current results
        result.rule_indices.insert(
            result.rule_indices.end(),
            child_info.rule_indices.begin(),
            child_info.rule_indices.end()
        );
        result.edge_sets.insert(
            result.edge_sets.end(),
            child_info.edge_sets.begin(),
            child_info.edge_sets.end()
        );
    }
    return result;
}
double get_rule_inside_em(ChartItem *root) {
    if (!root || !root->rule_ptr) {
        return -std::numeric_limits<double>::infinity();
    }
    double s = root->rule_ptr->log_rule_weight;
    for (auto child:root->children) {
        s += child->log_inside_prob;
    }
    return s;
}
DerivationInfo ExtractRuleIndicesAndEdges_EMInside(ChartItem *root_ptr) {
    DerivationInfo result;
    if (!root_ptr || root_ptr->em_greedy_deriv == VISITED) return result;

    ChartItem *best_root = root_ptr;
    double best_root_weight = get_rule_inside_em(root_ptr);

    ChartItem *curr_root = root_ptr;
    do {
        curr_root->em_greedy_deriv = VISITED;
        double curr_weight = get_rule_inside_em(curr_root);
        if (curr_root->rule_ptr && curr_weight > best_root_weight) {
            best_root_weight = curr_weight;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    // Add both the rule index and edge set from the best root
    result.rule_indices.push_back(best_root->shrg_index);
    result.edge_sets.push_back(best_root->edge_set);

    for(auto child_ptr : best_root->children) {
        // ChartItem *best_child = child_ptr;
        // double best_weight = child_ptr->rule_ptr ? child_ptr->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();
        //
        // ChartItem *curr_ptr = child_ptr->next_ptr;
        // while(curr_ptr != child_ptr) {
        //     if(curr_ptr->rule_ptr && curr_ptr->rule_ptr->log_rule_weight > best_weight) {
        //         best_weight = curr_ptr->rule_ptr->log_rule_weight;
        //         best_child = curr_ptr;
        //     }
        //     curr_ptr = curr_ptr->next_ptr;
        // }

        // Recursively get information from children
        auto child_info = ExtractRuleIndicesAndEdges_EMInside(child_ptr);

        // Combine the child information with our current results
        result.rule_indices.insert(
            result.rule_indices.end(),
            child_info.rule_indices.begin(),
            child_info.rule_indices.end()
        );
        result.edge_sets.insert(
            result.edge_sets.end(),
            child_info.edge_sets.begin(),
            child_info.edge_sets.end()
        );
    }
    return result;
}

ChartItem* FindNodeWithIndex(ChartItem* start_ptr, int target_index) {
    if (!start_ptr) return nullptr;

    ChartItem* curr_ptr = start_ptr;
    do {
        if (curr_ptr->shrg_index == target_index) {
            return curr_ptr;
        }
        for(auto rule :curr_ptr->rule_ptr->cfg_rules){
            if(rule.shrg_index == target_index){
                return curr_ptr;
            }
        }

        curr_ptr = curr_ptr->next_ptr;
    } while (curr_ptr != start_ptr);

    return nullptr;
}

// Helper function to check if an index exists in remaining indices
bool ContainsIndex(const std::vector<int>& indices, int target) {
    return std::find(indices.begin(), indices.end(), target) != indices.end();
}

// Helper function to remove an index from a vector (only first occurrence)
std::vector<int> RemoveIndex(const std::vector<int>& indices, int target) {
    std::vector<int> result;
    bool removed = false;
    for (int idx : indices) {
        if (idx == target && !removed) {
            removed = true;  // Skip only first occurrence
        } else {
            result.push_back(idx);
        }
    }
    return result;
}

// std::optional<DerivationInfo> TryExtractGoldDerivation(
//     ChartItem* root_ptr,
//     const std::vector<int>& remaining_indices) {
//
//     if (remaining_indices.empty()) {
//         return DerivationInfo();
//     }
//
//     // Try each remaining index as the potential root
//     for (int current_index : remaining_indices) {
//         ChartItem* current_node = FindNodeWithIndex(root_ptr, current_index);
//         if (!current_node) continue;
//
//         // Remove current index from remaining indices
//         auto new_remaining = RemoveIndex(remaining_indices, current_index);
//
//         // If no more indices to process and this is a leaf node
//         if (new_remaining.empty() && current_node->children.empty()) {
//             DerivationInfo result;
//             result.rule_indices.push_back(current_index);
//             result.edge_sets.push_back(current_node->edge_set);
//             return result;
//         }
//
//         // If this node has children, we need to distribute remaining indices among them
//         std::vector<std::optional<DerivationInfo>> child_results;
//         bool valid_distribution = true;
//
//         // Try to find valid derivations for each child
//         for (auto child_ptr : current_node->children) {
//             // Find indices that could belong to this child's subtree
//             std::vector<int> possible_child_indices;
//             for (int idx : new_remaining) {
//                 ChartItem* temp = child_ptr;
//                 do {
//                     if (temp->shrg_index == idx) {
//                         possible_child_indices.push_back(idx);
//                         break;
//                     }
//                     temp = temp->next_ptr;
//                 } while (temp != child_ptr);
//             }
//
//             auto child_result = TryExtractGoldDerivation(child_ptr, possible_child_indices);
//             if (!child_result) {
//                 valid_distribution = false;
//                 break;
//             }
//             child_results.push_back(child_result);
//         }
//
//         // If we found valid derivations for all children
//         if (valid_distribution && !child_results.empty()) {
//             DerivationInfo result;
//             result.rule_indices.push_back(current_index);
//             result.edge_sets.push_back(current_node->edge_set);
//
//             // Combine all child results
//             for (const auto& child_result : child_results) {
//                 const auto& child_info = child_result.value();
//                 result.rule_indices.insert(
//                     result.rule_indices.end(),
//                     child_info.rule_indices.begin(),
//                     child_info.rule_indices.end()
//                 );
//                 result.edge_sets.insert(
//                     result.edge_sets.end(),
//                     child_info.edge_sets.begin(),
//                     child_info.edge_sets.end()
//                 );
//             }
//             return result;
//         }
//     }
//
//     return std::nullopt;
// }
//
// std::optional<DerivationInfo> ExtractGoldDerivationTree(
//     ChartItem* root_ptr,
//     const std::vector<int>& gold_indices) {
//
//     return TryExtractGoldDerivation(root_ptr, gold_indices);
// }

// bool IndexExistsInSubtree(ChartItem* root_ptr, int target_index) {
//     if (!root_ptr) return false;
//
//     // Check in current linked list
//     ChartItem* curr_ptr = root_ptr;
//     do {
//         if (curr_ptr->shrg_index == target_index) {
//             return true;
//         }
//         for(auto rule :curr_ptr->rule_ptr->cfg_rules){
//             if(rule.shrg_index == target_index){
//                 return true;
//             }
//         }
//         curr_ptr = curr_ptr->next_ptr;
//     } while (curr_ptr != root_ptr);
//
//     // Check in all children recursively
//     for (auto child_ptr : root_ptr->children) {
//         if (IndexExistsInSubtree(child_ptr, target_index)) {
//             return true;
//         }
//     }
//
//     return false;
// }

std::vector<int> RemoveIndices(const std::vector<int>& indices, const std::vector<int>& to_remove) {
    // Track count of each index to remove
    std::unordered_map<int, int> remove_counts;
    for (int idx : to_remove) {
        remove_counts[idx]++;
    }
    std::vector<int> result;
    for (int idx : indices) {
        if (remove_counts[idx] > 0) {
            remove_counts[idx]--;  // Decrement count, skip this occurrence
        } else {
            result.push_back(idx);
        }
    }
    return result;
}

std::vector<std::vector<int>> GetPossibleDistributions(
    const std::vector<ChartItem*>& children,
    const std::vector<int>& indices) {

    std::vector<std::vector<int>> distributions(children.size());

    // For each index, add it to the distribution of any child that contains it
    for (int idx : indices) {
        for (size_t i = 0; i < children.size(); ++i) {
            if (IndexExistsInSubtree(children[i], idx)) {
                distributions[i].push_back(idx);
            }
        }
    }

    return distributions;
}

std::optional<DerivationInfo> TryExtractGoldDerivation(
    ChartItem* root_ptr,
    const std::vector<int>& remaining_indices) {

    if (remaining_indices.empty()) {
        return DerivationInfo();
    }

    // Try each remaining index as the potential root
    for (int current_index : remaining_indices) {
        // Look for this index in the current linked list
        ChartItem* current_node = FindNodeWithIndex(root_ptr, current_index);
        if (!current_node) continue;

        // Remove current index from remaining indices
        auto new_remaining = RemoveIndex(remaining_indices, current_index);

        // If this is a leaf node, return just this node's info
        if (current_node->children.empty()) {
            DerivationInfo result;
            result.rule_indices.push_back(current_index);
            result.edge_sets.push_back(current_node->edge_set);
            return result;
        }

        // Process children one at a time
        DerivationInfo result;
        result.rule_indices.push_back(current_index);
        result.edge_sets.push_back(current_node->edge_set);

        auto remaining_for_children = new_remaining;
        bool all_children_valid = true;

        // Process each child iteratively
        for (auto child_ptr : current_node->children) {
            auto child_result = TryExtractGoldDerivation(child_ptr, remaining_for_children);
            if (!child_result) {
                all_children_valid = false;
                break;
            }

            // Add child's results to current results
            const auto& child_info = *child_result;
            result.rule_indices.insert(
                result.rule_indices.end(),
                child_info.rule_indices.begin(),
                child_info.rule_indices.end()
            );
            result.edge_sets.insert(
                result.edge_sets.end(),
                child_info.edge_sets.begin(),
                child_info.edge_sets.end()
            );

            // Update remaining indices for next child
            remaining_for_children = RemoveIndices(remaining_for_children, child_info.rule_indices);
        }

        if (all_children_valid) {
            return result;
        }
    }

    return std::nullopt;
}

// Cleaner gold derivation extraction using multiset with backtracking
std::optional<DerivationInfo> ExtractGoldDerivation(
    ChartItem* node,
    std::multiset<int>& gold_indices)
{
    if (!node) return std::nullopt;

    // Try each alternative at this position (next_ptr chain)
    ChartItem* ptr = node;
    do {
        int rule_idx = ptr->shrg_index;

        // Is this rule in our gold set?
        auto it = gold_indices.find(rule_idx);
        if (it != gold_indices.end()) {
            // Save full state before making any modifications for proper backtracking
            auto saved_gold = gold_indices;

            // Remove one occurrence
            gold_indices.erase(it);

            DerivationInfo result;
            result.rule_indices.push_back(rule_idx);
            result.edge_sets.push_back(ptr->edge_set);

            // Match all children
            bool valid = true;
            for (auto* child : ptr->children) {
                auto child_result = ExtractGoldDerivation(child, gold_indices);
                if (!child_result) {
                    valid = false;
                    break;
                }
                result.rule_indices.insert(result.rule_indices.end(),
                    child_result->rule_indices.begin(),
                    child_result->rule_indices.end());
                result.edge_sets.insert(result.edge_sets.end(),
                    child_result->edge_sets.begin(),
                    child_result->edge_sets.end());
            }

            if (valid) return result;

            // Backtrack: restore ALL consumed indices (including this node and children)
            gold_indices = saved_gold;
        }

        ptr = ptr->next_ptr;
    } while (ptr && ptr != node);

    return std::nullopt;
}

// Wrapper that takes a vector and converts to multiset
std::optional<DerivationInfo> ExtractGoldDerivation(
    ChartItem* root,
    const std::vector<int>& gold_indices_vec)
{
    std::multiset<int> gold(gold_indices_vec.begin(), gold_indices_vec.end());
    return ExtractGoldDerivation(root, gold);
}

// std::optional<DerivationInfo> TryExtractGoldDerivation(
//     ChartItem* root_ptr,
//     const std::vector<int>& remaining_indices) {
//
//     if (remaining_indices.empty()) {
//         return DerivationInfo();
//     }
//
//     // Try each remaining index as the potential root
//     for (int current_index : remaining_indices) {
//         // Look for this index in the current linked list
//         ChartItem* current_node = FindNodeWithIndex(root_ptr, current_index);
//         if (!current_node) continue;
//
//         // Remove current index from remaining indices
//         auto new_remaining = RemoveIndex(remaining_indices, current_index);
//
//         // If no more indices to process and this is a leaf node
//         if (current_node->children.empty()) {
//             DerivationInfo result;
//             result.rule_indices.push_back(current_index);
//             result.edge_sets.push_back(current_node->edge_set);
//             return result;
//         }
//
//         // Get possible distributions of remaining indices among children
//         auto distributions = GetPossibleDistributions(current_node->children, new_remaining);
//         if (distributions.empty()) continue;  // No valid distribution found
//
//         // Try to find valid derivations for each child with its indices
//         std::vector<std::optional<DerivationInfo>> child_results;
//         bool valid_distribution = true;
//
//         for (size_t i = 0; i < current_node->children.size(); ++i) {
//             auto child_result = TryExtractGoldDerivation(
//                 current_node->children[i],
//                 distributions[i]
//             );
//
//             if (!child_result) {
//                 valid_distribution = false;
//                 break;
//             }
//             child_results.push_back(child_result);
//         }
//
//         // If we found valid derivations for all children
//         if (valid_distribution && !child_results.empty()) {
//             DerivationInfo result;
//             result.rule_indices.push_back(current_index);
//             result.edge_sets.push_back(current_node->edge_set);
//
//             // Combine all child results
//             for (const auto& child_result : child_results) {
//                 const auto& child_info = child_result.value();
//                 result.rule_indices.insert(
//                     result.rule_indices.end(),
//                     child_info.rule_indices.begin(),
//                     child_info.rule_indices.end()
//                 );
//                 result.edge_sets.insert(
//                     result.edge_sets.end(),
//                     child_info.edge_sets.begin(),
//                     child_info.edge_sets.end()
//                 );
//             }
//             return result;
//         }
//     }
//
//     return std::nullopt;
// }

std::optional<DerivationInfo> ExtractGoldDerivationTree(
    ChartItem* root_ptr,
    const std::vector<int>& gold_indices) {

    return TryExtractGoldDerivation(root_ptr, gold_indices);
}


double get_rule_inside(ChartItem *root) {
    double s = root->score;
    for (auto child:root->children) {
        s += child->log_inside_prob;
    }
    return s;
}

DerivationInfo ExtractRuleIndicesAndEdges_CountGreedy(ChartItem *root_ptr) {
    DerivationInfo result;
    if (!root_ptr || root_ptr->em_greedy_deriv == VISITED) return result;

    ChartItem *best_root = root_ptr;
    double best_root_score = get_rule_inside(root_ptr);

    ChartItem *curr_root = root_ptr;
    do {
        curr_root->count_greedy_deriv = VISITED;
        double curr_inside = get_rule_inside(curr_root);
        if(curr_inside > best_root_score) {
            best_root_score = curr_inside;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    result.rule_indices.push_back(best_root->shrg_index);
    result.edge_sets.push_back(best_root->edge_set);

    for(auto child_ptr : best_root->children) {

        auto child_info = ExtractRuleIndicesAndEdges_EMGreedy(child_ptr);

        result.rule_indices.insert(
            result.rule_indices.end(),
            child_info.rule_indices.begin(),
            child_info.rule_indices.end()
        );
        result.edge_sets.insert(
            result.edge_sets.end(),
            child_info.edge_sets.begin(),
            child_info.edge_sets.end()
        );
    }
    return result;
}

DerivationInfo ExtractRuleIndicesAndEdges_ScoreGreedy(ChartItem *root_ptr) {
    DerivationInfo result;
    if (!root_ptr || root_ptr->em_greedy_deriv == VISITED) return result;

    ChartItem *best_root = root_ptr;
    double best_root_score = root_ptr->score;

    ChartItem *curr_root = root_ptr;
    do {
        curr_root->count_greedy_deriv = VISITED;
        double curr_inside = curr_root->score;
        if(curr_inside > best_root_score) {
            best_root_score = curr_inside;
            best_root = curr_root;
        }
        curr_root = curr_root->next_ptr;
    } while(curr_root != root_ptr);

    result.rule_indices.push_back(best_root->shrg_index);
    result.edge_sets.push_back(best_root->edge_set);

    for(auto child_ptr : best_root->children) {

        auto child_info = ExtractRuleIndicesAndEdges_ScoreGreedy(child_ptr);

        result.rule_indices.insert(
            result.rule_indices.end(),
            child_info.rule_indices.begin(),
            child_info.rule_indices.end()
        );
        result.edge_sets.insert(
            result.edge_sets.end(),
            child_info.edge_sets.begin(),
            child_info.edge_sets.end()
        );
    }
    return result;
}

bool IndexExistsInSubtree(ChartItem* root_ptr, int target_index) {
    if (!root_ptr) return false;

    // Check in current linked list
    ChartItem* curr_ptr = root_ptr;
    do {
        if (curr_ptr->shrg_index == target_index) {
            return true;
        }
        for(auto rule :curr_ptr->rule_ptr->cfg_rules){
            if(rule.shrg_index == target_index){
                return true;
            }
        }
        curr_ptr = curr_ptr->next_ptr;
    } while (curr_ptr != root_ptr);

    // Check in all children recursively
    for (auto child_ptr : root_ptr->children) {
        if (IndexExistsInSubtree(child_ptr, target_index)) {
            return true;
        }
    }

    return false;
}

// Keep track of how many times each index appears in the gold indices
// std::map<int, int> GetIndexCounts(const std::vector<int>& indices) {
//     std::map<int, int> counts;
//     for (int idx : indices) {
//         counts[idx]++;
//     }
//     return counts;
// }

// std::optional<DerivationInfo> TryExtractGoldDerivation(
//     ChartItem* root_ptr,
//     const std::vector<int>& remaining_indices,
//     std::map<int, int>& remaining_counts) {  // Track remaining counts for each index
//
//     if (remaining_indices.empty()) {
//         return DerivationInfo();
//     }
//
//     // Try each remaining index as the potential root
//     for (int current_index : remaining_indices) {
//         // Skip if we've used all instances of this index
//         if (remaining_counts[current_index] <= 0) continue;
//
//         // Look for this index in the current linked list
//         ChartItem* current_node = FindNodeWithIndex(root_ptr, current_index);
//         if (!current_node) continue;
//
//         // Temporarily decrease the count for this index
//         remaining_counts[current_index]--;
//
//         // Create new remaining indices vector, keeping duplicate indices if their count > 0
//         std::vector<int> new_remaining;
//         for (int idx : remaining_indices) {
//             if (idx != current_index || remaining_counts[idx] > 0) {
//                 new_remaining.push_back(idx);
//             }
//         }
//
//         // If no more indices to process and this is a leaf node
//         if (new_remaining.empty() && current_node->children.empty()) {
//             DerivationInfo result;
//             result.rule_indices.push_back(current_index);
//             result.edge_sets.push_back(current_node->edge_set);
//             return result;
//         }
//
//         // Get possible distributions of remaining indices among children
//         auto distributions = GetPossibleDistributions(current_node->children, new_remaining);
//
//         // Try each possible distribution of the remaining indices
//         for (size_t attempt = 0; attempt < (1 << new_remaining.size()); ++attempt) {
//             // Create temporary counts for this attempt
//             auto temp_counts = remaining_counts;
//             std::vector<std::vector<int>> child_indices(current_node->children.size());
//
//             // Distribute indices to children based on the current attempt
//             for (size_t i = 0; i < new_remaining.size(); ++i) {
//                 if (attempt & (1 << i)) {
//                     // Try assigning to first valid child
//                     for (size_t j = 0; j < current_node->children.size(); ++j) {
//                         if (IndexExistsInSubtree(current_node->children[j], new_remaining[i])) {
//                             child_indices[j].push_back(new_remaining[i]);
//                             break;
//                         }
//                     }
//                 } else {
//                     // Try assigning to last valid child
//                     for (int j = current_node->children.size() - 1; j >= 0; --j) {
//                         if (IndexExistsInSubtree(current_node->children[j], new_remaining[i])) {
//                             child_indices[j].push_back(new_remaining[i]);
//                             break;
//                         }
//                     }
//                 }
//             }
//
//             // Try to find valid derivations for each child with its indices
//             std::vector<std::optional<DerivationInfo>> child_results;
//             bool valid_distribution = true;
//
//             for (size_t i = 0; i < current_node->children.size(); ++i) {
//                 if (!child_indices[i].empty()) {
//                     auto child_result = TryExtractGoldDerivation(
//                         current_node->children[i],
//                         child_indices[i],
//                         temp_counts
//                     );
//
//                     if (!child_result) {
//                         valid_distribution = false;
//                         break;
//                     }
//                     child_results.push_back(child_result);
//                 }
//             }
//
//             // If we found valid derivations for all children
//             if (valid_distribution) {
//                 DerivationInfo result;
//                 result.rule_indices.push_back(current_index);
//                 result.edge_sets.push_back(current_node->edge_set);
//
//                 // Combine all child results
//                 for (const auto& child_result : child_results) {
//                     const auto& child_info = child_result.value();
//                     result.rule_indices.insert(
//                         result.rule_indices.end(),
//                         child_info.rule_indices.begin(),
//                         child_info.rule_indices.end()
//                     );
//                     result.edge_sets.insert(
//                         result.edge_sets.end(),
//                         child_info.edge_sets.begin(),
//                         child_info.edge_sets.end()
//                     );
//                 }
//
//                 // This distribution worked
//                 remaining_counts = temp_counts;  // Update the counts
//                 return result;
//             }
//         }
//
//         // If we get here, no distribution worked with this root
//         // Restore the count for backtracking
//         remaining_counts[current_index]++;
//     }
//
//     return std::nullopt;
// }

// Wrapper function
// std::optional<DerivationInfo> ExtractGoldDerivationTree(
//     ChartItem* root_ptr,
//     const std::vector<int>& gold_indices) {
//
//     auto remaining_counts = GetIndexCounts(gold_indices);
//     return TryExtractGoldDerivation(root_ptr, gold_indices, remaining_counts);
// }
//
// Derivation &FindBestDerivation_EMGreedy(shrg::ChartItem *root) {
//     Derivation derivation;
//     if (!root) return derivation;
//
//     // Find best root alternative
//     ChartItem *best_root = root;
//     double best_root_weight = root->rule_ptr ? root->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();
//
//     ChartItem *curr_root = root->next_ptr;
//     while(curr_root != root) {
//         double current_weight = curr_root->rule_ptr ? curr_root->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();
//         if(current_weight > best_root_weight) {
//             best_root_weight = current_weight;
//             best_root = curr_root;
//         }
//         curr_root = curr_root->next_ptr;
//     }
//
//     int root_index = addDerivationNode(derivation, best_root, nullptr);
//
//     for(auto child_ptr : best_root->children) {
//         ChartItem *best_child = child_ptr;
//         double best_weight = child_ptr->rule_ptr ? child_ptr->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();
//         const SHRG::CFGRule* best_cfg = nullptr;
//
//         ChartItem *curr_ptr = child_ptr->next_ptr;
//         while(curr_ptr != child_ptr) {
//             double current_weight = curr_ptr->rule_ptr ? curr_ptr->rule_ptr->log_rule_weight : -std::numeric_limits<double>::infinity();
//             if(current_weight > best_weight) {
//                 best_weight = current_weight;
//                 best_child = curr_ptr;
//                 best_cfg = curr_ptr->rule_ptr ? &curr_ptr->rule_ptr->cfg_rules[0] : nullptr;
//             }
//             curr_ptr = curr_ptr->next_ptr;
//         }
//
//         Derivation child_derivation =  FindBestDerivation_EMGreedy(best_child);
//         if (!child_derivation.empty()) {
//             derivation[root_index].children.push_back(derivation.size());
//             derivation.insert(derivation.end(), child_derivation.begin(), child_derivation.end());
//         }
//     }
//     return derivation;
// }
//
// Derivation &FindBestDerivation_EMInside(ChartItem *root_ptr) {
//     Derivation derivation;
//     if (!root_ptr) return derivation;
//
//     // Find best root alternative using inside scores
//     ChartItem *best_root = root_ptr;
//     double best_root_score = root_ptr->log_inside_prob;
//
//     ChartItem *curr_root = root_ptr->next_ptr;
//     while(curr_root != root_ptr) {
//         if(curr_root->log_inside_prob > best_root_score) {
//             best_root_score = curr_root->log_inside_prob;
//             best_root = curr_root;
//         }
//         curr_root = curr_root->next_ptr;
//     }
//
//     int root_index = addDerivationNode(derivation, best_root, nullptr);
//
//     for(auto child_ptr : best_root->children) {
//         ChartItem *best_child = child_ptr;
//         double best_score = child_ptr->log_inside_prob;
//
//         ChartItem *curr_ptr = child_ptr->next_ptr;
//         while(curr_ptr != child_ptr) {
//             if(curr_ptr->log_inside_prob > best_score) {
//                 best_score = curr_ptr->log_inside_prob;
//                 best_child = curr_ptr;
//             }
//             curr_ptr = curr_ptr->next_ptr;
//         }
//
//         Derivation child_derivation = FindBestDerivation_EMInside(best_child);
//         if (!child_derivation.empty()) {
//             derivation[root_index].children.push_back(derivation.size());
//             derivation.insert(derivation.end(), child_derivation.begin(), child_derivation.end());
//         }
//     }
//     return derivation;
// }
//
// Derivation &FindBestDerivation_CountGreedy(ChartItem *root_ptr) {
//     Derivation derivation;
//     if (!root_ptr) return derivation;
//
//     // Find best root alternative using count-based scores
//     ChartItem *best_root = root_ptr;
//     double best_root_score = root_ptr->score;
//
//     ChartItem *curr_root = root_ptr->next_ptr;
//     while(curr_root != root_ptr) {
//         if(curr_root->score > best_root_score) {
//             best_root_score = curr_root->score;
//             best_root = curr_root;
//         }
//         curr_root = curr_root->next_ptr;
//     }
//
//     int root_index = addDerivationNode(derivation, best_root, nullptr);
//
//     for(auto child_ptr : best_root->children) {
//         ChartItem *best_child = child_ptr;
//         double best_score = child_ptr->score;
//
//         ChartItem *curr_ptr = child_ptr->next_ptr;
//         while(curr_ptr != child_ptr) {
//             if(curr_ptr->score > best_score) {
//                 best_score = curr_ptr->score;
//                 best_child = curr_ptr;
//             }
//             curr_ptr = curr_ptr->next_ptr;
//         }
//
//         Derivation child_derivation = FindBestDerivation_CountGreedy(best_child);
//         if (!child_derivation.empty()) {
//             derivation[root_index].children.push_back(derivation.size());
//             derivation.insert(derivation.end(), child_derivation.begin(), child_derivation.end());
//         }
//     }
//     return derivation;
// }
//
//
// Derivation &FindBestDerivation_CountInside(ChartItem *root_ptr) {
//     Derivation derivation;
//     if (!root_ptr) return derivation;
//
//     // Find best root alternative using computed inside scores
//     ChartItem *best_root = root_ptr;
//     double best_root_score = ComputeInsideCount(root_ptr);
//
//     ChartItem *curr_root = root_ptr->next_ptr;
//     while(curr_root != root_ptr) {
//         double curr_score = ComputeInsideCount(curr_root);
//         if(curr_score > best_root_score) {
//             best_root_score = curr_score;
//             best_root = curr_root;
//         }
//         curr_root = curr_root->next_ptr;
//     }
//
//     int root_index = addDerivationNode(derivation, best_root, nullptr);
//
//     for(auto child_ptr : best_root->children) {
//         ChartItem *best_child = child_ptr;
//         double best_score = ComputeInsideCount(child_ptr);
//
//         ChartItem *curr_ptr = child_ptr->next_ptr;
//         while(curr_ptr != child_ptr) {
//             double curr_score = ComputeInsideCount(curr_ptr);
//             if(curr_score > best_score) {
//                 best_score = curr_score;
//                 best_child = curr_ptr;
//             }
//             curr_ptr = curr_ptr->next_ptr;
//         }
//
//         Derivation child_derivation = FindBestDerivation_CountInside(best_child);
//         if (!child_derivation.empty()) {
//             derivation[root_index].children.push_back(derivation.size());
//             derivation.insert(derivation.end(), child_derivation.begin(), child_derivation.end());
//         }
//     }
//     return derivation;
// }
}
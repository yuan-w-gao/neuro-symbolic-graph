#include "em_viterbi.hpp"
#include <fstream>
#include <vector>
#include <future>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ctime>

namespace shrg::em {

ViterbiEM::ViterbiEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs,
                     Context *context, double threshold)
    : EMBase(shrg_rules, graphs, context, threshold) {
    prev_ll = 0.0;
    rule_dict = getRuleDict();
}
ViterbiEM::ViterbiEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold, std::string dir, int timeout_seconds):EMBase(shrg_rules, graphs, context, threshold) {
    prev_ll = 0.0;
    rule_dict = getRuleDict();
    output_dir = std::move(dir);
    time_out_in_seconds = timeout_seconds;
}
bool ViterbiEM::validateProbabilities() {
    // Check rule probabilities sum to 1 for each label
    for(auto& pair : rule_dict) {
        double sum = 0.0;
        for(auto rule : pair.second) {
            double prob = std::exp(rule->log_rule_weight);
            if(prob < 0.0 || prob > 1.0) {
                std::cerr << "Invalid probability: " << prob << std::endl;
                return false;
            }
            sum += prob;
        }
        // Allow for small numerical errors
        if(std::abs(sum - 1.0) > 1e-6) {
            std::cerr << "Probabilities don't sum to 1: " << sum << std::endl;
            return false;
        }
    }
    return true;
}
ChartItem* ViterbiEM::findBestChild(ChartItem* parent, const SHRG* grammar, int edge_index) {
    ChartItem* child = generator->FindChartItemByEdge(parent, grammar->nonterminal_edges[edge_index]);
    if (!child) return nullptr;

    // Find alternative with highest score
    ChartItem* best = child;
    float best_score = child->score;

    ChartItem* curr = child->next_ptr;
    do {
        if (curr->score > best_score) {
            best = curr;
            best_score = curr->score;
        }
        curr = curr->next_ptr;
    }while (curr != child);

    return best;
}

void ViterbiEM::buildBestParseRelationships(ChartItem* root, int level) {
    if (!root) return;

    std::queue<std::pair<ChartItem*, int>> queue;
    queue.push({root, level});

    while (!queue.empty()) {
        auto [current, current_level] = queue.front();
        queue.pop();

        if (current_level > current->level) {
            current->level = current_level;
        }

        if (current->child_visited_status != EMBase::VISITED) {
            const SHRG* grammar = current->attrs_ptr->grammar_ptr;
            size_t child_count = grammar->nonterminal_edges.size();
            current->children.reserve(child_count);

            // Find best child for each edge
            for (size_t i = 0; i < child_count; ++i) {
                ChartItem* best_child = findBestChild(current, grammar, i);
                if (best_child) {
                    current->children.push_back(best_child);
                    queue.push({best_child, current_level + 1});
                }
            }

            // Create sibling vectors for best parse only
            for (size_t i = 0; i < current->children.size(); ++i) {
                std::vector<ChartItem*> siblings;
                for (size_t j = 0; j < current->children.size(); ++j) {
                    if (i != j) {
                        siblings.push_back(current->children[j]);
                    }
                }
                current->children[i]->parents_sib.clear();  // Clear any existing relationships
                current->children[i]->parents_sib.push_back({current, std::move(siblings)});
            }

            current->child_visited_status = EMBase::VISITED;
        } else {
            for (ChartItem* child : current->children) {
                queue.push({child, current_level + 1});
            }
        }
    }
}

void ViterbiEM::run() {
    std::cout << "Running Viterbi EM Training...\n";
    clock_t t1, t2;
    unsigned long training_size = graphs.size();
    int iteration = 0;
    ll = 0;
    setInitialWeights(rule_dict);
    std::vector<double> history[shrg_rules.size()];
    std::vector<double> history_graph_ll[training_size];
    for(int i = 0; i < shrg_rules.size(); i++){
        history[i].push_back(shrg_rules[i]->log_rule_weight);
    }

    std::vector<double> times;

    std::vector<double> lls;

    do {
        prev_ll = ll;
        ll = 0;
        t1 = clock();

        // Original loop - commented out
        for(int i = 0; i < training_size; i++) {
            // std::cout << i << std::endl;
            if(i % 200 == 0) {
                std::cout << "Processing graph " << i << "\n";
            }

            EdsGraph graph = graphs[i];
            auto code = context->Parse(graph);

            if(code == ParserError::kNone) {
                ChartItem* root = context->parser->Result();

                // std::cout << "buildBestParse"<< std::endl;
                buildBestParseRelationships(root, 0);  // Build relationships only for best parse
                // std::cout << "adding rule pointer"<< std::endl;
                addRulePointer(root);
                // if (iteration == 1 && i == 3) {
                    // std::cout << "stop here" << std::endl;
                // }

                // std::cout << "inside"<< std::endl;
                double pw = computeViterbiInside(root);
                // std::cout << "outside"<< std::endl;
                computeViterbiOutside(root);
                // std::cout << "count" << std::endl;
                computeExpectedCount(root, pw);

                ll += pw;
                history_graph_ll[i].push_back(pw);
            }
        }

        // Fork-based timeout implementation
        // for(int i = 0; i < training_size; i++) {
        //     EdsGraph graph = graphs[i];
        //     std::cout << i << ":" << graph.nodes.size() << std::endl;
        //
        //     pid_t pid = fork();
        //     if (pid == 0) {
        //         // Child process
        //         auto code = context->Parse(graph);
        //         if(code == ParserError::kNone) {
        //             ChartItem* root = context->parser->Result();
        //
        //             buildBestParseRelationships(root, 0);  // Build relationships only for best parse
        //             addRulePointer(root);
        //
        //             double pw = computeViterbiInside(root);
        //             computeViterbiOutside(root);
        //             computeExpectedCount(root, pw);
        //
        //             ll += pw;
        //             history_graph_ll[i].push_back(pw);
        //         }
        //         exit(0);
        //     } else {
        //         // Parent process - wait with timeout
        //         int status;
        //         time_t start_time = time(NULL);
        //
        //         while (true) {
        //             int wait_result = waitpid(pid, &status, WNOHANG);
        //             if (wait_result > 0) {
        //                 // Child finished
        //                 break;
        //             }
        //             if (wait_result < 0) {
        //                 break;
        //             }
        //
        //             time_t current_time = time(NULL);
        //             double elapsed = difftime(current_time, start_time);
        //
        //             if (elapsed > time_out_in_seconds) {
        //                 // Timeout - kill child
        //                 kill(pid, SIGKILL);
        //                 waitpid(pid, &status, 0);
        //                 std::cout << "Graph " << i << " timed out after " << elapsed << "s, skipping\n";
        //                 break;
        //             }
        //
        //             usleep(500000); // Sleep 500ms between checks
        //         }
        //         // Read results from child process
        //     }
        // }

        updateEM();
        if (!validateProbabilities()) {
            std::cerr << "Warning: Invalid probabilities detected in iteration "
                      << iteration << std::endl;
        }

        for(int i = 0; i < shrg_rules.size(); i++){
            history[i].push_back(shrg_rules[i]->log_rule_weight);
        }

        t2 = clock();
        double time_diff = (double)(t2 - t1)/CLOCKS_PER_SEC;
        times.push_back(time_diff);
        std::cout << "Iteration: " << iteration
                  << "\nLog likelihood: " << ll
                  << "\nTime: " << time_diff << " seconds\n\n";

        lls.push_back(ll);

        if(!(output_dir == "N")){
            writeHistoryToDir(output_dir, history, shrg_rules.size());
            writeGraphLLToDir(output_dir, history_graph_ll, training_size);
            writeLLToDir(output_dir, lls, iteration);
            writeTimesToDir(output_dir, times, iteration);
        }

        iteration++;
    } while(!converged());
}

void ViterbiEM::reorganizeBestParse(ChartItem* root) {
    // Base case
    if (!root || !root->next_ptr) return;

    // Find best score and item in this level
    ChartItem* curr = root;
    ChartItem* best = root;
    float best_score = root->score;

    while (curr->next_ptr) {
        if (curr->next_ptr->score > best_score) {
            best = curr->next_ptr;
            best_score = curr->next_ptr->score;
        }
        curr = curr->next_ptr;
    }

    // If best isn't already first, swap it to front
    if (best != root) {
        best->Swap(*root);
    }

    // Recursively process children
    for (ChartItem* child : root->children) {
        reorganizeBestParse(child);
    }
}

// double ViterbiEM::computeViterbiInside(ChartItem* root) {
//     if(root->inside_visited_status == ChartItem::kVisited) {
//         return root->log_inside_prob;
//     }
//
//     // Only compute for first (best) rule after reorganization
//     double log_inside = root->rule_ptr->log_rule_weight;
//
//     // Compute children recursively
//     for(ChartItem* child : root->children) {
//         log_inside += computeViterbiInside(child);
//     }
//
//     root->log_inside_prob = log_inside;
//     root->inside_visited_status = ChartItem::kVisited;
//
//     return log_inside;
// }
double ViterbiEM::computeViterbiInside(ChartItem* root) {
    if(root->inside_visited_status == ChartItem::kVisited) {
        return root->log_inside_prob;
    }

    // Initialize with rule weight
    double log_inside = root->rule_ptr->log_rule_weight;

    // Safety check for valid log probability
    if (!std::isfinite(log_inside)) {
        std::cerr << "Warning: Invalid log probability in rule weight" << std::endl;
        log_inside = ChartItem::log_zero;
    }

    // Compute children recursively
    for(ChartItem* child : root->children) {
        double child_prob = computeViterbiInside(child);
        if (std::isfinite(child_prob)) {
            log_inside += child_prob;
        } else {
            std::cerr << "Warning: Invalid child probability" << std::endl;
        }
    }

    // Ensure we don't have invalid log probabilities
    if (!std::isfinite(log_inside)) {
        log_inside = ChartItem::log_zero;
    }

    root->log_inside_prob = log_inside;
    root->inside_visited_status = ChartItem::kVisited;

    return log_inside;
}

void ViterbiEM::computeViterbiOutside(ChartItem* root) {
    NodeLevelPQ pq;
    root->log_outside_prob = 0.0;
    pq.push(root);

    while(!pq.empty()) {
        ChartItem* node = pq.top();
        pq.pop();

        if(!node->parents_sib.empty()) {
            double log_outside = ChartItem::log_zero;

            for(auto& parent_sib : node->parents_sib) {
                ChartItem* parent = getParent(parent_sib);

                    std::vector<ChartItem*> siblings = getSiblings(parent_sib);

                    log_outside = parent->rule_ptr->log_rule_weight;
                    log_outside += parent->log_outside_prob;

                    for(auto sib : siblings) {
                        ChartItem* sib_head = sib;
                        log_outside += sib_head->log_inside_prob;
                    }
            }

            node->log_outside_prob = log_outside;
            node->outside_visited_status = ChartItem::kVisited;
        }

        for(ChartItem* child : node->children) {
            pq.push(child);
        }
    }
}

void ViterbiEM::computeExpectedCount(ChartItem* root, double pw) {
    if(root->count_visited_status == ChartItem::kVisited) {
        return;
    }

    double curr_log_count = root->rule_ptr->log_rule_weight;
    curr_log_count += root->log_outside_prob;
    curr_log_count -= pw;
    if(!is_normal_count(curr_log_count)){
        std::cout << "count";
    }

    for(ChartItem* child : root->children) {
        curr_log_count += child->log_inside_prob;
        if(!is_normal_count(curr_log_count)){
            std::cout << "count";
        }
    }

    root->log_sent_rule_count = curr_log_count;
    root->rule_ptr->log_count = addLogs(root->rule_ptr->log_count, curr_log_count);
    if(!is_normal_count(root->rule_ptr->log_count)){
        std::cout << "count";
    }

    root->count_visited_status = ChartItem::kVisited;

    // Process children recursively
    for(ChartItem* child : root->children) {
        computeExpectedCount(child, pw);
    }
}

void ViterbiEM::updateEM() {
    const double smoothing_factor = 1e-10;  // Small constant for smoothing
    LabelCount total_count;
    LabelToRule::iterator it;

    // First pass: compute totals with smoothing
    for(it = rule_dict.begin(); it != rule_dict.end(); it++) {
        RuleVector v = it->second;
        double log_total_count = ChartItem::log_zero;

        // Add smoothing to all rules
        for (auto rule : v) {
            double smoothed_count = addLogs(rule->log_count, std::log(smoothing_factor));
            log_total_count = addLogs(log_total_count, smoothed_count);
        }
        total_count[it->first] = log_total_count;
    }

    // Second pass: update rule weights with smoothing
    for(int i = 0; i < shrg_rules.size(); i++) {
        auto rule = shrg_rules[i];
        LabelHash l = rule->label_hash;

        // Add smoothing to count
        double smoothed_count = addLogs(rule->log_count, std::log(smoothing_factor));
        double new_phi = smoothed_count - total_count[l];

        // Ensure we don't have invalid log probabilities
        if (std::isfinite(new_phi)) {
            rule->log_rule_weight = new_phi;
        } else {
            // If we get an invalid value, use a very small probability
            rule->log_rule_weight = std::log(smoothing_factor);
        }

        // Verify the new weight is valid
        assert(std::isfinite(rule->log_rule_weight));
        assert(rule->log_rule_weight <= 0.0);  // Log probabilities should be <= 0
    }
}
bool ViterbiEM::converged() const {
    return std::abs(ll - prev_ll) <= threshold;
}
LabelToRule ViterbiEM::getRuleDict(){
    LabelToRule dict;
    for(auto rule:shrg_rules){
        dict[rule->label_hash].push_back(rule);
    }

    //remove duplicates
    LabelToRule::iterator it;
    for(it = dict.begin(); it != dict.end(); it++){
        std::set<SHRG*>s(it->second.begin(), it->second.end());
        dict[it->first] = RuleVector(s.begin(), s.end());
    }
    return dict;
}

} // namespace shrg::em
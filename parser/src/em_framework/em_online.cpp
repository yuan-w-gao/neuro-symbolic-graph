//
// Created by Yuan Gao on 07/02/2025.
//

#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ctime>

#include "em_online.hpp"

namespace shrg::em {
    OnlineEM::OnlineEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold)
        : EMBase(shrg_rules, graphs, context, threshold) {
        prev_ll = 0.0;
        rule_dict = getRuleDict();
        total_examples = graphs.size();
        examples_seen = 0;

        // Initialize previous weights
        for (auto rule : shrg_rules) {
            prev_weights[rule] = rule->log_rule_weight;
        }
    }

    OnlineEM::OnlineEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs,
            Context *context, double threshold,  std::string dir, int timeout_seconds)
            :EMBase(shrg_rules, graphs, context, threshold) {
        prev_ll = 0.0;
        rule_dict = getRuleDict();
        total_examples = graphs.size();
        examples_seen = 0;

        // Initialize previous weights
        for (auto rule : shrg_rules) {
            prev_weights[rule] = rule->log_rule_weight;
        }
        output_dir = std::move(dir);
        time_out_in_seconds = timeout_seconds;
    }

    void OnlineEM::computeExpectedCount(ChartItem *root, double pw) {
        if(root->count_visited_status == VISITED){
            return ;
        }
        ChartItem *ptr = root;

        do{
            double curr_log_count = ptr->rule_ptr->log_rule_weight;
            //        assert(is_negative(curr_log_count));

            curr_log_count += ptr->log_outside_prob;
            curr_log_count -= pw;

            for(ChartItem *child:ptr->children){
                curr_log_count += child->log_inside_prob;
            }

            ptr->log_sent_rule_count = curr_log_count;
            ptr->rule_ptr->log_count = addLogs(ptr->rule_ptr->log_count, curr_log_count);

            ptr->count_visited_status = VISITED;
            for(ChartItem *child:ptr->children){
                computeExpectedCount(child, pw);
            }
            ptr = ptr->next_ptr;
        }while(ptr != root);
    }

    void OnlineEM::run()  {
        std::cout << "Running Online EM Training...\n";
        clock_t t1, t2;
        int iteration = 0;
        ll = 0;
        setInitialWeights(rule_dict);

        std::vector<double> history[shrg_rules.size()];
        std::vector<double> history_graph_ll[total_examples];
        for(int i = 0; i < shrg_rules.size(); i++){
            history[i].push_back(shrg_rules[i]->log_rule_weight);
        }

        std::vector<double> lls;
        std::vector<double> times;

        // Create indices for shuffling
        std::vector<size_t> indices(graphs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());

        do {
            prev_ll = ll;
            ll = 0;
            t1 = clock();

            // Shuffle indices for random example processing
            std::shuffle(indices.begin(), indices.end(), g);

            // Original loop - commented out
            for (size_t i = 0; i < indices.size(); i++) {
                examples_seen++;

                // Clear counts for this example
                clearRuleCount();

                // Process single example
                EdsGraph& graph = graphs[indices[i]];
                auto code = context->Parse(graph);

                if (code == ParserError::kNone) {
                    ChartItem* root = context->parser->Result();
                    addParentPointerOptimized(root, 0);
                    addRulePointer(root);

                    double pw = computeInside(root);
                    computeOutside(root);
                    computeExpectedCount(root, pw);

                    ll += pw;

                    // Update weights after each example
                    updateEM();
                }

                // Log progress periodically
                if ((i + 1) % 1000 == 0) {
                    std::cout << "Processed " << (i + 1) << " examples\n";
                }
            }

            // Fork-based timeout implementation
            // for (size_t i = 0; i < indices.size(); i++) {
            //     examples_seen++;
            //     size_t idx = indices[i];
            //     EdsGraph& graph = graphs[idx];
            //     std::cout << idx << ":" << graph.nodes.size() << std::endl;
            //
            //     // Clear counts for this example
            //     clearRuleCount();
            //
            //     pid_t pid = fork();
            //     if (pid == 0) {
            //         // Child process
            //         auto code = context->Parse(graph);
            //         if (code == ParserError::kNone) {
            //             ChartItem* root = context->parser->Result();
            //             addParentPointerOptimized(root, 0);
            //             addRulePointer(root);
            //
            //             double pw = computeInside(root);
            //             computeOutside(root);
            //             computeExpectedCount(root, pw);
            //             ll += pw;
            //
            //             // Update weights after each example
            //             updateEM();
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
            //                 std::cout << "Graph " << idx << " timed out after " << elapsed << "s, skipping\n";
            //                 break;
            //             }
            //
            //             usleep(500000); // Sleep 500ms between checks
            //         }
            //         // Read results from child process
            //     }
            //
            //     // Log progress periodically
            //     if ((i + 1) % 1000 == 0) {
            //         std::cout << "Processed " << (i + 1) << " examples\n";
            //     }
            // }

            for(int i = 0; i < shrg_rules.size(); i++){
                history[i].push_back(shrg_rules[i]->log_rule_weight);
            }
            lls.push_back(ll);

            t2 = clock();
            double time_diff = (double)(t2 - t1)/CLOCKS_PER_SEC;
            times.push_back(time_diff);
            std::cout << "Iteration: " << iteration
                      << "\nLog likelihood: " << ll
                      << "\nTime: " << time_diff << " seconds\n\n";

            if(!(output_dir == "N")){
                writeHistoryToDir(output_dir, history, shrg_rules.size());
                writeGraphLLToDir(output_dir, history_graph_ll, total_examples);
                writeLLToDir(output_dir, lls, iteration);
                writeTimesToDir(output_dir, times, iteration);
            }

            iteration++;
        } while(!converged());
    }



    void OnlineEM::updateEM() {
        // Learning rate decreases with number of examples seen
        double learning_rate = 1.0 / std::sqrt(examples_seen);
        double retain_rate = 1.0 - learning_rate;

        // Group rules by labelHash
        std::unordered_map<LabelHash, std::vector<std::pair<SHRG*, double>>>
            label_to_unnormalized;

        // First pass: compute unnormalized weights
        for (auto& pair : rule_dict) {
            LabelHash label = pair.first;

            // Get total counts for this label in current example
            double log_total_count = ChartItem::log_zero;
            for (auto rule : pair.second) {
                if (rule->log_count != ChartItem::log_zero) {
                    log_total_count = addLogs(log_total_count, rule->log_count);
                }
            }

            // Compute unnormalized weights for each rule
            for (auto rule : pair.second) {
                double new_phi;
                if (rule->log_count != ChartItem::log_zero) {
                    // Rule appears in example
                    new_phi = rule->log_count - log_total_count;
                } else {
                    // Rule not in example
                    new_phi = prev_weights[rule];
                }

                // Add smoothing
                double smoothed_new = addLogs(new_phi, std::log(smoothing_factor));
                double smoothed_prev = addLogs(prev_weights[rule],
                                             std::log(smoothing_factor));

                // Combine weights with learning rate
                double unnormalized = std::log(
                    retain_rate * std::exp(smoothed_prev) +
                    learning_rate * std::exp(smoothed_new)
                );

                label_to_unnormalized[label].push_back({rule, unnormalized});
            }
        }

        // Second pass: normalize within each labelHash group
        for (auto& pair : label_to_unnormalized) {
            // Compute sum for normalization
            double log_sum = ChartItem::log_zero;
            for (const auto& rule_weight : pair.second) {
                log_sum = addLogs(log_sum, rule_weight.second);
            }

            // Normalize and update weights
            for (const auto& rule_weight : pair.second) {
                SHRG* rule = rule_weight.first;
                double normalized = rule_weight.second - log_sum;

                // Verify normalization
                assert(normalized <= 0.0);
                assert(std::isfinite(normalized));

                // Update weights
                prev_weights[rule] = normalized;
                rule->log_rule_weight = normalized;
            }
        }
    }

    bool OnlineEM::converged() const  {
        return std::abs(ll - prev_ll) <= threshold;
    }

    LabelToRule OnlineEM::getRuleDict() {
        LabelToRule dict;
        for (auto rule : shrg_rules) {
            dict[rule->label_hash].push_back(rule);
        }

        // Remove duplicates
        for (auto& pair : dict) {
            std::set<SHRG*> s(pair.second.begin(), pair.second.end());
            pair.second = RuleVector(s.begin(), s.end());
        }
        return dict;
    }
};// namespace shrg::em
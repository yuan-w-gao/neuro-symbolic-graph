//
// Created by Yuan Gao on 07/02/2025.
//
#include "em_batch.hpp"
#include <fstream>
#include <future>
#include <numeric>
#include <vector>
#include <random>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ctime>

namespace shrg::em {
BatchEM::BatchEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs,
            Context *context, double threshold, int batch_size)
        : EMBase(shrg_rules, graphs, context, threshold),
          batch_size_(batch_size) {
    prev_ll = 0.0;
    rule_dict = getRuleDict();
    total_examples = graphs.size();

    // Initialize previous weights for all rules
    for (auto rule : shrg_rules) {
        prev_weights[rule] = rule->log_rule_weight;
    }
}

BatchEM::BatchEM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs,
            Context *context, double threshold, int batch_size, std::string dir, int timeout_seconds)
        : EMBase(shrg_rules, graphs, context, threshold),
          batch_size_(batch_size) {
    prev_ll = 0.0;
    rule_dict = getRuleDict();
    total_examples = graphs.size();

    // Initialize previous weights for all rules
    for (auto rule : shrg_rules) {
        prev_weights[rule] = rule->log_rule_weight;
    }
    output_dir = std::move(dir);
    time_out_in_seconds = timeout_seconds;
}


void BatchEM::computeExpectedCount(ChartItem *root, double pw) {
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

void BatchEM::run() {
        std::cout << "Running Batch EM Training...\n";
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

        std::vector<size_t> indices(graphs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());

        do {
            prev_ll = ll;
            ll = 0;
            t1 = clock();

            // Shuffle indices for random batch selection
            std::shuffle(indices.begin(), indices.end(), g);

            // Process batches
            size_t total_batches = (indices.size() + batch_size_ - 1) / batch_size_;
            size_t batch_num = 0;

            for (size_t batch_start = 0; batch_start < indices.size();
                 batch_start += batch_size_) {

                size_t batch_end = std::min(batch_start + batch_size_,
                                          indices.size());

                batch_num++;
                clearRuleCount();

                // Original loop - commented out
                for (size_t i = batch_start; i < batch_end; i++) {
                    size_t idx = indices[i];
                    EdsGraph& graph = graphs[idx];

                    // Progress tracking for every graph
                    std::cout << "\r[iter " << iteration << "] " << graph.sentence_id
                              << " (" << (i + 1) << "/" << indices.size() << ")" << std::flush;

                    auto code = context->Parse(graph);
                    if (code == ParserError::kNone) {
                        ChartItem* root = context->parser->Result();
                        addParentPointerOptimized(root, 0);
                        addRulePointer(root);

                        double pw = computeInside(root);
                        computeOutside(root);
                        computeExpectedCount(root, pw);

                        ll += pw;
                    }
                }

                // Fork-based timeout implementation
                // for (size_t i = batch_start; i < batch_end; i++) {
                //     size_t idx = indices[i];
                //     EdsGraph& graph = graphs[idx];
                //     std::cout << idx << ":" << graph.nodes.size() << std::endl;
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
                // }

                curr_batch_size = batch_end - batch_start;
                // updateBatchWeights(batch_end - batch_start);
                updateEM();
            }
            std::cout << std::endl;  // Newline after progress

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
void BatchEM::verifyNormalization() {
    const double epsilon = 1e-6;  // Tolerance for floating point comparison

    for (const auto& pair : rule_dict) {
        double sum = 0.0;
        for (const auto& rule : pair.second) {
            sum += std::exp(rule->log_rule_weight);
        }
        if (std::abs(sum - 1.0) > epsilon) {
            std::cerr << "Warning: Rules for labelHash " << pair.first
                      << " sum to " << sum << std::endl;
        }
    }
}
void BatchEM::updateEM() {
    int current_batch_size = curr_batch_size;
    double batch_weight = static_cast<double>(current_batch_size) /
                            total_examples;
        double prev_weight = 1.0 - batch_weight;

        std::unordered_map<LabelHash, std::vector<std::pair<SHRG*, double>>>
            label_to_unnormalized;

        for (auto& pair : rule_dict) {
            LabelHash label = pair.first;

            double log_total_count = ChartItem::log_zero;
            for (auto rule : pair.second) {
                if (rule->log_count != ChartItem::log_zero) {
                    log_total_count = addLogs(log_total_count, rule->log_count);
                }
            }

            for (auto rule : pair.second) {
                double new_phi;
                if (rule->log_count != ChartItem::log_zero) {
                    new_phi = rule->log_count - log_total_count;
                } else {
                    new_phi = prev_weights[rule];
                }

                double smoothed_new = addLogs(new_phi, std::log(smoothing_factor));
                double smoothed_prev = addLogs(prev_weights[rule],
                                             std::log(smoothing_factor));

                double unnormalized = std::log(
                    prev_weight * std::exp(smoothed_prev) +
                    batch_weight * std::exp(smoothed_new)
                );

                label_to_unnormalized[label].push_back({rule, unnormalized});
            }
        }

        for (auto& pair : label_to_unnormalized) {
            double log_sum = ChartItem::log_zero;
            for (const auto& rule_weight : pair.second) {
                log_sum = addLogs(log_sum, rule_weight.second);
            }

            for (const auto& rule_weight : pair.second) {
                SHRG* rule = rule_weight.first;
                double normalized = rule_weight.second - log_sum;

                assert(normalized <= 0.0);  // log probability should be <= 0
                assert(std::isfinite(normalized));

                // Update weights
                prev_weights[rule] = normalized;
                rule->log_rule_weight = normalized;
            }
        }

        verifyNormalization();
}

bool BatchEM::converged() const {
    return std::abs(ll - prev_ll) <= threshold;
}

LabelToRule BatchEM::getRuleDict() {
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
}
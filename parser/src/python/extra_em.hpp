//
// Python bindings for C++ EM algorithms
//
#ifndef EXTRA_EM_HPP
#define EXTRA_EM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <functional>
#include <ctime>
#include <limits>
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

#include "../manager.hpp"
#include "../em_framework/em_base.hpp"
#include "../em_framework/em.hpp"

namespace shrg {

// Type aliases
using RuleVector = std::vector<SHRG*>;
using LabelToRule = std::map<LabelHash, RuleVector>;
using LabelCount = std::map<LabelHash, double>;

// Helper: log-sum-exp
inline double em_addLogs(double a, double b) {
    if (a == ChartItem::log_zero) return b;
    if (b == ChartItem::log_zero) return a;
    if (a > b) {
        return a + log1p(exp(b - a));
    }
    return b + log1p(exp(a - b));
}

struct EMResult {
    std::vector<double> final_weights;
    std::vector<std::vector<double>> weight_history;
    std::vector<double> log_likelihoods;
    int num_iterations;
    bool converged;
    double elapsed_time;
};

// Run batch EM training and return results
inline EMResult run_batch_em(
    Manager& manager,
    double convergence_threshold = 1e-4,
    int max_iterations = 100,
    int max_graphs = -1,  // -1 means all graphs
    bool verbose = true
) {
    EMResult result;
    result.converged = false;

    // Use grammars (grammar objects, 0 to hrg_size-1) for consistent indexing
    // This matches Python EM export_forest which uses grammar_index = grammar_ptr - grammars.data()
    auto& grammars = manager.grammars;
    auto& graphs = manager.edsgraphs;

    if (grammars.empty()) {
        throw std::runtime_error("No grammars loaded");
    }
    if (graphs.empty()) {
        throw std::runtime_error("No graphs loaded");
    }

    // Allocate and initialize contexts
    // Always reinitialize to ensure parser references current grammars
    if (manager.contexts.empty()) {
        manager.Allocate(1);
    }
    manager.InitAll("linear", verbose);

    Context* context = manager.contexts[0];
    Generator* generator = context->parser->GetGenerator();

    size_t num_graphs = graphs.size();
    if (max_graphs > 0 && static_cast<size_t>(max_graphs) < num_graphs) {
        num_graphs = static_cast<size_t>(max_graphs);
    }

    // Initialize weight history - use grammar object count, not shrg_index count
    size_t num_rules = grammars.size();
    result.weight_history.resize(num_rules);

    // Get rule dictionary and set initial weights
    // Iterate over grammar objects (not shrg_rules) for consistent indexing
    LabelToRule rule_dict;
    for (auto& grammar : grammars) {
        rule_dict[grammar.label_hash].push_back(&grammar);
    }

    // Set initial weights (no duplicates when iterating grammars directly)
    for (auto& pair : rule_dict) {
        double initial_weight = std::log(1.0 / pair.second.size());
        for (auto* rule : pair.second) {
            rule->log_rule_weight = initial_weight;
        }
    }

    // Record initial weights using grammar object indices
    for (size_t i = 0; i < num_rules; i++) {
        result.weight_history[i].push_back(grammars[i].log_rule_weight);
    }

    clock_t start_time = clock();

    double prev_ll = -std::numeric_limits<double>::infinity();
    double ll = 0;
    int iteration = 0;

    // Helper function to add parent pointers
    std::function<void(ChartItem*, int)> addParentPointer = [&](ChartItem* root, int level) {
        ChartItem* ptr = root;
        do {
            if (level > ptr->level) {
                ptr->level = level;
            }

            const SHRG* rule = ptr->attrs_ptr->grammar_ptr;
            if (ptr->child_visited_status != em::EMBase::VISITED) {
                for (auto* edge_ptr : rule->nonterminal_edges) {
                    ChartItem* child = generator->FindChartItemByEdge(ptr, edge_ptr);
                    ptr->children.push_back(child);
                    addParentPointer(child, ptr->level + 1);
                }
                for (size_t i = 0; i < ptr->children.size(); i++) {
                    std::vector<ChartItem*> sib;
                    for (size_t j = 0; j < ptr->children.size(); j++) {
                        if (j != i) {
                            sib.push_back(ptr->children[j]);
                        }
                    }
                    auto res = std::make_tuple(ptr, sib);
                    ptr->children[i]->parents_sib.push_back(res);
                }
                ptr->child_visited_status = em::EMBase::VISITED;
            } else {
                for (auto* child : ptr->children) {
                    addParentPointer(child, ptr->level + 1);
                }
            }
            ptr = ptr->next_ptr;
        } while (ptr != root);
    };

    // Helper function to add rule pointers
    // Use grammar object index (pointer arithmetic) instead of shrg_index for consistent indexing
    std::function<void(ChartItem*)> addRulePointer = [&](ChartItem* root) {
        if (root->rule_visited == em::EMBase::VISITED) return;

        ChartItem* ptr = root;
        do {
            // Compute grammar object index using pointer arithmetic
            // This matches Python export_forest which uses: grammar_ptr - grammars.data()
            ptr->rule_ptr = const_cast<SHRG*>(ptr->attrs_ptr->grammar_ptr);
            ptr->rule_visited = em::EMBase::VISITED;

            for (ChartItem* child : ptr->children) {
                addRulePointer(child);
            }
            ptr = ptr->next_ptr;
        } while (ptr != root);
    };

    // Helper function to compute inside
    std::function<double(ChartItem*)> computeInside = [&](ChartItem* root) -> double {
        if (root->inside_visited_status == em::EMBase::VISITED) {
            return root->log_inside_prob;
        }

        ChartItem* ptr = root;
        double log_inside = ChartItem::log_zero;

        do {
            double curr_log_inside = ptr->rule_ptr->log_rule_weight;
            for (ChartItem* child : ptr->children) {
                curr_log_inside += computeInside(child);
            }
            log_inside = em_addLogs(log_inside, curr_log_inside);
            ptr = ptr->next_ptr;
        } while (ptr != root);

        do {
            ptr->log_inside_prob = log_inside;
            ptr->inside_visited_status = em::EMBase::VISITED;
            ptr = ptr->next_ptr;
        } while (ptr != root);

        return log_inside;
    };

    // Helper struct for priority queue
    struct LessThanByLevel {
        bool operator()(ChartItem* lhs, ChartItem* rhs) const {
            return lhs->level > rhs->level;
        }
    };
    using NodeLevelPQ = std::priority_queue<ChartItem*, std::vector<ChartItem*>, LessThanByLevel>;

    // Helper function to compute outside
    std::function<void(ChartItem*)> computeOutside = [&](ChartItem* root_ptr) {
        NodeLevelPQ pq;
        ChartItem* ptr = root_ptr;

        do {
            ptr->log_outside_prob = 0.0;
            ptr = ptr->next_ptr;
        } while (ptr != root_ptr);

        pq.push(ptr);

        auto computeOutsideNode = [&](ChartItem* root, NodeLevelPQ& queue) {
            ChartItem* p = root;
            do {
                if (p->parents_sib.empty()) {
                    p = p->next_ptr;
                    continue;
                }

                double log_outside = ChartItem::log_zero;
                for (auto& parent_sib : p->parents_sib) {
                    ChartItem* parent = std::get<0>(parent_sib);
                    auto& siblings = std::get<1>(parent_sib);

                    double curr_log_outside = parent->rule_ptr->log_rule_weight;
                    curr_log_outside += parent->log_outside_prob;

                    for (auto* sib : siblings) {
                        curr_log_outside += sib->log_inside_prob;
                    }
                    log_outside = em_addLogs(log_outside, curr_log_outside);
                }
                p->log_outside_prob = log_outside;
                p->outside_visited_status = em::EMBase::VISITED;
                p = p->next_ptr;
            } while (p != root);

            double root_log_outside = root->log_outside_prob;

            do {
                if (p->outside_visited_status != em::EMBase::VISITED) {
                    p->log_outside_prob = root_log_outside;
                    p->outside_visited_status = em::EMBase::VISITED;
                }
                for (ChartItem* child : p->children) {
                    queue.push(child);
                }
                p = p->next_ptr;
            } while (p != root);
        };

        while (!pq.empty()) {
            ChartItem* node = pq.top();
            computeOutsideNode(node, pq);
            pq.pop();
        }
    };

    // Helper function to compute expected counts
    std::function<void(ChartItem*, double)> computeExpectedCount = [&](ChartItem* root, double pw) {
        if (root->count_visited_status == em::EMBase::VISITED) return;

        ChartItem* ptr = root;
        do {
            double curr_log_count = ptr->rule_ptr->log_rule_weight;
            curr_log_count += ptr->log_outside_prob;
            curr_log_count -= pw;

            for (ChartItem* child : ptr->children) {
                curr_log_count += child->log_inside_prob;
            }

            ptr->rule_ptr->log_count = em_addLogs(ptr->rule_ptr->log_count, curr_log_count);
            ptr->count_visited_status = em::EMBase::VISITED;

            for (ChartItem* child : ptr->children) {
                computeExpectedCount(child, pw);
            }
            ptr = ptr->next_ptr;
        } while (ptr != root);
    };

    // Main EM loop
    while (iteration < max_iterations) {
        ll = 0;
        clock_t iter_start = clock();

        // Clear counts for all grammar objects
        for (auto& grammar : grammars) {
            grammar.log_count = ChartItem::log_zero;
        }

        // E-step: compute expected counts for all graphs
        int processed = 0;
        for (size_t i = 0; i < num_graphs; i++) {
            EdsGraph& graph = graphs[i];

            if (verbose) {
                std::cout << "\r[iter " << iteration << "] " << graph.sentence_id
                          << " (" << (i + 1) << "/" << num_graphs << ")" << std::flush;
            }

            auto code = context->Parse(graph);
            if (code == ParserError::kNone) {
                ChartItem* root = context->parser->Result();
                addParentPointer(root, 0);
                addRulePointer(root);

                double pw = computeInside(root);
                computeOutside(root);
                computeExpectedCount(root, pw);

                ll += pw;
                processed++;
            }
            context->ReleaseMemory();
        }

        // M-step: update weights
        LabelCount total_count;
        for (auto& pair : rule_dict) {
            double log_total = ChartItem::log_zero;
            for (auto* rule : pair.second) {
                log_total = em_addLogs(log_total, rule->log_count);
            }
            total_count[pair.first] = log_total;
        }

        // Update weights for each grammar object
        for (auto& grammar : grammars) {
            double new_phi;
            if (grammar.log_count == ChartItem::log_zero) {
                if (rule_dict[grammar.label_hash].size() == 1) {
                    new_phi = 0.0;
                } else {
                    new_phi = ChartItem::log_zero;
                }
            } else {
                new_phi = grammar.log_count - total_count[grammar.label_hash];
            }
            grammar.log_rule_weight = new_phi;
        }

        // Record history using grammar object indices
        for (size_t i = 0; i < num_rules; i++) {
            result.weight_history[i].push_back(grammars[i].log_rule_weight);
        }
        result.log_likelihoods.push_back(ll);

        if (verbose) {
            clock_t iter_end = clock();
            double iter_time = static_cast<double>(iter_end - iter_start) / CLOCKS_PER_SEC;
            std::cout << "\n  LL: " << ll << ", time: " << iter_time << "s" << std::endl;
        }

        // Check convergence
        double improvement = ll - prev_ll;
        if (improvement >= 0 && improvement < convergence_threshold && iteration > 0) {
            result.converged = true;
            iteration++;
            break;
        }

        prev_ll = ll;
        iteration++;
    }

    clock_t end_time = clock();
    result.elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
    result.num_iterations = iteration;

    // Copy final weights using grammar object indices
    result.final_weights.resize(num_rules);
    for (size_t i = 0; i < num_rules; i++) {
        result.final_weights[i] = grammars[i].log_rule_weight;
    }

    if (verbose) {
        std::cout << "\nEM completed in " << result.num_iterations << " iterations, "
                  << result.elapsed_time << "s" << std::endl;
    }

    return result;
}

// Simpler function that just trains and returns final weights
inline std::vector<double> train_em_simple(
    Manager& manager,
    double convergence_threshold = 1e-4,
    int max_iterations = 100,
    bool verbose = true
) {
    EMResult result = run_batch_em(manager, convergence_threshold, max_iterations, -1, verbose);
    return result.final_weights;
}

// =============================================================================
// Optimized EM - uses forest caching and O(n log n) outside algorithm
// =============================================================================

struct OptimizedEMResult {
    std::vector<double> final_weights;
    std::vector<std::vector<double>> weight_history;
    std::vector<double> log_likelihoods;
    std::vector<double> iteration_times;
    int num_iterations;
    bool converged;
    double total_time;
    int num_forests_cached;

    // Per-graph metrics (if profiling enabled)
    std::vector<em::GraphMetrics> graph_metrics;
};

// Run optimized EM with forest caching
inline OptimizedEMResult run_em_optimized(
    Manager& manager,
    double convergence_threshold = 1e-4,
    int max_iterations = 100,
    const std::string& output_dir = "N",
    bool enable_profiling = false,
    bool use_safe_mode = false,
    int timeout_seconds = 5,
    const std::vector<std::string>& skip_graphs = {},
    bool verbose = true,
    bool parser_verbose = false  // Detailed parser stats (usually too noisy)
) {
    OptimizedEMResult result;
    result.converged = false;
    result.num_iterations = 0;
    result.num_forests_cached = 0;

    // Use grammars (grammar objects, 0 to hrg_size-1) for consistent indexing
    // This matches Python EM export_forest which uses grammar_index = grammar_ptr - grammars.data()
    auto& grammars = manager.grammars;
    auto& graphs = manager.edsgraphs;

    if (grammars.empty()) {
        throw std::runtime_error("No grammars loaded");
    }
    if (graphs.empty()) {
        throw std::runtime_error("No graphs loaded");
    }

    // Create vector of pointers to grammar objects for em::EM compatibility
    // em::EM takes std::vector<SHRG*>&, so we create pointers to grammars
    std::vector<SHRG*> grammar_ptrs;
    grammar_ptrs.reserve(grammars.size());
    for (auto& g : grammars) {
        grammar_ptrs.push_back(&g);
    }

    // Allocate and initialize context
    if (manager.contexts.empty()) {
        manager.Allocate(1);
    }
    manager.InitAll("linear", parser_verbose);  // Use parser_verbose for detailed parser stats

    Context* context = manager.contexts[0];

    // Convert skip_graphs vector to set
    std::unordered_set<std::string> skip_set(skip_graphs.begin(), skip_graphs.end());

    // Create the optimized EM instance with grammar object pointers
    em::EM em_trainer(grammar_ptrs, graphs, context, convergence_threshold,
                      output_dir, timeout_seconds, skip_set);

    // Set EM-level verbose (progress messages, skip notifications, summaries)
    em_trainer.setVerbose(verbose);

    if (enable_profiling) {
        em_trainer.enableProfiling(true);
    }

    // Initialize weights
    em_trainer.initializeWeights();

    // Record initial weights using grammar object indices
    size_t num_rules = grammars.size();
    result.weight_history.resize(num_rules);
    for (size_t i = 0; i < num_rules; i++) {
        result.weight_history[i].push_back(grammars[i].log_rule_weight);
    }

    clock_t start_time = clock();

    // Run the optimized EM (this does forest caching internally)
    if (use_safe_mode) {
        em_trainer.run_safe();
    } else {
        em_trainer.run();
    }

    clock_t end_time = clock();
    result.total_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    // Copy results from EM trainer using grammar object indices
    result.final_weights.resize(num_rules);
    for (size_t i = 0; i < num_rules; i++) {
        result.final_weights[i] = grammars[i].log_rule_weight;
    }

    // Copy log likelihood history and iteration info
    result.log_likelihoods = em_trainer.getLogLikelihoodHistory();
    result.iteration_times = em_trainer.getIterationTimes();
    result.num_iterations = em_trainer.getNumIterations();
    result.converged = em_trainer.hasConverged();

    // Copy profiling metrics if enabled
    if (enable_profiling) {
        result.graph_metrics = em_trainer.graph_metrics_;
    }

    // Get number of cached forests
    result.num_forests_cached = em_trainer.getNumCachedForests();

    if (verbose) {
        std::cout << "\nOptimized EM completed in " << result.total_time << "s"
                  << ", " << result.num_iterations << " iterations"
                  << ", " << result.num_forests_cached << " cached forests"
                  << ", final LL: " << (result.log_likelihoods.empty() ? -std::numeric_limits<double>::infinity() : result.log_likelihoods.back())
                  << std::endl;
    }

    return result;
}

// Wrapper class for Python that holds the EM trainer state
class OptimizedEMTrainer {
public:
    OptimizedEMTrainer(
        Manager& manager,
        double convergence_threshold = 1e-4,
        const std::string& output_dir = "N",
        int timeout_seconds = 5,
        const std::vector<std::string>& skip_graphs = {}
    ) : manager_(manager),
        convergence_threshold_(convergence_threshold),
        output_dir_(output_dir),
        timeout_seconds_(timeout_seconds),
        skip_set_(skip_graphs.begin(), skip_graphs.end()),
        em_trainer_(nullptr),
        forests_cached_(false) {

        if (manager_.grammars.empty()) {
            throw std::runtime_error("No grammars loaded");
        }
        if (manager_.edsgraphs.empty()) {
            throw std::runtime_error("No graphs loaded");
        }

        // Create vector of pointers to grammar objects for em::EM compatibility
        // em::EM takes std::vector<SHRG*>&, so we create pointers to grammars
        grammar_ptrs_.reserve(manager_.grammars.size());
        for (auto& g : manager_.grammars) {
            grammar_ptrs_.push_back(&g);
        }

        // Allocate and initialize context
        if (manager_.contexts.empty()) {
            manager_.Allocate(1);
        }
        manager_.InitAll("linear", false);
    }

    ~OptimizedEMTrainer() {
        if (em_trainer_) {
            delete em_trainer_;
        }
    }

    void enable_profiling(bool enable = true) {
        profiling_enabled_ = enable;
    }

    // Parse all graphs and cache derivation forests (Phase 1)
    int cache_forests(bool verbose = true) {
        if (em_trainer_) {
            delete em_trainer_;
        }

        Context* context = manager_.contexts[0];
        // Use grammar object pointers for consistent indexing
        em_trainer_ = new em::EM(
            grammar_ptrs_,
            manager_.edsgraphs,
            context,
            convergence_threshold_,
            output_dir_,
            timeout_seconds_,
            skip_set_
        );

        if (profiling_enabled_) {
            em_trainer_->enableProfiling(true);
        }

        em_trainer_->initializeWeights();

        // Parse and cache forests
        if (verbose) {
            std::cout << "Caching derivation forests..." << std::endl;
        }

        // The run() method handles both parsing and EM iterations
        // For just caching, we'd need a separate method
        // For now, this is handled within run()
        forests_cached_ = true;

        return 0;  // Will be updated after first iteration
    }

    // Run full EM training
    OptimizedEMResult run(bool use_safe_mode = false, bool verbose = true) {
        OptimizedEMResult result;

        if (!em_trainer_) {
            Context* context = manager_.contexts[0];
            // Use grammar object pointers for consistent indexing
            em_trainer_ = new em::EM(
                grammar_ptrs_,
                manager_.edsgraphs,
                context,
                convergence_threshold_,
                output_dir_,
                timeout_seconds_,
                skip_set_
            );

            if (profiling_enabled_) {
                em_trainer_->enableProfiling(true);
            }

            em_trainer_->initializeWeights();
        }

        clock_t start_time = clock();

        if (use_safe_mode) {
            em_trainer_->run_safe();
        } else {
            em_trainer_->run();
        }

        clock_t end_time = clock();
        result.total_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

        // Copy results from EM trainer using grammar object indices
        size_t num_rules = manager_.grammars.size();
        result.final_weights.resize(num_rules);
        for (size_t i = 0; i < num_rules; i++) {
            result.final_weights[i] = manager_.grammars[i].log_rule_weight;
        }

        // Copy log likelihood history and iteration info
        result.log_likelihoods = em_trainer_->getLogLikelihoodHistory();
        result.iteration_times = em_trainer_->getIterationTimes();
        result.num_iterations = em_trainer_->getNumIterations();
        result.converged = em_trainer_->hasConverged();

        if (profiling_enabled_ && em_trainer_) {
            result.graph_metrics = em_trainer_->graph_metrics_;
        }

        result.num_forests_cached = em_trainer_->getForests().size();

        return result;
    }

    // Get current weights using grammar object indices
    std::vector<double> get_weights() const {
        std::vector<double> weights(manager_.grammars.size());
        for (size_t i = 0; i < manager_.grammars.size(); i++) {
            weights[i] = manager_.grammars[i].log_rule_weight;
        }
        return weights;
    }

    // Set weights using grammar object indices
    void set_weights(const std::vector<double>& weights) {
        if (weights.size() != manager_.grammars.size()) {
            throw std::runtime_error("Weight vector size mismatch");
        }
        for (size_t i = 0; i < weights.size(); i++) {
            manager_.grammars[i].log_rule_weight = weights[i];
        }
    }

    // Run validation (compare original vs optimized outside)
    void run_validation() {
        if (!em_trainer_) {
            throw std::runtime_error("Must call cache_forests() or run() first");
        }
        em_trainer_->runValidation();
    }

    // Print profiling summary
    void print_metrics_summary() {
        if (!em_trainer_) {
            throw std::runtime_error("Must call run() first");
        }
        em_trainer_->printMetricsSummary();
    }

    // Get graph metrics
    std::vector<em::GraphMetrics> get_graph_metrics() const {
        if (!em_trainer_) {
            return {};
        }
        return em_trainer_->graph_metrics_;
    }

private:
    Manager& manager_;
    double convergence_threshold_;
    std::string output_dir_;
    int timeout_seconds_;
    std::unordered_set<std::string> skip_set_;
    std::vector<SHRG*> grammar_ptrs_;  // Pointers to grammar objects for em::EM
    em::EM* em_trainer_;
    bool forests_cached_;
    bool profiling_enabled_ = false;
};

// Result of safe graph index checking
struct SafeGraphIndices {
    std::vector<int> safe_indices;      // Indices of graphs that are safe to parse
    std::vector<std::string> safe_ids;  // Sentence IDs of safe graphs
    int num_tested;
    int num_safe;
    int num_skipped;
    int num_oom;
    double total_time;
};

// Test which graphs are safe to parse using fork-based OOM protection
// Returns indices of safe graphs - caller then parses only those
inline SafeGraphIndices test_safe_graph_indices(
    Manager& manager,
    int timeout_seconds = 10,
    int max_graphs = -1,
    bool verbose = true
) {
    SafeGraphIndices result;
    result.num_tested = 0;
    result.num_safe = 0;
    result.num_skipped = 0;
    result.num_oom = 0;

    auto& graphs = manager.edsgraphs;

    if (graphs.empty()) {
        throw std::runtime_error("No graphs loaded");
    }

    // Allocate and initialize context
    if (manager.contexts.empty()) {
        manager.Allocate(1);
    }
    manager.InitAll("linear", false);

    Context* context = manager.contexts[0];

    size_t num_graphs = graphs.size();
    if (max_graphs > 0 && static_cast<size_t>(max_graphs) < num_graphs) {
        num_graphs = static_cast<size_t>(max_graphs);
    }

    clock_t start_time = clock();

    for (size_t i = 0; i < num_graphs; i++) {
        EdsGraph& graph = graphs[i];
        result.num_tested++;

        if (verbose) {
            std::cout << "\r[testing] " << graph.sentence_id
                      << " (" << (i + 1) << "/" << num_graphs << ")" << std::flush;
        }

        // Fork child to test if parse is safe
        pid_t pid = fork();
        if (pid == 0) {
            // Child: test parse with timeout
            alarm(timeout_seconds);
            auto code = context->Parse(graph);
            context->ReleaseMemory();
            _exit(code == ParserError::kNone ? 0 : 1);
        }

        if (pid < 0) {
            // Fork failed - skip
            result.num_skipped++;
            continue;
        }

        // Parent: wait for child
        int status;
        waitpid(pid, &status, 0);

        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            if (WIFSIGNALED(status)) {
                int sig = WTERMSIG(status);
                if (sig == SIGKILL) {
                    result.num_oom++;
                    if (verbose) {
                        std::cout << " [OOM]" << std::flush;
                    }
                } else if (sig == SIGALRM) {
                    result.num_skipped++;
                    if (verbose) {
                        std::cout << " [TIMEOUT]" << std::flush;
                    }
                } else {
                    result.num_skipped++;
                }
            } else {
                result.num_skipped++;
            }
            continue;
        }

        // Child succeeded - this graph is safe
        result.safe_indices.push_back(static_cast<int>(i));
        result.safe_ids.push_back(graph.sentence_id);
        result.num_safe++;
    }

    if (verbose) {
        std::cout << "\nTested " << result.num_tested
                  << " graphs: " << result.num_safe << " safe"
                  << ", " << result.num_skipped << " skipped"
                  << ", " << result.num_oom << " OOM" << std::endl;
    }

    result.total_time = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
    return result;
}

} // namespace shrg

#endif // EXTRA_EM_HPP

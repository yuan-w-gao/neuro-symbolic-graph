//
// Created by Yuan Gao on 04/06/2024.
//
#include "em.hpp"
#include <cmath>
#include <iostream>
#include <utility>
#include <fstream>
#include <vector>
#include <future>
#include <set>
#include <algorithm>
#include <limits>
#include <queue>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <iomanip>
#include <numeric>

// Memory tracking for macOS
#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace {
    // Get current process memory usage in MB
    double getMemoryUsageMB() {
#ifdef __APPLE__
        struct mach_task_basic_info info;
        mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
        kern_return_t kerr = task_info(mach_task_self(),
                                       MACH_TASK_BASIC_INFO,
                                       (task_info_t)&info,
                                       &size);
        if (kerr == KERN_SUCCESS) {
            return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
        }
#endif
        return -1.0;  // Unknown
    }
}


namespace shrg::em {
 EM::EM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold)
    : EMBase(shrg_rules, graphs, context, threshold){
    prev_ll = 0.0;
    rule_dict = getRuleDict();
}

EM::EM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold, std::string dir) : EMBase(shrg_rules, graphs, context, threshold){
    prev_ll = 0.0;
    rule_dict = getRuleDict();
    output_dir = std::move(dir);
}

EM::EM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold, std::string dir, int timeout_seconds) : EMBase(shrg_rules, graphs, context, threshold){
    prev_ll = 0.0;
    rule_dict = getRuleDict();
    output_dir = std::move(dir);
    time_out_in_seconds = timeout_seconds;
}

EM::EM(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context, double threshold, std::string dir, int timeout_seconds, const std::unordered_set<std::string>& skip_graphs) : EMBase(shrg_rules, graphs, context, threshold){
    prev_ll = 0.0;
    rule_dict = getRuleDict();
    output_dir = std::move(dir);
    time_out_in_seconds = timeout_seconds;
    skip_graphs_ = skip_graphs;
}

void EM::initializeWeights() {
    setInitialWeights(rule_dict);
}

// ============================================================================
// Forest Caching Support
// ============================================================================

void EM::enableCaching(const std::string& cache_dir) {
    if (cache_dir.empty()) {
        caching_enabled_ = false;
        cache_.reset();
        return;
    }

    cache_ = std::make_unique<forest_cache::ForestCache>(cache_dir);
    grammar_hash_ = computeGrammarHash();
    cache_->set_grammar_hash(grammar_hash_);
    caching_enabled_ = true;

    if (verbose_) {
        std::cout << "Forest caching enabled: " << cache_dir
                  << " (grammar hash: " << std::hex << grammar_hash_ << std::dec << ")\n";
    }
}

size_t EM::getCacheHits() const {
    return cache_ ? cache_->cache_hits() : 0;
}

size_t EM::getCacheMisses() const {
    return cache_ ? cache_->cache_misses() : 0;
}

uint32_t EM::computeGrammarHash() const {
    // Hash all rule contents
    std::string content;
    for (const auto* rule : shrg_rules) {
        if (rule) {
            content += std::to_string(rule->label_hash);
            content += ":";
            content += std::to_string(rule->terminal_edges.size());
            content += ";";
        }
    }
    return forest_cache::ForestCache::compute_hash(content);
}

uint32_t EM::computeGraphHash(const EdsGraph& graph) const {
    // Hash graph content
    std::string content = graph.sentence_id;
    content += ":";
    content += std::to_string(graph.nodes.size());
    content += ":";
    content += std::to_string(graph.edges.size());
    // Add edge labels for more specific hash
    for (const auto& edge : graph.edges) {
        content += std::to_string(edge.label);
        content += ",";
    }
    return forest_cache::ForestCache::compute_hash(content);
}

double EM::computeInside(ChartItem *root){
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

void EM::computeOutsideNode(ChartItem *root, NodeLevelPQ &pq){
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
            // assert(is_negative(curr_log_outside));
            curr_log_outside += parent->log_outside_prob;
            // assert(is_negative(curr_log_outside));

            for(auto sib:siblings){
                curr_log_outside += sib->log_inside_prob;
                // assert(is_negative(curr_log_outside));
            }

            log_outside = addLogs(log_outside, curr_log_outside);
            // assert(is_negative(log_outside));
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

void EM::computeOutside(ChartItem *root){
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

bool EM::parentsOutsideReady(ChartItem* node) const {
    if (!node) {
        return true;
    }

    ChartItem* ptr = node;
    do {
        for (const auto& parent_sib : ptr->parents_sib) {
            ChartItem* parent = std::get<0>(parent_sib);
            if (parent && parent->outside_visited_status != VISITED) {
                return false;
            }
        }
        ptr = ptr->next_ptr;
    } while (ptr != node);

    return true;
}

void EM::computeOutsideChainOptimized(ChartItem* root) {
    if (!root) {
        return;
    }

    ChartItem* ptr = root;
    int safety1 = 0;

    do {
        if (!ptr->parents_sib.empty()) {
            double log_outside = ChartItem::log_zero;

            for (const auto& parent_sib : ptr->parents_sib) {
                ChartItem* parent = std::get<0>(parent_sib);
                if (!parent || !parent->rule_ptr) continue;  // Safety check
                const std::vector<ChartItem*>& siblings = std::get<1>(parent_sib);

                double curr_log_outside = parent->rule_ptr->log_rule_weight + parent->log_outside_prob;
                for (ChartItem* sibling : siblings) {
                    if (sibling) {  // Safety check
                        curr_log_outside += sibling->log_inside_prob;
                    }
                }

                log_outside = addLogs(log_outside, curr_log_outside);
            }

            ptr->log_outside_prob = log_outside;
            ptr->outside_visited_status = VISITED;
        }

        ptr = ptr->next_ptr;
        if (++safety1 > 10000 || !ptr) break;  // Safety check
    } while (ptr != root);

    double root_log_outside = root->log_outside_prob;

    ptr = root;
    int safety2 = 0;
    do {
        if (ptr->outside_visited_status != VISITED) {
            ptr->log_outside_prob = root_log_outside;
            ptr->outside_visited_status = VISITED;
        }

        ptr = ptr->next_ptr;
        if (++safety2 > 10000 || !ptr) break;  // Safety check
    } while (ptr != root);
}

void EM::computeOutside_optimized(ChartItem *root){
    if (!root) {
        return;
    }

    std::unordered_set<ChartItem*> all_items;
    collectAllReachableItems(root, all_items);

    if (all_items.empty()) {
        root->log_outside_prob = 0.0;
        root->outside_visited_status = VISITED;
        return;
    }

    std::unordered_map<ChartItem*, std::size_t> pending_parent_counts;
    pending_parent_counts.reserve(all_items.size());

    std::queue<ChartItem*> ready;

    for (ChartItem* node : all_items) {
        node->log_outside_prob = 0.0;
        node->outside_visited_status = ChartItem::kEmpty;

        std::size_t parent_count = node->parents_sib.size();
        pending_parent_counts[node] = parent_count;
        if (parent_count == 0) {
            ready.push(node);
        }
    }

    while (!ready.empty()) {
        ChartItem* node = ready.front();
        ready.pop();

        if (!node || node->outside_visited_status == VISITED) {
            continue;
        }

        computeOutsideChainOptimized(node);

        ChartItem* ptr = node;
        do {
            pending_parent_counts[ptr] = 0;
            for (ChartItem* child : ptr->children) {
                if (!child) {
                    continue;
                }

                auto it = pending_parent_counts.find(child);
                if (it == pending_parent_counts.end()) {
                    continue;
                }

                if (it->second > 0) {
                    --(it->second);
                }

                if (it->second == 0) {
                    ready.push(child);
                }
            }
            ptr = ptr->next_ptr;
        } while (ptr && ptr != node);
    }

    for (ChartItem* node : all_items) {
        if (node->outside_visited_status != VISITED) {
            computeOutsideChainOptimized(node);
        }
    }
}

// ============================================================================
// computeOutsideFixed: O(n) alternative to computeOutside
// Uses topological ordering to avoid exponential child-pushing
// ============================================================================
// =============================================================================
// computeOutsideFixed: Optimized O(n log n) version of computeOutside
//
// The original has O(chain^depth * n log n) complexity because it pushes the
// same child to the priority queue multiple times (once per alternative in
// parent's chain). This version tracks which nodes are already queued.
// =============================================================================
void EM::computeOutsideFixed(ChartItem* root) {
    if (!root) return;

    NodeLevelPQ pq;
    std::unordered_set<ChartItem*> in_queue;  // Track what's already in the queue

    // Initialize root chain to outside = 0.0 (same as original)
    ChartItem* ptr = root;
    do {
        ptr->log_outside_prob = 0.0;
        ptr = ptr->next_ptr;
    } while (ptr && ptr != root);

    // Add root to queue
    pq.push(root);
    in_queue.insert(root);

    while (!pq.empty()) {
        ChartItem* node = pq.top();
        pq.pop();

        // Process the entire chain (same logic as original computeOutsideNode)
        ptr = node;
        do {
            // Skip if already visited
            if (ptr->outside_visited_status == VISITED) {
                ptr = ptr->next_ptr;
                continue;
            }

            // Skip items with no parents - this preserves root's initialized 0.0
            // (This is the key fix - without this, root gets overwritten to -inf)
            if (ptr->parents_sib.empty()) {
                ptr = ptr->next_ptr;
                continue;
            }

            // Compute outside from all parents
            double log_outside = ChartItem::log_zero;
            for (const auto& parent_sib : ptr->parents_sib) {
                ChartItem* parent = std::get<0>(parent_sib);
                if (!parent || !parent->rule_ptr) continue;
                const auto& siblings = std::get<1>(parent_sib);

                double curr = parent->rule_ptr->log_rule_weight + parent->log_outside_prob;
                for (ChartItem* sib : siblings) {
                    if (sib) curr += sib->log_inside_prob;
                }
                log_outside = addLogs(log_outside, curr);
            }
            ptr->log_outside_prob = log_outside;
            ptr->outside_visited_status = VISITED;

            ptr = ptr->next_ptr;
        } while (ptr && ptr != node);

        // Propagate root's outside to unvisited alternatives and push children
        double node_outside = node->log_outside_prob;
        ptr = node;
        do {
            if (ptr->outside_visited_status != VISITED) {
                ptr->log_outside_prob = node_outside;
                ptr->outside_visited_status = VISITED;
            }

            // Push children - but only if not already in queue (THIS IS THE KEY FIX)
            for (ChartItem* child : ptr->children) {
                if (child && !in_queue.count(child)) {
                    pq.push(child);
                    in_queue.insert(child);
                }
            }

            ptr = ptr->next_ptr;
        } while (ptr && ptr != node);
    }
}

// ============================================================================
// Validation: Compare original computeOutside vs computeOutsideFixed
// ============================================================================
bool EM::validateOutsideImplementations(ChartItem* root, double tolerance) {
    if (!root) return true;

    // Collect all reachable items
    std::unordered_set<ChartItem*> all_items;
    collectAllReachableItems(root, all_items);

    if (all_items.empty()) return true;

    // Run original implementation and capture results
    computeOutside(root);

    std::unordered_map<ChartItem*, double> original_values;
    for (ChartItem* item : all_items) {
        original_values[item] = item->log_outside_prob;
    }

    // Reset flags for all items
    for (ChartItem* item : all_items) {
        item->outside_visited_status = ChartItem::kEmpty;
        item->log_outside_prob = ChartItem::log_zero;
    }

    // Run fixed implementation
    computeOutsideFixed(root);

    // Compare results
    bool all_match = true;
    double max_diff = 0.0;
    int mismatch_count = 0;

    int orig_inf_count = 0;
    int fixed_inf_count = 0;

    for (ChartItem* item : all_items) {
        double orig = original_values[item];
        double fixed = item->log_outside_prob;

        if (orig == ChartItem::log_zero) orig_inf_count++;
        if (fixed == ChartItem::log_zero) fixed_inf_count++;

        double diff = std::abs(orig - fixed);

        if (diff > max_diff && diff != std::numeric_limits<double>::infinity()) {
            max_diff = diff;
        }

        if (diff > tolerance) {
            all_match = false;
            mismatch_count++;
            if (mismatch_count <= 5) {  // Only print first 5 mismatches
                std::cerr << "  MISMATCH: orig=" << orig
                          << " fixed=" << fixed
                          << " diff=" << diff << "\n";
            }
        }
    }

    std::cout << "Validation: " << (all_match ? "PASSED" : "FAILED")
              << " max_diff=" << max_diff;
    if (!all_match) {
        std::cout << " (" << mismatch_count << " mismatches, orig_inf=" << orig_inf_count
                  << ", fixed_inf=" << fixed_inf_count << ")";
    }
    std::cout << "\n";

    return all_match;
}

// ============================================================================
// validateFullEMCycle: Compare rule weights after running full EM with both
// outside implementations. This is an end-to-end validation.
// ============================================================================
bool EM::validateFullEMCycle(double tolerance) {
    const int max_iterations = 10;

    std::cout << "\n=== FULL EM CYCLE VALIDATION ===\n";
    std::cout << "Running " << max_iterations << " EM iterations, validating after each...\n";

    // Track which graphs to process (skip pathological ones)
    std::vector<size_t> valid_graph_indices;
    for (size_t i = 0; i < graphs.size(); i++) {
        EdsGraph& graph = graphs[i];
        auto code = context->Parse(graph);
        if (code != ParserError::kNone) continue;

        ChartItem* root = context->parser->Result();
        addParentPointerOptimized(root, 0);
        addRulePointer(root);

        GraphMetrics metrics;
        computeForestMetrics(root, metrics);
        if (metrics.forest_size <= 300 && metrics.max_chain_length <= 15) {
            valid_graph_indices.push_back(i);
        }
    }
    std::cout << "Processing " << valid_graph_indices.size() << " graphs (skipping pathological cases)\n\n";

    // Initialize weights
    setInitialWeights(rule_dict);

    // Save initial weights
    std::vector<double> init_weights(shrg_rules.size());
    for (size_t i = 0; i < shrg_rules.size(); i++) {
        init_weights[i] = shrg_rules[i]->log_rule_weight;
    }

    // Storage for weight history from both implementations
    std::vector<std::vector<double>> original_weight_history(max_iterations);
    std::vector<double> original_ll_history(max_iterations);

    // === Run EM with ORIGINAL computeOutside ===
    std::cout << "Running EM with original computeOutside...\n";
    for (int iter = 0; iter < max_iterations; iter++) {
        // Reset counts
        for (auto rule : shrg_rules) {
            rule->log_count = ChartItem::log_zero;
        }

        double ll = 0.0;
        for (size_t idx : valid_graph_indices) {
            EdsGraph& graph = graphs[idx];
            auto code = context->Parse(graph);
            if (code != ParserError::kNone) continue;

            ChartItem* root = context->parser->Result();
            addParentPointerOptimized(root, 0);
            addRulePointer(root);

            double pw = computeInside(root);
            computeOutside(root);  // ORIGINAL
            computeExpectedCount(root, pw);
            ll += pw;
        }

        // Update weights
        updateEM();
        original_ll_history[iter] = ll;

        // Save weights after this iteration
        original_weight_history[iter].resize(shrg_rules.size());
        for (size_t i = 0; i < shrg_rules.size(); i++) {
            original_weight_history[iter][i] = shrg_rules[i]->log_rule_weight;
        }

        std::cout << "  iteration " << iter << ": ll=" << ll << "\n";
    }

    // === Reset and run EM with OPTIMIZED computeOutsideFixed ===
    std::cout << "\nRunning EM with optimized computeOutsideFixed...\n";

    // Restore initial weights
    for (size_t i = 0; i < shrg_rules.size(); i++) {
        shrg_rules[i]->log_rule_weight = init_weights[i];
    }

    bool all_iterations_match = true;
    for (int iter = 0; iter < max_iterations; iter++) {
        // Reset counts
        for (auto rule : shrg_rules) {
            rule->log_count = ChartItem::log_zero;
        }

        double ll = 0.0;
        for (size_t idx : valid_graph_indices) {
            EdsGraph& graph = graphs[idx];
            auto code = context->Parse(graph);
            if (code != ParserError::kNone) continue;

            ChartItem* root = context->parser->Result();
            addParentPointerOptimized(root, 0);
            addRulePointer(root);

            double pw = computeInside(root);
            computeOutsideFixed(root);  // OPTIMIZED
            computeExpectedCount(root, pw);
            ll += pw;
        }

        // Update weights
        updateEM();

        // Compare log likelihood
        double ll_diff = std::abs(original_ll_history[iter] - ll);

        // Compare weights
        double max_weight_diff = 0.0;
        int weight_mismatches = 0;
        for (size_t i = 0; i < shrg_rules.size(); i++) {
            double orig = original_weight_history[iter][i];
            double fixed = shrg_rules[i]->log_rule_weight;

            bool orig_inf = (orig == ChartItem::log_zero);
            bool fixed_inf = (fixed == ChartItem::log_zero);

            if (orig_inf && fixed_inf) continue;
            if (orig_inf != fixed_inf) {
                weight_mismatches++;
                continue;
            }

            double diff = std::abs(orig - fixed);
            if (diff > max_weight_diff) max_weight_diff = diff;
            if (diff > tolerance) weight_mismatches++;
        }

        bool iter_match = (ll_diff < tolerance) && (weight_mismatches == 0);
        if (!iter_match) all_iterations_match = false;

        std::cout << "  iteration " << iter << ": ll=" << ll
                  << " (diff=" << ll_diff << ")"
                  << " weights_max_diff=" << max_weight_diff
                  << (iter_match ? " MATCH" : " MISMATCH") << "\n";
    }

    std::cout << "\n=== EM CYCLE VALIDATION RESULT ===\n";
    if (all_iterations_match) {
        std::cout << "All " << max_iterations << " iterations MATCH!\n";
    } else {
        std::cout << "WARNING: Some iterations have mismatches.\n";
    }

    return all_iterations_match;
}

// ============================================================================
// runValidation: Test mode to validate both implementations on first N graphs
// ============================================================================
void EM::runValidation() {
    std::cout << "=== VALIDATION MODE ===\n";
    std::cout << "Comparing computeOutside vs computeOutsideFixed\n\n";

    int validated = 0;
    int failed = 0;
    int skipped = 0;
    // Limit to 30 graphs to avoid hitting pathological cases that crash original computeOutside
    size_t total = std::min((size_t)30, graphs.size());

    for (size_t i = 0; i < total; i++) {
        EdsGraph& graph = graphs[i];
        std::cout << "Validating " << graph.sentence_id << " (" << (i+1) << "/" << total << ")... " << std::flush;

        auto code = context->Parse(graph);
        if (code != ParserError::kNone) {
            std::cout << "SKIP (parse failed)\n";
            skipped++;
            continue;
        }

        ChartItem* root = context->parser->Result();
        addParentPointerOptimized(root, 0);
        addRulePointer(root);

        // Check forest size and chain length - skip if too large (would crash original implementation)
        GraphMetrics metrics;
        computeForestMetrics(root, metrics);
        if (metrics.forest_size > 300 || metrics.max_chain_length > 15) {
            std::cout << "SKIP (forest=" << metrics.forest_size
                      << ", chain=" << metrics.max_chain_length << ")\n";
            skipped++;
            continue;
        }

        // Must compute inside first (outside depends on inside values)
        computeInside(root);

        // Validate outside implementations
        if (validateOutsideImplementations(root)) {
            validated++;
        } else {
            failed++;
        }
    }

    std::cout << "\n=== OUTSIDE VALIDATION SUMMARY ===\n";
    std::cout << "Passed:  " << validated << "/" << total << "\n";
    std::cout << "Failed:  " << failed << "/" << total << "\n";
    std::cout << "Skipped: " << skipped << "/" << total << "\n";

    if (failed == 0) {
        std::cout << "\nOutside validation passed!\n";
    } else {
        std::cout << "\nWARNING: Outside validation failed. Do not use computeOutsideFixed yet.\n";
        return;
    }

    // Also validate full EM cycle (rule weights after update)
    bool em_valid = validateFullEMCycle();

    std::cout << "\n=== FINAL VALIDATION RESULT ===\n";
    if (failed == 0 && em_valid) {
        std::cout << "All validations passed! Safe to switch to computeOutsideFixed.\n";
    } else {
        std::cout << "WARNING: Some validations failed.\n";
    }
}

void writeStringsToFile(const std::vector<std::string>& strings, const std::string& filename) {
    std::ofstream outFile(filename); // Open the file for writing

    if (!outFile) { // Check if the file was opened successfully
        std::cerr << "Error: Could not open the file " << filename << " for writing." << std::endl;
        return;
    }

    for (const auto& str : strings) { // Iterate over the vector of strings
        outFile << str << std::endl; // Write each string followed by a newline
    }

    outFile.close(); // Close the file
}

//void EM::run() {
//    std::cout << "Training Time~ \n";
//    clock_t t1,t2;
//    unsigned long training_size = graphs.size();
//    int iteration = 0;
//    ll = 0;
//    setInitialWeights(rule_dict);
//    std::vector<double> history[shrg_rules.size()];
//    std::vector<double> history_graph_ll[training_size];
//    for(int i = 0; i < shrg_rules.size(); i++){
//        history[i].push_back(shrg_rules[i]->log_rule_weight);
//    }
//    EM_EVALUATE eval = EM_EVALUATE(this);
//    EVALUATE_TREE tree_eval = EVALUATE_TREE(this);
//
//    std::vector<double> lls;
//    std::vector<double> bleus;
//    do{
//        prev_ll = ll;
//        ll = 0;
//        t1 = clock();
//        for(int i = 0; i < training_size; i++){
////            if(i % 200 == 0){
////                std::cout << i << "\n";
////            }
//            std::cout << i << "\n";
//            EdsGraph graph = graphs[i];
//            auto code = context->Parse(graph);
//            if(code == ParserError::kNone) {
//                ChartItem *root = context->parser->Result();
//                addParentPointerOptimized(root, 0);
//                addRulePointer(root);
//
//                double pw = computeInside(root);
//                computeOutside(root);
//                computeExpectedCount(root, pw);
//
////                ll = addLogs(ll, pw);
//                ll += pw;
//                history_graph_ll[i].push_back(pw);
//            }
//        }
//        updateEM();
//        for(int i = 0; i < shrg_rules.size(); i++){
//            history[i].push_back(shrg_rules[i]->log_rule_weight);
//        }
//        std::pair<double, double> p = eval.f1_and_bleu();
//        std::vector<std::string> sentences = eval.getSentences();
//        double bleu = eval.bleu();
//        bleus.push_back(bleu);
//        double tree_score = tree_eval.evaluateAll();
//        clearRuleCount();
//        t2 = clock();
//        double time_diff = (double)(t2 - t1)/CLOCKS_PER_SEC;
//        std::cout << "iteration: " << iteration << "\nlog likelihood: " << ll << ", bleu: " << p.first << ", f1: " << p.second;
//
//        std::cout << ", in " << time_diff << " seconds \n\n";
//        lls.push_back(ll);
//        writeStringsToFile(sentences, "~/Desktop/sentences_"+std::to_string
//                                          (iteration));
//        if(! (output_dir == "N")){
//            writeHistoryToDir(output_dir, history, shrg_rules.size());
//            writeGraphLLToDir(output_dir, history_graph_ll, training_size);
//            writeLLToDir(output_dir, lls, iteration);
//        }
//
//        iteration++;
//    }while(!converged());
//
//
//}

inline std::string to_string(ParserError error) {
    switch (error) {
    case ParserError::kNone:
        return "None";
    case ParserError::kNoResult:
        return "NoResult";
    case ParserError::kOutOfMemory:
        return "OutOfMemory";
    case ParserError::kTooLarge:
        return "TooLarge";
    case ParserError::kUnInitialized:
        return "UnInitialized";
    case ParserError::kUnknown:
        return "Unknown";
    default:
        return "Invalid";
    }
}

void EM::run() {
    std::cout << "Training with cached derivation forests...\n";
    clock_t t1, t2;
    unsigned long training_size = graphs.size();

    // Clear any previous profiling data
    if (profiling_enabled_) {
        graph_metrics_.clear();
        graph_metrics_.reserve(training_size);
    }

    // ========== Phase 1: Parse all graphs ONCE and cache derivation forests ==========
    std::cout << "Phase 1: Parsing and caching derivation forests...\n";
    t1 = clock();

    struct CachedForest {
        ChartItem* root;
        std::string sentence_id;
        int original_index;
        size_t metrics_index;  // Index into graph_metrics_ for this forest
    };
    std::vector<CachedForest> cached_forests;
    cached_forests.reserve(training_size);

    size_t cache_hit_count = 0;
    size_t cache_miss_count = 0;

    for (int i = 0; i < training_size; i++) {
        EdsGraph& graph = graphs[i];

        // Skip graphs in the skip list
        if (!skip_graphs_.empty() && skip_graphs_.count(graph.sentence_id)) {
            if (verbose_) {
                std::cout << "\r[parsing] SKIPPING " << graph.sentence_id
                          << " (" << (i + 1) << "/" << training_size << ")" << std::flush;
            }
            continue;
        }

        if (verbose_) {
            std::cout << "\r[parsing] " << graph.sentence_id
                      << " (" << (i + 1) << "/" << training_size << ")" << std::flush;
        }

        // Start profiling for this graph
        GraphMetrics metrics;
        if (profiling_enabled_) {
            metrics.sentence_id = graph.sentence_id;
            metrics.node_count = graph.nodes.size();
            metrics.edge_count = graph.edges.size();
        }

        // Try to load from cache first
        ChartItem* persistent_root = nullptr;
        uint32_t graph_hash = 0;

        if (caching_enabled_ && cache_) {
            graph_hash = computeGraphHash(graph);
            persistent_root = cache_->load(graph.sentence_id, graph_hash, persistent_pool_);
            if (persistent_root) {
                cache_hit_count++;
                // Re-link rule pointers on the loaded forest
                addRulePointer(persistent_root);

                if (profiling_enabled_) {
                    computeForestMetrics(persistent_root, metrics);
                }

                size_t metrics_idx = graph_metrics_.size();
                if (profiling_enabled_) {
                    graph_metrics_.push_back(metrics);
                }

                cached_forests.push_back({persistent_root, graph.sentence_id, i, metrics_idx});
                continue;  // Skip parsing, use cached forest
            }
            cache_miss_count++;
        }

        auto parse_start = std::chrono::high_resolution_clock::now();
        auto code = context->Parse(graph);
        auto parse_end = std::chrono::high_resolution_clock::now();

        if (profiling_enabled_) {
            metrics.parse_time_ms = std::chrono::duration<double, std::milli>(parse_end - parse_start).count();
        }

        if (code == ParserError::kNone) {
            ChartItem* root = context->parser->Result();
            addParentPointerOptimized(root, 0);
            addRulePointer(root);

            // Deep copy to persistent storage before parser clears its pool
            auto deep_copy_start = std::chrono::high_resolution_clock::now();
            persistent_root = deepCopyDerivationForest(root, persistent_pool_);
            auto deep_copy_end = std::chrono::high_resolution_clock::now();

            if (profiling_enabled_) {
                metrics.deep_copy_time_ms = std::chrono::duration<double, std::milli>(deep_copy_end - deep_copy_start).count();
            }

            // Re-link rule pointers on the persistent copy
            addRulePointer(persistent_root);

            // Save to cache
            if (caching_enabled_ && cache_ && persistent_root) {
                if (graph_hash == 0) {
                    graph_hash = computeGraphHash(graph);
                }
                cache_->save(graph.sentence_id, graph_hash, persistent_root);
            }

            // Compute forest metrics for profiling
            if (profiling_enabled_) {
                computeForestMetrics(persistent_root, metrics);
            }

            size_t metrics_idx = graph_metrics_.size();
            if (profiling_enabled_) {
                graph_metrics_.push_back(metrics);
            }

            cached_forests.push_back({persistent_root, graph.sentence_id, i, metrics_idx});
        } else if (profiling_enabled_) {
            // Still record metrics for failed parses
            graph_metrics_.push_back(metrics);
        }
    }

    t2 = clock();
    double parse_time = (double)(t2 - t1) / CLOCKS_PER_SEC;
    if (verbose_) {
        std::cout << "\nParsing complete: " << cached_forests.size() << " forests cached in "
                  << parse_time << " seconds";
        if (caching_enabled_) {
            std::cout << " (cache hits: " << cache_hit_count
                      << ", misses: " << cache_miss_count << ")";
        }
        std::cout << "\n\n";
    }

    // Write parse-phase metrics if profiling enabled
    if (profiling_enabled_ && !(output_dir == "N")) {
        writeMetricsToCSV(output_dir + "parse_metrics.csv");
        if (verbose_) {
            std::cout << "Parse metrics written to " << output_dir << "parse_metrics.csv\n";
        }
    }

    // ========== Phase 2: Run EM iterations on cached forests ==========
    if (verbose_) {
        std::cout << "Phase 2: Running EM iterations...\n";
    }

    // Scale convergence threshold by number of cached forests
    // (threshold is per-graph, e.g., 0.01 means converge when improvement < 0.01 * num_graphs)
    double scaled_threshold = threshold * cached_forests.size();
    if (verbose_) {
        std::cout << "  (convergence threshold: " << std::fixed << std::setprecision(2)
                  << scaled_threshold << " = " << threshold << " * "
                  << cached_forests.size() << " graphs)\n";
    }

    int iteration = 0;
    ll = 0;
    setInitialWeights(rule_dict);

    // Use vectors instead of VLAs
    std::vector<std::vector<double>> history(shrg_rules.size());
    std::vector<std::vector<double>> history_graph_ll(training_size);

    for (size_t i = 0; i < shrg_rules.size(); i++) {
        history[i].push_back(shrg_rules[i]->log_rule_weight);
    }

    std::vector<double> lls;
    std::vector<double> times;

    do {
        prev_ll = ll;
        ll = 0;
        t1 = clock();

        // Reset visited flags for all cached forests (with timing if profiling)
        for (size_t i = 0; i < cached_forests.size(); i++) {
            auto& cf = cached_forests[i];

            if (profiling_enabled_ && cf.metrics_index < graph_metrics_.size()) {
                auto reset_start = std::chrono::high_resolution_clock::now();
                resetVisitedFlags(cf.root);
                auto reset_end = std::chrono::high_resolution_clock::now();
                // Accumulate across iterations
                graph_metrics_[cf.metrics_index].reset_flags_time_ms +=
                    std::chrono::duration<double, std::milli>(reset_end - reset_start).count();
            } else {
                resetVisitedFlags(cf.root);
            }
        }

        // Process all cached forests
        for (size_t i = 0; i < cached_forests.size(); i++) {
            auto& cf = cached_forests[i];

            if (verbose_) {
                std::cout << "\r[iter " << iteration << "] " << cf.sentence_id
                          << " (" << (i + 1) << "/" << cached_forests.size() << ")" << std::flush;
            }

            if (profiling_enabled_ && cf.metrics_index < graph_metrics_.size()) {
                auto& metrics = graph_metrics_[cf.metrics_index];

                auto inside_start = std::chrono::high_resolution_clock::now();
                double pw = computeInside(cf.root);
                auto inside_end = std::chrono::high_resolution_clock::now();
                metrics.inside_time_ms += std::chrono::duration<double, std::milli>(inside_end - inside_start).count();

                auto outside_start = std::chrono::high_resolution_clock::now();
                computeOutsideFixed(cf.root);  // Use optimized O(n log n) version
                auto outside_end = std::chrono::high_resolution_clock::now();
                metrics.outside_time_ms += std::chrono::duration<double, std::milli>(outside_end - outside_start).count();

                auto expected_start = std::chrono::high_resolution_clock::now();
                computeExpectedCount(cf.root, pw);
                auto expected_end = std::chrono::high_resolution_clock::now();
                metrics.expected_count_time_ms += std::chrono::duration<double, std::milli>(expected_end - expected_start).count();

                // Update total EM time (accumulated across iterations)
                metrics.total_em_time_ms = metrics.reset_flags_time_ms + metrics.inside_time_ms +
                                           metrics.outside_time_ms + metrics.expected_count_time_ms;

                ll += pw;
                history_graph_ll[cf.original_index].push_back(pw);
            } else {
                double pw = computeInside(cf.root);
                computeOutsideFixed(cf.root);  // Use optimized O(n log n) version
                computeExpectedCount(cf.root, pw);
                ll += pw;
                history_graph_ll[cf.original_index].push_back(pw);
            }
        }
        if (verbose_) {
            std::cout << std::endl;
        }

        updateEM();

        for (size_t i = 0; i < shrg_rules.size(); i++) {
            history[i].push_back(shrg_rules[i]->log_rule_weight);
        }

        clearRuleCount();
        t2 = clock();
        double time_diff = (double)(t2 - t1) / CLOCKS_PER_SEC;
        times.push_back(time_diff);

        if (verbose_) {
            std::cout << "iteration: " << iteration
                      << "\nlog likelihood: " << ll
                      << ", in " << time_diff << " seconds \n\n";
        }

        lls.push_back(ll);

        if (!(output_dir == "N")) {
            writeHistoryToDir(output_dir, history.data(), shrg_rules.size());
            writeGraphLLToDir(output_dir, history_graph_ll.data(), training_size);
            writeLLToDir(output_dir, lls, iteration);
            writeTimesToDir(output_dir, times, iteration);
        }

        iteration++;
    } while (std::abs(ll - prev_ll) > scaled_threshold);  // Use scaled threshold

    // Write final metrics with EM timing included
    if (profiling_enabled_ && !(output_dir == "N")) {
        writeMetricsToCSV(output_dir + "em_metrics.csv");
        if (verbose_) {
            std::cout << "\nFinal metrics written to " << output_dir << "em_metrics.csv\n";
            printMetricsSummary();
        }
    }

    // Populate member variables for Python binding access
    ll_history_ = lls;
    iteration_times_ = times;
    num_iterations_ = iteration;
    converged_ = (std::abs(ll - prev_ll) <= scaled_threshold);  // Use scaled threshold
    num_cached_forests_ = cached_forests.size();
}

void EM::run_safe() {
    if (verbose_) {
        std::cout << "Training with cached derivation forests (safe mode)...\n";
    }
    clock_t t1, t2;
    unsigned long training_size = graphs.size();
    double initial_mem = getMemoryUsageMB();

    if (profiling_enabled_) {
        graph_metrics_.clear();
        graph_metrics_.reserve(training_size);
    }

    // ========== Phase 1: Parse with fork-based protection ==========
    if (verbose_) {
        std::cout << "Phase 1: Parsing and caching derivation forests...\n";
        std::cout << "  (fork safety: timeout=" << time_out_in_seconds << "s per graph)\n";
    }
    t1 = clock();

    struct CachedForest {
        ChartItem* root;
        std::string sentence_id;
        int original_index;
        size_t metrics_index;
    };
    std::vector<CachedForest> cached_forests;
    cached_forests.reserve(training_size);
    int skipped_count = 0;
    size_t cache_hit_count = 0;
    size_t cache_miss_count = 0;

    for (int i = 0; i < training_size; i++) {
        EdsGraph& graph = graphs[i];

        if (!skip_graphs_.empty() && skip_graphs_.count(graph.sentence_id)) {
            continue;
        }

        // Try to load from cache first (skip fork test if cached)
        uint32_t graph_hash = 0;
        if (caching_enabled_ && cache_) {
            graph_hash = computeGraphHash(graph);
            ChartItem* cached_root = cache_->load(graph.sentence_id, graph_hash, persistent_pool_);
            if (cached_root) {
                cache_hit_count++;
                addRulePointer(cached_root);

                GraphMetrics metrics;
                if (profiling_enabled_) {
                    metrics.sentence_id = graph.sentence_id;
                    metrics.node_count = graph.nodes.size();
                    metrics.edge_count = graph.edges.size();
                    computeForestMetrics(cached_root, metrics);
                }

                size_t metrics_idx = graph_metrics_.size();
                if (profiling_enabled_) {
                    graph_metrics_.push_back(metrics);
                }

                cached_forests.push_back({cached_root, graph.sentence_id, i, metrics_idx});
                continue;  // Skip fork test and parsing
            }
            cache_miss_count++;
        }

        double current_mem = getMemoryUsageMB();
        if (verbose_) {
            std::cout << "\r[parsing] " << graph.sentence_id
                      << " (" << (i + 1) << "/" << training_size << ")"
                      << " [mem: " << std::fixed << std::setprecision(0) << current_mem << "MB"
                      << ", cached: " << cached_forests.size() << "]" << std::flush;
        }

        // Fork a child to test if parsing is safe
        pid_t pid = fork();
        if (pid == 0) {
            // Child: attempt parse AND deep copy with time limit, then exit
            // This tests actual memory usage, not just parse success
            alarm(time_out_in_seconds);
            auto child_code = context->Parse(graph);
            if (child_code == ParserError::kNone) {
                // Also test the memory-intensive deep copy operation
                ChartItem* root = context->parser->Result();
                addParentPointerOptimized(root, 0);
                addRulePointer(root);
                // Create temporary pool and test deep copy
                utils::MemoryPool<ChartItem> test_pool;
                deepCopyDerivationForest(root, test_pool);
            }
            _exit(child_code == ParserError::kNone ? 0 : 1);
        }

        if (pid < 0) {
            // fork failed - fall through and parse directly
            std::cerr << "\nWarning: fork() failed for " << graph.sentence_id << ", parsing directly\n";
        } else {
            // Parent: wait for child result
            int status;
            waitpid(pid, &status, 0);

            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                std::string reason;
                if (WIFSIGNALED(status)) {
                    int sig = WTERMSIG(status);
                    if (sig == SIGKILL) reason = "OOM killed";
                    else if (sig == SIGALRM) reason = "timeout (" + std::to_string(time_out_in_seconds) + "s)";
                    else reason = "signal " + std::to_string(sig);
                } else {
                    reason = "parse failed";
                }
                if (verbose_) {
                    std::cout << "\n  [SKIP] " << graph.sentence_id << " - " << reason << "\n";
                }
                skipped_count++;
                continue;
            }
        }

        // Child succeeded (or fork failed) - parse for real in parent
        GraphMetrics metrics;
        if (profiling_enabled_) {
            metrics.sentence_id = graph.sentence_id;
            metrics.node_count = graph.nodes.size();
            metrics.edge_count = graph.edges.size();
        }

        auto parse_start = std::chrono::high_resolution_clock::now();
        auto code = context->Parse(graph);
        auto parse_end = std::chrono::high_resolution_clock::now();

        if (profiling_enabled_) {
            metrics.parse_time_ms = std::chrono::duration<double, std::milli>(parse_end - parse_start).count();
        }

        if (code == ParserError::kNone) {
            ChartItem* root = context->parser->Result();
            addParentPointerOptimized(root, 0);
            addRulePointer(root);

            auto deep_copy_start = std::chrono::high_resolution_clock::now();
            ChartItem* persistent_root = deepCopyDerivationForest(root, persistent_pool_);
            auto deep_copy_end = std::chrono::high_resolution_clock::now();

            if (profiling_enabled_) {
                metrics.deep_copy_time_ms = std::chrono::duration<double, std::milli>(deep_copy_end - deep_copy_start).count();
            }

            addRulePointer(persistent_root);

            // Save to cache
            if (caching_enabled_ && cache_ && persistent_root) {
                if (graph_hash == 0) {
                    graph_hash = computeGraphHash(graph);
                }
                cache_->save(graph.sentence_id, graph_hash, persistent_root);
            }

            if (profiling_enabled_) {
                computeForestMetrics(persistent_root, metrics);
            }

            size_t metrics_idx = graph_metrics_.size();
            if (profiling_enabled_) {
                graph_metrics_.push_back(metrics);
            }

            cached_forests.push_back({persistent_root, graph.sentence_id, i, metrics_idx});

            // Debug: print memory every 500 forests
            if (verbose_ && cached_forests.size() % 500 == 0) {
                double mem_now = getMemoryUsageMB();
//                std::cout << "\n  [DEBUG] After " << cached_forests.size() << " forests: "
//                          << std::fixed << std::setprecision(1) << mem_now << " MB"
//                          << " (+" << (mem_now - initial_mem) << " MB from start)\n";
            }
        } else if (profiling_enabled_) {
            graph_metrics_.push_back(metrics);
        }
    }

    t2 = clock();
    double parse_time = (double)(t2 - t1) / CLOCKS_PER_SEC;
    double final_mem = getMemoryUsageMB();
    if (verbose_) {
        std::cout << "\nParsing complete: " << cached_forests.size() << " forests cached, "
                  << skipped_count << " skipped (unsafe), in "
                  << parse_time << " seconds";
        if (caching_enabled_) {
            std::cout << " (cache hits: " << cache_hit_count
                      << ", misses: " << cache_miss_count << ")";
        }
        std::cout << "\n";
        std::cout << "  [DEBUG] Final memory: " << std::fixed << std::setprecision(1) << final_mem << " MB"
                  << " (+" << (final_mem - initial_mem) << " MB from start)\n\n";
    }

    if (profiling_enabled_ && !(output_dir == "N")) {
        writeMetricsToCSV(output_dir + "parse_metrics.csv");
    }

    // ========== Phase 2: EM iterations (identical to run()) ==========
    if (verbose_) {
        std::cout << "Phase 2: Running EM iterations...\n";
    }

    // Scale convergence threshold by number of cached forests
    // (threshold is per-graph, e.g., 0.01 means converge when improvement < 0.01 * num_graphs)
    double scaled_threshold = threshold * cached_forests.size();
    if (verbose_) {
        std::cout << "  (convergence threshold: " << std::fixed << std::setprecision(2)
                  << scaled_threshold << " = " << threshold << " * "
                  << cached_forests.size() << " graphs)\n";
    }

    int iteration = 0;
    ll = 0;
    setInitialWeights(rule_dict);

    std::vector<std::vector<double>> history(shrg_rules.size());
    std::vector<std::vector<double>> history_graph_ll(training_size);

    for (size_t i = 0; i < shrg_rules.size(); i++) {
        history[i].push_back(shrg_rules[i]->log_rule_weight);
    }

    std::vector<double> lls;
    std::vector<double> times;

    do {
        prev_ll = ll;
        ll = 0;
        t1 = clock();

        for (size_t i = 0; i < cached_forests.size(); i++) {
            auto& cf = cached_forests[i];
            resetVisitedFlags(cf.root);
        }

        for (size_t i = 0; i < cached_forests.size(); i++) {
            auto& cf = cached_forests[i];

            if (verbose_) {
                std::cout << "\r[iter " << iteration << "] " << cf.sentence_id
                          << " (" << (i + 1) << "/" << cached_forests.size() << ")" << std::flush;
            }

            double pw = computeInside(cf.root);
            computeOutsideFixed(cf.root);
            computeExpectedCount(cf.root, pw);
            ll += pw;
            history_graph_ll[cf.original_index].push_back(pw);
        }
        if (verbose_) {
            std::cout << std::endl;
        }

        updateEM();

        for (size_t i = 0; i < shrg_rules.size(); i++) {
            history[i].push_back(shrg_rules[i]->log_rule_weight);
        }

        clearRuleCount();
        t2 = clock();
        double time_diff = (double)(t2 - t1) / CLOCKS_PER_SEC;
        times.push_back(time_diff);

        if (verbose_) {
            std::cout << "iteration: " << iteration
                      << "\nlog likelihood: " << ll
                      << ", in " << time_diff << " seconds \n\n";
        }

        lls.push_back(ll);

        if (!(output_dir == "N")) {
            writeHistoryToDir(output_dir, history.data(), shrg_rules.size());
            writeGraphLLToDir(output_dir, history_graph_ll.data(), training_size);
            writeLLToDir(output_dir, lls, iteration);
            writeTimesToDir(output_dir, times, iteration);
        }

        iteration++;
    } while (std::abs(ll - prev_ll) > scaled_threshold);  // Use scaled threshold

    if (profiling_enabled_ && !(output_dir == "N")) {
        writeMetricsToCSV(output_dir + "em_metrics.csv");
        if (verbose_) {
            printMetricsSummary();
        }
    }

    // Populate member variables for Python binding access
    ll_history_ = lls;
    iteration_times_ = times;
    num_iterations_ = iteration;
    converged_ = (std::abs(ll - prev_ll) <= scaled_threshold);  // Use scaled threshold
    num_cached_forests_ = cached_forests.size();
}

void EM::run_1iter() {
    std::cout << "Training Time~ \n";
    clock_t t1,t2, t3, t4;
    unsigned long training_size = graphs.size();
    int iteration = 0;
    ll = 0;
    setInitialWeights(rule_dict);
    std::vector<double> history[shrg_rules.size()];
    std::vector<double> history_graph_ll[training_size];
    for(int i = 0; i < shrg_rules.size(); i++){
        history[i].push_back(shrg_rules[i]->log_rule_weight);
    }

    std::vector<double> lls;
    std::vector<double> bleus;
    std::vector<double> f1s;
     std::vector<double> times;

        prev_ll = ll;
        ll = 0;
        t1 = clock();

        double time_diff;

        for(int i = 0; i < 100; i++) {
            EdsGraph graph = graphs[i];
            t3=clock();
            auto code = context->Parse(graph);

                if(code == ParserError::kNone) {

                    ChartItem *root = context->parser->Result();
                    addParentPointerOptimized(root, 0);
                    addRulePointer(root);
                    t4=clock();
                    time_diff = t4 - t3;
                    // std::cout << "parsing and computation took: " << time_diff << std::endl;

                    t3 = clock();
                    double pw = computeInside(root);
                    t4 = clock();
                    time_diff = (double)(t4 - t3);
                    // std::cout << "persistent root " << i << " inside: " << time_diff;
                    t3 = clock();
                    computeOutsideFixed(root);  // Use optimized O(n log n) version
                    t4 = clock();
                    time_diff = (double)(t4 - t3);
                    std::cout << ", outside: " << time_diff;
                    t3 = clock();
                    computeExpectedCount(root, pw);
                    t4 = clock();
                    time_diff = (double)(t4 - t3);
                    std::cout << ", Expected: " << time_diff << std::endl;



                    // double pw = computeInside(root);
                    // // std::cout << i << ":" << pw << std::endl;
                    // computeOutside(root);
                    // computeExpectedCount(root, pw);
                    ll += pw;
                    history_graph_ll[i].push_back(pw);

                }

        }

        updateEM();
        for(int i = 0; i < shrg_rules.size(); i++){
            history[i].push_back(shrg_rules[i]->log_rule_weight);
        }

        clearRuleCount();
        t2 = clock();
        time_diff = (double)(t2 - t1)/CLOCKS_PER_SEC;
        times.push_back(time_diff);

        std::cout << "iteration: " << iteration
                  << "\nlog likelihood: " << ll
                  << ", in " << time_diff << " seconds \n\n";

        lls.push_back(ll);

        if(!(output_dir == "N")){
            writeHistoryToDir(output_dir, history, shrg_rules.size());
            writeGraphLLToDir(output_dir, history_graph_ll, training_size);
            writeLLToDir(output_dir, lls, iteration);
            writeTimesToDir(output_dir, times, iteration);
        }

        iteration++;
}

void EM::run_from_saved() {
    std::cout << "Training Time~ \n";
    clock_t t1,t2;
    unsigned long training_size = graphs.size();
    int iteration = 0;
    ll = 0;
    setInitialWeights(rule_dict);
    std::vector<double> history[shrg_rules.size()];
    std::vector<double> history_graph_ll[training_size];
    for(int i = 0; i < shrg_rules.size(); i++){
        history[i].push_back(shrg_rules[i]->log_rule_weight);
    }
//    EM_EVALUATE eval = EM_EVALUATE(this);

    std::vector<double> lls;
    std::vector<double> times;

    for(int i = 0; i < training_size; i++) {
        EdsGraph graph = graphs[i];
        auto code = context->Parse(graph);
        std::cout << to_string(code);
        if (code == ParserError::kNone) {
            ChartItem *root = context->parser->Result();
            addParentPointerOptimized(root, 0);
            addRulePointer(root);
            forests.push_back(root);
            lemmas.push_back(graph.lemma_sequence);
        }
    }

    do{
        prev_ll = ll;
        ll = 0;
        t1 = clock();
        for(int i = 0; i < forests.size(); i++){
//            if(i %200 == 0){
//                std::cout << i << "\n";
//            }
            std::cout << i << "\n";
            ChartItem *root = forests[i];
            double pw = computeInside(root);
            computeOutsideFixed(root);  // Use optimized O(n log n) version
            computeExpectedCount(root, pw);

            std::cout << "bababa";
            std::cout << pw;

            ll += pw;
            history_graph_ll[i].push_back(pw);
        }
        updateEM();
        for(int i = 0; i < shrg_rules.size(); i++){
            history[i].push_back(shrg_rules[i]->log_rule_weight);
        }

//        auto metrics = eval.evaluateAll_fromSaved_noTree();
//        bleus.push_back(metrics.bleu);
        clearRuleCount();
        t2 = clock();
        double time_diff = (double)(t2 - t1)/CLOCKS_PER_SEC;
        times.push_back(time_diff);

        std::cout << "iteration: " << iteration
                  << "\nlog likelihood: " << ll
//                  << ", bleu: " << metrics.bleu
//                  << ", f1: " << metrics.f1
                  << ", in " << time_diff << " seconds \n\n";

        lls.push_back(ll);
//        writeStringsToFile(metrics.generated_sentences,
//                           output_dir + "sentences_" + std::to_string(iteration));

        if(!(output_dir == "N")){
            writeHistoryToDir(output_dir, history, shrg_rules.size());
            writeGraphLLToDir(output_dir, history_graph_ll, training_size);
            writeLLToDir(output_dir, lls, iteration);
            writeTimesToDir(output_dir, times, iteration);
        }

        iteration++;
    }while(!converged());
}

// Add helper function for writing tree scores
void EM::writeTreeScoresToDir(const std::string& filename,
                              const std::vector<double>& scores) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    for (double score : scores) {
        outFile << score << std::endl;
    }
    outFile.close();
}

bool EM::converged() const {
    return std::abs(ll - prev_ll) <= threshold;
}

void EM::computeExpectedCount(ChartItem *root, double pw) {
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



void EM::updateEM() {
    LabelCount total_count;
    LabelToRule::iterator it;

    for(it = rule_dict.begin(); it != rule_dict.end(); it++) {
        RuleVector v = it->second;
        double log_total_count = ChartItem::log_zero;
        for (auto rule : v) {
            log_total_count = addLogs(log_total_count, rule->log_count);
            if(!is_normal_count(log_total_count)){
                std::cout << "count";
            }
//            assert(is_normal_count(log_total_count));
        }
        total_count[it->first] = log_total_count;
    }
    max_change = 0.0;
    for(int i = 0; i < shrg_rules.size(); i++) {
        auto rule = shrg_rules[i];
        LabelHash l = rule->label_hash;
        double new_phi;

        // special case 1: the rule doesn't appear
        if (rule->log_count == ChartItem::log_zero) {
            // if the rule is the only one with the label
            if (rule_dict[l].size() == 1) {
                new_phi = 0.0;
            } else {
                new_phi = ChartItem::log_zero;
            }
        } else {
            new_phi = rule->log_count - total_count[l];
//            assert(is_negative(new_phi));
        }
        double change = abs(rule->log_rule_weight - new_phi);
        if(change > max_change){
            max_change = change;
            max_change_ind = i;
        }
        rule->log_rule_weight = new_phi;
    }
}

LabelToRule EM::getRuleDict(){
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

// Deep copy functions for persistent derivation forests
ChartItem* EM::deepCopyChartItem(ChartItem* original,
                                 std::unordered_map<ChartItem*, ChartItem*>& copied_map,
                                 utils::MemoryPool<ChartItem>& persistent_pool) {
    if (!original) return nullptr;

    // Check if already copied (handles cycles and shared nodes)
    auto it = copied_map.find(original);
    if (it != copied_map.end()) {
        return it->second;
    }

    // Create a copy of the original item in persistent memory
    ChartItem* copy = persistent_pool.Push();

    // Copy all essential data
    copy->attrs_ptr = original->attrs_ptr;
    copy->edge_set = original->edge_set;
    copy->boundary_node_mapping = original->boundary_node_mapping;
    copy->level = original->level;
    copy->score = original->score;
    copy->status = original->status;

    // Copy EM-related probabilities and counts
    copy->log_inside_prob = original->log_inside_prob;
    copy->log_outside_prob = original->log_outside_prob;
    copy->log_sent_rule_count = original->log_sent_rule_count;
    copy->log_inside_count = original->log_inside_count;

    // Don't copy rule pointers - they will be set by addRulePointer() after deep copy
    copy->shrg_index = -1;  // Will be set by addRulePointer()
    copy->rule_ptr = nullptr;  // Will be set by addRulePointer()

    // Copy status flags (reset to unvisited state for fresh EM iteration)
    copy->inside_visited_status = ChartItem::kEmpty;
    copy->outside_visited_status = ChartItem::kEmpty;
    copy->count_visited_status = ChartItem::kEmpty;
    copy->child_visited_status = ChartItem::kEmpty;
    copy->update_status = ChartItem::kEmpty;
    copy->rule_visited = ChartItem::kEmpty;

    // Copy derivation scoring fields
    copy->em_greedy_score = original->em_greedy_score;
    copy->em_greedy_deriv = original->em_greedy_deriv;
    copy->em_inside_score = original->em_inside_score;
    copy->em_inside_deriv = original->em_inside_deriv;
    copy->count_greedy_score = original->count_greedy_score;
    copy->count_greedy_deriv = original->count_greedy_deriv;
    copy->count_inside_score = original->count_inside_score;
    copy->count_inside_deriv = original->count_inside_deriv;

    // Register the copy early to handle circular references
    copied_map[original] = copy;

    // Initialize pointers to null first
    copy->next_ptr = nullptr;
    copy->left_ptr = nullptr;
    copy->right_ptr = nullptr;

    return copy;
}

void EM::deepCopyDerivationRelations(ChartItem* original,
                                     ChartItem* copy,
                                     std::unordered_map<ChartItem*, ChartItem*>& copied_map,
                                     utils::MemoryPool<ChartItem>& persistent_pool) {
    // Copy children relationships
    copy->children.clear();
    copy->children.reserve(original->children.size());
    for (ChartItem* child : original->children) {
        ChartItem* child_copy = deepCopyChartItem(child, copied_map, persistent_pool);
        copy->children.push_back(child_copy);
    }

    // Copy parent-sibling relationships
    copy->parents_sib.clear();
    copy->parents_sib.reserve(original->parents_sib.size());
    for (const auto& parent_sib_tuple : original->parents_sib) {
        ChartItem* parent = std::get<0>(parent_sib_tuple);
        const std::vector<ChartItem*>& siblings = std::get<1>(parent_sib_tuple);

        ChartItem* parent_copy = deepCopyChartItem(parent, copied_map, persistent_pool);
        std::vector<ChartItem*> siblings_copy;
        siblings_copy.reserve(siblings.size());

        for (ChartItem* sibling : siblings) {
            ChartItem* sibling_copy = deepCopyChartItem(sibling, copied_map, persistent_pool);
            siblings_copy.push_back(sibling_copy);
        }

        copy->parents_sib.emplace_back(parent_copy, std::move(siblings_copy));
    }

    // Copy next_ptr chain (alternative derivations for same subgraph)
    if (original->next_ptr && original->next_ptr != original) {
        copy->next_ptr = deepCopyChartItem(original->next_ptr, copied_map, persistent_pool);
    } else if (original->next_ptr == original) {
        // Self-loop case: mark for later fixup
        copy->next_ptr = copy;
    }

    // Copy left_ptr and right_ptr if they exist (for binary derivations)
    if (original->left_ptr) {
        copy->left_ptr = deepCopyChartItem(original->left_ptr, copied_map, persistent_pool);
    }
    if (original->right_ptr) {
        copy->right_ptr = deepCopyChartItem(original->right_ptr, copied_map, persistent_pool);
    }
}

void EM::collectAllReachableItems(ChartItem* root, std::unordered_set<ChartItem*>& all_items) {
    if (!root || all_items.count(root)) {
        return;
    }

    all_items.insert(root);

    // Follow children
    for (ChartItem* child : root->children) {
        collectAllReachableItems(child, all_items);
    }

    // Follow parent-sibling relationships
    for (const auto& parent_sib_tuple : root->parents_sib) {
        ChartItem* parent = std::get<0>(parent_sib_tuple);
        collectAllReachableItems(parent, all_items);

        for (ChartItem* sibling : std::get<1>(parent_sib_tuple)) {
            collectAllReachableItems(sibling, all_items);
        }
    }

    // Follow next_ptr chain (but avoid infinite loops)
    if (root->next_ptr && root->next_ptr != root) {
        collectAllReachableItems(root->next_ptr, all_items);
    }

    // Follow left_ptr and right_ptr
    if (root->left_ptr) {
        collectAllReachableItems(root->left_ptr, all_items);
    }
    if (root->right_ptr) {
        collectAllReachableItems(root->right_ptr, all_items);
    }
}

ChartItem* EM::deepCopyDerivationForest(ChartItem* root, utils::MemoryPool<ChartItem>& persistent_pool) {
    if (!root) return nullptr;

    // Step 1: Collect all reachable ChartItems from the root
    std::unordered_set<ChartItem*> all_items;
    collectAllReachableItems(root, all_items);

    // std::cout << "Deep copying derivation forest with " << all_items.size() << " items" << std::endl;

    // Step 2: Create copies of all items (structure only, no relationships yet)
    std::unordered_map<ChartItem*, ChartItem*> copied_map;
    for (ChartItem* item : all_items) {
        deepCopyChartItem(item, copied_map, persistent_pool);
    }

    // Step 3: Fix up all relationships between copied items
    for (ChartItem* original : all_items) {
        ChartItem* copy = copied_map[original];
        deepCopyDerivationRelations(original, copy, copied_map, persistent_pool);
    }

    // Return the copied root
    return copied_map[root];
}

void EM::resetVisitedFlags(ChartItem* root) {
    if (!root) return;

    std::unordered_set<ChartItem*> visited;
    resetVisitedFlagsRecursive(root, visited);
}

void EM::resetVisitedFlagsRecursive(ChartItem* item, std::unordered_set<ChartItem*>& visited) {
    if (!item || visited.count(item)) return;

    visited.insert(item);

    // Reset all the critical visited flags
    ChartItem* ptr = item;
    do {
        ptr->inside_visited_status = ChartItem::kEmpty;
        ptr->outside_visited_status = ChartItem::kEmpty;
        ptr->count_visited_status = ChartItem::kEmpty;
        ptr->child_visited_status = ChartItem::kEmpty;
        // Keep rule_visited as VISITED since rule pointers are valid

        ptr = ptr->next_ptr;
    } while (ptr && ptr != item);

    // Recursively reset flags for all reachable items
    for (ChartItem* child : item->children) {
        resetVisitedFlagsRecursive(child, visited);
    }

    for (const auto& parent_sib_tuple : item->parents_sib) {
        resetVisitedFlagsRecursive(std::get<0>(parent_sib_tuple), visited);
        for (ChartItem* sibling : std::get<1>(parent_sib_tuple)) {
            resetVisitedFlagsRecursive(sibling, visited);
        }
    }

    if (item->left_ptr) {
        resetVisitedFlagsRecursive(item->left_ptr, visited);
    }
    if (item->right_ptr) {
        resetVisitedFlagsRecursive(item->right_ptr, visited);
    }
}


// ===================== PROFILING FUNCTIONS =====================

size_t EM::countForestSize(ChartItem* root) {
    if (!root) return 0;
    std::unordered_set<ChartItem*> all_items;
    collectAllReachableItems(root, all_items);
    return all_items.size();
}

void EM::computeForestMetrics(ChartItem* root, GraphMetrics& metrics) {
    if (!root) return;

    std::unordered_set<ChartItem*> all_items;
    collectAllReachableItems(root, all_items);
    metrics.forest_size = all_items.size();

    size_t max_chain = 0;
    size_t max_children = 0;
    size_t max_parents = 0;

    for (ChartItem* item : all_items) {
        // Count chain length
        size_t chain_len = 1;
        ChartItem* ptr = item->next_ptr;
        while (ptr && ptr != item) {
            chain_len++;
            ptr = ptr->next_ptr;
        }
        if (chain_len > max_chain) max_chain = chain_len;

        // Count children
        if (item->children.size() > max_children) {
            max_children = item->children.size();
        }

        // Count parents
        if (item->parents_sib.size() > max_parents) {
            max_parents = item->parents_sib.size();
        }
    }

    metrics.max_chain_length = max_chain;
    metrics.max_children = max_children;
    metrics.max_parents = max_parents;
}

void EM::writeMetricsToCSV(const std::string& filepath) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open metrics file: " << filepath << std::endl;
        return;
    }

    // Write header
    out << "sentence_id,nodes,edges,forest_size,max_chain,max_children,max_parents,"
        << "parse_ms,deep_copy_ms,reset_ms,inside_ms,outside_ms,expected_ms,total_em_ms\n";

    // Write data
    for (const auto& m : graph_metrics_) {
        out << m.sentence_id << ","
            << m.node_count << ","
            << m.edge_count << ","
            << m.forest_size << ","
            << m.max_chain_length << ","
            << m.max_children << ","
            << m.max_parents << ","
            << std::fixed << std::setprecision(2)
            << m.parse_time_ms << ","
            << m.deep_copy_time_ms << ","
            << m.reset_flags_time_ms << ","
            << m.inside_time_ms << ","
            << m.outside_time_ms << ","
            << m.expected_count_time_ms << ","
            << m.total_em_time_ms << "\n";
    }

    out.close();
    std::cout << "Metrics written to: " << filepath << std::endl;
}

void EM::printMetricsSummary() {
    if (graph_metrics_.empty()) {
        std::cout << "No metrics collected.\n";
        return;
    }

    // Find slowest graphs
    std::vector<size_t> indices(graph_metrics_.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort by total_em_time descending
    std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
        return graph_metrics_[a].total_em_time_ms > graph_metrics_[b].total_em_time_ms;
    });

    std::cout << "\n===== PROFILING SUMMARY =====\n";
    std::cout << "Total graphs: " << graph_metrics_.size() << "\n\n";

    // Aggregate stats
    double total_parse = 0, total_inside = 0, total_outside = 0, total_expected = 0;
    size_t max_forest = 0;
    for (const auto& m : graph_metrics_) {
        total_parse += m.parse_time_ms;
        total_inside += m.inside_time_ms;
        total_outside += m.outside_time_ms;
        total_expected += m.expected_count_time_ms;
        if (m.forest_size > max_forest) max_forest = m.forest_size;
    }

    std::cout << "Time breakdown (total ms):\n";
    std::cout << "  Parsing:        " << std::fixed << std::setprecision(0) << total_parse << "\n";
    std::cout << "  Inside:         " << total_inside << "\n";
    std::cout << "  Outside:        " << total_outside << "\n";
    std::cout << "  Expected Count: " << total_expected << "\n";
    std::cout << "  Max forest size: " << max_forest << "\n\n";

    std::cout << "Top 10 slowest graphs (by EM time):\n";
    std::cout << std::setw(20) << "sentence_id" << " | "
              << std::setw(8) << "forest" << " | "
              << std::setw(8) << "chain" << " | "
              << std::setw(8) << "parse" << " | "
              << std::setw(8) << "inside" << " | "
              << std::setw(8) << "outside" << " | "
              << std::setw(8) << "total" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t i = 0; i < std::min(size_t(10), indices.size()); i++) {
        const auto& m = graph_metrics_[indices[i]];
        std::cout << std::setw(20) << m.sentence_id << " | "
                  << std::setw(8) << m.forest_size << " | "
                  << std::setw(8) << m.max_chain_length << " | "
                  << std::setw(8) << std::fixed << std::setprecision(1) << m.parse_time_ms << " | "
                  << std::setw(8) << m.inside_time_ms << " | "
                  << std::setw(8) << m.outside_time_ms << " | "
                  << std::setw(8) << m.total_em_time_ms << "\n";
    }
    std::cout << "=============================\n\n";
}

// void EM::run() {
//     std::vector<ChartItem*> persistent_roots;
//
//     // Parse all graphs and store persistent copies
//      clock_t t1,t2, t3, t4;
//      t1 = clock();
//      auto num_trees = std::min<std::size_t>(1000, graphs.size());
//     for (int i = 0; i < num_trees; i++) {
//         EdsGraph graph = graphs[i];
//         t3 = clock();
//         auto code = context->Parse(graph);
//
//         if (code == ParserError::kNone) {
//             ChartItem* root = context->parser->Result();
//             addParentPointerOptimized(root, 0);
//             addRulePointer(root);
//
//             // Create persistent copy before chart gets cleared
//             ChartItem* persistent_root = deepCopyDerivationForest(root, persistent_pool_);
//
//             // CRITICAL: Re-run addRulePointer to ensure persistent copies point to correct SHRG objects
//             addRulePointer(persistent_root);
//
//             persistent_roots.push_back(persistent_root);
//         }
//         t4 = clock();
//         double time_diff = (double)(t4 - t3) / CLOCKS_PER_SEC;
//         std::cout << "graph " << i << ": " << time_diff << std::endl;
//     }
//      t2 = clock();
//      double time_diff = (double)(t2 - t1)/CLOCKS_PER_SEC;
//      std::cout << "parsing took: " << time_diff << std::endl << std::flush;
//      setInitialWeights(rule_dict);
//      auto alloc_start = clock();
//      unsigned long training_size = graphs.size();
//      std::vector<std::vector<double>> history(shrg_rules.size());
//      std::vector<std::vector<double>> history_graph_ll(training_size);
//      auto alloc_end = clock();
//      std::cout << "Allocation took: " << (double)(alloc_end - alloc_start) / CLOCKS_PER_SEC << std::endl << std::flush;
//      std::cout << "Training Time~ " << std::endl << std::flush;
//
//      // unsigned long training_size = graphs.size();
//      int iteration = 0;
//      ll = 0;
//      // setInitialWeights(rule_dict);
//      // std::vector<double> history[shrg_rules.size()];
//      // std::vector<double> history_graph_ll[training_size];
//      t1 = clock();
//      for(int i = 0; i < shrg_rules.size(); i++){
//          history[i].push_back(shrg_rules[i]->log_rule_weight);
//      }
//     t2 = clock();
//      time_diff = (double)(t2 - t1)/CLOCKS_PER_SEC;
//      std::cout << "Weight Initialization: " << time_diff << std::endl << std::flush;
//      std::vector<double> lls;
//      std::vector<double> bleus;
//      std::vector<double> f1s;
//       std::vector<double> times;
//      // setInitialWeights(rule_dict);
//
//     do {
//         t1 = clock();
//         prev_ll = ll;
//         ll = 0;
//
//         // CRITICAL: Reset all visited flags before each EM iteration
//         t3 = clock();
//         for (ChartItem* persistent_root : persistent_roots) {
//             resetVisitedFlags(persistent_root);
//         }
//         t4 = clock();
//         time_diff = (double)(t4 - t3)/CLOCKS_PER_SEC;
//         std::cout << "reset flags took: " << time_diff << std::endl << std::flush;
//
//         for (int i = 0; i < persistent_roots.size(); i++){
//             auto persistent_root = persistent_roots[i];
//             // if (i % 200 == 0) {
//             //     std::cout << i << std::endl;
//             // }
//         // for (ChartItem* persistent_root : persistent_roots) {
//             // std::cout << "Rule 0 weight: " << shrg_rules[0]->log_rule_weight << std::endl;
//             // std::cout << "Persistent root rule weight: " << persistent_roots[0]->rule_ptr->log_rule_weight << std::endl;
//             t3 = clock();
//             double pw = computeInside(persistent_root);
//             t4 = clock();
//             time_diff = (double)(t4 - t3)/CLOCKS_PER_SEC;
//             std::cout << "persistent root " << i << " inside: " << time_diff;
//             t3 = clock();
//             computeOutside(persistent_root);
//             // computeOutside_optimized(persistent_root);
//             t4 = clock();
//             time_diff = (double)(t4 - t3)/CLOCKS_PER_SEC;
//             std::cout << ", outside: " << time_diff;
//             t3 = clock();
//             computeExpectedCount(persistent_root, pw);
//             t4 = clock();
//             time_diff = (double)(t4 - t3)/CLOCKS_PER_SEC;
//             std::cout << ", Expected: " << time_diff << std::endl;
//             ll += pw;
//             // std::cout << pw << "\n";
//         }
//
//         updateEM();
//         clearRuleCount();
//         for(int i = 0; i < shrg_rules.size(); i++){
//             history[i].push_back(shrg_rules[i]->log_rule_weight);
//         }
//         t2 = clock();
//         double time_diff = (double)(t2 - t1)/CLOCKS_PER_SEC;
//         times.push_back(time_diff);
//
//         std::cout << "iteration: " << iteration
//                   << "\nlog likelihood: " << ll
//                   << ", in " << time_diff << " seconds \n\n";
//
//         lls.push_back(ll);
//
//         if(!(output_dir == "N")){
//             writeHistoryToDir(output_dir, history.data(), shrg_rules.size());
//             writeGraphLLToDir(output_dir, history_graph_ll.data(), training_size);
//             writeLLToDir(output_dir, lls, iteration);
//             writeTimesToDir(output_dir, times, iteration);
//         }
//
//         iteration++;
//
//     } while (!converged());
// }


} // namespace shrg::em

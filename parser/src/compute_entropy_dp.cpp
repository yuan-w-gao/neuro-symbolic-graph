/**
 * @file compute_entropy_dp.cpp
 * @brief Compute derivation entropy using bottom-up DP (no outside pass needed)
 *
 * Usage: compute_entropy_dp <parser_type> <grammar_file> <graph_file> <output_file> <weight_mode>
 *
 * This uses the recursive entropy formula:
 *   H(v) = log Z(v) - sum_a r(a|v) log w(a) + sum_a r(a|v) sum_c H(c)
 *
 * Advantages over inside-outside method:
 *   - Single bottom-up pass (no outside computation needed)
 *   - Naturally handles sharing via memoization
 *   - More numerically stable
 *
 * Weight modes:
 *   uniform              - Equal weights per label group
 *   oracle               - MLE weights from training counts (cfg_rule.score)
 *   <weight_history_path> - Load trained weights from file (uses final iteration)
 */

#include "ambiguity_metrics/ambiguity_metrics.hpp"

// Parser includes
#include "manager.hpp"
#include "graph_parser/parser_base.hpp"
#include "em_framework/em.hpp"
#include "em_framework/em_utils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <limits>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <chrono>

using namespace lexcxg;

// Helper for robust double parsing (handles subnormals, inf, nan)
static inline void trim_inplace(std::string& s) {
    auto issp = [](unsigned char c){ return std::isspace(c); };
    while (!s.empty() && issp(s.front())) s.erase(s.begin());
    while (!s.empty() && issp(s.back()))  s.pop_back();
    if (!s.empty() && s.back() == '\r') s.pop_back();
}

static inline std::string lower_copy(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static inline bool parse_double_robust(const std::string& raw, double& out) {
    std::string s = raw;
    trim_inplace(s);
    if (s.empty()) return false;

    std::string lo = lower_copy(s);
    if (lo == "inf" || lo == "+inf" || lo == "infinity") {
        out = std::numeric_limits<double>::infinity();
        return true;
    }
    if (lo == "-inf" || lo == "-infinity") {
        out = -std::numeric_limits<double>::infinity();
        return true;
    }
    if (lo == "nan") {
        out = std::numeric_limits<double>::quiet_NaN();
        return true;
    }

    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) return false;
    out = v;
    return true;
}

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <parser_type> <grammar_file> <graph_file> <output_file> <weight_mode>\n";
    std::cerr << "\n";
    std::cerr << "Parser types: linear, tree_v1, tree_v2\n";
    std::cerr << "\n";
    std::cerr << "Weight modes:\n";
    std::cerr << "  uniform               - Equal weights per label group\n";
    std::cerr << "  oracle                - MLE weights from training counts\n";
    std::cerr << "  <weight_history_path> - Load trained weights from file\n";
    std::cerr << "\n";
    std::cerr << "Computes derivation entropy using bottom-up DP:\n";
    std::cerr << "  H(v) = log Z(v) - sum_a r(a|v) log w(a) + sum_a r(a|v) sum_c H(c)\n";
    std::cerr << "\n";
    std::cerr << "Output TSV columns:\n";
    std::cerr << "  graph_id, entropy_dp, log_Z, num_or_nodes, time_ms\n";
}

/**
 * @brief Load final weights from weight_history file
 * Format: rule_index,weight_iter0,weight_iter1,...,weight_iterN
 * Returns map from rule_index to final weight (last column)
 */
std::map<int, double> loadFinalWeights(const std::string& filepath) {
    std::map<int, double> weights;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open weight file: " << filepath << "\n";
        return weights;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;

        // First value is rule index
        if (!std::getline(ss, token, ',')) continue;
        int rule_index = std::stoi(token);

        // Read all remaining values, keeping the last one
        double final_weight = 0.0;
        while (std::getline(ss, token, ',')) {
            double val;
            if (parse_double_robust(token, val)) {
                final_weight = val;
            }
        }

        weights[rule_index] = final_weight;
    }

    return weights;
}

/**
 * @brief Apply weights to SHRG rules based on mode
 */
void applyWeights(std::vector<shrg::SHRG*>& shrg_rules,
                  const std::string& weight_mode,
                  shrg::em::EM& em_helper) {

    if (weight_mode == "uniform") {
        // Use uniform weights (already handled by initializeWeights)
        em_helper.initializeWeights();
        std::cout << "Using uniform weights\n";
    }
    else if (weight_mode == "oracle") {
        // Use MLE weights from training counts (cfg_rule.score)
        for (shrg::SHRG* rule : shrg_rules) {
            if (rule && !rule->cfg_rules.empty()) {
                // Use the score from the best CFG rule (already log probability)
                rule->log_rule_weight = rule->best_cfg_ptr->score;
            }
        }
        std::cout << "Using oracle weights (MLE from counts)\n";
    }
    else {
        // Load trained weights from file
        std::map<int, double> weights = loadFinalWeights(weight_mode);
        if (weights.empty()) {
            std::cerr << "Warning: No weights loaded, falling back to uniform\n";
            em_helper.initializeWeights();
        } else {
            int applied = 0;
            for (const auto& [idx, weight] : weights) {
                if (idx >= 0 && idx < static_cast<int>(shrg_rules.size()) && shrg_rules[idx] != nullptr) {
                    shrg_rules[idx]->log_rule_weight = weight;
                    applied++;
                }
            }
            std::cout << "Loaded " << weights.size() << " weights, applied " << applied << " from " << weight_mode << "\n";
        }
    }
}

/**
 * @brief Result structure for DP entropy computation
 */
struct EntropyDPResult {
    int graph_id;
    double entropy;
    double log_Z;
    int num_or_nodes;
    double time_ms;
    bool success;
};

void writeTsvOutput(
    const std::string& output_file,
    const std::vector<EntropyDPResult>& results
) {
    std::ofstream out(output_file);
    if (!out) {
        std::cerr << "Error: Cannot open output file: " << output_file << "\n";
        return;
    }

    // Header
    out << "graph_id\tentropy_dp\tlog_Z\tnum_or_nodes\ttime_ms\n";

    double sum_entropy = 0, sum_log_Z = 0, sum_time = 0;
    int valid_count = 0;

    for (const auto& r : results) {
        out << r.graph_id << "\t"
            << std::fixed << std::setprecision(6)
            << r.entropy << "\t"
            << r.log_Z << "\t"
            << r.num_or_nodes << "\t"
            << std::setprecision(3) << r.time_ms << "\n";

        if (r.success) {
            sum_entropy += r.entropy;
            sum_log_Z += r.log_Z;
            sum_time += r.time_ms;
            valid_count++;
        }
    }

    out.close();

    // Print summary to stdout
    if (valid_count > 0) {
        std::cout << "\n=== Summary ===\n";
        std::cout << "Total graphs: " << results.size() << "\n";
        std::cout << "Successfully parsed: " << valid_count << "\n";
        std::cout << "Average entropy: " << std::fixed << std::setprecision(4) << (sum_entropy / valid_count) << "\n";
        std::cout << "Average log_Z: " << std::setprecision(4) << (sum_log_Z / valid_count) << "\n";
        std::cout << "Average time: " << std::setprecision(3) << (sum_time / valid_count) << " ms\n";
        std::cout << "Total time: " << std::setprecision(1) << sum_time << " ms\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        printUsage(argv[0]);
        return 1;
    }

    std::string parser_type = argv[1];
    std::string grammar_file = argv[2];
    std::string graph_file = argv[3];
    std::string output_file = argv[4];
    std::string weight_mode = argv[5];

    std::cout << "Parser type: " << parser_type << "\n";
    std::cout << "Grammar: " << grammar_file << "\n";
    std::cout << "Graphs: " << graph_file << "\n";
    std::cout << "Output: " << output_file << "\n";
    std::cout << "Weight mode: " << weight_mode << "\n\n";


    shrg::Manager* manager = &shrg::Manager::manager;
    manager->Allocate(1);


    std::cout << "Loading grammars...\n";
    manager->LoadGrammars(grammar_file);

    std::cout << "Loading graphs...\n";
    manager->LoadGraphs(graph_file);


    shrg::Context* context = manager->contexts[0];
    context->Init(parser_type, false, 100);

    std::vector<shrg::SHRG*> shrg_rules = manager->shrg_rules;
    shrg::em::EM em_helper(shrg_rules, manager->edsgraphs, context, 1.0, "N", 5);

    applyWeights(shrg_rules, weight_mode, em_helper);

    std::vector<EntropyDPResult> results;

    size_t num_graphs = manager->edsgraphs.size();

    std::cout << "Processing " << num_graphs << " graphs...\n";

    auto total_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_graphs; ++i) {
        if (i % 100 == 0) {
            std::cout << "  Processing graph " << i << "/" << num_graphs << "\r" << std::flush;
        }

        EntropyDPResult result;
        result.graph_id = static_cast<int>(i);
        result.entropy = 0.0;
        result.log_Z = -std::numeric_limits<double>::infinity();
        result.num_or_nodes = 0;
        result.time_ms = 0.0;
        result.success = false;

        shrg::ParserError error = context->Parse(static_cast<int>(i));

        if (error != shrg::ParserError::kNone) {
            results.push_back(result);
            continue;
        }

        shrg::ChartItem* root = context->parser->Result();
        if (!root) {
            results.push_back(result);
            continue;
        }

        em_helper.addParentPointerOptimized(root, 0);
        em_helper.addRulePointer(root);

        auto start = std::chrono::high_resolution_clock::now();

        double log_Z, entropy;
        ComputePartitionAndEntropyDP(root, log_Z, entropy);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        result.entropy = entropy;
        result.log_Z = log_Z;
        result.time_ms = elapsed.count();
        result.success = true;

        std::unordered_set<shrg::ChartItem*> seen_or_nodes;
        std::function<void(shrg::ChartItem*)> countOrNodes = [&](shrg::ChartItem* node) {
            if (!node) return;
            shrg::ChartItem* canonical = GetCanonicalNode(node);
            if (seen_or_nodes.count(canonical)) return;
            seen_or_nodes.insert(canonical);

            shrg::ChartItem* ptr = node;
            do {
                for (shrg::ChartItem* child : ptr->children) {
                    countOrNodes(child);
                }
                ptr = ptr->next_ptr;
            } while (ptr && ptr != node);
        };
        countOrNodes(root);
        result.num_or_nodes = static_cast<int>(seen_or_nodes.size());

        results.push_back(result);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;

    std::cout << "\nProcessed " << results.size() << " graphs in "
              << std::fixed << std::setprecision(2) << total_elapsed.count() << " seconds.\n";

    // Write results
    writeTsvOutput(output_file, results);

    std::cout << "\nResults written to: " << output_file << "\n";

    return 0;
}

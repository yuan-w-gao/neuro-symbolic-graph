/**
 * @file comprehensive_eval.cpp
 * @brief Comprehensive evaluation: entropy, BLEU, and parsing F1 in a single pass
 *
 * Usage: comprehensive_eval <parser_type> <grammar_file> <graph_file> <output_dir> <weight_file> [gold_derivations_file]
 *
 * Output files:
 *   entropy.tsv              - per-graph entropy metrics
 *   bleu.tsv                 - per-graph BLEU scores
 *   f1.tsv                   - per-graph F1 scores
 *   em.txt                   - EM weight-based sentences
 *   baseline.txt             - uniform sampling sentences
 *   oracle.txt               - oracle (count-based) sentences
 *   original_sentences.txt   - gold sentences from corpus
 *   reference_sentences.txt  - lemma sequences from corpus
 *   summary.json             - overall summary metrics
 */

#include "manager.hpp"
#include "forest_cache.hpp"
#include "ambiguity_metrics/ambiguity_metrics.hpp"
#include "em_framework/find_derivations.hpp"
#include "em_framework/em.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <iomanip>
#include <limits>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <csetjmp>

using namespace shrg;

// ============================================================================
// Timeout handling for parent process parses
// ============================================================================
static sigjmp_buf g_timeout_jmp;
static volatile sig_atomic_t g_timeout_flag = 0;

static void sigalrm_handler(int) {
    g_timeout_flag = 1;
    siglongjmp(g_timeout_jmp, 1);
}

// ============================================================================
// Helper Functions
// ============================================================================

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

inline bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <parser_type> <grammar_file> <graph_file> <output_dir> <weight_file> [options]\n";
    std::cerr << "\nParser types: linear, tree_v1, tree_v2\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --gold <file>       Gold derivations file\n";
    std::cerr << "  --cache-dir <dir>   Directory for forest cache (skips re-parsing)\n";
    std::cerr << "\nOutput files:\n";
    std::cerr << "  entropy.tsv, bleu.tsv, f1.tsv       - per-graph metrics\n";
    std::cerr << "  em.txt, baseline.txt, oracle.txt   - generated sentences\n";
    std::cerr << "  original_sentences.txt             - gold sentences\n";
    std::cerr << "  reference_sentences.txt            - lemma sequences\n";
    std::cerr << "  summary.json                       - overall summary\n";
}

// ============================================================================
// Derivation Finding (from evaluate.cpp)
// ============================================================================

float FindBestDerivationOracle(Generator *generator, ChartItem *root_ptr) {
    if (!root_ptr) return -std::numeric_limits<float>::infinity();
    if (root_ptr->status == VISITED)
        return root_ptr->score;

    float log_sum_scores = 0;
    ChartItem *ptr = root_ptr;
    do {
        log_sum_scores += std::exp(ptr->score);
        ptr = ptr->next_ptr;
    } while (ptr && ptr != root_ptr);
    log_sum_scores = std::log(log_sum_scores);

    float max_score = -std::numeric_limits<float>::infinity();
    ChartItem *max_subgraph_ptr = root_ptr;

    ptr = root_ptr;
    do {
        float current_score = ptr->score - log_sum_scores;

        if (ptr->attrs_ptr && ptr->attrs_ptr->grammar_ptr) {
            const SHRG *grammar_ptr = ptr->attrs_ptr->grammar_ptr;
            for (auto edge_ptr : grammar_ptr->nonterminal_edges) {
                ChartItem *child = generator->FindChartItemByEdge(ptr, edge_ptr);
                if (child) {
                    current_score += FindBestDerivationOracle(generator, child);
                }
            }
        }

        if (max_score < current_score) {
            max_score = current_score;
            max_subgraph_ptr = ptr;
        }

        ptr = ptr->next_ptr;
    } while (ptr != root_ptr);

    if (max_subgraph_ptr != root_ptr)
        root_ptr->Swap(*max_subgraph_ptr);

    root_ptr->status = VISITED;
    root_ptr->score = max_score;
    return root_ptr->score;
}

float FindBestDerivationWeight(ChartItem *root_ptr, Generator *generator) {
    if (!root_ptr) return -std::numeric_limits<float>::infinity();
    // Use em_greedy_deriv for visited tracking (em_greedy_score is used by SampleDerivationTree)
    // This frees up 'status' to store the CFG rule index for generation
    if (root_ptr->em_greedy_deriv == VISITED)
        return root_ptr->score;

    ChartItem *ptr = root_ptr;
    double max_weight = -std::numeric_limits<double>::infinity();
    ChartItem *max_subgraph_ptr = root_ptr;

    do {
        double current_weight = 0;
        if (ptr->attrs_ptr && ptr->attrs_ptr->grammar_ptr) {
            const SHRG *grammar_ptr = ptr->attrs_ptr->grammar_ptr;
            current_weight = grammar_ptr->log_rule_weight;

            // Set the CFG rule index for proper word ordering during generation
            // SelectRule() in parser_base.hpp uses status as CFG index
            auto &cfg_rules = grammar_ptr->cfg_rules;
            if (!cfg_rules.empty()) {
                auto cfg_ptr = std::max_element(cfg_rules.begin(), cfg_rules.end());
                ptr->status = static_cast<int>(cfg_ptr - cfg_rules.begin());
            }

            for (auto edge_ptr : grammar_ptr->nonterminal_edges) {
                ChartItem *child = generator->FindChartItemByEdge(ptr, edge_ptr);
                if (child) {
                    current_weight += FindBestDerivationWeight(child, generator);
                }
            }
        }

        if (max_weight < current_weight) {
            max_weight = current_weight;
            max_subgraph_ptr = ptr;
        }

        ptr = ptr->next_ptr;
    } while (ptr && ptr != root_ptr);

    if (max_subgraph_ptr != root_ptr)
        root_ptr->Swap(*max_subgraph_ptr);

    root_ptr->em_greedy_deriv = VISITED;  // Mark as visited
    root_ptr->score = static_cast<float>(max_weight);
    return root_ptr->score;
}

ChartItem* SampleDerivationTree(ChartItem* root_ptr, std::mt19937& gen) {
    if (!root_ptr) return nullptr;
    if (root_ptr->em_greedy_score == VISITED)
        return root_ptr;

    ChartItem* ptr = root_ptr;
    std::vector<ChartItem*> choices;

    do {
        choices.push_back(ptr);
        ptr = ptr->next_ptr;
    } while (ptr && ptr != root_ptr);

    if (choices.empty()) return root_ptr;

    std::uniform_int_distribution<size_t> dist(0, choices.size() - 1);
    size_t chosen_idx = dist(gen);
    ChartItem* chosen_ptr = choices[chosen_idx];

    if (chosen_ptr != root_ptr) {
        root_ptr->Swap(*chosen_ptr);
    }

    for (auto child : root_ptr->children) {
        if (child) {
            SampleDerivationTree(child, gen);
        }
    }

    root_ptr->em_greedy_score = VISITED;
    return root_ptr;
}

// ============================================================================
// Weight Loading
// ============================================================================

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

        if (!std::getline(ss, token, ',')) continue;
        int rule_index = std::stoi(token);

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

void applyWeights(std::vector<SHRG*>& shrg_rules, const std::string& weight_file) {
    std::map<int, double> weights = loadFinalWeights(weight_file);
    if (weights.empty()) {
        std::cerr << "Warning: No weights loaded from " << weight_file << "\n";
        return;
    }

    int applied = 0;
    for (auto& kv : weights) {
        int idx = kv.first;
        double weight = kv.second;
        if (idx >= 0 && idx < static_cast<int>(shrg_rules.size()) && shrg_rules[idx] != nullptr) {
            shrg_rules[idx]->log_rule_weight = weight;
            applied++;
        }
    }
    std::cout << "Applied " << applied << " weights from " << weight_file << "\n";
}

// ============================================================================
// Gold Derivation Loading
// ============================================================================

// Format: graph_idx<TAB>rule_id1,rule_id2,...
std::map<int, std::vector<int>> loadGoldDerivations(const std::string& filepath) {
    std::map<int, std::vector<int>> gold;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open gold derivations file: " << filepath << "\n";
        return gold;
    }

    std::string line;
    while (std::getline(file, line)) {
        trim_inplace(line);
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string graph_idx_str, rules_str;

        if (!std::getline(ss, graph_idx_str, '\t')) continue;
        if (!std::getline(ss, rules_str)) continue;

        // Try to parse as integer, skip if not a number
        int graph_idx;
        try {
            graph_idx = std::stoi(graph_idx_str);
        } catch (...) {
            continue;  // Skip non-numeric graph IDs
        }

        std::vector<int> rule_ids;

        std::stringstream rules_ss(rules_str);
        std::string rule_id_str;
        while (std::getline(rules_ss, rule_id_str, ',')) {
            trim_inplace(rule_id_str);
            if (!rule_id_str.empty()) {
                try {
                    rule_ids.push_back(std::stoi(rule_id_str));
                } catch (...) {
                    // Skip invalid entries
                }
            }
        }

        gold[graph_idx] = rule_ids;
    }

    std::cout << "Loaded gold derivations for " << gold.size() << " graphs\n";
    return gold;
}

// Load gold edge indices for edge set F1
// Format: graph_id<TAB>num_edge_sets<TAB>edge_set1;edge_set2;...
// Each edge_set is: idx1,idx2,idx3
std::map<int, std::vector<std::vector<int>>> loadGoldEdgeSets(const std::string& filepath) {
    std::map<int, std::vector<std::vector<int>>> gold;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open gold edge sets file: " << filepath << "\n";
        return gold;
    }

    std::string line;
    while (std::getline(file, line)) {
        trim_inplace(line);
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string graph_idx_str, num_str, edge_sets_str;

        if (!std::getline(ss, graph_idx_str, '\t')) continue;
        if (!std::getline(ss, num_str, '\t')) continue;
        std::getline(ss, edge_sets_str);  // May be empty

        // Try to parse as integer, skip if not a number
        int graph_idx;
        try {
            graph_idx = std::stoi(graph_idx_str);
        } catch (...) {
            continue;
        }

        std::vector<std::vector<int>> edge_sets;

        if (!edge_sets_str.empty()) {
            std::stringstream sets_ss(edge_sets_str);
            std::string set_str;
            while (std::getline(sets_ss, set_str, ';')) {
                if (set_str.empty()) continue;

                std::vector<int> indices;
                std::stringstream idx_ss(set_str);
                std::string idx_str;
                while (std::getline(idx_ss, idx_str, ',')) {
                    trim_inplace(idx_str);
                    if (!idx_str.empty()) {
                        try {
                            indices.push_back(std::stoi(idx_str));
                        } catch (...) {
                            // Skip invalid
                        }
                    }
                }
                if (!indices.empty()) {
                    edge_sets.push_back(indices);
                }
            }
        }

        gold[graph_idx] = edge_sets;
    }

    std::cout << "Loaded gold edge sets for " << gold.size() << " graphs\n";
    return gold;
}

// ============================================================================
// BLEU Computation
// ============================================================================

/**
 * Remove $s$...$/s$ annotation patterns from a sentence.
 * The generator produces tokens like: banana$s$￨#￨n￨1￨#￨sg￨3￨#￨#$/s$
 * This removes the $s$...$/s$ parts, leaving just: banana
 */
std::string removeAnnotations(const std::string& sentence) {
    std::string result = sentence;

    // Remove all $s$...$/s$ patterns
    while (true) {
        size_t start = result.find("$s$");
        if (start == std::string::npos) break;

        size_t end = result.find("$/s$", start);
        if (end == std::string::npos) {
            // No closing tag, remove from $s$ to end
            result = result.substr(0, start);
            break;
        }

        // Remove $s$...$/s$ (inclusive)
        result = result.substr(0, start) + result.substr(end + 4);
    }

    return result;
}

/**
 * Tokenize sentence by extracting alphanumeric word sequences (like Python's \w+)
 * Returns lowercase tokens.
 */
std::vector<std::string> tokenize(const std::string& sentence) {
    // First remove annotations
    std::string clean_sentence = removeAnnotations(sentence);

    std::vector<std::string> tokens;
    std::string current_token;

    for (size_t i = 0; i < clean_sentence.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(clean_sentence[i]);

        // Check if alphanumeric or underscore (like \w in regex)
        if (std::isalnum(c) || c == '_') {
            current_token += static_cast<char>(std::tolower(c));
        } else {
            // End of token
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        }
    }

    // Don't forget the last token
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }

    return tokens;
}

std::map<std::vector<std::string>, int> getNgrams(const std::vector<std::string>& tokens, int n) {
    std::map<std::vector<std::string>, int> ngrams;
    if (static_cast<int>(tokens.size()) < n) return ngrams;

    for (size_t i = 0; i <= tokens.size() - static_cast<size_t>(n); ++i) {
        std::vector<std::string> ngram(tokens.begin() + static_cast<long>(i),
                                        tokens.begin() + static_cast<long>(i + static_cast<size_t>(n)));
        ngrams[ngram]++;
    }
    return ngrams;
}

double computeSentenceBleu(const std::string& candidate, const std::string& reference, int max_n_default = 4) {
    std::vector<std::string> cand_tokens = tokenize(candidate);
    std::vector<std::string> ref_tokens = tokenize(reference);

    if (cand_tokens.empty() || ref_tokens.empty()) return 0.0;

    // Dynamically adjust max_n based on sentence length (like Python implementation)
    int max_n = std::min({max_n_default,
                          static_cast<int>(cand_tokens.size()),
                          static_cast<int>(ref_tokens.size())});
    if (max_n == 0) return 0.0;

    std::vector<double> precisions;
    for (int n = 1; n <= max_n; ++n) {
        std::map<std::vector<std::string>, int> cand_ngrams = getNgrams(cand_tokens, n);
        std::map<std::vector<std::string>, int> ref_ngrams = getNgrams(ref_tokens, n);

        int clipped = 0, total = 0;
        for (auto& kv : cand_ngrams) {
            total += kv.second;
            auto it = ref_ngrams.find(kv.first);
            if (it != ref_ngrams.end()) {
                clipped += std::min(kv.second, it->second);
            }
        }

        if (total == 0) {
            precisions.push_back(0.0);
        } else {
            precisions.push_back(static_cast<double>(clipped) / static_cast<double>(total));
        }
    }

    // Check for zero precisions
    for (size_t i = 0; i < precisions.size(); ++i) {
        if (precisions[i] == 0.0) return 0.0;
    }

    double log_precision = 0.0;
    for (size_t i = 0; i < precisions.size(); ++i) {
        log_precision += std::log(precisions[i]);
    }
    log_precision /= static_cast<double>(precisions.size());

    double bp = 1.0;
    if (cand_tokens.size() < ref_tokens.size()) {
        bp = std::exp(1.0 - static_cast<double>(ref_tokens.size()) / static_cast<double>(cand_tokens.size()));
    }

    return bp * std::exp(log_precision);
}

// ============================================================================
// F1 Computation
// ============================================================================

struct F1Result {
    double precision;
    double recall;
    double f1;
    F1Result() : precision(0), recall(0), f1(0) {}
};

F1Result computeF1(const std::set<int>& predicted, const std::set<int>& gold) {
    F1Result result;

    if (predicted.empty() && gold.empty()) {
        result.precision = 1.0;
        result.recall = 1.0;
        result.f1 = 1.0;
        return result;
    }
    if (predicted.empty() || gold.empty()) {
        return result;
    }

    std::set<int> intersection;
    std::set_intersection(predicted.begin(), predicted.end(),
                          gold.begin(), gold.end(),
                          std::inserter(intersection, intersection.begin()));

    result.precision = static_cast<double>(intersection.size()) / static_cast<double>(predicted.size());
    result.recall = static_cast<double>(intersection.size()) / static_cast<double>(gold.size());

    if (result.precision + result.recall > 0) {
        result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall);
    }

    return result;
}

// Compute F1 over edge sets (constituents)
// Each edge set is represented as a sorted vector of edge indices
F1Result computeEdgeSetF1(
    const std::vector<EdgeSet>& predicted_edge_sets,
    const std::vector<std::vector<int>>& gold_edge_sets)
{
    F1Result result;

    // Convert predicted edge sets to comparable format (sorted vectors of indices)
    std::set<std::vector<int>> predicted_set;
    for (const auto& es : predicted_edge_sets) {
        std::vector<int> indices;
        for (size_t i = 0; i < MAX_GRAPH_EDGE_COUNT; ++i) {
            if (es[i]) {
                indices.push_back(static_cast<int>(i));
            }
        }
        if (!indices.empty()) {
            predicted_set.insert(indices);
        }
    }

    // Convert gold edge sets to set
    std::set<std::vector<int>> gold_set;
    for (const auto& indices : gold_edge_sets) {
        if (!indices.empty()) {
            std::vector<int> sorted_indices = indices;
            std::sort(sorted_indices.begin(), sorted_indices.end());
            gold_set.insert(sorted_indices);
        }
    }

    if (predicted_set.empty() && gold_set.empty()) {
        result.precision = 1.0;
        result.recall = 1.0;
        result.f1 = 1.0;
        return result;
    }
    if (predicted_set.empty() || gold_set.empty()) {
        return result;
    }

    // Count intersection
    int intersection_count = 0;
    for (const auto& pred : predicted_set) {
        if (gold_set.count(pred) > 0) {
            intersection_count++;
        }
    }

    result.precision = static_cast<double>(intersection_count) / static_cast<double>(predicted_set.size());
    result.recall = static_cast<double>(intersection_count) / static_cast<double>(gold_set.size());

    if (result.precision + result.recall > 0) {
        result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall);
    }

    return result;
}

// ============================================================================
// Result Structure
// ============================================================================

struct GraphResult {
    int graph_idx;
    std::string graph_id;

    // Entropy
    double entropy;
    double log_Z;
    int num_or_nodes;

    // Sentences
    std::string em_sentence;
    std::string baseline_sentence;
    std::string oracle_sentence;
    std::string original_sentence;
    std::string lemma_sequence;

    // BLEU
    double bleu_em;
    double bleu_baseline;
    double bleu_oracle;

    // F1 (rule-based, legacy)
    std::vector<int> predicted_rules;
    std::vector<int> gold_rules;
    F1Result f1_result;

    // F1 (edge set based)
    std::vector<EdgeSet> predicted_edge_sets;
    std::vector<std::vector<int>> gold_edge_sets;
    F1Result edge_set_f1_result;

    // Meta
    bool parse_success;
    double time_ms;

    GraphResult() : graph_idx(0), entropy(0), log_Z(-std::numeric_limits<double>::infinity()),
                    num_or_nodes(0), bleu_em(0), bleu_baseline(0), bleu_oracle(0),
                    parse_success(false), time_ms(0) {}
};

// ============================================================================
// Output Writers
// ============================================================================

void writeStringsToFile(const std::vector<std::string>& strings, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Could not open " << filename << "\n";
        return;
    }
    for (size_t i = 0; i < strings.size(); ++i) {
        outFile << strings[i] << "\n";
    }
}

void writeEntropyTsv(const std::string& filepath, const std::vector<GraphResult>& results) {
    std::ofstream out(filepath);
    out << "graph_idx\tgraph_id\tentropy\tlog_Z\tnum_or_nodes\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const GraphResult& r = results[i];
        out << r.graph_idx << "\t" << r.graph_id << "\t"
            << std::fixed << std::setprecision(6) << r.entropy << "\t"
            << r.log_Z << "\t" << r.num_or_nodes << "\n";
    }
}

void writeBleuTsv(const std::string& filepath, const std::vector<GraphResult>& results) {
    std::ofstream out(filepath);
    out << "graph_idx\tgraph_id\tbleu_em\tbleu_baseline\tbleu_oracle\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const GraphResult& r = results[i];
        out << r.graph_idx << "\t" << r.graph_id << "\t"
            << std::fixed << std::setprecision(6)
            << r.bleu_em << "\t" << r.bleu_baseline << "\t" << r.bleu_oracle << "\n";
    }
}

void writeF1Tsv(const std::string& filepath, const std::vector<GraphResult>& results) {
    std::ofstream out(filepath);
    out << "graph_idx\tgraph_id\tprecision\trecall\tf1\tedge_set_precision\tedge_set_recall\tedge_set_f1\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const GraphResult& r = results[i];
        out << r.graph_idx << "\t" << r.graph_id << "\t"
            << std::fixed << std::setprecision(6)
            << r.f1_result.precision << "\t"
            << r.f1_result.recall << "\t"
            << r.f1_result.f1 << "\t"
            << r.edge_set_f1_result.precision << "\t"
            << r.edge_set_f1_result.recall << "\t"
            << r.edge_set_f1_result.f1 << "\n";
    }
}

void writeSummaryJson(const std::string& filepath, const std::vector<GraphResult>& results,
                      double total_time_sec) {
    int total = static_cast<int>(results.size());
    int parsed = 0;
    double sum_entropy = 0, sum_bleu_em = 0, sum_bleu_baseline = 0, sum_bleu_oracle = 0;
    double sum_precision = 0, sum_recall = 0, sum_f1 = 0;
    double sum_es_precision = 0, sum_es_recall = 0, sum_es_f1 = 0;
    int f1_count = 0;
    int es_f1_count = 0;

    for (size_t i = 0; i < results.size(); ++i) {
        const GraphResult& r = results[i];
        if (r.parse_success) {
            parsed++;
            sum_entropy += r.entropy;
            sum_bleu_em += r.bleu_em;
            sum_bleu_baseline += r.bleu_baseline;
            sum_bleu_oracle += r.bleu_oracle;
            if (!r.gold_rules.empty()) {
                sum_precision += r.f1_result.precision;
                sum_recall += r.f1_result.recall;
                sum_f1 += r.f1_result.f1;
                f1_count++;
            }
            if (!r.gold_edge_sets.empty()) {
                sum_es_precision += r.edge_set_f1_result.precision;
                sum_es_recall += r.edge_set_f1_result.recall;
                sum_es_f1 += r.edge_set_f1_result.f1;
                es_f1_count++;
            }
        }
    }

    std::ofstream out(filepath);
    out << "{\n";
    out << "  \"total_graphs\": " << total << ",\n";
    out << "  \"parsed_graphs\": " << parsed << ",\n";
    out << "  \"parse_rate\": " << std::fixed << std::setprecision(4)
        << (total > 0 ? static_cast<double>(parsed) / static_cast<double>(total) : 0.0) << ",\n";
    out << "  \"mean_entropy\": " << std::setprecision(6)
        << (parsed > 0 ? sum_entropy / parsed : 0.0) << ",\n";
    out << "  \"mean_bleu_em\": " << (parsed > 0 ? sum_bleu_em / parsed : 0.0) << ",\n";
    out << "  \"mean_bleu_baseline\": " << (parsed > 0 ? sum_bleu_baseline / parsed : 0.0) << ",\n";
    out << "  \"mean_bleu_oracle\": " << (parsed > 0 ? sum_bleu_oracle / parsed : 0.0) << ",\n";
    out << "  \"mean_rule_precision\": " << (f1_count > 0 ? sum_precision / f1_count : 0.0) << ",\n";
    out << "  \"mean_rule_recall\": " << (f1_count > 0 ? sum_recall / f1_count : 0.0) << ",\n";
    out << "  \"mean_rule_f1\": " << (f1_count > 0 ? sum_f1 / f1_count : 0.0) << ",\n";
    out << "  \"rule_f1_graphs_evaluated\": " << f1_count << ",\n";
    out << "  \"mean_edge_set_precision\": " << (es_f1_count > 0 ? sum_es_precision / es_f1_count : 0.0) << ",\n";
    out << "  \"mean_edge_set_recall\": " << (es_f1_count > 0 ? sum_es_recall / es_f1_count : 0.0) << ",\n";
    out << "  \"mean_edge_set_f1\": " << (es_f1_count > 0 ? sum_es_f1 / es_f1_count : 0.0) << ",\n";
    out << "  \"edge_set_f1_graphs_evaluated\": " << es_f1_count << ",\n";
    out << "  \"total_time_sec\": " << std::setprecision(2) << total_time_sec << "\n";
    out << "}\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 6) {
        printUsage(argv[0]);
        return 1;
    }

    std::string parser_type = argv[1];
    std::string grammar_file = argv[2];
    std::string graph_file = argv[3];
    std::string output_dir = argv[4];
    std::string weight_file = argv[5];
    std::string gold_file;
    std::string cache_dir;

    // Parse optional arguments
    for (int i = 6; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gold" && i + 1 < argc) {
            gold_file = argv[++i];
        } else if (arg == "--cache-dir" && i + 1 < argc) {
            cache_dir = argv[++i];
        } else if (!arg.empty() && arg[0] != '-') {
            // Legacy: positional argument for gold file
            if (gold_file.empty()) {
                gold_file = arg;
            }
        }
    }

    create_directory(output_dir);

    std::cout << "=== Comprehensive Evaluation ===\n";
    std::cout << "Parser type: " << parser_type << "\n";
    std::cout << "Grammar: " << grammar_file << "\n";
    std::cout << "Graphs: " << graph_file << "\n";
    std::cout << "Output: " << output_dir << "\n";
    std::cout << "Weights: " << weight_file << "\n";
    if (!gold_file.empty()) {
        std::cout << "Gold derivations: " << gold_file << "\n";
    }
    if (!cache_dir.empty()) {
        std::cout << "Cache dir: " << cache_dir << "\n";
    }
    std::cout << "\n";

    Manager* manager = &Manager::manager;
    manager->Allocate(1);

    std::cout << "Loading grammars...\n";
    manager->LoadGrammars(grammar_file);

    std::cout << "Loading graphs...\n";
    manager->LoadGraphs(graph_file);

    Context* context = manager->contexts[0];
    context->Init(parser_type, false, 100);

    std::vector<SHRG*> shrg_rules = manager->shrg_rules;
    em::EM em_helper(shrg_rules, manager->edsgraphs, context, 1.0, "N", 5);

    applyWeights(shrg_rules, weight_file);

    // Initialize forest cache if cache_dir specified
    std::unique_ptr<forest_cache::ForestCache> forest_cache_ptr;
    uint32_t grammar_hash = 0;
    if (!cache_dir.empty()) {
        forest_cache_ptr = std::make_unique<forest_cache::ForestCache>(cache_dir);

        // Compute grammar hash for cache validation
        std::string grammar_content;
        for (const auto* rule : shrg_rules) {
            if (rule) {
                grammar_content += std::to_string(rule->label_hash);
                grammar_content += ":";
                grammar_content += std::to_string(rule->terminal_edges.size());
                grammar_content += ";";
            }
        }
        grammar_hash = forest_cache::ForestCache::compute_hash(grammar_content);
        forest_cache_ptr->set_grammar_hash(grammar_hash);

        std::cout << "Forest caching enabled (grammar hash: " << std::hex << grammar_hash << std::dec << ")\n";
    }

    // Create attrs_pool for cached forests (needed for sentence generation)
    std::vector<GrammarAttributes> attrs_pool = forest_cache::ForestCache::create_attrs_pool(shrg_rules);

    // Persistent memory pool for cached forests
    utils::MemoryPool<ChartItem> persistent_pool;
    size_t cache_hit_count = 0;
    size_t cache_miss_count = 0;

    std::map<int, std::vector<int>> gold_derivations;
    std::map<int, std::vector<std::vector<int>>> gold_edge_sets;
    if (!gold_file.empty()) {
        gold_derivations = loadGoldDerivations(gold_file);

        // Try to load gold edge indices file (created by convert_gold_derivations.py --with-graph)
        std::string gold_edge_file = gold_file;
        size_t dot_pos = gold_edge_file.rfind('.');
        if (dot_pos != std::string::npos) {
            gold_edge_file = gold_edge_file.substr(0, dot_pos) + "_edge_indices.txt";
        } else {
            gold_edge_file = gold_edge_file + "_edge_indices.txt";
        }
        // Also try standard name in same directory
        std::string gold_dir = gold_file.substr(0, gold_file.rfind('/') + 1);
        std::string gold_edge_file2 = gold_dir + "gold_edge_indices.txt";

        std::ifstream test_file(gold_edge_file);
        if (test_file.good()) {
            test_file.close();
            gold_edge_sets = loadGoldEdgeSets(gold_edge_file);
        } else {
            std::ifstream test_file2(gold_edge_file2);
            if (test_file2.good()) {
                test_file2.close();
                gold_edge_sets = loadGoldEdgeSets(gold_edge_file2);
            }
        }
    }

    Generator* generator = context->parser->GetGenerator();

    std::random_device rd;
    std::mt19937 rng(rd());

    std::vector<GraphResult> results;
    size_t num_graphs = manager->edsgraphs.size();

    // Timeout for parsing (in seconds) - skip graphs that take too long
    const int parse_timeout_seconds = 10;

    std::cout << "Processing " << num_graphs << " graphs...\n";
    std::cout << "  (fork safety: timeout=" << parse_timeout_seconds << "s per graph)\n";

    std::chrono::high_resolution_clock::time_point total_start = std::chrono::high_resolution_clock::now();
    int skipped_count = 0;

    for (size_t i = 0; i < num_graphs; ++i) {
        if (i % 100 == 0) {
            std::cout << "  Processing graph " << i << "/" << num_graphs << "\r" << std::flush;
        }

        GraphResult result;
        result.graph_idx = static_cast<int>(i);
        result.graph_id = manager->edsgraphs[i].sentence_id;
        result.original_sentence = manager->edsgraphs[i].sentence;
        result.lemma_sequence = manager->edsgraphs[i].lemma_sequence;

        std::map<int, std::vector<int>>::iterator git = gold_derivations.find(static_cast<int>(i));
        if (git != gold_derivations.end()) {
            result.gold_rules = git->second;
        }

        // Load gold edge sets if available
        auto git_es = gold_edge_sets.find(static_cast<int>(i));
        if (git_es != gold_edge_sets.end()) {
            result.gold_edge_sets = git_es->second;
        }

        std::chrono::high_resolution_clock::time_point parse_start = std::chrono::high_resolution_clock::now();

        // ========== Try cache first ==========
        ChartItem* cached_root = nullptr;
        uint32_t graph_hash = 0;
        bool cache_hit = false;

        if (forest_cache_ptr) {
            // Compute graph hash for cache lookup
            std::string graph_content = result.graph_id + ":" +
                std::to_string(manager->edsgraphs[i].nodes.size()) + ":" +
                std::to_string(manager->edsgraphs[i].edges.size());
            for (const auto& edge : manager->edsgraphs[i].edges) {
                graph_content += std::to_string(edge.label) + ",";
            }
            graph_hash = forest_cache::ForestCache::compute_hash(graph_content);

            cached_root = forest_cache_ptr->load(result.graph_id, graph_hash, persistent_pool);
            if (cached_root) {
                cache_hit = true;
                cache_hit_count++;
                // Restore both rule_ptr and attrs_ptr for full functionality
                forest_cache::ForestCache::restore_all_pointers(cached_root, shrg_rules, attrs_pool);
            } else {
                cache_miss_count++;
            }
        }

        // If cache hit, skip parsing entirely
        if (cache_hit && cached_root) {
            result.parse_success = true;

            // CRITICAL: Set the graph pointer for the generator (needed for sentence generation)
            context->parser->SetGraph(&manager->edsgraphs[i]);

            // 1. Compute Entropy
            double log_Z = 0, entropy = 0;
            lexcxg::ComputePartitionAndEntropyDP(cached_root, log_Z, entropy);
            result.entropy = entropy;
            result.log_Z = log_Z;

            // Count OR-nodes
            std::unordered_set<ChartItem*> seen_or_nodes;
            std::function<void(ChartItem*)> countOrNodes = [&](ChartItem* node) {
                if (!node) return;
                ChartItem* canonical = lexcxg::GetCanonicalNode(node);
                if (seen_or_nodes.count(canonical)) return;
                seen_or_nodes.insert(canonical);

                ChartItem* ptr = node;
                do {
                    for (size_t c = 0; c < ptr->children.size(); ++c) {
                        countOrNodes(ptr->children[c]);
                    }
                    ptr = ptr->next_ptr;
                } while (ptr && ptr != node);
            };
            countOrNodes(cached_root);
            result.num_or_nodes = static_cast<int>(seen_or_nodes.size());

            // 2. Baseline: uniform sampling
            SampleDerivationTree(cached_root, rng);
            {
                Derivation deriv;
                generator->Generate(cached_root, deriv, result.baseline_sentence);
            }

            // 3. EM: weight-based
            FindBestDerivationWeight(cached_root, generator);
            {
                Derivation deriv;
                generator->Generate(cached_root, deriv, result.em_sentence);
            }

            // Extract predicted rule IDs and edge sets for F1
            DerivationInfo deriv_info = ExtractRuleIndicesAndEdges_EMGreedy(cached_root);
            result.predicted_rules = deriv_info.rule_indices;
            result.predicted_edge_sets = deriv_info.edge_sets;

            // For oracle, we would need a fresh parse (skip for cached)
            result.oracle_sentence = result.em_sentence;  // Fallback

            // Compute BLEU scores
            result.bleu_em = computeSentenceBleu(result.em_sentence, result.lemma_sequence);
            result.bleu_baseline = computeSentenceBleu(result.baseline_sentence, result.lemma_sequence);
            result.bleu_oracle = computeSentenceBleu(result.oracle_sentence, result.lemma_sequence);

            // Compute F1 scores
            if (!result.gold_rules.empty()) {
                std::set<int> pred_set(result.predicted_rules.begin(), result.predicted_rules.end());
                std::set<int> gold_set(result.gold_rules.begin(), result.gold_rules.end());
                result.f1_result = computeF1(pred_set, gold_set);
            }
            if (!result.gold_edge_sets.empty()) {
                result.edge_set_f1_result = computeEdgeSetF1(
                    result.predicted_edge_sets, result.gold_edge_sets);
            }

            std::chrono::high_resolution_clock::time_point parse_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = parse_end - parse_start;
            result.time_ms = elapsed.count();

            results.push_back(result);
            continue;  // Skip to next graph
        }

        // ========== Fork-based timeout protection ==========
        // Match em.cpp's run_safe() pattern: test parse AND forest processing
        pid_t pid = fork();
        if (pid == 0) {
            // Child: test full processing pipeline (parse + forest operations)
            signal(SIGALRM, SIG_DFL);  // Ensure default handler (terminate process)
            alarm(parse_timeout_seconds);  // Kernel timer - sends SIGALRM after timeout

            auto child_code = context->Parse(static_cast<int>(i));
            if (child_code == ParserError::kNone) {
                ChartItem* child_root = context->parser->Result();
                if (child_root) {
                    // Test the operations that can hang (like em.cpp does)
                    em_helper.addParentPointerOptimized(child_root, 0);
                    em_helper.addRulePointer(child_root);
                }
            }
            alarm(0);  // Cancel alarm if completed
            _exit(child_code == ParserError::kNone ? 0 : 1);
        }

        bool should_skip = false;
        if (pid < 0) {
            // fork failed - parse directly (no protection)
            std::cerr << "\nWarning: fork() failed for " << result.graph_id << ", parsing directly\n";
        } else {
            // Parent: blocking wait for child (child will self-terminate via SIGALRM if timeout)
            int status;
            waitpid(pid, &status, 0);

            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                std::string reason;
                if (WIFSIGNALED(status)) {
                    int sig = WTERMSIG(status);
                    if (sig == SIGKILL) reason = "OOM killed";
                    else if (sig == SIGALRM) reason = "timeout (" + std::to_string(parse_timeout_seconds) + "s)";
                    else reason = "signal " + std::to_string(sig);
                } else {
                    reason = "parse failed";
                }
                std::cout << "\n  [SKIP] " << result.graph_id << " - " << reason << "\n";
                skipped_count++;
                should_skip = true;
            }
        }

        if (should_skip) {
            results.push_back(result);
            continue;
        }

        // ========== First Parse: Baseline + EM (with alarm timeout) ==========
        // Set up SIGALRM handler for timeout protection
        struct sigaction sa_new, sa_old;
        sa_new.sa_handler = sigalrm_handler;
        sigemptyset(&sa_new.sa_mask);
        sa_new.sa_flags = 0;
        sigaction(SIGALRM, &sa_new, &sa_old);

        g_timeout_flag = 0;
        ParserError error = ParserError::kUnknown;
        ChartItem* root = nullptr;

        if (sigsetjmp(g_timeout_jmp, 1) == 0) {
            alarm(parse_timeout_seconds);
            error = context->Parse(static_cast<int>(i));
            alarm(0);  // Cancel alarm
        } else {
            // Timeout occurred via siglongjmp
            std::cout << "\n  [SKIP] " << result.graph_id << " - parent parse timeout (" << parse_timeout_seconds << "s)\n";
            sigaction(SIGALRM, &sa_old, nullptr);  // Restore handler
            skipped_count++;
            results.push_back(result);
            continue;
        }

        sigaction(SIGALRM, &sa_old, nullptr);  // Restore handler

        if (error != ParserError::kNone) {
            results.push_back(result);
            continue;
        }

        root = context->parser->Result();
        if (!root) {
            results.push_back(result);
            continue;
        }

        result.parse_success = true;

        em_helper.addParentPointerOptimized(root, 0);
        em_helper.addRulePointer(root);

        // Save to cache if caching is enabled
        if (forest_cache_ptr && root) {
            // Deep copy to persistent pool before saving
            ChartItem* persistent_root = em_helper.deepCopyDerivationForest(root, persistent_pool);
            em_helper.addRulePointer(persistent_root);

            if (graph_hash == 0) {
                std::string graph_content = result.graph_id + ":" +
                    std::to_string(manager->edsgraphs[i].nodes.size()) + ":" +
                    std::to_string(manager->edsgraphs[i].edges.size());
                for (const auto& edge : manager->edsgraphs[i].edges) {
                    graph_content += std::to_string(edge.label) + ",";
                }
                graph_hash = forest_cache::ForestCache::compute_hash(graph_content);
            }

            forest_cache_ptr->save(result.graph_id, graph_hash, persistent_root);
        }

        // 1. Compute Entropy
        double log_Z = 0, entropy = 0;
        lexcxg::ComputePartitionAndEntropyDP(root, log_Z, entropy);
        result.entropy = entropy;
        result.log_Z = log_Z;

        // Count OR-nodes
        std::unordered_set<ChartItem*> seen_or_nodes;
        std::function<void(ChartItem*)> countOrNodes = [&](ChartItem* node) {
            if (!node) return;
            ChartItem* canonical = lexcxg::GetCanonicalNode(node);
            if (seen_or_nodes.count(canonical)) return;
            seen_or_nodes.insert(canonical);

            ChartItem* ptr = node;
            do {
                for (size_t c = 0; c < ptr->children.size(); ++c) {
                    countOrNodes(ptr->children[c]);
                }
                ptr = ptr->next_ptr;
            } while (ptr && ptr != node);
        };
        countOrNodes(root);
        result.num_or_nodes = static_cast<int>(seen_or_nodes.size());

        // 2. Baseline: uniform sampling
        SampleDerivationTree(root, rng);
        {
            Derivation deriv;
            generator->Generate(root, deriv, result.baseline_sentence);
        }

        // 3. EM: weight-based
        FindBestDerivationWeight(root, generator);
        {
            Derivation deriv;
            generator->Generate(root, deriv, result.em_sentence);
        }

        // Extract predicted rule IDs and edge sets for F1
        DerivationInfo deriv_info = ExtractRuleIndicesAndEdges_EMGreedy(root);
        result.predicted_rules = deriv_info.rule_indices;
        result.predicted_edge_sets = deriv_info.edge_sets;

        // ========== Second Parse: Oracle (with alarm timeout) ==========
        sigaction(SIGALRM, &sa_new, &sa_old);
        g_timeout_flag = 0;

        if (sigsetjmp(g_timeout_jmp, 1) == 0) {
            alarm(parse_timeout_seconds);
            error = context->Parse(static_cast<int>(i));
            alarm(0);  // Cancel alarm

            if (error == ParserError::kNone) {
                root = context->parser->Result();
                if (root) {
                    em_helper.addParentPointerOptimized(root, 0);
                    em_helper.addRulePointer(root);

                    FindBestDerivationOracle(generator, root);
                    {
                        Derivation deriv;
                        generator->Generate(root, deriv, result.oracle_sentence);
                    }
                }
            }
        }
        // If timeout on oracle parse, just skip oracle (don't skip entire graph)

        sigaction(SIGALRM, &sa_old, nullptr);  // Restore handler

        // Compute BLEU scores (compare against lemma_sequence, not original_sentence)
        result.bleu_em = computeSentenceBleu(result.em_sentence, result.lemma_sequence);
        result.bleu_baseline = computeSentenceBleu(result.baseline_sentence, result.lemma_sequence);
        result.bleu_oracle = computeSentenceBleu(result.oracle_sentence, result.lemma_sequence);

        // Compute rule-based F1 if gold available (legacy)
        if (!result.gold_rules.empty()) {
            std::set<int> pred_set(result.predicted_rules.begin(), result.predicted_rules.end());
            std::set<int> gold_set(result.gold_rules.begin(), result.gold_rules.end());
            result.f1_result = computeF1(pred_set, gold_set);
        }

        // Compute edge set F1 if gold edge sets available (preferred)
        if (!result.gold_edge_sets.empty()) {
            result.edge_set_f1_result = computeEdgeSetF1(
                result.predicted_edge_sets, result.gold_edge_sets);
        }

        std::chrono::high_resolution_clock::time_point parse_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = parse_end - parse_start;
        result.time_ms = elapsed.count();

        results.push_back(result);
    }

    std::chrono::high_resolution_clock::time_point total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;

    std::cout << "\nProcessed " << results.size() << " graphs in "
              << std::fixed << std::setprecision(2) << total_elapsed.count() << " seconds.";
    if (skipped_count > 0) {
        std::cout << " (" << skipped_count << " skipped due to timeout/OOM)";
    }
    if (forest_cache_ptr) {
        std::cout << " (cache hits: " << cache_hit_count
                  << ", misses: " << cache_miss_count << ")";
    }
    std::cout << "\n\n";

    // Write outputs
    std::cout << "Writing outputs...\n";

    writeEntropyTsv(output_dir + "/entropy.tsv", results);
    writeBleuTsv(output_dir + "/bleu.tsv", results);
    writeF1Tsv(output_dir + "/f1.tsv", results);

    // Collect sentences
    std::vector<std::string> em_sentences, baseline_sentences, oracle_sentences;
    std::vector<std::string> original_sentences, lemma_sequences;
    for (size_t i = 0; i < results.size(); ++i) {
        em_sentences.push_back(results[i].em_sentence);
        baseline_sentences.push_back(results[i].baseline_sentence);
        oracle_sentences.push_back(results[i].oracle_sentence);
        original_sentences.push_back(results[i].original_sentence);
        lemma_sequences.push_back(results[i].lemma_sequence);
    }

    writeStringsToFile(em_sentences, output_dir + "/em.txt");
    writeStringsToFile(baseline_sentences, output_dir + "/baseline.txt");
    writeStringsToFile(oracle_sentences, output_dir + "/oracle.txt");
    writeStringsToFile(original_sentences, output_dir + "/original_sentences.txt");
    writeStringsToFile(lemma_sequences, output_dir + "/reference_sentences.txt");
    writeSummaryJson(output_dir + "/summary.json", results, total_elapsed.count());

    // Print summary
    int parsed = 0;
    double sum_entropy = 0, sum_bleu_em = 0, sum_f1 = 0, sum_es_f1 = 0;
    int f1_count = 0, es_f1_count = 0;

    for (size_t i = 0; i < results.size(); ++i) {
        const GraphResult& r = results[i];
        if (r.parse_success) {
            parsed++;
            sum_entropy += r.entropy;
            sum_bleu_em += r.bleu_em;
            if (!r.gold_rules.empty()) {
                sum_f1 += r.f1_result.f1;
                f1_count++;
            }
            if (!r.gold_edge_sets.empty()) {
                sum_es_f1 += r.edge_set_f1_result.f1;
                es_f1_count++;
            }
        }
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Total graphs: " << results.size() << "\n";
    std::cout << "Parsed: " << parsed << " (" << std::setprecision(1)
              << (100.0 * parsed / static_cast<double>(results.size())) << "%)\n";
    std::cout << "Mean entropy: " << std::setprecision(4)
              << (parsed > 0 ? sum_entropy / parsed : 0.0) << "\n";
    std::cout << "Mean BLEU (EM): " << (parsed > 0 ? sum_bleu_em / parsed : 0.0) << "\n";
    if (f1_count > 0) {
        std::cout << "Mean Rule F1: " << (sum_f1 / f1_count)
                  << " (over " << f1_count << " graphs with gold)\n";
    }
    if (es_f1_count > 0) {
        std::cout << "Mean Edge Set F1: " << (sum_es_f1 / es_f1_count)
                  << " (over " << es_f1_count << " graphs with gold edge sets)\n";
    }
    std::cout << "\nResults written to: " << output_dir << "\n";

    return 0;
}

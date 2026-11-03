//
// compute_baseline.cpp
// Compute baseline parsing (F1) and generation (BLEU) using uniform distribution over rules
//
// Usage: compute_baseline <config> <grammars> <graphs> <output_dir>
//

#include "manager.hpp"
#include "graph_parser/parser_utils.hpp"
#include "em_framework/find_derivations.hpp"
#include "em_framework/em.hpp"

#include <fstream>
#include <random>
#include <map>
#include <sstream>
#include <unordered_set>

using namespace shrg;

// Write a single EdgeSet to string
std::string EdgeSetToString(const EdgeSet& edge_set) {
    std::string result;
    for (size_t i = 0; i < MAX_GRAPH_EDGE_COUNT; ++i) {
        result += edge_set[i] ? '1' : '0';
    }
    return result;
}

// Write derivation info map to file
void WriteDerivationMap(const std::map<std::string, DerivationInfo>& deriv_map,
                        const std::string& filename) {
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    for (const auto& pair : deriv_map) {
        const std::string& graph_id = pair.first;
        const DerivationInfo& info = pair.second;

        out_file << "Graph_ID: " << graph_id << "\n";
        out_file << "Rule_Indices:";
        for (const auto& idx : info.rule_indices) {
            out_file << " " << idx;
        }
        out_file << "\n";
        out_file << "Edge_Sets:";
        for (const auto& edge_set : info.edge_sets) {
            out_file << " " << EdgeSetToString(edge_set);
        }
        out_file << "\n\n";
    }
    out_file.close();
}

// Write strings to file
void writeStrings(const std::vector<std::string>& strings, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    for (const auto& str : strings) {
        outFile << str << std::endl;
    }
    outFile.close();
}

// Thread-local random generator
thread_local std::mt19937 g_baseline_rng(std::random_device{}());

// Uniform random sample from a vector
template<typename T>
T* uniformSample(const std::vector<T*>& items) {
    if (items.empty()) return nullptr;
    std::uniform_int_distribution<size_t> dis(0, items.size() - 1);
    return items[dis(g_baseline_rng)];
}

// Uniform random index from range [0, n)
size_t uniformIndex(size_t n) {
    if (n == 0) return 0;
    std::uniform_int_distribution<size_t> dis(0, n - 1);
    return dis(g_baseline_rng);
}

// Special status value to indicate node has been visited
const int BASELINE_VISITED_FLAG = -9999;

// Global counters for ambiguity analysis
int g_total_nodes = 0;
int g_ambiguous_nodes = 0;
int g_total_choices = 0;
int g_multi_cfg_nodes = 0;

// Select uniform derivation and record it, then swap to root position for generation
// Also sets chart_item->status to a random CFG rule index for generation
void SelectUniformDerivation(ChartItem* root_ptr, DerivationInfo& deriv_info,
                             std::unordered_set<ChartItem*>& visited) {
    if (!root_ptr || visited.count(root_ptr)) {
        return;
    }

    // Collect all alternatives at this node
    std::vector<ChartItem*> alternatives;
    ChartItem* ptr = root_ptr;
    do {
        if (ptr->rule_ptr) {
            alternatives.push_back(ptr);
        }
        ptr = ptr->next_ptr;
    } while (ptr && ptr != root_ptr);

    // Update global counters
    g_total_nodes++;
    g_total_choices += alternatives.size();
    if (alternatives.size() > 1) g_ambiguous_nodes++;

    if (alternatives.empty()) {
        return;
    }

    // Check CFG rule count
    if (root_ptr->attrs_ptr && root_ptr->attrs_ptr->grammar_ptr) {
        if (root_ptr->attrs_ptr->grammar_ptr->cfg_rules.size() > 1)
            g_multi_cfg_nodes++;
    }

    // Uniformly sample one alternative (SHRG rule)
    ChartItem* chosen = uniformSample(alternatives);
    if (!chosen || !chosen->rule_ptr) {
        return;
    }

    // Record this rule
    deriv_info.rule_indices.push_back(chosen->shrg_index);
    deriv_info.edge_sets.push_back(chosen->edge_set);

    // Swap chosen to root position so Generator uses it
    if (chosen != root_ptr) {
        root_ptr->Swap(*chosen);
    }

    // Mark all alternatives as visited
    ptr = root_ptr;
    do {
        visited.insert(ptr);
        ptr = ptr->next_ptr;
    } while (ptr && ptr != root_ptr);

    // Set status to a RANDOM CFG rule index for generation
    // This is critical: SelectRule uses status to pick CFG rule
    // If status < 0, it uses best_cfg_ptr (most frequent), which biases results!
    if (root_ptr->attrs_ptr && root_ptr->attrs_ptr->grammar_ptr) {
        size_t num_cfg_rules = root_ptr->attrs_ptr->grammar_ptr->cfg_rules.size();
        if (num_cfg_rules > 0) {
            root_ptr->status = uniformIndex(num_cfg_rules);
        } else {
            root_ptr->status = 0;
        }
    }

    // Recursively process children
    for (auto child : root_ptr->children) {
        SelectUniformDerivation(child, deriv_info, visited);
    }
}

int main(int argc, char* argv[]) {
    auto *manager = &Manager::manager;
    manager->Allocate(1);

    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <config> <grammars> <graphs> <output_dir>" << std::endl;
        std::cout << "\nComputes baseline parsing and generation using uniform distribution over rules." << std::endl;
        std::cout << "\nOutputs:" << std::endl;
        std::cout << "  base_edges.txt       - Rule indices and edge sets for F1 computation" << std::endl;
        std::cout << "  baselines.txt        - Generated sentences for BLEU computation" << std::endl;
        std::cout << "  gold_edges.txt       - Gold derivation edges" << std::endl;
        std::cout << "  reference_sentences.txt - Reference sentences" << std::endl;
        std::cout << "  original_sentences.txt  - Original sentences" << std::endl;
        return 1;
    }

    manager->LoadGrammars(argv[2]);
    manager->LoadGraphs(argv[3]);
    auto &context = manager->contexts[0];
    context->Init(argv[1], false, 100);

    std::string outDir = std::string(argv[4]);
    if (outDir.back() != '/') outDir += '/';

    std::vector<SHRG*> shrg_rules = manager->shrg_rules;

    // Initialize all rules with uniform weights (log(1) = 0)
    for (auto rule : shrg_rules) {
        if (rule) {
            rule->log_rule_weight = 0.0;  // Uniform distribution
        }
    }

    Generator* generator = context->parser->GetGenerator();

    // Create EM model for helper functions
    shrg::em::EM model(shrg_rules, manager->edsgraphs, context, 1, outDir, 1);

    std::vector<std::string> baselines;
    std::vector<std::string> lemmas;
    std::vector<std::string> originals;
    std::map<std::string, DerivationInfo> base_map;
    std::map<std::string, DerivationInfo> gold_map;

    int processed = 0;
    int failed = 0;

    // Reset global counters
    g_total_nodes = 0;
    g_ambiguous_nodes = 0;
    g_total_choices = 0;
    g_multi_cfg_nodes = 0;

    std::cout << "Processing " << manager->edsgraphs.size() << " graphs with uniform baseline..." << std::endl;

    for (auto& graph : manager->edsgraphs) {
        auto code = context->parser->Parse(graph);

        if (code == ParserError::kNone) {
            ChartItem* root = context->parser->Result();
            if (root) {
                // Add children and rule pointers using EMBase methods
                model.addChildren(root);
                model.addRulePointer(root);

                // Extract baseline derivation using uniform sampling
                DerivationInfo base_info;
                std::unordered_set<ChartItem*> visited;
                SelectUniformDerivation(root, base_info, visited);

                // Generate using the selected derivation
                Derivation deriv;
                std::string baseline_sent;
                generator->Generate(root, deriv, baseline_sent);

                baselines.push_back(baseline_sent);
                base_map[graph.sentence_id] = base_info;

                // Store reference data
                lemmas.push_back(graph.lemma_sequence);
                originals.push_back(graph.sentence);

                processed++;
            } else {
                failed++;
            }
        } else {
            failed++;
        }

        // Second parse for gold derivation (using oracle/count-based scores)
        code = context->parser->Parse(graph);
        if (code == ParserError::kNone) {
            ChartItem* root = context->parser->Result();
            if (root) {
                model.addChildren(root);
                model.addRulePointer(root);

                // Compute inside scores based on occurrence counts
                model.computeInside(root);

                // Extract gold derivation using count-based inside scores
                DerivationInfo gold_info = ExtractRuleIndicesAndEdges_CountGreedy(root);
                gold_map[graph.sentence_id] = gold_info;
            }
        }
    }

    std::cout << "Processed: " << processed << ", Failed: " << failed << std::endl;
    std::cout << "\n=== Ambiguity Analysis ===" << std::endl;
    std::cout << "Total nodes: " << g_total_nodes << std::endl;
    std::cout << "Ambiguous nodes (>1 SHRG alt): " << g_ambiguous_nodes
              << " (" << (100.0 * g_ambiguous_nodes / g_total_nodes) << "%)" << std::endl;
    std::cout << "Avg SHRG choices per node: " << (1.0 * g_total_choices / g_total_nodes) << std::endl;
    std::cout << "Nodes with >1 CFG rules: " << g_multi_cfg_nodes
              << " (" << (100.0 * g_multi_cfg_nodes / g_total_nodes) << "%)" << std::endl;
    std::cout << "Writing output files to: " << outDir << std::endl;

    // Write outputs
    WriteDerivationMap(base_map, outDir + "base_edges.txt");
    WriteDerivationMap(gold_map, outDir + "gold_edges.txt");
    writeStrings(baselines, outDir + "baselines.txt");
    writeStrings(lemmas, outDir + "reference_sentences.txt");
    writeStrings(originals, outDir + "original_sentences.txt");

    std::cout << "Done. Output files:" << std::endl;
    std::cout << "  " << outDir << "base_edges.txt" << std::endl;
    std::cout << "  " << outDir << "gold_edges.txt" << std::endl;
    std::cout << "  " << outDir << "baselines.txt" << std::endl;
    std::cout << "  " << outDir << "reference_sentences.txt" << std::endl;
    std::cout << "  " << outDir << "original_sentences.txt" << std::endl;

    return 0;
}

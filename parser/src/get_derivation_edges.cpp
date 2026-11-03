//
// Created by Yuan on 9/26/25.
//
//
// Created by Yuan Gao on 11/2/22.
//
#include "manager.hpp"

// #include "python/inside_outside.hpp"
// #include "graph_parser/synchronous_hyperedge_replacement_grammar.hpp"
#include "em_framework/em_base.hpp"
#include "em_framework/em.hpp"
#include "em_framework/find_derivations.hpp"

// // #include "em_framework/em_hmm.hpp"
// #include "python/find_best_derivation.cpp"
// #include "test.cpp"
// #include "em_framework/em_evaluate/eval_deriv.hpp"
// #include "em_framework/em_utils.hpp"
// #include "em_framework/variational_inference.hpp"

#include <set>
#include <iostream>
#include <map>
#include <chrono>
#include <random>
#include <string>
#include <optional>
#include <functional>

using namespace shrg;

// std::vector<std::vector<double>> LoadProbabilities(const std::string &prob_file){
//     // std::cout << "Loading probs from :" << prob_file << std::endl;
//     std::vector<std::vector<double>> rule_probs;
//     std::ifstream file(prob_file);
//     std::string line;
//     while(std::getline(file, line)){
//         std::stringstream ss(line);
//         std::string item;
//         std::vector<double> probs;
//
//         std::getline(ss, item, ',');
//         while(std::getline(ss, item, ',')){
//             probs.push_back(std::stod(item));
//         }
//
//         rule_probs.push_back(probs);
//     }
//     return rule_probs;
// }
#include <limits>
#include <cctype>
#include <cerrno>
#include <cmath>

static inline void trim_inplace(std::string& s) {
    auto issp = [](unsigned char c){ return std::isspace(c); };
    while (!s.empty() && issp(s.front())) s.erase(s.begin());
    while (!s.empty() && issp(s.back()))  s.pop_back();
    if (!s.empty() && s.back() == '\r') s.pop_back(); // handle CRLF
}

// Replace Unicode minus (U+2212, UTF-8: E2 88 92) with ASCII '-'
static inline void normalize_minus(std::string& s) {
    for (size_t i = 0; i + 2 < s.size(); ++i) {
        if ((unsigned char)s[i] == 0xE2 &&
            (unsigned char)s[i+1] == 0x88 &&
            (unsigned char)s[i+2] == 0x92) {
            s[i] = '-';
            s.erase(i+1, 2);
        }
    }
}

static inline std::string lower_copy(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static inline void log_token_bytes(const std::string& tok, size_t line_no, size_t field_idx, const char* msg) {
    std::cerr << "[LoadProbabilities] " << msg
              << " at line " << line_no << " field " << field_idx
              << " text:'" << tok << "' bytes:";
    std::ios::fmtflags f(std::cerr.flags());
    for (unsigned char c : tok) std::cerr << " " << std::hex << std::uppercase << (int)c;
    std::cerr.flags(f);
    std::cerr << "\n";
}

static inline bool parse_double_token(const std::string& raw, double& out,
                                      size_t line_no, size_t field_idx)
{
    std::string s = raw;
    trim_inplace(s);
    normalize_minus(s);
    if (s.empty()) {
        log_token_bytes(raw, line_no, field_idx, "Empty field; skipping");
        return false;
    }

    // Handle inf / -inf / nan (case-insensitive)
    std::string lo = lower_copy(s);
    if (lo == "inf" || lo == "+inf" || lo == "infinity" || lo == "+infinity") {
        out =  std::numeric_limits<double>::infinity();
        return true;
    }
    if (lo == "-inf" || lo == "-infinity") {
        out = -std::numeric_limits<double>::infinity();
        return true;
    }
    if (lo == "nan" || lo == "+nan" || lo == "-nan") {
        out = std::numeric_limits<double>::quiet_NaN();
        return true;
    }

    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);

    if (end == s.c_str()) {
        log_token_bytes(raw, line_no, field_idx, "No numeric content; skipping");
        return false;
    }
    if (*end != '\0') {
        // Trailing junk; accept the parsed prefix but warn.
        log_token_bytes(raw, line_no, field_idx, "Trailing characters after number");
    }

    if (errno == ERANGE) {
        if (std::isinf(v)) {
            // Overflow: clamp to +/- DBL_MAX
            out = std::copysign(std::numeric_limits<double>::max(), v);
            log_token_bytes(raw, line_no, field_idx, "Overflow; clamping to ±DBL_MAX");
            return true;
        }
        if (v == 0.0) {
            // Underflow: clamp to signed zero
            const bool neg = (!s.empty() && s[0] == '-');
            out = neg ? -0.0 : 0.0;
            log_token_bytes(raw, line_no, field_idx, "Underflow; clamping to signed zero");
            return true;
        }
    }

    out = v;
    return true;
}

std::vector<std::vector<double>> LoadProbabilities(const std::string& prob_file) {
    std::ifstream file(prob_file);
    if (!file) throw std::runtime_error("Cannot open: " + prob_file);

    std::vector<std::vector<double>> rule_probs;
    std::string line;
    size_t line_no = 0;

    while (std::getline(file, line)) {
        ++line_no;
        std::stringstream ss(line);
        std::string item;
        std::vector<double> probs;

        // Discard the first CSV field (e.g., rule id)
        std::getline(ss, item, ',');

        size_t field_idx = 0; // 1-based index for probability fields
        while (std::getline(ss, item, ',')) {
            ++field_idx;
            double val;
            if (parse_double_token(item, val, line_no, field_idx)) {
                probs.push_back(val);
            } else {
                // Skipped invalid/empty token; continue.
            }
        }

        rule_probs.push_back(std::move(probs));
    }

    return rule_probs;
}


std::string EdgeSetToString(const shrg::EdgeSet& edge_set) {
    std::string result = "";
    for (size_t i = 0; i < shrg::MAX_GRAPH_EDGE_COUNT; ++i) {
        result += edge_set[i] ? '1' : '0';
    }
    return result;
}

// Parse a string back to EdgeSet
shrg::EdgeSet StringToEdgeSet(const std::string& str) {
    shrg::EdgeSet edge_set;
    for (size_t i = 0; i < str.length() && i < shrg::MAX_GRAPH_EDGE_COUNT; ++i) {
        if (str[i] == '1') {
            edge_set.set(i);
        }
    }
    return edge_set;
}
void WriteDerivationInfoMap(const std::map<std::string, shrg::DerivationInfo>& deriv_map, const std::string& filename) {
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    for (const auto& [graph_id, info] : deriv_map) {
        // Write graph ID
        out_file << "Graph_ID: " << graph_id << "\n";

        // Write rule indices
        out_file << "Rule_Indices:";
        for (const auto& idx : info.rule_indices) {
            out_file << " " << idx;
        }
        out_file << "\n";

        // Write edge sets
        out_file << "Edge_Sets:";
        for (const auto& edge_set : info.edge_sets) {
            out_file << " " << EdgeSetToString(edge_set);
        }
        out_file << "\n\n";
    }

    out_file.close();
}

std::map<std::string, std::vector<int>> LoadGoldDerivationIndices(const std::string& filename) {
    std::map<std::string, std::vector<int>> result;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        // std::cout << "Processing line: '" << line << "'" << std::endl;

        std::istringstream line_stream(line);
        std::string token;

        // Get graph ID (string, e.g., "wsj00b-20041037")
        std::getline(line_stream, token, ':');
        std::string graph_id = token;
        // std::cout << "Graph ID: " << graph_id << std::endl;

        // Get number of indices (we don't actually need this)
        std::getline(line_stream, token, ':');
        // std::cout << "Number field: '" << token << "'" << std::endl;

        // Get indices
        std::getline(line_stream, token);
        // std::cout << "Indices string: '" << token << "'" << std::endl;

        std::istringstream indices_stream(token);
        std::vector<int> indices;

        // Parse comma-separated indices
        std::string index_str;
        while (std::getline(indices_stream, index_str, ',')) {
            // std::cout << "Raw index_str: '" << index_str << "'" << std::endl;

            // Trim whitespace
            index_str.erase(0, index_str.find_first_not_of(" \t"));
            index_str.erase(index_str.find_last_not_of(" \t") + 1);

            // std::cout << "Trimmed index_str: '" << index_str << "'" << std::endl;

            if (!index_str.empty()) {
                int index = std::stoi(index_str);
                // std::cout << "Parsed index: " << index << std::endl;
                indices.push_back(index);
                // std::cout << "Vector size after push_back: " << indices.size() << std::endl;
            }
        }

        // std::cout << "Final indices vector size: " << indices.size() << std::endl;
        // std::cout << "Indices contents: ";
        // for (int idx : indices) {
            // std::cout << idx << " ";
        // }
        // std::cout << std::endl;

        result[graph_id] = indices;
        // std::cout << "Added to result map for graph_id " << graph_id << std::endl;
        // std::cout << "Result map now has " << result.size() << " entries" << std::endl;
        // std::cout << "---" << std::endl;
    }

    file.close();
    return result;
}




int main(int argc, char *argv[]) {
    clock_t t1,t2, t3, t4;

    auto *manager = &shrg::Manager::manager;
    manager->Allocate(1);
    if (argc < 7) {
        LOG_ERROR("Usage: get_deriv <parser_type> <grammar_path> <graph_path> <output_dir> <weight_history> <gold_indices>");
        return 1;
    }

    manager->LoadGrammars(argv[2]);
    manager->LoadGraphs(argv[3]);
    auto &context = manager->contexts[0];
    context->Init(argv[1], false , 100 );


    auto graphs = manager->edsgraphs;
    std::vector<shrg::SHRG *>shrg_rules = manager->shrg_rules;
    // auto Generator* generator = context->parser->GetGenerator();
    std::vector<shrg::ChartItem*> forest;
    // for (auto graph:graphs) {
    //     auto code = context->parser->Parse(graph);
    //     if(code == ParserError::kNone) {
    //         ChartItem *root = context->parser->Result();
    //         forest.push_back(root);
    //         addParentPointer(root, generator, 0);
    //         addRulePointer(root, shrg_rules);
    //         double pw = computeInside(root);
    //         computeOutside(root);
    //     }
    // }
    // EM(shrg_rules, graphs, context, 0.05);

    // std::cout << "Rules Len: " << shrg_rules.size() << std::endl;
    std::string outDir = std::string(argv[4]);
    // std::string gold_file = argv[6];
    // int num_iteration = 1;
    shrg::em::EM model = shrg::em::EM(shrg_rules, graphs, context, 10, argv[4], 5);
    // // shrg::vi::VariationalInference model = shrg::vi::VariationalInference(shrg_rules, graphs, context, 30);
    // model.run();
    std::vector<std::vector<double>> probs = LoadProbabilities(argv[5]);

    bool equals = probs.size() == shrg_rules.size();

    std::cout << argv[1] << "prob_size: " << probs.size() << "; rule_size: " << shrg_rules.size() << std::endl;
    //
    int num_iteration = probs[0].size() - 1;

    for(size_t i = 0; i < shrg_rules.size(); i++){
        auto rule = shrg_rules[i];
        rule->log_rule_weight = probs[i][num_iteration];
    }
    //
    //
    // // em::EVALUATE_DERIVATION eval_deriv = em::EVALUATE_DERIVATION(&model);
    //
    // // std::map<std::string, std::vector<int>> em_g, em_i, or_g, or_i, base;
    //
    // std::cout << "Getting Derivs" << std::endl;
    //
    std::map<std::string, shrg::DerivationInfo> em_map, oracle_map, gold_map;
    std::map<std::string, shrg::DerivationInfo> base_map;

    std::string gold_file = argv[6];
    std::map<std::string, std::vector<int>> gold_deriv = LoadGoldDerivationIndices(gold_file);
    std::cout << "Loaded " << gold_deriv.size() << " gold derivations" << std::endl;

    std::random_device rd;
    std::mt19937 rng(rd());
    for( int i = 0; i < graphs.size(); i++){
        if (i%200 == 0) {
            std::cout << i << std::endl;
        }
        auto graph = graphs[i];
        auto code = context->parser->Parse(graph);
        if(code == shrg::ParserError::kNone) {
            shrg::ChartItem *root = context->parser->Result();
            model.addParentPointerOptimized(root, 0);
            model.addRulePointer(root);
            // model.computeInside(root);
            shrg::DerivationInfo oracle= ExtractRuleIndicesAndEdges_ScoreGreedy(root);
            shrg::DerivationInfo em = ExtractRuleIndicesAndEdges_EMGreedy(root);
            shrg::DerivationInfo base = ExtractDerivation_uniform(root);

            // Extract gold derivation
            std::string y = graph.sentence_id.substr(graph.sentence_id.find('/') + 1);
            if (gold_deriv.find(y) != gold_deriv.end()) {
                std::vector<int> gold_ind = gold_deriv[y];

                // Debug: print gold indices and available shrg_index values
                static int debug_count = 0;
                if (debug_count < 3) {
                    std::cout << "DEBUG graph " << y << ": gold_indices = [";
                    for (size_t i = 0; i < gold_ind.size(); i++) {
                        std::cout << gold_ind[i] << (i < gold_ind.size()-1 ? "," : "");
                    }
                    std::cout << "], available shrg_index in forest: [";
                    shrg::ChartItem* ptr = root;
                    std::set<int> seen_indices;
                    std::function<void(shrg::ChartItem*)> collect_indices = [&](shrg::ChartItem* node) {
                        if (!node) return;
                        shrg::ChartItem* p = node;
                        do {
                            if (seen_indices.find(p->shrg_index) == seen_indices.end()) {
                                seen_indices.insert(p->shrg_index);
                            }
                            for (auto* child : p->children) {
                                collect_indices(child);
                            }
                            p = p->next_ptr;
                        } while (p && p != node);
                    };
                    collect_indices(root);
                    bool first = true;
                    for (int idx : seen_indices) {
                        if (!first) std::cout << ",";
                        std::cout << idx;
                        first = false;
                    }
                    std::cout << "]" << std::endl;
                    debug_count++;
                }

                std::optional<shrg::DerivationInfo> gold = ExtractGoldDerivation(root, gold_ind);
                if (gold) {
                    gold_map[graph.sentence_id] = *gold;
                } else {
                    std::cout << "Could not find valid derivation for graph " << graph.sentence_id << std::endl;
                }
            }

            base_map[graph.sentence_id] = base;
            oracle_map[graph.sentence_id] = oracle;
            em_map[graph.sentence_id] = em;
        }
    }
    WriteDerivationInfoMap(base_map, outDir+"base_edges.txt");
    WriteDerivationInfoMap(oracle_map, outDir+"oracle_edges.txt");
    WriteDerivationInfoMap(em_map, outDir+"em_edges.txt");
    WriteDerivationInfoMap(gold_map, outDir+"gold_edges.txt");
    std::cout << "Wrote " << gold_map.size() << " gold, " << em_map.size() << " em, "
              << oracle_map.size() << " oracle, " << base_map.size() << " base edges" << std::endl;
    //
    // for( auto &graph:manager->edsgraphs){
    //     auto code = context->parser->Parse(graph);
    //     if(code == ParserError::kNone) {
    //         ChartItem *root = context->parser->Result();
    //         model.addParentPointerOptimized(root, 0);
    //         model.addRulePointer(root);
    //         model.computeInside(root);
    //         // DerivationInfo oracle = ExtractRuleIndicesAndEdges_CountGreedy(root);
    //         // DerivationInfo base = ExtractDerivation_sampled(root);
    //         DerivationInfo em = ExtractRuleIndicesAndEdges_EMInside(root);
    //         // int y = std::stoi(graph.sentence_id.substr(graph.sentence_id.find('/') + 1));
    //         // std::vector<int> gold_ind = gold_deriv[y];
    //         // std::optional<DerivationInfo> gold = ExtractGoldDerivationTree(root, gold_ind);
    //         // if (gold){
    //         //     gold_map[graph.sentence_id] = *gold;
    //         // }else{
    //         //     std::cout << "Could not find valid derivation for graph " << graph.sentence_id << std::endl;
    //         // }
    //         // base_map[graph.sentence_id] = base;
    //         // oracle_map[graph.sentence_id] = oracle;
    //         em_map[graph.sentence_id] = em;
    //     }
    // }
    //
    // WriteDerivationInfoMap(em_map,outDir+"em_edges_iter1.txt" );
    // WriteDerivationInfoMap(oracle_map, outDir+"oracle_edges_iter1.txt");
    // // WriteDerivationInfoMap(gold_map, outDir+"gold_edges.txt");
    // //
    // // for (int i = 0; i < graphs.size(); i ++) {
    // //     if (i%200 == 0) {
    // //         std::cout << i << std::endl;
    // //     }
    // //
    // //     auto graph = graphs[i];
    // //     auto code = context->Parse(graph);
    // //     if (code == ParserError::kNone) {
    // //         ChartItem *root = context->parser->Result();
    // //         std::string graph_id = graph.sentence_id;
    // //         model.addParentPointerOptimized(root, 0);
    // //         model.addRulePointer(root);
    // //
    // //         std::vector<int> ei = eval_deriv.get_rule_indices(root, em::EVALUATE_DERIVATION::EM_Inside_deriv);
    // //         std::vector<int> eg =eval_deriv.get_rule_indices(root, em::EVALUATE_DERIVATION::EM_Greedy_deriv);
    // //         std::vector<int> og = eval_deriv.get_rule_indices(root, em::EVALUATE_DERIVATION::Count_Greedy_deriv);
    // //         std::vector<int> oi =eval_deriv.get_rule_indices(root, em::EVALUATE_DERIVATION::Count_Inside_deriv);
    // //         // std::vector<int> b =eval_deriv.get_rule_indices(root, em::EVALUATE_DERIVATION::Baseline_Sample_deriv);
    // //         em_g[graph_id] = eg;
    // //         em_i[graph_id] = ei;
    // //         or_g[graph_id] = og;
    // //         or_i[graph_id] = oi;
    // //
    // //     }
    // // }
    // // std::string eg_f = dir + "eg.txt";
    // // std::string ei_f = dir + "ei.txt";
    // // std::string og_f = dir + "og.txt";
    // // std::string oi_f = dir + "oi.txt";
    // // // std::string b_f = dir + "b.txt";
    // // exportMapToFile(em_g, eg_f);
    // // exportMapToFile(em_i, ei_f);
    // // exportMapToFile(or_g, og_f);
    // // exportMapToFile(or_i, oi_f);


    return 0;
}

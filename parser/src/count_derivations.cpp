/**
 * @file count_derivations.cpp
 * @brief Count the number of derivation trees for each graph in a grammar
 *
 * Usage: count_derivations <parser_type> <grammar_file> <graph_file> <output_file>
 *
 * Output format:
 *   graph_id    count    log_count
 *   ...
 *   AVERAGE     avg_count    avg_log_count
 */

#include "ambiguity_metrics/ambiguity_metrics.hpp"
#include "manager.hpp"
#include "graph_parser/parser_base.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace shrg;

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <parser_type> <grammar_file> <graph_file> <output_file>\n";
    std::cerr << "\n";
    std::cerr << "Parser types: linear, tree_v1, tree_v2\n";
    std::cerr << "\n";
    std::cerr << "Computes the number of derivation trees for each graph.\n";
    std::cerr << "Output format: graph_id <tab> count <tab> log_count\n";
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printUsage(argv[0]);
        return 1;
    }

    std::string parser_type = argv[1];
    std::string grammar_file = argv[2];
    std::string graph_file = argv[3];
    std::string output_file = argv[4];

    // Initialize the manager
    Manager* manager = &Manager::manager;
    manager->Allocate(1);

    // Load grammars and graphs
    std::cerr << "Loading grammars from " << grammar_file << "..." << std::endl;
    if (!manager->LoadGrammars(grammar_file)) {
        std::cerr << "Error: Failed to load grammars" << std::endl;
        return 1;
    }

    std::cerr << "Loading graphs from " << graph_file << "..." << std::endl;
    if (!manager->LoadGraphs(graph_file)) {
        std::cerr << "Error: Failed to load graphs" << std::endl;
        return 1;
    }

    // Initialize parser context
    Context* context = manager->contexts[0];
    context->Init(parser_type, false, 100);

    Generator* generator = context->parser->GetGenerator();

    std::ofstream out(output_file);
    if (!out) {
        std::cerr << "Error: Cannot open output file: " << output_file << std::endl;
        return 1;
    }

    // Header
    out << "graph_id\tcount\tlog_count\n";

    size_t num_graphs = manager->edsgraphs.size();
    std::cerr << "Processing " << num_graphs << " graphs..." << std::endl;

    double sum_log_count = 0.0;
    int valid_count = 0;
    int parse_failed = 0;

    for (size_t i = 0; i < num_graphs; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == num_graphs) {
            std::cerr << "  Processing graph " << (i + 1) << "/" << num_graphs << "\r" << std::flush;
        }

        const EdsGraph& graph = manager->edsgraphs[i];

        // Parse the graph
        ParserError error = context->Parse(static_cast<int>(i));

        if (error != ParserError::kNone) {
            // Failed to parse
            out << graph.sentence_id << "\t0\t-inf\n";
            parse_failed++;
            continue;
        }

        ChartItem* root = context->parser->Result();
        if (!root) {
            out << graph.sentence_id << "\t0\t-inf\n";
            parse_failed++;
            continue;
        }

        // Count derivations using the efficient method (children only, no parents_sib)
        double log_count = lexcxg::CountDerivationsLog(root, generator);
        double count = std::exp(log_count);

        // Cap display count at 1e100 for readability
        if (count > 1e100) {
            count = 1e100;
        }

        out << graph.sentence_id << "\t"
            << std::fixed << std::setprecision(0) << count << "\t"
            << std::setprecision(4) << log_count << "\n";

        sum_log_count += log_count;
        valid_count++;
    }

    std::cerr << std::endl;

    // Write average
    double avg_log_count = (valid_count > 0) ? sum_log_count / valid_count : 0.0;
    double avg_count = std::exp(avg_log_count);
    if (avg_count > 1e100) {
        avg_count = 1e100;
    }

    out << "AVERAGE\t"
        << std::fixed << std::setprecision(0) << avg_count << "\t"
        << std::setprecision(4) << avg_log_count << "\n";

    out.close();

    // Summary to stderr
    std::cerr << "Results written to " << output_file << std::endl;
    std::cerr << "  Total graphs: " << num_graphs << std::endl;
    std::cerr << "  Successfully parsed: " << valid_count << std::endl;
    std::cerr << "  Parse failed: " << parse_failed << std::endl;
    std::cerr << "  Average log(count): " << std::fixed << std::setprecision(4) << avg_log_count << std::endl;

    return 0;
}

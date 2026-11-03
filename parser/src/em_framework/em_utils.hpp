//
// Created by Yuan Gao on 04/06/2024.
//

#ifndef SHRG_GRAPH_PARSER_EM_UTILS_HPP
#define SHRG_GRAPH_PARSER_EM_UTILS_HPP

#include <fstream>
#include "em_types.hpp"
#include "../graph_parser/parser_chart_item.hpp"

namespace shrg{
    using namespace std;
    double addLogs(double a, double b);
    bool is_negative(double log_prob);
    ChartItem* getParent(ParentTup &p);
    std::vector<ChartItem*> getSiblings(ParentTup &p);
    bool is_normal_count(double c);
    void setInitialWeights(LabelToRule &dict);
    void writeHistoryToFile(std::string filename, std::vector<double> history[], int size);
    void writeLLToFile(std::string filename, std::vector<double> lls, int size);
    void writeHistoryToDir(std::string dir, std::vector<double> history[], int size);
    void writeLLToDir(std::string dir, std::vector<double> lls, int size);
    void writeGraphLLToDir(std::string dir, std::vector<double> history[], int size);
void writeTimesToDir(std::string dir, std::vector<double> lls, int size);

double ComputeInsideCount(ChartItem *root);
double sanitizeLogProb(double logProb);

void load_weights(std::vector<SHRG*> shrg_rules, std::string file_name);
std::map<int, std::vector<double>> load_weights(std::string file_name);
void apply_weights(std::map<int, std::vector<double>> weights, std::vector<SHRG*>shrg_rules, int iter);
void exportMapToFile(const std::map<std::string, std::vector<int>>& map, const std::string& filename);
void clear_deriv_flags(ChartItem *root);
}


#endif // SHRG_GRAPH_PARSER_EM_UTILS_HPP

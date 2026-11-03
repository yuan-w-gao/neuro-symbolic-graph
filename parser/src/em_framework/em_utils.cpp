//
// Created by Yuan Gao on 04/06/2024.
//

#include <fstream>
#include <limits>
#include <cctype>
#include <cerrno>
#include <cmath>
#include "em_utils.hpp"

namespace shrg{
using namespace std;

// Helper for robust double parsing (handles subnormals, inf, nan)
static inline void trim_inplace_util(std::string& s) {
    auto issp = [](unsigned char c){ return std::isspace(c); };
    while (!s.empty() && issp(s.front())) s.erase(s.begin());
    while (!s.empty() && issp(s.back()))  s.pop_back();
}

static inline bool parse_double_robust_util(const std::string& raw, double& out) {
    std::string s = raw;
    trim_inplace_util(s);
    if (s.empty()) return false;

    if (s == "inf" || s == "+inf") { out = std::numeric_limits<double>::infinity(); return true; }
    if (s == "-inf") { out = -std::numeric_limits<double>::infinity(); return true; }
    if (s == "nan") { out = std::numeric_limits<double>::quiet_NaN(); return true; }

    errno = 0;
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return end != s.c_str();
}

constexpr static const double epsilon = 1e-10;
double addLogs(double a, double b){
    if(a == ChartItem::log_zero){
        return b;
    }
    if (b == ChartItem::log_zero)
        return a;
    if(a > b){
        return a + log1p(exp(b - a));
    }
    return b + log1p(exp(a - b));
}

bool is_negative(double log_prob){
    if(log_prob <= 0){
        return true;
    }
    return false;
}

double sanitizeLogProb(double logProb) {

    if (logProb <= 0) {
        return logProb;
    }

    // if (logProb > epsilon) {
    //     std::cout << "ERROR: Log probability is significantly positive: " << logProb << std::endl;
    //     // throw std::runtime_error("Log probability is significantly positive");
    // }else{
    //     std::cout << "WARNING: Log prob small positive: " << logProb << std::endl;
    // }

    return 0.0;  // Clamp small positives to 0
}

ChartItem* getParent(ParentTup &p){
    return std::get<0>(p);
}

std::vector<ChartItem*> getSiblings(ParentTup &p){
    return std::get<1>(p);
}

void setInitialWeights(LabelToRule &dict){
    LabelToRule::iterator it;
    for(it = dict.begin(); it != dict.end(); it++){
        RuleVector v = it->second;
        for(auto r:v){
            r->log_rule_weight = std::log(1.0/v.size());
        }
    }
}

void writeHistoryToFile(std::string filename, std::vector<double> history[], int size){
    std::ofstream outFile(filename);
    for(int i = 0; i < size; i ++){
        outFile << i;
        for(double j : history[i]){
            outFile << "," << j;
        }
        outFile << "\n";
    }
    outFile.close();
}

void writeLLToFile(std::string filename, std::vector<double> lls, int size){
    std::ofstream outFile(filename);
    for(int i = 0; i < size; i++){
        outFile << lls[i] << "\n";
    }
    outFile.close();
}

bool is_normal_count(double c){
    if(!isnormal(c) && c != 0.0 && c != ChartItem::log_zero){
        return false;
    }
    return true;
}

void writeHistoryToDir(std::string dir, std::vector<double> history[], int size){
    std::string filename = dir + "weight_history";
    std::ofstream outFile(filename);
    for(int i = 0; i < size; i ++){
        outFile << i;
        for(double j : history[i]){
            outFile << "," << j;
        }
        outFile << "\n";
    }
    outFile.close();
}

void writeLLToDir(std::string dir, std::vector<double> lls, int size){
    std::string filename = dir + "lls";
    std::ofstream outFile(filename);
    for(int i = 0; i < size; i++){
        outFile << lls[i] << "\n";
    }
    outFile.close();
}

void writeTimesToDir(std::string dir, std::vector<double> lls, int size){
    std::string filename = dir + "time_diff";
    std::ofstream outFile(filename);
    for(int i = 0; i < size; i++){
        outFile << lls[i] << "\n";
    }
    outFile.close();
}

void writeGraphLLToDir(std::string dir, std::vector<double> history[], int size){
    std::string filename = dir + "graph_ll";
    std::ofstream outFile(filename);
    for(int i = 0; i < size; i ++){
        outFile << i;
        for(double j : history[i]){
            outFile << "," << j;
        }
        outFile << "\n";
    }
    outFile.close();
}

int letterToIndex(char letter) {
    return letter - 'a';
}

void loadValidationFile(string filename,
              vector<vector<int>>& transitionMatrix,
              unordered_map<int, int>& endProb,
              unordered_map<int, int>& startingProb,
              vector<vector<int>>& omissionMatrix) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Could not open the file!" << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string fromStateStr, toStateStr;
        int shrgIndex;
        ss >> fromStateStr >> toStateStr >> shrgIndex;

        if (fromStateStr == "None") {
            continue;
        }

        int fromState = -1;
        if (fromStateStr == "ROOT") {
            fromState = -2;
        } else if (fromStateStr != "END") {
            fromState = stoi(fromStateStr);
        }

        if (toStateStr == "END") {
            if (fromState != -2) {  // Ignore ROOT to END
                endProb[fromState] = shrgIndex;
            }
        } else if (fromState == -2) {
            int toState = stoi(toStateStr);
            startingProb[toState] = shrgIndex;
        } else if (islower(toStateStr[0]) && toStateStr.size() == 1) {
            // It's an omission rule
            int toState = letterToIndex(toStateStr[0]);
            if (omissionMatrix.size() <= fromState) {
                omissionMatrix.resize(fromState + 1, vector<int>(26, -1));
            }
            omissionMatrix[fromState][toState] = shrgIndex;
        } else {
            int toState = stoi(toStateStr);
            if (transitionMatrix.size() <= toState) {
                transitionMatrix.resize(toState + 1, vector<int>(transitionMatrix.size(), -1));
            }
            if (transitionMatrix[toState].size() <= fromState) {
                transitionMatrix[toState].resize(fromState + 1, -1);
            }
            transitionMatrix[toState][fromState] = shrgIndex;
        }
    }

    file.close();
}

double ComputeInsideCount(ChartItem *root) {
    if (root->inside_visited_status == VISITED) {
       return root->log_inside_count;
    }

    ChartItem *ptr = root;
    double log_inside = ChartItem::log_zero;

    do {
        double curr_log_inside = ptr->score;
        curr_log_inside = sanitizeLogProb(curr_log_inside);

        double log_children = 0.0;
        for (ChartItem *child:ptr->children) {
            log_children += ComputeInsideCount(child);
        }
        log_children = sanitizeLogProb(log_children);

        curr_log_inside += log_children;
        curr_log_inside = sanitizeLogProb(curr_log_inside);

        log_inside = addLogs(log_inside, curr_log_inside);
        log_inside = sanitizeLogProb(log_inside);

        ptr = ptr->next_ptr;
    }while (ptr != root);

    do {
        ptr->log_inside_count = log_inside;
        ptr->inside_visited_status = VISITED;
        ptr = ptr->next_ptr;
    }while (ptr != root);

    return log_inside;
}



void clear_flags_helper(ChartItem* root, std::unordered_set<ChartItem*> visited) {
    if (!root || visited.count(root) > 0) {
        return;
    }
    visited.insert(root);

    ChartItem *ptr = root;
    do {
        ptr->inside_visited_status = ChartItem::kEmpty;
        ptr->outside_visited_status = ChartItem::kEmpty;
        ptr->count_visited_status = ChartItem::kEmpty;
        ptr->child_visited_status = ChartItem::kEmpty;
        ptr->update_status = ChartItem::kEmpty;

        ptr->em_greedy_deriv = ChartItem::kEmpty;
        ptr->em_greedy_score = ChartItem::kEmpty;
        ptr->em_inside_deriv = ChartItem::kEmpty;
        ptr->em_inside_score = ChartItem::kEmpty;
        ptr->count_greedy_deriv = ChartItem::kEmpty;
        ptr->count_greedy_score = ChartItem:: kEmpty;
        ptr->count_inside_deriv = ChartItem::kEmpty;
        ptr->count_inside_score = ChartItem::kEmpty;

        for (auto child:ptr->children) {
            clear_flags_helper(child, visited);
        }
        ptr = ptr->next_ptr;
    }while (ptr != root);
}
void clear_all_flags(ChartItem *root) {
    std::unordered_set<ChartItem*> visited;
    clear_flags_helper(root, visited);
}

void clear_derivflags_helper(ChartItem* root, std::unordered_set<ChartItem*> visited) {
    if (!root || visited.count(root) > 0) {
        return;
    }
    visited.insert(root);

    ChartItem *ptr = root;
    do {
        ptr->em_greedy_deriv = ChartItem::kEmpty;
        ptr->em_greedy_score = ChartItem::kEmpty;
        ptr->em_inside_deriv = ChartItem::kEmpty;
        ptr->em_inside_score = ChartItem::kEmpty;
        ptr->count_greedy_deriv = ChartItem::kEmpty;
        ptr->count_greedy_score = ChartItem:: kEmpty;
        ptr->count_inside_deriv = ChartItem::kEmpty;
        ptr->count_inside_score = ChartItem::kEmpty;

        for (auto child:ptr->children) {
            clear_flags_helper(child, visited);
        }
        ptr = ptr->next_ptr;
    }while (ptr != root);
}

void clear_deriv_flags(ChartItem *root) {
    std::unordered_set<ChartItem*> visited;
    clear_derivflags_helper(root, visited);
}

void load_weights(std::vector<SHRG *>shrg_rules, std::string file_name) {
    OPEN_IFSTREAM(is, file_name, return);

    int rule_count;
    is >> rule_count;
    assert(rule_count == shrg_rules.size());

    for (int i = 0; i < rule_count; i++) {
        int rule_index;
        is >> rule_index;
        double weight;
        is >> weight;
        shrg_rules[rule_index]->log_rule_weight = weight;
    }
}

std::map<int, std::vector<double>> load_weights(std::string file_name) {
    std::map<int, std::vector<double>> weights;
    OPEN_IFSTREAM(is, file_name, return weights);


    std::string line;

    while (std::getline(is, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        int index = std::stoi(value);

        std::vector<double> values;
        while (std::getline(ss, value, ',')) {
            double val;
            if (parse_double_robust_util(value, val)) {
                values.push_back(val);
            }
        }

        weights[index] = values;
    }

    return weights;
}

void apply_weights(std::map<int, std::vector<double>> weights, std::vector<SHRG*>shrg_rules, int iter) {
    for (auto it = weights.begin(); it != weights.end(); ++it) {
        int ind = it->first;
        const std::vector<double> &value = it->second;
        shrg_rules[ind]->log_rule_weight = value[iter];
    }
}

void exportMapToFile(const std::map<std::string, std::vector<int>>& map, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    for (const auto& [key, vec] : map) {
        // Write key first
        outFile << key << " : ";

        // Write vector size
        outFile << vec.size() << " : ";

        // Write vector elements
        for (size_t i = 0; i < vec.size(); ++i) {
            outFile << vec[i];
            if (i < vec.size() - 1) {
                outFile << ",";
            }
        }
        outFile << "\n";
    }
    outFile.close();
}

}
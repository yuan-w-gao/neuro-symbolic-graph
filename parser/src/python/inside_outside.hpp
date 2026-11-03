//
// Created by Yuan Gao on 5/28/23.
//
#include "../manager.hpp"
#include <queue>
#include <map>
#include <vector>


namespace shrg{
    extern const double ZERO_LOG;
    extern const int VISITED_INSIDE_FLAG;

    double computeInside(ChartItem *root);
    void computeOutsideNode(ChartItem *root, std::queue<ChartItem*> *queue);
    void computeOutside(ChartItem *root_ptr);
    struct LessThanByLevel;
    void computeOutside2Node(Generator *generator, ChartItem *root, std::priority_queue<ChartItem*, std::vector<ChartItem*>, LessThanByLevel> *queue);
    void computeOutside2(Generator *generator, ChartItem *root_ptr);
    void trainSingleSentence(Generator *generator, ChartItem *root_ptr, double pw, std::map<Label, double> &label_prob);
    double trainIterate(Generator *generator, std::vector<ChartItem *> forests, std::vector<SHRG> const &grammars);
    void train(std::vector<SHRG> const &grammars, std::vector<ChartItem*> forests, Generator *generator, double threshold);
    void addParentPointer(ChartItem *root, Generator *generator, int level);
    void addRulePointer(ChartItem *root, Generator *generator, std::vector<SHRG *> &shrg_rules);
    void addRulePointer_online(ChartItem *root, Generator *generator, std::vector<SHRG *> &shrg_rules);
    void preCompute(ChartItem *root, Generator *generator, std::vector<SHRG *> &shrg_rules);
    void addParentPointerIter(ChartItem *root, Generator *generator, int level);
    void preComputeIter(ChartItem *root, Generator *generator, std::vector<SHRG *> &shrg_rules);
    void computeOutside2NodeIter(Generator *generator, ChartItem *root, std::priority_queue<ChartItem*, std::vector<ChartItem*>, LessThanByLevel> *queue);
    void computeOutside2Iter(Generator *generator, ChartItem *root_ptr);
    void syncParentPointer(ChartItem* root, Generator *generator);
    void EM(std::vector<SHRG *> &shrg_rules, std::vector<EdsGraph> &graphs,
        Context *context, double threshold);
}

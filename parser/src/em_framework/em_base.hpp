//
// Created by Yuan Gao on 03/06/2024.
//

#ifndef SHRG_GRAPH_PARSER_EM_H
#define SHRG_GRAPH_PARSER_EM_H

#include <utility>

#include "../graph_parser/generator.hpp"
#include "../graph_parser/parser_chart_item.hpp"
#include "../manager.hpp"
#include "em_types.hpp"
//#include <queue>

namespace shrg {
namespace em {

class EMBase {
  public:
    static const int VISITED;
    EMBase(RuleVector &shrg_rules, std::vector<EdsGraph> &graphs, Context *context,
           double threshold)
        : graphs(graphs), shrg_rules(shrg_rules), context(context), threshold(threshold) {
        ll = 0;
        output_dir = "N";
        generator = context->parser->GetGenerator();
    }

    Generator* getGenerator() { return generator; }
    std::vector<EdsGraph>& getGraphs(){return graphs;}
    std::vector<ChartItem*> &getForests(){return forests;}
    Context* getContext(){return context;}
    std::vector<std::string>& getLemmaSentences(){return lemmas;}
    RuleVector &getRules(){return shrg_rules;}

    void unloadGraphs(){graphs.clear();}
    void setGraphs(std::vector<EdsGraph> &new_graphs){graphs = new_graphs;}

    void addParentPointer(ChartItem *root, int level);
    void addChildren(ChartItem* root);
    void addParentPointerOptimized(ChartItem *root, int level);
    void addRulePointer(ChartItem *root);
    double computeInside(ChartItem *root);
    void computeOutsideNode(ChartItem *root, NodeLevelPQ &pq);
    void computeOutside(ChartItem *root);
    virtual void run() = 0;

    float FindBestScoreWeight(ChartItem *root_ptr);
    Derivation& FindBestDerivation_EMGreedy(ChartItem *root_ptr);
  protected:
    std::vector<EdsGraph> &graphs;
    RuleVector &shrg_rules;
    Context *context;
    Generator *generator;
    double threshold;
    double ll;
    std::string output_dir;
    std::vector<ChartItem*> forests;
    std::vector<std::string> lemmas;
    int time_out_in_seconds = 5;

    virtual bool converged() const = 0;
    virtual void computeExpectedCount(ChartItem *root, double pw) = 0;
    virtual void updateEM() = 0;

    void clearRuleCount();
};

}
}
#endif // SHRG_GRAPH_PARSER_EM_H

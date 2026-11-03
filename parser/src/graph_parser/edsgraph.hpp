#pragma once

#include <array>
#include <string>
#include <vector>

#include "hyper_graph.hpp"
#include "tokens.hpp"

namespace shrg {

using Lemma = int;
using POSTag = char;
using Sense = int;

struct GraphNode {
    Label label;
    POSTag pos_tag;
    std::string lemma;
    std::string sense;
    std::string carg;
    std::array<std::string, 5> properties;
    std::string id;

    bool is_lexical = false;
};

class EdsGraph : public HyperGraph<GraphNode> {
  public:
    std::string sentence;
    std::string lemma_sequence;
    std::string sentence_id;
    int top_index = -1;

    static bool Load(const std::string &input_file, std::vector<EdsGraph> &edsgraphs,
                     TokenSet &label_set);

    static bool Load(const std::string &input_file, std::vector<EdsGraph> &edsgraphs,
                     TokenSet &label_set, //
                     std::vector<int> &random, bool sort_edsgraphs = false);
};

} // namespace shrg

// Local Variables:
// mode: c++
//  End:

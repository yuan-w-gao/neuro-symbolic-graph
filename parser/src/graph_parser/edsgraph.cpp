#include <algorithm>
#include <fstream>
#include <limits>
#include <random>

#include "edsgraph.hpp"

static const std::unordered_map<std::string, std::string> TABLE{
    {"Jan", "January"},  {"Feb", "February"},  {"Mar", "March"},       {"Apr", "April"},
    {"Jun", "June"},     {"Jul", "July"},      {"Aug", "August"},      {"Sep", "September"},
    {"Oct", "October"},  {"Nov", "November"},  {"Dec", "December"},    {"Mon", "Monday"},
    {"Tue", "Tuesday"},  {"Wed", "Wednesday"}, {"Thu", "Thursday"},    {"Fri", "Friday"},
    {"Sat", "Saturday"}, {"Sun", "Sunday"},    {"1", "first"},         {"2", "second"},
    {"3", "third"},      {"4", "fourth"},      {"5", "fifth"},         {"6", "sixth"},
    {"7", "seventh"},    {"8", "eighth"},      {"9", "ninth"},         {"10", "tenth"},
    {"11", "eleventh"},  {"12", "twelfth"},    {"13", "thirteenth"},   {"14", "fourteenth"},
    {"15", "fifteenth"}, {"16", "sixteenth"},  {"17", "seventeenth"},  {"18", "eighteenth"},
    {"19", "nineteen"},  {"20", "twentieth"},  {"30", "thirtieth"},    {"40", "fortieth"},
    {"50", "fiftieth"},  {"60", "sixtieth"},   {"70", "seventieth"},   {"80", "eightieth"},
    {"90", "ninetieth"}, {"100", "hundredth"}, {"1000", "thousandth"}, {"1000000", "millionth"}};

namespace shrg {

void PostProcess(EdsGraph::Node &node, TokenSet &label_set) {
    auto MOFY_INDEX = label_set.Index("mofy");
    auto DOFW_INDEX = label_set.Index("dofw");
    auto ORD_INDEX = label_set.Index("ord");

    if (node.carg == "1000000")
        node.carg = "million";
    else if (node.carg == "1000000000")
        node.carg = "billion";
    else if (node.carg == "1000000000000")
        node.carg = "trillion";
    else if (node.carg == "US")
        node.carg = "u.s.";
    else if (node.label == MOFY_INDEX || node.label == DOFW_INDEX ||
             node.label == ORD_INDEX) { // abbrev
        auto it = TABLE.find(node.carg);
        if (it != TABLE.end())
            node.carg = it->second;
    }

    auto pos = node.lemma.find('/');
    if (pos != std::string::npos)
        node.lemma.erase(pos);
    if (node.lemma.back() == '-')
        node.lemma.pop_back();
    if (node.carg.back() == '-')
        node.carg.pop_back();
}

std::istream &LoadEdsGraph(std::istream &is, EdsGraph &edsgraph, TokenSet &label_set) {
    using Node = EdsGraph::Node;
    using Edge = EdsGraph::Edge;

    int node_count;
    std::string token;

    is >> edsgraph.sentence_id;
    is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(is, edsgraph.sentence);
    std::getline(is, edsgraph.lemma_sequence);
    is >> node_count;

    if (!is || edsgraph.sentence.empty()) {
        is.setstate(std::ios_base::failbit);
        return is;
    }
    edsgraph.nodes.resize(node_count);
    for (int i = 0; i < node_count; ++i) {
        int ignored_index;
        Node &node = edsgraph.nodes[i];

        is >> ignored_index >> node.id >> token;
        node.is_lexical = !token.empty() && token[0] == '_';
        node.label = label_set.Get(token);
        if (node.label == -1) {
            auto pos = std::string::npos;
            if (node.is_lexical)
                pos = token.find('_', 1);
            if (pos != std::string::npos) {
                token.replace(1, pos - 1, "X");
                node.label = label_set.Get(token);
            }
        }

        is >> node.lemma;
        is >> node.pos_tag;
        is >> node.sense;
        is >> node.carg;

        for (int i = 0; i < 5; ++i)
            is >> node.properties[i];

        PostProcess(node, label_set);
    }

    int edge_count;
    is >> edsgraph.top_index >> edge_count;
    edsgraph.edges.resize(edge_count + node_count);
    for (int i = 0; i < node_count; ++i) {
        Edge &edge = edsgraph.edges[i];
        Node &node = edsgraph.nodes[i];

        node.index = i;
        node.linked_edges.push_back(&edge);

        edge.index = i;
        edge.linked_nodes.push_back(&node);
        edge.label = edsgraph.nodes[i].label;
        edge.is_terminal = true;
    }

    for (int i = 0; i < edge_count; ++i) {
        Edge &edge = edsgraph.edges[i + node_count];
        int from_index, to_index;
        is >> from_index >> to_index >> token;

        Node &from_node = edsgraph.nodes[from_index];
        Node &to_node = edsgraph.nodes[to_index];
        from_node.linked_edges.push_back(&edge);
        to_node.linked_edges.push_back(&edge);

        edge.index = i + node_count;
        edge.linked_nodes.push_back(&from_node);
        edge.linked_nodes.push_back(&to_node);
        edge.label = label_set.Index(token);
        edge.is_terminal = true;
    }
    return is;
}

bool EdsGraph::Load(const std::string &input_file, std::vector<EdsGraph> &edsgraphs,
                    TokenSet &label_set) {
    OPEN_IFSTREAM(is, input_file, return false);

    LOG_INFO("Loading Graphs ... < " << input_file);

    int graph_count;
    is >> graph_count;
    edsgraphs.resize(graph_count);

    bool flag = true;
    for (EdsGraph &edsgraph : edsgraphs)
        if (!LoadEdsGraph(is, edsgraph, label_set)) {
            flag = false;
            break;
        }

    is.close();

    if (flag)
        LOG_INFO("Loaded " << graph_count << " Graphs");

    return flag;
}

bool EdsGraph::Load(const std::string &input_file, std::vector<EdsGraph> &edsgraphs,
                    TokenSet &label_set, //
                    std::vector<int> &random, bool sort_edsgraphs) {
    if (!Load(input_file, edsgraphs, label_set))
        return false;

    int graph_count = edsgraphs.size();
    random.reserve(graph_count);
    for (int i = 0; i < graph_count; ++i)
        random.push_back(i);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(random.begin(), random.end(), g);
//    std::random_shuffle(random.begin(), random.end());

    if (sort_edsgraphs)
        std::stable_sort(random.begin(), random.end(), //
                         [&edsgraphs](int i, int j) {
                             return edsgraphs[i].nodes.size() < edsgraphs[j].nodes.size();
                         });
    return true;
}

} // namespace shrg

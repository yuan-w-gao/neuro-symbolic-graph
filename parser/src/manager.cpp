#include "manager.hpp"

#include "graph_parser/parser_linear.hpp"
#include "graph_parser/parser_tree_index_v1.hpp"
#include "graph_parser/parser_tree_index_v2.hpp"
#include "graph_parser/parser_tree_v1.hpp"
#include "graph_parser/parser_tree_v2.hpp"

namespace shrg {

Manager Manager::manager;

using TreeSHRGParserV1 = tree_v1::TreeSHRGParser;
using TreeSHRGParserV2 = tree_v2::TreeSHRGParser;
using IndexedTreeSHRGParserV1 = tree_index_v1::TreeSHRGParser;
using IndexedTreeSHRGParserV2 = tree_index_v2::TreeSHRGParser;

void Initialize() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::srand(std::time(nullptr));
}

bool Manager::LoadGrammars(const std::string &input_file, const std::string &filter) {
    grammars.clear();
    shrg_rules.clear();
    label_set.Clear();

    int num_shrg_rules = SHRG::Load(input_file, grammars, label_set);
    if (num_shrg_rules == 0)
        return false;

    label_set.Freeze();

    if (filter == "disconnected")
        SHRG::FilterDisconneted(grammars);
    else if (filter != "none")
        throw std::runtime_error("Unsupported filter: " + filter);

    shrg_rules.resize(num_shrg_rules, nullptr);
    for (auto &grammar : grammars)
        for (auto &cfg_rule : grammar.cfg_rules)
            shrg_rules[cfg_rule.shrg_index] = &grammar;

    return true;
}

bool Manager::LoadGraphs(const std::string &input_file) {
    edsgraphs.clear();
    return EdsGraph::Load(input_file, edsgraphs, label_set);
}

void Manager::Allocate(uint num_contexts) {
    uint old_size = contexts.size();
    if (num_contexts <= old_size)
        return;
    contexts.resize(num_contexts);
    for (uint i = old_size; i < num_contexts; ++i) {
        contexts[i] = new Context(this);
        contexts[i]->manager_ptr = this;
    }
}

Manager::~Manager() {
    for (auto context_ptr : contexts)
        delete context_ptr;
}

template <typename Parser>
std::unique_ptr<Parser> CreateTreeParser(const std::vector<SHRG> &grammars,
                                         const std::string &decomposer_type,
                                         const TokenSet &label_set) {
    using namespace tree;
    using Node = typename Parser::TreeNode;

    if (decomposer_type.empty() || decomposer_type == "naive")
        return std::make_unique<Parser>(grammars, TreeDecomposerTpl<Node, NaiveDecomposer>(),
                                        label_set);
    else if (decomposer_type == "terminal_first")
        return std::make_unique<Parser>(grammars, //
                                        TreeDecomposerTpl<Node, TerminalFirstDecomposer>(),
                                        label_set);
    else if (decomposer_type == "best")
        return std::make_unique<Parser>(grammars, //
                                        TreeDecomposerTpl<Node, MinimumWidthDecomposer>(),
                                        label_set);

    throw std::runtime_error("Unknown decomposer type: " + decomposer_type);
}

void Context::Init(const std::string &type, bool verbose, uint max_pool_size) {
    auto &grammars = manager_ptr->grammars;
    auto &label_set = manager_ptr->label_set;
    if (type == "linear")
        parser = std::make_unique<linear::LinearSHRGParser>(grammars, label_set);
    else {
        auto pos = type.find('/');
        std::string parser_type(type);
        std::string decomposer_type;
        if (pos != std::string::npos) {
            parser_type = type.substr(0, pos);
            decomposer_type = type.substr(pos + 1);
        }

        if (parser_type == "tree_v1")
            parser = CreateTreeParser<TreeSHRGParserV1>(grammars, decomposer_type, label_set);
        else if (parser_type == "tree_v2")
            parser = CreateTreeParser<TreeSHRGParserV2>(grammars, decomposer_type, label_set);
        else if (parser_type == "tree_index_v1")
            parser =
                CreateTreeParser<IndexedTreeSHRGParserV1>(grammars, decomposer_type, label_set);
        else if (parser_type == "tree_index_v2")
            parser =
                CreateTreeParser<IndexedTreeSHRGParserV2>(grammars, decomposer_type, label_set);
        else {
            parser.release();
            throw std::runtime_error("Unknown parser type: " + type);
        }
    }

    this->type = type;
    parser->SetVerbose(verbose);
    parser->SetPoolSize(max_pool_size);
}

std::size_t Context::CountChartItems(ChartItem *chart_item_ptr) {
    if (!Check())
        throw std::runtime_error("empty context");

    if (!chart_item_ptr)
        chart_item_ptr = parser->Result();

    if (!chart_item_ptr->attrs_ptr)
        throw std::runtime_error("empty chart item");

    return parser->GetGenerator()->CountChartItems(chart_item_ptr);
}

const ChartItem *Context::Result() {
    if (!Check())
        return nullptr;
    return parser->Result();
}

ParserError Context::Parse(int index) { return Parse(manager_ptr->edsgraphs[index]); }

ParserError Context::Parse(const EdsGraph &graph) {
    if (!Check())
        return ParserError::kUnInitialized;
    Clear();
    return parser->Parse(graph);
}

bool Context::Generate() {
    if (!Check())
        return false;
    auto generator = parser->GetGenerator();
    if (best_item_ptr == nullptr)
        best_item_ptr = generator->BestResult();

    if (!best_item_ptr)
        return false;

    derivation.clear();
    sentence.clear();
    return generator->Generate(best_item_ptr, derivation, sentence) >= 0;
}

} // namespace shrg

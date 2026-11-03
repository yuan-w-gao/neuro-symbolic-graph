#pragma once

#include "graph_parser/generator.hpp"
#include "graph_parser/parser_base.hpp"

namespace shrg {

void Initialize();

class Manager;
class Context {
    friend class Manager;

  private:
    explicit Context(const Manager *manager) : manager_ptr(manager){};

    void Clear() {
        best_item_ptr = nullptr;
        sentence.clear();
        derivation.clear();
    }

  public:
    const Manager *manager_ptr = nullptr;

    std::string type;
    std::unique_ptr<SHRGParserBase> parser;
    std::string sentence;
    Derivation derivation;
    ChartItem *best_item_ptr = nullptr;

    bool Check() const {
        if (!parser.get()) {
            LOG_ERROR("parser is not created yet");
            return false;
        }
        return true;
    }

    void Init(const std::string &type, bool verbose = true, uint max_pool_size = 50);

    void ReleaseMemory() {
        if (Check()) {
            Clear();
            parser->MemoryPool().Reset();
        }
    }

    std::size_t PoolSize() const { return parser->MemoryPool().Size(); }

    const ChartItem *GetChartItem(std::size_t index) const {
        if (index >= PoolSize())
            throw std::runtime_error("Out of range");
        return parser->MemoryPool()[index];
    }

    ParserError Parse(int index);

    ParserError Parse(const EdsGraph &edsgraph);

    bool Generate();

    void SetChartItem(ChartItem &chart_item) { best_item_ptr = &chart_item; }

    std::size_t CountChartItems(ChartItem *chart_item_ptr = nullptr);

    const ChartItem *Result();
};

class Manager {
  private:
    Manager(){};

  public:
    ~Manager();

    std::vector<SHRG *> shrg_rules;
    std::vector<Context *> contexts;
    std::vector<SHRG> grammars;
    std::vector<EdsGraph> edsgraphs;
    TokenSet label_set;

    std::map<std::string, std::vector<int>> gold_derivations;

    bool LoadGrammars(const std::string &input_file, const std::string &filter = "disconnected");

    bool LoadGraphs(const std::string &input_file);

    bool LoadDerivations(const std::string &input_file);

    void Allocate(uint num_contexts = 1);

    // init all context
    void InitAll(const std::string &type, bool verbose = true, uint max_pool_size = 25) {
        for (auto context_ptr : contexts)
            context_ptr->Init(type, verbose, max_pool_size);
    }

    static Manager manager;
};

} // namespace shrg

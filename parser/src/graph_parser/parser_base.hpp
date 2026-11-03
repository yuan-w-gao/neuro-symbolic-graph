#pragma  once

#include "../include/memory_utils.hpp"

#include "sparsehash/dense_hash_map"

#include "parser_debug.hpp"
#include "parser_utils.hpp"

namespace shrg {

using ChartItemList = std::vector<ChartItem *>;
class ChartItemSet {
  private:
    std::vector<ChartItem *> items_;
    google::dense_hash_map<Ref<ChartItem>, int> item_to_index_;

    static const Ref<ChartItem> empty_key_;

  public:
    using iterator = decltype(items_.begin());
    using const_iterator = decltype(items_.cbegin());

    ChartItemSet() { // google dense_hash_map need an empty_key
        item_to_index_.set_empty_key(empty_key_);
    }

    const ChartItemList &AsList() const { return items_; }

    void Clear() {
        items_.clear();
        item_to_index_.clear(); // TODO: optimize memory use
    }

    iterator begin() { return items_.begin(); }
    iterator end() { return items_.end(); }
    const_iterator begin() const { return items_.cbegin(); }
    const_iterator end() const { return items_.cend(); }

    const ChartItem *operator[](int index) const { return items_[index]; }
    ChartItem *operator[](int index) { return items_[index]; }

    const decltype(items_) &Graphs() { return items_; }

    std::size_t Size() const { return items_.size(); }

    bool Empty() const { return items_.empty(); }

    // Try to insert a new subgraph
    bool TryInsert(ChartItem *chart_item_ptr) {
        Ref<ChartItem> subgraph_ref(*chart_item_ptr);
        auto result = item_to_index_.insert({subgraph_ref, items_.size()});
        if (!result.second) { // insertion failed
            // Because a subgraph may have multiple derivations, the `subgraph' here is actually a
            // tuple of subgraph and derivation. (graph part and derivation part) But if two such
            // tuples have the same graph part, they can be viewed the as the same `subgraph` in
            // the rest of parsing process. So here we keep only one entry for these subgraphs.

            // this subgraph should not be already inserted into cycle list
            assert(!chart_item_ptr->next_ptr);

            items_[result.first->second]->Push(chart_item_ptr);

            assert(chart_item_ptr->next_ptr); // this subgraph should be inserted into cycle list
            return false;
        }
        items_.push_back(chart_item_ptr);
        if (!chart_item_ptr->next_ptr)
            chart_item_ptr->next_ptr = chart_item_ptr;
        return true;
    }
};

struct GrammarAttributes {
    const SHRG *grammar_ptr = nullptr; // points to the grammar

    const SHRG::CFGRule *SelectRule(const ChartItem *chart_item_ptr) {
        int cfg_index = chart_item_ptr->status;
        if (cfg_index < 0) // select the most frequently used rule
            return grammar_ptr->best_cfg_ptr;
        return &grammar_ptr->cfg_rules[cfg_index];
    }
};

enum class ParserError {
    kNone = 0,
    kNoResult = 1,
    kOutOfMemory = 2,
    kTooLarge = 3,
    kUnInitialized = 4,
    kUnknown = 5
};

constexpr const char *ToString(ParserError v) {
    switch (v) {
    case ParserError::kNone:
        return "None";
    case ParserError::kNoResult:
        return "NoResult";
    case ParserError::kUnknown:
        return "Unknown";
    case ParserError::kTooLarge:
        return "TooLarge";
    case ParserError::kOutOfMemory:
        return "OutOfMemory";
    default:
        return "???";
    }
}

#define DEFINE_GETTER(modifier, type, name, getter)                                                \
  public:                                                                                          \
    type Get##getter() const { return name; }                                                      \
    modifier:                                                                                      \
    type name;

class Generator;
class SHRGParserBase {
    friend class Generator;

  protected:
    const char *parser_type_;
    bool verbose_ = true;
    uint max_pool_size_ = 1024; // unlimited

    DEFINE_GETTER(protected, uint64_t, num_grammars_available_, NumGrammarsAvailable);
    DEFINE_GETTER(protected, uint64_t, num_terminal_subgraphs_, NumTerminalSubgraphs);
    DEFINE_GETTER(protected, uint64_t, num_passive_items_, NumPassiveItems);
    DEFINE_GETTER(protected, uint64_t, num_active_items_, NumActiveItems);
    DEFINE_GETTER(protected, uint64_t, num_succ_merge_operations_, NumSuccMergeOps);
    DEFINE_GETTER(protected, uint64_t, num_total_merge_operations_, NumTotalMergeOps);
    DEFINE_GETTER(protected, uint64_t, num_indexing_keys_, NumIndexingKeys);

    // point to current edsgraph
    const EdsGraph *graph_ptr_;
    EdgeSet all_edges_in_graph_;

    // point to SHRG grammars
    const std::vector<SHRG> &grammars_;
    const TokenSet &label_set_;

    // the only start symbol in grammars
    Label start_symbol_;
    ChartItem *matched_item_ptr_; // head pointer of parsing results
    utils::MemoryPool<ChartItem> items_pool_;

    ParserError BeforeParse(const EdsGraph &graph);

    void ClearChart();

    template <typename ResultMap> void SetCompleteItem(const ResultMap &results, int label_offset) {
        auto it = results.find(static_cast<uint64_t>(start_symbol_) << label_offset);
        if (it != results.end()) {
            for (auto chart_item_ptr : it->second.passive_items)
                if (chart_item_ptr->edge_set == all_edges_in_graph_) {
                    // there should be only one possible solution
                    assert(matched_item_ptr_ == nullptr);
                    matched_item_ptr_ = chart_item_ptr;
#ifdef NDEBUG
                    // check in DEBUG mode
                    break;
#endif
                }
        }
    }

  public:
    SHRGParserBase(const char *parser_type, const std::vector<SHRG> &grammars,
                   const TokenSet &label_set)
        : parser_type_(parser_type), //
          grammars_(grammars),       //
          label_set_(label_set),     //
          start_symbol_(label_set.Get("ROOT")) {}

    utils::MemoryPool<ChartItem> &MemoryPool() { return items_pool_; };

    void SetVerbose(bool verbose) { verbose_ = verbose; }
    void SetStartSymbol(Label start_symbol) { start_symbol_ = start_symbol; }
    void SetPoolSize(uint max_pool_size) { max_pool_size_ = max_pool_size; }

    const EdsGraph *Graph() const { return graph_ptr_; }
    void SetGraph(const EdsGraph* graph) { graph_ptr_ = graph; }
    bool IsVerbose() const { return verbose_; }
    const char *Type() const { return parser_type_; }

    ChartItem *Result() { return matched_item_ptr_; }

    virtual ParserError Parse(const EdsGraph &graph) = 0;

    virtual Generator *GetGenerator() = 0;

    virtual const ChartItemList *GetItemsByLabelHash(LabelHash label_hash) { return nullptr; }

    virtual ~SHRGParserBase() {}
};

// Precompute boundary nodes of a production
void PrecomputeBoundaryNodesForHRG(NodeMapping &boundary_nodes_of_hrg, const SHRG &grammar,
                                   const EdgeSet &matched_edges);

} // namespace shrg

// Local Variables:
// mode: c++
// End:

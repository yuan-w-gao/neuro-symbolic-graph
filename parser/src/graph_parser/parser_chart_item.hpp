#pragma once

#include <array>
#include <bitset>
#include <boost/functional/hash.hpp>
// #include <boost/functional/hash/hash_fix.hpp>
#include <unordered_set>
#include <vector>
#include <tuple>

#include "edsgraph.hpp"
#include "synchronous_hyperedge_replacement_grammar.hpp"

namespace shrg {

const int MAX_GRAPH_EDGE_COUNT = 256;
const int MAX_GRAPH_NODE_COUNT = 256;
const int MAX_GRAMMAR_BOUNDARY_NODE_COUNT = 16;

using EdgeSet = std::bitset<MAX_GRAPH_EDGE_COUNT>;
using NodeSet = std::bitset<MAX_GRAPH_NODE_COUNT>;
union NodeMapping {
    using T1 = std::array<uint8_t, MAX_GRAMMAR_BOUNDARY_NODE_COUNT>;
    using T4 = std::array<uint32_t, MAX_GRAMMAR_BOUNDARY_NODE_COUNT / 4>;
    using T8 = std::array<uint64_t, MAX_GRAMMAR_BOUNDARY_NODE_COUNT / 8>;
    T1 m1;
    T4 m4;
    T8 m8;

    uint8_t &operator[](int index) { return m1[index]; }

    uint8_t operator[](int index) const { return m1[index]; }

    bool operator==(const NodeMapping &other) const { return m8 == other.m8; }

    constexpr std::size_t size() const noexcept { return MAX_GRAMMAR_BOUNDARY_NODE_COUNT; }
};

template <typename T> class Ref {
  public:
    Ref() : ptr_(nullptr) {}
    explicit Ref(const T &t) : ptr_(std::addressof(t)) {}
    Ref(T &&t) = delete;

    operator T &() const { return *ptr_; }
    const T &get() const { return *ptr_; }
    const T *get_pointer() const { return ptr_; }

  private:
    const T *ptr_;
};

template <typename T> Ref<T> GetRef(const T &t) { return Ref<T>(t); }

struct GrammarAttributes;

class ChartItem {
  public:
    static const int kEmpty = -1;
    static const int kExpanded = -100;
    static const int kVisited = -1000;
    static constexpr double ZERO_LOG = 3000.0;
    static constexpr double log_zero = -std::numeric_limits<double>::infinity();

  public:
    GrammarAttributes *attrs_ptr; // the attributes of the grammar that the item belongs to

    ChartItem *next_ptr = nullptr;
    ChartItem *left_ptr = nullptr;
    ChartItem *right_ptr = nullptr;

    EdgeSet edge_set; // edge set of EdsGraph::Edge
    // mapping from boundary nodes of SHRG (SHRG::Node) to boundary nodes of EDS (EdsGraph::Node,
    // the index starts from 1)
    NodeMapping boundary_node_mapping;

    std::vector<ChartItem*> children;
    std::vector<std::tuple<ChartItem*, std::vector<ChartItem*>>> parents_sib;

    int level = -1;

    float score = 1.0; // initially above zero
    int status = kEmpty;
    int inside_visited_status = kEmpty;
    int outside_visited_status = kEmpty;
    int count_visited_status = kEmpty;
    int child_visited_status = kEmpty;
    int update_status = kEmpty;

    int em_greedy_score = kEmpty;
    int em_greedy_deriv = kEmpty;
    int em_inside_score = kEmpty;
    int em_inside_deriv = kEmpty;
    int count_greedy_score = kEmpty;
    int count_greedy_deriv = kEmpty;
    int count_inside_score = kEmpty;
    int count_inside_deriv = kEmpty;

    double log_inside_prob = 0.0;
    double log_outside_prob = log_zero;
    double log_sent_rule_count = log_zero;
    double log_inside_count = 0.0;


    int shrg_index = -1;
    shrg::SHRG *rule_ptr = nullptr;

    int rule_visited = kEmpty;

    ChartItem() : attrs_ptr(nullptr), boundary_node_mapping{} {}

    explicit ChartItem(GrammarAttributes *attrs_ptr, //
                       const EdgeSet &edge_set_ = 0, //
                       const NodeMapping &node_mapping_ = {})
        : attrs_ptr(attrs_ptr), edge_set(edge_set_), boundary_node_mapping(node_mapping_) {}

    void Swap(ChartItem &other) {
        std::swap(attrs_ptr, other.attrs_ptr);
        std::swap(left_ptr, other.left_ptr);
        std::swap(right_ptr, other.right_ptr);
        std::swap(score, other.score);
        std::swap(status, other.status);

        std::swap(log_inside_prob, other.log_inside_prob);
        std::swap(log_outside_prob, other.log_outside_prob);
        std::swap(log_sent_rule_count, other.log_sent_rule_count);
        std::swap(inside_visited_status, other.inside_visited_status);
        std::swap(outside_visited_status, other.outside_visited_status);
        std::swap(count_visited_status, other.count_visited_status);
        std::swap(rule_visited, other.rule_visited);
        std::swap(child_visited_status, other.child_visited_status);
        std::swap(children, other.children);
        std::swap(parents_sib, other.parents_sib);
        std::swap(rule_ptr, other.rule_ptr);
        std::swap(shrg_index, other.shrg_index);
    }

    ChartItem *Pop() {
        ChartItem *ptr = next_ptr;
        next_ptr = nullptr;
        return ptr;
    }

    void Push(ChartItem *chart_item_ptr) {
        chart_item_ptr->next_ptr = next_ptr;
        next_ptr = chart_item_ptr;
    }


};

template <typename T> inline bool operator==(const Ref<T> &ref1, const Ref<T> &ref2) {
    return ref1.get() == ref2.get();
}

inline bool operator==(const ChartItem &v1, const ChartItem &v2) {
    return v1.edge_set == v2.edge_set && v1.boundary_node_mapping == v2.boundary_node_mapping;
}

} // namespace shrg

namespace std {

using shrg::ChartItem;
using shrg::NodeMapping;
using shrg::Ref;
template <> struct hash<Ref<ChartItem>> {
    // !!! the hash function must be mark as const
    std::size_t operator()(const Ref<ChartItem> &ref) const {
        auto &v = ref.get();
        auto hash_value = std::hash<decltype(v.edge_set)>()(v.edge_set);
        boost::hash_combine(hash_value, v.boundary_node_mapping);
        return hash_value;
    }
};

template <> struct hash<NodeMapping> {
    // !!! the hash function must be mark as const
    std::size_t operator()(const NodeMapping &key) const { return boost::hash_value(key.m8); }
};

} // namespace std

namespace boost {

using shrg::NodeMapping;
template <> struct hash<NodeMapping> {
    // !!! the hash function must be mark as const
    std::size_t operator()(const NodeMapping &key) const { return boost::hash_value(key.m8); }
};

} // namespace boost

REGISTER_TYPE_NAME(shrg::ChartItem);

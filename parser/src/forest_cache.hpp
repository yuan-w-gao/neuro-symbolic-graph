//
// Forest Caching for SHRG Parser
// Serializes derivation forests to disk to skip re-parsing across runs.
//

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <limits>

#include "graph_parser/parser_chart_item.hpp"
#include "graph_parser/parser_base.hpp"  // For GrammarAttributes
#include "include/memory_utils.hpp"

namespace shrg {
namespace forest_cache {

// Version number for cache format compatibility
constexpr uint32_t CACHE_VERSION = 1;

// Magic number to identify cache files
constexpr uint32_t CACHE_MAGIC = 0x46525354;  // "FRST"

/**
 * Serialized representation of a ChartItem.
 * Pointers are converted to indices for disk storage.
 */
struct SerializedNode {
    // EdgeSet is a std::bitset<256> which is 32 bytes
    uint64_t edge_set_data[4];  // 256 bits = 32 bytes

    // NodeMapping is 16 bytes
    uint8_t boundary_mapping[16];

    float score;
    int level;
    int shrg_index;

    // Pointer fields as indices (-1 if null)
    int32_t next_index;
    int32_t left_index;
    int32_t right_index;

    // Children count (children stored separately)
    uint32_t children_count;

    // Parents count (parents stored separately)
    uint32_t parents_count;

    // EM-related probabilities
    double log_inside_prob;
    double log_outside_prob;
    double log_sent_rule_count;
    double log_inside_count;

    // Derivation scoring fields
    int em_greedy_score;
    int em_greedy_deriv;
    int em_inside_score;
    int em_inside_deriv;
    int count_greedy_score;
    int count_greedy_deriv;
    int count_inside_score;
    int count_inside_deriv;

    SerializedNode()
        : edge_set_data{0}, boundary_mapping{0}, score(1.0f), level(-1),
          shrg_index(-1), next_index(-1), left_index(-1), right_index(-1),
          children_count(0), parents_count(0),
          log_inside_prob(0.0), log_outside_prob(-std::numeric_limits<double>::infinity()),
          log_sent_rule_count(-std::numeric_limits<double>::infinity()), log_inside_count(0.0),
          em_greedy_score(-1), em_greedy_deriv(-1), em_inside_score(-1), em_inside_deriv(-1),
          count_greedy_score(-1), count_greedy_deriv(-1), count_inside_score(-1), count_inside_deriv(-1) {}
};

/**
 * Serialized parent-sibling relationship.
 * Each parent has a reference and a list of sibling indices.
 */
struct SerializedParentSib {
    int32_t parent_index;
    std::vector<int32_t> sibling_indices;
};

/**
 * Header for a serialized forest file.
 */
struct ForestHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t grammar_hash;  // Simple hash for grammar validation
    uint32_t graph_hash;    // Simple hash for graph validation
    int32_t root_index;
    uint32_t node_count;
    uint32_t total_children;  // Total number of children across all nodes
    uint32_t total_parents;   // Total number of parent-sib entries across all nodes

    ForestHeader()
        : magic(CACHE_MAGIC), version(CACHE_VERSION), grammar_hash(0), graph_hash(0),
          root_index(-1), node_count(0), total_children(0), total_parents(0) {}
};

/**
 * ForestCache manages disk caching of derivation forests.
 *
 * Cache directory structure:
 *   {cache_dir}/
 *     metadata.bin       - Grammar hash and validation info
 *     forests/
 *       {graph_id}.bin   - Per-graph serialized forests
 */
class ForestCache {
public:
    /**
     * Create a ForestCache with the specified cache directory.
     * @param cache_dir Directory to store cached forests
     */
    explicit ForestCache(const std::string& cache_dir);

    /**
     * Set the grammar hash for validation.
     * Cached forests are only valid if the grammar hash matches.
     * @param hash Hash of the grammar file content
     */
    void set_grammar_hash(uint32_t hash);

    /**
     * Compute a simple hash for a string (grammar or graph content).
     */
    static uint32_t compute_hash(const std::string& content);

    /**
     * Check if a valid cached forest exists for the given graph.
     * @param graph_id Unique identifier for the graph
     * @param graph_hash Hash of the graph content
     * @return true if valid cache exists
     */
    bool has_valid_cache(const std::string& graph_id, uint32_t graph_hash) const;

    /**
     * Save a derivation forest to disk.
     * @param graph_id Unique identifier for the graph
     * @param graph_hash Hash of the graph content
     * @param root Root ChartItem of the forest
     */
    void save(const std::string& graph_id, uint32_t graph_hash, ChartItem* root);

    /**
     * Load a derivation forest from disk.
     * @param graph_id Unique identifier for the graph
     * @param graph_hash Hash of the graph content (for validation)
     * @param pool Memory pool to allocate loaded nodes in
     * @return Root ChartItem of the loaded forest, or nullptr if invalid/missing
     */
    ChartItem* load(const std::string& graph_id, uint32_t graph_hash,
                    utils::MemoryPool<ChartItem>& pool);

    /**
     * Clear all cached forests.
     */
    void clear();

    /**
     * Restore rule pointers on a loaded forest using shrg_index.
     * This is needed because attrs_ptr is not serialized.
     * @param root Root of the loaded forest
     * @param shrg_rules Vector of SHRG rule pointers indexed by shrg_index
     */
    static void restore_rule_pointers(ChartItem* root, const std::vector<SHRG*>& shrg_rules);

    /**
     * Restore all pointers (rule_ptr and attrs_ptr) on a loaded forest.
     * This makes the cached forest fully functional for sentence generation.
     * @param root Root of the loaded forest
     * @param shrg_rules Vector of SHRG rule pointers indexed by shrg_index
     * @param attrs_pool Pool of GrammarAttributes, one per rule (created by create_attrs_pool)
     */
    static void restore_all_pointers(ChartItem* root, const std::vector<SHRG*>& shrg_rules,
                                     std::vector<GrammarAttributes>& attrs_pool);

    /**
     * Create a pool of GrammarAttributes for use with cached forests.
     * Each GrammarAttributes points to its corresponding SHRG rule.
     * @param shrg_rules Vector of SHRG rule pointers
     * @return Vector of GrammarAttributes, one per rule
     */
    static std::vector<GrammarAttributes> create_attrs_pool(const std::vector<SHRG*>& shrg_rules);

    /**
     * Get cache directory path.
     */
    const std::string& cache_dir() const { return cache_dir_; }

    /**
     * Get number of cache hits.
     */
    size_t cache_hits() const { return cache_hits_; }

    /**
     * Get number of cache misses.
     */
    size_t cache_misses() const { return cache_misses_; }

private:
    std::string cache_dir_;
    std::string forests_dir_;
    uint32_t grammar_hash_;
    mutable size_t cache_hits_;
    mutable size_t cache_misses_;

    /**
     * Get the path for a graph's cached forest.
     */
    std::string get_forest_path(const std::string& graph_id) const;

    /**
     * Sanitize graph_id for use as filename.
     */
    static std::string sanitize_filename(const std::string& graph_id);

    /**
     * Collect all reachable ChartItems from a root.
     */
    static void collect_all_items(ChartItem* root,
                                  std::unordered_map<ChartItem*, int32_t>& item_to_index);

    /**
     * Serialize a ChartItem to a SerializedNode.
     */
    static void serialize_node(ChartItem* item,
                               const std::unordered_map<ChartItem*, int32_t>& item_to_index,
                               SerializedNode& node,
                               std::vector<int32_t>& children_out,
                               std::vector<SerializedParentSib>& parents_out);

    /**
     * Deserialize a SerializedNode to a ChartItem.
     */
    static void deserialize_node(const SerializedNode& node, ChartItem* item);

    /**
     * Restore pointer relationships after deserializing all nodes.
     */
    static void restore_relationships(
        const std::vector<SerializedNode>& nodes,
        const std::vector<std::vector<int32_t>>& all_children,
        const std::vector<std::vector<SerializedParentSib>>& all_parents,
        std::vector<ChartItem*>& items);
};

}  // namespace forest_cache
}  // namespace shrg

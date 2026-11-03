//
// Forest Caching Implementation
// Serializes derivation forests to disk to skip re-parsing across runs.
//

#include "forest_cache.hpp"

#include <fstream>
#include <iostream>
#include <cstring>
#include <queue>
#include <sys/stat.h>
#include <cerrno>
#include <cctype>

namespace shrg {
namespace forest_cache {

namespace {

// Create directory if it doesn't exist
bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

// Check if file exists
bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// Convert EdgeSet (std::bitset<256>) to 4 uint64_t values
void edgeset_to_data(const EdgeSet& es, uint64_t data[4]) {
    // EdgeSet is a std::bitset<256> = 32 bytes = 4 uint64_t
    // We need to extract bits manually since bitset doesn't have direct access
    for (int i = 0; i < 4; i++) {
        data[i] = 0;
        for (int j = 0; j < 64; j++) {
            if (es[i * 64 + j]) {
                data[i] |= (1ULL << j);
            }
        }
    }
}

// Convert 4 uint64_t values to EdgeSet
void data_to_edgeset(const uint64_t data[4], EdgeSet& es) {
    es.reset();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 64; j++) {
            if (data[i] & (1ULL << j)) {
                es.set(i * 64 + j);
            }
        }
    }
}

// Convert NodeMapping to bytes
void nodemapping_to_data(const NodeMapping& nm, uint8_t data[16]) {
    std::memcpy(data, nm.m1.data(), 16);
}

// Convert bytes to NodeMapping
void data_to_nodemapping(const uint8_t data[16], NodeMapping& nm) {
    std::memcpy(nm.m1.data(), data, 16);
}

}  // namespace

// ============================================================================
// ForestCache Implementation
// ============================================================================

ForestCache::ForestCache(const std::string& cache_dir)
    : cache_dir_(cache_dir),
      forests_dir_(cache_dir + "/forests"),
      grammar_hash_(0),
      cache_hits_(0),
      cache_misses_(0) {
    // Create cache directories
    create_directory(cache_dir_);
    create_directory(forests_dir_);
}

void ForestCache::set_grammar_hash(uint32_t hash) {
    grammar_hash_ = hash;
}

uint32_t ForestCache::compute_hash(const std::string& content) {
    // Simple FNV-1a hash
    uint32_t hash = 2166136261u;
    for (char c : content) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 16777619u;
    }
    return hash;
}

std::string ForestCache::sanitize_filename(const std::string& graph_id) {
    std::string result;
    result.reserve(graph_id.size());
    for (char c : graph_id) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
            result += c;
        } else {
            // Replace invalid chars with underscore
            result += '_';
        }
    }
    return result;
}

std::string ForestCache::get_forest_path(const std::string& graph_id) const {
    return forests_dir_ + "/" + sanitize_filename(graph_id) + ".bin";
}

bool ForestCache::has_valid_cache(const std::string& graph_id, uint32_t graph_hash) const {
    std::string path = get_forest_path(graph_id);
    if (!file_exists(path)) {
        return false;
    }

    // Read and validate header
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    ForestHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!file || file.gcount() != sizeof(header)) {
        return false;
    }

    // Validate magic, version, and hashes
    if (header.magic != CACHE_MAGIC ||
        header.version != CACHE_VERSION ||
        header.grammar_hash != grammar_hash_ ||
        header.graph_hash != graph_hash) {
        return false;
    }

    return true;
}

void ForestCache::collect_all_items(ChartItem* root,
                                     std::unordered_map<ChartItem*, int32_t>& item_to_index) {
    if (!root || item_to_index.count(root)) {
        return;
    }

    // BFS to collect all items
    std::queue<ChartItem*> queue;
    queue.push(root);

    while (!queue.empty()) {
        ChartItem* item = queue.front();
        queue.pop();

        if (item_to_index.count(item)) {
            continue;
        }

        int32_t index = static_cast<int32_t>(item_to_index.size());
        item_to_index[item] = index;

        // Helper lambda to enqueue item if not already indexed
        auto enqueue_if_new = [&](ChartItem* ptr) {
            if (ptr && !item_to_index.count(ptr)) {
                queue.push(ptr);
            }
        };

        // Traverse next_ptr chain - just queue them, don't index yet
        // They'll be indexed when popped from queue
        ChartItem* ptr = item->next_ptr;
        while (ptr && ptr != item) {
            enqueue_if_new(ptr);
            ptr = ptr->next_ptr;
        }

        // Add children
        for (ChartItem* child : item->children) {
            enqueue_if_new(child);
        }

        // Add items from parents_sib
        for (const auto& parent_sib : item->parents_sib) {
            ChartItem* parent = std::get<0>(parent_sib);
            enqueue_if_new(parent);
            for (ChartItem* sib : std::get<1>(parent_sib)) {
                enqueue_if_new(sib);
            }
        }

        // Add left_ptr and right_ptr
        enqueue_if_new(item->left_ptr);
        enqueue_if_new(item->right_ptr);
    }
}

void ForestCache::serialize_node(ChartItem* item,
                                  const std::unordered_map<ChartItem*, int32_t>& item_to_index,
                                  SerializedNode& node,
                                  std::vector<int32_t>& children_out,
                                  std::vector<SerializedParentSib>& parents_out) {
    // Serialize EdgeSet
    edgeset_to_data(item->edge_set, node.edge_set_data);

    // Serialize NodeMapping
    nodemapping_to_data(item->boundary_node_mapping, node.boundary_mapping);

    // Copy scalar fields
    node.score = item->score;
    node.level = item->level;
    node.shrg_index = item->shrg_index;

    // Convert pointers to indices
    auto get_index = [&](ChartItem* ptr) -> int32_t {
        if (!ptr) return -1;
        auto it = item_to_index.find(ptr);
        return (it != item_to_index.end()) ? it->second : -1;
    };

    node.next_index = get_index(item->next_ptr);
    node.left_index = get_index(item->left_ptr);
    node.right_index = get_index(item->right_ptr);

    // Serialize children
    children_out.clear();
    for (ChartItem* child : item->children) {
        children_out.push_back(get_index(child));
    }
    node.children_count = static_cast<uint32_t>(children_out.size());

    // Serialize parents_sib
    parents_out.clear();
    for (const auto& parent_sib : item->parents_sib) {
        SerializedParentSib ps;
        ps.parent_index = get_index(std::get<0>(parent_sib));
        for (ChartItem* sib : std::get<1>(parent_sib)) {
            ps.sibling_indices.push_back(get_index(sib));
        }
        parents_out.push_back(ps);
    }
    node.parents_count = static_cast<uint32_t>(parents_out.size());

    // Copy EM-related fields
    node.log_inside_prob = item->log_inside_prob;
    node.log_outside_prob = item->log_outside_prob;
    node.log_sent_rule_count = item->log_sent_rule_count;
    node.log_inside_count = item->log_inside_count;

    // Copy derivation scoring fields
    node.em_greedy_score = item->em_greedy_score;
    node.em_greedy_deriv = item->em_greedy_deriv;
    node.em_inside_score = item->em_inside_score;
    node.em_inside_deriv = item->em_inside_deriv;
    node.count_greedy_score = item->count_greedy_score;
    node.count_greedy_deriv = item->count_greedy_deriv;
    node.count_inside_score = item->count_inside_score;
    node.count_inside_deriv = item->count_inside_deriv;
}

void ForestCache::deserialize_node(const SerializedNode& node, ChartItem* item) {
    // Deserialize EdgeSet
    data_to_edgeset(node.edge_set_data, item->edge_set);

    // Deserialize NodeMapping
    data_to_nodemapping(node.boundary_mapping, item->boundary_node_mapping);

    // Copy scalar fields
    item->score = node.score;
    item->level = node.level;
    item->shrg_index = node.shrg_index;

    // Pointer fields will be restored later by restore_relationships

    // Reset status flags for fresh EM iteration
    item->status = ChartItem::kEmpty;
    item->inside_visited_status = ChartItem::kEmpty;
    item->outside_visited_status = ChartItem::kEmpty;
    item->count_visited_status = ChartItem::kEmpty;
    item->child_visited_status = ChartItem::kEmpty;
    item->update_status = ChartItem::kEmpty;
    item->rule_visited = ChartItem::kEmpty;

    // Copy EM-related fields
    item->log_inside_prob = node.log_inside_prob;
    item->log_outside_prob = node.log_outside_prob;
    item->log_sent_rule_count = node.log_sent_rule_count;
    item->log_inside_count = node.log_inside_count;

    // Copy derivation scoring fields
    item->em_greedy_score = node.em_greedy_score;
    item->em_greedy_deriv = node.em_greedy_deriv;
    item->em_inside_score = node.em_inside_score;
    item->em_inside_deriv = node.em_inside_deriv;
    item->count_greedy_score = node.count_greedy_score;
    item->count_greedy_deriv = node.count_greedy_deriv;
    item->count_inside_score = node.count_inside_score;
    item->count_inside_deriv = node.count_inside_deriv;

    // Rule pointer will be set by addRulePointer after loading
    item->rule_ptr = nullptr;
    item->attrs_ptr = nullptr;
}

void ForestCache::restore_relationships(
    const std::vector<SerializedNode>& nodes,
    const std::vector<std::vector<int32_t>>& all_children,
    const std::vector<std::vector<SerializedParentSib>>& all_parents,
    std::vector<ChartItem*>& items) {

    auto get_ptr = [&](int32_t idx) -> ChartItem* {
        if (idx < 0 || idx >= static_cast<int32_t>(items.size())) {
            return nullptr;
        }
        return items[idx];
    };

    for (size_t i = 0; i < nodes.size(); i++) {
        ChartItem* item = items[i];
        const SerializedNode& node = nodes[i];

        // Restore pointer fields
        item->next_ptr = get_ptr(node.next_index);
        item->left_ptr = get_ptr(node.left_index);
        item->right_ptr = get_ptr(node.right_index);

        // Handle self-loop for next_ptr
        if (node.next_index == static_cast<int32_t>(i)) {
            item->next_ptr = item;
        }

        // Restore children
        item->children.clear();
        item->children.reserve(all_children[i].size());
        for (int32_t child_idx : all_children[i]) {
            item->children.push_back(get_ptr(child_idx));
        }

        // Restore parents_sib
        item->parents_sib.clear();
        item->parents_sib.reserve(all_parents[i].size());
        for (const auto& ps : all_parents[i]) {
            ChartItem* parent = get_ptr(ps.parent_index);
            std::vector<ChartItem*> siblings;
            siblings.reserve(ps.sibling_indices.size());
            for (int32_t sib_idx : ps.sibling_indices) {
                siblings.push_back(get_ptr(sib_idx));
            }
            item->parents_sib.emplace_back(parent, std::move(siblings));
        }
    }
}

void ForestCache::save(const std::string& graph_id, uint32_t graph_hash,
                        ChartItem* root) {
    if (!root) {
        return;
    }

    // Collect all reachable items and assign indices
    std::unordered_map<ChartItem*, int32_t> item_to_index;
    collect_all_items(root, item_to_index);

    if (item_to_index.empty()) {
        return;
    }

    // Create index-ordered vector of items
    std::vector<ChartItem*> ordered_items(item_to_index.size());
    for (const auto& kv : item_to_index) {
        ordered_items[kv.second] = kv.first;
    }

    // Serialize all nodes
    std::vector<SerializedNode> nodes(ordered_items.size());
    std::vector<std::vector<int32_t>> all_children(ordered_items.size());
    std::vector<std::vector<SerializedParentSib>> all_parents(ordered_items.size());

    uint32_t total_children = 0;
    uint32_t total_parents = 0;

    for (size_t i = 0; i < ordered_items.size(); i++) {
        serialize_node(ordered_items[i], item_to_index, nodes[i],
                       all_children[i], all_parents[i]);
        total_children += static_cast<uint32_t>(all_children[i].size());
        total_parents += static_cast<uint32_t>(all_parents[i].size());
    }

    // Prepare header
    ForestHeader header;
    header.magic = CACHE_MAGIC;
    header.version = CACHE_VERSION;
    header.grammar_hash = grammar_hash_;
    header.graph_hash = graph_hash;
    header.root_index = item_to_index[root];
    header.node_count = static_cast<uint32_t>(nodes.size());
    header.total_children = total_children;
    header.total_parents = total_parents;

    // Write to file
    std::string path = get_forest_path(graph_id);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        std::cerr << "Warning: Cannot create cache file: " << path << "\n";
        return;
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write nodes
    file.write(reinterpret_cast<const char*>(nodes.data()),
               static_cast<std::streamsize>(nodes.size() * sizeof(SerializedNode)));

    // Write children (flat array with counts embedded in nodes)
    for (size_t i = 0; i < all_children.size(); i++) {
        if (!all_children[i].empty()) {
            file.write(reinterpret_cast<const char*>(all_children[i].data()),
                       static_cast<std::streamsize>(all_children[i].size() * sizeof(int32_t)));
        }
    }

    // Write parents_sib
    for (size_t i = 0; i < all_parents.size(); i++) {
        // Write count of parent_sib entries (already in node)
        for (const auto& ps : all_parents[i]) {
            file.write(reinterpret_cast<const char*>(&ps.parent_index), sizeof(int32_t));
            uint32_t sib_count = static_cast<uint32_t>(ps.sibling_indices.size());
            file.write(reinterpret_cast<const char*>(&sib_count), sizeof(uint32_t));
            if (sib_count > 0) {
                file.write(reinterpret_cast<const char*>(ps.sibling_indices.data()),
                           static_cast<std::streamsize>(sib_count * sizeof(int32_t)));
            }
        }
    }

    file.close();
}

ChartItem* ForestCache::load(const std::string& graph_id, uint32_t graph_hash,
                              utils::MemoryPool<ChartItem>& pool) {
    std::string path = get_forest_path(graph_id);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        cache_misses_++;
        return nullptr;
    }

    // Read header
    ForestHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!file || file.gcount() != sizeof(header)) {
        cache_misses_++;
        return nullptr;
    }

    // Validate header
    if (header.magic != CACHE_MAGIC ||
        header.version != CACHE_VERSION ||
        header.grammar_hash != grammar_hash_ ||
        header.graph_hash != graph_hash) {
        cache_misses_++;
        return nullptr;
    }

    if (header.node_count == 0) {
        cache_misses_++;
        return nullptr;
    }

    // Read nodes
    std::vector<SerializedNode> nodes(header.node_count);
    file.read(reinterpret_cast<char*>(nodes.data()),
              static_cast<std::streamsize>(header.node_count * sizeof(SerializedNode)));

    if (!file) {
        cache_misses_++;
        return nullptr;
    }

    // Read children
    std::vector<std::vector<int32_t>> all_children(header.node_count);
    for (size_t i = 0; i < header.node_count; i++) {
        uint32_t count = nodes[i].children_count;
        if (count > 0) {
            all_children[i].resize(count);
            file.read(reinterpret_cast<char*>(all_children[i].data()),
                      static_cast<std::streamsize>(count * sizeof(int32_t)));
        }
    }

    // Read parents_sib
    std::vector<std::vector<SerializedParentSib>> all_parents(header.node_count);
    for (size_t i = 0; i < header.node_count; i++) {
        uint32_t count = nodes[i].parents_count;
        all_parents[i].reserve(count);
        for (uint32_t j = 0; j < count; j++) {
            SerializedParentSib ps;
            file.read(reinterpret_cast<char*>(&ps.parent_index), sizeof(int32_t));
            uint32_t sib_count;
            file.read(reinterpret_cast<char*>(&sib_count), sizeof(uint32_t));
            if (sib_count > 0) {
                ps.sibling_indices.resize(sib_count);
                file.read(reinterpret_cast<char*>(ps.sibling_indices.data()),
                          static_cast<std::streamsize>(sib_count * sizeof(int32_t)));
            }
            all_parents[i].push_back(std::move(ps));
        }
    }

    if (!file) {
        cache_misses_++;
        return nullptr;
    }

    // Allocate all nodes in the memory pool
    std::vector<ChartItem*> items(header.node_count);
    for (size_t i = 0; i < header.node_count; i++) {
        items[i] = pool.Push();
        deserialize_node(nodes[i], items[i]);
    }

    // Restore all pointer relationships
    restore_relationships(nodes, all_children, all_parents, items);

    cache_hits_++;

    // Return root
    if (header.root_index >= 0 && header.root_index < static_cast<int32_t>(items.size())) {
        return items[header.root_index];
    }

    return nullptr;
}

void ForestCache::clear() {
    // Remove all files in forests directory
    // Note: This is a simple implementation; a more robust version
    // would iterate through the directory
    cache_hits_ = 0;
    cache_misses_ = 0;
}

void ForestCache::restore_rule_pointers(ChartItem* root, const std::vector<SHRG*>& shrg_rules) {
    if (!root || root->rule_visited == ChartItem::kVisited) {
        return;
    }

    ChartItem* ptr = root;
    do {
        // Use pre-stored shrg_index instead of accessing attrs_ptr
        int grammar_index = ptr->shrg_index;
        if (grammar_index >= 0 && grammar_index < static_cast<int>(shrg_rules.size())) {
            ptr->rule_ptr = shrg_rules[grammar_index];
        } else {
            ptr->rule_ptr = nullptr;
        }

        ptr->rule_visited = ChartItem::kVisited;

        // Recursively restore for children
        for (ChartItem* child : ptr->children) {
            restore_rule_pointers(child, shrg_rules);
        }

        ptr = ptr->next_ptr;
    } while (ptr && ptr != root);
}

std::vector<GrammarAttributes> ForestCache::create_attrs_pool(const std::vector<SHRG*>& shrg_rules) {
    std::vector<GrammarAttributes> pool(shrg_rules.size());
    for (size_t i = 0; i < shrg_rules.size(); i++) {
        pool[i].grammar_ptr = shrg_rules[i];
    }
    return pool;
}

void ForestCache::restore_all_pointers(ChartItem* root, const std::vector<SHRG*>& shrg_rules,
                                       std::vector<GrammarAttributes>& attrs_pool) {
    if (!root) {
        return;
    }

    // Use BFS to traverse ALL reachable nodes, including via left_ptr/right_ptr
    // This is crucial because FindChartItemByEdge navigates via left_ptr/right_ptr
    std::queue<ChartItem*> queue;
    queue.push(root);

    while (!queue.empty()) {
        ChartItem* item = queue.front();
        queue.pop();

        if (!item || item->rule_visited == ChartItem::kVisited) {
            continue;
        }

        // Process this item and all items in its next_ptr cycle
        ChartItem* ptr = item;
        do {
            if (ptr->rule_visited == ChartItem::kVisited) {
                ptr = ptr->next_ptr;
                continue;
            }

            int grammar_index = ptr->shrg_index;
            if (grammar_index >= 0 &&
                grammar_index < static_cast<int>(shrg_rules.size()) &&
                grammar_index < static_cast<int>(attrs_pool.size())) {
                ptr->rule_ptr = shrg_rules[grammar_index];
                ptr->attrs_ptr = &attrs_pool[grammar_index];
            } else {
                ptr->rule_ptr = nullptr;
                ptr->attrs_ptr = nullptr;
            }

            ptr->rule_visited = ChartItem::kVisited;

            // Queue all reachable nodes: children, left_ptr, right_ptr
            for (ChartItem* child : ptr->children) {
                if (child && child->rule_visited != ChartItem::kVisited) {
                    queue.push(child);
                }
            }
            if (ptr->left_ptr && ptr->left_ptr->rule_visited != ChartItem::kVisited) {
                queue.push(ptr->left_ptr);
            }
            if (ptr->right_ptr && ptr->right_ptr->rule_visited != ChartItem::kVisited) {
                queue.push(ptr->right_ptr);
            }

            ptr = ptr->next_ptr;
        } while (ptr && ptr != item);
    }
}

}  // namespace forest_cache
}  // namespace shrg

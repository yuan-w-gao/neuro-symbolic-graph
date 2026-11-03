#include <array>
#include <iterator>
#include <sstream>

#include "extra_chart_item.hpp"
#include "extra_context.hpp"
#include "extra_edge_set.hpp"
#include "extra_edsgraph.hpp"
#include "extra_shrg.hpp"
#include "extra_ambiguity_metrics.hpp"
#include "extra_em.hpp"
#include "multi_threads_runner.hpp"

using namespace pybind11::literals;

#define LAMBDA(expr) []() { return (expr); }

#define LAMBDA_EXPR(type, expr) [](type &self) { return (expr); }

#define LAMBDA_WITH_INT(type, expr) [](const type &self, int index) { return (expr); }

#define LAMBDA_GET_LABEL(type)                                                                     \
    [](const type &self) { return Manager::manager.label_set[self.label]; }

#define LAMBDA_GET_SIZE(type, property) [](const type &self) { return self.property.size(); }

#define LAMBDA_ITER(type, iter, policy)                                                            \
    [](type &self) { return make_iterator(std::begin(iter), std::end(iter), policy); }

namespace shrg {

struct DerivationWrapper {
    const Derivation &v;
};

template <typename T> std::string ObjectToString(const T &self) {
    std::ostringstream os;
    debug::Printer{nullptr, Manager::manager.label_set}(&self, os);
    return os.str();
}

template <> std::string ObjectToString(const Derivation &self) {
    std::ostringstream os;
    debug::Printer{nullptr, Manager::manager.label_set}.Ln(self, self[0], 0, os);
    return os.str();
}

} // namespace shrg

PYBIND11_MODULE(pyshrg, m) {
    using namespace shrg;
    using namespace pybind11;

    m.doc() = "C++ hrg parser"; // optional module docstring

    m.attr("MAX_SHRG_EDGE_COUNT") = MAX_SHRG_EDGE_COUNT;
    m.attr("MAX_GRAPH_EDGE_COUNT") = MAX_GRAPH_EDGE_COUNT;
    m.attr("MAX_GRAMMAR_BOUNDARY_NODE_COUNT") = MAX_GRAMMAR_BOUNDARY_NODE_COUNT;

    m.def("initialize", &Initialize);
    m.def("get_manager", LAMBDA(&Manager::manager), return_value_policy::reference);

    class_<EdgeSet>(m, "EdgeSet") //
        .def(init<int64_t>())
        .def("__contains__", LAMBDA_WITH_INT(EdgeSet, self.test(index)))
        .def("to_edge_indices", &EdgeSet_ToIndices)
        .def("to_node_indices",
             overload_cast<const EdgeSet &, const EdsGraph &>(&EdgeSet_ToNodeList));

    class_<ChartItem>(m, "ChartItem") //
        .def_readonly("next", &ChartItem::next_ptr)
        .def_readonly("edge_set", &ChartItem::edge_set)
        .def_property_readonly(
            "grammar",
            LAMBDA_EXPR(ChartItem, (self.attrs_ptr ? self.attrs_ptr->grammar_ptr : nullptr)),
            return_value_policy::reference)
        .def_property_readonly("grammar_index", &ChartItem_GrammarIndex)
        .def_property_readonly("size", LAMBDA_EXPR(ChartItem, (self.edge_set.count())))
        .def_readwrite("cfg_index", &ChartItem::status)
        .def_readwrite("score", &ChartItem::score)
        .def_property_readonly("node_mapping",
                               LAMBDA_EXPR(ChartItem, (self.boundary_node_mapping.m1)))
        .def("to_string", &ChartItem_ToString)
        .def("to_dot", &ChartItem_ToDot)
        .def("to_list", overload_cast<const ChartItem &, const EdsGraph &>(&ChartItem_ToList))
        .def("swap", &ChartItem::Swap)
        .def("all", LAMBDA_ITER(ChartItem, self, return_value_policy::reference))
        .def("__hash__", LAMBDA_EXPR(ChartItem, std::hash<Ref<ChartItem>>()(Ref<ChartItem>(self))));

    class_<ChartItemSet>(m, "ChartItemSet") //
        .def_property_readonly("size", &ChartItemSet::Size)
        .def("__len__", &ChartItemSet::Size)
        .def("__iter__",                      //
             LAMBDA_EXPR(ChartItemSet, self), //
             return_value_policy::reference)
        .def("__getitem__",                              //
             LAMBDA_WITH_INT(ChartItemSet, self[index]), //
             return_value_policy::reference);

    class_<SHRG> shrg_class(m, "SHRG");

    class_<SHRG::Node>(shrg_class, "Node") //
        .def_readonly("index", &SHRG::Node::index)
        .def_readonly("is_external", &SHRG::Node::is_external)
        .def_readonly("linked_edges", &SHRG::Node::linked_edges);

    class_<SHRG::Edge>(shrg_class, "Edge") //
        .def_readonly("label_index", &SHRG::Edge::label)
        .def_readonly("index", &SHRG::Edge::index)
        .def_readonly("is_terminal", &SHRG::Edge::is_terminal)
        .def_readonly("linked_nodes", &SHRG::Edge::linked_nodes)
        .def_property_readonly("label", LAMBDA_GET_LABEL(SHRG::Edge))
        .def("__str__", &ObjectToString<SHRG::Edge>)
        .def("__repr__", &ObjectToString<SHRG::Edge>);

    class_<SHRG::CFGItem>(shrg_class, "CFGItem") //
        .def_readonly("label_index", &SHRG::CFGItem::label)
        .def_readonly("edge", &SHRG::CFGItem::aligned_edge_ptr)
        .def_property_readonly("label", LAMBDA_GET_LABEL(SHRG::CFGItem));

    class_<SHRG::CFGRule>(shrg_class, "CFGRule") //
        .def_readonly("label_index", &SHRG::CFGRule::label)
        .def_readonly("shrg_index", &SHRG::CFGRule::shrg_index)
        .def("get",                                              //
             LAMBDA_WITH_INT(SHRG::CFGRule, &self.items[index]), //
             return_value_policy::reference)
        .def("iter_items", //
             LAMBDA_ITER(SHRG::CFGRule, self.items, return_value_policy::reference))
        .def_property_readonly("size", LAMBDA_GET_SIZE(SHRG::CFGRule, items))
        .def_property_readonly("label", LAMBDA_GET_LABEL(SHRG::CFGRule));

    shrg_class //
        .def_readonly("label_index", &SHRG::label)
        .def_readonly("label_hash", &SHRG::label_hash)
        .def_readonly("terminal_edges", &SHRG::terminal_edges)
        .def_readonly("nonterminal_edges", &SHRG::nonterminal_edges)
        .def_readonly("external_nodes", &SHRG::external_nodes)
        .def_readonly("num_occurences", &SHRG::num_occurences)
        .def_readwrite("log_rule_weight", &SHRG::log_rule_weight)
        .def_property_readonly("is_empty", LAMBDA_EXPR(SHRG, self.IsEmpty()))
        .def_property_readonly("size", LAMBDA_GET_SIZE(SHRG, cfg_rules))
        .def_property_readonly("edges", LAMBDA_EXPR(SHRG, self.fragment.edges))
        .def_property_readonly("nodes", LAMBDA_EXPR(SHRG, self.fragment.nodes))
        .def_property_readonly("label", LAMBDA_GET_LABEL(SHRG))
        .def("iter_cfgs", LAMBDA_ITER(SHRG, self.cfg_rules, return_value_policy::reference))
        .def("get", LAMBDA_WITH_INT(SHRG, &self.cfg_rules[index]), return_value_policy::reference)
        .def("cfg_at", &SHRG_CFGToString);

    class_<EdsGraph> edsgraph_class(m, "EdsGraph");
    class_<EdsGraph::Node>(edsgraph_class, "Node") //
        .def_readonly("index", &EdsGraph::Node::index)
        .def_readonly("id", &EdsGraph::Node::id)
        .def_readonly("linked_edges", &EdsGraph::Node::linked_edges)
        .def_readonly("label_index", &EdsGraph::Node::label)
        .def_readonly("lemma", &EdsGraph::Node::lemma)
        .def_readonly("sense", &EdsGraph::Node::sense)
        .def_readonly("pos_tag", &EdsGraph::Node::pos_tag)
        .def_readonly("carg", &EdsGraph::Node::carg)
        .def_readonly("properties", &EdsGraph::Node::properties)
        .def_property_readonly("label", LAMBDA_GET_LABEL(EdsGraph::Node))
        .def("__str__", &ObjectToString<EdsGraph::Node>)
        .def("__repr__", &ObjectToString<EdsGraph::Node>);

    class_<EdsGraph::Edge>(edsgraph_class, "Edge") //
        .def_readonly("index", &EdsGraph::Edge::index)
        .def_readonly("label_index", &EdsGraph::Edge::label)
        .def_readonly("linked_nodes", &EdsGraph::Edge::linked_nodes)
        .def_property_readonly("label", LAMBDA_GET_LABEL(EdsGraph::Edge))
        .def("to_tuple", &EdsGraphEdge_ToTuple, "use_index"_a = false)
        .def("__str__", &ObjectToString<EdsGraph::Edge>)
        .def("__repr__", &ObjectToString<EdsGraph::Edge>);

    edsgraph_class //
        .def(init<>())
        .def_readonly("sentence", &EdsGraph::sentence)
        .def_readonly("sentence_id", &EdsGraph::sentence_id)
        .def_readonly("lemma_sequence", &EdsGraph::lemma_sequence)
        .def_readonly("top_index", &EdsGraph::top_index)
        .def_readonly("edges", &EdsGraph::edges)
        .def_readonly("nodes", &EdsGraph::nodes)
        .def_property_readonly("num_nodes", LAMBDA_GET_SIZE(EdsGraph, nodes))
        .def_property_readonly("num_edges", LAMBDA_GET_SIZE(EdsGraph, edges))
        .def_property_readonly("edges_map", &EdsGraph_EdgesMap);

    class_<DerivationWrapper>(m, "Derivation") //
        .def_property_readonly("size", LAMBDA_GET_SIZE(DerivationWrapper, v))
        .def("__len__", LAMBDA_GET_SIZE(DerivationWrapper, v))
        .def("__iter__", LAMBDA_ITER(DerivationWrapper, self.v, return_value_policy::reference))
        .def("__getitem__", //
             LAMBDA_WITH_INT(DerivationWrapper, &self.v[index]),
             return_value_policy::reference_internal)
        .def("__repr__", LAMBDA_EXPR(DerivationWrapper, ObjectToString(self.v)));

    class_<DerivationNode>(m, "DerivationNode")
        .def_readonly("grammar", &DerivationNode::grammar_ptr)
        .def_readonly("cfg", &DerivationNode::cfg_ptr)
        .def_readonly("item", &DerivationNode::item_ptr)
        .def_readonly("children", &DerivationNode::children);

    enum_<ParserError>(m, "ParserError") //
        .value("kNone", ParserError::kNone)
        .value("kNoResult", ParserError::kNoResult)
        .value("kOutOfMemory", ParserError::kOutOfMemory)
        .value("kTooLarge", ParserError::kTooLarge)
        .value("kUnInitialized", ParserError::kUnInitialized)
        .value("KUnknown", ParserError::kUnknown);

    class_<Context>(m, "Context") //
        .def_readonly("manager", &Context::manager_ptr)
        .def_readonly("type", &Context::type)
        .def_readonly("sentence", &Context::sentence)
        .def_readonly("best_item", &Context::best_item_ptr)
        .def_property_readonly("derivation",
                               LAMBDA_EXPR(Context, DerivationWrapper{self.derivation}))
        .def_property_readonly("num_grammars_available",
                               LAMBDA_EXPR(Context, self.parser->GetNumGrammarsAvailable()))
        .def_property_readonly("num_terminal_subgraphs",
                               LAMBDA_EXPR(Context, self.parser->GetNumTerminalSubgraphs()))
        .def_property_readonly("num_active_items",
                               LAMBDA_EXPR(Context, self.parser->GetNumActiveItems()))
        .def_property_readonly("num_passive_items",
                               LAMBDA_EXPR(Context, self.parser->GetNumPassiveItems()))
        .def_property_readonly("num_succ_merge_ops",
                               LAMBDA_EXPR(Context, self.parser->GetNumSuccMergeOps()))
        .def_property_readonly("num_total_merge_ops",
                               LAMBDA_EXPR(Context, self.parser->GetNumTotalMergeOps()))
        .def_property_readonly("num_indexing_keys",
                               LAMBDA_EXPR(Context, self.parser->GetNumIndexingKeys()))
        .def_property_readonly("result_item", &Context::Result, return_value_policy::reference)
        .def("set_best_item", &Context::SetChartItem)
        .def("init", &Context::Init, "type"_a, "verbose"_a = false, "max_pool_size"_a = 25)
        .def("parse", overload_cast<const EdsGraph &>(&Context::Parse))
        .def("parse", overload_cast<int>(&Context::Parse))
        .def("generate", &Context::Generate)
        .def("split_item",
             overload_cast<const Context &, ChartItem &, const EdsGraph &>(&Context_SplitItem))
        .def("split_item", overload_cast<const Context &, ChartItem &>(&Context_SplitItem))
        .def("count_items", &Context::CountChartItems, "subgraph"_a = nullptr)
        .def("export_derivation", &Context_ExportDerivation)
        .def("find_best_derivation", &Context_FindBestDerivation)
        .def("release_memory", &Context::ReleaseMemory)
        .def("pool_size", &Context::PoolSize)
        .def("get_item", &Context::GetChartItem, return_value_policy::reference)
        // Ambiguity metrics
        .def("compute_inside_outside", &Context_ComputeInsideOutside,
             "Compute inside-outside probabilities, returns log partition function")
        .def("compute_entropy", &Context_ComputeEntropy,
             "Compute derivation entropy (requires inside-outside to be computed)")
        .def("compute_entropy_with_partition", &Context_ComputeEntropyWithPartition,
             "Compute entropy with explicit log partition", "log_partition"_a)
        .def("compute_derivation_count", &Context_ComputeDerivationCount,
             "Compute expected number of derivations")
        .def("compute_log_derivation_count", &Context_ComputeLogDerivationCount,
             "Compute log of derivation count (for overflow safety)")
        .def("compute_forest_stats", &Context_ComputeForestStats,
             "Compute forest statistics (nodes, edges, depth, branching)")
        .def("compute_all_metrics", &Context_ComputeAllMetrics,
             "Compute all ambiguity metrics at once")
        .def("get_log_partition", &Context_GetLogPartition,
             "Get log partition function (inside probability at root)");

    class_<Manager>(m, "Manager") //
        .def_property_readonly("hrg_size", LAMBDA_GET_SIZE(Manager, grammars))
        .def_property_readonly("shrg_size", LAMBDA_GET_SIZE(Manager, shrg_rules))
        .def_property_readonly("graph_size", LAMBDA_GET_SIZE(Manager, edsgraphs))
        .def_property_readonly("context_size", LAMBDA_GET_SIZE(Manager, contexts))
        .def("allocate", &Manager::Allocate, "num_contexts"_a = 4)
        .def("get_hrg",                                          //
             LAMBDA_WITH_INT(Manager, &self.grammars.at(index)), //
             return_value_policy::reference)
        .def("get_shrg",                                          //
             LAMBDA_WITH_INT(Manager, self.shrg_rules.at(index)), //
             return_value_policy::reference)
        .def("get_graph",                                         //
             LAMBDA_WITH_INT(Manager, &self.edsgraphs.at(index)), //
             return_value_policy::reference)
        .def("get_context",                                     //
             LAMBDA_WITH_INT(Manager, self.contexts.at(index)), //
             return_value_policy::reference)
        .def("iter_contexts", LAMBDA_ITER(Manager, self.contexts, return_value_policy::reference))
        .def("iter_graphs", LAMBDA_ITER(Manager, self.edsgraphs, return_value_policy::reference))
        .def("iter_hrgs", LAMBDA_ITER(Manager, self.grammars, return_value_policy::reference))
        .def("index_of_hrg", [](const Manager &self,
                                const SHRG &grammar) { return &grammar - self.grammars.data(); })
        .def("load_grammars", &Manager::LoadGrammars, "input_file"_a, "filter"_a = "none")
        .def("load_graphs", &Manager::LoadGraphs)
        .def("init_all", &Manager::InitAll, "type"_a, "verbose"_a = false, "max_pool_size"_a = 25)
        .def("freeze_tokens", LAMBDA_EXPR(Manager, self.label_set.Freeze()));

    class_<Runner>(m, "Runner") //
        .def(init<Manager &, bool>(), "manager"_a, "verbose"_a = false)
        .def("__call__", &Runner::Run);

    // EM Result struct
    class_<EMResult>(m, "EMResult")
        .def_readonly("final_weights", &EMResult::final_weights)
        .def_readonly("weight_history", &EMResult::weight_history)
        .def_readonly("log_likelihoods", &EMResult::log_likelihoods)
        .def_readonly("num_iterations", &EMResult::num_iterations)
        .def_readonly("converged", &EMResult::converged)
        .def_readonly("elapsed_time", &EMResult::elapsed_time);

    // EM training functions (basic - re-parses every iteration)
    m.def("run_em", &run_batch_em,
          "Run batch EM training (basic version - re-parses every iteration)",
          "manager"_a,
          "convergence_threshold"_a = 1e-4,
          "max_iterations"_a = 100,
          "max_graphs"_a = -1,
          "verbose"_a = true);

    m.def("train_em_simple", &train_em_simple,
          "Train EM and return final weights",
          "manager"_a,
          "convergence_threshold"_a = 1e-4,
          "max_iterations"_a = 100,
          "verbose"_a = true);

    // ==========================================================================
    // Optimized EM - forest caching + O(n log n) outside algorithm
    // ==========================================================================

    // GraphMetrics struct for profiling
    class_<em::GraphMetrics>(m, "GraphMetrics")
        .def_readonly("sentence_id", &em::GraphMetrics::sentence_id)
        .def_readonly("node_count", &em::GraphMetrics::node_count)
        .def_readonly("edge_count", &em::GraphMetrics::edge_count)
        .def_readonly("forest_size", &em::GraphMetrics::forest_size)
        .def_readonly("max_chain_length", &em::GraphMetrics::max_chain_length)
        .def_readonly("max_children", &em::GraphMetrics::max_children)
        .def_readonly("max_parents", &em::GraphMetrics::max_parents)
        .def_readonly("parse_time_ms", &em::GraphMetrics::parse_time_ms)
        .def_readonly("deep_copy_time_ms", &em::GraphMetrics::deep_copy_time_ms)
        .def_readonly("reset_flags_time_ms", &em::GraphMetrics::reset_flags_time_ms)
        .def_readonly("inside_time_ms", &em::GraphMetrics::inside_time_ms)
        .def_readonly("outside_time_ms", &em::GraphMetrics::outside_time_ms)
        .def_readonly("expected_count_time_ms", &em::GraphMetrics::expected_count_time_ms)
        .def_readonly("total_em_time_ms", &em::GraphMetrics::total_em_time_ms);

    // OptimizedEMResult struct
    class_<OptimizedEMResult>(m, "OptimizedEMResult")
        .def_readonly("final_weights", &OptimizedEMResult::final_weights)
        .def_readonly("weight_history", &OptimizedEMResult::weight_history)
        .def_readonly("log_likelihoods", &OptimizedEMResult::log_likelihoods)
        .def_readonly("iteration_times", &OptimizedEMResult::iteration_times)
        .def_readonly("num_iterations", &OptimizedEMResult::num_iterations)
        .def_readonly("converged", &OptimizedEMResult::converged)
        .def_readonly("total_time", &OptimizedEMResult::total_time)
        .def_readonly("num_forests_cached", &OptimizedEMResult::num_forests_cached)
        .def_readonly("graph_metrics", &OptimizedEMResult::graph_metrics);

    // Optimized EM function
    m.def("run_em_optimized", &run_em_optimized,
          "Run optimized EM training with forest caching and O(n log n) outside algorithm.\n\n"
          "This is significantly faster than run_em() because:\n"
          "1. Parses graphs once and caches derivation forests\n"
          "2. Uses optimized outside algorithm (O(n log n) vs O(chain^depth * n))\n"
          "3. Uses persistent memory pool for forest storage\n",
          "manager"_a,
          "convergence_threshold"_a = 1e-4,
          "max_iterations"_a = 100,
          "output_dir"_a = "N",
          "enable_profiling"_a = false,
          "use_safe_mode"_a = false,
          "timeout_seconds"_a = 5,
          "skip_graphs"_a = std::vector<std::string>{},
          "verbose"_a = true,
          "parser_verbose"_a = false);

    // OptimizedEMTrainer class for more control
    class_<OptimizedEMTrainer>(m, "OptimizedEMTrainer")
        .def(init<Manager&, double, const std::string&, int, const std::vector<std::string>&>(),
             "Create an optimized EM trainer.\n\n"
             "Args:\n"
             "    manager: The Manager containing rules and graphs\n"
             "    convergence_threshold: Stop when LL improvement < threshold\n"
             "    output_dir: Directory for output files (\"N\" for none)\n"
             "    timeout_seconds: Timeout for parsing each graph\n"
             "    skip_graphs: List of sentence IDs to skip\n",
             "manager"_a,
             "convergence_threshold"_a = 1e-4,
             "output_dir"_a = "N",
             "timeout_seconds"_a = 5,
             "skip_graphs"_a = std::vector<std::string>{})
        .def("enable_profiling", &OptimizedEMTrainer::enable_profiling,
             "Enable per-graph profiling metrics",
             "enable"_a = true)
        .def("run", &OptimizedEMTrainer::run,
             "Run full EM training with forest caching",
             "use_safe_mode"_a = false,
             "verbose"_a = true)
        .def("get_weights", &OptimizedEMTrainer::get_weights,
             "Get current rule log-weights")
        .def("set_weights", &OptimizedEMTrainer::set_weights,
             "Set rule log-weights",
             "weights"_a)
        .def("run_validation", &OptimizedEMTrainer::run_validation,
             "Run validation comparing original vs optimized outside algorithm")
        .def("print_metrics_summary", &OptimizedEMTrainer::print_metrics_summary,
             "Print profiling metrics summary")
        .def("get_graph_metrics", &OptimizedEMTrainer::get_graph_metrics,
             "Get per-graph profiling metrics");

    // Safe graph index testing for MLE/neural training
    class_<SafeGraphIndices>(m, "SafeGraphIndices")
        .def_readonly("safe_indices", &SafeGraphIndices::safe_indices)
        .def_readonly("safe_ids", &SafeGraphIndices::safe_ids)
        .def_readonly("num_tested", &SafeGraphIndices::num_tested)
        .def_readonly("num_safe", &SafeGraphIndices::num_safe)
        .def_readonly("num_skipped", &SafeGraphIndices::num_skipped)
        .def_readonly("num_oom", &SafeGraphIndices::num_oom)
        .def_readonly("total_time", &SafeGraphIndices::total_time);

    m.def("test_safe_graph_indices", &test_safe_graph_indices,
          "Test which graphs are safe to parse using fork-based OOM protection.\n\n"
          "Uses fork() to test each parse in a child process. If the child\n"
          "crashes (OOM/timeout), the parent survives and marks that graph unsafe.\n\n"
          "Returns indices of safe graphs. Caller then parses only those graphs.\n",
          "manager"_a,
          "timeout_seconds"_a = 10,
          "max_graphs"_a = -1,
          "verbose"_a = true);
}

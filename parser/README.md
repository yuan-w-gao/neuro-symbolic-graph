# SHRG parser (vendored)

Vendored, pruned copy of the C++ SHRG parser from the `parser` submodule of
`graph_optimization`, upstream commit `7c291ef12b2accd20101fc5d5ed10a2fc143a56e`
(2026-04-03). The parser and EM code are from previous work; the C++ sources are unchanged.

**Differences from upstream — build configuration only, no C++ source changes:**
- `CMakeLists.txt` defaults to `Release` instead of hard-setting `Debug`. Upstream's
  `SET(CMAKE_BUILD_TYPE Debug)` overrode `-DCMAKE_BUILD_TYPE=Release` from the build
  script, so every earlier build was Debug. With `NDEBUG`, `SetCompleteItem`
  (`parser_base.hpp:159`) breaks at the first complete item instead of asserting it is unique.
- Targets pruned to `shrg`, `forest_cache`, `em_legacy`, `ambiguity_metrics`, `pyshrg`,
  `comprehensive_eval`, `get_deriv`, `compute_entropy_dp`, plus optional `generate`,
  `compute_baseline`, `count_derivations`. Hard-coded macOS include path removed.
- Dropped sources: `run_*`, `em_debugger`, `em_verify`, `count_nodes`, `check_parser`,
  `evaluate`, `get_*_sentences`, `get_deriv_rule_indices`, `compute_ambiguity_metrics`,
  `test_forest_cache`, `test.cpp`, `em_framework/{em_hmm, variational_inference, em_debug,
  em_data_processor}`, `python/inside_outside.cpp.old`, and the duplicate
  `third-party/{pybind11-2.5.0, sparsehash-c11-2.11.1}`. The kept `third-party/pybind11` is v2.13.6 (a2e59f0), vendored without its `.git`.

Build instructions are in the top-level README.

import os

import pytest
import torch

from graph_grammar import PackedForest, expected_counts, export_forest, inside, outside

try:
    import pyshrg
    PYSHRG_AVAILABLE = True
except ImportError:
    PYSHRG_AVAILABLE = False


pytestmark = pytest.mark.skipif(not PYSHRG_AVAILABLE, reason="pyshrg module not available")


class TestExportForest:
    @pytest.fixture
    def forest(self):
        pyshrg.initialize()
        manager = pyshrg.get_manager()

        for path in ["data/test", "../data/test", "parser/data/test"]:
            grammar_file = os.path.join(path, "test.mapping.txt")
            graph_file = os.path.join(path, "test.graph.txt")
            if os.path.exists(grammar_file) and os.path.exists(graph_file):
                break
        else:
            pytest.skip("No test data found")

        manager.load_grammars(grammar_file)
        manager.load_graphs(graph_file)
        manager.allocate(num_contexts=1)
        manager.init_all(type="linear", verbose=False)

        context, graph = manager.get_context(0), manager.get_graph(0)
        error = context.parse(graph)
        if error != pyshrg.ParserError.kNone:
            pytest.skip(f"Parse failed with error: {error}")
        return export_forest(context, graph)

    def test_export_produces_valid_forest(self, forest):
        assert isinstance(forest, PackedForest)
        assert len(forest.nodes) > 0
        assert len(forest.hyperedges) > 0
        assert len(forest.roots) > 0

        for root_idx in forest.roots:
            assert forest.nodes[root_idx].is_root
        for edge in forest.hyperedges:
            assert 0 <= edge.parent_node < len(forest.nodes)
            for child in edge.child_nodes:
                assert 0 <= child < len(forest.nodes)

    def test_inside_outside_numerical_stability(self, forest):
        rule_log_probs = torch.zeros(forest.num_rules)

        inside_values, total_log_prob = inside(forest, rule_log_probs)
        outside_values = outside(forest, rule_log_probs, inside_values)

        assert not torch.isnan(inside_values).any()
        assert not torch.isinf(inside_values).any()
        # Outside may be -inf for unreachable nodes, so only NaN is checked.
        assert not torch.isnan(outside_values).any()
        assert not torch.isnan(total_log_prob)

    def test_expected_counts_non_negative(self, forest):
        counts = expected_counts(forest, torch.zeros(forest.num_rules))
        assert (counts >= 0).all()


class TestCreateSyntheticData:
    def test_create_synthetic_grammar_and_graph(self, tmp_path):
        grammar_file = tmp_path / "test.mapping.txt"
        graph_file = tmp_path / "test.graph.txt"

        grammar_file.write_text("# Synthetic grammar\n")
        graph_file.write_text("# Synthetic graph\n")

        assert grammar_file.exists()
        assert graph_file.exists()


class TestPyshrgBasicUsage:
    def test_initialize(self):
        pyshrg.initialize()
        assert pyshrg.get_manager() is not None

    def test_constants(self):
        assert hasattr(pyshrg, 'MAX_SHRG_EDGE_COUNT')
        assert hasattr(pyshrg, 'MAX_GRAPH_EDGE_COUNT')
        assert hasattr(pyshrg, 'MAX_GRAMMAR_BOUNDARY_NODE_COUNT')

    def test_parser_error_enum(self):
        assert hasattr(pyshrg, 'ParserError')
        assert hasattr(pyshrg.ParserError, 'kNone')
        assert hasattr(pyshrg.ParserError, 'kNoResult')

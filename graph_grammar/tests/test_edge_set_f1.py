import pickle

import pytest

from graph_grammar.evaluation.parsing_f1 import (
    compute_f1,
    convert_gold_edge_tuples_to_indices,
    extract_edge_sets_from_gold_derivation,
    find_gold_derivation_in_forest,
    load_gold_derivations_raw,
    load_gold_edge_indices_from_file,
    load_gold_rule_indices_from_file,
)
from graph_grammar.forest import ForestNode, HyperEdge, PackedForest


class TestConvertGoldEdgeTuples:
    def test_basic_conversion(self):
        edges_map = {('x10', 'named'): 0, ('_2', 'proper_q'): 1, ('_2', 'x10', 'BV'): 2}

        result = convert_gold_edge_tuples_to_indices({('x10', 'named'), ('_2', 'proper_q')}, edges_map)

        assert result == frozenset({0, 1})

    def test_missing_tuples(self):
        result = convert_gold_edge_tuples_to_indices({('x10', 'named'), ('missing', 'tuple')},
                                                     {('x10', 'named'): 0})
        assert result == frozenset({0})

    def test_empty_tuples(self):
        assert convert_gold_edge_tuples_to_indices(set(), {('x10', 'named'): 0}) == frozenset()

    def test_binary_edge_tuples(self):
        edges_map = {('x1', 'x2', 'ARG1'): 0, ('x1', 'x3', 'ARG2'): 1, ('x1', 'predicate'): 2}

        result = convert_gold_edge_tuples_to_indices({('x1', 'x2', 'ARG1'), ('x1', 'predicate')}, edges_map)

        assert result == frozenset({0, 2})


class TestEdgeSetF1:
    def test_perfect_match(self):
        sets = {frozenset({0, 1}), frozenset({2, 3})}
        assert compute_f1(sets, set(sets)) == (1.0, 1.0, 1.0)

    def test_partial_match(self):
        predicted = {frozenset({0, 1}), frozenset({2, 3})}
        gold = {frozenset({0, 1}), frozenset({4, 5})}

        precision, recall, f1 = compute_f1(predicted, gold)

        assert precision == 0.5
        assert recall == 0.5
        assert f1 == 0.5

    def test_no_match(self):
        _, _, f1 = compute_f1({frozenset({0, 1})}, {frozenset({2, 3})})
        assert f1 == 0.0

    def test_empty_predicted(self):
        _, _, f1 = compute_f1(set(), {frozenset({0, 1})})
        assert f1 == 0.0

    def test_empty_both(self):
        _, _, f1 = compute_f1(set(), set())
        assert f1 == 1.0


class TestLoadGoldDerivations:
    def test_load_raw_format(self, tmp_path):
        test_data = {
            'graph1': [
                (0, {('x1', 'pred1')}, None),
                (1, {('x2', 'pred2'), ('x1', 'x2', 'ARG')}, ('x1',)),
            ],
            'graph2': [(5, {('e1', 'event')}, None)],
        }
        path = tmp_path / "derivations.p"
        path.write_bytes(pickle.dumps(test_data))

        result = load_gold_derivations_raw(str(path))

        assert 'graph1' in result
        assert 'graph2' in result
        assert len(result['graph1']) == 2
        assert result['graph1'][0][0] == 0
        assert ('x1', 'pred1') in result['graph1'][0][1]


class TestEdgeSetF1Integration:
    def test_realistic_derivation_comparison(self):
        gold = {frozenset({0, 1}), frozenset({2, 3, 4}), frozenset({0, 1, 2, 3, 4})}
        predicted = {frozenset({0, 1}), frozenset({2, 3}), frozenset({0, 1, 2, 3, 4})}

        precision, recall, _ = compute_f1(predicted, gold)

        assert precision == pytest.approx(2/3)
        assert recall == pytest.approx(2/3)


class TestLoadGoldRuleIndicesFromFile:
    def load(self, tmp_path, content):
        path = tmp_path / "gold.txt"
        path.write_text(content)
        return load_gold_rule_indices_from_file(str(path))

    def test_colon_format(self, tmp_path):
        result = self.load(tmp_path, "0:1:42\n1:2:10,15\n2:3:5,10,20\n")
        assert result == {0: [42], 1: [10, 15], 2: [5, 10, 20]}

    def test_tab_format(self, tmp_path):
        result = self.load(tmp_path, "# loop_index\trule_indices\n0\t42\n1\t10,15\n2\t5,10,20\n")
        assert result == {0: [42], 1: [10, 15], 2: [5, 10, 20]}


class TestLoadGoldEdgeIndicesFromFile:
    def test_basic_loading(self, tmp_path):
        path = tmp_path / "gold_edges.txt"
        path.write_text("# loop_index\tnum_edge_sets\tedge_index_sets\n"
                        "0\t1\t0,1,2\n1\t2\t0,1;2,3,4\n2\t3\t0;1,2;3\n")

        result = load_gold_edge_indices_from_file(str(path))

        assert result[0] == [frozenset({0, 1, 2})]
        assert len(result[1]) == 2
        assert frozenset({0, 1}) in result[1]
        assert frozenset({2, 3, 4}) in result[1]
        assert len(result[2]) == 3


@pytest.fixture
def two_derivation_forest():
    """Root covers edges 0-4 via rule 10 -> (20, 30) or rule 11 -> (21, 31)."""
    def node(i, incoming, edge_indices, root=False):
        return ForestNode(id=i, edge_set_hash=i, boundary_mapping=(0,), incoming_edges=incoming,
                          is_terminal=not root, is_root=root, edge_indices=edge_indices)

    nodes = [
        node(0, [0, 3], (0, 1, 2, 3, 4), root=True),
        node(1, [1], (0, 1)),
        node(2, [2], (2, 3, 4)),
        node(3, [4], (0, 1, 2)),
        node(4, [5], (3, 4)),
    ]
    hyperedges = [
        HyperEdge(id=0, rule_id=10, parent_node=0, child_nodes=(1, 2), log_prob=0.0),
        HyperEdge(id=1, rule_id=20, parent_node=1, child_nodes=(), log_prob=0.0),
        HyperEdge(id=2, rule_id=30, parent_node=2, child_nodes=(), log_prob=0.0),
        HyperEdge(id=3, rule_id=11, parent_node=0, child_nodes=(3, 4), log_prob=0.0),
        HyperEdge(id=4, rule_id=21, parent_node=3, child_nodes=(), log_prob=0.0),
        HyperEdge(id=5, rule_id=31, parent_node=4, child_nodes=(), log_prob=0.0),
    ]
    return PackedForest(nodes=nodes, hyperedges=hyperedges, roots=[0], num_rules=50, graph_id="test")


class TestFindGoldDerivationInForest:
    def test_find_derivation_path1(self, two_derivation_forest):
        forest = two_derivation_forest

        derivation = find_gold_derivation_in_forest(forest, [10, 20, 30])

        assert derivation is not None
        assert {forest.hyperedges[e].rule_id for _, e in derivation} == {10, 20, 30}

    def test_find_derivation_path2(self, two_derivation_forest):
        forest = two_derivation_forest

        derivation = find_gold_derivation_in_forest(forest, [11, 21, 31])

        assert derivation is not None
        assert {forest.hyperedges[e].rule_id for _, e in derivation} == {11, 21, 31}

    def test_derivation_not_found(self, two_derivation_forest):
        assert find_gold_derivation_in_forest(two_derivation_forest, [99, 100, 101]) is None

    def test_extract_edge_sets(self, two_derivation_forest):
        edge_sets = extract_edge_sets_from_gold_derivation(two_derivation_forest, [10, 20, 30])

        assert edge_sets is not None
        assert frozenset({0, 1, 2, 3, 4}) in edge_sets
        assert frozenset({0, 1}) in edge_sets
        assert frozenset({2, 3, 4}) in edge_sets

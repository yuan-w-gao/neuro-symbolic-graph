import pytest

from graph_grammar.forest import ForestNode, HyperEdge, PackedForest


def create_single_rule_forest(rule_id, num_rules):
    node = ForestNode(id=0, edge_set_hash=hash((rule_id,)), boundary_mapping=(),
                      incoming_edges=[0], is_terminal=True, is_root=True)
    edge = HyperEdge(id=0, rule_id=rule_id, parent_node=0, child_nodes=())
    return PackedForest(nodes=[node], hyperedges=[edge], roots=[0],
                        num_rules=num_rules, graph_id=f"single_rule_{rule_id}")


def forests_with_counts(counts, num_rules):
    """counts[r] single-rule forests for each rule r."""
    return [create_single_rule_forest(rule, num_rules)
            for rule, n in enumerate(counts) for _ in range(n)]


def create_binary_choice_forest(num_rules=2):
    assert num_rules >= 2
    node = ForestNode(id=0, edge_set_hash=hash((0, 1)), boundary_mapping=(),
                      incoming_edges=[0, 1], is_terminal=True, is_root=True)
    edges = [
        HyperEdge(id=0, rule_id=0, parent_node=0, child_nodes=()),
        HyperEdge(id=1, rule_id=1, parent_node=0, child_nodes=()),
    ]
    return PackedForest(nodes=[node], hyperedges=edges, roots=[0],
                        num_rules=num_rules, graph_id="binary_choice")


def create_simple_forest():
    """S -> A B, A -> a, B -> b as rules 0, 2, 3 of 5; one derivation."""
    nodes = [
        ForestNode(id=0, edge_set_hash=hash((0, 1, 2)), boundary_mapping=(),
                   incoming_edges=[0], is_terminal=False, is_root=True),
        ForestNode(id=1, edge_set_hash=hash((0,)), boundary_mapping=(),
                   incoming_edges=[1], is_terminal=True, is_root=False),
        ForestNode(id=2, edge_set_hash=hash((1,)), boundary_mapping=(),
                   incoming_edges=[2], is_terminal=True, is_root=False),
    ]
    hyperedges = [
        HyperEdge(id=0, rule_id=0, parent_node=0, child_nodes=(1, 2)),
        HyperEdge(id=1, rule_id=2, parent_node=1, child_nodes=()),
        HyperEdge(id=2, rule_id=3, parent_node=2, child_nodes=()),
    ]
    return PackedForest(nodes=nodes, hyperedges=hyperedges, roots=[0],
                        num_rules=5, graph_id="test_ab")


def create_ambiguous_forest():
    """S -> A B | C D (rules 0, 1); each child has two leaf rules: 8 derivations."""
    nodes = [
        ForestNode(id=0, edge_set_hash=hash((0, 1)), boundary_mapping=(),
                   incoming_edges=[0, 1], is_terminal=False, is_root=True),
        ForestNode(id=1, edge_set_hash=hash((0,)), boundary_mapping=(),
                   incoming_edges=[2, 3], is_terminal=True, is_root=False),
        ForestNode(id=2, edge_set_hash=hash((1,)), boundary_mapping=(),
                   incoming_edges=[4, 5], is_terminal=True, is_root=False),
    ]
    hyperedges = [
        HyperEdge(id=0, rule_id=0, parent_node=0, child_nodes=(1, 2)),
        HyperEdge(id=1, rule_id=1, parent_node=0, child_nodes=(1, 2)),
        HyperEdge(id=2, rule_id=2, parent_node=1, child_nodes=()),
        HyperEdge(id=3, rule_id=4, parent_node=1, child_nodes=()),
        HyperEdge(id=4, rule_id=3, parent_node=2, child_nodes=()),
        HyperEdge(id=5, rule_id=5, parent_node=2, child_nodes=()),
    ]
    return PackedForest(nodes=nodes, hyperedges=hyperedges, roots=[0],
                        num_rules=6, graph_id="test_xy_ambiguous")


@pytest.fixture
def ab_forest():
    """S -> A B, A -> a, B -> b as rules 0, 1, 2 of 3."""
    nodes = [
        ForestNode(id=0, edge_set_hash=hash((0, 1)), boundary_mapping=(0, 1), is_root=True),
        ForestNode(id=1, edge_set_hash=hash((0,)), boundary_mapping=(0,)),
        ForestNode(id=2, edge_set_hash=hash((1,)), boundary_mapping=(1,)),
    ]
    hyperedges = [
        HyperEdge(id=0, rule_id=0, parent_node=0, child_nodes=(1, 2)),
        HyperEdge(id=1, rule_id=1, parent_node=1, child_nodes=()),
        HyperEdge(id=2, rule_id=2, parent_node=2, child_nodes=()),
    ]
    nodes[0].incoming_edges = [0]
    nodes[1].incoming_edges = [1]
    nodes[2].incoming_edges = [2]
    return PackedForest(nodes=nodes, hyperedges=hyperedges, roots=[0], num_rules=3)


@pytest.fixture
def two_choice_forest():
    """A single root derived by rule 0 or rule 1."""
    nodes = [ForestNode(id=0, edge_set_hash=hash((0,)), boundary_mapping=(0,), is_root=True)]
    hyperedges = [
        HyperEdge(id=0, rule_id=0, parent_node=0, child_nodes=()),
        HyperEdge(id=1, rule_id=1, parent_node=0, child_nodes=()),
    ]
    nodes[0].incoming_edges = [0, 1]
    return PackedForest(nodes=nodes, hyperedges=hyperedges, roots=[0], num_rules=2)

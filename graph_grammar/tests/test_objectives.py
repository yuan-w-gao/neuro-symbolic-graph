import pytest
import torch

from graph_grammar.forest import ForestNode, HyperEdge, PackedForest
from graph_grammar.learning.objectives import (
    analyze_forest,
    compare_objectives,
    compute_batch_entropy,
    compute_derivation_entropy,
    train_constrained_min_entropy,
    train_max_entropy,
    train_min_entropy,
    train_mle,
)


def leaf_forest(rule_ids, num_rules):
    """A single root node with one leaf edge per rule in rule_ids."""
    nodes = [ForestNode(id=0, edge_set_hash=hash((0,)), boundary_mapping=(0,), is_root=True)]
    hyperedges = [HyperEdge(id=i, rule_id=r, parent_node=0, child_nodes=()) for i, r in enumerate(rule_ids)]
    nodes[0].incoming_edges = list(range(len(rule_ids)))
    return PackedForest(nodes=nodes, hyperedges=hyperedges, roots=[0], num_rules=num_rules)


@pytest.fixture
def three_way_ambiguous():
    return leaf_forest([0, 1, 2], num_rules=3)


class TestEntropyComputation:
    def test_uniform_entropy(self, two_choice_forest):
        entropy = compute_derivation_entropy(two_choice_forest, torch.tensor([-0.693, -0.693]))
        assert abs(entropy.item() - 0.693) < 0.01

    def test_peaked_entropy(self, two_choice_forest):
        entropy = compute_derivation_entropy(two_choice_forest, torch.tensor([0.0, -10.0]))
        assert entropy.item() < 0.1

    def test_unambiguous_entropy(self):
        entropy = compute_derivation_entropy(leaf_forest([0], num_rules=2), torch.tensor([-0.5, -0.5]))
        assert abs(entropy.item()) < 0.01

    def test_three_way_uniform_entropy(self, three_way_ambiguous):
        uniform = torch.log(torch.tensor([1/3, 1/3, 1/3]))
        entropy = compute_derivation_entropy(three_way_ambiguous, uniform)
        assert abs(entropy.item() - 1.099) < 0.01

    def test_entropy_is_positive(self, two_choice_forest):
        for _ in range(10):
            random_probs = torch.log_softmax(torch.randn(2), dim=0)
            entropy = compute_derivation_entropy(two_choice_forest, random_probs)
            assert entropy.item() >= -0.001


class TestMLETraining:
    def test_converges_on_ambiguous(self, two_choice_forest):
        result = train_mle([two_choice_forest] * 20, num_rules=2, max_iterations=100, verbose=False)

        probs = torch.exp(result.rule_log_probs)
        assert abs(probs[0].item() - 0.5) < 0.05
        assert abs(probs[1].item() - 0.5) < 0.05

    def test_learns_data_distribution(self):
        forests = [leaf_forest([0], 2) for _ in range(7)] + [leaf_forest([1], 2) for _ in range(3)]

        result = train_mle(forests, num_rules=2, max_iterations=200, verbose=False)

        probs = torch.exp(result.rule_log_probs)
        assert abs(probs[0].item() - 0.7) < 0.05
        assert abs(probs[1].item() - 0.3) < 0.05


class TestMinEntropyTraining:
    def test_produces_peaked_distribution(self, two_choice_forest):
        result = train_min_entropy([two_choice_forest] * 20, num_rules=2, max_iterations=500, verbose=False)
        assert max(torch.exp(result.rule_log_probs)).item() > 0.9

    def test_achieves_low_entropy(self, three_way_ambiguous):
        forests = [three_way_ambiguous] * 30
        result = train_min_entropy(forests, num_rules=3, max_iterations=500, verbose=False)
        assert compute_batch_entropy(forests, result.rule_log_probs).item() < 0.1


class TestMaxEntropyTraining:
    def test_maintains_uniform_on_symmetric(self, two_choice_forest):
        result = train_max_entropy([two_choice_forest] * 20, num_rules=2, max_iterations=100, verbose=False)
        assert abs(torch.exp(result.rule_log_probs)[0].item() - 0.5) < 0.05

    def test_achieves_high_entropy(self, three_way_ambiguous):
        forests = [three_way_ambiguous] * 20
        result = train_max_entropy(forests, num_rules=3, max_iterations=100, verbose=False)
        assert compute_batch_entropy(forests, result.rule_log_probs).item() > 1.0


class TestConstrainedMinEntropy:
    def test_maintains_likelihood(self):
        forests = [leaf_forest([0], 2) for _ in range(10)]

        result = train_constrained_min_entropy(forests, num_rules=2, likelihood_weight=10.0,
                                               max_iterations=300, verbose=False)

        assert torch.exp(result.rule_log_probs)[0].item() > 0.5


class TestCompareObjectives:
    def test_returns_all_objectives(self, two_choice_forest):
        results = compare_objectives([two_choice_forest] * 10, num_rules=2, max_iterations=50, verbose=False)

        for name in ['mle', 'min_entropy', 'max_entropy', 'mdl']:
            assert name in results

    def test_all_produce_valid_distributions(self, two_choice_forest):
        results = compare_objectives([two_choice_forest] * 10, num_rules=2, max_iterations=50, verbose=False)

        for name, result in results.items():
            probs = torch.exp(result.rule_log_probs)
            assert (probs > 0).all(), name
            assert (probs <= 1).all(), name
            assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4), name


class TestAnalyzeForest:
    def test_analysis_keys(self, two_choice_forest):
        analysis = analyze_forest(two_choice_forest, torch.tensor([-0.693, -0.693]))

        assert analysis['valid']
        for key in ['log_prob', 'entropy', 'best_log_prob', 'best_rules']:
            assert key in analysis

    def test_empty_forest_analysis(self):
        empty = PackedForest(nodes=[], hyperedges=[], roots=[], num_rules=2)
        assert not analyze_forest(empty, torch.tensor([-0.693, -0.693]))['valid']

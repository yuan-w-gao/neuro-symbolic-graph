import math

import torch

from graph_grammar.inference import expected_counts, inside, outside, sample, viterbi
from graph_grammar.tests.conftest import create_ambiguous_forest, create_simple_forest

SIMPLE_PROBS = [0.6, 0.4, 1.0, 1.0, 1.0]


class TestInside:
    def test_simple_forest(self):
        forest = create_simple_forest()
        rule_log_probs = torch.log(torch.tensor(SIMPLE_PROBS))

        inside_values, total_log_prob = inside(forest, rule_log_probs)

        assert abs(total_log_prob.item() - math.log(0.6)) < 1e-5
        assert abs(inside_values[1].item() - math.log(1.0)) < 1e-5
        assert abs(inside_values[2].item() - math.log(1.0)) < 1e-5

    def test_ambiguous_forest(self):
        forest = create_ambiguous_forest()
        rule_log_probs = torch.log(torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0, 1.0]))

        _, total_log_prob = inside(forest, rule_log_probs)

        # Each child sums two leaf rules of prob 1, so each root edge is 0.5 * 2 * 2.
        assert abs(total_log_prob.item() - math.log(4.0)) < 1e-5

    def test_gradient_flows(self):
        forest = create_simple_forest()
        rule_log_probs = torch.log(torch.tensor(SIMPLE_PROBS))
        rule_log_probs.requires_grad_(True)

        _, total_log_prob = inside(forest, rule_log_probs)
        total_log_prob.backward()

        assert rule_log_probs.grad is not None
        assert abs(rule_log_probs.grad[0].item()) > 0
        assert abs(rule_log_probs.grad[1].item()) < 1e-10


class TestOutside:
    def test_root_outside(self):
        forest = create_simple_forest()
        rule_log_probs = torch.log(torch.tensor(SIMPLE_PROBS))

        inside_values, _ = inside(forest, rule_log_probs)
        outside_values = outside(forest, rule_log_probs, inside_values)

        assert abs(outside_values[0].item()) < 1e-5

    def test_inside_outside_product(self):
        forest = create_simple_forest()
        rule_log_probs = torch.log(torch.tensor(SIMPLE_PROBS))

        inside_values, total_log_prob = inside(forest, rule_log_probs)
        outside_values = outside(forest, rule_log_probs, inside_values)

        root_product = inside_values[0] + outside_values[0]
        assert abs(root_product.item() - total_log_prob.item()) < 1e-5


class TestExpectedCounts:
    def test_counts_sum_to_derivation_size(self):
        counts = expected_counts(create_simple_forest(), torch.log(torch.tensor(SIMPLE_PROBS)))
        assert abs(counts.sum().item() - 3.0) < 1e-5

    def test_counts_match_usage(self):
        counts = expected_counts(create_simple_forest(), torch.log(torch.tensor(SIMPLE_PROBS)))

        for rule, expected in enumerate([1.0, 0.0, 1.0, 1.0, 0.0]):
            assert abs(counts[rule].item() - expected) < 1e-5


class TestViterbi:
    def test_finds_best(self):
        forest = create_ambiguous_forest()
        rule_log_probs = torch.log(torch.tensor([0.8, 0.2, 1.0, 1.0, 1.0, 1.0]))

        _, derivation = viterbi(forest, rule_log_probs)

        edge_rules = [forest.hyperedges[e].rule_id for _, e in derivation]
        assert 0 in edge_rules

    def test_probability_matches(self):
        forest = create_simple_forest()
        rule_log_probs = torch.log(torch.tensor(SIMPLE_PROBS))

        best_log_prob, derivation = viterbi(forest, rule_log_probs)

        manual_log_prob = sum(rule_log_probs[forest.hyperedges[e].rule_id].item()
                              for _, e in derivation)
        assert abs(best_log_prob.item() - manual_log_prob) < 1e-5


class TestSample:
    def test_samples_valid_derivations(self):
        forest = create_simple_forest()
        rule_log_probs = torch.log(torch.tensor(SIMPLE_PROBS))

        samples = sample(forest, rule_log_probs, num_samples=10)

        assert len(samples) == 10
        for s in samples:
            assert isinstance(s, list)
            for node_idx, edge_idx in s:
                assert 0 <= node_idx < len(forest.nodes)
                assert 0 <= edge_idx < len(forest.hyperedges)

    def test_sample_distribution(self):
        forest = create_ambiguous_forest()
        rule_log_probs = torch.log(torch.tensor([0.8, 0.2, 1.0, 1.0, 1.0, 1.0]))

        torch.manual_seed(42)
        samples = sample(forest, rule_log_probs, num_samples=1000)

        rules = [forest.hyperedges[e].rule_id for s in samples for _, e in s]
        r0_ratio = rules.count(0) / (rules.count(0) + rules.count(1))
        assert 0.7 < r0_ratio < 0.9, f"R0 ratio {r0_ratio}"


class TestPackedForest:
    def test_topo_order(self):
        forest = create_simple_forest()
        position = {n: i for i, n in enumerate(forest.topo_order)}
        assert position[1] < position[0]
        assert position[2] < position[0]

    def test_to_tensors(self):
        forest = create_simple_forest()
        tensors = forest.to_tensors()

        for key in ['rule_ids', 'parent_nodes', 'child_nodes', 'topo_order', 'roots']:
            assert key in tensors
        assert len(tensors['rule_ids']) == len(forest.hyperedges)

    def test_num_derivations(self):
        assert create_simple_forest().num_derivations() == 1
        assert create_ambiguous_forest().num_derivations() == 8

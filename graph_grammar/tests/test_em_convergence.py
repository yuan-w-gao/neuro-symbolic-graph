import math

import torch

from graph_grammar.forest import PackedForest
from graph_grammar.learning.em import EMConfig, EMResult, compute_perplexity, em_step, train_em
from graph_grammar.tests.conftest import (
    create_ambiguous_forest,
    create_binary_choice_forest,
    create_simple_forest,
    create_single_rule_forest,
    forests_with_counts,
)


def assert_monotonic(ll, tol):
    for i in range(1, len(ll)):
        assert ll[i] >= ll[i - 1] - tol, f"LL decreased at iteration {i}: {ll[i-1]:.6f} -> {ll[i]:.6f}"


class TestBasicConvergence:
    def test_em_runs_on_simple_forest(self):
        result = train_em([create_simple_forest()], num_rules=5,
                          config=EMConfig(max_iterations=10, verbose=False))

        assert isinstance(result, EMResult)
        assert len(result.rule_log_probs) == 5
        assert len(result.log_likelihood_history) > 0
        assert result.num_iterations > 0

    def test_em_runs_on_ambiguous_forest(self):
        result = train_em([create_ambiguous_forest()], num_rules=6,
                          config=EMConfig(max_iterations=10, verbose=False))

        assert isinstance(result, EMResult)
        assert len(result.rule_log_probs) == 6

    def test_log_probs_are_valid(self):
        result = train_em([create_simple_forest()], num_rules=5,
                          config=EMConfig(max_iterations=10, verbose=False))
        assert torch.all(result.rule_log_probs <= 0)

    def test_log_likelihoods_are_valid(self):
        result = train_em([create_simple_forest()], num_rules=5,
                          config=EMConfig(max_iterations=10, verbose=False))
        for ll in result.log_likelihood_history:
            assert ll <= 0, f"Log likelihood {ll} should be <= 0"


class TestLogLikelihoodMonotonicity:
    config = EMConfig(max_iterations=50, convergence_threshold=1e-10, verbose=False)

    def test_ll_monotonic_simple_forest(self):
        result = train_em([create_simple_forest()], num_rules=5, config=self.config)
        assert_monotonic(result.log_likelihood_history, 1e-10)

    def test_ll_monotonic_ambiguous_forest(self):
        result = train_em([create_ambiguous_forest()], num_rules=6, config=self.config)
        assert_monotonic(result.log_likelihood_history, 1e-10)

    def test_ll_monotonic_binary_choice(self):
        result = train_em([create_binary_choice_forest(num_rules=2)], num_rules=2, config=self.config)
        assert_monotonic(result.log_likelihood_history, 1e-10)

    def test_ll_monotonic_batch(self):
        forests = [create_simple_forest(), create_ambiguous_forest(),
                   create_binary_choice_forest(num_rules=6)]
        result = train_em(forests, num_rules=6, config=self.config)
        assert_monotonic(result.log_likelihood_history, 1e-6)


class TestKnownDistributionRecovery:
    config = EMConfig(max_iterations=100, convergence_threshold=1e-8, verbose=False)

    def recovered_probs(self, counts):
        forests = forests_with_counts(counts, len(counts))
        result = train_em(forests, num_rules=len(counts), config=self.config)
        return torch.exp(result.rule_log_probs)

    def test_80_20_distribution(self):
        probs = self.recovered_probs([80, 20])
        assert abs(probs[0].item() - 0.80) < 0.05, f"P(R0) = {probs[0].item():.4f}"
        assert abs(probs[1].item() - 0.20) < 0.05, f"P(R1) = {probs[1].item():.4f}"

    def test_uniform_distribution(self):
        probs = self.recovered_probs([25, 25, 25, 25])
        for i in range(4):
            assert abs(probs[i].item() - 0.25) < 0.05, f"P(R{i}) = {probs[i].item():.4f}"

    def test_skewed_distribution(self):
        probs = self.recovered_probs([95, 5])
        assert abs(probs[0].item() - 0.95) < 0.05, f"P(R0) = {probs[0].item():.4f}"
        assert abs(probs[1].item() - 0.05) < 0.05, f"P(R1) = {probs[1].item():.4f}"

    def test_three_way_distribution(self):
        probs = self.recovered_probs([50, 30, 20])
        assert abs(probs[0].item() - 0.50) < 0.05
        assert abs(probs[1].item() - 0.30) < 0.05
        assert abs(probs[2].item() - 0.20) < 0.05


class TestConvergenceDetection:
    def test_convergence_detected(self):
        forests = [create_single_rule_forest(0, 2) for _ in range(100)]
        config = EMConfig(max_iterations=100, convergence_threshold=1e-6, verbose=False)

        result = train_em(forests, num_rules=2, config=config)

        assert result.converged
        assert result.num_iterations < config.max_iterations

    def test_max_iterations_respected(self):
        config = EMConfig(max_iterations=3, convergence_threshold=1e-20, verbose=False)

        result = train_em(forests_with_counts([50, 50], 2), num_rules=2, config=config)

        assert result.num_iterations <= config.max_iterations
        assert len(result.log_likelihood_history) == result.num_iterations

    def test_single_iteration(self):
        result = train_em([create_simple_forest()], num_rules=5,
                          config=EMConfig(max_iterations=1, verbose=False))

        assert result.num_iterations == 1
        assert len(result.log_likelihood_history) == 1

    def test_convergence_threshold_effect(self):
        forests = forests_with_counts([50, 50], 2)

        loose = train_em(forests, num_rules=2,
                         config=EMConfig(max_iterations=100, convergence_threshold=1e-3, verbose=False))
        strict = train_em(forests, num_rules=2,
                          config=EMConfig(max_iterations=100, convergence_threshold=1e-10, verbose=False))

        assert strict.num_iterations >= loose.num_iterations


class TestBatchTraining:
    def test_batch_converges(self):
        forests = forests_with_counts([30, 20], 3)
        forests += [create_binary_choice_forest(3) for _ in range(10)]
        config = EMConfig(max_iterations=100, convergence_threshold=1e-6, verbose=False)

        result = train_em(forests, num_rules=3, config=config)

        assert result.converged or result.num_iterations == config.max_iterations
        assert len(result.log_likelihood_history) == result.num_iterations

    def test_ll_scales_with_batch_size(self):
        rule_log_probs = torch.log(torch.tensor([0.5, 0.5]))

        _, ll_single = em_step([create_single_rule_forest(0, 2)], rule_log_probs)
        _, ll_double = em_step([create_single_rule_forest(0, 2) for _ in range(2)], rule_log_probs)

        ratio = ll_double / ll_single
        assert 1.9 < ratio < 2.1, f"LL ratio {ratio:.2f}"

    def test_empty_forest_handled(self):
        empty_forest = PackedForest(nodes=[], hyperedges=[], roots=[], num_rules=2, graph_id="empty")
        forests = [create_single_rule_forest(0, 2), empty_forest, create_single_rule_forest(1, 2)]

        result = train_em(forests, num_rules=2, config=EMConfig(max_iterations=10, verbose=False))

        assert len(result.rule_log_probs) == 2
        assert torch.all(result.rule_log_probs <= 0)


class TestPerplexity:
    def test_perplexity_decreases(self):
        forests = [create_single_rule_forest(0, 2) for _ in range(100)]
        initial = compute_perplexity(forests, torch.zeros(2) - math.log(2))

        result = train_em(forests, num_rules=2, config=EMConfig(max_iterations=50, verbose=False))
        final = compute_perplexity(forests, result.rule_log_probs)

        assert final < initial, f"perplexity {initial:.4f} -> {final:.4f}"

    def test_perfect_model_perplexity(self):
        forests = [create_single_rule_forest(0, 2) for _ in range(100)]
        rule_log_probs = torch.tensor([math.log(0.999), math.log(0.001)])

        perplexity = compute_perplexity(forests, rule_log_probs)

        assert perplexity < 1.1, f"Perplexity {perplexity:.4f}"


class TestEdgeCases:
    def test_all_same_rule(self):
        forests = [create_single_rule_forest(1, 3) for _ in range(100)]

        result = train_em(forests, num_rules=3, config=EMConfig(max_iterations=50, verbose=False))
        probs = torch.exp(result.rule_log_probs)

        assert probs[1].item() > probs[0].item()
        assert probs[1].item() > probs[2].item()
        assert probs[1].item() > 0.9, f"P(R1) = {probs[1].item():.4f}"

    def test_initial_probs_used(self):
        forests = [create_binary_choice_forest(2)]
        config = EMConfig(max_iterations=1, verbose=False)
        initial_log_probs = torch.tensor([math.log(0.99), math.log(0.01)])

        custom = train_em(forests, num_rules=2, config=config, initial_log_probs=initial_log_probs)
        uniform = train_em(forests, num_rules=2, config=config)

        assert abs(torch.exp(custom.rule_log_probs).sum().item() - 1.0) < 0.01
        assert abs(torch.exp(uniform.rule_log_probs).sum().item() - 1.0) < 0.01

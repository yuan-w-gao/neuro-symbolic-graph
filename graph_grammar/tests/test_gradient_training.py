import math

import pytest
import torch

from graph_grammar.forest import PackedForest
from graph_grammar.learning.em import EMConfig, train_em
from graph_grammar.learning.gradient import (
    GradientConfig,
    GradientResult,
    RuleLogitParameterization,
    compare_training_methods,
    compute_batch_log_likelihood,
    train_gradient,
)
from graph_grammar.tests.conftest import (
    create_ambiguous_forest,
    create_binary_choice_forest,
    create_simple_forest,
    create_single_rule_forest,
    forests_with_counts,
)


def train_simple(**config):
    return train_gradient([create_simple_forest()], num_rules=5,
                          config=GradientConfig(verbose=False, **config))


class TestBasicGradientConvergence:
    def test_gradient_runs_on_simple_forest(self):
        result = train_simple(max_iterations=10)

        assert isinstance(result, GradientResult)
        assert len(result.rule_log_probs) == 5
        assert len(result.log_likelihood_history) > 0
        assert result.num_iterations > 0

    def test_gradient_runs_on_ambiguous_forest(self):
        result = train_gradient([create_ambiguous_forest()], num_rules=6,
                                config=GradientConfig(max_iterations=10, verbose=False))

        assert isinstance(result, GradientResult)
        assert len(result.rule_log_probs) == 6

    def test_log_probs_are_valid(self):
        assert torch.all(train_simple(max_iterations=10).rule_log_probs <= 0)

    def test_probs_sum_to_one(self):
        probs = torch.exp(train_simple(max_iterations=10).rule_log_probs)
        assert abs(probs.sum().item() - 1.0) < 1e-5, f"Probs sum to {probs.sum().item()}"

    def test_gradient_norm_tracked(self):
        result = train_simple(max_iterations=10)

        assert result.gradient_norm_history is not None
        assert len(result.gradient_norm_history) == len(result.log_likelihood_history)


class TestGradientLogLikelihood:
    def test_ll_improves_simple_forest(self):
        ll = train_simple(max_iterations=50, learning_rate=0.5).log_likelihood_history
        assert ll[-1] >= ll[0] - 0.1, f"LL {ll[0]:.4f} -> {ll[-1]:.4f}"

    def test_ll_improves_ambiguous_forest(self):
        result = train_gradient([create_ambiguous_forest()], num_rules=6,
                                config=GradientConfig(max_iterations=50, learning_rate=0.5, verbose=False))
        ll = result.log_likelihood_history
        assert ll[-1] >= ll[0] - 0.1, f"LL {ll[0]:.4f} -> {ll[-1]:.4f}"

    def test_ll_improves_batch(self):
        config = GradientConfig(max_iterations=100, learning_rate=0.5, verbose=False)
        result = train_gradient(forests_with_counts([80, 20], 2), num_rules=2, config=config)
        ll = result.log_likelihood_history
        assert ll[-1] >= ll[0], f"LL {ll[0]:.4f} -> {ll[-1]:.4f}"


class TestKnownDistributionRecoveryGradient:
    config = GradientConfig(max_iterations=200, learning_rate=0.5, verbose=False)

    def recovered_probs(self, counts):
        forests = forests_with_counts(counts, len(counts))
        result = train_gradient(forests, num_rules=len(counts), config=self.config)
        return torch.exp(result.rule_log_probs)

    def test_80_20_distribution(self):
        probs = self.recovered_probs([80, 20])
        # Looser than EM: the patience stop can end training early.
        assert abs(probs[0].item() - 0.80) < 0.10, f"P(R0) = {probs[0].item():.4f}"
        assert abs(probs[1].item() - 0.20) < 0.10, f"P(R1) = {probs[1].item():.4f}"

    def test_uniform_distribution(self):
        probs = self.recovered_probs([25, 25, 25, 25])
        for i in range(4):
            assert abs(probs[i].item() - 0.25) < 0.05, f"P(R{i}) = {probs[i].item():.4f}"

    def test_skewed_distribution(self):
        probs = self.recovered_probs([95, 5])
        assert abs(probs[0].item() - 0.95) < 0.05, f"P(R0) = {probs[0].item():.4f}"
        assert abs(probs[1].item() - 0.05) < 0.05, f"P(R1) = {probs[1].item():.4f}"


class TestEMGradientComparison:
    def train_both(self, forests):
        em = train_em(forests, num_rules=2, config=EMConfig(max_iterations=100, verbose=False))
        grad = train_gradient(forests, num_rules=2,
                              config=GradientConfig(max_iterations=200, learning_rate=0.5, verbose=False))
        return em, grad

    def test_both_converge_to_same_distribution(self):
        em, grad = self.train_both(forests_with_counts([80, 20], 2))

        em_probs = torch.exp(em.rule_log_probs)
        grad_probs = torch.exp(grad.rule_log_probs)
        for i in range(2):
            diff = abs(em_probs[i].item() - grad_probs[i].item())
            assert diff < 0.10, f"R{i}: EM={em_probs[i].item():.4f}, Grad={grad_probs[i].item():.4f}"

    def test_similar_final_ll(self):
        em, grad = self.train_both(forests_with_counts([50, 50], 2))

        ll_diff = abs(em.log_likelihood_history[-1] - grad.log_likelihood_history[-1])
        assert ll_diff < 1.0, f"LL difference {ll_diff:.4f}"

    def test_compare_utility_function(self):
        result = compare_training_methods(
            forests=forests_with_counts([60, 40], 2),
            num_rules=2,
            em_config=EMConfig(max_iterations=50, verbose=False),
            gradient_config=GradientConfig(max_iterations=100, learning_rate=0.5, verbose=False),
        )

        assert len(result.em_log_probs) == 2
        assert len(result.gradient_log_probs) == 2
        assert result.em_final_ll < 0
        assert result.gradient_final_ll < 0
        assert result.log_prob_difference < 0.5, f"Log prob difference {result.log_prob_difference:.4f}"


class TestRuleLogitParameterization:
    def test_initialization_uniform(self):
        probs = torch.exp(RuleLogitParameterization(num_rules=4).get_rule_log_probs())
        for i in range(4):
            assert abs(probs[i].item() - 0.25) < 1e-5, f"P(R{i}) = {probs[i].item():.4f}"

    def test_initialization_from_log_probs(self):
        initial_log_probs = torch.tensor([math.log(0.5), math.log(0.3), math.log(0.2)])
        param = RuleLogitParameterization(num_rules=3, initial_log_probs=initial_log_probs)

        probs = torch.exp(param.get_rule_log_probs())

        assert probs[0] > probs[1] > probs[2]

    def test_grouped_normalization(self):
        param = RuleLogitParameterization(num_rules=4, rule_to_label=torch.tensor([0, 0, 1, 1]))

        probs = torch.exp(param.get_rule_log_probs())

        group0_sum = probs[0].item() + probs[1].item()
        group1_sum = probs[2].item() + probs[3].item()
        assert abs(group0_sum - 1.0) < 1e-5, f"Group 0 sums to {group0_sum:.4f}"
        assert abs(group1_sum - 1.0) < 1e-5, f"Group 1 sums to {group1_sum:.4f}"

    def test_gradient_flow(self):
        param = RuleLogitParameterization(num_rules=3)

        loss = -param.get_rule_log_probs()[0]
        loss.backward()

        assert param.logits.grad is not None
        assert not torch.all(param.logits.grad == 0)

    def test_forward_equals_get_rule_log_probs(self):
        param = RuleLogitParameterization(num_rules=4)
        assert torch.allclose(param.get_rule_log_probs(), param.forward())


class TestDifferentOptimizers:
    def p_r0(self, **config):
        forests = [create_single_rule_forest(0, 2) for _ in range(100)]
        result = train_gradient(forests, num_rules=2, config=GradientConfig(verbose=False, **config))
        return torch.exp(result.rule_log_probs)[0].item()

    def test_adam_optimizer(self):
        p = self.p_r0(max_iterations=50, optimizer="adam", learning_rate=0.5)
        assert p > 0.9, f"P(R0) = {p:.4f}"

    def test_sgd_optimizer(self):
        p = self.p_r0(max_iterations=100, optimizer="sgd", learning_rate=1.0)
        assert p > 0.8, f"P(R0) = {p:.4f}"

    def test_lbfgs_optimizer(self):
        p = self.p_r0(max_iterations=50, optimizer="lbfgs", learning_rate=1.0)
        assert p > 0.9, f"P(R0) = {p:.4f}"

    def test_invalid_optimizer_raises(self):
        config = GradientConfig(optimizer="invalid_optimizer", verbose=False)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            train_gradient([create_single_rule_forest(0, 2)], num_rules=2, config=config)


class TestConvergenceDetection:
    def test_convergence_detected(self):
        forests = [create_single_rule_forest(0, 2) for _ in range(100)]
        config = GradientConfig(max_iterations=500, convergence_threshold=1e-6,
                                learning_rate=0.5, verbose=False)

        result = train_gradient(forests, num_rules=2, config=config)

        assert result.converged or result.num_iterations < config.max_iterations

    def test_max_iterations_respected(self):
        config = GradientConfig(max_iterations=5, convergence_threshold=1e-20, verbose=False)

        result = train_gradient(forests_with_counts([50, 50], 2), num_rules=2, config=config)

        assert result.num_iterations <= config.max_iterations
        assert len(result.log_likelihood_history) == result.num_iterations


class TestBatchLogLikelihood:
    def test_single_forest_ll(self):
        rule_log_probs = torch.tensor([math.log(0.8), math.log(0.2)])

        ll = compute_batch_log_likelihood([create_single_rule_forest(0, 2)], rule_log_probs)

        assert abs(ll.item() - math.log(0.8)) < 1e-5, f"LL = {ll.item():.4f}"

    def test_batch_ll_is_sum(self):
        forests = [create_single_rule_forest(0, 2), create_single_rule_forest(1, 2)]
        rule_log_probs = torch.tensor([math.log(0.6), math.log(0.4)])

        batch_ll = compute_batch_log_likelihood(forests, rule_log_probs)

        expected = math.log(0.6) + math.log(0.4)
        assert abs(batch_ll.item() - expected) < 1e-5, f"Batch LL = {batch_ll.item():.4f}, expected {expected:.4f}"

    def test_ll_is_differentiable(self):
        rule_log_probs = torch.tensor([math.log(0.5), math.log(0.5)], requires_grad=True)

        ll = compute_batch_log_likelihood([create_single_rule_forest(0, 2)], rule_log_probs)
        ll.backward()

        assert rule_log_probs.grad is not None


class TestEdgeCases:
    def test_empty_forest_handled(self):
        empty_forest = PackedForest(nodes=[], hyperedges=[], roots=[], num_rules=2, graph_id="empty")
        forests = [create_single_rule_forest(0, 2), empty_forest, create_single_rule_forest(1, 2)]

        result = train_gradient(forests, num_rules=2, config=GradientConfig(max_iterations=10, verbose=False))

        assert len(result.rule_log_probs) == 2
        assert torch.all(result.rule_log_probs <= 0)

    def test_initial_probs_used(self):
        initial_log_probs = torch.tensor([math.log(0.99), math.log(0.01)])

        result = train_gradient([create_binary_choice_forest(2)], num_rules=2,
                                config=GradientConfig(max_iterations=1, verbose=False),
                                initial_log_probs=initial_log_probs)

        probs = torch.exp(result.rule_log_probs)
        assert probs[0].item() > probs[1].item()


class TestLabelBasedNormalization:
    def test_label_normalization_preserves_groups(self):
        config = GradientConfig(max_iterations=100, learning_rate=0.5, verbose=False)

        result = train_gradient(forests_with_counts([40, 10, 30, 20], 4), num_rules=4,
                                config=config, rule_to_label=torch.tensor([0, 0, 1, 1]))

        probs = torch.exp(result.rule_log_probs)
        group0_sum = probs[0].item() + probs[1].item()
        group1_sum = probs[2].item() + probs[3].item()
        assert abs(group0_sum - 1.0) < 0.01, f"Group 0 sums to {group0_sum:.4f}"
        assert abs(group1_sum - 1.0) < 0.01, f"Group 1 sums to {group1_sum:.4f}"

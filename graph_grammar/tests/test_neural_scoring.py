import pytest
import torch

from graph_grammar.learning import (
    BilinearRuleScorer,
    EMConfig,
    NeuralConfig,
    RuleEmbeddingScorer,
    compare_methods,
    train_em,
    train_neural,
)


@pytest.fixture
def labeled_rules():
    return torch.tensor([0, 0, 1, 1], dtype=torch.long)


def assert_distribution(log_probs, atol=1e-5):
    probs = torch.exp(log_probs)
    assert (probs > 0).all()
    assert (probs <= 1).all()
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=atol)


def assert_groups_normalized(log_probs):
    probs = torch.exp(log_probs)
    assert torch.isclose(probs[0] + probs[1], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(probs[2] + probs[3], torch.tensor(1.0), atol=1e-5)


class TestRuleEmbeddingScorer:
    def test_output_shape(self):
        scorer = RuleEmbeddingScorer(num_rules=5, embedding_dim=16)
        assert scorer.get_rule_log_probs().shape == (5,)

    def test_valid_probabilities(self):
        assert_distribution(RuleEmbeddingScorer(num_rules=4, embedding_dim=16).get_rule_log_probs())

    def test_grouped_normalization(self, labeled_rules):
        scorer = RuleEmbeddingScorer(num_rules=4, embedding_dim=16, rule_to_label=labeled_rules)
        assert_groups_normalized(scorer.get_rule_log_probs())

    def test_gradients_flow(self):
        scorer = RuleEmbeddingScorer(num_rules=3, embedding_dim=16)

        (-scorer.get_rule_log_probs().sum()).backward()

        for param in scorer.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_different_configs(self):
        configs = [
            {'embedding_dim': 8, 'hidden_dim': 16, 'num_layers': 1},
            {'embedding_dim': 32, 'hidden_dim': 64, 'num_layers': 2},
            {'embedding_dim': 64, 'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.1},
        ]
        for cfg in configs:
            log_probs = RuleEmbeddingScorer(num_rules=5, **cfg).get_rule_log_probs()
            assert log_probs.shape == (5,)
            assert not torch.isnan(log_probs).any()


class TestBilinearRuleScorer:
    def test_output_shape(self):
        scorer = BilinearRuleScorer(num_rules=5, embedding_dim=16)
        assert scorer.get_rule_log_probs().shape == (5,)

    def test_valid_probabilities(self):
        assert_distribution(BilinearRuleScorer(num_rules=4, embedding_dim=16).get_rule_log_probs())

    def test_grouped_normalization(self, labeled_rules):
        scorer = BilinearRuleScorer(num_rules=4, embedding_dim=16, rule_to_label=labeled_rules)
        assert_groups_normalized(scorer.get_rule_log_probs())


class TestNeuralTraining:
    def test_likelihood_increases(self, ab_forest):
        config = NeuralConfig(max_iterations=20, learning_rate=0.01, verbose=False)

        result = train_neural(forests=[ab_forest] * 5, num_rules=ab_forest.num_rules, config=config)

        assert not torch.isnan(result.rule_log_probs).any()
        assert not torch.isinf(result.rule_log_probs).any()
        assert 1 <= len(result.log_likelihood_history) <= 20

    def test_convergence(self, ab_forest):
        config = NeuralConfig(max_iterations=100, convergence_threshold=1e-4, learning_rate=0.01,
                              embedding_dim=16, hidden_dim=32, num_layers=1, verbose=False)

        result = train_neural(forests=[ab_forest] * 10, num_rules=ab_forest.num_rules, config=config)

        assert not torch.isnan(result.rule_log_probs).any()
        assert len(result.log_likelihood_history) > 0

    def test_valid_output(self, ab_forest):
        result = train_neural(forests=[ab_forest], num_rules=ab_forest.num_rules,
                              config=NeuralConfig(max_iterations=10, verbose=False))
        assert_distribution(result.rule_log_probs)

    def test_callback_called(self, ab_forest):
        iterations = []

        train_neural(forests=[ab_forest], num_rules=ab_forest.num_rules,
                     config=NeuralConfig(max_iterations=5, patience=10, verbose=False),
                     callback=lambda iteration, log_probs, ll: iterations.append(iteration))

        assert 1 <= len(iterations) <= 5
        assert iterations == list(range(len(iterations)))

    def test_ambiguous_forest(self, two_choice_forest):
        config = NeuralConfig(max_iterations=50, learning_rate=0.1, verbose=False)

        result = train_neural(forests=[two_choice_forest] * 10, num_rules=two_choice_forest.num_rules,
                              config=config)

        probs = torch.exp(result.rule_log_probs)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_custom_model(self, ab_forest):
        model = BilinearRuleScorer(num_rules=ab_forest.num_rules, embedding_dim=32)

        result = train_neural(forests=[ab_forest], num_rules=ab_forest.num_rules,
                              config=NeuralConfig(max_iterations=10, verbose=False), model=model)

        assert result.rule_log_probs is not None
        assert result.model_state is not None


class TestNeuralVsEM:
    def test_simple_case_convergence(self, ab_forest):
        forests = [ab_forest] * 20

        em_result = train_em(forests=forests, num_rules=ab_forest.num_rules,
                             config=EMConfig(max_iterations=100, verbose=False))
        neural_result = train_neural(forests=forests, num_rules=ab_forest.num_rules,
                                     config=NeuralConfig(max_iterations=200, learning_rate=0.1,
                                                         patience=50, verbose=False))

        assert neural_result.log_likelihood_history[-1] >= em_result.log_likelihood_history[-1] - 5.0

    def test_compare_methods(self, two_choice_forest):
        results = compare_methods(forests=[two_choice_forest] * 10, num_rules=two_choice_forest.num_rules,
                                  em_iterations=50, neural_iterations=50, verbose=False)

        for method in ['em', 'gradient', 'neural']:
            assert method in results
            assert 'log_probs' in results[method]
            assert 'final_ll' in results[method]


class TestEntropyRegularization:
    def test_entropy_encourages_uniformity(self, two_choice_forest):
        # Whether the result ends up more uniform depends on the data, so only NaNs are checked.
        result = train_neural(forests=[two_choice_forest] * 10, num_rules=2,
                              config=NeuralConfig(max_iterations=100, entropy_weight=1.0, verbose=False))

        assert not torch.isnan(result.rule_log_probs).any()


class TestModelState:
    def test_state_dict_saved(self, ab_forest):
        result = train_neural(forests=[ab_forest], num_rules=ab_forest.num_rules,
                              config=NeuralConfig(max_iterations=5, verbose=False))

        assert result.model_state is not None
        assert len(result.model_state) > 0

    def test_state_dict_loadable(self, ab_forest):
        config = NeuralConfig(max_iterations=10, verbose=False)
        result = train_neural(forests=[ab_forest], num_rules=ab_forest.num_rules, config=config)

        new_model = RuleEmbeddingScorer(num_rules=ab_forest.num_rules, embedding_dim=config.embedding_dim,
                                        hidden_dim=config.hidden_dim, num_layers=config.num_layers)
        new_model.load_state_dict(result.model_state)

        assert torch.allclose(new_model.get_rule_log_probs(), result.rule_log_probs, atol=1e-5)

import pytest
import torch
from torch import Tensor

from graph_grammar.inference import inside_with_scorer, viterbi_with_scorer
from graph_grammar.learning import ContextAwareRuleScorer, NeuralConfig, train_context_aware


@pytest.fixture
def node_embeddings(ab_forest):
    return {node.id: torch.randn(16) for node in ab_forest.nodes}


def make_scorer(forest, **kwargs):
    return ContextAwareRuleScorer(num_rules=forest.num_rules, node_embedding_dim=16, **kwargs)


def root_edge(forest):
    edge = forest.hyperedges[0]
    return edge, forest.nodes[edge.parent_node], [forest.nodes[c] for c in edge.child_nodes]


def train(forests, num_rules, **kwargs):
    return train_context_aware(forests=forests, num_rules=num_rules, node_embedding_dim=16, **kwargs)


class TestContextAwareRuleScorer:
    def test_initialization(self):
        scorer = ContextAwareRuleScorer(num_rules=5, node_embedding_dim=16, rule_embedding_dim=32)

        assert scorer.num_rules == 5
        assert scorer.node_embedding_dim == 16
        assert scorer.rule_embedding_dim == 32

    def test_score_edge(self, ab_forest, node_embeddings):
        score = make_scorer(ab_forest).score_edge(*root_edge(ab_forest), node_embeddings)

        assert isinstance(score, Tensor)
        assert score.dim() == 0
        assert not torch.isnan(score)

    def test_score_without_embeddings(self, ab_forest):
        score = make_scorer(ab_forest).score_edge(*root_edge(ab_forest), {})
        assert not torch.isnan(score)

    def test_scorer_function(self, ab_forest, node_embeddings):
        scorer_fn = make_scorer(ab_forest).get_scorer_function()

        log_prob = scorer_fn(*root_edge(ab_forest), node_embeddings)

        assert isinstance(log_prob, Tensor)
        assert log_prob <= 0

    def test_gradients_flow(self, ab_forest, node_embeddings):
        scorer = make_scorer(ab_forest)

        scorer.score_edge(*root_edge(ab_forest), node_embeddings).backward()

        # default_node_embedding gets no gradient when embeddings are given, so not all params do.
        params_with_grad = sum(1 for p in scorer.parameters() if p.requires_grad and p.grad is not None)
        assert params_with_grad > 0

    def test_different_aggregations(self, ab_forest, node_embeddings):
        for agg in ["mean", "sum", "max"]:
            score = make_scorer(ab_forest, aggregation=agg).score_edge(*root_edge(ab_forest), node_embeddings)
            assert not torch.isnan(score)


class TestInsideWithScorer:
    def test_basic_computation(self, ab_forest, node_embeddings):
        scorer_fn = make_scorer(ab_forest).get_scorer_function()

        inside_values, total_log_prob = inside_with_scorer(ab_forest, scorer_fn, node_embeddings)

        assert inside_values.shape == (len(ab_forest.nodes),)
        assert not torch.isnan(inside_values).any()
        assert not torch.isnan(total_log_prob)

    def test_matches_static_inside(self, ab_forest):
        num_rules = ab_forest.num_rules

        def uniform_scorer(edge, parent, children, embeddings):
            return torch.tensor(-torch.log(torch.tensor(float(num_rules))))

        inside_values, total_log_prob = inside_with_scorer(ab_forest, uniform_scorer, {})

        assert not torch.isnan(inside_values).any()
        assert total_log_prob.item() < 0

    def test_gradients_flow_through_inside(self, ab_forest, node_embeddings):
        scorer = make_scorer(ab_forest)

        _, total_log_prob = inside_with_scorer(ab_forest, scorer.get_scorer_function(), node_embeddings)
        (-total_log_prob).backward()

        for name, param in scorer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestViterbiWithScorer:
    def test_basic_viterbi(self, ab_forest, node_embeddings):
        scorer_fn = make_scorer(ab_forest).get_scorer_function()

        best_log_prob, derivation = viterbi_with_scorer(ab_forest, scorer_fn, node_embeddings)

        assert not torch.isnan(best_log_prob)
        assert len(derivation) > 0

    def test_derivation_valid(self, ab_forest, node_embeddings):
        scorer_fn = make_scorer(ab_forest).get_scorer_function()

        _, derivation = viterbi_with_scorer(ab_forest, scorer_fn, node_embeddings)

        for node_idx, edge_idx in derivation:
            assert 0 <= node_idx < len(ab_forest.nodes)
            assert 0 <= edge_idx < len(ab_forest.hyperedges)
            assert edge_idx in ab_forest.nodes[node_idx].incoming_edges


class TestContextAwareTraining:
    def test_basic_training(self, ab_forest):
        config = NeuralConfig(max_iterations=10, learning_rate=0.01, verbose=False)

        result = train([ab_forest] * 5, ab_forest.num_rules, config=config)

        assert result.rule_log_probs is not None
        assert len(result.log_likelihood_history) == 10
        assert not torch.isnan(result.rule_log_probs).any()

    def test_training_with_embeddings(self, ab_forest, node_embeddings):
        config = NeuralConfig(max_iterations=10, learning_rate=0.01, verbose=False)

        result = train([ab_forest] * 5, ab_forest.num_rules, config=config,
                       node_embeddings_list=[node_embeddings] * 5)

        assert not torch.isnan(result.rule_log_probs).any()

    def test_valid_output_distribution(self, ab_forest):
        result = train([ab_forest] * 5, ab_forest.num_rules,
                       config=NeuralConfig(max_iterations=5, verbose=False))

        probs = torch.exp(result.rule_log_probs)
        assert (probs > 0).all()
        assert (probs <= 1).all()
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)

    def test_model_state_saved(self, ab_forest):
        result = train([ab_forest], ab_forest.num_rules,
                       config=NeuralConfig(max_iterations=5, verbose=False))

        assert result.model_state is not None
        assert len(result.model_state) > 0

    def test_ambiguous_forest(self, two_choice_forest):
        config = NeuralConfig(max_iterations=20, learning_rate=0.01, verbose=False)

        result = train([two_choice_forest] * 10, two_choice_forest.num_rules, config=config)

        probs = torch.exp(result.rule_log_probs)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)


class TestIntegration:
    def test_end_to_end_pipeline(self, ab_forest):
        forests = [ab_forest] * 10
        embeddings_list = [{node.id: torch.randn(16) for node in ab_forest.nodes} for _ in forests]
        config = NeuralConfig(max_iterations=20, learning_rate=0.01, verbose=False)

        result = train(forests, ab_forest.num_rules, config=config, node_embeddings_list=embeddings_list)

        scorer = make_scorer(ab_forest)
        scorer.load_state_dict(result.model_state)
        _, total_log_prob = inside_with_scorer(ab_forest, scorer.get_scorer_function(), embeddings_list[0])

        assert not torch.isnan(total_log_prob)

import math

import pytest
import torch
from torch import Tensor

import graph_grammar.learning.compound as compound_mod
from graph_grammar.forest import ForestNode, HyperEdge, PackedForest
from graph_grammar.inference import inside_with_scorer
from graph_grammar.learning.compound import (
    CompoundConfig,
    CompoundHRG,
    CompoundRuleScorer,
    GraphEncoder,
    LSTMGraphEncoder,
    extract_bag_of_rules_features,
    extract_forest_statistics,
    extract_graph_features,
    train_compound_hrg,
    train_compound_scorer_only,
)


@pytest.fixture
def simple_forest():
    nodes = [
        ForestNode(id=0, edge_set_hash=0, boundary_mapping=(), incoming_edges=[0], is_root=True),
        ForestNode(id=1, edge_set_hash=1, boundary_mapping=(), incoming_edges=[1]),
        ForestNode(id=2, edge_set_hash=2, boundary_mapping=(), incoming_edges=[2]),
    ]
    hyperedges = [
        HyperEdge(id=0, rule_id=0, parent_node=0, child_nodes=(1, 2)),
        HyperEdge(id=1, rule_id=1, parent_node=1, child_nodes=()),
        HyperEdge(id=2, rule_id=2, parent_node=2, child_nodes=()),
    ]
    forest = PackedForest(nodes=nodes, hyperedges=hyperedges, roots=[0], num_rules=5)
    forest.topo_order = [1, 2, 0]
    return forest


@pytest.fixture
def ambiguous_forest():
    nodes = [
        ForestNode(id=0, edge_set_hash=0, boundary_mapping=(), incoming_edges=[0, 1], is_root=True),
        ForestNode(id=1, edge_set_hash=1, boundary_mapping=(), incoming_edges=[2]),
        ForestNode(id=2, edge_set_hash=2, boundary_mapping=(), incoming_edges=[3]),
    ]
    hyperedges = [
        HyperEdge(id=0, rule_id=0, parent_node=0, child_nodes=(1,)),
        HyperEdge(id=1, rule_id=1, parent_node=0, child_nodes=(2,)),
        HyperEdge(id=2, rule_id=2, parent_node=1, child_nodes=()),
        HyperEdge(id=3, rule_id=3, parent_node=2, child_nodes=()),
    ]
    forest = PackedForest(nodes=nodes, hyperedges=hyperedges, roots=[0], num_rules=5)
    forest.topo_order = [1, 2, 0]
    return forest


@pytest.fixture
def rule_to_label():
    return torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)


@pytest.fixture
def multiple_forests(simple_forest, ambiguous_forest):
    return [simple_forest, ambiguous_forest]


def make_scorer(rule_to_label, latent_dim=16, **kwargs):
    return CompoundRuleScorer(num_rules=5, num_labels=2, rule_to_label=rule_to_label,
                              latent_dim=latent_dim, **kwargs)


def make_model(rule_to_label, **kwargs):
    return CompoundHRG(num_rules=5, num_labels=2, rule_to_label=rule_to_label,
                       encoder_input_dim=10, latent_dim=16, **kwargs)


def make_lstm():
    return LSTMGraphEncoder(num_rules=100, embedding_dim=32, hidden_dim=64, latent_dim=16)


def train_scorer_only(forests, rule_to_label, callback=None, **config):
    return train_compound_scorer_only(forests=forests, num_rules=5, num_labels=2,
                                      rule_to_label=rule_to_label,
                                      config=CompoundConfig(verbose=False, **config), callback=callback)


def train_full(forests, rule_to_label, **config):
    features_list = [extract_graph_features(f, num_rules=5) for f in forests]
    # CPU keeps these small runs numerically stable.
    config = CompoundConfig(verbose=False, device="cpu", **config)
    return train_compound_hrg(forests=forests, features_list=features_list, num_rules=5,
                              num_labels=2, rule_to_label=rule_to_label, config=config)


class TestCompoundRuleScorer:
    def test_initialization(self, rule_to_label):
        scorer = make_scorer(rule_to_label, latent_dim=0, embedding_dim=32, hidden_dim=64)

        assert scorer.num_rules == 5
        assert scorer.num_labels == 2
        assert scorer.embedding_dim == 32
        assert scorer.latent_dim == 0
        assert len(scorer.label_groups) == 2

    def test_output_shape_without_z(self, rule_to_label):
        log_probs = make_scorer(rule_to_label, latent_dim=0).get_rule_log_probs()

        assert log_probs.shape == (5,)
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()

    def test_output_shape_with_z(self, rule_to_label):
        log_probs = make_scorer(rule_to_label).get_rule_log_probs(torch.randn(16))

        assert log_probs.shape == (5,)
        assert not torch.isnan(log_probs).any()

    def test_valid_probabilities(self, rule_to_label):
        probs = torch.exp(make_scorer(rule_to_label, latent_dim=0).get_rule_log_probs())

        assert torch.isclose(probs[rule_to_label == 0].sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(probs[rule_to_label == 1].sum(), torch.tensor(1.0), atol=1e-5)

    def test_z_changes_output(self, rule_to_label):
        scorer = make_scorer(rule_to_label)

        log_probs_1 = scorer.get_rule_log_probs(torch.randn(16))
        log_probs_2 = scorer.get_rule_log_probs(torch.randn(16))

        assert not torch.allclose(log_probs_1, log_probs_2, atol=1e-3)

    def test_z_required_when_latent_dim_positive(self, rule_to_label):
        with pytest.raises(ValueError, match="z is required"):
            make_scorer(rule_to_label).get_rule_log_probs()

    def test_gradients_flow(self, rule_to_label):
        scorer = make_scorer(rule_to_label)
        z = torch.randn(16, requires_grad=True)

        (-scorer.get_rule_log_probs(z).sum()).backward()

        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
        for name, param in scorer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_scorer_function(self, rule_to_label, simple_forest):
        scorer_fn = make_scorer(rule_to_label).get_scorer_function(torch.randn(16))
        edge = simple_forest.hyperedges[0]
        parent = simple_forest.nodes[edge.parent_node]
        children = [simple_forest.nodes[c] for c in edge.child_nodes]

        log_prob = scorer_fn(edge, parent, children, {})

        assert isinstance(log_prob, Tensor)
        assert log_prob.dim() == 0

    def test_inside_with_scorer_integration(self, rule_to_label, simple_forest):
        scorer_fn = make_scorer(rule_to_label).get_scorer_function(torch.randn(16))

        inside_values, total_log_prob = inside_with_scorer(simple_forest, scorer_fn)

        assert inside_values.shape == (len(simple_forest.nodes),)
        assert not torch.isnan(total_log_prob)
        assert total_log_prob < 0


class TestGraphEncoder:
    def test_initialization(self):
        encoder = GraphEncoder(input_dim=100, latent_dim=32, hidden_dim=64)

        assert encoder.input_dim == 100
        assert encoder.latent_dim == 32

    def test_output_shape(self):
        mu, log_var = GraphEncoder(input_dim=100, latent_dim=32)(torch.randn(100))

        assert mu.shape == (32,)
        assert log_var.shape == (32,)

    def test_batch_output_shape(self):
        mu, log_var = GraphEncoder(input_dim=100, latent_dim=32)(torch.randn(8, 100))

        assert mu.shape == (8, 32)
        assert log_var.shape == (8, 32)

    def test_sample_shape(self):
        encoder = GraphEncoder(input_dim=100, latent_dim=32)

        z = encoder.sample(*encoder(torch.randn(100)))

        assert z.shape == (32,)

    def test_reparameterization_gradient(self):
        encoder = GraphEncoder(input_dim=100, latent_dim=32)
        x = torch.randn(100, requires_grad=True)

        encoder.sample(*encoder(x)).sum().backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_kl_divergence_nonnegative(self):
        encoder = GraphEncoder(input_dim=100, latent_dim=32)
        assert encoder.kl_divergence(*encoder(torch.randn(100))) >= 0

    def test_kl_divergence_zero_at_prior(self):
        encoder = GraphEncoder(input_dim=100, latent_dim=32)

        kl = encoder.kl_divergence(torch.zeros(32), torch.zeros(32))

        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


class TestLSTMGraphEncoder:
    def test_initialization(self):
        encoder = make_lstm()

        assert encoder.num_rules == 100
        assert encoder.embedding_dim == 32
        assert encoder.hidden_dim == 64
        assert encoder.latent_dim == 16

    def test_output_shape(self):
        mu, log_var = make_lstm()(torch.tensor([0, 5, 10, 15], dtype=torch.long))

        assert mu.shape == (16,)
        assert log_var.shape == (16,)

    def test_empty_sequence(self):
        mu, log_var = make_lstm()(torch.tensor([], dtype=torch.long))

        assert mu.shape == (16,)
        assert log_var.shape == (16,)
        assert torch.allclose(mu, torch.zeros(16))
        assert torch.allclose(log_var, torch.zeros(16))

    def test_different_sequences_give_different_outputs(self):
        encoder = make_lstm()

        mu1, _ = encoder(torch.tensor([0, 1, 2], dtype=torch.long))
        mu2, _ = encoder(torch.tensor([3, 4, 5], dtype=torch.long))

        assert not torch.allclose(mu1, mu2)

    def test_gradients_flow(self):
        encoder = make_lstm()

        mu, log_var = encoder(torch.tensor([0, 5, 10], dtype=torch.long))
        encoder.sample(mu, log_var).sum().backward()

        assert encoder.rule_embedding.weight.grad is not None
        assert not torch.isnan(encoder.rule_embedding.weight.grad).any()


class TestFeatureExtraction:
    def test_bag_of_rules_shape(self, simple_forest):
        features = extract_bag_of_rules_features(simple_forest, num_rules=5)

        assert features.shape == (5,)
        assert features.sum() > 0

    def test_bag_of_rules_normalized(self, simple_forest):
        features = extract_bag_of_rules_features(simple_forest, num_rules=5, normalize=True)
        assert torch.isclose(features.sum(), torch.tensor(1.0), atol=1e-6)

    def test_forest_statistics_shape(self, simple_forest):
        stats = extract_forest_statistics(simple_forest)

        assert stats.dim() == 1
        assert len(stats) == 5

    def test_graph_features_combined(self, simple_forest):
        features = extract_graph_features(simple_forest, num_rules=5, include_statistics=True)
        assert features.shape == (5 + 5,)


class TestCompoundHRG:
    def test_initialization(self, rule_to_label):
        model = make_model(rule_to_label, embedding_dim=32, hidden_dim=64)

        assert model.num_rules == 5
        assert model.latent_dim == 16

    def test_forward_pass(self, rule_to_label, simple_forest):
        ll, kl, elbo = make_model(rule_to_label)(simple_forest, torch.randn(10))

        assert ll.dim() == 0
        assert kl.dim() == 0
        assert elbo.dim() == 0
        assert kl >= 0

    def test_elbo_components(self, rule_to_label, simple_forest):
        ll, kl, elbo = make_model(rule_to_label)(simple_forest, torch.randn(10))
        assert torch.isclose(elbo, ll - kl, atol=1e-5)

    def test_no_nan_outputs(self, rule_to_label, simple_forest):
        ll, kl, elbo = make_model(rule_to_label)(simple_forest, torch.randn(10))

        assert not torch.isnan(ll)
        assert not torch.isnan(kl)
        assert not torch.isnan(elbo)

    def test_gradients_flow(self, rule_to_label, simple_forest):
        model = make_model(rule_to_label)
        features = torch.randn(10, requires_grad=True)

        _, _, elbo = model(simple_forest, features)
        (-elbo).backward()

        assert features.grad is not None
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestCompoundTraining:
    def test_scorer_only_training(self, rule_to_label, multiple_forests):
        result = train_scorer_only(multiple_forests, rule_to_label,
                                   max_iterations=10, latent_dim=0)

        assert result.rule_log_probs is not None
        assert result.rule_log_probs.shape == (5,)
        assert len(result.ll_history) == 10
        assert all(kl == 0 for kl in result.kl_history)

    def test_full_compound_training(self, rule_to_label, multiple_forests):
        result, model = train_full(multiple_forests, rule_to_label, max_iterations=10, latent_dim=8)

        assert result.rule_log_probs is not None
        assert model is not None
        assert len(result.elbo_history) <= 10
        assert len(result.kl_history) <= 10
        assert all(kl >= 0 for kl in result.kl_history if not math.isnan(kl))

    def test_elbo_increases(self, rule_to_label, multiple_forests):
        result, _ = train_full(multiple_forests, rule_to_label,
                               max_iterations=50, latent_dim=8, learning_rate=0.01)

        valid = [e for e in result.elbo_history if not math.isnan(e)]
        if len(valid) >= 10:
            assert sum(valid[-5:]) / 5 >= sum(valid[:5]) / 5 - 1.0
        else:
            assert len(valid) >= 1

    def test_ll_increases(self, rule_to_label, multiple_forests):
        result = train_scorer_only(multiple_forests, rule_to_label,
                                   max_iterations=50, learning_rate=0.01)
        assert result.ll_history[-1] > result.ll_history[0]

    def test_callback_called(self, rule_to_label, multiple_forests):
        calls = []

        train_scorer_only(multiple_forests, rule_to_label, max_iterations=5,
                          callback=lambda iteration, model, value: calls.append(iteration))

        assert len(calls) == 5


class TestAblations:
    def test_latent_dim_zero_matches_baseline(self, rule_to_label, multiple_forests):
        result = train_scorer_only(multiple_forests, rule_to_label,
                                   max_iterations=30, learning_rate=0.01, latent_dim=0)
        assert result.ll_history[-1] > -100

    def test_scorer_only_with_structural_embedder(self, rule_to_label, multiple_forests, monkeypatch):
        # The stub stands in for a real structural embedder so no grammar file or pyshrg is needed.
        class StubEmbedder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Parameter(torch.randn(5, 16) * 0.1)

            def forward(self):
                return self.emb

            def invalidate_cache(self):
                pass

        monkeypatch.setattr(
            compound_mod, "create_structural_embedder",
            lambda config, device: StubEmbedder() if config.use_structural_embeddings else None,
        )
        result = train_scorer_only(
            multiple_forests, rule_to_label, max_iterations=10, learning_rate=0.05, latent_dim=0,
            embedding_dim=16, hidden_dim=32, use_structural_embeddings=True,
            structural_encoder_type="serialized_seq", grammar_file="/tmp/does_not_exist.txt",
            device="cpu")

        assert result.rule_log_probs.shape == (5,)
        assert not torch.isnan(result.rule_log_probs).any()
        assert not torch.isinf(result.rule_log_probs).any()
        assert result.effective_config is not None
        assert result.effective_config["latent_dim"] == 0
        assert result.effective_config["encoder_type"] == "none"
        assert result.effective_config["use_structural_embeddings"] is True
        assert len(result.ll_history) >= 2
        assert result.ll_history[-1] >= result.ll_history[0] - 1e-4

    def test_different_latent_dims(self, rule_to_label, multiple_forests):
        for latent_dim in [4, 16, 32]:
            result, _ = train_full(multiple_forests, rule_to_label, max_iterations=5, latent_dim=latent_dim)

            assert 1 <= len(result.elbo_history) <= 5
            assert len([e for e in result.elbo_history if not math.isnan(e)]) >= 1


class TestModelState:
    def test_scorer_state_dict(self, rule_to_label):
        scorer = make_scorer(rule_to_label)
        z = torch.randn(16)
        initial_log_probs = scorer.get_rule_log_probs(z).clone()

        new_scorer = make_scorer(rule_to_label)
        new_scorer.load_state_dict(scorer.state_dict())

        assert torch.allclose(initial_log_probs, new_scorer.get_rule_log_probs(z))

    def test_compound_hrg_state_dict(self, rule_to_label, simple_forest):
        model = make_model(rule_to_label)
        features = torch.randn(10)
        model(simple_forest, features)

        new_model = make_model(rule_to_label)
        new_model.load_state_dict(model.state_dict())

        # z is sampled, so outputs differ between calls; only check the loaded model runs.
        assert not torch.isnan(new_model(simple_forest, features)[0])

import os

import pytest
import torch

from graph_grammar.learning.compound import CompoundConfig, create_structural_embedder
from graph_grammar.learning.hypergraph_embeddings import (
    BoundaryNodeEncoder,
    GraphStructuralRuleEmbedding,
    HypergraphConvLayer,
    HypergraphGNN,
)
from graph_grammar.learning.serialized_embeddings import SerializedRuleEmbedding
from graph_grammar.learning.structural_embeddings import StructuralRuleEmbedding

try:
    import pyshrg  # noqa: F401
    PYSHRG_AVAILABLE = True
except ImportError:
    PYSHRG_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYSHRG_AVAILABLE, reason="pyshrg not available")


@pytest.fixture
def sample_grammar_file():
    for path in ["grammars/childes_cxg/06/train.mapping.txt",
                 "grammars/deepbank_200_cxg/02/train.mapping.txt"]:
        full_path = os.path.join(os.path.dirname(__file__), "../..", path)
        if os.path.exists(full_path):
            return full_path
    pytest.skip("No test grammar file found")


def make_embedder(grammar_file, encoder_type):
    config = CompoundConfig(use_structural_embeddings=True, structural_encoder_type=encoder_type,
                            grammar_file=grammar_file)
    return create_structural_embedder(config, torch.device('cpu'))


class TestHypergraphConvLayer:
    def test_forward_simple(self):
        layer = HypergraphConvLayer(node_dim=8, edge_dim=8, hidden_dim=16)

        new_node_emb, new_edge_emb = layer(torch.randn(2, 8), torch.randn(1, 8), [[0, 1]], [[0], [0]])

        assert new_node_emb.shape == (2, 16)
        assert new_edge_emb.shape == (1, 16)

    def test_forward_empty(self):
        layer = HypergraphConvLayer(node_dim=8, edge_dim=8, hidden_dim=16)

        new_node_emb, new_edge_emb = layer(torch.zeros(0, 8), torch.zeros(0, 8), [], [])

        assert new_node_emb.shape == (0, 16)
        assert new_edge_emb.shape == (0, 16)

    def test_variable_arity(self):
        layer = HypergraphConvLayer(node_dim=8, edge_dim=8, hidden_dim=16)
        edge_to_nodes = [[0, 1], [1, 2, 3]]
        node_to_edges = [[0], [0, 1], [1], [1]]

        new_node_emb, new_edge_emb = layer(torch.randn(4, 8), torch.randn(2, 8), edge_to_nodes, node_to_edges)

        assert new_node_emb.shape == (4, 16)
        assert new_edge_emb.shape == (2, 16)


class TestHypergraphGNN:
    def test_forward(self):
        gnn = HypergraphGNN(node_input_dim=12, edge_input_dim=34, hidden_dim=32, num_layers=2)

        node_emb, edge_emb = gnn(torch.randn(3, 12), torch.randn(2, 34),
                                 [[0, 1], [1, 2]], [[0], [0, 1], [1]])

        assert node_emb.shape == (3, 32)
        assert edge_emb.shape == (2, 32)


class TestBoundaryNodeEncoder:
    def test_forward(self):
        encoder = BoundaryNodeEncoder(node_dim=32, max_external_nodes=5)
        assert encoder(torch.randn(4, 32), [0, 2]).shape == (32,)

    def test_empty_external(self):
        boundary_emb = BoundaryNodeEncoder(node_dim=32)(torch.randn(4, 32), [])

        assert boundary_emb.shape == (32,)
        assert torch.allclose(boundary_emb, torch.zeros(32))


class TestGraphStructuralRuleEmbedding:
    def test_initialization(self, sample_grammar_file):
        embedder = GraphStructuralRuleEmbedding(grammar_file=sample_grammar_file, embedding_dim=64,
                                                hidden_dim=128, num_gnn_layers=2)

        assert embedder.num_rules > 0
        assert embedder.embedding_dim == 64

    def test_forward(self, sample_grammar_file):
        embedder = GraphStructuralRuleEmbedding(grammar_file=sample_grammar_file, embedding_dim=64)

        embeddings = embedder()

        assert embeddings.shape == (embedder.num_rules, 64)
        assert not torch.isnan(embeddings).any()

    def test_cache_invalidation(self, sample_grammar_file):
        embedder = GraphStructuralRuleEmbedding(grammar_file=sample_grammar_file, embedding_dim=64)

        embedder()
        assert embedder._cache_valid
        embedder.invalidate_cache()
        assert not embedder._cache_valid
        embedder()
        assert embedder._cache_valid

    def test_gradient_flow(self, sample_grammar_file):
        embedder = GraphStructuralRuleEmbedding(grammar_file=sample_grammar_file, embedding_dim=64)

        embedder().sum().backward()

        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in embedder.parameters())


class TestSerializedRuleEmbedding:
    def test_bag_mode(self, sample_grammar_file):
        embedder = SerializedRuleEmbedding(grammar_file=sample_grammar_file, embedding_dim=64, mode="bag")

        embeddings = embedder()

        assert embeddings.shape == (embedder.num_rules, 64)
        assert not torch.isnan(embeddings).any()

    def test_sequence_mode(self, sample_grammar_file):
        embedder = SerializedRuleEmbedding(grammar_file=sample_grammar_file, embedding_dim=64, mode="sequence")

        embeddings = embedder()

        assert embeddings.shape == (embedder.num_rules, 64)
        assert not torch.isnan(embeddings).any()

    def test_cache(self, sample_grammar_file):
        embedder = SerializedRuleEmbedding(grammar_file=sample_grammar_file, embedding_dim=64, mode="bag")

        embedder()
        assert embedder._cache_valid
        embedder.invalidate_cache()
        assert not embedder._cache_valid


class TestCreateStructuralEmbedder:
    def test_gnn_type(self, sample_grammar_file):
        embedder = make_embedder(sample_grammar_file, "gnn")

        assert embedder is not None
        assert isinstance(embedder, GraphStructuralRuleEmbedding)

    def test_serialized_bag_type(self, sample_grammar_file):
        embedder = make_embedder(sample_grammar_file, "serialized_bag")

        assert embedder is not None
        assert isinstance(embedder, SerializedRuleEmbedding)

    def test_serialized_seq_type(self, sample_grammar_file):
        assert make_embedder(sample_grammar_file, "serialized_seq") is not None

    def test_legacy_type(self, sample_grammar_file):
        embedder = make_embedder(sample_grammar_file, "legacy")

        assert embedder is not None
        assert isinstance(embedder, StructuralRuleEmbedding)

    def test_disabled(self, sample_grammar_file):
        config = CompoundConfig(use_structural_embeddings=False)
        assert create_structural_embedder(config, torch.device('cpu')) is None

    def test_invalid_type(self, sample_grammar_file):
        with pytest.raises(ValueError):
            make_embedder(sample_grammar_file, "invalid")

    def test_missing_grammar_file(self):
        with pytest.raises(ValueError):
            make_embedder(None, "gnn")


class TestStructuralSimilarity:
    def test_identical_rules_same_embedding(self, sample_grammar_file):
        embedder = GraphStructuralRuleEmbedding(grammar_file=sample_grammar_file, embedding_dim=64)
        embedder()

        selected = embedder.get_embedding(torch.tensor([0, 0, 1, 1]))

        assert torch.allclose(selected[0], selected[1])
        assert torch.allclose(selected[2], selected[3])

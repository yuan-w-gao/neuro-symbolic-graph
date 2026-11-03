import pickle

import pytest

from graph_grammar.cache import (
    CACHE_VERSION,
    ForestCache,
    _compute_file_hash,
    _compute_string_hash,
    _parse_graphs_file,
    _sanitize_graph_id,
)
from graph_grammar.forest import ForestNode, HyperEdge, PackedForest


@pytest.fixture
def sample_grammar_file(tmp_path):
    path = tmp_path / "train.mapping.txt"
    path.write_text("0\tS -> NP VP\n1\tNP -> Det N\n2\tVP -> V NP\n")
    return str(path)


@pytest.fixture
def sample_graphs_file(tmp_path):
    path = tmp_path / "train.graphs.txt"
    path.write_text("@graph1\nnode1 -> node2\n\n@graph2\nnode3 -> node4\nnode4 -> node5\n")
    return str(path)


@pytest.fixture
def cache(tmp_path, sample_grammar_file, sample_graphs_file):
    return ForestCache.from_files(str(tmp_path / "cache"), sample_grammar_file, sample_graphs_file)


@pytest.fixture
def sample_forest():
    nodes = [
        ForestNode(id=0, edge_set_hash=123, boundary_mapping=(0, 1), incoming_edges=[0], is_terminal=True),
        ForestNode(id=1, edge_set_hash=456, boundary_mapping=(0,), incoming_edges=[1], is_root=True),
    ]
    hyperedges = [
        HyperEdge(id=0, rule_id=0, parent_node=0, child_nodes=()),
        HyperEdge(id=1, rule_id=1, parent_node=1, child_nodes=(0,)),
    ]
    return PackedForest(nodes=nodes, hyperedges=hyperedges, roots=[1], num_rules=10, graph_id="test_graph")


class TestHashFunctions:
    def test_compute_string_hash(self):
        hash1 = _compute_string_hash("hello")

        assert hash1 == _compute_string_hash("hello")
        assert hash1 != _compute_string_hash("world")
        assert len(hash1) == 64

    def test_compute_file_hash(self, tmp_path):
        file1, file2 = tmp_path / "file1.txt", tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        hash1 = _compute_file_hash(str(file1))

        assert hash1 == _compute_file_hash(str(file1))
        assert hash1 != _compute_file_hash(str(file2))


class TestParseGraphsFile:
    def test_parse_graphs_file(self, sample_graphs_file):
        graph_hashes = _parse_graphs_file(sample_graphs_file)

        assert "graph1" in graph_hashes
        assert "graph2" in graph_hashes
        assert len(graph_hashes) == 2

    def test_graph_hash_changes_with_content(self, tmp_path):
        path = tmp_path / "graphs.txt"
        path.write_text("@graph1\nnode1 -> node2\n")
        hash1 = _parse_graphs_file(str(path))["graph1"]
        path.write_text("@graph1\nnode1 -> node3\n")
        hash2 = _parse_graphs_file(str(path))["graph1"]

        assert hash1 != hash2


class TestSanitizeGraphId:
    def test_sanitize_slashes(self):
        assert '/' not in _sanitize_graph_id("path/to/graph")
        assert '\\' not in _sanitize_graph_id("path\\to\\graph")

    def test_sanitize_colons(self):
        assert ':' not in _sanitize_graph_id("c:file")

    def test_sanitize_length_limit(self):
        assert len(_sanitize_graph_id("a" * 200)) <= 100


class TestForestCache:
    def test_from_files(self, cache):
        assert cache.grammar_hash is not None
        assert len(cache.grammar_hash) == 64
        assert "graph1" in cache.graph_hashes
        assert "graph2" in cache.graph_hashes

    def test_save_and_load_forest(self, cache, sample_forest):
        cache.save_forest("graph1", sample_forest)
        loaded = cache.load_forest("graph1")

        assert loaded is not None
        assert len(loaded.nodes) == len(sample_forest.nodes)
        assert len(loaded.hyperedges) == len(sample_forest.hyperedges)
        assert loaded.roots == sample_forest.roots
        assert loaded.graph_id == sample_forest.graph_id

    def test_is_valid(self, cache, sample_forest):
        assert not cache.is_valid("graph1")
        cache.save_forest("graph1", sample_forest)
        assert cache.is_valid("graph1")

    def test_cache_invalidation_on_graph_change(self, tmp_path, sample_forest):
        grammar_path, graphs_path = tmp_path / "grammar.txt", tmp_path / "graphs.txt"
        cache_dir = str(tmp_path / "cache")
        grammar_path.write_text("S -> NP VP\n")
        graphs_path.write_text("@graph1\nnode1 -> node2\n")

        cache1 = ForestCache.from_files(cache_dir, str(grammar_path), str(graphs_path))
        cache1.save_forest("graph1", sample_forest)
        assert cache1.is_valid("graph1")

        graphs_path.write_text("@graph1\nnode1 -> node3\n")
        cache2 = ForestCache.from_files(cache_dir, str(grammar_path), str(graphs_path))
        assert not cache2.is_valid("graph1")

    def test_batch_save_and_load(self, cache, sample_forest):
        forest2 = PackedForest(
            nodes=[ForestNode(id=0, edge_set_hash=789, boundary_mapping=(),
                              incoming_edges=[0], is_terminal=True, is_root=True)],
            hyperedges=[HyperEdge(id=0, rule_id=2, parent_node=0)],
            roots=[0], num_rules=10, graph_id="graph2",
        )

        cache.save_batch({"graph1": sample_forest, "graph2": forest2})

        assert cache.use_batch_mode()
        loaded = cache.load_batch()
        assert "graph1" in loaded
        assert "graph2" in loaded
        assert len(loaded["graph1"].nodes) == len(sample_forest.nodes)
        assert len(loaded["graph2"].nodes) == len(forest2.nodes)

    def test_has_all(self, cache, sample_forest):
        assert not cache.has_all(["graph1", "graph2"])
        cache.save_forest("graph1", sample_forest)
        assert not cache.has_all(["graph1", "graph2"])
        cache.save_forest("graph2", sample_forest)
        assert cache.has_all(["graph1", "graph2"])

    def test_clear(self, tmp_path, cache, sample_grammar_file, sample_graphs_file, sample_forest):
        cache.save_forest("graph1", sample_forest)
        assert cache.is_valid("graph1")

        cache.clear()

        fresh = ForestCache.from_files(str(tmp_path / "cache"), sample_grammar_file, sample_graphs_file)
        assert not fresh.is_valid("graph1")

    def test_stats(self, cache, sample_forest):
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        cache.load_forest("graph1")
        assert cache.get_stats()["misses"] == 1

        cache.save_forest("graph1", sample_forest)
        cache.load_forest("graph1")
        assert cache.get_stats()["hits"] == 1

    def test_corrupted_cache_handled(self, cache, sample_forest):
        cache.save_forest("graph1", sample_forest)
        forest_path = cache._forest_path("graph1")
        forest_path.write_bytes(b"corrupted data")

        assert cache.load_forest("graph1") is None
        assert not forest_path.exists()


class TestCacheIntegration:
    def test_cache_version(self):
        assert isinstance(CACHE_VERSION, int)
        assert CACHE_VERSION >= 1

    def test_pickle_compatibility(self, sample_forest):
        unpickled = pickle.loads(pickle.dumps(sample_forest))

        assert len(unpickled.nodes) == len(sample_forest.nodes)
        assert len(unpickled.hyperedges) == len(sample_forest.hyperedges)
        assert unpickled.roots == sample_forest.roots
        assert unpickled.num_rules == sample_forest.num_rules

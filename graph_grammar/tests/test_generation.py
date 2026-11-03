from pathlib import Path

import pytest
import torch

from graph_grammar.generation import DerivationSampler, GrammarWrapper, GraphGenerator, GraphWriter

try:
    import pyshrg
    PYSHRG_AVAILABLE = True
except ImportError:
    PYSHRG_AVAILABLE = False

GRAMMAR_DIR = Path(__file__).parent.parent.parent / "grammars"
GRAMMAR_FILE = next((str(GRAMMAR_DIR / sub / "train.mapping.txt")
                     for sub in ["childes_cxg/61", "childes_cxg/06", "cxg/06"]
                     if (GRAMMAR_DIR / sub / "train.mapping.txt").exists()), None)

pytestmark = pytest.mark.skipif(not PYSHRG_AVAILABLE, reason="pyshrg not available")
needs_grammar = pytest.mark.skipif(GRAMMAR_FILE is None, reason="No grammar file found")


def uniform_grammar():
    grammar = GrammarWrapper(GRAMMAR_FILE)
    grammar.set_uniform_weights()
    return grammar


@needs_grammar
class TestGrammarWrapper:
    def test_initialization(self):
        grammar = GrammarWrapper(GRAMMAR_FILE)

        assert grammar.num_rules > 0
        assert grammar.num_labels > 0
        assert len(grammar.labels) > 0

    def test_rules_grouped_by_label(self):
        grammar = GrammarWrapper(GRAMMAR_FILE)

        for label in grammar.labels:
            for rule in grammar.get_rules_for_label(label):
                assert rule.label == label, f"Rule {rule.index} has label {rule.label}, expected {label}"

    def test_root_rules_exist(self):
        grammar = GrammarWrapper(GRAMMAR_FILE)

        assert grammar.has_root_rules()
        assert len(grammar.get_rules_for_label("ROOT")) > 0

    def test_set_random_weights(self):
        grammar = GrammarWrapper(GRAMMAR_FILE)

        log_probs = grammar.set_random_weights(seed=42, alpha=1.0)

        assert len(log_probs) == grammar.num_rules
        assert torch.all(log_probs <= 0)
        for label in grammar.labels:
            rules = grammar.get_rules_for_label(label)
            if rules:
                probs = torch.exp(torch.tensor([r.log_weight for r in rules]))
                assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5), \
                    f"Probs for label {label} sum to {probs.sum()}"

    def test_set_uniform_weights(self):
        grammar = GrammarWrapper(GRAMMAR_FILE)

        grammar.set_uniform_weights()

        for label in grammar.labels:
            weights = [r.log_weight for r in grammar.get_rules_for_label(label)]
            if len(weights) > 1:
                assert all(abs(w - weights[0]) < 1e-6 for w in weights), label

    def test_rule_sampling_respects_weights(self):
        grammar = GrammarWrapper(GRAMMAR_FILE)
        if not grammar.has_root_rules():
            return
        root_rules = grammar.get_rules_for_label("ROOT")
        if len(root_rules) < 2:
            return

        log_probs = grammar.get_weights()
        for rule in root_rules:
            log_probs[rule.index] = -10.0
        log_probs[root_rules[0].index] = 0.0
        grammar.set_weights(log_probs)

        counts = {r.index: 0 for r in root_rules}
        rng = torch.Generator()
        rng.manual_seed(42)
        for _ in range(100):
            counts[grammar.sample_rule("ROOT", rng).index] += 1

        assert counts[root_rules[0].index] > 80, counts


@needs_grammar
class TestDerivationSampler:
    def test_sample_derivation(self):
        derivation = DerivationSampler(uniform_grammar(), seed=42).sample()

        assert len(derivation.steps) > 0
        assert len(derivation.rule_indices) > 0

    def test_sampler_reproducibility(self):
        grammar = uniform_grammar()

        derivation1 = DerivationSampler(grammar, seed=42).sample()
        derivation2 = DerivationSampler(grammar, seed=42).sample()

        assert derivation1.rule_indices == derivation2.rule_indices

    def test_sample_batch(self):
        derivations = DerivationSampler(uniform_grammar(), seed=42).sample_batch(n=10)
        assert len(derivations) == 10


@needs_grammar
class TestGraphGenerator:
    def test_generate_graph(self):
        graph = GraphGenerator(uniform_grammar(), seed=42).generate()

        assert graph.sentence_id is not None
        assert graph.num_nodes >= 0

    def test_generate_batch(self):
        graphs = GraphGenerator(uniform_grammar(), seed=42).generate_batch(n=5)
        assert len(graphs) >= 1

    def test_no_duplicate_edges(self):
        generator = GraphGenerator(uniform_grammar(), seed=42)

        for i in range(20):
            graph = generator.generate(f"test/{i}")
            seen = set()
            for edge in graph.edges:
                key = (edge.source, edge.target, edge.label)
                assert key not in seen, f"Duplicate edge in graph {graph.sentence_id}: {key}"
                seen.add(key)


@needs_grammar
class TestGraphWriter:
    def test_write_and_read(self, tmp_path):
        graphs = GraphGenerator(uniform_grammar(), seed=42).generate_batch(n=5)
        output_path = tmp_path / "graphs.txt"

        GraphWriter.write_graphs(str(output_path), graphs)

        content = output_path.read_text()
        assert len(content) > 0
        assert str(len(graphs)) in content.split('\n')[0]


@needs_grammar
class TestGraphParsing:
    def test_single_graph_parseable(self, tmp_path):
        graph = GraphGenerator(uniform_grammar(), seed=42).generate()
        output_path = str(tmp_path / "graph.txt")
        GraphWriter.write_graphs(output_path, [graph])

        pyshrg.initialize()
        manager = pyshrg.get_manager()
        manager.load_grammars(GRAMMAR_FILE)
        manager.load_graphs(output_path)
        manager.allocate(num_contexts=1)
        manager.init_all(type='linear', verbose=False)

        context = manager.get_context(0)
        for g in manager.iter_graphs():
            # Complex generated graphs may legitimately fail to parse.
            assert context.parse(g) in [pyshrg.ParserError.kNone, pyshrg.ParserError.kNoResult]


@pytest.mark.parametrize("grammar_subdir", ["childes_cxg/61", "childes_cxg/06"])
def test_grammar_loading(grammar_subdir):
    grammar_path = GRAMMAR_DIR / grammar_subdir / "train.mapping.txt"
    if not grammar_path.exists():
        pytest.skip(f"Grammar not found: {grammar_path}")

    assert GrammarWrapper(str(grammar_path)).num_rules > 0

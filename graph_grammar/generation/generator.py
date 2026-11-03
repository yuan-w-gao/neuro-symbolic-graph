from collections import defaultdict
from dataclasses import dataclass, field

from .grammar import GrammarWrapper
from .sampler import DerivationSampler

# Single-character POS tags of the ERG predicate naming convention (C++ expects one char).
KNOWN_POS = frozenset("nvapqcdxu")
EVENT_LABELS = {"unknown", "loc_nonsp", "poss", "compound"}


@dataclass
class NodeInfo:
    index: int
    node_id: str
    label: str
    lemma: str = "#"
    pos_tag: str = "#"
    sense: str = "#"
    carg: str = "#"
    properties: list[str] = field(default_factory=lambda: ["#"] * 5)


@dataclass
class EdgeInfo:
    source: int
    target: int
    label: str


@dataclass
class GeneratedGraph:
    sentence_id: str
    sentence: str = ""
    lemma_sequence: str = ""
    nodes: list[NodeInfo] = field(default_factory=list)
    edges: list[EdgeInfo] = field(default_factory=list)
    top_index: int = 0
    rule_indices: list[int] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def is_connected(self) -> bool:
        if self.num_nodes <= 1:
            return True
        neighbours = defaultdict(set)
        for edge in self.edges:
            neighbours[edge.source].add(edge.target)
            neighbours[edge.target].add(edge.source)
        seen = {0}
        stack = [0]
        while stack:
            for other in neighbours[stack.pop()] - seen:
                seen.add(other)
                stack.append(other)
        return len(seen) == self.num_nodes

    def has_valid_structure(self) -> bool:
        return self.num_nodes > 0 and self.is_connected()


def _node_id(index, label):
    if "_v_" in label or label in EVENT_LABELS:
        return f"e{index}"
    if "_n_" in label or label == "pron":
        return f"x{index}"
    return f"i{index}"


def _lemma(label):
    if label.endswith("_q"):
        return label[:-2]
    if label.startswith("_"):
        parts = label[1:].split("_")
        return "_".join(parts[:-2]) if len(parts) >= 3 else parts[0]
    return label


def _pos(label):
    if label.endswith("_q"):
        return "q"
    parts = label[1:].split("_") if label.startswith("_") else label.split("_")
    # _lemma_pos_sense (_come_v_1), _lemma_pos (_of_p) and nonlexical_pos (focus_d) put the
    # POS in different slots; only membership in KNOWN_POS tells them apart.
    if len(parts) >= 3 and parts[-2] in KNOWN_POS:
        return parts[-2]
    if len(parts) >= 2 and parts[-1] in KNOWN_POS:
        return parts[-1]
    return "#"


def _sense(label):
    if label.endswith("_q") or not label.startswith("_"):
        return "#"
    parts = label[1:].split("_")
    return parts[-1] if len(parts) >= 3 else "#"


class GraphGenerator:
    """Samples derivations and reads EDS graphs off their terminal edges."""

    def __init__(self, grammar: GrammarWrapper, sampler: DerivationSampler | None = None,
                 seed: int | None = None):
        self.grammar = grammar
        self.sampler = sampler if sampler is not None else DerivationSampler(grammar, seed=seed)
        self._next_graph_id = 0

    def set_seed(self, seed: int):
        self.sampler.set_seed(seed)

    def generate(self, sentence_id: str | None = None) -> GeneratedGraph:
        derivation = self.sampler.sample()
        if sentence_id is None:
            sentence_id = f"gen/{self._next_graph_id}"
            self._next_graph_id += 1

        # Unary terminal edges label a node; longer ones link their first node to each of the rest.
        node_labels = {}
        edges = {}
        for label, nodes in derivation.terminal_edges:
            if len(nodes) == 1:
                if node_labels.get(nodes[0], "unknown") == "unknown":
                    node_labels[nodes[0]] = label
            elif nodes:
                source, *targets = nodes
                node_labels.setdefault(source, "unknown")
                for target in targets:
                    edges[label, source, target] = None
                    node_labels.setdefault(target, "unknown")

        index_of = {old: new for new, old in enumerate(sorted(node_labels))}
        nodes = []
        for old, new in index_of.items():
            label = node_labels[old]
            nodes.append(NodeInfo(index=new, node_id=_node_id(new, label), label=label,
                                  lemma=_lemma(label), pos_tag=_pos(label), sense=_sense(label)))
        top_index = next((n.index for n in nodes if n.node_id.startswith("e")), 0)
        lemma_sequence = " ".join(n.lemma for n in nodes if n.lemma not in ("#", "unknown")) or "unknown"

        return GeneratedGraph(
            sentence_id=sentence_id,
            sentence=lemma_sequence + " .",
            lemma_sequence=lemma_sequence,
            nodes=nodes,
            edges=[EdgeInfo(index_of[src], index_of[tgt], label) for label, src, tgt in edges],
            top_index=top_index,
            rule_indices=derivation.rule_indices,
        )

    def generate_batch(self, n: int, id_prefix: str = "gen", validate: bool = True,
                       max_retries_per_graph: int = 10) -> list[GeneratedGraph]:
        """Up to n graphs; failed or (with validate) disconnected samples count as attempts."""
        graphs = []
        attempts = 0
        while len(graphs) < n and attempts < n * max_retries_per_graph:
            attempts += 1
            try:
                graph = self.generate(f"{id_prefix}/{len(graphs)}")
            except RuntimeError:
                continue
            if not validate or graph.has_valid_structure():
                graphs.append(graph)
        return graphs

    def generate_with_retry(self, sentence_id: str | None = None,
                            max_retries: int = 10) -> GeneratedGraph | None:
        for _ in range(max_retries):
            try:
                return self.generate(sentence_id)
            except RuntimeError:
                continue
        return None

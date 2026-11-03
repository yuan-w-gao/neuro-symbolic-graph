from collections import deque
from dataclasses import dataclass, field

import torch

from .grammar import GrammarWrapper


@dataclass
class DerivationStep:
    rule_index: int
    rule_label: str
    node_mapping: dict[int, int]      # fragment node -> graph node
    external_mapping: dict[int, int]  # external position -> graph node


@dataclass
class Derivation:
    steps: list[DerivationStep] = field(default_factory=list)
    terminal_edges: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    rule_indices: list[int] = field(default_factory=list)


class DerivationSampler:
    """Top-down, breadth-first sampling of derivations from the grammar's current weights."""

    def __init__(self, grammar: GrammarWrapper, seed: int | None = None, max_depth: int = 100,
                 max_terminal_edges: int | None = None):
        self.grammar = grammar
        self.max_depth = max_depth
        # Depth alone does not bound a supercritical grammar: breadth-first, a derivation
        # can grow without limit while staying shallow.
        self.max_terminal_edges = max_terminal_edges
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def set_seed(self, seed: int):
        self.rng.manual_seed(seed)

    def sample(self, start_symbol: str | None = None) -> Derivation:
        if start_symbol is None:
            start_symbol = self.grammar.get_start_symbol()

        derivation = Derivation()
        next_node_id = 0
        queue = deque([(start_symbol, {}, 0)])
        while queue:
            label, parent_mapping, depth = queue.popleft()
            if depth > self.max_depth:
                raise RuntimeError(f"Derivation exceeded max depth {self.max_depth}")
            if self.max_terminal_edges is not None and \
                    len(derivation.terminal_edges) > self.max_terminal_edges:
                raise RuntimeError(
                    f"Derivation exceeded {self.max_terminal_edges} terminal edges")

            try:
                rule = self.grammar.sample_rule(label, self.rng, arity=len(parent_mapping) or None)
            except ValueError:
                continue

            node_mapping = {}
            external_mapping = {}
            for i, ext_node in enumerate(rule.external_nodes):
                if i in parent_mapping:
                    graph_node = parent_mapping[i]
                else:
                    graph_node = next_node_id
                    next_node_id += 1
                node_mapping[ext_node] = graph_node
                external_mapping[i] = graph_node
            for node in rule.nodes:
                if node.index not in node_mapping:
                    node_mapping[node.index] = next_node_id
                    next_node_id += 1

            derivation.steps.append(DerivationStep(rule.index, rule.label, node_mapping, external_mapping))
            derivation.rule_indices.append(rule.index)
            for edge in rule.terminal_edges:
                derivation.terminal_edges.append(
                    (edge.label, tuple(node_mapping[n] for n in edge.linked_nodes)))
            for edge in rule.nonterminal_edges:
                child_mapping = {i: node_mapping[n] for i, n in enumerate(edge.linked_nodes)}
                queue.append((edge.label, child_mapping, depth + 1))

        return derivation

    def sample_batch(self, n: int, start_symbol: str | None = None) -> list[Derivation]:
        return [self.sample(start_symbol) for _ in range(n)]

    def sample_with_retry(self, start_symbol: str | None = None,
                          max_retries: int = 10) -> Derivation | None:
        for _ in range(max_retries):
            try:
                return self.sample(start_symbol)
            except RuntimeError:
                continue
        return None

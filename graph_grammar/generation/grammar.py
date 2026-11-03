import math
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FragmentEdge:
    label: str
    label_index: int
    is_terminal: bool
    linked_nodes: tuple[int, ...]


@dataclass
class FragmentNode:
    index: int
    is_external: bool
    linked_edges: tuple[int, ...]


@dataclass
class RuleInfo:
    index: int
    label: str
    label_index: int
    log_weight: float
    terminal_edges: list[FragmentEdge]
    nonterminal_edges: list[FragmentEdge]
    external_nodes: list[int]
    nodes: list[FragmentNode]
    num_occurrences: int = 0


def _edges(edges, is_terminal):
    return [FragmentEdge(e.label, e.label_index, is_terminal, tuple(n.index for n in e.linked_nodes))
            for e in edges]


def _rule_info(index, hrg):
    return RuleInfo(
        index=index,
        label=hrg.label,
        label_index=hrg.label_index,
        log_weight=hrg.log_rule_weight,
        terminal_edges=_edges(hrg.terminal_edges, True),
        nonterminal_edges=_edges(hrg.nonterminal_edges, False),
        external_nodes=[n.index for n in hrg.external_nodes],
        nodes=[FragmentNode(n.index, n.is_external, tuple(n.linked_edges)) for n in hrg.nodes],
        num_occurrences=hrg.num_occurences,
    )


class GrammarWrapper:
    """The HRG rules of a train.mapping.txt, grouped by LHS label."""

    def __init__(self, grammar_file: str):
        import pyshrg

        pyshrg.initialize()
        self.manager = pyshrg.get_manager()
        self.manager.load_grammars(grammar_file)

        self._rules = [_rule_info(i, hrg) for i, hrg in enumerate(self.manager.iter_hrgs())]
        self._rules_by_label = defaultdict(list)
        self._label_to_id = {}
        self._id_to_label = {}
        for rule in self._rules:
            self._rules_by_label[rule.label].append(rule)
            if rule.label not in self._label_to_id:
                self._label_to_id[rule.label] = rule.label_index
                self._id_to_label[rule.label_index] = rule.label
        self._log_probs = None

    @property
    def num_rules(self) -> int:
        return len(self._rules)

    @property
    def labels(self) -> list[str]:
        return list(self._rules_by_label.keys())

    @property
    def num_labels(self) -> int:
        return len(self._rules_by_label)

    def get_rule(self, rule_idx: int) -> RuleInfo:
        return self._rules[rule_idx]

    def get_rules_for_label(self, label: str) -> list[RuleInfo]:
        return self._rules_by_label.get(label, [])

    def get_label_id(self, label: str) -> int:
        return self._label_to_id.get(label, -1)

    def get_label_name(self, label_id: int) -> str:
        return self._id_to_label.get(label_id, "")

    def get_rule_to_label_tensor(self) -> Tensor:
        return torch.tensor([r.label_index for r in self._rules], dtype=torch.long)

    def sample_rule(self, label: str, rng: torch.Generator | None = None,
                    arity: int | None = None) -> RuleInfo:
        rules = self.get_rules_for_label(label)
        if not rules:
            raise ValueError(f"No rules for label '{label}'")
        # The parser matches on (label, arity), so a nonterminal is only rewritten by rules
        # with as many external nodes as it has attachments; weights normalize within that group.
        if arity is not None:
            rules = [r for r in rules if len(r.external_nodes) == arity]
            if not rules:
                raise ValueError(f"No rules for label '{label}' with arity {arity}")
        if len(rules) == 1:
            return rules[0]

        probs = torch.softmax(torch.tensor([r.log_weight for r in rules]), dim=0)
        if rng is None:
            return rules[torch.multinomial(probs, 1).item()]
        return rules[torch.multinomial(probs, 1, generator=rng).item()]

    def set_weights(self, log_probs: Tensor):
        if len(log_probs) != self.num_rules:
            raise ValueError(f"Expected {self.num_rules} log probs, got {len(log_probs)}")
        self._log_probs = log_probs.clone()
        for rule, log_prob in zip(self._rules, log_probs):
            rule.log_weight = log_prob.item()
        # iter_hrgs yields references; get_shrg(i) returns copies, so assigning to those is a no-op.
        for rule_idx, hrg in enumerate(self.manager.iter_hrgs()):
            hrg.log_rule_weight = log_probs[rule_idx].item()

    def shrg_to_hrg_index(self) -> Tensor:
        # Several SHRGs (semantic + syntactic side) share one HRG (semantic side). Training
        # indexes SHRGs, generation indexes HRGs; only the manager knows which maps to which.
        return torch.tensor(
            [self.manager.index_of_hrg(self.manager.get_shrg(i))
             for i in range(self.manager.shrg_size)],
            dtype=torch.long,
        )

    def marginalize_shrg_log_probs(self, shrg_log_probs: Tensor) -> Tensor:
        """log P(shrg | label) -> log P(hrg | label), summing out the syntactic side."""
        mapping = self.shrg_to_hrg_index()
        if len(shrg_log_probs) != len(mapping):
            raise ValueError(
                f"expected {len(mapping)} SHRG log probs, got {len(shrg_log_probs)}")
        size = self.num_rules
        peak = torch.full((size,), float("-inf")).scatter_reduce(
            0, mapping, shrg_log_probs, reduce="amax", include_self=True)
        totals = torch.zeros(size).scatter_add(
            0, mapping, (shrg_log_probs - peak[mapping]).exp())
        out = peak + totals.clamp(min=1e-300).log()

        for rules in self._rules_by_label.values():
            idx = torch.tensor([r.index for r in rules], dtype=torch.long)
            out[idx] = out[idx] - torch.logsumexp(out[idx], dim=0)
        return out

    def get_weights(self) -> Tensor:
        if self._log_probs is not None:
            return self._log_probs.clone()
        return torch.tensor([r.log_weight for r in self._rules])

    def set_random_weights(self, seed: int | None = None, alpha: float = 1.0) -> Tensor:
        """Draws each label's rule distribution from a symmetric Dirichlet(alpha)."""
        if seed is not None:
            torch.manual_seed(seed)
        log_probs = torch.zeros(self.num_rules)
        for rules in self._rules_by_label.values():
            probs = torch.distributions.Dirichlet(torch.full((len(rules),), alpha)).sample()
            for rule, p in zip(rules, probs):
                log_probs[rule.index] = torch.log(p)
        self.set_weights(log_probs)
        return log_probs

    def set_uniform_weights(self) -> Tensor:
        log_probs = torch.zeros(self.num_rules)
        for rules in self._rules_by_label.values():
            log_probs[[r.index for r in rules]] = -math.log(len(rules))
        self.set_weights(log_probs)
        return log_probs

    def has_root_rules(self) -> bool:
        return "ROOT" in self._rules_by_label

    def get_start_symbol(self) -> str:
        if "ROOT" in self._rules_by_label:
            return "ROOT"
        if self.labels:
            return self.labels[0]
        raise ValueError("Grammar has no rules")

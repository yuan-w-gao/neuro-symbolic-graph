import math
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import Tensor

from ..inference import expected_counts


def delexicalize(label: str) -> str:
    """``_permission_n_1`` -> ``_X_n_1``; non-lexical labels pass through."""
    # Same rule as the C++ parser for unknown lexical labels (edsgraph.cpp:76-84).
    if not label.startswith("_"):
        return label
    cut = label.find("_", 1)
    return "_X" + label[cut:] if cut != -1 else "_X"


def rule_signature(rule) -> str:
    """The rule with lexical predicates anonymized; shared by its lexical variants."""
    terminals = sorted(
        (delexicalize(e.label), tuple(e.linked_nodes)) for e in rule.terminal_edges)
    nonterminals = sorted(
        (e.label, tuple(e.linked_nodes)) for e in rule.nonterminal_edges)
    return (f"{rule.label}|{len(rule.external_nodes)}|"
            f"{terminals}|{nonterminals}|{tuple(sorted(rule.external_nodes))}")


def build_signatures(grammar) -> tuple[list[int], list[list[int]]]:
    """Returns ``(sig_of_rule, rules_of_sig)`` over HRG rule indices."""
    by_signature = {}
    sig_of_rule = [0] * grammar.num_rules
    rules_of_sig = []
    for rule in grammar._rules:
        key = rule_signature(rule)
        if key not in by_signature:
            by_signature[key] = len(rules_of_sig)
            rules_of_sig.append([])
        sig = by_signature[key]
        sig_of_rule[rule.index] = sig
        rules_of_sig[sig].append(rule.index)
    return sig_of_rule, rules_of_sig


def _groups(grammar):
    # Normalize per (label, arity), not per label: that is the partition the C++ EM and
    # the generator use (181 vs 352 groups on redwoods_pool4k).
    groups = defaultdict(list)
    for label, rules in grammar._rules_by_label.items():
        for r in rules:
            groups[(label, len(r.external_nodes))].append(r.index)
    return list(groups.values())


def estimate_flat(counts: Tensor, grammar, alpha: float = 0.0) -> Tensor:
    """Per-group multinomial with a symmetric Dirichlet(alpha) prior; alpha=0 is the MLE."""
    log_probs = torch.full((grammar.num_rules,), -math.inf, dtype=torch.float64)
    for indices in _groups(grammar):
        idx = torch.tensor(indices, dtype=torch.long)
        c = counts[idx].double() + alpha
        total = c.sum()
        if total <= 0:
            log_probs[idx] = -math.log(len(indices))
            continue
        with torch.no_grad():
            log_probs[idx] = torch.where(
                c > 0, (c / total).log(), torch.full_like(c, -math.inf))
    return log_probs.float()


def _backoff_weights(kind, indices, sig_of_rule, rules_of_sig, sig_counts, sig_distinct, alpha):
    # "mass" gives each member its whole class's count, so large classes swamp the group;
    # "per_member" splits it, "continuation" counts distinct observed members (Kneser-Ney).
    sigs = [sig_of_rule[i] for i in indices]
    if kind == "mass":
        weights = [sig_counts[s] + alpha for s in sigs]
    elif kind == "per_member":
        weights = [sig_counts[s] / max(len(rules_of_sig[s]), 1) + alpha for s in sigs]
    elif kind == "continuation":
        weights = [sig_distinct[s] + alpha for s in sigs]
    else:
        raise ValueError(f"unknown backoff_kind {kind!r}")
    return torch.tensor(weights, dtype=torch.float64)


def estimate_backoff(counts: Tensor, grammar, sig_of_rule: Sequence[int],
                     rules_of_sig: Sequence[Sequence[int]],
                     alpha: float = 0.0, discount: float = 0.75,
                     backoff_kind: str = "continuation") -> Tensor:
    """Absolute discounting within each group, the reserved mass spread by signature class.

    p(rule) = max(c_rule - d, 0) / c_group + lambda_group * p_backoff(rule)
    """
    log_probs = torch.full((grammar.num_rules,), -math.inf, dtype=torch.float64)
    sig_counts = defaultdict(float)
    sig_distinct = defaultdict(int)
    for index in range(grammar.num_rules):
        sig_counts[sig_of_rule[index]] += float(counts[index])
        if counts[index] > 0:
            sig_distinct[sig_of_rule[index]] += 1

    for indices in _groups(grammar):
        idx = torch.tensor(indices, dtype=torch.long)
        c = counts[idx].double()
        total = float(c.sum())
        if total <= 0:
            log_probs[idx] = -math.log(len(indices))
            continue

        observed = int((c > 0).sum())
        reserved = min(discount * observed / total, 0.9) if observed else 1.0
        weights = _backoff_weights(backoff_kind, indices, sig_of_rule, rules_of_sig,
                                   sig_counts, sig_distinct, alpha)
        if float(weights.sum()) <= 0:
            weights = torch.ones(len(indices), dtype=torch.float64)
        backoff = weights / weights.sum()

        probs = torch.clamp(c - discount, min=0.0) / total + reserved * backoff
        probs = probs / probs.sum()
        log_probs[idx] = torch.where(
            probs > 0, probs.log(), torch.full_like(probs, -math.inf))
    return log_probs.float()


def rule_counts_from_forests(forests, num_rules: int,
                             rule_log_probs: Tensor | None = None) -> Tensor:
    """Posterior rule counts under rule_log_probs, or one count per hyperedge if None."""
    counts = torch.zeros(num_rules, dtype=torch.float64)
    if rule_log_probs is None:
        for forest in forests:
            for edge in forest.hyperedges:
                if 0 <= edge.rule_id < num_rules:
                    counts[edge.rule_id] += 1.0
        return counts

    for forest in forests:
        if forest.roots:
            counts += expected_counts(forest, rule_log_probs).double()
    return counts

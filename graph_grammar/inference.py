from collections import defaultdict

import torch
from torch import Tensor

from .forest import PackedForest


def _check_rule_ids(forest: PackedForest, rule_log_probs: Tensor) -> None:
    # Raise rather than score an unknown rule as log 1, which silently inflates the likelihood.
    if not forest.hyperedges:
        return
    lo = min(e.rule_id for e in forest.hyperedges)
    hi = max(e.rule_id for e in forest.hyperedges)
    if lo < 0 or hi >= len(rule_log_probs):
        raise IndexError(
            f"forest {forest.graph_id!r} references rule ids [{lo}, {hi}] but "
            f"rule_log_probs has length {len(rule_log_probs)}. Size the weight "
            f"vector from the grammar (manager.hrg_size), not from a single "
            f"forest's num_rules.")


def _root_log_prob(forest, values, device):
    if not forest.roots:
        return torch.tensor(float('-inf'), device=device)
    return torch.logsumexp(values[torch.tensor(forest.roots, device=device)], dim=0)


def _scorer_device(node_embeddings):
    if node_embeddings:
        return next(iter(node_embeddings.values())).device
    return torch.device('cpu')


def _scorer_edge_score(forest, rule_scorer, node_embeddings):
    def score(edge, parent):
        children = [forest.nodes[c] for c in edge.child_nodes]
        return rule_scorer(edge, parent, children, node_embeddings)
    return score


def _inside(forest, edge_score, device):
    values = torch.full((len(forest.nodes),), float('-inf'), device=device)
    for node_idx in forest.topo_order:
        node = forest.nodes[node_idx]
        if not node.incoming_edges:
            values[node_idx] = 0.0
            continue
        edge_log_probs = []
        for edge_idx in node.incoming_edges:
            edge = forest.hyperedges[edge_idx]
            log_prob = edge_score(edge, node)
            for child_idx in edge.child_nodes:
                log_prob = log_prob + values[child_idx]
            edge_log_probs.append(log_prob)
        values[node_idx] = torch.logsumexp(torch.stack(edge_log_probs), dim=0)
    return values, _root_log_prob(forest, values, device)


def _outside(forest, edge_score, inside_values, device):
    values = torch.full((len(forest.nodes),), float('-inf'), device=device)
    for root_idx in forest.roots:
        values[root_idx] = 0.0
    for node_idx in reversed(forest.topo_order):
        for edge_idx in forest.nodes[node_idx].incoming_edges:
            edge = forest.hyperedges[edge_idx]
            parent_idx = edge.parent_node
            log_rule_prob = edge_score(edge, forest.nodes[parent_idx])
            for i, child_idx in enumerate(edge.child_nodes):
                contrib = values[parent_idx] + log_rule_prob
                for j, sibling_idx in enumerate(edge.child_nodes):
                    if i != j:
                        contrib = contrib + inside_values[sibling_idx]
                values[child_idx] = torch.logaddexp(values[child_idx], contrib)
    return values


def _viterbi(forest, edge_score, device):
    num_nodes = len(forest.nodes)
    best_values = torch.full((num_nodes,), float('-inf'), device=device)
    backpointers = [-1] * num_nodes
    for node_idx in forest.topo_order:
        node = forest.nodes[node_idx]
        if not node.incoming_edges:
            best_values[node_idx] = 0.0
            continue
        best_edge_idx = -1
        best_log_prob = float('-inf')
        for edge_idx in node.incoming_edges:
            edge = forest.hyperedges[edge_idx]
            log_prob = edge_score(edge, node).item()
            for child_idx in edge.child_nodes:
                log_prob += best_values[child_idx].item()
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_edge_idx = edge_idx
        best_values[node_idx] = best_log_prob
        backpointers[node_idx] = best_edge_idx

    if not forest.roots:
        return torch.tensor(float('-inf'), device=device), []
    best_root = max(forest.roots, key=lambda r: best_values[r].item())

    derivation = []

    def extract(node_idx):
        edge_idx = backpointers[node_idx]
        if edge_idx < 0:
            return
        derivation.append((node_idx, edge_idx))
        for child_idx in forest.hyperedges[edge_idx].child_nodes:
            extract(child_idx)

    extract(best_root)
    return best_values[best_root], derivation


def inside(forest: PackedForest, rule_log_probs: Tensor) -> tuple[Tensor, Tensor]:
    """Returns (log inside per node, log P(graph))."""
    _check_rule_ids(forest, rule_log_probs)
    return _inside(forest, lambda e, _: rule_log_probs[e.rule_id], rule_log_probs.device)


def inside_with_entropy(forest: PackedForest, rule_log_probs: Tensor) -> tuple[Tensor, Tensor]:
    """Returns (log P(graph), H(derivation | graph) in nats), both differentiable.

    Accumulated bottom-up in the entropy semiring, because outside() cannot be backpropagated
    through (in-place logaddexp writes) and log P - sum E[c_r] log p_r has zero gradient
    with the counts detached. Per node: H_n = -sum_e q_e (w_e - Z_n) + sum_e q_e sum_c H_c.
    """
    device = rule_log_probs.device
    if not forest.roots:
        return (torch.tensor(float("-inf"), device=device),
                torch.tensor(0.0, device=device))

    node_z = {}
    node_h = {}
    for node_idx in forest.topo_order or range(len(forest.nodes)):
        node = forest.nodes[node_idx]
        if not node.incoming_edges:
            node_z[node_idx] = torch.tensor(0.0, device=device)
            node_h[node_idx] = torch.tensor(0.0, device=device)
            continue

        weights = []
        child_entropy = []
        for edge_idx in node.incoming_edges:
            edge = forest.hyperedges[edge_idx]
            if edge.rule_id < 0 or edge.rule_id >= len(rule_log_probs):
                continue
            weight = rule_log_probs[edge.rule_id]
            below = torch.tensor(0.0, device=device)
            for child_idx in edge.child_nodes:
                weight = weight + node_z[child_idx]
                below = below + node_h[child_idx]
            weights.append(weight)
            child_entropy.append(below)

        if not weights:
            node_z[node_idx] = torch.tensor(float("-inf"), device=device)
            node_h[node_idx] = torch.tensor(0.0, device=device)
            continue

        stacked = torch.stack(weights)
        total = torch.logsumexp(stacked, dim=0)
        posterior = torch.softmax(stacked, dim=0)
        local = -(posterior * (stacked - total)).sum()
        inherited = (posterior * torch.stack(child_entropy)).sum()
        node_z[node_idx] = total
        node_h[node_idx] = local + inherited

    root_weights = torch.stack([node_z[r] for r in forest.roots])
    total_log_prob = torch.logsumexp(root_weights, dim=0)
    root_posterior = torch.softmax(root_weights, dim=0)
    root_local = -(root_posterior * (root_weights - total_log_prob)).sum()
    root_inherited = (root_posterior * torch.stack([node_h[r] for r in forest.roots])).sum()
    return total_log_prob, root_local + root_inherited


def derivation_entropy(forest: PackedForest, rule_log_probs: Tensor) -> Tensor:
    return inside_with_entropy(forest, rule_log_probs)[1]


def outside(forest: PackedForest, rule_log_probs: Tensor, inside_values: Tensor) -> Tensor:
    _check_rule_ids(forest, rule_log_probs)
    return _outside(forest, lambda e, _: rule_log_probs[e.rule_id], inside_values, rule_log_probs.device)


def expected_counts(
    forest: PackedForest,
    rule_log_probs: Tensor,
    inside_values: Tensor | None = None,
    outside_values: Tensor | None = None,
) -> Tensor:
    device = rule_log_probs.device
    num_rules = len(rule_log_probs)

    if inside_values is None:
        inside_values, total_log_prob = inside(forest, rule_log_probs)
    else:
        total_log_prob = _root_log_prob(forest, inside_values, device)
    if outside_values is None:
        outside_values = outside(forest, rule_log_probs, inside_values)

    # Keyed on the rules this forest uses, so the cost does not grow with the grammar size.
    contributions = defaultdict(list)
    for edge in forest.hyperedges:
        if edge.rule_id < 0 or edge.rule_id >= num_rules:
            continue
        log_contrib = outside_values[edge.parent_node] + rule_log_probs[edge.rule_id]
        for child_idx in edge.child_nodes:
            log_contrib = log_contrib + inside_values[child_idx]
        contributions[edge.rule_id].append(log_contrib - total_log_prob)

    counts = torch.zeros(num_rules, device=device)
    for rule_id, contribs in contributions.items():
        counts[rule_id] = torch.logsumexp(torch.stack(contribs), dim=0).exp()
    return counts


def viterbi(forest: PackedForest, rule_log_probs: Tensor) -> tuple[Tensor, list]:
    """Returns (best log prob, [(node_idx, edge_idx), ...] in preorder)."""
    _check_rule_ids(forest, rule_log_probs)
    return _viterbi(forest, lambda e, _: rule_log_probs[e.rule_id], rule_log_probs.device)


def sample(
    forest: PackedForest,
    rule_log_probs: Tensor,
    inside_values: Tensor | None = None,
    num_samples: int = 1,
) -> list:
    """Ancestral samples, each a list of (node_idx, edge_idx)."""
    device = rule_log_probs.device
    if inside_values is None:
        inside_values, _ = inside(forest, rule_log_probs)

    def sample_subtree(node_idx, derivation):
        node = forest.nodes[node_idx]
        if not node.incoming_edges:
            return
        edge_log_probs = []
        for edge_idx in node.incoming_edges:
            edge = forest.hyperedges[edge_idx]
            log_prob = rule_log_probs[edge.rule_id]
            for child_idx in edge.child_nodes:
                log_prob = log_prob + inside_values[child_idx]
            edge_log_probs.append(log_prob)
        probs = torch.softmax(torch.stack(edge_log_probs), dim=0)
        chosen = node.incoming_edges[torch.multinomial(probs, 1).item()]
        derivation.append((node_idx, chosen))
        for child_idx in forest.hyperedges[chosen].child_nodes:
            sample_subtree(child_idx, derivation)

    samples = []
    for _ in range(num_samples):
        derivation = []
        if forest.roots:
            root_values = inside_values[torch.tensor(forest.roots, device=device)]
            root_choice = torch.multinomial(torch.softmax(root_values, dim=0), 1).item()
            sample_subtree(forest.roots[root_choice], derivation)
        samples.append(derivation)
    return samples


def inside_with_scorer(forest: PackedForest, rule_scorer, node_embeddings: dict | None = None):
    """inside() with rule_scorer(edge, parent_node, child_nodes, node_embeddings) -> log prob."""
    device = _scorer_device(node_embeddings)
    if node_embeddings is None:
        node_embeddings = {}
    return _inside(forest, _scorer_edge_score(forest, rule_scorer, node_embeddings), device)


def outside_with_scorer(forest: PackedForest, rule_scorer, inside_values: Tensor,
                        node_embeddings: dict | None = None) -> Tensor:
    if node_embeddings is None:
        node_embeddings = {}
    return _outside(forest, _scorer_edge_score(forest, rule_scorer, node_embeddings),
                    inside_values, inside_values.device)


def viterbi_with_scorer(forest: PackedForest, rule_scorer, node_embeddings: dict | None = None):
    if node_embeddings is None:
        node_embeddings = {}
    return _viterbi(forest, _scorer_edge_score(forest, rule_scorer, node_embeddings),
                    _scorer_device(node_embeddings))

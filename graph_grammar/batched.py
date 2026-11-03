"""Inside over a whole corpus at once: forests concatenated, swept level by level.

Agrees with inference.inside() (leaves are log 1, rootless forests -inf) and, unlike
inference.outside(), can be backpropagated through, so d logZ / d log p gives expected counts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .forest import PackedForest

NEG_INF = float("-inf")


def _segment_logsumexp(values, segments, size, device, dtype):
    """logsumexp of `values` grouped by `segments`; returns (result, touched segments)."""
    # The shift is detached: exact for logsumexp and keeps the max out of autograd.
    shift = torch.full((size,), NEG_INF, device=device, dtype=dtype)
    shift = shift.scatter_reduce(0, segments, values.detach(), reduce="amax", include_self=False)
    per_value = shift[segments]
    # An all -inf segment would give -inf - -inf = nan; shift by 0 there instead.
    safe = torch.where(torch.isfinite(per_value), per_value, torch.zeros_like(per_value))
    acc = torch.zeros((size,), device=device, dtype=dtype)
    acc = acc.index_add(0, segments, torch.exp(values - safe))
    touched = torch.unique(segments)
    return torch.log(acc[touched]) + shift[touched], touched


class BatchedForest:
    def __init__(self, edge_parent, edge_rule, edge_children, level_slices, leaf_nodes,
                 root_index, root_forest, num_nodes, num_forests, graph_ids):
        self.edge_parent = edge_parent
        self.edge_rule = edge_rule
        self.edge_children = edge_children      # [E, 2], self.dummy where absent
        self.level_slices = level_slices        # contiguous edge ranges, one per depth
        self.leaf_nodes = leaf_nodes
        self.root_index = root_index
        self.root_forest = root_forest          # which forest each root belongs to
        self.num_nodes = num_nodes
        self.num_forests = num_forests
        self.graph_ids = graph_ids
        self.dummy = num_nodes                  # extra slot with inside 0.0

    @classmethod
    def from_forests(cls, forests: Sequence[PackedForest]) -> BatchedForest:
        edge_parent, edge_rule, edge_c0, edge_c1, edge_depth = [], [], [], [], []
        leaf_nodes, root_index, root_forest, graph_ids = [], [], [], []

        offset = 0
        for fi, forest in enumerate(forests):
            graph_ids.append(forest.graph_id)
            n = len(forest.nodes)

            depth = [0] * n
            for node_idx in forest.topo_order or range(n):
                node = forest.nodes[node_idx]
                if not node.incoming_edges:
                    leaf_nodes.append(offset + node_idx)
                    continue
                best = 0
                for edge_idx in node.incoming_edges:
                    d = 1
                    for child in forest.hyperedges[edge_idx].child_nodes:
                        d = max(d, depth[child] + 1)
                    best = max(best, d)
                depth[node_idx] = best

            for edge in forest.hyperedges:
                kids = edge.child_nodes
                if len(kids) > 2:
                    raise ValueError(
                        f"forest {forest.graph_id!r} has a hyperedge with {len(kids)} "
                        f"children; the batched inside assumes at most 2 (the "
                        f"extracted grammars are binary: 0 or 2 nonterminals)")
                edge_parent.append(offset + edge.parent_node)
                edge_rule.append(edge.rule_id)
                edge_c0.append(offset + kids[0] if len(kids) > 0 else -1)
                edge_c1.append(offset + kids[1] if len(kids) > 1 else -1)
                edge_depth.append(depth[edge.parent_node])

            for r in forest.roots:
                root_index.append(offset + r)
                root_forest.append(fi)
            offset += n

        dummy = offset
        c0 = torch.tensor([x if x >= 0 else dummy for x in edge_c0], dtype=torch.long)
        c1 = torch.tensor([x if x >= 0 else dummy for x in edge_c1], dtype=torch.long)

        depth_t = torch.tensor(edge_depth, dtype=torch.long)
        order = torch.argsort(depth_t, stable=True)
        sorted_depth = depth_t[order]

        level_slices = []
        if len(sorted_depth):
            bounds = (torch.nonzero(sorted_depth[1:] != sorted_depth[:-1]).flatten() + 1).tolist()
            level_slices = [slice(s, e) for s, e in zip([0] + bounds, bounds + [len(sorted_depth)])]

        return cls(
            edge_parent=torch.tensor(edge_parent, dtype=torch.long)[order],
            edge_rule=torch.tensor(edge_rule, dtype=torch.long)[order],
            edge_children=torch.stack([c0[order], c1[order]], dim=1),
            level_slices=level_slices,
            leaf_nodes=torch.tensor(leaf_nodes, dtype=torch.long),
            root_index=torch.tensor(root_index, dtype=torch.long),
            root_forest=torch.tensor(root_forest, dtype=torch.long),
            num_nodes=offset,
            num_forests=len(forests),
            graph_ids=graph_ids,
        )

    def to(self, device) -> BatchedForest:
        for name in ("edge_parent", "edge_rule", "edge_children", "leaf_nodes",
                     "root_index", "root_forest"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @property
    def num_levels(self) -> int:
        return len(self.level_slices)

    def log_partition(self, rule_log_probs: Tensor, edge_bias: Tensor | None = None) -> Tensor:
        """Per-forest log Z. `edge_bias` (one entry per hyperedge) is for edge_posteriors."""
        if self.edge_rule.numel():
            hi = int(self.edge_rule.max())
            if hi >= len(rule_log_probs) or int(self.edge_rule.min()) < 0:
                raise IndexError(
                    f"batched forest references rule ids up to {hi} but "
                    f"rule_log_probs has length {len(rule_log_probs)}")

        device = rule_log_probs.device
        dtype = rule_log_probs.dtype
        size = self.num_nodes + 1

        inside = torch.full((size,), NEG_INF, device=device, dtype=dtype)
        zero_slots = torch.cat([
            self.leaf_nodes.to(device),
            torch.tensor([self.dummy], device=device, dtype=torch.long)])
        inside = inside.index_fill(0, zero_slots, 0.0)

        for sl in self.level_slices:
            kids = self.edge_children[sl]
            w = rule_log_probs[self.edge_rule[sl]] + inside[kids[:, 0]] + inside[kids[:, 1]]
            if edge_bias is not None:
                w = w + edge_bias[sl]
            vals, touched = _segment_logsumexp(w, self.edge_parent[sl], size, device, dtype)
            inside = inside.index_copy(0, touched, vals)

        out = torch.full((self.num_forests,), NEG_INF, device=device, dtype=dtype)
        if self.root_index.numel():
            vals, touched = _segment_logsumexp(
                inside[self.root_index], self.root_forest, self.num_forests, device, dtype)
            out = out.index_copy(0, touched, vals)
        return out

    def total_log_likelihood(self, rule_log_probs: Tensor, edge_bias: Tensor | None = None) -> Tensor:
        """Skips forests with no derivation."""
        z = self.log_partition(rule_log_probs, edge_bias=edge_bias)
        return z[torch.isfinite(z)].sum()

    def edge_posteriors(self, rule_log_probs: Tensor) -> Tensor:
        """P(hyperedge used), in this object's (depth-sorted) edge order."""
        n_edges = int(self.edge_rule.numel())
        if n_edges == 0:
            return torch.zeros(0, dtype=rule_log_probs.dtype)
        bias = torch.zeros(n_edges, dtype=rule_log_probs.dtype,
                           device=rule_log_probs.device, requires_grad=True)
        self.total_log_likelihood(rule_log_probs.detach(), edge_bias=bias).backward()
        return bias.grad.detach()

    def node_posteriors(self, rule_log_probs: Tensor) -> Tensor:
        """P(forest node used); a node is used at most once, so it sums its incoming edges."""
        ep = self.edge_posteriors(rule_log_probs)
        out = torch.zeros(self.num_nodes + 1, dtype=ep.dtype, device=ep.device)
        out = out.index_add(0, self.edge_parent.to(ep.device), ep)
        return out[: self.num_nodes]

    def expected_counts(self, rule_log_probs: Tensor, num_rules: int) -> Tensor:
        x = rule_log_probs.detach().clone().requires_grad_(True)
        self.total_log_likelihood(x).backward()
        return x.grad.detach()

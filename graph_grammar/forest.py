from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class ForestNode:
    id: int
    edge_set_hash: int
    boundary_mapping: tuple[int, ...]
    incoming_edges: list[int] = field(default_factory=list)
    is_terminal: bool = False
    is_root: bool = False
    edge_indices: tuple[int, ...] | None = None


@dataclass
class HyperEdge:
    id: int
    rule_id: int
    parent_node: int
    child_nodes: tuple[int, ...] = ()
    cfg_rule_id: int = 0
    log_prob: float = 0.0


@dataclass
class PackedForest:
    nodes: list[ForestNode]
    hyperedges: list[HyperEdge]
    roots: list[int]
    num_rules: int
    topo_order: list[int] = field(default_factory=list)
    graph_id: Any = None

    def __post_init__(self):
        if not self.topo_order:
            self.topo_order = self._compute_topo_order()

    def _compute_topo_order(self) -> list[int]:
        """Children before parents."""
        n = len(self.nodes)
        children_of = [set() for _ in range(n)]
        parents_of = [set() for _ in range(n)]
        for edge in self.hyperedges:
            for child in edge.child_nodes:
                children_of[edge.parent_node].add(child)
                parents_of[child].add(edge.parent_node)

        pending = [len(c) for c in children_of]
        queue = deque(i for i in range(n) if pending[i] == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for parent in parents_of[node]:
                pending[parent] -= 1
                if pending[parent] == 0:
                    queue.append(parent)

        if len(order) < n:
            done = set(order)
            order.extend(i for i in range(n) if i not in done)
        return order

    def to_tensors(self, device: str = 'cpu') -> dict[str, Tensor]:
        num_edges = len(self.hyperedges)
        num_nodes = len(self.nodes)
        max_children = max((len(e.child_nodes) for e in self.hyperedges), default=0)
        max_incoming = max((len(n.incoming_edges) for n in self.nodes), default=0)

        def zeros(*shape):
            return torch.zeros(*shape, dtype=torch.long, device=device)

        rule_ids = zeros(num_edges)
        parent_nodes = zeros(num_edges)
        child_nodes = zeros(num_edges, max(max_children, 1))
        num_children = zeros(num_edges)
        for i, edge in enumerate(self.hyperedges):
            rule_ids[i] = edge.rule_id
            parent_nodes[i] = edge.parent_node
            num_children[i] = len(edge.child_nodes)
            for j, child in enumerate(edge.child_nodes):
                child_nodes[i, j] = child

        node_to_edges = zeros(num_nodes, max(max_incoming, 1))
        num_incoming = zeros(num_nodes)
        for i, node in enumerate(self.nodes):
            num_incoming[i] = len(node.incoming_edges)
            for j, edge_id in enumerate(node.incoming_edges):
                node_to_edges[i, j] = edge_id

        return {
            'rule_ids': rule_ids,
            'parent_nodes': parent_nodes,
            'child_nodes': child_nodes,
            'num_children': num_children,
            'node_to_edges': node_to_edges,
            'num_incoming': num_incoming,
            'topo_order': torch.tensor(self.topo_order, dtype=torch.long, device=device),
            'roots': torch.tensor(self.roots, dtype=torch.long, device=device),
        }

    def __len__(self) -> int:
        return len(self.nodes)

    def num_derivations(self) -> int:
        count = {}
        for node_idx in self.topo_order:
            node = self.nodes[node_idx]
            if not node.incoming_edges:
                count[node_idx] = 1
                continue
            total = 0
            for edge_idx in node.incoming_edges:
                prod = 1
                for child in self.hyperedges[edge_idx].child_nodes:
                    prod *= count.get(child, 1)
                total += prod
            count[node_idx] = total
        return sum(count.get(r, 0) for r in self.roots)

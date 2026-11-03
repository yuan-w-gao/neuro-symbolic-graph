import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def load_grammar_rules(grammar_file, synchronous=False):
    """(manager, rules): the SHRG rules if synchronous, else the HRG rules (forest index space)."""
    import pyshrg

    pyshrg.initialize()
    manager = pyshrg.get_manager()

    def size():
        return manager.shrg_size if synchronous else manager.hrg_size

    # Re-loading the singleton corrupts contexts/forests the caller still holds (segfault in BLEU eval).
    if size() == 0:
        manager.load_grammars(grammar_file)
    get = manager.get_shrg if synchronous else manager.get_hrg
    return manager, [get(i) for i in range(size())]


def is_empty_rule(rule):
    return rule is None or rule.is_empty


def edges_of(rule):
    return [*rule.terminal_edges, *rule.nonterminal_edges]


def label_vocab(labels):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for label in sorted(set(labels)):
        vocab.setdefault(label, len(vocab))
    return vocab


def edge_label_vocab(rules):
    return label_vocab(edge.label for rule in rules if not is_empty_rule(rule)
                       for edge in edges_of(rule))


def pad_rows(rows):
    """Right-pad integer rows with zeros to (len(rows), max(1, longest)); returns (padded, mask)."""
    width = max(max(len(row) for row in rows), 1)
    padded = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for i, row in enumerate(rows):
        if row:
            padded[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = True
    return padded, mask


class CachedRuleEmbedding(nn.Module):
    """forward() computes all rule embeddings once and reuses them until invalidate_cache()."""

    _cached_embeddings = None

    def forward(self) -> Tensor:
        if self._cached_embeddings is None:
            self._cached_embeddings = self._compute_embeddings()
        return self._cached_embeddings

    @property
    def _cache_valid(self):
        return self._cached_embeddings is not None

    def invalidate_cache(self):
        self._cached_embeddings = None

    @property
    def num_rules(self) -> int:
        return self.extractor.num_rules

    def get_embedding(self, rule_ids: Tensor) -> Tensor:
        return self.forward()[rule_ids]


@dataclass
class HypergraphData:
    rule_idx: int
    num_nodes: int
    external_node_indices: list[int]  # -1 for an external node missing from rule.nodes
    node_is_external: list[bool]
    node_external_position: list[int]  # -1 if internal
    node_degree: list[int]
    node_terminal_degree: list[int]
    node_nonterminal_degree: list[int]
    num_edges: int
    edge_labels: list[str]
    edge_label_indices: list[int]
    edge_is_terminal: list[bool]
    edge_arities: list[int]
    edge_to_nodes: list[list[int]]
    node_to_edges: list[list[int]]

    @property
    def is_empty(self) -> bool:
        return self.num_nodes == 0 and self.num_edges == 0


def _empty_hypergraph(rule_idx):
    return HypergraphData(rule_idx, 0, [], [], [], [], [], [], 0, [], [], [], [], [], [])


def _hypergraph(rule_idx, rule, vocab):
    nodes = list(rule.nodes)
    num_nodes = len(nodes)
    position = {node.index: i for i, node in enumerate(nodes)}
    external = [position.get(node.index, -1) for node in rule.external_nodes]

    is_external = [False] * num_nodes
    external_position = [-1] * num_nodes
    for pos, i in enumerate(external):
        if i >= 0:
            is_external[i] = True
            external_position[i] = pos

    degree = [0] * num_nodes
    degree_by_kind = {True: [0] * num_nodes, False: [0] * num_nodes}
    labels, is_terminal, edge_to_nodes = [], [], []
    node_to_edges = [[] for _ in range(num_nodes)]
    for terminal, edges in ((True, rule.terminal_edges), (False, rule.nonterminal_edges)):
        for edge in edges:
            linked = [position.get(node.index, -1) for node in edge.linked_nodes]
            linked = [i for i in linked if i >= 0]
            for i in linked:
                degree[i] += 1
                degree_by_kind[terminal][i] += 1
                node_to_edges[i].append(len(labels))
            labels.append(edge.label)
            is_terminal.append(terminal)
            edge_to_nodes.append(linked)

    return HypergraphData(
        rule_idx=rule_idx,
        num_nodes=num_nodes,
        external_node_indices=external,
        node_is_external=is_external,
        node_external_position=external_position,
        node_degree=degree,
        node_terminal_degree=degree_by_kind[True],
        node_nonterminal_degree=degree_by_kind[False],
        num_edges=len(labels),
        edge_labels=labels,
        edge_label_indices=[vocab[label] for label in labels],
        edge_is_terminal=is_terminal,
        edge_arities=[len(linked) for linked in edge_to_nodes],
        edge_to_nodes=edge_to_nodes,
        node_to_edges=node_to_edges,
    )


class HypergraphStructureExtractor:
    def __init__(self, grammar_file: str):
        self.manager, rules = load_grammar_rules(grammar_file)
        self.edge_label_vocab = edge_label_vocab(rules)
        self.structures = [
            _empty_hypergraph(i) if is_empty_rule(rule) else _hypergraph(i, rule, self.edge_label_vocab)
            for i, rule in enumerate(rules)]

    @property
    def num_rules(self) -> int:
        return len(self.structures)

    @property
    def edge_vocab_size(self) -> int:
        return len(self.edge_label_vocab)

    @property
    def max_external_nodes(self) -> int:
        return max((len(s.external_node_indices) for s in self.structures), default=0)


class HypergraphConvLayer(nn.Module):
    """Node -> edge attention, then edge -> node attention over the updated edges."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_to_edge_key = nn.Linear(node_dim, hidden_dim)
        self.node_to_edge_query = nn.Linear(edge_dim, hidden_dim)
        self.node_to_edge_value = nn.Linear(node_dim, hidden_dim)
        self.edge_update = nn.Linear(hidden_dim + edge_dim, hidden_dim)
        self.edge_to_node_key = nn.Linear(hidden_dim, hidden_dim)
        self.edge_to_node_query = nn.Linear(node_dim, hidden_dim)
        self.edge_to_node_value = nn.Linear(hidden_dim, hidden_dim)
        self.node_update = nn.Linear(hidden_dim + node_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def _attend(self, queries, keys, values, neighbours):
        out = torch.zeros(queries.shape[0], self.hidden_dim, device=queries.device)
        for i, members in enumerate(neighbours):
            if not members:
                continue
            members = torch.tensor(members, dtype=torch.long, device=queries.device)
            scores = torch.matmul(queries[i:i + 1], keys[members].t()) * self.scale
            attn = self.dropout(F.softmax(scores, dim=-1))
            out[i] = torch.matmul(attn, values[members]).squeeze(0)
        return out

    def forward(self, node_emb: Tensor, edge_emb: Tensor, edge_to_nodes: list[list[int]],
                node_to_edges: list[list[int]]) -> tuple[Tensor, Tensor]:
        num_nodes, num_edges = node_emb.shape[0], edge_emb.shape[0]
        if num_nodes == 0 or num_edges == 0:
            return (torch.zeros(num_nodes, self.hidden_dim, device=node_emb.device),
                    torch.zeros(num_edges, self.hidden_dim, device=node_emb.device))

        edge_agg = self._attend(self.node_to_edge_query(edge_emb), self.node_to_edge_key(node_emb),
                                self.node_to_edge_value(node_emb), edge_to_nodes)
        new_edge_emb = F.relu(self.edge_norm(self.edge_update(torch.cat([edge_agg, edge_emb], dim=-1))))

        node_agg = self._attend(self.edge_to_node_query(node_emb),
                                self.edge_to_node_key(new_edge_emb),
                                self.edge_to_node_value(new_edge_emb), node_to_edges)
        new_node_emb = F.relu(self.node_norm(self.node_update(torch.cat([node_agg, node_emb], dim=-1))))
        return new_node_emb, new_edge_emb


class HypergraphGNN(nn.Module):
    def __init__(self, node_input_dim: int, edge_input_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.node_input_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_input_proj = nn.Linear(edge_input_dim, hidden_dim)
        self.layers = nn.ModuleList([HypergraphConvLayer(hidden_dim, hidden_dim, hidden_dim, dropout)
                                     for _ in range(num_layers)])

    def forward(self, node_features: Tensor, edge_features: Tensor,
                edge_to_nodes: list[list[int]], node_to_edges: list[list[int]]) -> tuple[Tensor, Tensor]:
        node_emb = F.relu(self.node_input_proj(node_features))
        edge_emb = F.relu(self.edge_input_proj(edge_features))
        for layer in self.layers:
            node_emb, edge_emb = layer(node_emb, edge_emb, edge_to_nodes, node_to_edges)
        return node_emb, edge_emb


class BoundaryNodeEncoder(nn.Module):
    """Self-attention over the external nodes plus their position, mean-pooled."""

    def __init__(self, node_dim: int, max_external_nodes: int = 10, num_heads: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.node_dim = node_dim
        self.max_external_nodes = max_external_nodes
        self.position_embedding = nn.Embedding(max_external_nodes, node_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=node_dim, num_heads=num_heads,
                                                    dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(node_dim)
        self.output_proj = nn.Linear(node_dim, node_dim)

    def forward(self, node_emb: Tensor, external_node_indices: list[int]) -> Tensor:
        device = node_emb.device
        valid = [i for i in external_node_indices if 0 <= i < node_emb.shape[0]]
        if not valid:
            return torch.zeros(self.node_dim, device=device)
        valid = valid[:self.max_external_nodes]
        ext_emb = node_emb[torch.tensor(valid, dtype=torch.long, device=device)]
        ext_emb = ext_emb + self.position_embedding(
            torch.arange(len(valid), dtype=torch.long, device=device))
        batch = ext_emb.unsqueeze(0)
        attn_out, _ = self.self_attention(batch, batch, batch)
        pooled = self.layer_norm(ext_emb + attn_out.squeeze(0)).mean(dim=0)
        return self.output_proj(pooled)


def node_feature_rows(data: HypergraphData) -> list[list[float]]:
    """Per node: is_external, external position one-hot (5), degrees / 10 (3), role one-hot (3)."""
    rows = []
    for i in range(data.num_nodes):
        position = [0.0] * 5
        if 0 <= data.node_external_position[i] < 5:
            position[data.node_external_position[i]] = 1.0
        has_terminal = data.node_terminal_degree[i] > 0
        has_nonterminal = data.node_nonterminal_degree[i] > 0
        if has_terminal and has_nonterminal:
            role = [0.0, 0.0, 1.0]
        elif has_terminal:
            role = [1.0, 0.0, 0.0]
        else:
            role = [0.0, 1.0, 0.0]
        rows.append([1.0 if data.node_is_external[i] else 0.0] + position +
                    [data.node_degree[i] / 10.0, data.node_terminal_degree[i] / 10.0,
                     data.node_nonterminal_degree[i] / 10.0] + role)
    return rows


class GraphStructuralRuleEmbedding(CachedRuleEmbedding):
    """Per-rule hypergraph GNN; mean node pool, mean edge pool and boundary encoding, projected."""

    NODE_FEATURE_DIM = 12

    def __init__(
        self,
        grammar_file: str,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        edge_embed_dim: int = 32,
        dropout: float = 0.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.extractor = HypergraphStructureExtractor(grammar_file)
        print(f"Hypergraph structures: {self.extractor.num_rules} rules, "
              f"{self.extractor.edge_vocab_size} edge labels, "
              f"at most {self.extractor.max_external_nodes} external nodes")

        self.edge_label_embedding = nn.Embedding(self.extractor.edge_vocab_size, edge_embed_dim,
                                                 padding_idx=0)
        self.edge_feature_dim = edge_embed_dim + 2  # + is_terminal, arity / 5
        self.gnn = HypergraphGNN(self.NODE_FEATURE_DIM, self.edge_feature_dim, hidden_dim,
                                 num_gnn_layers, dropout)
        self.boundary_encoder = BoundaryNodeEncoder(
            hidden_dim, max(10, self.extractor.max_external_nodes), num_heads=2, dropout=dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.to(self.device)

    def _build_node_features(self, data):
        if data.num_nodes == 0:
            return torch.zeros(0, self.NODE_FEATURE_DIM, device=self.device)
        return torch.tensor(node_feature_rows(data), dtype=torch.float32, device=self.device)

    def _build_edge_features(self, data):
        if data.num_edges == 0:
            return torch.zeros(0, self.edge_feature_dim, device=self.device)
        labels = self.edge_label_embedding(
            torch.tensor(data.edge_label_indices, dtype=torch.long, device=self.device))
        is_terminal = torch.tensor([1.0 if t else 0.0 for t in data.edge_is_terminal],
                                   dtype=torch.float32, device=self.device).unsqueeze(-1)
        arity = torch.tensor([a / 5.0 for a in data.edge_arities],
                             dtype=torch.float32, device=self.device).unsqueeze(-1)
        return torch.cat([labels, is_terminal, arity], dim=-1)

    def _mean_or_zeros(self, emb):
        if emb.shape[0] > 0:
            return emb.mean(dim=0)
        return torch.zeros(self.hidden_dim, device=self.device)

    def _encode_single_rule(self, data):
        if data.is_empty:
            return torch.zeros(self.embedding_dim, device=self.device)
        node_emb, edge_emb = self.gnn(self._build_node_features(data),
                                      self._build_edge_features(data),
                                      data.edge_to_nodes, data.node_to_edges)
        node_pool = self._mean_or_zeros(node_emb)
        edge_pool = self._mean_or_zeros(edge_emb)
        boundary = self.boundary_encoder(node_emb, data.external_node_indices)
        return self.output_proj(torch.cat([node_pool, edge_pool, boundary], dim=0))

    def _compute_embeddings(self) -> Tensor:
        return torch.stack([self._encode_single_rule(d) for d in self.extractor.structures], dim=0)

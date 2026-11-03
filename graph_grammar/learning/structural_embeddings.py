from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .hypergraph_embeddings import edges_of, is_empty_rule, label_vocab, load_grammar_rules, pad_rows
from .serialized_embeddings import lstm_final_state, masked_sum


@dataclass
class RuleStructure:
    rule_idx: int
    lhs_label: str
    lhs_label_idx: int
    rhs_labels: list[str]
    rhs_label_indices: list[int]
    terminal_edge_labels: list[str]
    terminal_edge_label_indices: list[int]
    nonterminal_edge_labels: list[str]
    nonterminal_edge_label_indices: list[int]
    num_nodes: int
    num_external_nodes: int
    edge_to_nodes: list[tuple[int, ...]]


def _first_cfg_labels(rule):
    cfg = next(iter(rule.iter_cfgs()), None)
    if cfg is None:
        return []
    return [cfg.get(i).label for i in range(cfg.size)]


class RuleStructureExtractor:
    """Indexed by SHRG rule, unlike the HRG-indexed extractors."""

    def __init__(self, grammar_file: str):
        self.manager, rules = load_grammar_rules(grammar_file, synchronous=True)
        rules = [None if is_empty_rule(rule) else rule for rule in rules]
        self.rhs_label_vocab = label_vocab(label for rule in rules if rule is not None
                                           for label in _first_cfg_labels(rule))
        self.edge_label_vocab = label_vocab(edge.label for rule in rules if rule is not None
                                            for edge in edges_of(rule))
        self.structures = [self._structure(i, rule) for i, rule in enumerate(rules)]

    def _structure(self, rule_idx, rule):
        if rule is None:
            return RuleStructure(rule_idx, '<EMPTY>', 0, [], [], [], [], [], [], 0, 0, [])
        rhs = _first_cfg_labels(rule)
        terminal = [edge.label for edge in rule.terminal_edges]
        nonterminal = [edge.label for edge in rule.nonterminal_edges]
        return RuleStructure(
            rule_idx=rule_idx,
            lhs_label=rule.label,
            lhs_label_idx=rule.label_index,
            rhs_labels=rhs,
            rhs_label_indices=[self.rhs_label_vocab[label] for label in rhs],
            terminal_edge_labels=terminal,
            terminal_edge_label_indices=[self.edge_label_vocab[label] for label in terminal],
            nonterminal_edge_labels=nonterminal,
            nonterminal_edge_label_indices=[self.edge_label_vocab[label] for label in nonterminal],
            num_nodes=len(list(rule.nodes)),
            num_external_nodes=len(list(rule.external_nodes)),
            edge_to_nodes=[tuple(node.index for node in edge.linked_nodes) for edge in edges_of(rule)],
        )

    @property
    def num_rules(self) -> int:
        return len(self.structures)

    @property
    def rhs_vocab_size(self) -> int:
        return len(self.rhs_label_vocab)

    @property
    def edge_vocab_size(self) -> int:
        return len(self.edge_label_vocab)


class RHSEncoder(nn.Module):
    """LSTM over the first CFG right-hand side of each rule."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64,
                 num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, sequences: Tensor, lengths: Tensor) -> Tensor:
        return lstm_final_state(self.lstm, self.embedding(sequences), lengths)


class RuleGraphEncoder(nn.Module):
    """MLP over mean terminal / nonterminal edge-label embeddings and scaled size counts."""

    def __init__(self, edge_vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64,
                 num_layers: int = 1):
        super().__init__()
        self.terminal_edge_embedding = nn.Embedding(edge_vocab_size, embed_dim, padding_idx=0)
        self.nonterminal_edge_embedding = nn.Embedding(edge_vocab_size, embed_dim, padding_idx=0)
        layers = []
        in_dim = embed_dim * 2 + 4
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, terminal_edges: Tensor, terminal_mask: Tensor, nonterminal_edges: Tensor,
                nonterminal_mask: Tensor, num_nodes: Tensor, num_external: Tensor) -> Tensor:
        terminal_sum = masked_sum(self.terminal_edge_embedding(terminal_edges), terminal_mask)
        terminal_count = terminal_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        nonterminal_sum = masked_sum(self.nonterminal_edge_embedding(nonterminal_edges),
                                     nonterminal_mask)
        nonterminal_count = nonterminal_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return self.mlp(torch.cat([
            terminal_sum / terminal_count,
            nonterminal_sum / nonterminal_count,
            terminal_count / 10.0,
            nonterminal_count / 10.0,
            num_nodes.unsqueeze(-1).float() / 10.0,
            num_external.unsqueeze(-1).float() / 5.0,
        ], dim=-1))


class StructuralRuleEmbedding(nn.Module):
    """RHS LSTM and edge-label graph encoder, combined by an MLP. Selected as the "legacy" encoder."""

    def __init__(
        self,
        grammar_file: str,
        embedding_dim: int = 64,
        rhs_embed_dim: int = 32,
        rhs_hidden_dim: int = 64,
        graph_embed_dim: int = 32,
        graph_hidden_dim: int = 64,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.embedding_dim = embedding_dim
        self.extractor = RuleStructureExtractor(grammar_file)
        print(f"Rule structures: {self.extractor.num_rules} rules, "
              f"{self.extractor.rhs_vocab_size} RHS labels, "
              f"{self.extractor.edge_vocab_size} edge labels")

        self.rhs_encoder = RHSEncoder(self.extractor.rhs_vocab_size, rhs_embed_dim, rhs_hidden_dim,
                                      bidirectional=True)
        self.graph_encoder = RuleGraphEncoder(self.extractor.edge_vocab_size, graph_embed_dim,
                                              graph_hidden_dim)
        self.combine = nn.Sequential(
            nn.Linear(self.rhs_encoder.output_dim + self.graph_encoder.output_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self._register_structures()
        self.to(self.device)

    def _register_structures(self):
        structures = self.extractor.structures
        rhs = [s.rhs_label_indices for s in structures]
        rhs_sequences, _ = pad_rows(rhs)
        terminal_edges, terminal_mask = pad_rows([s.terminal_edge_label_indices for s in structures])
        nonterminal_edges, nonterminal_mask = pad_rows(
            [s.nonterminal_edge_label_indices for s in structures])
        self.register_buffer('rhs_sequences', rhs_sequences)
        self.register_buffer('rhs_lengths', torch.tensor([max(len(r), 1) for r in rhs], dtype=torch.long))
        self.register_buffer('terminal_edges', terminal_edges)
        self.register_buffer('terminal_mask', terminal_mask)
        self.register_buffer('nonterminal_edges', nonterminal_edges)
        self.register_buffer('nonterminal_mask', nonterminal_mask)
        self.register_buffer('num_nodes', torch.tensor([s.num_nodes for s in structures], dtype=torch.long))
        self.register_buffer('num_external',
                             torch.tensor([s.num_external_nodes for s in structures], dtype=torch.long))

    @property
    def num_rules(self) -> int:
        return self.extractor.num_rules

    def forward(self) -> Tensor:
        rhs_emb = self.rhs_encoder(self.rhs_sequences, self.rhs_lengths)
        graph_emb = self.graph_encoder(self.terminal_edges, self.terminal_mask, self.nonterminal_edges,
                                       self.nonterminal_mask, self.num_nodes, self.num_external)
        return self.combine(torch.cat([rhs_emb, graph_emb], dim=-1))

    def get_embedding(self, rule_ids: Tensor) -> Tensor:
        return self.forward()[rule_ids]

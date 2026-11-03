from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from .hypergraph_embeddings import (
    CachedRuleEmbedding,
    edge_label_vocab,
    is_empty_rule,
    load_grammar_rules,
    pad_rows,
)


@dataclass
class SerializedRuleData:
    rule_idx: int
    edge_labels: list[str]
    edge_label_indices: list[int]
    edge_is_terminal: list[bool]
    num_nodes: int
    num_external_nodes: int


class SerializedStructureExtractor:
    def __init__(self, grammar_file: str):
        self.manager, rules = load_grammar_rules(grammar_file)
        self.edge_label_vocab = edge_label_vocab(rules)
        self.structures = [self._serialize(i, rule) for i, rule in enumerate(rules)]

    def _serialize(self, rule_idx, rule):
        if is_empty_rule(rule):
            return SerializedRuleData(rule_idx, [], [], [], 0, 0)
        terminal = [edge.label for edge in rule.terminal_edges]
        nonterminal = [edge.label for edge in rule.nonterminal_edges]
        labels = terminal + nonterminal
        return SerializedRuleData(
            rule_idx=rule_idx,
            edge_labels=labels,
            edge_label_indices=[self.edge_label_vocab[label] for label in labels],
            edge_is_terminal=[True] * len(terminal) + [False] * len(nonterminal),
            num_nodes=len(list(rule.nodes)),
            num_external_nodes=len(list(rule.external_nodes)),
        )

    @property
    def num_rules(self) -> int:
        return len(self.structures)

    @property
    def edge_vocab_size(self) -> int:
        return len(self.edge_label_vocab)


def masked_sum(emb: Tensor, mask: Tensor) -> Tensor:
    return (emb * mask.unsqueeze(-1).float()).sum(dim=1)


def lstm_final_state(lstm: nn.LSTM, embedded: Tensor, lengths: Tensor) -> Tensor:
    """Last layer's final hidden state (forward and backward concatenated if bidirectional)."""
    packed = pack_padded_sequence(embedded, lengths.clamp(min=1).cpu(), batch_first=True,
                                  enforce_sorted=False)
    _, (h_n, _) = lstm(packed)
    if lstm.bidirectional:
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)
    return h_n[-1]


class BagOfEdgesEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64,
                 output_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.terminal_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.nonterminal_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layers = []
        in_dim = embed_dim * 2 + 5
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, terminal_edges: Tensor, terminal_mask: Tensor, nonterminal_edges: Tensor,
                nonterminal_mask: Tensor, count_features: Tensor) -> Tensor:
        terminal_sum = masked_sum(self.terminal_embedding(terminal_edges), terminal_mask)
        nonterminal_sum = masked_sum(self.nonterminal_embedding(nonterminal_edges), nonterminal_mask)
        return self.mlp(torch.cat([terminal_sum, nonterminal_sum, count_features], dim=-1))


class SequenceEncoder(nn.Module):
    """LSTM over the edge labels in extraction order, which is arbitrary."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64,
                 output_dim: int = 64, num_layers: int = 1, bidirectional: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.output_proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, sequences: Tensor, lengths: Tensor) -> Tensor:
        return self.output_proj(lstm_final_state(self.lstm, self.embedding(sequences), lengths))


class SerializedRuleEmbedding(CachedRuleEmbedding):
    """Edge labels and counts only, no connectivity: the baseline for the GNN embedding."""

    def __init__(
        self,
        grammar_file: str,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        mode: str = "bag",  # bag or sequence
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.extractor = SerializedStructureExtractor(grammar_file)
        print(f"Serialized structures ({mode}): {self.extractor.num_rules} rules, "
              f"{self.extractor.edge_vocab_size} edge labels")

        encoders = {"bag": BagOfEdgesEncoder, "sequence": SequenceEncoder}
        if mode not in encoders:
            raise ValueError(f"Unknown mode: {mode}. Use 'bag' or 'sequence'.")
        self.encoder = encoders[mode](vocab_size=self.extractor.edge_vocab_size, embed_dim=embed_dim,
                                      hidden_dim=hidden_dim, output_dim=embedding_dim)
        if mode == "bag":
            self._register_bag()
        else:
            self._register_sequences()
        self.to(self.device)

    def _register_bag(self):
        structures = self.extractor.structures
        terminal = [[i for i, t in zip(s.edge_label_indices, s.edge_is_terminal) if t]
                    for s in structures]
        nonterminal = [[i for i, t in zip(s.edge_label_indices, s.edge_is_terminal) if not t]
                       for s in structures]
        terminal_edges, terminal_mask = pad_rows(terminal)
        nonterminal_edges, nonterminal_mask = pad_rows(nonterminal)
        counts = torch.tensor(
            [[len(s.edge_label_indices) / 10.0, len(t) / 10.0, len(nt) / 10.0,
              s.num_nodes / 10.0, s.num_external_nodes / 5.0]
             for s, t, nt in zip(structures, terminal, nonterminal)],
            dtype=torch.float32)
        self.register_buffer('terminal_edges', terminal_edges)
        self.register_buffer('terminal_mask', terminal_mask)
        self.register_buffer('nonterminal_edges', nonterminal_edges)
        self.register_buffer('nonterminal_mask', nonterminal_mask)
        self.register_buffer('count_features', counts)

    def _register_sequences(self):
        rows = [s.edge_label_indices for s in self.extractor.structures]
        self.register_buffer('sequences', pad_rows(rows)[0])
        self.register_buffer('lengths', torch.tensor([max(len(r), 1) for r in rows], dtype=torch.long))

    def _compute_embeddings(self) -> Tensor:
        if self.mode == "bag":
            return self.encoder(self.terminal_edges, self.terminal_mask, self.nonterminal_edges,
                                self.nonterminal_mask, self.count_features)
        return self.encoder(self.sequences, self.lengths)

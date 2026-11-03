import math
import random
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..forest import PackedForest
from ..inference import inside, inside_with_entropy, inside_with_scorer
from .em import EarlyStop
from .neural import _mlp


@dataclass
class CompoundConfig:
    max_iterations: int = 100
    learning_rate: float = 0.001
    relative_threshold: float = 0.001
    # Maximize LL - entropy_weight * H(d|G). The summed likelihood does not care how mass is
    # spread over one graph's derivations, but generation samples a single derivation.
    entropy_weight: float = 0.0
    # Compute H on this many random forests per step and rescale (the entropy pass costs ~9x
    # the inside pass); 0 = every forest.
    entropy_subsample: int = 0
    forest_batch_size: int = 0  # unused
    patience: int = 20
    optimizer: str = "adam"  # adam, sgd or adamw
    weight_decay: float = 0.0
    grad_clip: float = 1.0  # 0 disables clipping

    min_log_prob: float = -20.0
    softmax_temperature: float = 1.0
    adaptive_temperature: bool = True

    convergence_threshold: float = None  # accepted from old callers, unused

    embedding_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 32
    num_mlp_layers: int = 2
    dropout: float = 0.0
    encoder_type: str = "mlp"  # mlp or lstm
    lstm_layers: int = 1

    use_structural_embeddings: bool = False
    structural_encoder_type: str = "gnn"  # gnn, gnn_reference, serialized_bag, serialized_seq, legacy
    structural_gnn_layers: int = 2
    structural_hidden_dim: int = 128
    structural_embed_dim: int = 64
    structural_edge_embed_dim: int = 32
    structural_dropout: float = 0.0
    grammar_file: str | None = None  # required with use_structural_embeddings

    # only for structural_encoder_type="legacy"
    structural_rhs_embed_dim: int = 32
    structural_rhs_hidden_dim: int = 64
    structural_graph_embed_dim: int = 32
    structural_graph_hidden_dim: int = 64

    num_z_samples: int = 1
    kl_weight: float = 1.0
    kl_warmup_iterations: int = 20
    free_bits: float = 0.0  # per-dimension KL floor
    batch_size: int = 0  # 0 = full batch
    lazy_features: bool = False  # unused: features_list=None is what makes features lazy

    lr_warmup_iterations: int = 0
    lstm_grad_clip: float = 0.5
    disable_auto_scaling: bool = False
    encoder_lr_scale: float = 1.0  # unused

    use_mixed_precision: bool = False  # CUDA only
    disable_cuda_cache_clear: bool = True
    use_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"

    quick_mode: bool = False
    quick_max_graphs: int = 50
    quick_max_iterations: int = 20
    quick_eval_interval: int = 5

    eval_entropy: bool = False
    eval_interval: int = 10
    eval_at_end: bool = True

    device: str = "auto"  # auto, cpu, cuda, cuda:0, mps, ...
    verbose: bool = True


@dataclass
class CompoundResult:
    rule_log_probs: Tensor | None
    elbo_history: list[float]
    ll_history: list[float]
    kl_history: list[float]
    converged: bool
    num_iterations: int
    model_state: dict | None = None
    effective_config: dict | None = None  # dimensions actually used after large-grammar scaling
    entropy_history: list[float] = field(default_factory=list)
    final_entropy: float | None = None
    final_perplexity: float | None = None


def get_device(device_str: str = "auto") -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_structural_embedder(config: CompoundConfig, device: torch.device) -> nn.Module | None:
    if not config.use_structural_embeddings:
        return None
    if not config.grammar_file:
        raise ValueError("grammar_file is required when use_structural_embeddings=True")

    kind = config.structural_encoder_type
    if kind in ("gnn", "gnn_reference"):
        # "gnn" is the batched implementation, equal to the reference up to float32 round-off
        if kind == "gnn":
            from .hypergraph_embeddings_batched import \
                BatchedGraphStructuralRuleEmbedding as Embedding
        else:
            from .hypergraph_embeddings import GraphStructuralRuleEmbedding as Embedding
        return Embedding(
            grammar_file=config.grammar_file,
            embedding_dim=config.structural_embed_dim,
            hidden_dim=config.structural_hidden_dim,
            num_gnn_layers=config.structural_gnn_layers,
            edge_embed_dim=config.structural_edge_embed_dim,
            dropout=config.structural_dropout,
            device=device,
        )
    if kind in ("serialized_bag", "serialized_seq"):
        from .serialized_embeddings import SerializedRuleEmbedding
        return SerializedRuleEmbedding(
            grammar_file=config.grammar_file,
            embedding_dim=config.structural_embed_dim,
            hidden_dim=config.structural_hidden_dim,
            embed_dim=config.structural_edge_embed_dim,
            mode="bag" if kind == "serialized_bag" else "sequence",
            device=device,
        )
    if kind == "legacy":
        from .structural_embeddings import StructuralRuleEmbedding
        return StructuralRuleEmbedding(
            grammar_file=config.grammar_file,
            embedding_dim=config.structural_embed_dim,
            rhs_embed_dim=config.structural_rhs_embed_dim,
            rhs_hidden_dim=config.structural_rhs_hidden_dim,
            graph_embed_dim=config.structural_graph_embed_dim,
            graph_hidden_dim=config.structural_graph_hidden_dim,
            device=device,
        )
    raise ValueError(
        f"Unknown structural_encoder_type: {kind}. "
        f"Use 'gnn', 'gnn_reference', 'serialized_bag', 'serialized_seq', or 'legacy'."
    )


class CompoundRuleScorer(nn.Module):
    """score(r) = MLP([lhs(A); rhs(r); z]), log-softmaxed within each LHS label group."""

    def __init__(
        self,
        num_rules: int,
        num_labels: int,
        rule_to_label: Tensor,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 0,
        num_layers: int = 2,
        dropout: float = 0.0,
        device: torch.device | None = None,
        min_log_prob: float = -20.0,
        softmax_temperature: float = 1.0,
        adaptive_temperature: bool = True,
        structural_embedder: nn.Module | None = None,
        rule_residual: bool = False,
    ):
        super().__init__()
        self.num_rules = num_rules
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.device = device or torch.device('cpu')
        self.min_log_prob = min_log_prob
        self.base_temperature = softmax_temperature
        self.adaptive_temperature = adaptive_temperature

        self.lhs_embedding = nn.Embedding(num_labels, embedding_dim)
        self.use_structural = structural_embedder is not None
        if self.use_structural:
            self.structural_embedder = structural_embedder

        # A per-rule scalar on top of the structural score: rules the encoder cannot tell
        # apart would otherwise get identical scores. Starts at zero.
        self.use_rule_residual = bool(rule_residual)
        if self.use_rule_residual:
            self.rule_residual = nn.Embedding(num_rules, 1)
            nn.init.zeros_(self.rule_residual.weight)
            self.rule_residual = self.rule_residual.to(self.device)
        else:
            self.rhs_embedding = nn.Embedding(num_rules, embedding_dim)

        self.mlp = _mlp(2 * embedding_dim + latent_dim, hidden_dim, num_layers, dropout)
        self.register_buffer('rule_to_label', rule_to_label.to(self.device))

        self.label_groups = [torch.where(self.rule_to_label == label)[0]
                             for label in range(num_labels)]
        self.max_group_size = max([0] + [len(g) for g in self.label_groups])
        # Softmax over a large group drives the losers to -inf; smooth it from 64 rules up.
        self.softmax_temperature = softmax_temperature
        if adaptive_temperature and self.max_group_size > 64:
            self.softmax_temperature = softmax_temperature * max(
                1.0, math.log2(self.max_group_size) / 6.0)
        self.to(self.device)

    def _grouped_log_softmax(self, scores: Tensor) -> Tensor:
        log_probs = torch.zeros(self.num_rules, device=self.device)
        for indices in self.label_groups:
            if len(indices) == 0:
                continue
            group = scores[indices]
            if self.softmax_temperature != 1.0:
                group = group / self.softmax_temperature
            group = F.log_softmax(group, dim=0)
            if self.min_log_prob < 0:
                group = torch.clamp(group, min=self.min_log_prob)
            log_probs[indices] = group
        return log_probs

    def get_rule_scores(self, z: Tensor | None = None) -> Tensor:
        lhs = self.lhs_embedding(self.rule_to_label)
        if self.use_structural:
            rhs = self.structural_embedder()
        else:
            rhs = self.rhs_embedding(torch.arange(self.num_rules, device=self.device))

        if self.latent_dim > 0:
            if z is None:
                raise ValueError("z is required when latent_dim > 0")
            combined = torch.cat([lhs, rhs, z.unsqueeze(0).expand(self.num_rules, -1)], dim=-1)
        else:
            combined = torch.cat([lhs, rhs], dim=-1)
        scores = self.mlp(combined).squeeze(-1)

        # getattr: whole-pickled scorers from before rule_residual existed lack the attribute
        if getattr(self, "use_rule_residual", False):
            idx = torch.arange(self.num_rules, device=self.device)
            scores = scores + self.rule_residual(idx).squeeze(-1)
        return scores

    def get_rule_log_probs(self, z: Tensor | None = None) -> Tensor:
        return self._grouped_log_softmax(self.get_rule_scores(z))

    def get_scorer_function(self, z: Tensor | None = None) -> Callable:
        """Returns scorer(edge, parent, children, node_embeddings) -> log-prob for inside_with_scorer."""
        log_probs = self.get_rule_log_probs(z)

        def scorer(edge, parent_node, child_nodes, node_embeddings):
            return log_probs[edge.rule_id]
        return scorer

    def forward(self, z: Tensor | None = None) -> Tensor:
        return self.get_rule_log_probs(z)


class _GaussianPosterior(nn.Module):
    """Diagonal Gaussian q(z|x) helpers; subclasses set the clamp bounds."""

    LOG_VAR_MIN = -10.0
    LOG_VAR_MAX = 10.0
    MU_MAX = 10.0

    def _clamp(self, mu, log_var):
        return (torch.clamp(mu, -self.MU_MAX, self.MU_MAX),
                torch.clamp(log_var, self.LOG_VAR_MIN, self.LOG_VAR_MAX))

    def sample(self, mu: Tensor, log_var: Tensor) -> Tensor:
        log_var = torch.clamp(log_var, self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def kl_divergence(self, mu: Tensor, log_var: Tensor, free_bits: float = 0.0) -> Tensor:
        """KL(q || N(0, I)), with free_bits as a floor on each dimension's KL."""
        mu, log_var = self._clamp(mu, log_var)
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        if free_bits and free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=float(free_bits))
        return torch.sum(kl_per_dim)


class GraphEncoder(_GaussianPosterior):
    """MLP over graph features -> (mu, log_var)."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self._clamp(self.mu_head(h), self.logvar_head(h))


class LSTMGraphEncoder(_GaussianPosterior):
    """(Bi)LSTM over the forest's rule sequence -> (mu, log_var)."""

    LOG_VAR_MIN = -10.0
    LOG_VAR_MAX = 1.0
    MU_MAX = 3.0
    MAX_SEQ_LEN = 500

    def __init__(self, num_rules: int, embedding_dim: int = 64, hidden_dim: int = 128,
                 latent_dim: int = 32, num_layers: int = 1, dropout: float = 0.0,
                 bidirectional: bool = True):
        super().__init__()
        self.num_rules = num_rules
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional

        self.rule_embedding = nn.Embedding(num_rules, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.layer_norm = nn.LayerNorm(out_dim)
        self.mu_head = nn.Linear(out_dim, latent_dim)
        self.logvar_head = nn.Linear(out_dim, latent_dim)

        nn.init.normal_(self.rule_embedding.weight, mean=0.0, std=min(0.1, 1.0 / math.sqrt(num_rules)))
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)  # forget gate
        # start close to the prior, with a small variance
        nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.logvar_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.logvar_head.bias, -4.0)

    def forward(self, rule_ids: Tensor) -> tuple[Tensor, Tensor]:
        if len(rule_ids) == 0:
            device = self.rule_embedding.weight.device
            return (torch.zeros(self.latent_dim, device=device),
                    torch.zeros(self.latent_dim, device=device))
        if len(rule_ids) > self.MAX_SEQ_LEN:
            rule_ids = rule_ids[torch.linspace(0, len(rule_ids) - 1, self.MAX_SEQ_LEN).long()]

        _, (h_n, _) = self.lstm(self.rule_embedding(rule_ids).unsqueeze(0))
        if self.bidirectional:
            h = torch.cat([h_n[-2, 0, :], h_n[-1, 0, :]], dim=0)
        else:
            h = h_n[-1, 0, :]
        h = self.layer_norm(h)
        return self._clamp(self.mu_head(h), self.logvar_head(h))


def extract_rule_sequence(forest: PackedForest) -> list[int]:
    """Rule ids of the forest's hyperedges in breadth-first order from the roots."""
    if not forest.roots or not forest.hyperedges:
        return []
    edges_of = defaultdict(list)
    for edge in forest.hyperedges:
        edges_of[edge.parent_node].append(edge)

    rule_ids = []
    seen = set()
    queue = deque(forest.roots)
    while queue:
        for edge in edges_of.get(queue.popleft(), ()):
            key = (edge.parent_node, edge.rule_id, tuple(edge.child_nodes))
            if key not in seen:
                seen.add(key)
                rule_ids.append(edge.rule_id)
                queue.extend(edge.child_nodes)
    return rule_ids


def extract_bag_of_rules_features(forest: PackedForest, num_rules: int,
                                  normalize: bool = True) -> Tensor:
    counts = torch.zeros(num_rules)
    for edge in forest.hyperedges:
        if 0 <= edge.rule_id < num_rules:
            counts[edge.rule_id] += 1
    if normalize and counts.sum() > 0:
        counts = counts / counts.sum()
    return counts


def extract_forest_statistics(forest: PackedForest) -> Tensor:
    num_nodes = len(forest.nodes)
    num_edges = len(forest.hyperedges)
    avg_branching = sum(len(e.child_nodes) for e in forest.hyperedges) / max(num_edges, 1)
    terminal_frac = sum(1 for n in forest.nodes if not n.incoming_edges) / max(num_nodes, 1)
    return torch.tensor([
        math.log1p(num_nodes) / 10.0,
        math.log1p(num_edges) / 10.0,
        math.log1p(len(forest.roots)) / 5.0,
        avg_branching / 5.0,
        terminal_frac,
    ], dtype=torch.float32)


def extract_graph_features(forest: PackedForest, num_rules: int,
                           include_statistics: bool = True) -> Tensor:
    """Bag of rules, optionally followed by the five forest statistics."""
    rule_features = extract_bag_of_rules_features(forest, num_rules)
    if include_statistics:
        return torch.cat([rule_features, extract_forest_statistics(forest)])
    return rule_features


class CompoundHRG(nn.Module):
    """z ~ N(0, I), p(r | A, z) from CompoundRuleScorer, q(z | x) from an MLP or LSTM encoder."""

    MIN_LOG_PROB = -100.0

    def __init__(
        self,
        num_rules: int,
        num_labels: int,
        rule_to_label: Tensor,
        encoder_input_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_mlp_layers: int = 2,
        dropout: float = 0.0,
        device: torch.device | None = None,
        min_log_prob: float = -20.0,
        softmax_temperature: float = 1.0,
        adaptive_temperature: bool = True,
        encoder_type: str = "mlp",
        lstm_layers: int = 1,
        structural_embedder: nn.Module | None = None,
        free_bits: float = 0.0,
    ):
        super().__init__()
        self.num_rules = num_rules
        self.num_labels = num_labels
        self.latent_dim = latent_dim
        self.device = device or torch.device('cpu')
        self.encoder_type = encoder_type
        self.free_bits = float(free_bits or 0.0)

        if encoder_type == "lstm":
            self.encoder = LSTMGraphEncoder(num_rules, embedding_dim, hidden_dim, latent_dim,
                                            lstm_layers, dropout, bidirectional=True)
        else:
            self.encoder = GraphEncoder(encoder_input_dim, latent_dim, hidden_dim,
                                        num_mlp_layers, dropout)
        self.scorer = CompoundRuleScorer(
            num_rules=num_rules,
            num_labels=num_labels,
            rule_to_label=rule_to_label,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_mlp_layers,
            dropout=dropout,
            device=device,
            min_log_prob=min_log_prob,
            softmax_temperature=softmax_temperature,
            adaptive_temperature=adaptive_temperature,
            structural_embedder=structural_embedder,
        )
        self.to(self.device)

    def encode(self, features_or_forest) -> tuple[Tensor, Tensor]:
        """Takes a feature tensor for the MLP encoder, a PackedForest for the LSTM one."""
        if self.encoder_type != "lstm":
            return self.encoder(features_or_forest.to(self.device))
        rule_ids = extract_rule_sequence(features_or_forest)
        if not rule_ids:
            return (torch.zeros(self.latent_dim, device=self.device),
                    torch.zeros(self.latent_dim, device=self.device))
        return self.encoder(torch.tensor(rule_ids, dtype=torch.long, device=self.device))

    def decode(self, forest: PackedForest, z: Tensor) -> Tensor:
        """log p(x | z) by the inside algorithm."""
        return inside_with_scorer(forest, self.scorer.get_scorer_function(z))[1]

    def forward(self, forest: PackedForest, features: Tensor | None = None,
                num_samples: int = 1) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (E_q[log p(x|z)], KL(q || p), ELBO); non-finite values fall back to constants."""
        if self.encoder_type == "lstm":
            mu, log_var = self.encode(forest)
        else:
            if features is not None and torch.isnan(features).any():
                features = torch.nan_to_num(features, nan=0.0)
            mu, log_var = self.encode(features)

        if torch.isnan(mu).any() or torch.isnan(log_var).any():
            floor = torch.tensor(self.MIN_LOG_PROB, device=self.device)
            kl = torch.tensor(0.0, device=self.device)
            return floor, kl, floor - kl

        # getattr: whole-pickled models from before free_bits existed lack the attribute
        kl = self.encoder.kl_divergence(mu, log_var, free_bits=getattr(self, 'free_bits', 0.0))
        if torch.isnan(kl) or torch.isinf(kl):
            kl = torch.tensor(0.0, device=self.device)

        total_ll = torch.tensor(0.0, device=self.device)
        for _ in range(num_samples):
            z = self.encoder.sample(mu, log_var)
            log_prob = None if torch.isnan(z).any() else self.decode(forest, z)
            if log_prob is None or torch.isinf(log_prob) or torch.isnan(log_prob):
                log_prob = torch.tensor(self.MIN_LOG_PROB, device=self.device)
            total_ll = total_ll + log_prob
        expected_ll = torch.clamp(total_ll / max(num_samples, 1), min=self.MIN_LOG_PROB)
        return expected_ll, kl, expected_ll - kl

    def get_posterior_mean_log_probs(self, features_or_forest) -> Tensor:
        mu, _ = self.encode(features_or_forest)
        return self.scorer.get_rule_log_probs(mu)


def _compute_entropy_metrics(forests, rule_log_probs):
    from ..evaluation.entropy import compute_batch_entropy
    entropy = compute_batch_entropy(forests, rule_log_probs)
    return {'entropy': entropy, 'perplexity': math.exp(entropy) if entropy < 20 else float('inf')}


def _structural_embedder(scorer):
    return getattr(scorer, 'structural_embedder', None)


def _invalidate_structural_cache(scorer):
    # backward() frees the graph the cached rule embeddings were built in
    embedder = _structural_embedder(scorer)
    if embedder is not None and hasattr(embedder, 'invalidate_cache'):
        embedder.invalidate_cache()


def _print_structural(config):
    print("  Structural embeddings enabled:")
    print(f"    Encoder type: {config.structural_encoder_type}")
    print(f"    Embed dim: {config.structural_embed_dim}, Hidden dim: {config.structural_hidden_dim}")
    if config.structural_encoder_type == "gnn":
        print(f"    GNN layers: {config.structural_gnn_layers}")


def _quick_subsample(forests, features_list, n):
    """n forests evenly spaced in order of size."""
    if len(forests) <= n:
        return forests, features_list
    by_size = sorted(range(len(forests)), key=lambda i: len(forests[i].hyperedges))
    step = len(by_size) / n
    selected = [by_size[int(i * step)] for i in range(n)]
    if features_list is not None:
        features_list = [features_list[i] for i in selected]
    return [forests[i] for i in selected], features_list


def _scaled_settings(config, num_rules, use_lstm):
    """(latent, hidden, embedding dim, lr, grad clip, lstm grad clip), shrunk for an LSTM
    encoder over a large rule vocabulary."""
    if not use_lstm or num_rules <= 1000 or config.disable_auto_scaling:
        return (config.latent_dim, config.hidden_dim, config.embedding_dim,
                config.learning_rate, config.grad_clip, config.lstm_grad_clip)
    if num_rules > 5000:
        return (min(config.latent_dim, 16), min(config.hidden_dim, 64),
                min(config.embedding_dim, 32), min(config.learning_rate, 0.0001), 0.25, 0.25)
    scale = 1000.0 / num_rules
    return (min(config.latent_dim, max(4, int(config.latent_dim * (scale ** 0.5)))),
            min(config.hidden_dim, max(32, int(config.hidden_dim * (scale ** 0.3)))),
            min(config.embedding_dim, max(16, int(config.embedding_dim * (scale ** 0.3)))),
            config.learning_rate * (scale ** 0.5),
            min(config.grad_clip, 0.5),
            min(config.lstm_grad_clip, 0.25))


def _elbo_terms(model, forests, features, num_samples, device):
    """Summed (ll, kl) over the forests whose terms are not NaN, and how many there were."""
    total_ll = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    count = 0
    for forest, feats in zip(forests, features):
        if not forest.roots:
            continue
        ll, kl, _ = model(forest, feats, num_samples=num_samples)
        if torch.isnan(ll) or torch.isnan(kl):
            continue
        total_ll = total_ll + ll
        total_kl = total_kl + kl
        count += 1
    return total_ll, total_kl, count


def _has_bad_grad(model):
    return any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
               for p in model.parameters())


def train_compound_hrg(
    forests: list[PackedForest],
    features_list: list[Tensor] | None,
    num_rules: int,
    num_labels: int,
    rule_to_label: Tensor,
    config: CompoundConfig | None = None,
    model: CompoundHRG | None = None,
    callback: Callable[[int, CompoundHRG, float], None] | None = None,
) -> tuple[CompoundResult, CompoundHRG]:
    """ELBO training. features_list=None extracts MLP features per batch on the fly."""
    config = config or CompoundConfig()
    device = get_device(config.device)
    use_lstm = config.encoder_type == "lstm"
    if config.verbose:
        print(f"  Using device: {device}, encoder: {config.encoder_type}")

    max_iters = config.max_iterations
    if config.quick_mode:
        max_iters = min(config.quick_max_iterations, config.max_iterations)
        forests, features_list = _quick_subsample(forests, features_list, config.quick_max_graphs)
        if config.verbose:
            print(f"  Quick mode: {len(forests)} graphs, {max_iters} iterations")
    lazy_features = features_list is None and not use_lstm

    latent_dim, hidden_dim, embedding_dim, lr, grad_clip, lstm_grad_clip = _scaled_settings(
        config, num_rules, use_lstm)
    if config.verbose and use_lstm and num_rules > 1000 and not config.disable_auto_scaling:
        print(f"  Large grammar ({num_rules} rules): latent {latent_dim}, hidden {hidden_dim}, "
              f"embedding {embedding_dim}, lr {lr:.6f}, grad clip {grad_clip}")

    if model is None:
        if use_lstm:
            encoder_input_dim = 0
        elif features_list:
            encoder_input_dim = features_list[0].shape[0]
        else:
            encoder_input_dim = extract_graph_features(forests[0], num_rules).shape[0]
        structural_embedder = create_structural_embedder(config, device)
        if structural_embedder is not None and config.verbose:
            _print_structural(config)
        model = CompoundHRG(
            num_rules=num_rules,
            num_labels=num_labels,
            rule_to_label=rule_to_label,
            encoder_input_dim=encoder_input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_mlp_layers=config.num_mlp_layers,
            dropout=config.dropout,
            device=device,
            min_log_prob=config.min_log_prob,
            softmax_temperature=config.softmax_temperature,
            adaptive_temperature=config.adaptive_temperature,
            encoder_type=config.encoder_type,
            lstm_layers=config.lstm_layers,
            structural_embedder=structural_embedder,
            free_bits=config.free_bits,
        )

    if config.use_torch_compile and hasattr(torch, 'compile'):
        try:
            model.scorer = torch.compile(model.scorer, mode=config.torch_compile_mode,
                                         fullgraph=False)
        except Exception as e:
            if config.verbose:
                print(f"  torch.compile failed: {e}, continuing without it")

    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    if config.optimizer not in optimizers:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    optimizer = optimizers[config.optimizer](model.parameters(), lr=lr,
                                             weight_decay=config.weight_decay)

    use_amp = config.use_mixed_precision and device.type == 'cuda'
    if use_amp:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler('cuda')

    eval_interval = config.quick_eval_interval if config.quick_mode else config.eval_interval
    kl_warmup = config.kl_warmup_iterations
    if config.quick_mode:
        kl_warmup = min(kl_warmup, max_iters // 2)

    # structural embeddings are shared by every forest, so they force one full batch
    structural = _structural_embedder(model.scorer) is not None
    batch_size = config.batch_size if config.batch_size > 0 and not structural else len(forests)

    elbo_history, ll_history, kl_history, entropy_history = [], [], [], []
    prev_elbo = None
    best_elbo = None
    stale = 0
    converged = False
    nan_iterations = 0
    start_time = time.time()

    for iteration in range(max_iters):
        model.train()
        optimizer.zero_grad()
        kl_weight = config.kl_weight
        if kl_warmup > 0:
            kl_weight = min(1.0, iteration / kl_warmup) * config.kl_weight

        warmup = config.lr_warmup_iterations
        if warmup > 0 and iteration <= warmup:
            for group in optimizer.param_groups:
                group['lr'] = lr * ((iteration + 1) / warmup) if iteration < warmup else lr

        num_batches = (len(forests) + batch_size - 1) // batch_size
        total_ll = 0.0
        total_kl = 0.0
        for start in range(0, num_batches * batch_size, batch_size):
            _invalidate_structural_cache(model.scorer)
            batch = forests[start:start + batch_size]
            if use_lstm:
                features = [None] * len(batch)
            elif lazy_features:
                features = [extract_graph_features(f, num_rules) for f in batch]
            else:
                features = features_list[start:start + batch_size]

            with autocast('cuda') if use_amp else nullcontext():
                batch_ll, batch_kl, count = _elbo_terms(model, batch, features,
                                                        config.num_z_samples, device)
                batch_loss = -(batch_ll - kl_weight * batch_kl) / num_batches
            if not torch.isnan(batch_loss) and count > 0 and batch_loss.requires_grad:
                (scaler.scale(batch_loss) if use_amp else batch_loss).backward()
            elif not use_amp:
                continue
            total_ll += batch_ll.item()
            total_kl += batch_kl.item()
            if device.type == 'cuda' and not config.disable_cuda_cache_clear:
                torch.cuda.empty_cache()

        if _has_bad_grad(model):
            nan_iterations += 1
            optimizer.zero_grad()
            if nan_iterations <= 5:
                print(f"    WARNING: NaN/Inf gradient at iter {iteration + 1} (count: {nan_iterations})")
            elif nan_iterations == 6:
                print("    WARNING: suppressing further NaN warnings")
            continue

        if use_amp:
            scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        if use_lstm and lstm_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=lstm_grad_clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        elbo = total_ll - kl_weight * total_kl
        elbo_history.append(elbo)
        ll_history.append(total_ll)
        kl_history.append(total_kl)
        if callback is not None:
            callback(iteration, model, elbo)

        if (iteration + 1) % max(1, max_iters // 10) == 0 or iteration == 0:
            eta = ""
            if iteration > 0:
                seconds = (time.time() - start_time) / (iteration + 1) * (max_iters - iteration - 1)
                eta = f", ETA: {seconds / 60:.1f}min" if seconds > 60 else f", ETA: {seconds:.0f}s"
            print(f"  [{100 * (iteration + 1) / max_iters:5.1f}%] Iter {iteration + 1}/{max_iters}: "
                  f"ELBO={elbo:.2f}, LL={total_ll:.2f}, KL={total_kl:.4f}{eta}")

        if config.eval_entropy and iteration >= kl_warmup and (iteration + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                prior_mean = torch.zeros(model.latent_dim, device=device)
                metrics = _compute_entropy_metrics(
                    forests, model.scorer.get_rule_log_probs(z=prior_mean))
                entropy_history.append(metrics['entropy'])
                if config.verbose:
                    print(f"    Entropy: {metrics['entropy']:.4f}, "
                          f"Perplexity: {metrics['perplexity']:.2f}")
            model.train()

        if math.isnan(elbo) or math.isinf(elbo):
            if config.verbose:
                print(f"Stopped at iteration {iteration + 1}: ELBO is {elbo}")
            break

        # the ELBO moves with the KL weight during warmup, so convergence is only judged after it
        if iteration >= kl_warmup:
            if best_elbo is None:
                best_elbo = elbo
            if prev_elbo is not None:
                relative = abs(elbo - prev_elbo) / (abs(elbo) + 1e-8)
                if relative < config.relative_threshold:
                    converged = True
                    if config.verbose:
                        print(f"Converged at iteration {iteration + 1} "
                              f"(relative improvement {relative:.6f})")
                    break
            if elbo > best_elbo + max(1.0, abs(best_elbo) * config.relative_threshold):
                best_elbo = elbo
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    converged = True
                    if config.verbose:
                        print(f"Stopped at iteration {iteration + 1} "
                              f"(no improvement for {config.patience} iterations)")
                    break
        prev_elbo = elbo

    model.eval()
    with torch.no_grad():
        if use_lstm:
            final_log_probs = model.get_posterior_mean_log_probs(forests[0])
        elif features_list:
            final_log_probs = model.get_posterior_mean_log_probs(features_list[0])
        else:
            final_log_probs = None

    final_entropy = final_perplexity = None
    if config.eval_at_end and final_log_probs is not None:
        try:
            metrics = _compute_entropy_metrics(forests, final_log_probs)
            final_entropy, final_perplexity = metrics['entropy'], metrics['perplexity']
            if config.verbose:
                print(f"  Final entropy: {final_entropy:.4f}, perplexity: {final_perplexity:.2f}")
        except Exception as e:
            if config.verbose:
                print(f"  Final entropy computation failed: {e}")

    if nan_iterations > 0:
        print(f"  {nan_iterations} of {iteration + 1} iterations had NaN gradients")
    if not elbo_history:
        print("  ERROR: no iteration completed")

    effective_config = {
        'latent_dim': latent_dim,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'encoder_type': config.encoder_type,
        'lstm_layers': config.lstm_layers,
    }
    result = CompoundResult(
        rule_log_probs=final_log_probs,
        elbo_history=elbo_history,
        ll_history=ll_history,
        kl_history=kl_history,
        converged=converged,
        num_iterations=len(elbo_history),
        model_state=model.state_dict(),
        effective_config=effective_config,
        entropy_history=entropy_history,
        final_entropy=final_entropy,
        final_perplexity=final_perplexity,
    )
    return result, model


def _likelihood_and_entropy(forests, rule_log_probs, config, iteration, device):
    """Summed log-likelihood, summed derivation entropy, number of forests scored."""
    entropy_idx = None
    if config.entropy_weight != 0.0 and config.entropy_subsample:
        # reseeded per iteration: every forest is seen over training, and runs reproduce
        live = [i for i, f in enumerate(forests) if f.roots]
        picker = random.Random(1000 + iteration)
        entropy_idx = set(picker.sample(live, min(config.entropy_subsample, len(live))))

    total_ll = torch.tensor(0.0, device=device)
    total_h = torch.tensor(0.0, device=device)
    num_scored = num_h = 0
    for i, forest in enumerate(forests):
        if not forest.roots:
            continue
        if config.entropy_weight != 0.0 and (entropy_idx is None or i in entropy_idx):
            log_prob, h = inside_with_entropy(forest, rule_log_probs)
            total_h = total_h + h
            num_h += 1
        else:
            log_prob = inside(forest, rule_log_probs)[1]
        total_ll = total_ll + log_prob
        num_scored += 1

    if num_h and num_h < num_scored:
        # the objective is a sum over forests, so the subsample estimate is scaled up to it
        total_h = total_h * (num_scored / num_h)
    return total_ll, total_h, num_scored


def train_compound_scorer_only(
    forests: list[PackedForest],
    num_rules: int,
    num_labels: int,
    rule_to_label: Tensor,
    config: CompoundConfig | None = None,
    callback: Callable[[int, CompoundRuleScorer, float], None] | None = None,
) -> CompoundResult:
    """The neural parameterization without z: maximizes LL - entropy_weight * H(d|G)."""
    if config is None:
        config = CompoundConfig()
        config.latent_dim = 0
    device = get_device(config.device)
    if config.verbose:
        print(f"  Using device: {device}")

    structural_embedder = create_structural_embedder(config, device)
    if structural_embedder is not None and config.verbose:
        _print_structural(config)
    scorer = CompoundRuleScorer(
        num_rules=num_rules,
        num_labels=num_labels,
        rule_to_label=rule_to_label,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=0,
        num_layers=config.num_mlp_layers,
        dropout=config.dropout,
        device=device,
        min_log_prob=config.min_log_prob,
        softmax_temperature=config.softmax_temperature,
        adaptive_temperature=config.adaptive_temperature,
        structural_embedder=structural_embedder,
    )
    # anything but "adam" gets SGD, adamw included
    make_optimizer = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
    optimizer = make_optimizer(scorer.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)

    stop = EarlyStop(config.relative_threshold, config.patience)
    ll_history = []
    entropy_history = []
    converged = False
    for iteration in range(config.max_iterations):
        scorer.train()
        _invalidate_structural_cache(scorer)
        optimizer.zero_grad()
        total_ll, total_h, num_scored = _likelihood_and_entropy(
            forests, scorer.get_rule_log_probs(), config, iteration, device)
        loss = -(total_ll - config.entropy_weight * total_h)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=config.grad_clip)
        optimizer.step()

        ll = total_ll.item()
        ll_history.append(ll)
        mean_h = total_h.item() / num_scored if num_scored else 0.0
        entropy_history.append(mean_h)
        # stopping tracks the objective: with entropy_weight > 0 the LL may fall as it rises
        objective = ll - config.entropy_weight * total_h.item()
        if callback is not None:
            callback(iteration, scorer, ll)
        if config.verbose and (iteration + 1) % 10 == 0:
            if config.entropy_weight != 0.0:
                print(f"Iteration {iteration + 1}: LL={ll:.4f}  H(d|G)={mean_h:.4f}  "
                      f"obj={objective:.4f}")
            else:
                print(f"Iteration {iteration + 1}: LL={ll:.4f}")

        if math.isnan(ll) or math.isinf(ll):
            if config.verbose:
                print(f"Stopped at iteration {iteration + 1}: LL is {ll}")
            break
        reason = stop(objective)
        if reason:
            converged = True
            if config.verbose:
                print(f"Stopped after {iteration + 1} iterations ({reason})")
            break

    scorer.eval()
    with torch.no_grad():
        _invalidate_structural_cache(scorer)
        final_log_probs = scorer.get_rule_log_probs()

    return CompoundResult(
        rule_log_probs=final_log_probs,
        elbo_history=ll_history,
        ll_history=ll_history,
        kl_history=[0.0] * len(ll_history),
        entropy_history=entropy_history,
        converged=converged,
        num_iterations=len(ll_history),
        model_state=scorer.state_dict(),
        effective_config={
            'latent_dim': 0,
            'embedding_dim': config.embedding_dim,
            'hidden_dim': config.hidden_dim,
            'encoder_type': 'none',
            'lstm_layers': config.lstm_layers,
            'use_structural_embeddings': config.use_structural_embeddings,
            'structural_encoder_type': config.structural_encoder_type,
        },
    )

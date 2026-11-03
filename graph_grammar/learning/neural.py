import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..forest import PackedForest
from ..inference import inside_with_scorer
from .em import EarlyStop
from .gradient import compute_batch_log_likelihood


@dataclass
class NeuralConfig:
    max_iterations: int = 100
    relative_threshold: float = 0.001
    patience: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"  # adam, sgd or adamw
    weight_decay: float = 0.0
    grad_clip: float = 1.0  # 0 disables clipping
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.0
    entropy_weight: float = 0.0
    verbose: bool = True
    convergence_threshold: float = None  # accepted from old callers, unused


@dataclass
class NeuralResult:
    rule_log_probs: Tensor
    log_likelihood_history: list[float]
    converged: bool
    num_iterations: int
    gradient_norm_history: list[float] | None = None
    model_state: dict | None = None


def _mlp(in_dim, hidden_dim, num_layers, dropout=0.0):
    layers = []
    for _ in range(num_layers - 1):
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


class _GroupedScorer(nn.Module):
    """Scores are log-softmaxed within each rule_to_label group (one global group if None)."""

    def _set_labels(self, rule_to_label, num_rules):
        if rule_to_label is not None:
            self.register_buffer('rule_to_label', rule_to_label.to(self.device))
            self.num_labels = self.rule_to_label.max().item() + 1
        else:
            self.register_buffer('rule_to_label',
                                 torch.zeros(num_rules, dtype=torch.long, device=self.device))
            self.num_labels = 1
        # indexed by label id, so empty groups stay in the list
        self.label_groups = [torch.where(self.rule_to_label == label)[0]
                             for label in range(self.num_labels)]

    def _normalize(self, scores):
        log_probs = torch.zeros(self.num_rules, device=self.device)
        for indices in self.label_groups:
            if len(indices) > 0:
                log_probs[indices] = F.log_softmax(scores[indices], dim=0)
        return log_probs


class RuleEmbeddingScorer(_GroupedScorer):
    def __init__(self, num_rules: int, embedding_dim: int = 64, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.0, rule_to_label: Tensor | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.num_rules = num_rules
        self.embedding_dim = embedding_dim
        self.device = device or torch.device('cpu')
        self.rule_embedding = nn.Embedding(num_rules, embedding_dim)
        self.mlp = _mlp(embedding_dim, hidden_dim, num_layers, dropout)
        self._set_labels(rule_to_label, num_rules)
        self.to(self.device)

    def get_rule_scores(self) -> Tensor:
        rule_ids = torch.arange(self.num_rules, device=self.device)
        return self.mlp(self.rule_embedding(rule_ids)).squeeze(-1)

    def get_rule_log_probs(self) -> Tensor:
        return self._normalize(self.get_rule_scores())

    def forward(self) -> Tensor:
        return self.get_rule_log_probs()


class BilinearRuleScorer(_GroupedScorer):
    """score(r) = lhs(r) W rhs(r) + b"""

    def __init__(self, num_rules: int, embedding_dim: int = 64,
                 rule_to_label: Tensor | None = None, device: torch.device | None = None):
        super().__init__()
        self.num_rules = num_rules
        self.embedding_dim = embedding_dim
        self.device = device or torch.device('cpu')
        self.lhs_embedding = nn.Embedding(num_rules, embedding_dim)
        self.rhs_embedding = nn.Embedding(num_rules, embedding_dim)
        self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, 1, bias=True)
        self._set_labels(rule_to_label, num_rules)
        self.to(self.device)

    def get_rule_log_probs(self) -> Tensor:
        rule_ids = torch.arange(self.num_rules, device=self.device)
        scores = self.bilinear(self.lhs_embedding(rule_ids), self.rhs_embedding(rule_ids))
        return self._normalize(scores.squeeze(-1))

    def forward(self) -> Tensor:
        return self.get_rule_log_probs()


class ContextAwareRuleScorer(_GroupedScorer):
    """Scores a rule from its own embedding plus the parent's and the aggregated children's."""

    def __init__(self, num_rules: int, node_embedding_dim: int, rule_embedding_dim: int = 64,
                 hidden_dim: int = 128, num_layers: int = 2, aggregation: str = "mean",
                 rule_to_label: Tensor | None = None, device: torch.device | None = None):
        super().__init__()
        self.num_rules = num_rules
        self.node_embedding_dim = node_embedding_dim
        self.rule_embedding_dim = rule_embedding_dim
        self.aggregation = aggregation  # mean, sum or max
        self.device = device or torch.device('cpu')
        self.rule_embedding = nn.Embedding(num_rules, rule_embedding_dim)
        self.default_node_embedding = nn.Parameter(torch.zeros(node_embedding_dim))
        self.mlp = _mlp(rule_embedding_dim + 2 * node_embedding_dim, hidden_dim, num_layers)
        self._set_labels(rule_to_label, num_rules)
        self.to(self.device)

    def _node_embedding(self, node, node_embeddings):
        return node_embeddings.get(node.id, self.default_node_embedding)

    def _aggregate_children(self, child_embeddings):
        if not child_embeddings:
            return self.default_node_embedding
        stacked = torch.stack(child_embeddings)
        if self.aggregation == "sum":
            return stacked.sum(dim=0)
        if self.aggregation == "max":
            return stacked.max(dim=0)[0]
        return stacked.mean(dim=0)

    def _score(self, rule_id, parent_node, child_nodes, node_embeddings):
        rule_emb = self.rule_embedding(torch.tensor(rule_id, device=self.device))
        parent_emb = self._node_embedding(parent_node, node_embeddings)
        children_emb = self._aggregate_children(
            [self._node_embedding(child, node_embeddings) for child in child_nodes])
        return self.mlp(torch.cat([rule_emb, parent_emb, children_emb])).squeeze(-1)

    def score_edge(self, edge, parent_node, child_nodes, node_embeddings) -> Tensor:
        return self._score(edge.rule_id, parent_node, child_nodes, node_embeddings)

    def get_scorer_function(self) -> Callable:
        """Returns scorer(edge, parent, children, node_embeddings) -> log-prob for inside_with_scorer."""
        def scorer(edge, parent_node, child_nodes, node_embeddings):
            group = self.label_groups[self.rule_to_label[edge.rule_id].item()]
            if len(group) == 1:
                return torch.tensor(0.0, device=self.device)
            # the competitors are rescored in this same context, so the softmax is exact
            score = self.score_edge(edge, parent_node, child_nodes, node_embeddings)
            rules = group.tolist()
            scores = [score if rule == edge.rule_id
                      else self._score(rule, parent_node, child_nodes, node_embeddings)
                      for rule in rules]
            return F.log_softmax(torch.stack(scores), dim=0)[rules.index(edge.rule_id)]
        return scorer

    def forward(self, edge, parent_node, child_nodes, node_embeddings) -> Tensor:
        return self.get_scorer_function()(edge, parent_node, child_nodes, node_embeddings)

    def _context_free_log_probs(self):
        rule_ids = torch.arange(self.num_rules, device=self.device)
        default = self.default_node_embedding.unsqueeze(0).expand(self.num_rules, -1)
        combined = torch.cat([self.rule_embedding(rule_ids), default, default], dim=1)
        return self._normalize(self.mlp(combined).squeeze(-1))


def compute_entropy_regularization(log_probs: Tensor, label_groups: list[Tensor]) -> Tensor:
    """Mean entropy over the groups with at least two rules."""
    total = 0.0
    num_groups = 0
    for indices in label_groups:
        if len(indices) > 1:
            group_log_probs = log_probs[indices]
            entropy = -(torch.exp(group_log_probs) * group_log_probs).sum()
            total = total + entropy
            num_groups += 1
    if num_groups == 0:
        return torch.tensor(0.0, device=log_probs.device)
    return total / num_groups


def compute_batch_log_likelihood_with_scorer(
    forests: list[PackedForest],
    scorer: ContextAwareRuleScorer,
    node_embeddings_list: list[dict[int, Tensor]] | None = None,
) -> Tensor:
    scorer_fn = scorer.get_scorer_function()
    total = torch.tensor(0.0, device=scorer.device)
    for i, forest in enumerate(forests):
        if not forest.roots:
            continue
        embeddings = {}
        if node_embeddings_list is not None and i < len(node_embeddings_list):
            embeddings = node_embeddings_list[i]
        total = total + inside_with_scorer(forest, scorer_fn, embeddings)[1]
    return total


def _make_optimizer(config, params):
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    name = config.optimizer.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    return optimizers[name](params, lr=config.learning_rate, weight_decay=config.weight_decay)


def _train(config, model, params, objective, final_log_probs, callback):
    """objective() -> (log_likelihood, loss); callback(iteration, ll)."""
    optimizer = _make_optimizer(config, params)
    stop = EarlyStop(config.relative_threshold, config.patience)
    history = []
    grad_norms = []

    def result(converged, iterations):
        with torch.no_grad():
            log_probs = final_log_probs()
        return NeuralResult(log_probs, history, converged, iterations, grad_norms,
                            model.state_dict())

    for iteration in range(config.max_iterations):
        optimizer.zero_grad()
        log_likelihood, loss = objective()
        loss.backward()
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in params if p.grad is not None) ** 0.5
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=config.grad_clip)
        optimizer.step()

        ll = log_likelihood.item()
        history.append(ll)
        grad_norms.append(grad_norm)
        if config.verbose:
            print(f"Iteration {iteration + 1}: log-likelihood = {ll:.4f}, grad_norm = {grad_norm:.6f}")
        if not math.isfinite(ll):
            if config.verbose:
                print(f"Stopped at iteration {iteration + 1}: log-likelihood is {ll}")
            return result(False, iteration + 1)
        if callback is not None:
            callback(iteration, ll)
        reason = stop(ll)
        if reason:
            if config.verbose:
                print(f"Stopped after {iteration + 1} iterations ({reason})")
            return result(True, iteration + 1)

    if config.verbose:
        print(f"Did not converge after {config.max_iterations} iterations")
    return result(False, config.max_iterations)


def train_neural(
    forests: list[PackedForest],
    num_rules: int,
    config: NeuralConfig | None = None,
    rule_to_label: Tensor | None = None,
    model: nn.Module | None = None,
    callback: Callable[[int, Tensor, float], None] | None = None,
) -> NeuralResult:
    config = config or NeuralConfig()
    device = torch.device('cpu')
    if model is None:
        model = RuleEmbeddingScorer(num_rules, config.embedding_dim, config.hidden_dim,
                                    config.num_layers, config.dropout, rule_to_label, device)
    else:
        model = model.to(device)

    def objective():
        rule_log_probs = model.get_rule_log_probs()
        log_likelihood = compute_batch_log_likelihood(forests, rule_log_probs)
        loss = -log_likelihood
        if config.entropy_weight > 0:
            entropy = compute_entropy_regularization(rule_log_probs, model.label_groups)
            loss = loss - config.entropy_weight * entropy
        return log_likelihood, loss

    def report(iteration, ll):
        with torch.no_grad():
            log_probs = model.get_rule_log_probs()
        callback(iteration, log_probs, ll)

    return _train(config, model, list(model.parameters()), objective,
                  lambda: model.get_rule_log_probs().clone(), report if callback is not None else None)


def train_context_aware(
    forests: list[PackedForest],
    num_rules: int,
    node_embedding_dim: int,
    config: NeuralConfig | None = None,
    rule_to_label: Tensor | None = None,
    node_embeddings_list: list[dict[int, Tensor]] | None = None,
    encoder: nn.Module | None = None,
    callback: Callable[[int, nn.Module, float], None] | None = None,
) -> NeuralResult:
    # rule_log_probs are scored with the default node embedding; encoder is optimized, never called
    config = config or NeuralConfig()
    scorer = ContextAwareRuleScorer(
        num_rules=num_rules, node_embedding_dim=node_embedding_dim,
        rule_embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
        num_layers=config.num_layers, rule_to_label=rule_to_label, device=torch.device('cpu'))
    params = list(scorer.parameters())
    if encoder is not None:
        params.extend(encoder.parameters())

    def objective():
        log_likelihood = compute_batch_log_likelihood_with_scorer(
            forests, scorer, node_embeddings_list)
        return log_likelihood, -log_likelihood

    return _train(config, scorer, params, objective, scorer._context_free_log_probs,
                  (lambda iteration, ll: callback(iteration, scorer, ll)) if callback is not None else None)


def compare_methods(
    forests: list[PackedForest],
    num_rules: int,
    rule_to_label: Tensor | None = None,
    em_iterations: int = 100,
    neural_iterations: int = 100,
    neural_lr: float = 0.01,
    verbose: bool = False,
) -> dict:
    from .em import EMConfig, train_em
    from .gradient import GradientConfig, train_gradient

    runs = {
        'em': train_em(forests=forests, num_rules=num_rules, rule_to_label=rule_to_label,
                       config=EMConfig(max_iterations=em_iterations, verbose=verbose)),
        'gradient': train_gradient(
            forests=forests, num_rules=num_rules, rule_to_label=rule_to_label,
            config=GradientConfig(max_iterations=neural_iterations, verbose=verbose)),
        'neural': train_neural(
            forests=forests, num_rules=num_rules, rule_to_label=rule_to_label,
            config=NeuralConfig(max_iterations=neural_iterations, learning_rate=neural_lr,
                                verbose=verbose)),
    }
    return {name: {'log_probs': r.rule_log_probs, 'final_ll': r.log_likelihood_history[-1],
                   'iterations': r.num_iterations, 'converged': r.converged}
            for name, r in runs.items()}

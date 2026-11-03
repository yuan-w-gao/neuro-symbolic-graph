import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..forest import PackedForest
from ..inference import expected_counts, inside, sample, viterbi
from .gradient import compute_batch_log_likelihood as compute_log_likelihood


@dataclass
class ObjectiveResult:
    rule_log_probs: Tensor
    objective_history: list[float]
    num_iterations: int
    converged: bool
    extra_info: dict | None = None


def compute_derivation_entropy(forest: PackedForest, rule_log_probs: Tensor) -> Tensor:
    """H(derivation | graph) = log P(graph) - sum_r E[c_r] log p_r."""
    if not forest.roots:
        return torch.tensor(0.0, device=rule_log_probs.device)
    inside_values, total_log_prob = inside(forest, rule_log_probs)
    counts = expected_counts(forest, rule_log_probs, inside_values.detach())
    return total_log_prob - (counts * rule_log_probs).sum()


def compute_batch_entropy(forests: list[PackedForest], rule_log_probs: Tensor) -> Tensor:
    """Mean derivation entropy over the forests that parse."""
    total = torch.tensor(0.0, device=rule_log_probs.device)
    count = 0
    for forest in forests:
        if forest.roots:
            total = total + compute_derivation_entropy(forest, rule_log_probs)
            count += 1
    if count > 0:
        return total / count
    return torch.tensor(0.0, device=rule_log_probs.device)


def compute_mdl_objective(forests: list[PackedForest], rule_log_probs: Tensor,
                          complexity_weight: float = 0.1) -> Tensor:
    """-log P(data) minus complexity_weight times the entropy of the rule distribution."""
    neg_ll = -compute_log_likelihood(forests, rule_log_probs)
    complexity = (torch.exp(rule_log_probs) * rule_log_probs).sum()
    return neg_ll + complexity_weight * complexity


def optimize_objective(
    forests: list[PackedForest],
    num_rules: int,
    objective_fn: Callable[[list[PackedForest], Tensor], Tensor],
    max_iterations: int = 100,
    learning_rate: float = 0.1,
    convergence_threshold: float = 1e-6,
    minimize: bool = True,
    verbose: bool = True,
    random_init: bool = False,
) -> ObjectiveResult:
    """Adam on one global softmax over rules."""
    if random_init:
        logits = torch.randn(num_rules, device=torch.device('cpu'), requires_grad=True)
    else:
        logits = torch.zeros(num_rules, device=torch.device('cpu'), requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)

    def result(iterations, converged):
        with torch.no_grad():
            log_probs = F.log_softmax(logits, dim=0).clone()
        return ObjectiveResult(log_probs, history, iterations, converged)

    history = []
    prev = math.inf
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        value = objective_fn(forests, F.log_softmax(logits, dim=0))
        (value if minimize else -value).backward()
        optimizer.step()

        value = value.item()
        history.append(value)
        if verbose:
            print(f"Iteration {iteration + 1}: objective = {value:.4f}")
        if abs(value - prev) < convergence_threshold:
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            return result(iteration + 1, True)
        prev = value

    if verbose:
        print(f"Did not converge after {max_iterations} iterations")
    return result(max_iterations, False)


def train_mle(forests: list[PackedForest], num_rules: int, **kwargs) -> ObjectiveResult:
    return optimize_objective(forests, num_rules, compute_log_likelihood, minimize=False, **kwargs)


def train_min_entropy(forests: list[PackedForest], num_rules: int, **kwargs) -> ObjectiveResult:
    # Uniform weights are a stationary point of the entropy, so start from random logits.
    kwargs.setdefault('random_init', True)
    return optimize_objective(forests, num_rules, compute_batch_entropy, minimize=True, **kwargs)


def train_constrained_min_entropy(forests: list[PackedForest], num_rules: int,
                                  likelihood_weight: float = 10.0, **kwargs) -> ObjectiveResult:
    def objective(forests, rule_log_probs):
        ll = compute_log_likelihood(forests, rule_log_probs)
        entropy = compute_batch_entropy(forests, rule_log_probs)
        return -likelihood_weight * ll + entropy

    kwargs.setdefault('random_init', True)
    return optimize_objective(forests, num_rules, objective, minimize=True, **kwargs)


def train_max_entropy(forests: list[PackedForest], num_rules: int,
                      likelihood_weight: float = 1.0, **kwargs) -> ObjectiveResult:
    def objective(forests, rule_log_probs):
        ll = compute_log_likelihood(forests, rule_log_probs)
        entropy = compute_batch_entropy(forests, rule_log_probs)
        return likelihood_weight * ll + entropy

    return optimize_objective(forests, num_rules, objective, minimize=False, **kwargs)


def train_mdl(forests: list[PackedForest], num_rules: int,
              complexity_weight: float = 0.1, **kwargs) -> ObjectiveResult:
    def objective(forests, rule_log_probs):
        return compute_mdl_objective(forests, rule_log_probs, complexity_weight)

    return optimize_objective(forests, num_rules, objective, minimize=True, **kwargs)


def train_contrastive(forests: list[PackedForest], num_rules: int, margin: float = 1.0,
                      num_negative_samples: int = 5, **kwargs) -> ObjectiveResult:
    """Margin loss pushing the Viterbi derivation above sampled ones."""
    def objective(forests, rule_log_probs):
        device = rule_log_probs.device
        total_loss = torch.tensor(0.0, device=device)
        for forest in forests:
            if not forest.roots:
                continue
            best_log_prob, _ = viterbi(forest, rule_log_probs)
            for derivation in sample(forest, rule_log_probs, num_samples=num_negative_samples):
                sample_log_prob = torch.tensor(0.0, device=device)
                for _, edge_idx in derivation:
                    rule_id = forest.hyperedges[edge_idx].rule_id
                    sample_log_prob = sample_log_prob + rule_log_probs[rule_id]
                total_loss = total_loss + F.relu(margin - (best_log_prob - sample_log_prob))
        return total_loss

    return optimize_objective(forests, num_rules, objective, minimize=True, **kwargs)


def analyze_forest(forest: PackedForest, rule_log_probs: Tensor) -> dict:
    if not forest.roots:
        return {'valid': False}
    _, total_log_prob = inside(forest, rule_log_probs)
    entropy = compute_derivation_entropy(forest, rule_log_probs)
    best_log_prob, best_derivation = viterbi(forest, rule_log_probs)
    counts = expected_counts(forest, rule_log_probs)
    return {
        'valid': True,
        'log_prob': total_log_prob.item(),
        'entropy': entropy.item(),
        'num_nodes': len(forest.nodes),
        'num_edges': len(forest.hyperedges),
        'best_log_prob': best_log_prob.item(),
        'best_derivation_length': len(best_derivation),
        'best_rules': [forest.hyperedges[e].rule_id for _, e in best_derivation],
        'expected_counts': counts.detach(),
        'num_derivations': forest.num_derivations(),
    }


def compare_objectives(forests: list[PackedForest], num_rules: int, max_iterations: int = 100,
                       verbose: bool = False) -> dict[str, ObjectiveResult]:
    results = {}
    for name, train in (("mle", train_mle), ("min_entropy", train_min_entropy),
                        ("max_entropy", train_max_entropy), ("mdl", train_mdl)):
        print(f"Training {name}...")
        results[name] = train(forests, num_rules, max_iterations=max_iterations, verbose=verbose)
    return results

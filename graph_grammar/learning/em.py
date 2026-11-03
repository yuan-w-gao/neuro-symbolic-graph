from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ..forest import PackedForest
from ..inference import expected_counts, inside


@dataclass
class EMConfig:
    max_iterations: int = 100
    relative_threshold: float = 0.001
    patience: int = 10
    smoothing: float = 1e-10
    verbose: bool = True
    convergence_threshold: float | None = None  # accepted from old callers, unused


@dataclass
class EMResult:
    rule_log_probs: Tensor
    log_likelihood_history: list[float]
    converged: bool
    num_iterations: int


class EarlyStop:
    """Called once per iteration with the log-likelihood; returns why to stop, or None."""

    def __init__(self, relative_threshold: float, patience: int):
        self.threshold = relative_threshold
        self.patience = patience
        self.prev = float('-inf')
        self.best = float('-inf')
        self.stale = 0

    def __call__(self, ll: float) -> str | None:
        relative = abs(ll - self.prev) / (abs(ll) + 1e-8)
        self.prev = ll
        if relative < self.threshold:
            return f"relative improvement {relative:.6f}"
        if ll > self.best + abs(self.best) * self.threshold:
            self.best = ll
            self.stale = 0
            return None
        self.stale += 1
        if self.stale >= self.patience:
            return f"no improvement for {self.patience} iterations"
        return None


def _accumulate_counts(forests, rule_log_probs, smoothing):
    total_counts = torch.zeros(len(rule_log_probs), device=rule_log_probs.device) + smoothing
    total_ll = 0.0
    for forest in forests:
        if not forest.roots:
            continue
        total_counts += expected_counts(forest, rule_log_probs)
        _, log_prob = inside(forest, rule_log_probs)
        total_ll += log_prob.item()
    return total_counts, total_ll


def em_step(forests: list[PackedForest], rule_log_probs: Tensor,
            smoothing: float = 1e-10) -> tuple[Tensor, float]:
    """One EM step with a single global normalization."""
    counts, ll = _accumulate_counts(forests, rule_log_probs, smoothing)
    total = counts.sum()
    if total > 0:
        return torch.log(counts / total), ll
    return rule_log_probs, ll


def em_step_by_label(forests: list[PackedForest], rule_log_probs: Tensor,
                     rule_to_label: Tensor, smoothing: float = 1e-10) -> tuple[Tensor, float]:
    """One EM step, normalizing within each rule_to_label group."""
    counts, ll = _accumulate_counts(forests, rule_log_probs, smoothing)
    num_rules = len(rule_log_probs)
    labels = [rule_to_label[r].item() for r in range(num_rules)]

    label_counts = torch.zeros(rule_to_label.max().item() + 1, device=rule_log_probs.device)
    for r, label in enumerate(labels):
        label_counts[label] += counts[r]

    new_log_probs = torch.zeros(num_rules, device=rule_log_probs.device)
    for r, label in enumerate(labels):
        if label_counts[label] > 0:
            new_log_probs[r] = torch.log(counts[r] / label_counts[label])
        else:
            new_log_probs[r] = rule_log_probs[r]
    return new_log_probs, ll


def train_em(
    forests: list[PackedForest],
    num_rules: int,
    config: EMConfig | None = None,
    rule_to_label: Tensor | None = None,
    initial_log_probs: Tensor | None = None,
    callback: Callable[[int, Tensor, float], None] | None = None,
) -> EMResult:
    config = config or EMConfig()
    if initial_log_probs is not None:
        rule_log_probs = initial_log_probs.clone()
    else:
        rule_log_probs = torch.zeros(num_rules) - torch.log(torch.tensor(float(num_rules)))

    stop = EarlyStop(config.relative_threshold, config.patience)
    history = []
    for iteration in range(config.max_iterations):
        if rule_to_label is not None:
            rule_log_probs, ll = em_step_by_label(
                forests, rule_log_probs, rule_to_label, config.smoothing)
        else:
            rule_log_probs, ll = em_step(forests, rule_log_probs, config.smoothing)
        history.append(ll)
        if config.verbose:
            print(f"Iteration {iteration + 1}: log-likelihood = {ll:.4f}")
        if callback is not None:
            callback(iteration, rule_log_probs, ll)

        reason = stop(ll)
        if reason:
            if config.verbose:
                print(f"Stopped after {iteration + 1} iterations ({reason})")
            return EMResult(rule_log_probs, history, True, iteration + 1)

    if config.verbose:
        print(f"Did not converge after {config.max_iterations} iterations")
    return EMResult(rule_log_probs, history, False, config.max_iterations)


def compute_perplexity(forests: list[PackedForest], rule_log_probs: Tensor) -> float:
    total, n = 0.0, 0
    for forest in forests:
        if forest.roots:
            total += inside(forest, rule_log_probs)[1].item()
            n += 1
    if n == 0:
        return float('inf')
    return torch.exp(torch.tensor(-total / n)).item()

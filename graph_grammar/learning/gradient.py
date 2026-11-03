from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from ..forest import PackedForest
from ..inference import inside
from .em import EarlyStop, EMConfig, train_em


@dataclass
class GradientConfig:
    max_iterations: int = 100
    relative_threshold: float = 0.001
    patience: int = 10
    learning_rate: float = 0.1
    optimizer: str = "adam"  # adam, sgd or lbfgs
    verbose: bool = True
    convergence_threshold: float | None = None  # accepted from old callers, unused


@dataclass
class GradientResult:
    rule_log_probs: Tensor
    log_likelihood_history: list[float]
    converged: bool
    num_iterations: int
    gradient_norm_history: list[float] | None = None


class RuleLogitParameterization(nn.Module):
    """Free logits, log-softmaxed within each rule_to_label group (one global group if None)."""

    def __init__(self, num_rules: int, rule_to_label: Tensor | None = None,
                 device: torch.device | None = None, initial_log_probs: Tensor | None = None):
        super().__init__()
        self.num_rules = num_rules
        self.device = device or torch.device('cpu')
        if rule_to_label is not None:
            self.rule_to_label = rule_to_label.to(self.device)
            self.num_labels = self.rule_to_label.max().item() + 1
        else:
            self.rule_to_label = torch.zeros(num_rules, dtype=torch.long, device=self.device)
            self.num_labels = 1

        groups = (torch.where(self.rule_to_label == label)[0] for label in range(self.num_labels))
        self.label_groups = [g for g in groups if len(g) > 0]

        if initial_log_probs is not None:
            logits = initial_log_probs.to(self.device).clone()
        else:
            logits = torch.zeros(num_rules, device=self.device)
        self.logits = nn.Parameter(logits)

    def get_rule_log_probs(self) -> Tensor:
        log_probs = torch.zeros(self.num_rules, device=self.device)
        for indices in self.label_groups:
            log_probs[indices] = torch.log_softmax(self.logits[indices], dim=0)
        return log_probs

    def forward(self) -> Tensor:
        return self.get_rule_log_probs()


def compute_batch_log_likelihood(forests: list[PackedForest], rule_log_probs: Tensor) -> Tensor:
    total = torch.tensor(0.0, device=rule_log_probs.device)
    for forest in forests:
        if forest.roots:
            total = total + inside(forest, rule_log_probs)[1]
    return total


def _make_optimizer(config, params):
    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate)
    if name == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate)
    if name == "lbfgs":
        # one LBFGS iteration per step(); the outer loop counts iterations
        return torch.optim.LBFGS(params, lr=config.learning_rate, max_iter=1,
                                 line_search_fn="strong_wolfe")
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def train_gradient(
    forests: list[PackedForest],
    num_rules: int,
    config: GradientConfig | None = None,
    rule_to_label: Tensor | None = None,
    initial_log_probs: Tensor | None = None,
    callback: Callable[[int, Tensor, float], None] | None = None,
) -> GradientResult:
    config = config or GradientConfig()
    model = RuleLogitParameterization(num_rules, rule_to_label, torch.device('cpu'),
                                      initial_log_probs)
    optimizer = _make_optimizer(config, model.parameters())
    lbfgs = config.optimizer.lower() == "lbfgs"

    def final_log_probs():
        with torch.no_grad():
            return model.get_rule_log_probs().clone()

    def closure():
        optimizer.zero_grad()
        loss = -compute_batch_log_likelihood(forests, model.get_rule_log_probs())
        loss.backward()
        return loss

    stop = EarlyStop(config.relative_threshold, config.patience)
    history = []
    grad_norms = []
    for iteration in range(config.max_iterations):
        if lbfgs:
            optimizer.step(closure)
            with torch.no_grad():
                ll = compute_batch_log_likelihood(forests, model.get_rule_log_probs()).item()
            grad_norm = model.logits.grad.norm().item()
        else:
            loss = closure()
            grad_norm = model.logits.grad.norm().item()
            optimizer.step()
            ll = -loss.item()

        history.append(ll)
        grad_norms.append(grad_norm)
        if config.verbose:
            print(f"Iteration {iteration + 1}: log-likelihood = {ll:.4f}, grad_norm = {grad_norm:.6f}")
        if callback is not None:
            with torch.no_grad():
                log_probs = model.get_rule_log_probs()
            callback(iteration, log_probs, ll)

        reason = stop(ll)
        if reason:
            if config.verbose:
                print(f"Stopped after {iteration + 1} iterations ({reason})")
            return GradientResult(final_log_probs(), history, True, iteration + 1, grad_norms)

    if config.verbose:
        print(f"Did not converge after {config.max_iterations} iterations")
    return GradientResult(final_log_probs(), history, False, config.max_iterations, grad_norms)


@dataclass
class ComparisonResult:
    em_log_probs: Tensor
    gradient_log_probs: Tensor
    em_final_ll: float
    gradient_final_ll: float
    em_iterations: int
    gradient_iterations: int
    log_prob_difference: float = field(default=0.0)

    def __post_init__(self):
        self.log_prob_difference = (
            (self.em_log_probs - self.gradient_log_probs).pow(2).sum().sqrt().item())


def compare_training_methods(
    forests: list[PackedForest],
    num_rules: int,
    em_config: EMConfig | None = None,
    gradient_config: GradientConfig | None = None,
    rule_to_label: Tensor | None = None,
    initial_log_probs: Tensor | None = None,
) -> ComparisonResult:
    em = train_em(forests, num_rules, em_config or EMConfig(verbose=False),
                  rule_to_label=rule_to_label, initial_log_probs=initial_log_probs)
    gd = train_gradient(forests, num_rules, gradient_config or GradientConfig(verbose=False),
                        rule_to_label=rule_to_label, initial_log_probs=initial_log_probs)
    return ComparisonResult(
        em_log_probs=em.rule_log_probs,
        gradient_log_probs=gd.rule_log_probs,
        em_final_ll=em.log_likelihood_history[-1],
        gradient_final_ll=gd.log_likelihood_history[-1],
        em_iterations=em.num_iterations,
        gradient_iterations=gd.num_iterations,
    )

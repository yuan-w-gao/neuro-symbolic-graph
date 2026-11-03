import torch
from torch import Tensor

from ..forest import PackedForest
from ..inference import expected_counts, inside


def compute_forest_entropy(forest: PackedForest, rule_log_probs: Tensor) -> float:
    if not forest.roots:
        return 0.0
    inside_values, log_z = inside(forest, rule_log_probs)
    counts = expected_counts(forest, rule_log_probs, inside_values)
    # H(d | forest) = log Z - E[sum_r count(r) log P(r)], clamped at 0 against rounding.
    entropy = -(counts * rule_log_probs).sum().item() + log_z.item()
    return max(0.0, entropy)


def compute_batch_entropy(forests: list[PackedForest], rule_log_probs: Tensor,
                          normalize: bool = True) -> float:
    total = 0.0
    count = 0
    for forest in forests:
        if forest.roots:
            total += compute_forest_entropy(forest, rule_log_probs)
            count += 1
    if normalize and count:
        return total / count
    return total


def compute_perplexity_from_entropy(entropy: float) -> float:
    return torch.exp(torch.tensor(entropy)).item()

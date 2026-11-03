import math
import re
from collections import Counter

from torch import Tensor

# The generator emits morphology as $s$...$/s$ blocks, which have no counterpart in the lemma reference.
_MORPH_BLOCK_RE = re.compile(r'\$s\$.*?\$/s\$')


def tokenize_sentence(sentence: str) -> list[str]:
    return _MORPH_BLOCK_RE.sub('', sentence).strip().lower().split()


def reference_from_graph(graph) -> str:
    # BLEU is scored against the lemma sequence; the surface sentence is only a fallback.
    return getattr(graph, 'lemma_sequence', None) or getattr(graph, 'sentence', '')


def set_rule_weights(manager, rule_log_probs: Tensor) -> bool:
    weights = rule_log_probs.detach().cpu().tolist()
    # Write through iter_hrgs: some pyshrg builds return copies from get_shrg(i), so writes to it are lost.
    written = 0
    for hrg, weight in zip(manager.iter_hrgs(), weights):
        hrg.log_rule_weight = weight
        written += 1
    if not written:
        return False

    readback = next(iter(manager.iter_hrgs())).log_rule_weight
    if abs(readback - weights[0]) > 1e-4:
        raise RuntimeError(
            f"set_rule_weights: wrote log_rule_weight={weights[0]:.6f} on rule 0 "
            f"but readback is {readback:.6f} — pyshrg binding may be read-only. "
            "BLEU and generation will use default grammar weights."
        )
    return True


def generate_sentence(context, graph) -> str:
    try:
        if context.parse(graph).name != 'kNone' or not context.generate():
            return ""
        return context.sentence
    except Exception:
        return ""


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_ngram_precision(candidate: list[str], reference: list[str], n: int) -> tuple[int, int]:
    """(clipped matches, candidate n-grams)."""
    if len(candidate) < n:
        return 0, 0
    reference_ngrams = _ngrams(reference, n)
    clipped = sum(min(count, reference_ngrams[ngram]) for ngram, count in _ngrams(candidate, n).items())
    return clipped, len(candidate) - n + 1


def _bleu(counts, candidate_len, reference_len):
    # Unsmoothed: any order without a match, e.g. every 4-gram of a sentence under 4 tokens, scores 0.
    if any(clipped == 0 for clipped, _ in counts):
        return 0.0
    log_precision = sum(math.log(clipped / total) for clipped, total in counts) / len(counts)
    brevity = 1.0 if candidate_len >= reference_len else math.exp(1 - reference_len / candidate_len)
    return brevity * math.exp(log_precision)


def compute_bleu_score(candidate: list[str], reference: list[str], max_n: int = 4) -> float:
    if not candidate or not reference:
        return 0.0
    counts = [compute_ngram_precision(candidate, reference, n) for n in range(1, max_n + 1)]
    return _bleu(counts, len(candidate), len(reference))


def _generate_tokens(context, graph, gold_sentence):
    return tokenize_sentence(generate_sentence(context, graph)), tokenize_sentence(gold_sentence)


def compute_bleu(manager, context, graphs: list, gold_sentences: list[str],
                 rule_log_probs: Tensor, max_n: int = 4) -> float:
    set_rule_weights(manager, rule_log_probs)

    if graphs:
        first = generate_sentence(context, graphs[0])
        shifted = rule_log_probs.detach().clone()
        shifted[: len(shifted) // 2] -= 25.0
        set_rule_weights(manager, shifted)
        second = generate_sentence(context, graphs[0])
        set_rule_weights(manager, rule_log_probs)
        if first and first == second:
            raise RuntimeError(
                "compute_bleu: the generator ignored a 25-nat change to half the rule "
                "weights, so BLEU here would score the untrained grammar regardless of "
                "the model. Use parser/build/comprehensive_eval for weight-sensitive "
                "BLEU. See tasks/todo.md item 54(b)."
            )

    clipped = [0] * max_n
    totals = [0] * max_n
    candidate_len = reference_len = 0
    for graph, gold_sentence in zip(graphs, gold_sentences):
        candidate, reference = _generate_tokens(context, graph, gold_sentence)
        if not candidate or not reference:
            continue
        candidate_len += len(candidate)
        reference_len += len(reference)
        for n in range(1, max_n + 1):
            c, t = compute_ngram_precision(candidate, reference, n)
            clipped[n - 1] += c
            totals[n - 1] += t

    if candidate_len == 0 or reference_len == 0:
        return 0.0
    return _bleu(list(zip(clipped, totals)), candidate_len, reference_len)


def compute_sentence_bleu_scores(manager, context, graphs: list, gold_sentences: list[str],
                                 rule_log_probs: Tensor, max_n: int = 4) -> list[float]:
    set_rule_weights(manager, rule_log_probs)
    return [compute_bleu_score(*_generate_tokens(context, graph, gold), max_n)
            for graph, gold in zip(graphs, gold_sentences)]

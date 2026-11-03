import pickle
from collections import Counter

from torch import Tensor

from ..forest import PackedForest
from ..inference import viterbi


def _data_lines(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                yield line


def _ints(text):
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def load_gold_rule_indices_from_file(indices_file: str) -> dict[int, list[int]]:
    """Reads `graph_id:count:i,j,...` (export_gold_indices.py) or `loop_index<TAB>i,j,...` lines."""
    result = {}
    for line in _data_lines(indices_file):
        if ':' in line:
            parts = line.split(':')
            if len(parts) < 3:
                continue
            key, indices = parts[0], parts[2]
        elif '\t' in line:
            key, indices = line.split('\t')[:2]
        else:
            continue
        try:
            result[int(key)] = _ints(indices)
        except ValueError:
            continue
    return result


def load_gold_edge_indices_from_file(indices_file: str) -> dict[int, list[frozenset[int]]]:
    """Reads `loop_index<TAB>count<TAB>a,b;c,d;...` lines."""
    result = {}
    for line in _data_lines(indices_file):
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        try:
            edge_sets = [frozenset(_ints(chunk)) for chunk in parts[2].split(';')]
            result[int(parts[0])] = [s for s in edge_sets if s]
        except ValueError:
            continue
    return result


def load_gold_derivations_raw(derivations_file: str) -> dict:
    with open(derivations_file, 'rb') as f:
        return pickle.load(f)


def load_gold_derivations(derivations_file: str) -> dict[str, list[tuple[int, frozenset]]]:
    return {graph_id: [(item[0], frozenset(item[1])) for item in steps
                       if len(item) >= 2 and isinstance(item[1], (frozenset, set))]
            for graph_id, steps in load_gold_derivations_raw(derivations_file).items()}


def load_gold_rule_ids(derivations_file: str) -> dict[str, list[int]]:
    return {graph_id: [item[0] for item in steps]
            for graph_id, steps in load_gold_derivations(derivations_file).items()}


def convert_gold_edge_tuples_to_indices(gold_edge_tuples, edges_map: dict) -> frozenset[int]:
    return frozenset(edges_map[t] for t in gold_edge_tuples if t in edges_map)


def _gold_index_sets(tuple_sets, edges_map):
    out = set()
    for tuples in tuple_sets:
        if isinstance(tuples, (set, frozenset)):
            indices = convert_gold_edge_tuples_to_indices(tuples, edges_map)
            if indices:
                out.add(indices)
    return out


def gold_fragments(gold_steps, edges_map: dict) -> set[frozenset[int]]:
    out = set()
    for _step, tuples in gold_steps:
        indices = convert_gold_edge_tuples_to_indices(tuples, edges_map)
        if indices:
            out.add(indices)
    return out


def _consume(counter, key):
    if key in counter:
        counter[key] -= 1
        if counter[key] == 0:
            del counter[key]


def find_gold_derivation_in_forest(forest: PackedForest, gold_rule_indices: list[int]) -> list | None:
    """A derivation using exactly the multiset gold_rule_indices, as (node_idx, edge_idx) pairs."""
    if not forest.roots or not gold_rule_indices:
        return None
    gold = Counter(gold_rule_indices)

    def search(node_idx, remaining):
        for edge_idx in forest.nodes[node_idx].incoming_edges:
            edge = forest.hyperedges[edge_idx]
            if remaining[edge.rule_id] <= 0:
                continue
            if not edge.child_nodes:
                return [(node_idx, edge_idx)]
            left = remaining.copy()
            _consume(left, edge.rule_id)
            derivation = [(node_idx, edge_idx)]
            for child_idx in edge.child_nodes:
                sub = search(child_idx, left)
                if sub is None:
                    break
                derivation.extend(sub)
                for _, child_edge in sub:
                    _consume(left, forest.hyperedges[child_edge].rule_id)
            else:
                return derivation
        return None

    for root_idx in forest.roots:
        derivation = search(root_idx, gold)
        if derivation is not None and Counter(forest.hyperedges[e].rule_id for _, e in derivation) == gold:
            return derivation
    return None


def _span(forest, node_idx):
    return set(forest.nodes[node_idx].edge_indices or ())


def _introduced_edges(forest, node_idx, edge_idx):
    covered = set()
    for child_idx in forest.hyperedges[edge_idx].child_nodes:
        covered |= _span(forest, child_idx)
    return _span(forest, node_idx) - covered


def derivation_spans(forest: PackedForest, derivation) -> set[frozenset[int]]:
    """Every graph edge beneath each step: the classic constituents."""
    return {frozenset(span) for node_idx, _ in derivation
            if (span := forest.nodes[node_idx].edge_indices or ())}


def derivation_fragments(forest: PackedForest, derivation) -> set[frozenset[int]]:
    """The edges each step introduces itself: its span minus its children's spans.

    Gold derivations in train.derivations.p store fragments, not spans; fragments
    partition the graph's edges.
    """
    fragments = set()
    for node_idx, edge_idx in derivation:
        local = _introduced_edges(forest, node_idx, edge_idx)
        if local:
            fragments.add(frozenset(local))
    return fragments


def find_gold_derivation_by_fragments(forest: PackedForest, gold) -> list | None:
    """A derivation in which every step introduces a gold fragment or nothing.

    Matching on rule indices does not work for train.derivations.p: the first field
    of each step there is a step ordinal, not a rule index.
    """
    allowed = set(gold)
    memo = {}

    def search(node_idx):
        if node_idx in memo:
            return memo[node_idx]
        memo[node_idx] = None  # cycle guard
        for edge_idx in forest.nodes[node_idx].incoming_edges:
            local = _introduced_edges(forest, node_idx, edge_idx)
            if local and frozenset(local) not in allowed:
                continue
            derivation = [(node_idx, edge_idx)]
            for child_idx in forest.hyperedges[edge_idx].child_nodes:
                sub = search(child_idx)
                if sub is None:
                    break
                derivation.extend(sub)
            else:
                memo[node_idx] = derivation
                return derivation
        return None

    for root_idx in forest.roots:
        found = search(root_idx)
        if found is not None:
            return found
    return None


def extract_edge_sets_from_gold_derivation(forest: PackedForest, gold_rule_indices: list[int]):
    derivation = find_gold_derivation_in_forest(forest, gold_rule_indices)
    if derivation is None:
        return None
    return derivation_spans(forest, derivation)


def extract_rule_ids_from_derivation(forest: PackedForest, derivation) -> list[int]:
    return [forest.hyperedges[edge_idx].rule_id for _, edge_idx in derivation]


def extract_edge_sets_from_derivation(forest: PackedForest, derivation) -> list[frozenset[int]]:
    return [frozenset(forest.nodes[node_idx].edge_indices or ()) for node_idx, _ in derivation]


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compute_f1(predicted: set, gold: set) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    correct = len(predicted & gold)
    precision = correct / len(predicted)
    recall = correct / len(gold)
    return precision, recall, _f1(precision, recall)


def _macro_average(scores):
    if not scores:
        return 0.0, 0.0, 0.0
    return tuple(sum(column) / len(scores) for column in zip(*scores))


def _macro_span_f1(forests_and_gold, rule_log_probs, skip_viterbi_errors=False):
    """Macro P/R/F1 of each forest's Viterbi spans against its gold spans."""
    scores = []
    for forest, gold in forests_and_gold:
        try:
            _, derivation = viterbi(forest, rule_log_probs)
        except Exception:
            if not skip_viterbi_errors:
                raise
            continue
        if derivation:
            scores.append(compute_f1(derivation_spans(forest, derivation), gold))
    return _macro_average(scores)


def _lookup_key(mapping, graph_id):
    for key in (graph_id, graph_id.split('/')[-1]):
        if key in mapping:
            return key
    return None


def compute_edge_set_f1(forests: list[PackedForest], graph_ids: list[str], gold_derivations_raw: dict,
                        edges_maps: dict, rule_log_probs: Tensor) -> tuple[float, float, float]:
    def forests_and_gold():
        for forest, graph_id in zip(forests, graph_ids):
            if not forest.roots:
                continue
            gold_key = _lookup_key(gold_derivations_raw, graph_id)
            map_key = _lookup_key(edges_maps, graph_id)
            if gold_key is None or map_key is None:
                continue
            steps = gold_derivations_raw[gold_key]
            yield forest, _gold_index_sets((s[1] for s in steps if len(s) >= 2), edges_maps[map_key])

    return _macro_span_f1(forests_and_gold(), rule_log_probs)


def compute_edge_set_f1_single(forest: PackedForest, graph_id: str, gold_edge_tuples_list: list,
                               edges_map: dict, rule_log_probs: Tensor) -> tuple[float, float, float]:
    if not forest.roots:
        return 0.0, 0.0, 0.0
    gold = _gold_index_sets(gold_edge_tuples_list, edges_map)
    _, derivation = viterbi(forest, rule_log_probs)
    if not derivation:
        return 0.0, 0.0, 0.0
    return compute_f1(derivation_spans(forest, derivation), gold)


def compute_edge_set_f1_from_rule_indices(forests: list[PackedForest], gold_rule_indices: dict[int, list[int]],
                                          rule_log_probs: Tensor) -> tuple[float, float, float]:
    """Gold spans come from locating the gold rule indices in each forest (forests indexed by loop index)."""
    def forests_and_gold():
        for loop_idx, forest in enumerate(forests):
            if not forest.roots or not gold_rule_indices.get(loop_idx):
                continue
            gold = extract_edge_sets_from_gold_derivation(forest, gold_rule_indices[loop_idx])
            if gold is not None:
                yield forest, gold

    return _macro_span_f1(forests_and_gold(), rule_log_probs, skip_viterbi_errors=True)


def compute_edge_set_f1_from_precomputed(forests: list[PackedForest],
                                         gold_edge_indices: dict[int, list[frozenset[int]]],
                                         rule_log_probs: Tensor) -> tuple[float, float, float]:
    forests_and_gold = ((forest, set(gold_edge_indices[loop_idx]))
                        for loop_idx, forest in enumerate(forests)
                        if forest.roots and gold_edge_indices.get(loop_idx))
    return _macro_span_f1(forests_and_gold, rule_log_probs, skip_viterbi_errors=True)


def _rule_id_sets(forests, graph_ids, gold_derivations, rule_log_probs):
    """(predicted, gold) rule-id sets per forest with a gold derivation and a Viterbi parse."""
    for forest, graph_id in zip(forests, graph_ids):
        suffix = graph_id.split('/')[-1]
        if not forest.roots or suffix not in gold_derivations:
            continue
        _, derivation = viterbi(forest, rule_log_probs)
        if derivation:
            yield (set(extract_rule_ids_from_derivation(forest, derivation)),
                   {item[0] for item in gold_derivations[suffix]})


def compute_parsing_f1(forests: list[PackedForest], graph_ids: list[str], gold_derivations: dict,
                       rule_log_probs: Tensor) -> tuple[float, float, float]:
    """Macro P/R/F1 of Viterbi rule ids against gold rule ids."""
    return _macro_average([compute_f1(predicted, gold) for predicted, gold
                           in _rule_id_sets(forests, graph_ids, gold_derivations, rule_log_probs)])


def compute_micro_f1(forests: list[PackedForest], graph_ids: list[str], gold_derivations: dict,
                     rule_log_probs: Tensor) -> tuple[float, float, float]:
    num_predicted = num_gold = num_correct = 0
    for predicted, gold in _rule_id_sets(forests, graph_ids, gold_derivations, rule_log_probs):
        num_predicted += len(predicted)
        num_gold += len(gold)
        num_correct += len(predicted & gold)
    if num_predicted == 0 and num_gold == 0:
        return 1.0, 1.0, 1.0
    precision = num_correct / num_predicted if num_predicted else 0.0
    recall = num_correct / num_gold if num_gold else 0.0
    return precision, recall, _f1(precision, recall)

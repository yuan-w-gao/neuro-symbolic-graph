import json
import os
import pickle
import random
from dataclasses import asdict, dataclass


@dataclass
class GraphEntry:
    sentence_id: str
    sentence: str
    lemma_sequence: str
    nodes: list[str]
    root_count: int
    edges: list[str]


@dataclass
class SplitStatistics:
    total_graphs: int
    train_graphs: int
    test_graphs: int
    total_rules: int
    train_rules: int
    test_rules: int
    shared_rules: int
    train_only_rules: int
    test_only_rules: int
    rule_overlap_ratio: float
    unseen_rule_ratio: float

    def __str__(self) -> str:
        return (
            f"Split Statistics:\n"
            f"  Graphs: {self.train_graphs} train, {self.test_graphs} test "
            f"({100*self.test_graphs/self.total_graphs:.1f}% test)\n"
            f"  Rules: {self.total_rules} total, {self.train_rules} in train, "
            f"{self.test_rules} in test\n"
            f"  Overlap: {self.shared_rules} shared rules "
            f"({100*self.rule_overlap_ratio:.1f}% of test rules seen in train)\n"
            f"  Unseen: {self.test_only_rules} rules in test only "
            f"({100*self.unseen_rule_ratio:.1f}% unseen)"
        )


@dataclass
class GraphSplit:
    train_graphs: list[GraphEntry]
    test_graphs: list[GraphEntry]
    train_indices: list[int]
    test_indices: list[int]
    train_rule_indices: set[int]
    test_rule_indices: set[int]
    statistics: SplitStatistics


def _skip_blank(lines, idx):
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx


def load_graphs_file(filepath: str) -> list[GraphEntry]:
    with open(filepath, encoding='utf-8') as f:
        lines = f.read().split('\n')
    idx = _skip_blank(lines, 0)
    total = int(lines[idx])
    idx += 1

    graphs = []
    for _ in range(total):
        idx = _skip_blank(lines, idx)
        if idx >= len(lines):
            break
        sentence_id, sentence, lemma_sequence = lines[idx], lines[idx + 1], lines[idx + 2]
        num_nodes = int(lines[idx + 3])
        idx += 4
        nodes = [lines[i] for i in range(idx, idx + num_nodes)]
        idx += num_nodes
        root_count = int(lines[idx])
        num_edges = int(lines[idx + 1])
        idx += 2
        edges = [lines[i] for i in range(idx, idx + num_edges)]
        idx += num_edges
        graphs.append(GraphEntry(sentence_id, sentence, lemma_sequence, nodes, root_count, edges))
    return graphs


def write_graphs_file(graphs: list[GraphEntry], filepath: str) -> None:
    lines = [str(len(graphs))]
    for g in graphs:
        lines += [g.sentence_id, g.sentence, g.lemma_sequence, str(len(g.nodes)), *g.nodes,
                  str(g.root_count), str(len(g.edges)), *g.edges, '']
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _leading_ints(items, allow_bare):
    ids = []
    for item in items:
        if isinstance(item, tuple) and item and isinstance(item[0], int):
            ids.append(item[0])
        elif allow_bare and isinstance(item, int):
            ids.append(item)
    return ids


def load_derivations_file(filepath: str) -> dict[str, list[int]]:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    derivations = {}
    for sent_id, deriv in data.items():
        rules = []
        if isinstance(deriv, list):
            rules = _leading_ints(deriv, allow_bare=True)
        elif isinstance(deriv, dict):
            if 'rules' in deriv:
                rules = list(deriv['rules'])
            elif 'derivation' in deriv:
                rules = _leading_ints(deriv['derivation'], allow_bare=False)
        derivations[sent_id] = rules
    return derivations


def load_split_indices(filepath: str) -> tuple[list[int], list[int]]:
    with open(filepath) as f:
        data = json.load(f)
    return data['train_indices'], data['test_indices']


def save_split_indices(train_indices: list[int], test_indices: list[int], filepath: str,
                       metadata: dict | None = None) -> None:
    data = {'train_indices': train_indices, 'test_indices': test_indices}
    if metadata:
        data['metadata'] = metadata
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def compute_rule_overlap(train_rule_indices: set[int], test_rule_indices: set[int],
                         total_rules: int | None = None) -> dict:
    shared = train_rule_indices & test_rule_indices
    test_only = test_rule_indices - train_rule_indices
    all_seen = train_rule_indices | test_rule_indices
    n_test = len(test_rule_indices)

    stats = {
        'train_rules': len(train_rule_indices),
        'test_rules': n_test,
        'shared_rules': len(shared),
        'train_only_rules': len(train_rule_indices - test_rule_indices),
        'test_only_rules': len(test_only),
        'all_seen_rules': len(all_seen),
        'overlap_ratio': len(shared) / n_test if n_test else 0.0,
        'unseen_ratio': len(test_only) / n_test if n_test else 0.0,
    }
    if total_rules is not None:
        stats['total_rules'] = total_rules
        stats['coverage_ratio'] = len(all_seen) / total_rules if total_rules > 0 else 0.0
    return stats


def _derivation_key(derivations, sent_id):
    if sent_id in derivations:
        return sent_id
    stripped = sent_id.split('/')[-1]
    return stripped if stripped in derivations else None


def create_train_test_split(
    graphs_file: str,
    derivations_file: str | None = None,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify_by: str | None = None,
) -> GraphSplit:
    graphs = load_graphs_file(graphs_file)
    indices = list(range(len(graphs)))
    random.seed(seed)
    random.shuffle(indices)
    n_test = int(len(graphs) * test_ratio)
    test_indices = sorted(indices[:n_test])
    train_indices = sorted(indices[n_test:])

    train_rules, test_rules = set(), set()
    total_rules = 0
    if derivations_file and os.path.exists(derivations_file):
        derivations = load_derivations_file(derivations_file)
        for split_indices, rules in ((train_indices, train_rules), (test_indices, test_rules)):
            for i in split_indices:
                key = _derivation_key(derivations, graphs[i].sentence_id)
                if key:
                    rules.update(derivations[key])
        all_rules = train_rules | test_rules
        total_rules = max(all_rules) + 1 if all_rules else 0

    overlap = compute_rule_overlap(train_rules, test_rules, total_rules)
    statistics = SplitStatistics(
        total_graphs=len(graphs),
        train_graphs=len(train_indices),
        test_graphs=len(test_indices),
        total_rules=total_rules,
        train_rules=overlap['train_rules'],
        test_rules=overlap['test_rules'],
        shared_rules=overlap['shared_rules'],
        train_only_rules=overlap['train_only_rules'],
        test_only_rules=overlap['test_only_rules'],
        rule_overlap_ratio=overlap['overlap_ratio'],
        unseen_rule_ratio=overlap['unseen_ratio'],
    )
    return GraphSplit(
        train_graphs=[graphs[i] for i in train_indices],
        test_graphs=[graphs[i] for i in test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
        train_rule_indices=train_rules,
        test_rule_indices=test_rules,
        statistics=statistics,
    )


def combine_graphs_files(input_files: list[str], output_file: str, prefix_ids: bool = True) -> int:
    all_graphs = []
    for i, filepath in enumerate(input_files):
        graphs = load_graphs_file(filepath)
        if prefix_ids:
            for g in graphs:
                g.sentence_id = f"src{i}_{g.sentence_id}"
        all_graphs.extend(graphs)
    write_graphs_file(all_graphs, output_file)
    return len(all_graphs)


def split_and_save(
    grammar_dir: str,
    output_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
    graphs_filename: str = "train.graphs.txt",
    derivations_filename: str = "train.derivations.p",
) -> GraphSplit:
    os.makedirs(output_dir, exist_ok=True)
    graphs_file = os.path.join(grammar_dir, graphs_filename)
    split = create_train_test_split(graphs_file, os.path.join(grammar_dir, derivations_filename),
                                    test_ratio=test_ratio, seed=seed)

    write_graphs_file(split.train_graphs, os.path.join(output_dir, "train.graphs.txt"))
    write_graphs_file(split.test_graphs, os.path.join(output_dir, "test.graphs.txt"))
    save_split_indices(split.train_indices, split.test_indices,
                       os.path.join(output_dir, "split_indices.json"),
                       metadata={'test_ratio': test_ratio, 'seed': seed,
                                 'source_graphs_file': graphs_file})
    with open(os.path.join(output_dir, "split_stats.json"), 'w') as f:
        json.dump(asdict(split.statistics), f, indent=2)
    return split

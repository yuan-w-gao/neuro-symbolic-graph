from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SyntheticEdge:
    label: str
    node_indices: list[int]
    is_terminal: bool = True


@dataclass
class CFGItem:
    label: str
    aligned_edge_idx: int = -1


@dataclass
class SyntheticRule:
    node_count: int
    edges: list[SyntheticEdge]
    external_nodes: list[int]
    cfg_label: str
    cfg_items: list[CFGItem]
    shrg_index: int = 0
    current_count: int = 1
    total_count: int = 1

    @property
    def arity(self) -> int:
        return len(self.external_nodes)

    @property
    def label_hash_key(self) -> tuple[str, int]:
        return (self.cfg_label, self.arity)

    def has_semantic_part(self) -> bool:
        return self.node_count > 0 and len(self.edges) > 0


class GrammarBuilder:

    def __init__(self):
        self.rules: list[SyntheticRule] = []

    def add_rule(self, rule: SyntheticRule) -> int:
        rule.shrg_index = len(self.rules)
        self.rules.append(rule)
        return rule.shrg_index

    def get_lhs_groups(self) -> dict[tuple[str, int], list[int]]:
        groups = {}
        for i, rule in enumerate(self.rules):
            groups.setdefault(rule.label_hash_key, []).append(i)
        return groups

    def get_rule_to_lhs_index(self) -> dict[int, int]:
        lhs_index = {lhs: i for i, lhs in enumerate(sorted(self.get_lhs_groups()))}
        return {i: lhs_index[rule.label_hash_key] for i, rule in enumerate(self.rules)}

    def validate(self) -> list[str]:
        issues = []
        for i, rule in enumerate(self.rules):
            for edge in rule.edges:
                ns = edge.node_indices
                if edge.is_terminal and len(ns) == 2 and ns[0] == ns[1]:
                    issues.append(f"Rule {i}: Self-loop detected on edge '{edge.label}'")
            for edge in rule.edges:
                for n in edge.node_indices:
                    if n < 0 or n >= rule.node_count:
                        issues.append(f"Rule {i}: Invalid node index {n} "
                                      f"(node_count={rule.node_count})")
            for n in rule.external_nodes:
                if n < 0 or n >= rule.node_count:
                    issues.append(f"Rule {i}: Invalid external node index {n}")
        return issues

    def write_mapping_file(self, path: str) -> None:
        issues = self.validate()
        if issues:
            raise ValueError("Grammar validation failed:\n" + "\n".join(issues))

        lines = [str(len(self.rules))]
        for rule in self.rules:
            has_sem = rule.has_semantic_part()
            lines.append("1" if has_sem else "0")
            if has_sem:
                lines.append(f"{rule.node_count} {len(rule.edges)}")
                for edge in rule.edges:
                    nodes = " ".join(map(str, edge.node_indices))
                    flag = "Y" if edge.is_terminal else "N"
                    lines.append(f"{edge.label} {len(edge.node_indices)} {nodes} {flag}")
                lines.append(f"{len(rule.external_nodes)} {' '.join(map(str, rule.external_nodes))}")
            lines.append("1")
            lines.append(f"{rule.shrg_index} {rule.current_count} {rule.total_count}")
            items = " ".join(f"{item.label} {item.aligned_edge_idx}" for item in rule.cfg_items)
            lines.append(f"{rule.cfg_label} {len(rule.cfg_items)} {items}")
            lines.append("")
        _write_lines(path, lines)

    def __len__(self) -> int:
        return len(self.rules)

    def __getitem__(self, idx: int) -> SyntheticRule:
        return self.rules[idx]


@dataclass
class SyntheticGraph:
    sentence_id: str
    nodes: list[tuple[str, str]]
    edges: list[tuple[int, int, str]] = field(default_factory=list)
    sentence: str | None = None
    lemma_sequence: str | None = None
    top_index: int = 0

    def __post_init__(self):
        if self.sentence is None:
            self.sentence = " ".join(label for _, label in self.nodes) + " ."
        if self.lemma_sequence is None:
            self.lemma_sequence = " ".join(label for _, label in self.nodes)


class SyntheticGraphWriter:

    def write_graphs(self, path: str, graphs: list[SyntheticGraph]) -> None:
        lines = [str(len(graphs))]
        for graph in graphs:
            lines += [graph.sentence_id, graph.sentence, graph.lemma_sequence, str(len(graph.nodes))]
            # lemma = label; pos, sense, carg and the five properties are placeholders
            for idx, (node_id, label) in enumerate(graph.nodes):
                lines.append(f"{idx}\t{node_id}\t{label}\t{label}" + "\t#" * 8)
            lines += [str(graph.top_index), str(len(graph.edges))]
            lines += [f"{src}\t{tgt}\t{label}" for src, tgt, label in graph.edges]
            lines.append("")
        _write_lines(path, lines)

    def write_single_graph(self, path: str, graph: SyntheticGraph) -> None:
        self.write_graphs(path, [graph])


def _write_lines(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write("\n".join(lines))

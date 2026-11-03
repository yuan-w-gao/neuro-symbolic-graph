from dataclasses import dataclass

import networkx as nx

NUM_NODE_FIELDS = 12
LABEL_LEVELS = ("none", "pos", "predicate")


@dataclass
class EdsGraphRecord:
    sentence_id: str
    sentence: str
    lemma_sequence: str
    node_ids: list[str]
    node_labels: list[str]
    node_lemmas: list[str]
    node_pos: list[str]
    node_senses: list[str]
    node_cargs: list[str]
    node_props: list[list[str]]
    edges: list[tuple[int, int, str]]
    top_index: int

    @property
    def num_nodes(self) -> int:
        return len(self.node_labels)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


def load_eds_graphs(path: str, limit: int | None = None) -> list[EdsGraphRecord]:
    with open(path) as f:
        lines = f.read().split("\n")
    count = int(lines[0].strip())
    if limit is not None:
        count = min(count, limit)

    cursor = 1
    records = []
    for _ in range(count):
        sentence_id = lines[cursor].strip()
        sentence, lemma_sequence = lines[cursor + 1], lines[cursor + 2]
        node_count = int(lines[cursor + 3].strip())
        cursor += 4

        nodes = []
        for _ in range(node_count):
            line = lines[cursor]
            cursor += 1
            fields = line.split("\t")
            if len(fields) < NUM_NODE_FIELDS:
                raise ValueError(f"{path}: node line has {len(fields)} fields, expected "
                                 f"{NUM_NODE_FIELDS}: {line!r}")
            nodes.append(fields)

        # top_index and edge_count are read with `>>` in C++, so they may share a line
        header = []
        while len(header) < 2:
            header += lines[cursor].split()
            cursor += 1
        top_index, edge_count = int(header[0]), int(header[1])

        edges = []
        for _ in range(edge_count):
            src, tgt, label = lines[cursor].split("\t")
            cursor += 1
            edges.append((int(src), int(tgt), label))

        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1

        records.append(EdsGraphRecord(
            sentence_id=sentence_id,
            sentence=sentence,
            lemma_sequence=lemma_sequence,
            node_ids=[f[1] for f in nodes],
            node_labels=[f[2] for f in nodes],
            node_lemmas=[f[3] for f in nodes],
            node_pos=[f[4] for f in nodes],
            node_senses=[f[5] for f in nodes],
            node_cargs=[f[6] for f in nodes],
            node_props=[f[7:12] for f in nodes],
            edges=edges,
            top_index=top_index,
        ))
    return records


def format_eds_graph(record: EdsGraphRecord) -> str:
    out = [record.sentence_id, record.sentence, record.lemma_sequence, str(record.num_nodes)]
    for i in range(record.num_nodes):
        out.append("\t".join([str(i), record.node_ids[i], record.node_labels[i],
                              record.node_lemmas[i], record.node_pos[i], record.node_senses[i],
                              record.node_cargs[i], *record.node_props[i]]))
    out += [str(record.top_index), str(record.num_edges)]
    out += [f"{src}\t{tgt}\t{label}" for src, tgt, label in record.edges]
    return "\n".join(out)


def write_eds_graphs(path: str, records: list[EdsGraphRecord]) -> None:
    with open(path, "w") as f:
        f.write(f"{len(records)}\n")
        for record in records:
            f.write(format_eds_graph(record) + "\n\n")


def node_label_of(record: EdsGraphRecord, index: int, level: str) -> str:
    if level == "none":
        return "*"
    if level == "pos":
        return record.node_pos[index]
    if level == "predicate":
        return record.node_labels[index]
    raise ValueError(f"unknown label level {level!r}; expected one of {LABEL_LEVELS}")


def to_networkx(record: EdsGraphRecord, label_level: str = "pos", directed: bool = False) -> nx.Graph:
    graph = nx.DiGraph() if directed else nx.Graph()
    for i in range(record.num_nodes):
        graph.add_node(i, label=node_label_of(record, i, label_level))
    for src, tgt, label in record.edges:
        if src != tgt and not graph.has_edge(src, tgt):
            graph.add_edge(src, tgt, label=label)
    return graph


def records_to_networkx(records: list[EdsGraphRecord], label_level: str = "pos",
                        directed: bool = False) -> list[nx.Graph]:
    return [to_networkx(r, label_level, directed) for r in records]


def node_label_vocab(records: list[EdsGraphRecord], level: str) -> list[str]:
    return sorted({node_label_of(r, i, level) for r in records for i in range(r.num_nodes)})


def edge_label_vocab(records: list[EdsGraphRecord]) -> list[str]:
    return sorted({label for r in records for _, _, label in r.edges})


def split_by_ids(records: list[EdsGraphRecord], held_ids: list[str]
                 ) -> tuple[list[EdsGraphRecord], list[EdsGraphRecord]]:
    held = set(held_ids)
    return ([r for r in records if r.sentence_id in held],
            [r for r in records if r.sentence_id not in held])

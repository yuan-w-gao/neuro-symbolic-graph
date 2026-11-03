from pathlib import Path

from .generator import EdgeInfo, GeneratedGraph, NodeInfo


class GraphWriter:
    """Writes graphs in the train.graphs.txt format pyshrg reads."""

    @staticmethod
    def format_node(node: NodeInfo) -> str:
        return "\t".join([str(node.index), node.node_id, node.label, node.lemma, node.pos_tag,
                          node.sense, node.carg, *(node.properties or ["#"] * 5)])

    @staticmethod
    def format_edge(edge: EdgeInfo) -> str:
        return f"{edge.source}\t{edge.target}\t{edge.label}"

    @staticmethod
    def format_graph(graph: GeneratedGraph) -> str:
        lines = [
            graph.sentence_id,
            graph.sentence or "unknown .",
            graph.lemma_sequence or "unknown",
            str(graph.num_nodes),
            *(GraphWriter.format_node(node) for node in graph.nodes),
            str(graph.top_index),
            str(graph.num_edges),
            *(GraphWriter.format_edge(edge) for edge in graph.edges),
        ]
        return "\n".join(lines)

    @staticmethod
    def write_graphs(path: str, graphs: list[GeneratedGraph], append: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a" if append else "w") as f:
            if not append:
                f.write(f"{len(graphs)}\n")
            for graph in graphs:
                f.write(GraphWriter.format_graph(graph) + "\n\n")

    @staticmethod
    def read_graph_count(path: str) -> int:
        with open(path) as f:
            return int(f.readline().strip())

    @staticmethod
    def update_count(path: str, new_count: int):
        with open(path) as f:
            content = f.read()
        rest = content[content.index("\n") + 1:]
        with open(path, "w") as f:
            f.write(f"{new_count}\n{rest}")


class CompactGraphWriter:
    """Human-readable dump, for inspection."""

    @staticmethod
    def format_graph(graph: GeneratedGraph) -> str:
        lines = [f"Graph {graph.sentence_id}:",
                 f"  Sentence: {graph.sentence}",
                 f"  Nodes ({graph.num_nodes}):"]
        lines += [f"    {node.index}: {node.node_id} = {node.label}" for node in graph.nodes]
        lines.append(f"  Edges ({graph.num_edges}):")
        lines += [f"    {edge.source} --{edge.label}--> {edge.target}" for edge in graph.edges]
        if graph.rule_indices:
            lines.append(f"  Rules: {graph.rule_indices}")
        return "\n".join(lines)

    @staticmethod
    def write_graphs(path: str, graphs: list[GeneratedGraph]):
        with open(path, "w") as f:
            for graph in graphs:
                f.write(CompactGraphWriter.format_graph(graph) + "\n\n")

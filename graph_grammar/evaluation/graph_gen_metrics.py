import hashlib
from collections import Counter

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist


def degree_histogram(graph: nx.Graph) -> np.ndarray:
    if graph.number_of_nodes() == 0:
        return np.zeros(1)
    return np.asarray(nx.degree_histogram(graph), dtype=float)


def clustering_histogram(graph: nx.Graph, bins: int = 100) -> np.ndarray:
    if graph.number_of_nodes() == 0:
        return np.zeros(bins)
    hist, _ = np.histogram(list(nx.clustering(graph).values()), bins=bins, range=(0.0, 1.0))
    return hist.astype(float)


def spectrum_histogram(graph: nx.Graph, bins: int = 200) -> np.ndarray:
    if graph.number_of_nodes() == 0:
        return np.zeros(bins)
    hist, _ = np.histogram(nx.normalized_laplacian_spectrum(graph), bins=bins, range=(-1e-5, 2.0))
    return hist.astype(float)


DESCRIPTORS = {
    "degree": degree_histogram,
    "clustering": clustering_histogram,
    "spectrum": spectrum_histogram,
}

# The kernel widths GraphRNN / GRAN report each descriptor at.
SIGMAS = {"degree": 1.0, "clustering": 0.1, "spectrum": 1.0}


def _stack_normalized(histograms, width):
    matrix = np.zeros((len(histograms), width), dtype=float)
    for i, hist in enumerate(histograms):
        matrix[i, : len(hist)] = hist
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return matrix / totals


def _mean_kernel(a, b, sigma):
    """Mean of exp(-TV^2 / 2 sigma^2) over all pairs of rows."""
    tv = 0.5 * cdist(a, b, metric="cityblock")
    return float(np.exp(-(tv ** 2) / (2 * sigma ** 2)).mean())


def mmd2(histograms_a, histograms_b, sigma: float) -> float:
    if not histograms_a or not histograms_b:
        return float("nan")
    width = max(max(len(h) for h in histograms_a), max(len(h) for h in histograms_b))
    a = _stack_normalized(histograms_a, width)
    b = _stack_normalized(histograms_b, width)
    return _mean_kernel(a, a, sigma) + _mean_kernel(b, b, sigma) - 2 * _mean_kernel(a, b, sigma)


def structural_mmds(generated, reference, which=("degree", "clustering", "spectrum")) -> dict[str, float]:
    return {name: mmd2([DESCRIPTORS[name](g) for g in generated],
                       [DESCRIPTORS[name](g) for g in reference], SIGMAS[name])
            for name in which}


def _distribution(counter, support):
    total = sum(counter.values())
    if total == 0:
        return np.zeros(len(support))
    return np.array([counter.get(key, 0) / total for key in support], dtype=float)


def total_variation(counts_a: Counter, counts_b: Counter) -> float:
    support = sorted(set(counts_a) | set(counts_b))
    return float(0.5 * np.abs(_distribution(counts_a, support) - _distribution(counts_b, support)).sum())


def node_label_counts(graphs) -> Counter:
    return Counter(data.get("label", "*") for g in graphs for _, data in g.nodes(data=True))


def edge_label_counts(graphs) -> Counter:
    return Counter(data.get("label", "*") for g in graphs for _, _, data in g.edges(data=True))


def node_label_tv(generated, reference) -> float:
    return total_variation(node_label_counts(generated), node_label_counts(reference))


def edge_label_tv(generated, reference) -> float:
    return total_variation(edge_label_counts(generated), edge_label_counts(reference))


def _ascii_id(label):
    # Must be the same in every graph: uniqueness and novelty compare hashes across graphs.
    return hashlib.blake2b(str(label).encode("utf-8"), digest_size=8).hexdigest()


def _ascii_relabelled(graph, node_attr, edge_attr):
    # weisfeiler_lehman_graph_hash ASCII-encodes labels and real corpora have non-ASCII
    # predicates; WL only compares labels for equality, so a stable digest is exact.
    out = nx.Graph()
    for node, data in graph.nodes(data=True):
        out.add_node(node, **({"label": _ascii_id(data.get("label"))} if node_attr else {}))
    for src, tgt, data in graph.edges(data=True):
        out.add_edge(src, tgt, **({"label": _ascii_id(data.get("label"))} if edge_attr else {}))
    return out


def graph_hash(graph: nx.Graph, use_labels: bool = True) -> str:
    """Weisfeiler-Lehman hash over node and edge labels, so isomorphic graphs collide."""
    if graph.number_of_nodes() == 0:
        return "empty"
    node_attr = use_labels and all("label" in d for _, d in graph.nodes(data=True))
    edge_attr = (use_labels and graph.number_of_edges() > 0
                 and all("label" in d for _, _, d in graph.edges(data=True)))
    return nx.weisfeiler_lehman_graph_hash(
        _ascii_relabelled(graph, node_attr, edge_attr),
        node_attr="label" if node_attr else None,
        edge_attr="label" if edge_attr else None,
        iterations=3)


def uniqueness_and_novelty(generated, training=None, use_labels: bool = True) -> dict[str, float | None]:
    hashes = [graph_hash(g, use_labels) for g in generated]
    unique = len(set(hashes)) / len(hashes) if hashes else float("nan")
    novel = None
    if training is not None:
        seen = {graph_hash(g, use_labels) for g in training}
        novel = sum(h not in seen for h in hashes) / len(hashes) if hashes else float("nan")
    return {"uniqueness": unique, "novelty": novel}


def evaluate(generated, reference, with_labels: bool = True, training=None) -> dict[str, float | None]:
    """All reported metrics; the label columns are None for models without labels."""
    results = dict(structural_mmds(generated, reference))
    results["node_label_tv"] = node_label_tv(generated, reference) if with_labels else None
    results["edge_label_tv"] = edge_label_tv(generated, reference) if with_labels else None
    nodes = np.array([g.number_of_nodes() for g in generated], dtype=float)
    edges = np.array([g.number_of_edges() for g in generated], dtype=float)
    results["num_graphs"] = float(len(generated))
    results["mean_nodes"] = float(nodes.mean())
    results["mean_edges"] = float(edges.mean())
    results["max_nodes"] = float(nodes.max())
    results["edges_per_node"] = float(edges.sum() / nodes.sum())
    results["connected_fraction"] = float(np.mean([
        g.number_of_nodes() > 0 and nx.is_connected(g) for g in generated]))
    results.update(uniqueness_and_novelty(generated, training, use_labels=with_labels))
    return results


def held_out_floor(reference_a, reference_b, with_labels: bool = True) -> dict[str, float | None]:
    """Metrics between two samples of real graphs: what a perfect model would score."""
    return evaluate(reference_a, reference_b, with_labels=with_labels)

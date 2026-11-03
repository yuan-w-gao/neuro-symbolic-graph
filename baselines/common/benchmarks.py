"""Standard graph-generation benchmarks, for checking the baselines against their own papers.

planar: 200 Delaunay triangulations of 64 uniform points (SPECTRE / DiGress).
community_small: 500 graphs of two p=0.7 E-R communities of 6-10 nodes (GraphRNN).

GraphRNN reports raw squared MMD; DiGress and SPECTRE report the ratio of a model's
MMD to the train/test MMD, which ratio_to_floor computes.
"""

import random

import networkx as nx
import numpy as np


def planar_graphs(count=200, nodes=64, seed=0) -> list[nx.Graph]:
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    graphs = []
    while len(graphs) < count:
        graph = nx.Graph()
        graph.add_nodes_from(range(nodes))
        for simplex in Delaunay(rng.random((nodes, 2))).simplices:
            for i in range(3):
                graph.add_edge(int(simplex[i]), int(simplex[(i + 1) % 3]))
        if graph.number_of_nodes() == nodes and nx.is_connected(graph):
            graphs.append(graph)
    return graphs


def community_small_graphs(count=500, seed=0) -> list[nx.Graph]:
    rng = random.Random(seed)
    graphs = []
    while len(graphs) < count:
        sizes = [rng.randint(6, 10), rng.randint(6, 10)]
        graph = nx.disjoint_union_all(
            [nx.gnp_random_graph(size, 0.7, seed=rng.randrange(2 ** 31)) for size in sizes])
        first = list(range(sizes[0]))
        second = list(range(sizes[0], sizes[0] + sizes[1]))
        for _ in range(max(1, int(0.05 * graph.number_of_nodes()))):
            graph.add_edge(rng.choice(first), rng.choice(second))
        if nx.is_connected(graph):
            graphs.append(graph)
    return graphs


BENCHMARKS = {
    "planar": planar_graphs,
    "community_small": community_small_graphs,
}

PUBLISHED = {
    "planar": {
        "scale": "ratio-to-floor",
        "source": "DiGress (Vignac et al. 2023), Table 1",
        "GraphRNN": {"degree": 24.5, "clustering": 9.0, "orbit": 2508.0},
        "DiGress": {"degree": 1.4, "clustering": 1.2, "orbit": 1.7},
    },
    "community_small": {
        "scale": "raw MMD^2",
        # GraphRNN used a Gaussian-EMD kernel; re-evaluations with the Gaussian-TV kernel
        # used here report about Deg 0.08, Clus 0.12, so that band is the check, not 0.03.
        "source": "GraphRNN (You et al. 2018), Table 2 — Gaussian-EMD kernel",
        "GraphRNN": {"degree": 0.03, "clustering": 0.03, "orbit": 0.01},
        "GraphRNN (TV-kernel re-eval)": {"degree": 0.08, "clustering": 0.12, "orbit": 0.04},
    },
}


def split(graphs, seed=0, train_fraction=0.8):
    shuffled = list(graphs)
    random.Random(seed).shuffle(shuffled)
    cut = int(train_fraction * len(shuffled))
    return shuffled[:cut], shuffled[cut:]


def ratio_to_floor(generated: dict, floor: dict) -> dict:
    """MMD(model, test) / MMD(train, test), the scale DiGress and SPECTRE report."""
    out = {}
    for key in ("degree", "clustering", "spectrum"):
        numerator, denominator = generated.get(key), floor.get(key)
        out[key] = None if numerator is None or not denominator else numerator / denominator
    return out


def load_training_graphs(train_file, benchmark, label_level, seed):
    """Training graphs for the GraphRNN and diffusion baselines: a benchmark's train split or an EDS corpus."""
    if benchmark:
        train, _ = split(BENCHMARKS[benchmark](seed=seed), seed=seed)
        return train
    from graph_grammar.data.graph_io import load_eds_graphs, records_to_networkx
    return [g for g in records_to_networkx(load_eds_graphs(train_file), label_level)
            if g.number_of_nodes() > 1]

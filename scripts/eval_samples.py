#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph_grammar.data.graph_io import load_eds_graphs, records_to_networkx
from graph_grammar.evaluation.graph_gen_metrics import evaluate

COLUMNS = [("degree", "Degree"), ("clustering", "Clustering"),
           ("spectrum", "Spectrum"), ("node_label_tv", "Node-label TV")]
DIAGNOSTICS = [("mean_nodes", "Nodes"), ("connected_fraction", "Connected"),
               ("uniqueness", "Unique"), ("novelty", "Novel")]


def parse_named(values: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    out = []
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"expected NAME=path, got {value!r}")
        name, _sep, path = value.rpartition("=")
        out.append((name, path))
    return out


def load_any(path: str) -> List[nx.Graph]:
    if path.endswith(".npz"):
        return [nx.from_numpy_array(a) for a in np.load(path).values()]
    if path.endswith(".pkl"):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    return records_to_networkx(load_eds_graphs(path))


def replicate_stats(graphs: Sequence[nx.Graph], test, train, replicates: int,
                    size: int, with_labels: bool, overlap: bool = False):
    if overlap:
        if len(graphs) < size:
            raise SystemExit(f"need at least {size} graphs, got {len(graphs)}")
        rng = np.random.default_rng(0)
        blocks = replicates
        subsets = [[graphs[i] for i in rng.choice(len(graphs), size, replace=False)]
                   for _ in range(blocks)]
    else:
        blocks = min(replicates, len(graphs) // size)
        if blocks < 1:
            raise SystemExit(f"need at least {size} graphs, got {len(graphs)}")
        subsets = [list(graphs[i * size:(i + 1) * size]) for i in range(blocks)]
    runs = [evaluate(subset, test, with_labels=with_labels, training=train)
            for subset in subsets]
    keys = [k for k, _ in COLUMNS + DIAGNOSTICS]
    out = {}
    for key in keys:
        values = np.array([r[key] for r in runs], dtype=float) \
            if runs[0][key] is not None else np.array([np.nan])
        out[key] = (float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0, blocks)
    return out


def cell(mean: float, sd: float) -> str:
    if mean != mean:
        return "—"
    if abs(mean) < 1e-3:
        return f"{mean:.2e} ± {sd:.0e}"
    return f"{mean:.4f} ± {sd:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True,
                        help="graphs the models were trained on; the novelty reference")
    parser.add_argument("--test", required=True)
    parser.add_argument("--floor-graphs",
                        help="real graphs for the floor row (default: --train). Set this "
                             "when --train varies between tables that must be compared, "
                             "as in the data-efficiency curve: the floor is a property of "
                             "the data, not of how much of it a model saw, so it has to be "
                             "the same at every point.")
    parser.add_argument("--eds", action="append")
    parser.add_argument("--npz", action="append", help="unlabeled adjacency pools")
    parser.add_argument("--pkl", action="append")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--out-dir", default="eval_samples")
    args = parser.parse_args()

    train = records_to_networkx(load_eds_graphs(args.train))
    test = records_to_networkx(load_eds_graphs(args.test))
    floor_source = (records_to_networkx(load_eds_graphs(args.floor_graphs))
                    if args.floor_graphs else train)

    models: List[Tuple[str, str, bool]] = []
    for name, path in parse_named(args.eds) + parse_named(args.pkl):
        models.append((name, path, True))
    for name, path in parse_named(args.npz):
        models.append((name, path, False))

    rows = [("Held-out data (floor)",
             replicate_stats(floor_source, test, None, args.replicates, args.size, True,
                             overlap=True))]
    for name, path, labeled in models:
        pool = load_any(path)
        rows.append((name, replicate_stats(pool, test, train, args.replicates,
                                           args.size, labeled)))
        print(f"  scored {name}: {len(pool)} graphs -> "
              f"{rows[-1][1]['degree'][2]} replicates", flush=True)

    width = max(len(n) for n, _ in rows) + 2
    header = f"{'Model':<{width}}" + "".join(f"{lab:>22}" for _, lab in COLUMNS)
    lines = [header, "-" * len(header)]
    for name, stats in rows:
        lines.append(f"{name:<{width}}" +
                     "".join(f"{cell(*stats[k][:2]):>22}" for k, _ in COLUMNS))
    lines += ["", f"{'Model':<{width}}" + "".join(f"{lab:>16}" for _, lab in DIAGNOSTICS)]
    for name, stats in rows:
        lines.append(f"{name:<{width}}" +
                     "".join(f"{stats[k][0]:>16.3f}" for k, _ in DIAGNOSTICS))
    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_replicates.txt").write_text(table + "\n")
    json.dump({n: {k: v for k, v in s.items()} for n, s in rows},
              open(out / "metrics_replicates.json", "w"), indent=1)
    print(f"\nwrote {out}/table_replicates.txt and metrics_replicates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph_grammar.generation.grammar import GrammarWrapper

import torch

from graph_grammar.batched import BatchedForest
from graph_grammar.cache import load_forests_with_cache


def load_weights(path: str, num_rules: int) -> torch.Tensor:
    w = torch.full((num_rules,), float("-inf"), dtype=torch.float64)
    seen = 0
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            if idx >= num_rules:
                raise SystemExit(
                    f"{path} has rule index {idx} but the grammar holds {num_rules}")
            v = parts[-1]
            x = float("-inf") if v in ("-inf", "-nan", "nan") else float(v)
            w[idx] = float("-inf") if x < -100 else x
            seen += 1
    if seen == 0:
        raise SystemExit(f"{path} contained no weights")
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grammar-dir", "-g", required=True)
    ap.add_argument("--test-file", default="test.split.graphs.txt")
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--mapping-file", default="train.mapping.txt",
                    help="grammar file inside --grammar-dir; must match the inventory "
                         "the weights were trained on")
    ap.add_argument("--weights", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    grammar_file = f"{args.grammar_dir}/{args.mapping_file}"
    grammar = GrammarWrapper(grammar_file)
    num_rules = grammar.num_rules

    forests, _i, _s, _nr, _c = load_forests_with_cache(
        grammar_file=grammar_file,
        graphs_file=f"{args.grammar_dir}/{args.test_file}",
        cache_dir=args.test_cache, batch_cache=True, verbose=False,
        timeout=args.timeout)
    forests = [f for f in forests if f.roots]
    batch = BatchedForest.from_forests(forests)
    print(f"  {len(forests)} held-out forests, {len(batch.edge_rule):,} hyperedges, "
          f"{batch.num_levels} levels\n")

    rows = []
    print(f"{'arm':38s} {'coverage':>9} {'LL/graph':>11} {'scored':>7}")
    for spec in args.weights:
        name, _, path = spec.rpartition("=")
        if not name:
            name = Path(path).parent.name
        w = load_weights(path, num_rules)
        z = batch.log_partition(w)
        finite = torch.isfinite(z)
        n = int(finite.sum())
        cov = n / len(forests) if forests else float("nan")
        ll = float(z[finite].mean()) if n else float("nan")
        rows.append({"arm": name, "weights": path, "coverage": cov,
                     "ll_per_scoreable_graph": ll, "scored": n,
                     "total": len(forests)})
        print(f"{name[:38]:38s} {100*cov:8.1f}% {ll:11.2f} {n:7d}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

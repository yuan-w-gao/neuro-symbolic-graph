#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph_grammar.generation.grammar import GrammarWrapper  # imports pyshrg; must precede torch

import torch

from graph_grammar.cache import load_forests_with_cache
from graph_grammar.inference import inside, inside_with_entropy

_spec = importlib.util.spec_from_file_location(
    "sample_hrg", str(PROJECT_ROOT / "scripts" / "sample.py"))
_sample_hrg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sample_hrg)


def parse_named(values) -> List[Tuple[str, str]]:
    out = []
    for value in values or []:
        name, _, path = value.rpartition("=")
        if not name:
            raise SystemExit(f"expected NAME=path, got {value!r}")
        out.append((name, path))
    return out


def group_entropy_fraction(log_probs: torch.Tensor, grammar) -> float:
    total, total_max = 0.0, 0.0
    for rules in grammar._rules_by_label.values():
        idx = torch.tensor([r.index for r in rules], dtype=torch.long)
        if len(idx) < 2:
            continue
        lp = torch.nan_to_num(log_probs[idx].double(), neginf=-60.0).clamp(min=-60.0)
        lp = lp - torch.logsumexp(lp, dim=0)
        h = float(-(lp.exp() * lp).sum())
        if h == h:
            total += h
            total_max += math.log(len(idx))
    return total / total_max


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grammar", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--weights", action="append", required=True, help="NAME=path")
    parser.add_argument("--forests", type=int, default=400)
    parser.add_argument("--out")
    args = parser.parse_args()

    grammar = GrammarWrapper(args.grammar)
    forests, _, _, num_rules, _ = load_forests_with_cache(
        grammar_file=args.grammar, graphs_file=args.graphs,
        cache_dir=args.cache_dir, batch_cache=True, verbose=False)
    forests = [f for f in forests if f.roots][: args.forests]
    mapping = grammar.shrg_to_hrg_index()
    print(f"{len(forests)} forests, {num_rules} SHRG rules")

    header = (f"{'model':<22}{'H(d|G)':>10}{'exp(H)':>9}{'p(r|A) ent':>12}"
              f"{'LL':>13}{'LL/graph':>10}")
    print(f"\n{header}\n{'-' * len(header)}")
    rows = []
    for name, path in parse_named(args.weights):
        hrg = _sample_hrg.load_weights_as_hrg(path, grammar)
        shrg = torch.full((num_rules,), -60.0)
        for i in range(min(num_rules, len(mapping))):
            shrg[i] = hrg[int(mapping[i])]
        with torch.no_grad():
            pairs = [inside_with_entropy(f, shrg) for f in forests]
        ll = float(sum(p[0] for p in pairs))
        ent = [float(p[1]) for p in pairs]
        ent = [e for e in ent if e == e]
        mean_h = sum(ent) / max(len(ent), 1)
        frac = group_entropy_fraction(hrg, grammar)
        rows.append({"name": name, "derivation_entropy": mean_h,
                     "group_entropy_fraction": frac, "ll": ll,
                     "ll_per_graph": ll / max(len(forests), 1)})
        print(f"{name:<22}{mean_h:>10.3f}{math.exp(mean_h):>9.1f}{100*frac:>11.1f}%"
              f"{ll:>13,.0f}{ll/max(len(forests),1):>10.2f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

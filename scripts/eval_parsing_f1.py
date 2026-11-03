#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph_grammar.generation.grammar import GrammarWrapper

import torch

from graph_grammar.cache import load_forests_with_cache
from graph_grammar.evaluation.parsing_f1 import (
    compute_f1, derivation_fragments, derivation_spans,
    find_gold_derivation_by_fragments, gold_fragments, load_gold_derivations)
from graph_grammar.inference import sample, viterbi

_spec_path = PROJECT_ROOT / "scripts" / "sample.py"


def load_weights(path: str, grammar) -> torch.Tensor:
    import importlib.util
    spec = importlib.util.spec_from_file_location("sample_hrg", str(_spec_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_weights_as_hrg(path, grammar)


def parse_named(values) -> List[Tuple[str, str]]:
    out = []
    for value in values or []:
        name, _sep, path = value.rpartition("=")
        if not name:
            raise SystemExit(f"expected NAME=path, got {value!r}")
        out.append((name, path))
    return out


def read_edges_maps(grammar_file: str, graphs_file: str) -> Dict[str, Dict[Tuple, int]]:
    import pyshrg
    pyshrg.initialize()
    manager = pyshrg.get_manager()
    manager.load_grammars(grammar_file)
    manager.load_graphs(graphs_file)
    out = {}
    for i in range(manager.graph_size):
        graph = manager.get_graph(i)
        out[graph.sentence_id.split("/")[-1]] = dict(graph.edges_map)
    return out


def evaluate(gold_by_graph, forests_by_graph, log_probs, level, seed=None,
             draws=5):
    read = derivation_fragments if level == "fragment" else derivation_spans
    precisions, recalls, f1s, scored = [], [], [], 0
    per_graph: Dict[str, float] = {}
    if seed is not None:
        torch.manual_seed(seed)
    for key, forest in forests_by_graph.items():
        gold = gold_by_graph.get(key)
        if not gold:
            continue
        values = []
        with torch.no_grad():
            if seed is None:
                _best, derivation = viterbi(forest, log_probs)
                derivations = [derivation] if derivation else []
            else:
                derivations = []
                for _ in range(draws):
                    drawn = sample(forest, log_probs, num_samples=1)[0]
                    derivations.append(drawn[1] if isinstance(drawn, tuple) else drawn)
        for derivation in derivations:
            predicted = read(forest, derivation)
            if predicted:
                values.append(compute_f1(predicted, gold))
        if not values:
            continue
        p = sum(v[0] for v in values) / len(values)
        r = sum(v[1] for v in values) / len(values)
        f1 = sum(v[2] for v in values) / len(values)
        precisions.append(p); recalls.append(r); f1s.append(f1)
        per_graph[key] = f1
        scored += 1
    n = max(len(f1s), 1)
    return {"precision": sum(precisions) / n, "recall": sum(recalls) / n,
            "f1": sum(f1s) / n, "scored": scored, "per_graph": per_graph}


def paired(result, control) -> Dict[str, object]:
    shared = sorted(set(result["per_graph"]) & set(control["per_graph"]))
    diffs = [result["per_graph"][k] - control["per_graph"][k] for k in shared]
    if not diffs:
        return {"delta_f1": 0.0, "n": 0, "t": 0.0, "p": "n/a"}
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / max(len(diffs) - 1, 1)
    se = math.sqrt(var / len(diffs))
    t = mean / se if se > 0 else 0.0
    try:
        from scipy import stats
        pvalue = f"{2 * stats.t.sf(abs(t), len(diffs) - 1):.3g}"
    except Exception:
        pvalue = "n/a"
    return {"delta_f1": mean, "n": len(diffs), "t": t, "p": pvalue}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grammar-dir", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--weights", action="append", help="NAME=path")
    parser.add_argument("--uniform", action="store_true", help="add a uniform control")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random-derivation floor")
    parser.add_argument("--out")
    args = parser.parse_args()

    grammar_file = f"{args.grammar_dir}/train.mapping.txt"
    edges_maps = read_edges_maps(grammar_file, args.test)
    grammar = GrammarWrapper(grammar_file)
    forests, graph_ids, _s, _num_rules, _c = load_forests_with_cache(
        grammar_file=grammar_file, graphs_file=args.test,
        cache_dir=args.cache_dir or f"{args.grammar_dir}/.cache_f1",
        batch_cache=True, verbose=False, timeout=args.timeout)

    raw_gold = load_gold_derivations(f"{args.grammar_dir}/train.derivations.p")
    forests_by_graph, gold_frag, gold_span, unresolved, unlocated = {}, {}, {}, 0, 0
    for forest, graph_id in zip(forests, graph_ids):
        key = graph_id.split("/")[-1]
        steps = raw_gold.get(key)
        if not forest.roots or not steps or key not in edges_maps:
            continue
        gold = gold_fragments(steps, edges_maps[key])
        if not gold:
            unresolved += 1
            continue
        forests_by_graph[key] = forest
        gold_frag[key] = gold
        located = find_gold_derivation_by_fragments(forest, gold)
        if located is None:
            unlocated += 1
        else:
            gold_span[key] = derivation_spans(forest, located)

    print(f"  {len(forests)} test forests, {len(raw_gold)} gold derivations, "
          f"{grammar.num_rules} rules")
    print(f"  {len(gold_frag)} graphs with resolvable gold fragments "
          f"({unresolved} unresolved); gold derivation located in the forest for "
          f"{len(gold_span)} ({unlocated} not present)")

    arms: Dict[str, torch.Tensor] = {}
    for name, path in parse_named(args.weights):
        arms[name] = load_weights(path, grammar)
    if args.uniform:
        vector = torch.zeros(grammar.num_rules)
        for rules in grammar._rules_by_label.values():
            idx = torch.tensor([r.index for r in rules], dtype=torch.long)
            vector[idx] = -math.log(len(rules))
        arms["uniform (control)"] = vector

    width = max(max(len(k) for k in arms), 24) + 2
    report: Dict[str, Dict] = {}
    for level, gold_by_graph in (("fragment", gold_frag), ("span", gold_span)):
        if not gold_by_graph:
            continue
        level_report = {}
        for name, log_probs in arms.items():
            level_report[name] = evaluate(gold_by_graph, forests_by_graph,
                                          log_probs, level)
        level_report["random derivation"] = evaluate(
            gold_by_graph, forests_by_graph, arms["uniform (control)"], level,
            seed=args.seed)

        title = ("fragment F1 -- which edges each rule application introduces"
                 if level == "fragment" else
                 "span F1 -- the full subtree each derivation step covers")
        header = (f"{'model':<{width}}{'precision':>11}{'recall':>9}{'F1':>9}"
                  f"{'graphs':>9}{'vs random':>11}{'t':>8}{'p':>10}")
        print(f"\n{title}\n{header}\n{'-' * len(header)}")
        floor = level_report["random derivation"]
        for name, result in level_report.items():
            cmp = paired(result, floor) if name != "random derivation" else None
            tail = (f"{cmp['delta_f1']:>11.4f}{cmp['t']:>8.2f}{cmp['p']:>10}"
                    if cmp else f"{'--':>11}{'--':>8}{'--':>10}")
            if cmp:
                result["paired_vs_random"] = cmp
            print(f"{name:<{width}}{result['precision']:>11.4f}{result['recall']:>9.4f}"
                  f"{result['f1']:>9.4f}{result['scored']:>9}{tail}")
        for result in level_report.values():
            result.pop("per_graph", None)
        report[level] = level_report

    print("\n  Fragment F1 asks whether the model carves the graph into the same "
          "pieces;\n  span F1 asks whether it assembles those pieces into the same "
          "tree. The random\n  arm draws derivations from the same forest, which is "
          "the floor any claim about\n  learned weights has to clear -- a model level "
          "with it has learned nothing the\n  grammar did not already encode.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

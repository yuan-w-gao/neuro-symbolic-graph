#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph_grammar.generation.grammar import GrammarWrapper
from graph_grammar.generation.generator import GraphGenerator
from graph_grammar.generation.writer import GraphWriter
from graph_grammar.data.graph_io import load_eds_graphs


def load_final_weights(path: str, size: int) -> torch.Tensor:
    weights = torch.full((size,), -20.0)
    seen = 0
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            idx = int(parts[0])
            if idx >= size:
                raise SystemExit(
                    f"{path} has index {idx} but the target space holds {size} "
                    f"rules — the weights are in the wrong index space")
            weights[idx] = float(parts[-1])
            seen += 1
    print(f"  loaded {seen} rule weights from {path}")
    return weights


def weight_index_space(path: str, grammar: GrammarWrapper) -> int:
    indices = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                indices.add(int(line.split(",")[0]))
    hrg, shrg = grammar.num_rules, grammar.manager.shrg_size
    highest = max(indices) if indices else -1
    if len(indices) == hrg:
        return hrg
    if len(indices) == shrg:
        return shrg
    if highest < hrg:
        print(f"  warning: {path} holds {len(indices)} of {hrg} HRG rules, "
              f"assuming the HRG index space")
        return hrg
    if highest < shrg:
        print(f"  warning: {path} holds {len(indices)} of {shrg} SHRG rules, "
              f"assuming the SHRG index space")
        return shrg
    raise SystemExit(f"{path} has rule index {highest}, beyond both the HRG "
                     f"({hrg}) and SHRG ({shrg}) rule counts")


def load_weights_as_hrg(path: str, grammar: GrammarWrapper) -> torch.Tensor:
    size = weight_index_space(path, grammar)
    weights = load_final_weights(path, size)
    if size == grammar.num_rules:
        return weights
    print(f"  weights are SHRG-indexed ({size}); marginalizing the syntactic "
          f"side out to {grammar.num_rules} HRG rules")
    return grammar.marginalize_shrg_log_probs(weights)


def size_distribution(path: str) -> Tuple[List[int], List[float]]:
    counts = Counter(record.num_nodes for record in load_eds_graphs(path))
    sizes = sorted(counts)
    total = sum(counts.values())
    return sizes, [counts[s] / total for s in sizes]


def sample_conditioned(propose, count: int, reference: str,
                       seed: int, batch: int = 2000, max_batches: int = 60):
    rng = random.Random(seed)
    sizes, probs = size_distribution(reference)
    targets = Counter(rng.choices(sizes, weights=probs, k=count))

    available: Dict[int, List] = {}
    accepted: List = []
    proposed = 0
    for _ in range(max_batches):
        outstanding = {n: k for n, k in targets.items() if k > len(available.get(n, ()))}
        if not outstanding:
            break
        for graph in propose(batch):
            available.setdefault(graph.num_nodes, []).append(graph)
        proposed += batch

    unmet = 0
    for n, wanted in sorted(targets.items()):
        have = available.get(n, [])
        take = min(wanted, len(have))
        accepted.extend(have[:take])
        unmet += wanted - take
    rng.shuffle(accepted)
    print(f"  size-conditioned: {proposed} proposals -> {len(accepted)} accepted "
          f"({100.0 * len(accepted) / max(proposed, 1):.1f}%)")
    if unmet:
        print(f"  WARNING: {unmet} target sizes unreachable after {max_batches} batches")
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", required=True, help="train.mapping.txt")
    parser.add_argument("--weights", help="weight_history file (static weights)")
    parser.add_argument("--model", help="compound model.pt (per-graph latent z)")
    parser.add_argument("--uniform", action="store_true", help="uniform weights")
    parser.add_argument("-n", "--num-graphs", type=int, default=1000)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition-size", metavar="GRAPHS",
                        help="EDS corpus (the TRAINING split) whose node-count "
                             "histogram the samples should match; sampling then "
                             "draws from p(G | |V| = n)")
    parser.add_argument("--proposal-batch", type=int, default=2000,
                        help="proposals per rejection-sampling round")
    parser.add_argument("--timing-json",
                        help="write sampling wall-clock and throughput here, in the same "
                             "schema the GraphRNN and DiGress runners emit, so the three "
                             "models land in one efficiency table")
    parser.add_argument("--max-batches", type=int, default=60,
                        help="give up after this many rounds (guards a grammar "
                             "whose size distribution cannot reach the targets)")
    args = parser.parse_args()

    grammar = GrammarWrapper(args.grammar)
    print(f"  grammar: {grammar.num_rules} rules, {grammar.num_labels} labels")

    torch.manual_seed(args.seed)
    generator = GraphGenerator(grammar, seed=args.seed)

    start = time.time()
    if args.model:
        propose = _compound_proposal(args, grammar, generator)
    else:
        if args.uniform:
            grammar.set_uniform_weights()
        else:
            grammar.set_weights(load_weights_as_hrg(args.weights, grammar))
        propose = lambda k: generator.generate_batch(k, validate=True)

    if args.condition_size:
        graphs = sample_conditioned(propose, args.num_graphs,
                                    args.condition_size, args.seed,
                                    batch=args.proposal_batch,
                                    max_batches=args.max_batches)
    else:
        graphs = propose(args.num_graphs)
    elapsed = time.time() - start

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    GraphWriter.write_graphs(args.output, graphs)
    mean_nodes = sum(g.num_nodes for g in graphs) / max(len(graphs), 1)
    print(f"  sampled {len(graphs)} graphs in {elapsed:.1f}s "
          f"({len(graphs)/max(elapsed,1e-9):.0f}/s), mean nodes {mean_nodes:.1f}")
    print(f"  wrote {args.output}")
    if args.timing_json:
        import json as _json
        kind = "hrg_compound" if args.model else ("hrg_uniform" if args.uniform else "hrg_em")
        rec = {"model": kind, "num_rules": grammar.num_rules,
               "num_samples": len(graphs),
               "sample_seconds": elapsed,
               "graphs_per_second": len(graphs) / max(elapsed, 1e-9),
               "seconds_per_graph": elapsed / max(len(graphs), 1),
               "mean_nodes": mean_nodes,
               "size_conditioned": bool(args.condition_size),
               "params": 0 if args.uniform else grammar.num_rules}
        Path(args.timing_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.timing_json, "w") as fh:
            _json.dump(rec, fh, indent=2)
        print(f"  timing -> {args.timing_json}: "
              f"{rec['graphs_per_second']:.1f} graphs/s, {rec['params']:,} learned params")
    return 0


def _load_compound(model_path: str, grammar_file: str):
    import torch as _torch
    from graph_grammar.learning.compound import (
        CompoundConfig, CompoundHRG, create_structural_embedder)

    checkpoint = _torch.load(model_path, map_location="cpu", weights_only=False)
    if not (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
        raise SystemExit(f"{model_path} is not a compound checkpoint")
    config = checkpoint["config"]
    state = checkpoint["model_state_dict"]
    rule_to_label = state["scorer.rule_to_label"]
    structural = "scorer.structural_embedder.gnn.node_input_proj.weight" in state

    embedder = None
    if structural:
        embedder = create_structural_embedder(
            CompoundConfig(
                use_structural_embeddings=True,
                structural_encoder_type="gnn",
                structural_gnn_layers=2,
                structural_hidden_dim=config.get("hidden_dim", 128),
                structural_embed_dim=config.get("embedding_dim", 64),
                grammar_file=grammar_file,
                device="cpu",
            ),
            _torch.device("cpu"),
        )

    model = CompoundHRG(
        num_rules=config["num_rules"],
        num_labels=config["num_labels"],
        rule_to_label=rule_to_label,
        encoder_input_dim=config["num_rules"] + 5,
        embedding_dim=config.get("embedding_dim", 64),
        hidden_dim=config.get("hidden_dim", 128),
        latent_dim=config.get("latent_dim", 32),
        device=_torch.device("cpu"),
        structural_embedder=embedder,
    )
    model.load_state_dict(state)
    model.eval()
    return model.scorer, config.get("latent_dim", 32)


def _compound_proposal(args, grammar: GrammarWrapper, generator: GraphGenerator):
    scorer, latent_dim = _load_compound(args.model, args.grammar)

    def propose(count: int):
        graphs = []
        resamples = 0
        with torch.no_grad():
            for _ in range(count):
                z = torch.randn(latent_dim)
                log_probs = scorer.get_rule_log_probs(z).reshape(-1).cpu()
                grammar.set_weights(grammar.marginalize_shrg_log_probs(log_probs))
                graph = generator.generate_with_retry(max_retries=10)
                if graph is None:
                    resamples += 1
                    continue
                graphs.append(graph)
        if resamples:
            print(f"  {resamples} graphs failed to sample after retries")
        return graphs

    return propose


if __name__ == "__main__":
    raise SystemExit(main())

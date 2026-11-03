#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from graph_grammar.generation.grammar import GrammarWrapper  # imports pyshrg; must precede torch

import torch
from rdkit import Chem

from graph_grammar.generation.generator import GraphGenerator
from graph_grammar.molecules.mol_io import graph_to_mol
from eval_heldout import load_weights


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grammar-dir", "-g", required=True)
    ap.add_argument("--mapping-file", default="train.mapping.txt")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--reference-smiles", help="training SMILES; enables novelty")
    ap.add_argument("-n", "--num-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    grammar = GrammarWrapper(f"{args.grammar_dir}/{args.mapping_file}")
    w = load_weights(args.weights, grammar.num_rules)
    grammar.set_weights(w.to(torch.float32))
    print(f"{grammar.num_rules} rules; sampling {args.num_samples} molecules")

    gen = GraphGenerator(grammar, seed=args.seed)
    smiles: list[str] = []
    sizes: list[int] = []
    reasons: Counter = Counter()
    attempted = 0
    for _ in range(args.num_samples):
        try:
            g = gen.generate()
        except Exception as exc:
            reasons[f"sample:{type(exc).__name__}"] += 1
            continue
        attempted += 1
        labels = [n.label for n in g.nodes]
        edges = [(e.source, e.target, e.label) for e in g.edges]
        pairs = [(min(u, v), max(u, v)) for u, v, _l in edges]
        if any(u == v for u, v in pairs):
            reasons["self-bond"] += 1
            continue
        if len(set(pairs)) != len(pairs):
            reasons["parallel-bond"] += 1
            continue
        mol = graph_to_mol(labels, edges)
        if mol is None:
            reasons["valence-violation"] += 1
            continue
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            reasons["smiles-failed"] += 1
            continue
        smiles.append(smi)
        sizes.append(mol.GetNumAtoms())

    valid = len(smiles)
    uniq = set(smiles)
    row = {"samples": attempted, "valid": valid,
           "validity": valid / attempted if attempted else 0.0,
           "unique": len(uniq),
           "uniqueness": len(uniq) / valid if valid else 0.0,
           "mean_heavy_atoms": (sum(sizes) / len(sizes)) if sizes else 0.0,
           "failures": dict(reasons)}

    print(f"\nvalidity    {valid}/{attempted} = {100*row['validity']:.2f}%")
    if reasons:
        print(f"   failures by cause: {dict(reasons)}")
    print(f"uniqueness  {len(uniq)}/{valid} = {100*row['uniqueness']:.2f}%")
    if args.reference_smiles:
        ref = {l.strip() for l in open(args.reference_smiles) if l.strip()}
        novel = [s for s in uniq if s not in ref]
        row["novelty"] = len(novel) / len(uniq) if uniq else 0.0
        row["reference_size"] = len(ref)
        print(f"novelty     {len(novel)}/{len(uniq)} = {100*row['novelty']:.2f}%  "
              f"(against {len(ref):,} training molecules)")
    print(f"mean heavy atoms {row['mean_heavy_atoms']:.1f}")
    if uniq:
        print("examples: " + ", ".join(list(uniq)[:5]))

    if args.out:
        import json
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(row, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

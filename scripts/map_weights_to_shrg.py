#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph_grammar.generation.grammar import GrammarWrapper  # imports pyshrg; must precede torch


def load_final_column(path: str) -> dict:
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            val = None
            for tok in parts[1:]:
                try:
                    val = float(tok)
                except ValueError:
                    continue
            if val is not None:
                out[idx] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grammar", "-g", required=True, help="train.mapping.txt")
    ap.add_argument("--weights", "-w", required=True, help="HRG-indexed weight_history")
    ap.add_argument("--mode", choices=["copy", "split"], default="copy")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--floor", type=float, default=-60.0,
                    help="value written for an HRG rule at -inf (the C++ reader has no -inf)")
    args = ap.parse_args()

    grammar = GrammarWrapper(args.grammar)
    s2h = grammar.shrg_to_hrg_index().tolist()
    hrg = load_final_column(args.weights)

    n_hrg = grammar.num_rules
    if hrg and max(hrg) >= n_hrg:
        raise SystemExit(
            f"{args.weights} holds index {max(hrg)} but the HRG space is {n_hrg} rules -- "
            f"this file is probably already SHRG-indexed; do not map it twice")

    share = Counter(s2h)
    written = missing = floored = 0
    with open(args.output, "w") as out:
        for shrg_idx, hrg_idx in enumerate(s2h):
            if hrg_idx < 0 or hrg_idx not in hrg:
                missing += 1
                continue
            w = hrg[hrg_idx]
            if not math.isfinite(w) or w < args.floor:
                w = args.floor
                floored += 1
            elif args.mode == "split":
                w -= math.log(share[hrg_idx])
            out.write(f"{shrg_idx},{w:.10f}\n")
            written += 1

    ident = sum(1 for i, h in enumerate(s2h) if i == h)
    print(f"HRG rules read      : {len(hrg):,} / {n_hrg:,}")
    print(f"SHRG rules written  : {written:,} / {len(s2h):,}  (mode={args.mode})")
    print(f"  no source weight  : {missing:,}")
    print(f"  floored at {args.floor}: {floored:,}")
    print(f"max SHRG sharing one HRG rule: {max(share.values())}")
    print(f"identity agreement  : {ident}/{len(s2h)} = {100*ident/len(s2h):.2f}%  "
          f"<- the fraction the unmapped file got right")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

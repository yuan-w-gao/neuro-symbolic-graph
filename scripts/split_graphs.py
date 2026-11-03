#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph_grammar.data.graph_io import load_eds_graphs, write_eds_graphs


def source_of(sentence_id: str) -> str:
    tail = sentence_id.split("/")[-1]
    return tail.rsplit("-", 1)[0] if "-" in tail else tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    records = load_eds_graphs(args.graphs)

    # whole sentences go to one side, so repeated sentences cannot straddle the split
    by_sentence = defaultdict(list)
    for record in records:
        by_sentence[record.sentence].append(record)
    by_source = defaultdict(list)
    for sentence, group in by_sentence.items():
        by_source[source_of(group[0].sentence_id)].append(group)

    rng = random.Random(args.seed)
    train, test = [], []
    for source in sorted(by_source):
        groups = by_source[source][:]
        rng.shuffle(groups)
        wanted = round(sum(len(g) for g in groups) * args.test_fraction)
        taken = 0
        for group in groups:
            if taken < wanted:
                test.extend(group)
                taken += len(group)
            else:
                train.extend(group)
    rng.shuffle(train)
    rng.shuffle(test)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / f"{args.prefix}train.split.graphs.txt"
    test_path = out / f"{args.prefix}test.split.graphs.txt"
    write_eds_graphs(str(train_path), train)
    write_eds_graphs(str(test_path), test)

    train_keys = {r.sentence for r in train}
    leak = sum(1 for r in test if r.sentence in train_keys)
    print(f"  {len(records)} graphs, {len(by_sentence)} distinct sentences, "
          f"{len(by_source)} sources")
    print(f"  train {len(train)}  test {len(test)}  ({args.test_fraction:.0%} held out)")
    print(f"  leak check: {leak} test sentences also appear in train")
    per_source = {s: sum(len(g) for g in gs) for s, gs in by_source.items()}
    print(f"  smallest source has {min(per_source.values())} graphs; "
          f"{sum(1 for v in per_source.values() if v < 5)} sources under 5")
    print(f"  wrote {train_path}\n  wrote {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

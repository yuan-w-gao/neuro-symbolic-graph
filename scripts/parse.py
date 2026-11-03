#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pyshrg

from graph_grammar.cache import ForestCache
from graph_grammar.export import export_forest


def main() -> int:
    import threading
    sys.setrecursionlimit(200000)
    threading.stack_size(512 * 1024 * 1024)
    result = {}
    t = threading.Thread(target=lambda: result.update(code=_main()))
    t.start()
    t.join()
    return result.get("code", 1)


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar-dir", "-g", required=True)
    ap.add_argument("--graph-file", default="train.split.graphs.txt")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--report-every", type=int, default=50)
    ap.add_argument("--pool", type=int, default=25,
                    help="chart pool blocks (Context::Init). A larger inventory makes more partial "
                         "items match, so a big grammar needs a bigger pool: the arity-4 program "
                         "grammar returns kOutOfMemory on some graphs at 25 that parse fine at 400.")
    args = ap.parse_args()

    grammar_file = f"{args.grammar_dir}/train.mapping.txt"
    graphs_file = f"{args.grammar_dir}/{args.graph_file}"

    pyshrg.initialize()
    manager = pyshrg.get_manager()
    manager.load_grammars(grammar_file)
    manager.load_graphs(graphs_file)
    manager.allocate(num_contexts=1)
    manager.init_all(type="linear", verbose=False, max_pool_size=args.pool)
    context = manager.get_context(0)

    cache = ForestCache.from_files(args.cache_dir, grammar_file, graphs_file)

    total = manager.graph_size
    mine = list(range(args.shard, total, args.num_shards))
    done = failed = skipped = 0
    start = time.time()
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} of {total} graphs",
          flush=True)

    for n, i in enumerate(mine):
        graph = manager.get_graph(i)
        gid = graph.sentence_id
        if cache.is_valid(gid):
            skipped += 1
            continue
        try:
            if context.parse(graph).name == "kNone":
                cache.save_forest(gid, export_forest(context, graph))
                done += 1
            else:
                failed += 1
        except Exception:
            failed += 1
        finally:
            context.release_memory()

        if (n + 1) % args.report_every == 0:
            el = time.time() - start
            rate = (done + failed) / el if el else 0
            left = (len(mine) - n - 1) / rate / 60 if rate else float("nan")
            print(f"  shard {args.shard}: {n+1}/{len(mine)} "
                  f"(ok {done}, fail {failed}, cached {skipped}) "
                  f"{rate:.2f} g/s, ~{left:.0f} min left", flush=True)

    el = time.time() - start
    print(f"shard {args.shard} DONE: parsed {done}, failed {failed}, "
          f"pre-cached {skipped} in {el/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

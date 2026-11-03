#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from graph_grammar.generation.grammar import GrammarWrapper  # imports pyshrg; must precede torch

import torch

from graph_grammar.batched import BatchedForest
from graph_grammar.cache import load_forests_with_cache


def group_ids(grammar, num_rules: int) -> torch.Tensor:
    keys = sorted({(label, len(rule.external_nodes))
                   for label, rules in grammar._rules_by_label.items()
                   for rule in rules})
    index = {k: i for i, k in enumerate(keys)}
    out = torch.zeros(num_rules, dtype=torch.long)
    for label, rules in grammar._rules_by_label.items():
        for rule in rules:
            if rule.index < num_rules:
                out[rule.index] = index[(label, len(rule.external_nodes))]
    return out


def uniform_weights(r2g: torch.Tensor, num_rules: int) -> torch.Tensor:
    w = torch.zeros(num_rules, dtype=torch.float64)
    for g in torch.unique(r2g):
        idx = (r2g == g).nonzero(as_tuple=True)[0]
        w[idx] = -math.log(len(idx))
    return w


def normalize_by_group(scores: torch.Tensor, r2g: torch.Tensor) -> torch.Tensor:
    m = torch.full((int(r2g.max()) + 1,), float("-inf"),
                   device=scores.device, dtype=scores.dtype)
    m = m.scatter_reduce(0, r2g, scores.detach(), reduce="amax", include_self=False)
    shifted = scores - m[r2g]
    denom = torch.zeros_like(m).index_add(0, r2g, torch.exp(shifted))
    return shifted - torch.log(denom)[r2g]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grammar-dir", "-g", required=True)
    ap.add_argument("--graph-file", default="train.split.graphs.txt")
    ap.add_argument("--mapping-file", default="train.mapping.txt",
                    help="grammar file inside --grammar-dir. Use this to train over a "
                         "modified inventory (e.g. train.mapping.txt.anonymous). The "
                         "forest cache keys on the grammar hash, so a different "
                         "inventory needs its own --cache-dir.")
    ap.add_argument("--cache-dir")
    ap.add_argument("--output-dir", "-o", required=True)
    ap.add_argument("--method", default="em",
                    choices=["em", "uniform", "neural", "flat", "backoff", "gd"])
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="symmetric Dirichlet alpha for --method flat, or the additive "
                         "term on top of the backoff distribution")
    ap.add_argument("--backoff-kind", default="continuation",
                    choices=["mass", "per_member", "continuation"])
    ap.add_argument("--discount", type=float, default=0.75,
                    help="absolute discount for --method backoff")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--residual-decay", type=float, default=0.0,
                    help="L2 penalty applied ONLY to the per-rule residual. With the residual "
                         "free, structural+residual is just a reparameterized lookup table and "
                         "the encoder can be ignored; penalising it forces the encoder to "
                         "explain what it can and leaves the residual as a correction.")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="clip global grad norm; 0 disables. The neural LL oscillates hard "
                         "(single steps swing >15k nats), and a run that happens to END on a "
                         "spike saves the spiked weights.")
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear LR warmup over this many steps, to avoid destructive early "
                         "updates from a randomly initialised encoder")
    ap.add_argument("--keep-best", action="store_true",
                    help="restore the parameters from the step with the best TRAINING "
                         "likelihood instead of keeping whatever the last step produced. Uses "
                         "no held-out data, so it is model selection, not leakage. Off by "
                         "default so previously reported runs stay reproducible.")
    ap.add_argument("--gnn-layers", type=int, default=2,
                    help="message-passing layers for --structural gnn")
    ap.add_argument("--rule-residual", action="store_true",
                    help="add a free per-rule scalar on top of the structural score, so the "
                         "encoder acts as a shared prior instead of a hard constraint")
    ap.add_argument("--structural", choices=["none", "gnn", "serialized_seq", "serialized_bag"],
                    default="none",
                    help="rule-embedding source for --method neural. 'none' is a free per-rule "
                         "lookup (no sharing); the others tie rules through their RHS structure. "
                         "Measured cost per step on pool4k/GPU: serialized_seq 0.07 s, "
                         "serialized_bag 2.5 s, gnn 155 s -- the GNN is a per-rule Python loop "
                         "and is only usable at low step counts.")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--device", default="cpu",
                    help="cpu is usually fastest: at ~23 depth levels the batched "
                         "inside is memory-bound and CUDA gives no speedup")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-7,
                    help="EM relative stopping tolerance")
    ap.add_argument("--smoothing", type=float, default=1e-10,
                    help="added to expected counts before the M-step. Matches "
                         "EMConfig.smoothing. Without it a rule that reaches exactly "
                         "zero count becomes -inf, which makes every forest needing it "
                         "unscoreable, which zeroes more rules: a death spiral that "
                         "empties the corpus by iteration 8.")
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grammar_file = f"{args.grammar_dir}/{args.mapping_file}"
    graphs_file = f"{args.grammar_dir}/{args.graph_file}"

    t0 = time.time()
    grammar = GrammarWrapper(grammar_file)
    forests, ids, _sents, num_rules, _cache = load_forests_with_cache(
        grammar_file=grammar_file, graphs_file=graphs_file,
        cache_dir=args.cache_dir or f"{args.grammar_dir}/.cache",
        batch_cache=True, verbose=False, timeout=args.timeout)
    ids = [i for i, f in zip(ids, forests) if f.roots]
    forests = [f for f in forests if f.roots]
    t_load = time.time() - t0

    device = torch.device(args.device)
    t0 = time.time()
    batch = BatchedForest.from_forests(forests).to(device)
    t_flat = time.time() - t0
    r2g = group_ids(grammar, num_rules).to(device)
    n_groups = int(r2g.max()) + 1
    print(f"  {len(forests)} forests, {batch.num_nodes:,} nodes, "
          f"{len(batch.edge_rule):,} hyperedges, {batch.num_levels} levels", flush=True)
    print(f"  {num_rules} rules in {n_groups} (LHS, arity) groups", flush=True)
    print(f"  load {t_load:.1f}s, flatten {t_flat:.1f}s", flush=True)

    history = []
    lls = []
    t_train = time.time()

    if args.method == "uniform":
        w = uniform_weights(r2g.cpu(), num_rules)
        history.append(w.clone())
        lls.append(float(batch.total_log_likelihood(w.to(device))))

    elif args.method == "em":
        w = uniform_weights(r2g.cpu(), num_rules).to(device)
        prev = None
        for it in range(args.steps):
            x = w.detach().clone().requires_grad_(True)
            ll = batch.total_log_likelihood(x)
            ll.backward()
            counts = x.grad.detach() + args.smoothing
            total = torch.zeros(n_groups, device=device, dtype=counts.dtype)
            total = total.index_add(0, r2g, counts)
            with torch.no_grad():
                w = torch.log(counts) - torch.log(total)[r2g]
                w = torch.nan_to_num(w, nan=float("-inf"), neginf=float("-inf"))
            cur = float(ll.detach())
            lls.append(cur)
            history.append(w.detach().cpu().clone())
            if prev is not None and abs(cur - prev) < abs(prev) * args.tol:
                print(f"  EM converged at iteration {it + 1}", flush=True)
                break
            prev = cur
            if (it + 1) % 10 == 0:
                print(f"  EM iter {it + 1}: LL {cur:,.2f}", flush=True)

    elif args.method in ("flat", "backoff"):
        from graph_grammar.learning.factored import (
            build_signatures, estimate_backoff, estimate_flat)
        w = uniform_weights(r2g.cpu(), num_rules).to(device)
        prev = None
        for it in range(args.steps):
            x = w.detach().clone().requires_grad_(True)
            ll = batch.total_log_likelihood(x)
            ll.backward()
            counts = x.grad.detach() + args.smoothing
            total = torch.zeros(n_groups, device=device, dtype=counts.dtype)
            total = total.index_add(0, r2g, counts)
            with torch.no_grad():
                w = torch.nan_to_num(torch.log(counts) - torch.log(total)[r2g],
                                     nan=float("-inf"), neginf=float("-inf"))
            cur = float(ll.detach())
            lls.append(cur)
            if prev is not None and abs(cur - prev) < abs(prev) * args.tol:
                break
            prev = cur
        x = w.detach().clone().requires_grad_(True)
        batch.total_log_likelihood(x).backward()
        counts = x.grad.detach().cpu()
        if args.method == "flat":
            weights = estimate_flat(counts, grammar, alpha=args.alpha)
        else:
            sig_of_rule, rules_of_sig = build_signatures(grammar)
            weights = estimate_backoff(counts, grammar, sig_of_rule, rules_of_sig,
                                       alpha=args.alpha, discount=args.discount,
                                       backoff_kind=args.backoff_kind)
        history.append(weights.double())
        print(f"  {args.method}: {int(torch.isfinite(weights).sum())}/{num_rules} rules "
              f"with non-zero probability", flush=True)

    elif args.method == "gd":
        logits = torch.zeros(num_rules, dtype=torch.float64, device=device,
                             requires_grad=True)
        opt = torch.optim.Adam([logits], lr=args.lr)
        every = max(1, args.steps // 20)
        prev, flat = None, 0
        for step in range(args.steps):
            opt.zero_grad()
            ll = batch.total_log_likelihood(normalize_by_group(logits, r2g))
            (-ll).backward()
            opt.step()
            cur = float(ll.detach())
            lls.append(cur)
            if step % every == 0:
                history.append(normalize_by_group(logits.detach(), r2g).cpu().clone())
            if step == 0 or (step + 1) % every == 0:
                print(f"  GD step {step + 1}: LL {cur:,.2f}", flush=True)
            if args.tol > 0 and prev is not None:
                flat = flat + 1 if cur - prev < abs(prev) * args.tol else 0
                if flat >= 10:
                    print(f"  GD plateaued at step {step + 1}", flush=True)
                    break
            prev = cur
        history.append(normalize_by_group(logits.detach(), r2g).cpu().clone())
        lls.append(float(batch.total_log_likelihood(history[-1].to(device))))

    else:
        from graph_grammar.learning.compound import CompoundRuleScorer
        struct = None
        if args.structural != "none":
            from graph_grammar.learning.compound import (CompoundConfig,
                                                         create_structural_embedder)
            struct = create_structural_embedder(CompoundConfig(
                use_structural_embeddings=True,
                structural_encoder_type=args.structural,
            structural_gnn_layers=args.gnn_layers,
                grammar_file=grammar_file,
                structural_embed_dim=args.embedding_dim,
                structural_hidden_dim=args.hidden_dim), device)
            print(f"  structural embedder: {args.structural}", flush=True)
        scorer = CompoundRuleScorer(
            num_rules=num_rules, num_labels=n_groups, rule_to_label=r2g,
            embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
            latent_dim=0, num_layers=2, dropout=0.0, device=device,
            structural_embedder=struct,
            rule_residual=args.rule_residual,
            min_log_prob=-1e30,
            softmax_temperature=1.0,
            adaptive_temperature=False,
        )
        if args.rule_residual and args.residual_decay > 0:
            resid = [p for n, p in scorer.named_parameters() if "rule_residual" in n]
            other = [p for n, p in scorer.named_parameters() if "rule_residual" not in n]
            opt = torch.optim.Adam(
                [{"params": other, "weight_decay": 0.0},
                 {"params": resid, "weight_decay": args.residual_decay}], lr=args.lr)
            print(f"  residual weight decay {args.residual_decay} on "
                  f"{sum(p.numel() for p in resid):,} params", flush=True)
        else:
            opt = torch.optim.Adam(scorer.parameters(), lr=args.lr)
        n_params = sum(p.numel() for p in scorer.parameters())
        print(f"  neural scorer: {n_params:,} parameters", flush=True)
        best, best_state, best_step = float("-inf"), None, -1
        base_lrs = [g["lr"] for g in opt.param_groups]
        for step in range(args.steps):
            if args.warmup > 0 and step < args.warmup:
                scale = (step + 1) / args.warmup
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * scale
            elif args.warmup > 0 and step == args.warmup:
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b
            opt.zero_grad()
            if struct is not None and hasattr(struct, "invalidate_cache"):
                struct.invalidate_cache()
            logp = scorer.get_rule_log_probs().to(torch.float64)
            ll = batch.total_log_likelihood(logp)
            (-ll).backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_clip)
            opt.step()
            cur = float(ll.detach())
            lls.append(cur)
            if cur > best:
                best, best_step = cur, step + 1
                if args.keep_best:
                    best_state = {k: v.detach().clone()
                                  for k, v in scorer.state_dict().items()}
            if step == 0 or (step + 1) % max(1, args.steps // 20) == 0:
                print(f"  [{100*(step+1)/args.steps:5.1f}%] step {step+1}: "
                      f"LL {cur:,.2f}", flush=True)
        if args.keep_best and best_state is not None:
            scorer.load_state_dict(best_state)
            if struct is not None and hasattr(struct, "invalidate_cache"):
                struct.invalidate_cache()
            print(f"  restored best-training-LL parameters from step {best_step} "
                  f"(LL {best:,.2f} vs last {cur:,.2f})", flush=True)
        with torch.no_grad():
            history.append(scorer.get_rule_log_probs().double().cpu().clone())
        torch.save({"state_dict": scorer.state_dict(),
                    "num_rules": num_rules, "num_labels": n_groups},
                   out / "model.pt")

    elapsed = time.time() - t_train

    checkpoints = history if len(history) <= 25 else \
        [history[i] for i in range(0, len(history), max(1, len(history) // 24))] + [history[-1]]
    with open(out / "weight_history", "w") as fh:
        cols = torch.stack(checkpoints, dim=1)
        for i in range(num_rules):
            fh.write(f"{i}," + ",".join(f"{v:.6g}" for v in cols[i].tolist()) + "\n")
    with open(out / "lls", "w") as fh:
        fh.write("\n".join(f"{v:.6f}" for v in lls) + "\n")
    with torch.no_grad():
        per_graph = batch.log_partition(history[-1].to(device)).cpu().tolist()
    with open(out / "graph_ll", "w") as fh:
        fh.write("graph_id\tlog_likelihood\n")
        fh.writelines(f"{g}\t{v:.6f}\n" for g, v in zip(ids, per_graph))
    final_ll = lls[-1] if lls else float("nan")
    json.dump({"method": args.method, "num_rules": num_rules, "num_groups": n_groups,
               "num_forests": len(forests), "steps_run": len(lls),
               "final_log_likelihood": final_ll, "train_seconds": elapsed,
               "load_seconds": t_load, "flatten_seconds": t_flat,
               "device": str(device), "lr": args.lr, "seed": args.seed},
              open(out / "config.json", "w"), indent=2)
    print(f"  {args.method}: final LL {final_ll:,.2f} in {elapsed:.1f}s "
          f"({len(lls)} steps) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

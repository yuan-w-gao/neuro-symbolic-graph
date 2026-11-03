#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph_grammar.generation.grammar import GrammarWrapper

import torch
import torch.nn as nn

from graph_grammar.batched import BatchedForest
from graph_grammar.cache import load_forests_with_cache
from graph_grammar.learning.compound import (CompoundRuleScorer, GraphEncoder,
                                             extract_graph_features)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train import group_ids, normalize_by_group


def batched_rule_log_probs(scorer, z, r2g, chunk=16, checkpoint=True):
    lhs = scorer.lhs_embedding(scorer.rule_to_label)
    if scorer.use_structural:
        rhs = scorer.structural_embedder()
    else:
        rhs = scorer.rhs_embedding(torch.arange(scorer.num_rules, device=scorer.device))
    base = torch.cat([lhs, rhs], dim=-1)
    resid = None
    if getattr(scorer, "use_rule_residual", False):
        resid = scorer.rule_residual(
            torch.arange(scorer.num_rules, device=scorer.device)).squeeze(-1)

    def block(zz, base_):
        c = zz.shape[0]
        comb = torch.cat([base_.unsqueeze(0).expand(c, -1, -1),
                          zz.unsqueeze(1).expand(-1, scorer.num_rules, -1)], dim=-1)
        s = scorer.mlp(comb).squeeze(-1)
        if resid is not None:
            s = s + resid
        return _grouped_log_softmax_batched(s, r2g)

    out = []
    for i in range(0, z.shape[0], chunk):
        zz = z[i:i + chunk]
        if checkpoint and torch.is_grad_enabled():
            out.append(torch.utils.checkpoint.checkpoint(block, zz, base, use_reentrant=False))
        else:
            out.append(block(zz, base))
    return torch.cat(out, 0)


def _grouped_log_softmax_batched(scores, r2g):
    B = scores.shape[0]
    ng = int(r2g.max()) + 1
    idx = r2g.unsqueeze(0).expand(B, -1)
    m = torch.full((B, ng), float("-inf"), device=scores.device, dtype=scores.dtype)
    m = m.scatter_reduce(1, idx, scores.detach(), reduce="amax", include_self=False)
    shifted = scores - m.gather(1, idx)
    denom = torch.zeros((B, ng), device=scores.device, dtype=scores.dtype)
    denom = denom.index_add(1, r2g, torch.exp(shifted))
    return shifted - torch.log(denom).gather(1, idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grammar-dir", "-g", required=True)
    ap.add_argument("--graph-file", default="train.split.graphs.txt")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--graphs", type=int, default=800)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--free-bits", type=float, default=0.0)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--kl-warmup", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--active-threshold", type=float, default=0.01)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out")
    args = ap.parse_args()

    dev = torch.device(args.device)
    gf = f"{args.grammar_dir}/train.mapping.txt"
    grammar = GrammarWrapper(gf)
    R = grammar.num_rules
    r2g = group_ids(grammar, R).to(dev)
    ng = int(r2g.max()) + 1

    forests, _i, _s, _n, _c = load_forests_with_cache(
        grammar_file=gf, graphs_file=f"{args.grammar_dir}/{args.graph_file}",
        cache_dir=args.cache_dir, batch_cache=True, verbose=False)
    forests = [f for f in forests if f.roots]
    forests.sort(key=lambda f: len(f.hyperedges))
    if len(forests) > args.graphs:
        step = len(forests) / args.graphs
        forests = [forests[int(i * step)] for i in range(args.graphs)]
    G = len(forests)

    batch = BatchedForest.from_forests(forests).to(dev)
    node_forest = torch.empty(batch.num_nodes + 1, dtype=torch.long)
    off = 0
    for gi, f in enumerate(forests):
        node_forest[off: off + len(f.nodes)] = gi
        off += len(f.nodes)
    node_forest[batch.num_nodes] = 0
    edge_graph = node_forest.to(dev)[batch.edge_parent]
    zeros_R = torch.zeros(R, dtype=torch.float64, device=dev)
    print(f"{G} graphs, {len(batch.edge_rule):,} hyperedges, {R:,} rules, "
          f"latent {args.latent_dim}, free_bits {args.free_bits}", flush=True)

    feats = torch.stack([extract_graph_features(f, R) for f in forests]).to(dev)
    enc = GraphEncoder(input_dim=R + 5, hidden_dim=args.hidden_dim,
                       latent_dim=args.latent_dim).to(dev)
    scorer = CompoundRuleScorer(num_rules=R, num_labels=ng, rule_to_label=r2g,
                                embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                                latent_dim=args.latent_dim, device=dev,
                                min_log_prob=-1e30, adaptive_temperature=False).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=args.lr)

    hist = []
    for it in range(args.iters):
        t0 = time.time()
        klw = min(1.0, (it + 1) / max(1, args.kl_warmup)) * args.kl_weight
        opt.zero_grad()
        mu, lv = enc(feats)
        lv = lv.clamp(-10, 10)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        kl_d = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
        kl = (torch.clamp(kl_d, min=args.free_bits) if args.free_bits > 0 else kl_d).sum(1)
        logp = batched_rule_log_probs(scorer, z, r2g, args.chunk).double()
        bias = logp[edge_graph, batch.edge_rule]
        lz = batch.log_partition(zeros_R, edge_bias=bias)
        fin = torch.isfinite(lz)
        elbo = (lz[fin] - klw * kl.double()[fin]).mean()
        (-elbo).backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(scorer.parameters()),
                                       1.0, foreach=False)
        opt.step()

        pd = kl_d.detach().mean(0).cpu()
        active = int((pd > args.active_threshold).sum())
        row = {"iter": it + 1, "ll": float(lz[fin].mean().detach()), "kl": float(kl[fin].mean().detach()),
               "active_units": active, "kl_weight": klw,
               "max_dim_kl": float(pd.max()), "mu_spread": float(mu.detach().std(0).mean())}
        hist.append(row)
        if it < 3 or (it + 1) % 10 == 0:
            print(f"  [{it+1:4d}] LL {row['ll']:10.2f}  KL {row['kl']:8.3f}  "
                  f"ACTIVE {active:2d}/{args.latent_dim}  maxKL/dim {row['max_dim_kl']:.3f}  "
                  f"sd(mu) {row['mu_spread']:.4f}  ({time.time()-t0:.2f}s)", flush=True)

    with torch.no_grad():
        zz = torch.randn(16, args.latent_dim, device=dev)
        lp = batched_rule_log_probs(scorer, zz, r2g, args.chunk).double()
        sym = []
        for i in range(0, 16, 2):
            a, b = lp[i], lp[i + 1]
            m = torch.isfinite(a) & torch.isfinite(b)
            sym.append(float(0.5 * ((a[m].exp() * (a[m] - b[m])).sum()
                                    + (b[m].exp() * (b[m] - a[m])).sum())))
        prior_sens = sum(sym) / len(sym)
        mu_p, _ = enc(feats)
        idx = torch.randperm(mu_p.shape[0], device=dev)[:16]
        lp2 = batched_rule_log_probs(scorer, mu_p[idx], r2g, args.chunk).double()
        sym2 = []
        for i in range(0, 16, 2):
            a, b = lp2[i], lp2[i + 1]
            m = torch.isfinite(a) & torch.isfinite(b)
            sym2.append(float(0.5 * ((a[m].exp() * (a[m] - b[m])).sum()
                                     + (b[m].exp() * (b[m] - a[m])).sum())))
        post_sens = sum(sym2) / len(sym2)
    true_kl = float(kl_d.detach().mean(0).sum())
    print(f"\nDECODER SENSITIVITY  prior z pairs {prior_sens:.4f} nats | "
          f"posterior-mean pairs {post_sens:.4f} nats")
    print(f"TRUE total KL/graph (unclamped) {true_kl:.4f}   "
          f"reported/clamped {hist[-1]['kl']:.4f}")
    print("A sensitivity near 0 means the scorer ignores z: the latent is decorative.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args), "history": hist,
                   "decoder_sensitivity_prior": prior_sens,
                   "decoder_sensitivity_posterior": post_sens,
                   "true_total_kl": true_kl}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

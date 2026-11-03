from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from rdkit import Chem

from graph_grammar.molecules.mol_io import BOND_LABEL, atom_label
from graph_grammar.grammar_io import (
    CFGItem, GrammarBuilder, SyntheticEdge, SyntheticRule)

ROOT_LABEL = "ROOT"

# Half-units, so an aromatic bond (1.5) stays integral.
BOND_UNITS: Dict[str, int] = {"SINGLE": 2, "DOUBLE": 4, "TRIPLE": 6, "AROMATIC": 3}


def atom_debt(atom: Chem.Atom) -> int:
    total = 0.0
    for b in atom.GetBonds():
        total += b.GetValenceContrib(atom)
    return int(round(total * 2))


def valence_atom_label(atom: Chem.Atom) -> str:
    charge = atom.GetFormalCharge()
    c = "c0" if charge == 0 else (f"cp{charge}" if charge > 0 else f"cm{-charge}")
    return f"{atom.GetSymbol()}_h{atom.GetTotalNumHs()}_{c}_v{atom_debt(atom)}"


def collect_atom_types(mols: Iterable[Chem.Mol]) -> Dict[str, int]:
    types: Dict[str, int] = {}
    for mol in mols:
        for a in mol.GetAtoms():
            lab, d = valence_atom_label(a), atom_debt(a)
            if types.setdefault(lab, d) != d:
                raise ValueError(
                    f"atom label {lab!r} implies debt {types[lab]} and {d}; the label does "
                    "not determine valence, so validity cannot be guaranteed")
    return types


def nt_label(debts: Sequence[int], adj=0) -> str:
    if not debts:
        return ROOT_LABEL
    if len(debts) == 1:
        return f"N_{debts[0]}"
    body = "N_" + "_".join(str(d) for d in debts)
    if len(debts) == 2:
        bit = adj if isinstance(adj, int) else adj[0]
        return f"{body}_x{bit}"
    return f"{body}_x" + "".join(str(b) for b in adj)


def _pairs(k: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def _canon(debts: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(debts, reverse=True))


def _emit(builder: GrammarBuilder, seen: set, n_nodes: int,
          edges: List[Tuple[str, Tuple[int, ...], bool]],
          survivors: Sequence[Tuple[int, int]], bonded: set) -> None:
    ordered = sorted(survivors, key=lambda p: (-p[1], p[0]))
    externals = [n for n, _d in ordered]
    bits = tuple(1 if frozenset((externals[i], externals[j])) in bonded else 0
                 for i, j in _pairs(len(externals)))
    lhs = nt_label([d for _n, d in ordered], bits if len(externals) != 2 else bits[0])
    key = (lhs, n_nodes, tuple(externals), tuple(sorted(edges)))
    if key in seen:
        return
    seen.add(key)
    builder.add_rule(SyntheticRule(
        node_count=n_nodes,
        edges=[SyntheticEdge(label=lab, node_indices=list(ns), is_terminal=term)
               for lab, ns, term in edges],
        external_nodes=externals,
        cfg_label=lhs,
        cfg_items=[CFGItem(label=lhs, aligned_edge_idx=-1)],
        shrg_index=len(builder.rules),
        current_count=1,
        total_count=1,
    ))


def _boundary_types(max_debt: int, max_arity: int, debt_step: int):
    debts = list(range(debt_step, max_debt + 1, debt_step))
    out = [((d,), (0,)) for d in debts]
    if max_arity >= 2:
        out += [((d1, d2), (b,)) for d1 in debts for d2 in debts if d2 <= d1
                for b in (0, 1)]
    if max_arity >= 3:
        out += [((d1, d2, d3), (b01, b02, b12))
                for d1 in debts for d2 in debts if d2 <= d1 for d3 in debts if d3 <= d2
                for b01 in (0, 1) for b02 in (0, 1) for b12 in (0, 1)]
    return out


def build_valence_grammar(atom_types: Dict[str, int], max_arity: int = 2,
                          bond_labels: Sequence[str] = ("SINGLE", "DOUBLE", "TRIPLE"),
                          debt_step: int = 1, both_orders: bool = False,
                          ) -> GrammarBuilder:
    builder = GrammarBuilder()
    seen: set = set()
    max_debt = max(atom_types.values()) if atom_types else 0
    boundaries = _boundary_types(max_debt, max_arity, debt_step)

    for lab, d in sorted(atom_types.items()):
        _emit(builder, seen, 1, [(lab, (0,), True)], [(0, d)] if d > 0 else [], set())

    for bond in bond_labels:
        t = BOND_UNITS[bond]
        for dA, adjA in boundaries:
            kA = len(dA)
            nA = (nt_label(dA, adjA if kA != 2 else adjA[0]), tuple(range(kA)), False)
            bondedA = {frozenset((i, j)) for (i, j), b in zip(_pairs(kA), adjA) if b}

            for i, j in _pairs(kA):
                if frozenset((i, j)) in bondedA or dA[i] < t or dA[j] < t:
                    continue
                surv = [(n, dA[n] - t if n in (i, j) else dA[n]) for n in range(kA)]
                surv = [(n, d) for n, d in surv if d > 0]
                ends = ([(i, j), (j, i)] if (both_orders or dA[i] == dA[j])
                        else ([(i, j)] if dA[i] > dA[j] else [(j, i)]))
                for u, v in ends:
                    _emit(builder, seen, kA, [nA, (bond, (u, v), True)], surv,
                          bondedA | {frozenset((i, j))})

            for dB, adjB in boundaries:
                off = kA
                kB = len(dB)
                nB = (nt_label(dB, adjB if kB != 2 else adjB[0]),
                      tuple(range(off, off + kB)), False)
                bondedB = {frozenset((off + i, off + j))
                           for (i, j), b in zip(_pairs(kB), adjB) if b}
                for i in range(kA):
                    for j in range(kB):
                        if dA[i] < t or dB[j] < t:
                            continue
                        surv = ([(n, dA[n]) for n in range(kA) if n != i] +
                                [(off + n, dB[n]) for n in range(kB) if n != j])
                        if dA[i] - t > 0:
                            surv.append((i, dA[i] - t))
                        if dB[j] - t > 0:
                            surv.append((off + j, dB[j] - t))
                        if len(surv) > max_arity:
                            continue
                        bonded = bondedA | bondedB | {frozenset((i, off + j))}
                        # Graphs order a bond by total debt, a rule only knows the remaining debt;
                        # both_orders emits both orientations so the two can't disagree.
                        if both_orders or dA[i] == dB[j]:
                            orders = [(i, off + j), (off + j, i)]
                        elif dA[i] > dB[j]:
                            orders = [(i, off + j)]
                        else:
                            orders = [(off + j, i)]
                        for u, v in orders:
                            _emit(builder, seen, off + kB,
                                  [nA, nB, (bond, (u, v), True)], surv, bonded)

    return builder

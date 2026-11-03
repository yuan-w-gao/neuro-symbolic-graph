from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from rdkit import Chem, RDLogger

from graph_grammar.grammar_io import SyntheticGraph

RDLogger.DisableLog("rdApp.*")

BOND_LABEL = {
    Chem.BondType.SINGLE: "SINGLE",
    Chem.BondType.DOUBLE: "DOUBLE",
    Chem.BondType.TRIPLE: "TRIPLE",
    Chem.BondType.AROMATIC: "AROMATIC",
}
LABEL_BOND = {v: k for k, v in BOND_LABEL.items()}


def atom_label(atom: Chem.Atom) -> str:
    charge = atom.GetFormalCharge()
    c = "c0" if charge == 0 else (f"cp{charge}" if charge > 0 else f"cm{-charge}")
    return (f"{atom.GetSymbol()}_h{atom.GetTotalNumHs()}_{c}"
            f"_a{int(atom.GetIsAromatic())}")


def parse_atom_label(label: str) -> Tuple[str, int, int, bool]:
    sym, h, c, last = label.split("_")
    charge = 0 if c == "c0" else (int(c[2:]) if c[1] == "p" else -int(c[2:]))
    aromatic = last[0] == "a" and bool(int(last[1:]))
    return sym, int(h[1:]), charge, aromatic


def kekulize(mol: Chem.Mol) -> Optional[Chem.Mol]:
    m = Chem.Mol(mol)
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:
        return None
    return m


@dataclass
class MolFilter:
    max_heavy_atoms: int = 20
    min_heavy_atoms: int = 3
    single_fragment: bool = True
    allowed_elements: Optional[frozenset] = frozenset(
        {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I"})
    allow_fused_rings: bool = True

    def accepts(self, mol: Chem.Mol) -> bool:
        n = mol.GetNumAtoms()
        if not (self.min_heavy_atoms <= n <= self.max_heavy_atoms):
            return False
        if self.single_fragment and len(Chem.GetMolFrags(mol)) > 1:
            return False
        if self.allowed_elements is not None:
            for a in mol.GetAtoms():
                if a.GetSymbol() not in self.allowed_elements:
                    return False
        for b in mol.GetBonds():
            if b.GetBondType() not in BOND_LABEL:
                return False
        if not self.allow_fused_rings:
            ri = mol.GetRingInfo()
            for i, r1 in enumerate(ri.BondRings()):
                for r2 in ri.BondRings()[i + 1:]:
                    if set(r1) & set(r2):
                        return False
        return True


def mol_to_graph(mol: Chem.Mol, sentence_id: str,
                 order_by: str = "label", label_fn=None) -> SyntheticGraph:
    label_fn = label_fn or atom_label
    nodes = [(f"a{a.GetIdx()}", label_fn(a)) for a in mol.GetAtoms()]
    # Hyperedge matching is positional, so bond endpoints must be ordered the way the grammar was
    # built ("label" for decompose.py, "debt" for valence.py) or nothing parses.
    if order_by == "debt":
        from graph_grammar.molecules.valence import atom_debt
        def key(i: int):
            a = mol.GetAtomWithIdx(i)
            return (-atom_debt(a), label_fn(a), i)
    elif order_by == "label":
        def key(i: int):
            return (label_fn(mol.GetAtomWithIdx(i)), i)
    else:
        raise ValueError(f"unknown order_by {order_by!r}")

    edges = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if key(v) < key(u):
            u, v = v, u
        edges.append((u, v, BOND_LABEL[b.GetBondType()]))
    smiles = Chem.MolToSmiles(mol)
    return SyntheticGraph(
        sentence_id=sentence_id,
        nodes=nodes,
        edges=edges,
        sentence=smiles,  # LoadEdsGraph fails on an empty sentence line
        lemma_sequence=" ".join(lbl for _i, lbl in nodes),
        top_index=0,
    )


def read_smiles(path: str, limit: Optional[int] = None,
                mol_filter: Optional[MolFilter] = None) -> Iterator[Tuple[str, Chem.Mol]]:
    kept = 0
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            smi = parts[0]
            mol_id = parts[1] if len(parts) > 1 else str(lineno)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if mol_filter is not None and not mol_filter.accepts(mol):
                continue
            yield f"nci-{mol_id}", mol
            kept += 1
            if limit is not None and kept >= limit:
                return


def graph_to_mol(node_labels: List[str],
                 edges: List[Tuple[int, int, str]]) -> Optional[Chem.Mol]:
    rw = Chem.RWMol()
    for label in node_labels:
        sym, n_h, charge, arom = parse_atom_label(label)
        a = Chem.Atom(sym)
        a.SetNumExplicitHs(n_h)
        a.SetNoImplicit(True)
        a.SetFormalCharge(charge)
        a.SetIsAromatic(arom)
        rw.AddAtom(a)
    placed = set()
    for src, tgt, label in edges:
        bt = LABEL_BOND.get(label)
        if bt is None:
            return None
        u, v = int(src), int(tgt)
        # The valence grammar can generate parallel bonds (its nonterminals don't record which
        # boundary atoms are already bonded); a molecule can't have them.
        if u == v or (u, v) in placed or (v, u) in placed:
            return None
        placed.add((u, v))
        rw.AddBond(u, v, bt)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol

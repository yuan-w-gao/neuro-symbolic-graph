from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, BRICS, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


DATASETS: Dict[str, Tuple[str, str, str, str]] = {
    "esol": ("data/moleculenet/delaney-processed.csv", "smiles",
             "measured log solubility in mols per litre", "regression"),
    "freesolv": ("data/moleculenet/SAMPL.csv", "smiles", "expt", "regression"),
    "lipo": ("data/moleculenet/Lipophilicity.csv", "smiles", "exp", "regression"),
    "bbbp": ("data/moleculenet/BBBP.csv", "smiles", "p_np", "classification"),
    "bace": ("data/moleculenet/bace.csv", "mol", "Class", "classification"),
}


def load_dataset(name: str) -> Tuple[List[str], np.ndarray, str]:
    import csv

    path, smi_col, y_col, task = DATASETS[name]
    smiles, ys = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            s = row[smi_col].strip()
            if not s or Chem.MolFromSmiles(s) is None:
                continue
            try:
                y = float(row[y_col])
            except (TypeError, ValueError):
                continue
            smiles.append(s)
            ys.append(y)
    return smiles, np.asarray(ys, dtype=np.float64), task


def scaffold_split(smiles: Sequence[str], frac_train: float = 0.8,
                   frac_valid: float = 0.1, seed: int = 0
                   ) -> Tuple[List[int], List[int], List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = ""
        groups.setdefault(scaf, []).append(i)

    rng = np.random.default_rng(seed)
    sets = list(groups.values())
    order = sorted(range(len(sets)), key=lambda k: (-len(sets[k]), rng.random()))

    n = len(smiles)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    train, valid, test = [], [], []
    for k in order:
        g = sets[k]
        if len(train) + len(g) <= n_train:
            train += g
        elif len(valid) + len(g) <= n_valid:
            valid += g
        else:
            test += g
    return train, valid, test


def ecfp4(mols: Sequence[Chem.Mol], n_bits: int = 2048) -> np.ndarray:
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits)
    out = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        out[i] = np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.float32)
    return out


def atom_pair(mols: Sequence[Chem.Mol], n_bits: int = 2048) -> np.ndarray:
    gen = AllChem.GetAtomPairGenerator(fpSize=n_bits)
    out = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        out[i] = np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.float32)
    return out


def maccs(mols: Sequence[Chem.Mol]) -> np.ndarray:
    out = np.zeros((len(mols), 167), dtype=np.float32)
    for i, m in enumerate(mols):
        out[i] = np.asarray(MACCSkeys.GenMACCSKeys(m), dtype=np.float32)
    return out


_DESC = [(n, f) for n, f in Descriptors.descList]


def rdkit_descriptors(mols: Sequence[Chem.Mol]) -> np.ndarray:
    out = np.zeros((len(mols), len(_DESC)), dtype=np.float64)
    for i, m in enumerate(mols):
        for j, (_n, f) in enumerate(_DESC):
            try:
                v = f(m)
            except Exception:
                v = 0.0
            out[i, j] = v if np.isfinite(v) else 0.0
    return out


def brics_fragments(mols: Sequence[Chem.Mol], vocab: List[str] | None = None,
                    train_idx: Sequence[int] | None = None,
                    max_vocab: int = 2048) -> Tuple[np.ndarray, List[str]]:
    decomp = []
    for m in mols:
        try:
            decomp.append(set(BRICS.BRICSDecompose(m)))
        except Exception:
            decomp.append(set())
    if vocab is None:
        from collections import Counter
        c: Counter = Counter()
        idx = range(len(mols)) if train_idx is None else train_idx
        for i in idx:
            c.update(decomp[i])
        vocab = [f for f, _n in c.most_common(max_vocab)]
    pos = {f: j for j, f in enumerate(vocab)}
    out = np.zeros((len(mols), len(vocab)), dtype=np.float32)
    for i, frags in enumerate(decomp):
        for f in frags:
            j = pos.get(f)
            if j is not None:
                out[i, j] = 1.0
    return out, vocab

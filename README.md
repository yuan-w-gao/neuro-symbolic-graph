# Graph grammar induction

Learning rule weights for synchronous hyperedge replacement grammars (SHRG). A C++ parser
builds a packed derivation forest for each graph; Python code runs inside/outside, trains
weights (EM, gradient, neural and compound models) and evaluates them (held-out likelihood,
parsing F1, derivation entropy, BLEU, graph-generation metrics). Two domains: semantic graphs
(CHILDES, Redwoods) and molecules (ZINC, MoleculeNet).

## Layout

```
parser/          C++ SHRG parser, vendored (see parser/README.md); builds the pyshrg module
graph_grammar/   Python library
  forest.py, export.py, cache.py, batched.py   packed forests, C++ export, on-disk forest cache
  inference.py   inside, outside, viterbi, sampling, entropy (log space, autograd)
  learning/      EM, gradient, neural, compound and structural rule models
  evaluation/    entropy, BLEU, parsing F1, graph-generation metrics
  generation/    top-down derivation sampling and graph writing
  data/          graph I/O and train/test splits
  molecules/     SMILES <-> graph, valence grammar, MoleculeNet features
  tests/
baselines/       GraphRNN and discrete-diffusion baselines for graph generation
extraction/      grammar extraction helpers (need PyDelphin, separate env)
```

Grammars, data and results are not in the repo.

## Setup

```bash
pip install -r requirements.txt     # rdkit / scikit-learn only needed for molecules

cd parser && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON=ON -DPython_EXECUTABLE="$(which python)"
cmake --build . -j --target pyshrg comprehensive_eval get_deriv compute_entropy_dp
cd ../..
# link the module into the repo root so `import pyshrg` works
lib=$(find parser/build -maxdepth 1 -name '*pyshrg*' \( -name '*.so' -o -name '*.dylib' \) | head -1)
ln -sf "$lib" "pyshrg$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

python -m pytest graph_grammar/tests -q
```

On newer GCC add `-DCMAKE_CXX_FLAGS="-D_HAS_AUTO_PTR_ETC=0"` to the cmake call (boost's
`hash.hpp`). If pyshrg fails to load inside conda, put the system libstdc++ first on
`LD_LIBRARY_PATH`. Import `pyshrg` before torch/rdkit, and sklearn before rdkit, or the
interpreter may segfault.

## Grammar format

A grammar directory holds `train.mapping.txt` (rules) and `train.graphs.txt` (graphs; the
count is on the first line), plus optionally `train.derivations.p` (gold derivations) and
train/test splits.

## Usage

```python
import pyshrg  # first
from graph_grammar import load_forests_with_cache, inside
from graph_grammar.learning import train_em, EMConfig

forests, ids, sentences, num_rules, _ = load_forests_with_cache(
    "G/train.mapping.txt", "G/train.graphs.txt", cache_dir="G/.cache")

result = train_em(forests, num_rules, EMConfig(max_iterations=50))
log_z = inside(forests[0], result.rule_log_probs)[1]
```

Parsing is the slow step, so forests are cached under `cache_dir` and reused.
Without further arguments EM normalizes over all rules at once. To normalize per
(LHS label, arity) group, pass `rule_to_label`, a LongTensor with the group id of each rule.

## Things to know

- Rule indices in forests and weight files are in HRG space. The C++ `comprehensive_eval`
  (BLEU) reads SHRG space. The two disagree on almost every rule, so map weights across
  before scoring generation (`GrammarWrapper.shrg_to_hrg_index` in `graph_grammar/generation`).
- `comprehensive_eval` is not deterministic (unseeded RNG for the baseline, tie-breaking in
  EM). On ~100 graphs BLEU moves by about ±0.006 between runs.
- About 30 tests need a grammar under `grammars/` (e.g. `grammars/childes_cxg/06`) and skip
  without one. With data present, `test_structural_embeddings.py::TestCreateStructuralEmbedder::test_gnn_type`
  fails: it expects the GNN embedder class from before batching.

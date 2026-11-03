#!/bin/bash
# Realization BLEU for one weight file through the C++ evaluator.
#
#   bash scripts/eval_bleu.sh GRAMMAR_DIR GRAPHS_FILE WEIGHT_FILE OUT_DIR
#
# Trainers write weights in HRG index space; comprehensive_eval reads SHRG space,
# so the weights are mapped first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
G="$1"; GRAPHS="$2"; W="$3"; OUT="$4"
mkdir -p "$OUT"
"$PYTHON" "$ROOT/scripts/map_weights_to_shrg.py" -g "$G/train.mapping.txt" -w "$W" \
    --mode split -o "$OUT/weights.shrg"
"$ROOT/parser/build/comprehensive_eval" linear "$G/train.mapping.txt" "$GRAPHS" "$OUT" \
    "$OUT/weights.shrg" > "$OUT/log.txt" 2>&1
grep -o '"mean_bleu_[a-z]*": [0-9.]*' "$OUT/summary.json"

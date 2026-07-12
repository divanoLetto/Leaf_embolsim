#!/usr/bin/env bash
# run_method.sh — Full pipeline: train → predict → evaluate
#
# Run from project root:
#   bash src/method/run_method.sh
#   bash src/method/run_method.sh --epochs 200 --batch-size 8
#
# All extra flags are forwarded to train.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -d "${REPO_ROOT}/data" ]]; then
    echo "[ERROR] Could not locate data/ directory under ${REPO_ROOT}"
    exit 1
fi

cd "${REPO_ROOT}"

if [[ ! -f "test.txt" ]]; then
    echo "[ERROR] test.txt not found. Run: python src/make_splits.py"
    exit 1
fi
mapfile -t TEST_SEQS < <(grep -v '^\s*#' test.txt | grep -v '^\s*$')

if [[ ${#TEST_SEQS[@]} -eq 0 ]]; then
    echo "[ERROR] test.txt is empty."
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  U-Net pipeline — train → predict → evaluate"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Test sequences (${#TEST_SEQS[@]}):"
for seq in "${TEST_SEQS[@]}"; do echo "    ${seq}"; done
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1 / 3 — Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 src/method/train.py "${@}"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2 / 3 — Inference on test set (${#TEST_SEQS[@]} sequences)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for seq in "${TEST_SEQS[@]}"; do
    echo "  → ${seq}"
    python3 src/method/predict.py --sequence "${seq}"
done
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3 / 3 — Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 src/method/evaluate.py --all
echo ""

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pipeline complete.                                      ║"
echo "║  Outputs:     src/method/outputs/<sequence>/           ║"
echo "║  Evaluation:  src/method/evaluation/<sequence>/        ║"
echo "║  Aggregate:   src/method/evaluation/summary_avg.txt    ║"
echo "║  Checkpoints: src/method/checkpoints/                  ║"
echo "╚══════════════════════════════════════════════════════════╝"

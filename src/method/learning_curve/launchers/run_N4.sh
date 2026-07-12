#!/usr/bin/env bash
# Lancia 10 seed (0 1 2 3 4 5 6 7 8 9) per N=4.
# Iperparametri: batch-size=64, lr=4e-4.
# Argomenti extra dopo "--" sono pass-through a run_one.py.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/_template.sh" 4 0 1 2 3 4 5 6 7 8 9 -- --batch-size 64 --lr 4e-4 "$@"

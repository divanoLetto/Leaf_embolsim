#!/usr/bin/env bash
# Lancia 8 seed (0 1 2 3 4 5 6 7) per N=12.
# Iperparametri: batch-size=64, lr=4e-4.
# Argomenti extra dopo "--" sono pass-through a run_one.py.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/_template.sh" 12 0 1 2 3 4 5 6 7 -- --batch-size 64 --lr 4e-4 "$@"

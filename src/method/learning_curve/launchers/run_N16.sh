#!/usr/bin/env bash
# Lancia 6 seed (0 1 2 3 4 5) per N=16.
# Iperparametri: batch-size=64, lr=4e-4.
# Argomenti extra dopo "--" sono pass-through a run_one.py.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/_template.sh" 16 0 1 2 3 4 5 -- --batch-size 64 --lr 4e-4 "$@"

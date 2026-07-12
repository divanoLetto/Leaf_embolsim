#!/usr/bin/env bash
# Lancia 3 seed (0 1 2) per N=32.
# Iperparametri: batch-size=64, lr=4e-4.
# Argomenti extra dopo "--" sono pass-through a run_one.py.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/_template.sh" 32 0 1 2 -- --batch-size 64 --lr 4e-4 "$@"

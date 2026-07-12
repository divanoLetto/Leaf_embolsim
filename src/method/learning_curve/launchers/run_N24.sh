#!/usr/bin/env bash
# Lancia 4 seed (0 1 2 3) per N=24.
# Iperparametri: batch-size=64, lr=4e-4.
# Argomenti extra dopo "--" sono pass-through a run_one.py.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/_template.sh" 24 0 1 2 3 -- --batch-size 64 --lr 4e-4 "$@"

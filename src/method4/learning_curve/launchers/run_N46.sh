#!/usr/bin/env bash
# Lancia 1 seed (0) per N=46.
# Iperparametri: batch-size=64, lr=4e-4.
# Argomenti extra dopo "--" sono pass-through a run_one.py.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/_template.sh" 46 0 -- --batch-size 64 --lr 4e-4 "$@"

#!/usr/bin/env bash
# Template — non eseguire direttamente. Vedi run_N{N}.sh.
#
# Uso interno (dai wrapper):
#   bash _template.sh <N> <seed1> [seed2] [seed3] ... [-- extra args to run_one.py]
#
# Per ogni (N, seed):
#   - se runs/N{N}_seed{S}/evaluation/summary_avg.csv esiste, run_one.py salta;
#   - altrimenti allena, predice, valuta, e salva wp_metric.json.
#
# Output:
#   - risultati per-run    : src/method/learning_curve/runs/N{N}_seed{S}/
#   - log testuale per-run : src/method/learning_curve/logs/N{N}_seed{S}.log
#
# Sicuro da killare: al rilancio i run già finiti vengono saltati.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LC_DIR}/../../.." && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: bash _template.sh <N> <seed1> [seed2 ...] [-- extra args]" >&2
    exit 2
fi

N="$1"; shift

SEEDS=()
EXTRA=()
parsing_seeds=1
for arg in "$@"; do
    if [[ "${arg}" == "--" ]]; then
        parsing_seeds=0
        continue
    fi
    if [[ ${parsing_seeds} -eq 1 ]]; then
        SEEDS+=("${arg}")
    else
        EXTRA+=("${arg}")
    fi
done

cd "${REPO_ROOT}"
mkdir -p "${LC_DIR}/logs"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
printf "║  Learning curve — N=%-3d   seeds: %-18s ║\n" "${N}" "${SEEDS[*]}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

START_ALL=$(date +%s)

for S in "${SEEDS[@]}"; do
    LOG="${LC_DIR}/logs/N${N}_seed${S}.log"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  N=${N}  seed=${S}   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  log → ${LOG}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    START=$(date +%s)
    python3 src/method/learning_curve/run_one.py --n "${N}" --seed "${S}" "${EXTRA[@]}" 2>&1 | tee "${LOG}"
    RC=${PIPESTATUS[0]}
    END=$(date +%s)
    ELAPSED=$(( END - START ))
    if [[ ${RC} -ne 0 ]]; then
        echo "[ERROR] N=${N} seed=${S} fallito (rc=${RC}) dopo ${ELAPSED}s. Vedi ${LOG}."
        echo "        Proseguo con il prossimo seed."
    else
        echo "[OK]    N=${N} seed=${S} completato in ${ELAPSED}s."
    fi
    echo ""
done

END_ALL=$(date +%s)
ELAPSED_ALL=$(( END_ALL - START_ALL ))
echo "╔══════════════════════════════════════════════════════════╗"
printf "║  N=%-3d  finito.  Tempo totale: %5d min (%4.1f h)        ║\n" \
    "${N}" $(( ELAPSED_ALL / 60 )) "$(awk "BEGIN{printf \"%.1f\", ${ELAPSED_ALL}/3600}")"
echo "╚══════════════════════════════════════════════════════════╝"

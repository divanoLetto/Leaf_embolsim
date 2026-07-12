#!/usr/bin/env bash
# Lancia TUTTI i launcher N=4 → N=46 in sequenza.
# Schema K(N)=10/8/8/6/4/3/3/1 (43 run totali).
# Iperparametri: batch-size=64, lr=4e-4.
#
# Set-and-forget. Sicuro da killare: ogni run_one.py salta se già completato.
#
# Stima budget (training tempo ~lineare in N, accelerato ~3x vs bs=16):
#   N=4  ×10 ≈  2.5h     N=24 ×4  ≈  4h
#   N=8  ×8  ≈  3h       N=32 ×3  ≈  4h
#   N=12 ×8  ≈  4h       N=40 ×3  ≈  5h
#   N=16 ×6  ≈  4h       N=46 ×1  ≈  2h
#   Totale: ~28h (~1.2 giorni di GPU).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START=$(date +%s)
for N in 4 8 12 16 24 32 40 46; do
    echo ""
    echo "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"
    echo "▓ Blocco N=${N}   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"
    bash "${SCRIPT_DIR}/run_N${N}.sh" "$@"
done
END=$(date +%s)
ELAPSED=$(( END - START ))
echo ""
echo "TUTTO FINITO. Tempo totale: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))min"
echo "Lancia ora: python src/method4/learning_curve/aggregate.py"

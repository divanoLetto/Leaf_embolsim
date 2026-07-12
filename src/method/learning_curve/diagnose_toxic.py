"""
diagnose_toxic.py — Fase A: identifica sequenze di training potenzialmente
"tossiche" usando solo i dati già prodotti dai 22 run di learning curve.

Tre analisi:

  1. Leave-one-in (LOI) dai 22 run: per ogni sequenza S del pool di training,
     confronta mean(wp_err) quando S è dentro vs quando è fuori. Δ_S grande e
     positivo = S "fa male" quando viene aggiunta. Solo sequenze con almeno
     3 run dentro E 3 run fuori (altrimenti il segnale è troppo debole).

  2. Run-pair diff: per ogni coppia (N, seed) consecutiva nello stesso seed
     (es. N=8 seed=2 → N=12 seed=2) in cui wp_err peggiora di molto, isola
     le sequenze "nuove arrivate" tra i due subset. Candidate dirette.

  3. Sospetti "uniti": intersezione tra le due liste = candidati più solidi.

Output (in learning_curve/diagnostics/):
  - loi_ranking.csv
  - run_pair_diffs.csv
  - shortlist.txt   (human-readable, sospetti ordinati per evidenza)

Run (da repo root):
  python src/method/learning_curve/diagnose_toxic.py
"""

import csv
import os
import sys
from collections import defaultdict
from statistics import mean, stdev

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUBSETS  = os.path.join(THIS_DIR, "subsets")
CSV_PATH = os.path.join(THIS_DIR, "learning_curve.csv")
OUT_DIR  = os.path.join(THIS_DIR, "diagnostics")


def load_runs() -> list[dict]:
    """Carica learning_curve.csv, parsando wp_err numerico."""
    rows = []
    with open(CSV_PATH) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "n":      int(r["n"]),
                    "seed":   int(r["seed"]),
                    "wp_err": float(r["abs_wp_error"]),
                    "iou":    float(r["iou_event"]),
                })
            except (KeyError, ValueError, TypeError):
                continue
    return rows


def load_subset(n: int, seed: int) -> set[str]:
    path = os.path.join(SUBSETS, f"train_N{n}_seed{seed}.txt")
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")}


def all_pool_sequences() -> set[str]:
    """Pool = unione di tutti i subset (= tutto train.txt)."""
    seqs = set()
    for f in os.listdir(SUBSETS):
        if f.endswith(".txt"):
            with open(os.path.join(SUBSETS, f)) as fh:
                seqs.update(ln.strip() for ln in fh if ln.strip())
    return seqs


# --------------------------------------------------------------------------
# Analysis 1: leave-one-in
# --------------------------------------------------------------------------

def leave_one_in(runs: list[dict], pool: set[str]) -> list[dict]:
    # Per ogni run carico il suo subset una volta sola
    run_subsets = {(r["n"], r["seed"]): load_subset(r["n"], r["seed"]) for r in runs}

    results = []
    for seq in sorted(pool):
        wp_in, wp_out = [], []
        for r in runs:
            sub = run_subsets[(r["n"], r["seed"])]
            (wp_in if seq in sub else wp_out).append(r["wp_err"])
        if len(wp_in) < 3 or len(wp_out) < 3:
            continue
        m_in,  m_out  = mean(wp_in),  mean(wp_out)
        results.append({
            "sequence": seq,
            "n_in":     len(wp_in),
            "n_out":    len(wp_out),
            "mean_err_in":  round(m_in,  5),
            "mean_err_out": round(m_out, 5),
            "delta":        round(m_in - m_out, 5),
            "std_in":   round(stdev(wp_in)  if len(wp_in)  > 1 else 0.0, 5),
            "std_out":  round(stdev(wp_out) if len(wp_out) > 1 else 0.0, 5),
        })
    # Δ positivo grande = tossico
    results.sort(key=lambda d: d["delta"], reverse=True)
    return results


# --------------------------------------------------------------------------
# Analysis 2: run-pair diff (within same seed, consecutive N)
# --------------------------------------------------------------------------

def run_pair_diffs(runs: list[dict]) -> list[dict]:
    by_seed = defaultdict(list)
    for r in runs:
        by_seed[r["seed"]].append(r)
    for s in by_seed:
        by_seed[s].sort(key=lambda r: r["n"])

    pairs = []
    for seed, rs in by_seed.items():
        for i in range(len(rs) - 1):
            a, b = rs[i], rs[i + 1]
            delta = b["wp_err"] - a["wp_err"]
            sub_a = load_subset(a["n"], seed)
            sub_b = load_subset(b["n"], seed)
            newly = sorted(sub_b - sub_a)
            pairs.append({
                "seed":      seed,
                "n_from":    a["n"],
                "n_to":      b["n"],
                "err_from":  round(a["wp_err"], 5),
                "err_to":    round(b["wp_err"], 5),
                "delta_err": round(delta, 5),
                "n_added":   len(newly),
                "added":     newly,
            })
    # Le coppie che peggiorano di più sono in testa
    pairs.sort(key=lambda p: p["delta_err"], reverse=True)
    return pairs


# --------------------------------------------------------------------------
# Analysis 3: combined shortlist
# --------------------------------------------------------------------------

def combined_shortlist(loi: list[dict], pairs: list[dict]) -> list[dict]:
    """Per ogni sequenza, conta in quanti run-pair "peggiorativi" appare
    tra i nuovi arrivati. Combina con il Δ di LOI."""
    # Soglia "peggioramento netto" = delta_err > +0.05 MPa
    bad_pairs = [p for p in pairs if p["delta_err"] > 0.05]

    incidents = defaultdict(list)   # seq -> lista di (seed, n_from, n_to, delta_err)
    for p in bad_pairs:
        for s in p["added"]:
            incidents[s].append((p["seed"], p["n_from"], p["n_to"], p["delta_err"]))

    loi_by_seq = {x["sequence"]: x for x in loi}

    combined = []
    for seq, incs in incidents.items():
        loi_entry = loi_by_seq.get(seq)
        combined.append({
            "sequence":          seq,
            "n_bad_pairs":       len(incs),
            "max_delta_pair":    max(d for _, _, _, d in incs),
            "loi_delta":         loi_entry["delta"] if loi_entry else None,
            "loi_n_in":          loi_entry["n_in"]  if loi_entry else None,
            "incidents":         incs,
        })
    # Ordine: prima chi appare in più pair cattivi, poi chi ha LOI delta alto
    combined.sort(key=lambda x: (x["n_bad_pairs"],
                                 x["loi_delta"] if x["loi_delta"] is not None else 0),
                  reverse=True)
    return combined


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def write_csv(path: str, rows: list[dict], cols: list[str]):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}  ({len(rows)} rows)")


def write_shortlist(path: str, loi: list[dict], pairs: list[dict],
                    combined: list[dict], all_runs: list[dict]):
    lines = []
    lines.append("=" * 78)
    lines.append("FASE A — Diagnosi sequenze tossiche (dai 22 run esistenti)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Run analizzati: {len(all_runs)}")
    lines.append(f"Best wp_err:    {min(r['wp_err'] for r in all_runs):.4f} "
                 f"(N={min(all_runs, key=lambda r: r['wp_err'])['n']}, "
                 f"seed={min(all_runs, key=lambda r: r['wp_err'])['seed']})")
    lines.append(f"Worst wp_err:   {max(r['wp_err'] for r in all_runs):.4f} "
                 f"(N={max(all_runs, key=lambda r: r['wp_err'])['n']}, "
                 f"seed={max(all_runs, key=lambda r: r['wp_err'])['seed']})")
    lines.append("")

    # Section 1: top run-pair degradation
    lines.append("─" * 78)
    lines.append("1) RUN-PAIR PEGGIORATIVI (stesso seed, N consecutivi, Δerr > +0.05 MPa)")
    lines.append("─" * 78)
    lines.append("   Quando si passa da N1 a N2 con lo stesso seed e l'errore peggiora,")
    lines.append("   il colpevole è probabilmente tra le sequenze 'nuove arrivate'.")
    lines.append("")
    bad = [p for p in pairs if p["delta_err"] > 0.05]
    if not bad:
        lines.append("   Nessuna coppia peggiora di più di 0.05. (= nessun segnale forte qui)")
    for p in bad[:8]:
        lines.append(f"  seed={p['seed']}  N={p['n_from']:>2} → {p['n_to']:>2}   "
                     f"err {p['err_from']:.4f} → {p['err_to']:.4f}   "
                     f"(Δ=+{p['delta_err']:.4f})")
        lines.append(f"    Aggiunte ({p['n_added']}):")
        for s in p["added"]:
            lines.append(f"      • {s}")
        lines.append("")

    # Section 2: LOI top sospetti
    lines.append("─" * 78)
    lines.append("2) LEAVE-ONE-IN — top 10 sequenze con Δ più positivo")
    lines.append("─" * 78)
    lines.append("   Δ = mean(wp_err | sequenza dentro) − mean(wp_err | sequenza fuori).")
    lines.append("   Δ > 0 = quando questa sequenza è nel training, l'errore tende ad")
    lines.append("   essere più alto. Segnale debole (~22 run), guardalo come indizio.")
    lines.append("")
    lines.append(f"  {'Δ':>8}  {'in':>3}  {'out':>3}  {'mean_in':>8}  {'mean_out':>9}  sequence")
    for x in loi[:10]:
        lines.append(f"  {x['delta']:>+8.4f}  {x['n_in']:>3}  {x['n_out']:>3}  "
                     f"{x['mean_err_in']:>8.4f}  {x['mean_err_out']:>9.4f}  {x['sequence']}")
    lines.append("")
    lines.append("  (top 5 con Δ più NEGATIVO — sequenze 'protettive', per confronto:)")
    lines.append("")
    for x in loi[-5:][::-1]:
        lines.append(f"  {x['delta']:>+8.4f}  {x['n_in']:>3}  {x['n_out']:>3}  "
                     f"{x['mean_err_in']:>8.4f}  {x['mean_err_out']:>9.4f}  {x['sequence']}")
    lines.append("")

    # Section 3: shortlist combinata
    lines.append("─" * 78)
    lines.append("3) SHORTLIST FINALE — sequenze sospette (ordine: evidenza decrescente)")
    lines.append("─" * 78)
    lines.append("   Criterio: appare tra le nuove arrivate di almeno 1 run-pair peggiorativo,")
    lines.append("   ordinate per (n_run_pair_cattivi, Δ_LOI).")
    lines.append("")
    if not combined:
        lines.append("   (lista vuota — nessun run-pair peggiora abbastanza)")
    for i, c in enumerate(combined, 1):
        loi_str = f"Δ_LOI={c['loi_delta']:+.4f}" if c["loi_delta"] is not None else "Δ_LOI=N/A"
        lines.append(f"  #{i:>2}  {c['sequence']}")
        lines.append(f"        in {c['n_bad_pairs']} run-pair cattivi  "
                     f"(max Δerr +{c['max_delta_pair']:.4f}),  {loi_str}")
        for seed, nf, nt, de in c["incidents"]:
            lines.append(f"          - seed={seed}, N={nf}→{nt}, Δerr=+{de:.4f}")
        lines.append("")

    lines.append("=" * 78)
    lines.append("PROSSIMI PASSI SUGGERITI (Fase B)")
    lines.append("=" * 78)
    lines.append("Per ogni sospetto della shortlist:")
    lines.append("  1. Ispeziona visivamente le immagini e le maschere GT della sequenza.")
    lines.append("     (annotazioni sbagliate? immagini sfocate? leaf parzialmente fuori frame?)")
    lines.append("  2. (opzionale) Run di verifica: due training a N=24, uno con la sequenza")
    lines.append("     dentro, uno senza. Se l'errore differisce nettamente, è confermato.")
    lines.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  → {path}")


def main() -> int:
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] {CSV_PATH} non trovato. Lancia prima aggregate.py", file=sys.stderr)
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    runs = load_runs()
    print(f"Caricati {len(runs)} run da {CSV_PATH}")
    pool = all_pool_sequences()
    print(f"Pool di training: {len(pool)} sequenze\n")

    print("1/3 leave-one-in ...")
    loi = leave_one_in(runs, pool)
    write_csv(os.path.join(OUT_DIR, "loi_ranking.csv"), loi,
              ["sequence", "delta", "mean_err_in", "mean_err_out",
               "n_in", "n_out", "std_in", "std_out"])

    print("2/3 run-pair diffs ...")
    pairs = run_pair_diffs(runs)
    # Per il CSV serializzo "added" come stringa
    pairs_csv = [dict(p, added=";".join(p["added"])) for p in pairs]
    write_csv(os.path.join(OUT_DIR, "run_pair_diffs.csv"), pairs_csv,
              ["seed", "n_from", "n_to", "err_from", "err_to", "delta_err",
               "n_added", "added"])

    print("3/3 shortlist combinata ...")
    combined = combined_shortlist(loi, pairs)
    write_shortlist(os.path.join(OUT_DIR, "shortlist.txt"),
                    loi, pairs, combined, runs)

    print("\nFatto. Apri:")
    print(f"  {os.path.join(OUT_DIR, 'shortlist.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

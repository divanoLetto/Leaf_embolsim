"""
aggregate.py — Collect learning-curve results across all runs/ subdirectories
and produce a CSV + plot.

Scans:
    src/method/learning_curve/runs/N{N}_seed{S}/{evaluation/summary_avg.csv,wp_metric.json}

Outputs:
    src/method/learning_curve/learning_curve.csv
    src/method/learning_curve/learning_curve.png

Run (from project root):
    python src/method/learning_curve/aggregate.py
"""

import csv
import json
import os
import re
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(THIS_DIR, "runs")
OUT_CSV  = os.path.join(THIS_DIR, "learning_curve.csv")
OUT_PNG  = os.path.join(THIS_DIR, "learning_curve.png")

RUN_RE = re.compile(r"^N(\d+)_seed(\d+)$")


def _read_summary_avg(eval_dir: str) -> dict | None:
    """Parse the event-frame mean metrics from summary_avg.txt."""
    path = os.path.join(eval_dir, "summary_avg.txt")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as fh:
        text = fh.read()
    # crude but robust enough for the fixed format
    for key, label in [
        ("iou_event",        r"Event frames only.*?Mean IoU:\s*([0-9.]+)"),
        ("dice_event",       r"Event frames only.*?Mean Dice:\s*([0-9.]+)"),
        ("f1_event",         r"Event frames only.*?Mean F1:\s*([0-9.]+)"),
        ("precision_event",  r"Event frames only.*?Mean Precision:\s*([0-9.]+)"),
        ("recall_event",     r"Event frames only.*?Mean Recall:\s*([0-9.]+)"),
        ("pred_fp50_frame",  r"Mean pred fp50 frame:\s*([0-9.\-]+)"),
        ("gt_fp50_frame",    r"Mean GT\s+fp50 frame:\s*([0-9.\-]+)"),
    ]:
        m = re.search(label, text, re.S)
        out[key] = float(m.group(1)) if m else None
    return out


def _read_wp(run_dir: str) -> dict:
    path = os.path.join(run_dir, "wp_metric.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def collect():
    rows = []
    if not os.path.isdir(RUNS_DIR):
        print(f"[ERROR] {RUNS_DIR} does not exist", file=sys.stderr)
        return rows
    for name in sorted(os.listdir(RUNS_DIR)):
        m = RUN_RE.match(name)
        if not m:
            continue
        n, seed = int(m.group(1)), int(m.group(2))
        run_dir  = os.path.join(RUNS_DIR, name)
        eval_dir = os.path.join(run_dir, "evaluation")
        summ = _read_summary_avg(eval_dir)
        if summ is None:
            print(f"  [skip] {name}: no summary_avg.txt")
            continue
        wp = _read_wp(run_dir)
        rows.append({
            "n": n, "seed": seed,
            **summ,
            "avg_pred_wp": wp.get("avg_pred_wp"),
            "avg_gt_wp":   wp.get("avg_gt_wp"),
            "abs_wp_error": wp.get("abs_wp_error"),
        })
    return rows


def write_csv(rows):
    if not rows:
        print("[WARN] no rows to write")
        return
    keys = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV → {OUT_CSV}")


def plot(rows):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available")
        return
    if not rows:
        return

    plt.rcParams.update({
        "font.size":        18,
        "axes.titlesize":   20,
        "axes.labelsize":   19,
        "xtick.labelsize":  16,
        "ytick.labelsize":  16,
        "figure.titlesize": 22,
    })

    # Aggregate (mean, std) over seeds for each N
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r)
    ns = sorted(by_n)

    def _series(key):
        means, stds = [], []
        for n in ns:
            vals = [r[key] for r in by_n[n] if r.get(key) is not None]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            else:
                means.append(np.nan); stds.append(0.0)
        return np.array(means), np.array(stds)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: |wp_pred - wp_gt|
    m, s = _series("abs_wp_error")
    axes[0].errorbar(ns, m, yerr=s, marker="o", capsize=4, lw=1.5)
    axes[0].set_xlabel("Training set size N")
    axes[0].set_ylabel(r"$|\mathrm{wp}_{\mathrm{pred}} - \mathrm{wp}_{\mathrm{GT}}|$  (MPa)")
    axes[0].set_title("Learning curve — wp error (lower is better)")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: IoU on event frames
    m, s = _series("iou_event")
    axes[1].errorbar(ns, m, yerr=s, marker="o", capsize=4, lw=1.5, color="C2")
    axes[1].set_xlabel("Training set size N")
    axes[1].set_ylabel("Mean IoU (event frames)")
    axes[1].set_title("Learning curve — mask IoU (higher is better)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("learning curve (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"PNG → {OUT_PNG}")


def main() -> int:
    rows = collect()
    print(f"Collected {len(rows)} runs\n")
    write_csv(rows)
    plot(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

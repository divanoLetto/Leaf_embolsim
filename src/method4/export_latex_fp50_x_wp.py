"""
export_latex_fp50_x_wp.py — Method 4: FP50-vs-water-potential plot for LaTeX.

Reproduces evaluation/fp50_x_wp_avg.png but WITHOUT the "fp50 GT Excel avg"
vertical line. Saved to latex/fp50_x_wp_avg.png.

Per-sequence cumulative areas are read back from each
evaluation/{seq}/cumulative_area.csv, so the full evaluation does not need to
be re-run.

Usage (from project root):
    python src/method4/export_latex_fp50_x_wp.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method4 import config
from method4.evaluate import _excel_wp_series, _fp50_frame


def _read_cum_area(seq_name: str):
    """Return (pred_cum, gt_cum) lists from evaluation/{seq}/cumulative_area.csv."""
    path = os.path.join(config.EVAL_DIR, seq_name, "cumulative_area.csv")
    if not os.path.isfile(path):
        return None, None
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["frame_index"]),
                         int(r["pred_cumulative_area"]),
                         int(r["gt_cumulative_area"])))
    rows.sort(key=lambda r: r[0])
    pred_cum = [r[1] for r in rows]
    gt_cum   = [r[2] for r in rows]
    return pred_cum, gt_cum


def _save_plot(all_pred_cum, all_gt_cum, seq_names, title, out_path):
    """As evaluate._save_avg_fp50_x_wp_plot but without the fp50 GT Excel line."""
    import matplotlib.pyplot as plt
    if not seq_names:
        return

    series = []  # (wp_sorted_asc, p_sorted, g_sorted, fp50_pred_wp, fp50_gt_wp)
    for name, pc, gc in zip(seq_names, all_pred_cum, all_gt_cum):
        imgs, wps = _excel_wp_series(name)
        if imgs is None or not pc or not gc:
            continue
        img_to_wp = dict(zip(imgs, wps))
        n = min(len(pc), len(gc))
        ptot = pc[-1] if pc else 0
        gtot = gc[-1] if gc else 0
        rows = []
        for i in range(n):
            wp = img_to_wp.get(i + 1)
            if wp is None:
                continue
            rows.append((wp,
                         pc[i] / ptot * 100 if ptot > 0 else 0,
                         gc[i] / gtot * 100 if gtot > 0 else 0))
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: r[0])  # ascending wp (most negative first)
        wp_arr = np.array([r[0] for r in rows])
        p_arr  = np.array([r[1] for r in rows])
        g_arr  = np.array([r[2] for r in rows])
        pred_i = _fp50_frame(pc)
        gt_i   = _fp50_frame(gc)
        fp50_pred_wp = img_to_wp.get(pred_i + 1) if pred_i >= 0 else None
        fp50_gt_wp   = img_to_wp.get(gt_i   + 1) if gt_i   >= 0 else None
        series.append((wp_arr, p_arr, g_arr, fp50_pred_wp, fp50_gt_wp))
    if not series:
        return

    # Common wp grid = intersection of [min, max] across sequences
    wp_lo = max(s[0].min() for s in series)
    wp_hi = min(s[0].max() for s in series)
    if not (wp_lo < wp_hi):
        return
    grid = np.linspace(wp_lo, wp_hi, 200)

    pred_curves, gt_curves = [], []
    for wp_arr, p_arr, g_arr, *_ in series:
        pred_curves.append(np.interp(grid, wp_arr, p_arr))
        gt_curves.append(  np.interp(grid, wp_arr, g_arr))
    pred_avg = np.mean(pred_curves, axis=0)
    gt_avg   = np.mean(gt_curves,   axis=0)

    def _mean(vals):
        v = [x for x in vals if x is not None]
        return float(np.mean(v)) if v else None
    avg_pred_wp = _mean([s[3] for s in series])
    avg_gt_wp   = _mean([s[4] for s in series])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, pred_avg, lw=2, label="Predicted (avg)")
    ax.plot(grid, gt_avg,   lw=2, label="GT mask (avg)", ls="--")
    for c in pred_curves:
        ax.plot(grid, c, lw=0.5, alpha=0.2, color="C0")
    for c in gt_curves:
        ax.plot(grid, c, lw=0.5, alpha=0.2, color="C1")

    def _vline(wp, color, ls, label):
        if wp is None:
            return
        ax.axvline(wp, color=color, ls=ls, lw=1.5, label=label)
        ax.text(wp, 102, f"{wp:.2f} MPa", color=color, rotation=90,
                ha="right", va="top", fontsize=8)

    if avg_pred_wp is not None:
        _vline(avg_pred_wp, "C0", ":", f"fp50 pred avg (wp={avg_pred_wp:.2f} MPa)")
    if avg_gt_wp is not None:
        _vline(avg_gt_wp,   "C1", ":", f"fp50 GT mask avg (wp={avg_gt_wp:.2f} MPa)")

    ax.invert_xaxis()
    ax.set_xlabel("Water potential, wp corrected (MPa)")
    ax.set_ylabel("% of embolism")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    latex_dir = os.path.join(config.MODULE_DIR, "latex")
    os.makedirs(latex_dir, exist_ok=True)

    seq_names = sorted(
        d for d in os.listdir(config.EVAL_DIR)
        if os.path.isdir(os.path.join(config.EVAL_DIR, d))
    )

    all_pred_cum, all_gt_cum, names = [], [], []
    for name in seq_names:
        pc, gc = _read_cum_area(name)
        if pc is None:
            print(f"  [SKIP] {name}: no cumulative_area.csv")
            continue
        all_pred_cum.append(pc)
        all_gt_cum.append(gc)
        names.append(name)

    out_path = os.path.join(latex_dir, "fp50_x_wp_avg.png")
    _save_plot(all_pred_cum, all_gt_cum, names,
               title="Method 4 — Avg FP50 vs water potential",
               out_path=out_path)
    print(f"\nWrote {out_path} ({len(names)} sequence(s))")


if __name__ == "__main__":
    main()

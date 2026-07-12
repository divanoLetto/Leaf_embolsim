"""
export_latex_discrete_fp50.py — Method 4: discrete FP50 plot for LaTeX.

Reproduces evaluation/{seq}/discrete_fp50.png but WITHOUT the "fp50 GT Excel"
vertical line. Saved to latex/{seq}_discrete_fp50.png.

Cumulative areas are read back from evaluation/{seq}/cumulative_area.csv, so the
full evaluation does not need to be re-run.

Usage (from project root):
    python src/method4/export_latex_discrete_fp50.py [SEQ_NAME ...]

With no arguments it defaults to Senecio_16_05_L3_Cavicam13_090725.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method4 import config
from method4.evaluate import _fp50_frame

DEFAULT_SEQS = ["Senecio_16_05_L3_Cavicam13_090725"]


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
    return [r[1] for r in rows], [r[2] for r in rows]


def _save_discrete_fp50(pred_cum, gt_cum, seq_name, out_path):
    """As evaluate._save_discrete_fp50 but without the fp50 GT Excel line."""
    import matplotlib.pyplot as plt
    frames     = list(range(len(pred_cum)))
    pred_total = pred_cum[-1] if pred_cum else 0
    gt_total   = gt_cum[-1]   if gt_cum   else 0
    pred_pct   = [v / pred_total * 100 if pred_total > 0 else 0 for v in pred_cum]
    gt_pct     = [v / gt_total   * 100 if gt_total   > 0 else 0 for v in gt_cum]
    pred_fp50  = _fp50_frame(pred_cum)
    gt_fp50    = _fp50_frame(gt_cum)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, pred_pct, label="Predicted", lw=1.5)
    ax.plot(frames, gt_pct,   label="GT",        lw=1.5, ls="--")
    if pred_fp50 >= 0:
        ax.axvline(pred_fp50, color="C0", ls=":", lw=1.5, label=f"fp50 pred (frame {pred_fp50})")
    if gt_fp50 >= 0:
        ax.axvline(gt_fp50,   color="C1", ls=":", lw=1.5, label=f"fp50 GT (frame {gt_fp50})")
    ax.set_xlabel("Frame pair index")
    ax.set_ylabel("% of embolism")
    ax.set_title(f"Discrete FP50 — {seq_name}")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    seqs = sys.argv[1:] or DEFAULT_SEQS
    latex_dir = os.path.join(config.MODULE_DIR, "latex")
    os.makedirs(latex_dir, exist_ok=True)

    for seq_name in seqs:
        pred_cum, gt_cum = _read_cum_area(seq_name)
        if pred_cum is None:
            print(f"  [SKIP] {seq_name}: no cumulative_area.csv")
            continue
        out_path = os.path.join(latex_dir, f"{seq_name}_discrete_fp50.png")
        _save_discrete_fp50(pred_cum, gt_cum, seq_name, out_path)
        print(f"  [OK]   {out_path}")


if __name__ == "__main__":
    main()

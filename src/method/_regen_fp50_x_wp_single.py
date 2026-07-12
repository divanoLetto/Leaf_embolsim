"""Regenerate evaluation/{seq}/fp50_x_wp.png with tweaks (no full re-eval)."""
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import config
from method.evaluate import _excel_wp_series, _fp50_frame

SEQ = "Senecio_16_05_L3_Cavicam13_090725"


def _read_cum_area(seq_name):
    path = os.path.join(config.EVAL_DIR, seq_name, "cumulative_area.csv")
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["frame_index"]),
                         int(r["pred_cumulative_area"]),
                         int(r["gt_cumulative_area"])))
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows], [r[2] for r in rows]


def main():
    pred_cum, gt_cum = _read_cum_area(SEQ)
    imgs, wps = _excel_wp_series(SEQ)
    img_to_wp = dict(zip(imgs, wps))

    n = min(len(pred_cum), len(gt_cum))
    x_wp, y_pred, y_gt = [], [], []
    pred_total = pred_cum[-1] if pred_cum else 0
    gt_total = gt_cum[-1] if gt_cum else 0
    for i in range(n):
        wp = img_to_wp.get(i + 1)
        if wp is None:
            continue
        x_wp.append(wp)
        y_pred.append(pred_cum[i] / pred_total * 100 if pred_total > 0 else 0)
        y_gt.append(gt_cum[i] / gt_total * 100 if gt_total > 0 else 0)

    pred_fp50 = _fp50_frame(pred_cum)
    gt_fp50 = _fp50_frame(gt_cum)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_wp, y_pred, label="Predicted", lw=1.5)
    ax.plot(x_wp, y_gt, label="GT (mask)", lw=1.5, ls="--")

    def _annot(frame_idx, color, ls, label):
        if frame_idx is None or frame_idx < 0:
            return
        wp = img_to_wp.get(frame_idx + 1)
        if wp is None:
            return
        ax.axvline(wp, color=color, ls=ls, lw=1.5,
                   label=f"{label} (wp={wp:.2f} MPa)")
        ax.text(wp, 102, f"{wp:.2f} MPa", color=color, rotation=90,
                ha="right", va="top", fontsize=8)

    _annot(pred_fp50, "C0", ":", "fp50 pred")
    _annot(gt_fp50, "C1", ":", "fp50 GT")

    ax.invert_xaxis()
    ax.set_xlabel("Water Potential (MPa)")
    ax.set_ylabel("% Embolism")
    ax.set_title(f"FP50 vs water potential — {SEQ}")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(config.EVAL_DIR, SEQ, "fp50_x_wp.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

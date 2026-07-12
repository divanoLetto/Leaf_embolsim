"""
export_latex_masks.py — Method 4: export cumulative-mask panels for LaTeX.

For every sequence folder under evaluation/ this writes two clean PNGs into
latex/:
    latex/{seq}_predicted.png   — predicted cumulative mask
    latex/{seq}_GT.png          — ground-truth cumulative mask

These are exactly the first two panels of evaluation/{seq}/cumulative_mask.png
(white background, black cumulative union), saved without titles/axes so they
drop straight into a figure.

Usage (from project root):
    python src/method4/export_latex_masks.py
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method4 import config
from method4.dataset import _list_frames, _list_masks, _load_mask


def _cumulative_unions(seq_name: str):
    """Reconstruct pred/GT cumulative unions exactly like evaluate_sequence."""
    seq_path = os.path.join(config.DATA_ROOT, seq_name)
    out_dir  = os.path.join(config.OUTPUT_DIR, seq_name)
    mask_dir = os.path.join(out_dir, "masks")

    if not os.path.isdir(mask_dir):
        return None, None

    frames     = _list_frames(seq_path)
    gt_paths   = _list_masks(seq_path)
    pred_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    n_pairs    = min(len(pred_files), len(gt_paths), len(frames) - 1)
    if n_pairs <= 0:
        return None, None

    pred_union = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.uint8)
    gt_union   = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.uint8)

    for i in range(n_pairs):
        pred = (np.array(
            Image.open(os.path.join(mask_dir, pred_files[i])).convert("L")
        ) > 128).astype(np.uint8)
        gt = _load_mask(gt_paths[i], config.IMG_HEIGHT, config.IMG_WIDTH)
        pred_union = np.maximum(pred_union, pred)
        gt_union   = np.maximum(gt_union,   gt)

    return pred_union, gt_union


def main():
    eval_dir  = config.EVAL_DIR
    latex_dir = os.path.join(config.MODULE_DIR, "latex")
    os.makedirs(latex_dir, exist_ok=True)

    seq_names = sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d))
    )

    n_done = 0
    for seq_name in seq_names:
        pred_union, gt_union = _cumulative_unions(seq_name)
        if pred_union is None:
            print(f"  [SKIP] {seq_name}: no predicted masks")
            continue

        pred_img = (255 - pred_union * 255).astype(np.uint8)
        gt_img   = (255 - gt_union   * 255).astype(np.uint8)

        Image.fromarray(pred_img).save(
            os.path.join(latex_dir, f"{seq_name}_predicted.png"))
        Image.fromarray(gt_img).save(
            os.path.join(latex_dir, f"{seq_name}_GT.png"))
        print(f"  [OK]   {seq_name}")
        n_done += 1

    print(f"\nWrote {n_done} sequence(s) to {latex_dir}")


if __name__ == "__main__":
    main()

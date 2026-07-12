"""
predict.py — Full-sequence inference.

Usage (from project root):
    python src/method/predict.py --sequence Senecio_17_11_L5_Cavicam12_210725
    python src/method/predict.py --all
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method import config
from method.dataset import _list_frames, _list_masks, _load_rgb, _load_mask, find_sequences
from method.model import UNet


def load_model(device: torch.device) -> torch.nn.Module:
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nRun train.py first."
        )
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = UNet(in_channels=6, dropout_p=config.DROPOUT_P).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    return model


def frames_to_tensor(
    rgb_t: np.ndarray, rgb_t1: np.ndarray, device: torch.device
) -> torch.Tensor:
    t  = torch.from_numpy(rgb_t.transpose(2, 0, 1).astype(np.float32) / 255.0)
    t1 = torch.from_numpy(rgb_t1.transpose(2, 0, 1).astype(np.float32) / 255.0)
    return torch.cat([t, t1], dim=0).unsqueeze(0).to(device)


def sliding_window_inference(
    model: torch.nn.Module,
    rgb_t: np.ndarray,
    rgb_t1: np.ndarray,
    device: torch.device,
    patch: int = config.PATCH_SIZE,
    stride: int = None,
) -> np.ndarray:
    """Tiled inference with overlapping patches; averages overlapping predictions."""
    if stride is None:
        stride = patch // 2

    H, W = rgb_t.shape[:2]
    sum_map   = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.float64)

    with torch.no_grad():
        for y0 in range(0, H - patch + 1, stride):
            for x0 in range(0, W - patch + 1, stride):
                y1, x1  = y0 + patch, x0 + patch
                inp     = frames_to_tensor(rgb_t[y0:y1, x0:x1], rgb_t1[y0:y1, x0:x1], device)
                pred    = model(inp).squeeze().detach().cpu().numpy()
                sum_map  [y0:y1, x0:x1] += pred
                count_map[y0:y1, x0:x1] += 1.0

        # Right strip
        for y0 in range(0, H - patch + 1, stride):
            x0 = W - patch
            y1, x1  = y0 + patch, x0 + patch
            inp  = frames_to_tensor(rgb_t[y0:y1, x0:x1], rgb_t1[y0:y1, x0:x1], device)
            pred = model(inp).squeeze().detach().cpu().numpy()
            sum_map  [y0:y1, x0:x1] += pred
            count_map[y0:y1, x0:x1] += 1.0

        # Bottom strip
        for x0 in range(0, W - patch + 1, stride):
            y0 = H - patch
            y1, x1  = y0 + patch, x0 + patch
            inp  = frames_to_tensor(rgb_t[y0:y1, x0:x1], rgb_t1[y0:y1, x0:x1], device)
            pred = model(inp).squeeze().detach().cpu().numpy()
            sum_map  [y0:y1, x0:x1] += pred
            count_map[y0:y1, x0:x1] += 1.0

        # Bottom-right corner
        y0, x0 = H - patch, W - patch
        inp  = frames_to_tensor(rgb_t[y0:, x0:][:patch, :patch], rgb_t1[y0:, x0:][:patch, :patch], device)
        pred = model(inp).squeeze().detach().cpu().numpy()
        sum_map  [y0:, x0:] += pred
        count_map[y0:, x0:] += 1.0

    heatmap = np.where(count_map > 0, sum_map / count_map, 0.0)
    return heatmap.astype(np.float32)


def predict_sequence(
    seq_name: str,
    threshold: float = config.THRESHOLD,
    device: torch.device = None,
    model: torch.nn.Module = None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(device)

    print(f"\n{'='*60}")
    print(f"Predicting: {seq_name}")
    print(f"{'='*60}")

    seq_path = os.path.join(config.DATA_ROOT, seq_name)
    frames   = _list_frames(seq_path)
    masks    = _list_masks(seq_path)
    n_pairs  = min(len(frames) - 1, len(masks)) if masks else len(frames) - 1

    if n_pairs <= 0:
        print("  [SKIP] No frame pairs found.")
        return

    out_dir  = os.path.join(config.OUTPUT_DIR, seq_name)
    hmap_dir = os.path.join(out_dir, "heatmaps")
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(hmap_dir, exist_ok=True)
    os.makedirs(mask_dir,  exist_ok=True)

    print(f"  {n_pairs} frame pairs to process …")
    rows = []

    for i in range(n_pairs):
        rgb_t  = _load_rgb(frames[i],   config.IMG_HEIGHT, config.IMG_WIDTH)
        rgb_t1 = _load_rgb(frames[i+1], config.IMG_HEIGHT, config.IMG_WIDTH)
        if rgb_t  is None: rgb_t  = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3), np.uint8)
        if rgb_t1 is None: rgb_t1 = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3), np.uint8)

        heatmap = sliding_window_inference(model, rgb_t, rgb_t1, device)
        binary  = (heatmap > threshold).astype(np.uint8)

        gt_count = 0
        if i < len(masks):
            gt = _load_mask(masks[i], config.IMG_HEIGHT, config.IMG_WIDTH)
            gt_count = int(gt.sum())

        stem = os.path.splitext(os.path.basename(frames[i]))[0]
        np.save(os.path.join(hmap_dir, f"{stem}.npy"), heatmap)
        Image.fromarray(binary * 255).save(os.path.join(mask_dir, f"{stem}.png"))

        rows.append((i, stem, int(binary.sum()), gt_count))

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:4d}/{n_pairs}]  pred_px={binary.sum():5d}  gt_px={gt_count:5d}")

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w") as fh:
        fh.write("frame_index,frame_name,n_black_pixels_predicted,n_black_pixels_gt\n")
        for row in rows:
            fh.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

    print(f"\n  Outputs → {out_dir}")
    print(f"  Summary CSV → {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run inference.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sequence", "-s")
    g.add_argument("--all", "-a", action="store_true")
    p.add_argument("--threshold", type=float, default=config.THRESHOLD)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    if args.all:
        seqs = find_sequences(config.DATA_ROOT)
        for name, _ in seqs:
            predict_sequence(name, threshold=args.threshold, device=device, model=model)
    else:
        predict_sequence(args.sequence, threshold=args.threshold, device=device, model=model)


if __name__ == "__main__":
    main()

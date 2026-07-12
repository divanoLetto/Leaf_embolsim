"""
train.py — Method 4: U-Net training with improved regularisation.

Changes vs Method 2:
  - Weight decay on Adam (WEIGHT_DECAY)
  - Early stopping: training halts when val loss does not improve for
    EARLY_STOP_PATIENCE epochs (with minimum delta EARLY_STOP_MIN_DELTA).
    The best checkpoint is always saved regardless of when training stops.
  - EPOCHS is an upper bound; the effective number of epochs is printed at the end.

Usage (from project root):
    python src/method4/train.py
    python src/method4/train.py --epochs 200 --lr 5e-5 --batch-size 4
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method4 import config
from method4.dataset import LeafPairDataset, load_split, make_loader
from method4.model import UNet
from method4.losses import CombinedLoss


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stops training when val loss has not improved by at least min_delta
    for `patience` consecutive epochs.

    Call .step(val_loss) each epoch; returns True when training should stop.
    .best_epoch and .best_loss reflect the epoch where the checkpoint was saved.
    """

    def __init__(self, patience: int, min_delta: float = 1e-5):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.best_epoch  = 0
        self._wait       = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self._wait      = 0
            return False   # continue
        self._wait += 1
        return self._wait >= self.patience   # True → stop


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("method4.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def train(args):
    torch.manual_seed(config.SEED)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    log_path = os.path.join(config.CHECKPOINT_DIR, "train.log")
    log = _setup_logger(log_path)

    train_seqs = config.TRAIN_SEQUENCES or load_split(config.TRAIN_SPLIT_FILE)
    val_seqs   = config.VAL_SEQUENCES   or load_split(config.VAL_SPLIT_FILE)

    log.info(f"Train sequences: {train_seqs}")
    log.info(f"Val   sequences: {val_seqs}\n")

    log.info("Building training dataset …")
    train_ds = LeafPairDataset(train_seqs, config.DATA_ROOT, augment=True)
    log.info("\nBuilding validation dataset …")
    val_ds   = LeafPairDataset(val_seqs,   config.DATA_ROOT, augment=False)
    log.info("")

    if len(train_ds) == 0:
        log.error("[ERROR] Training dataset is empty.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        drop_last=False,
    )

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model     = UNet(in_channels=6, dropout_p=config.DROPOUT_P).to(device)
    criterion = CombinedLoss(
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        focal_weight=config.LOSS_FOCAL_W,
        dice_weight=config.LOSS_DICE_W,
    )
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )
    stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=config.EARLY_STOP_MIN_DELTA,
    )

    best_val_loss = float("inf")
    best_ckpt     = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    history       = {"train": [], "val": []}

    log.info(
        f"\nStarting training: up to {args.epochs} epochs, "
        f"batch={args.batch_size}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, "
        f"early_stop_patience={args.early_stop_patience}"
    )
    log.info("-" * 60)

    stopped_early = False
    last_epoch    = args.epochs

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            x    = batch["input"].to(device)
            y    = batch["target"].to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            total_train_loss += loss.item() * x.size(0)
            n_train += x.size(0)

        train_loss = total_train_loss / max(n_train, 1)
        val_loss   = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        elapsed = time.time() - t0
        marker  = " ← best" if val_loss < best_val_loss else ""
        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"time={elapsed:.1f}s{marker}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "train_seqs":  train_seqs,
                "val_seqs":    val_seqs,
            }, best_ckpt)

        if stopper.step(val_loss, epoch):
            log.info(
                f"\nEarly stopping triggered at epoch {epoch}. "
                f"Best val loss {stopper.best_loss:.6f} at epoch {stopper.best_epoch}."
            )
            stopped_early = True
            last_epoch    = epoch
            break

    if not stopped_early:
        log.info(f"\nReached max epochs ({args.epochs}).")

    log.info(f"Best val loss: {best_val_loss:.6f} (epoch {stopper.best_epoch})")
    log.info(f"Best checkpoint → {best_ckpt}")

    hist_path = os.path.join(config.CHECKPOINT_DIR, "history.json")
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)

    _save_loss_curve(history, config.CHECKPOINT_DIR, stopper.best_epoch, log)


def _save_loss_curve(history: dict, out_dir: str, best_epoch: int = None, log=None):
    csv_path = os.path.join(out_dir, "loss_curve.csv")
    with open(csv_path, "w") as fh:
        fh.write("epoch,train_loss,val_loss\n")
        for i, (tr, vl) in enumerate(zip(history["train"], history["val"]), 1):
            fh.write(f"{i},{tr:.8f},{vl:.8f}\n")
    if log: log.info(f"Loss curve CSV → {csv_path}")

    try:
        import matplotlib.pyplot as plt
        module_dir = os.path.dirname(os.path.abspath(__file__))
        png_paths  = [
            os.path.join(out_dir, "loss_curve.png"),
            os.path.join(module_dir, "train_curves.png"),
        ]
        epochs = range(1, len(history["train"]) + 1)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs, history["train"], label="Train")
        ax.plot(epochs, history["val"],   label="Val", linestyle="--")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.5,
                       label=f"Best (ep {best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Combined loss (Focal + Dice)")
        ax.set_title("Method 4 — Training curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for p in png_paths:
            fig.savefig(p, dpi=120)
            if log: log.info(f"Loss curve PNG → {p}")
        plt.close(fig)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Method 4 U-Net.")
    p.add_argument("--epochs",               type=int,   default=config.EPOCHS)
    p.add_argument("--lr",                   type=float, default=config.LR)
    p.add_argument("--batch-size",           type=int,   default=config.BATCH_SIZE)
    p.add_argument("--focal-alpha",          type=float, default=config.FOCAL_ALPHA)
    p.add_argument("--focal-gamma",          type=float, default=config.FOCAL_GAMMA)
    p.add_argument("--weight-decay",         type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--early-stop-patience",  type=int,   default=config.EARLY_STOP_PATIENCE)
    p.add_argument("--workers",              type=int,   default=8)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

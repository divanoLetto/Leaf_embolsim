#!/usr/bin/env python3
"""
make_splits.py — Generate train.txt, val.txt, test.txt for the leaf embolism dataset.

Split design (biologically motivated, three generalisation levels):
--------------------------------------------------------------------
Each of val and test contains sequences from all three difficulty levels:

  LEAF-SPLIT   : same population + same mother plant in train, only this leaf is new.
                 → easiest generalisation (intra-plant)

  MOTHER-SPLIT : same mother plant appears in train under a different population,
                 but this population is never seen in train.
                 → medium generalisation (mother known, population new)

  POP-SPLIT    : the population is completely absent from train AND the mother plant
                 appears nowhere else in the dataset (so it is also truly unseen).
                 → hardest generalisation (both population and mother new)

Concrete assignments
--------------------
  VAL  (10 seq):
    pop-split    → Senecio_23_08_*  (mother 08, unique to pop 23)
    mother-split → Senecio_22_12_*  (mother 12 seen in train via pop 07)

  TEST (11 seq):
    pop-split    → Senecio_16_05_*  (mother 05, unique to pop 16)
    mother-split → Senecio_10_11_*  (mother 11 seen in train via pop 07, 17, 21;
                                     pop 10 partially in train via 10_03)
    leaf-split   → Senecio_17_11_L5_* only (mother 11 + pop 17 both in train via L1-L4
                                             and via pop 07, 21)

  TRAIN (53 seq): everything else

Usage
-----
    python src/make_splits.py                     # writes to project root
    python src/make_splits.py --data-root /other/path
    python src/make_splits.py --dry-run           # print without writing
"""

import argparse
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Split rules — edit here to change the split
# ---------------------------------------------------------------------------

# Entire populations assigned to val / test (all their sequences go there).
VAL_POPULATIONS  = {"23", "22"}   # pop 23 = pop-split, pop 22 = mother-split
TEST_POPULATIONS = {"16", "10"}   # pop 16 = pop-split, pop 10 = mother-split
                                  # (pop 10_03 stays in train; only 10_11 → test
                                  #  but we assign the whole pop and keep 10_03
                                  #  via TRAIN_INDIVIDUAL override below)

# Individual sequences forced into train regardless of their population.
TRAIN_INDIVIDUAL = {
    "Senecio_10_03_",   # mother 03 (unique) stays in train; only 10_11 goes to test
}

# Individual sequences assigned to test regardless of population.
# Format: substring match against the sequence folder name.
TEST_INDIVIDUAL = {
    "Senecio_17_11_L5_",   # leaf-split: L5 only; L1-L4 stay in train
}


def classify(seq_name: str) -> str:
    """Return 'train', 'val', or 'test' for a sequence folder name."""
    # TRAIN_INDIVIDUAL: highest priority — forces back to train
    for substr in TRAIN_INDIVIDUAL:
        if substr in seq_name:
            return "train"

    # TEST_INDIVIDUAL: next priority
    for substr in TEST_INDIVIDUAL:
        if substr in seq_name:
            return "test"

    # Population-level assignment
    parts = seq_name.split("_")
    if len(parts) < 3:
        return "train"
    pop = parts[1]

    if pop in VAL_POPULATIONS:
        return "val"
    if pop in TEST_POPULATIONS:
        return "test"
    return "train"


def find_sequences(data_root: str):
    """Return sorted list of sequence folder names (directories containing PNGs)."""
    root = Path(data_root)
    seqs = sorted(
        e.name for e in root.iterdir()
        if e.is_dir() and list(e.glob("*.png"))
    )
    return seqs


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test split files.")
    parser.add_argument("--data-root", default=None,
                        help="Path to data/ directory (default: auto-detected from script location)")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to write split .txt files (default: project root)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print split without writing files")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent

    data_root = Path(args.data_root) if args.data_root else repo_root / "data"
    out_dir   = Path(args.out_dir)   if args.out_dir   else repo_root

    if not data_root.is_dir():
        raise SystemExit(f"[ERROR] data root not found: {data_root}")

    seqs = find_sequences(str(data_root))
    if not seqs:
        raise SystemExit(f"[ERROR] No sequences found in {data_root}")

    splits = {"train": [], "val": [], "test": []}
    for seq in seqs:
        splits[classify(seq)].append(seq)

    # ---- Summary ----
    print(f"Data root : {data_root}")
    print(f"Sequences : {len(seqs)} total\n")

    labels = {
        "train": "TRAIN",
        "val":   "VAL  ",
        "test":  "TEST ",
    }
    for split in ("train", "val", "test"):
        print(f"  {labels[split]}  ({len(splits[split])} sequences)")
        for seq in splits[split]:
            tag = ""
            for substr in TRAIN_INDIVIDUAL:
                if substr in seq:
                    tag = "  [train-override]"
            for substr in TEST_INDIVIDUAL:
                if substr in seq:
                    tag = "  [leaf-split]"
            parts = seq.split("_")
            if len(parts) >= 2:
                pop = parts[1]
                if pop in VAL_POPULATIONS and split == "val":
                    tag = "  [pop-split]" if pop == "23" else "  [mother-split]"
                if pop in TEST_POPULATIONS and split == "test":
                    tag = "  [pop-split]" if pop == "16" else "  [mother-split]"
            print(f"    {seq}{tag}")
        print()

    if args.dry_run:
        print("[dry-run] No files written.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.txt"
        with open(path, "w") as fh:
            fh.write("\n".join(splits[split]) + "\n")
        print(f"  Written: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
make_subsets.py — Generate nested training subsets for the learning-curve experiment.

For each seed, the full train.txt is shuffled once with that seed; the first N
entries become the subset for size N. By construction, subset(N) is a prefix of
subset(M) when N < M and the same seed is used — i.e. nested subsets.

Outputs (one file per (N, seed)):
    src/method4/learning_curve/subsets/train_N{N}_seed{S}.txt

Run (from project root):
    python src/method4/learning_curve/make_subsets.py
"""

import os
import random
import sys

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
TRAIN_TXT   = os.path.join(REPO_ROOT, "train.txt")
SUBSETS_DIR = os.path.join(THIS_DIR, "subsets")

N_VALUES = [4, 8, 12, 16, 24, 32, 40, 46]
SEEDS    = list(range(10))   # 0..9 — copre lo schema K(N)=10/8/8/6/4/3/3/1


def load_train_list(path: str) -> list[str]:
    with open(path) as fh:
        items = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    return items


def main() -> int:
    if not os.path.exists(TRAIN_TXT):
        print(f"[ERROR] {TRAIN_TXT} not found", file=sys.stderr)
        return 1

    full = load_train_list(TRAIN_TXT)
    total = len(full)
    os.makedirs(SUBSETS_DIR, exist_ok=True)

    print(f"Full train list: {total} sequences")
    print(f"N grid: {N_VALUES}")
    print(f"Seeds:  {SEEDS}\n")

    for seed in SEEDS:
        rng = random.Random(seed)
        permuted = full.copy()
        rng.shuffle(permuted)
        for n in N_VALUES:
            if n > total:
                print(f"[WARN] N={n} > {total}, skipping")
                continue
            subset = permuted[:n]
            out = os.path.join(SUBSETS_DIR, f"train_N{n}_seed{seed}.txt")
            with open(out, "w") as fh:
                for s in subset:
                    fh.write(s + "\n")
            print(f"  seed={seed} N={n:>2} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

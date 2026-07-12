"""
run_one.py — One learning-curve run: train + predict + evaluate for a given (N, seed).

All outputs go to:
    src/method4/learning_curve/runs/N{N}_seed{S}/{checkpoints,outputs,evaluation}/

The original src/method4/outputs/ and src/method4/evaluation/ produced by the
full 46-sample run are NEVER touched. We achieve this by monkey-patching the
relevant paths on `method4.config` before invoking train/predict/evaluate.

Resumability: skipped if runs/N{N}_seed{S}/evaluation/summary_avg.csv exists.

Run (from project root):
    python src/method4/learning_curve/run_one.py --n 8 --seed 0
    python src/method4/learning_curve/run_one.py --n 8 --seed 0 --force
"""

import argparse
import json
import os
import sys
import time

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
SRC_DIR   = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

sys.path.insert(0, SRC_DIR)


def _patch_config(run_dir: str, train_split_file: str):
    """Redirect method4.config paths to the per-run directory."""
    from method4 import config
    config.TRAIN_SPLIT_FILE = train_split_file
    config.CHECKPOINT_DIR   = os.path.join(run_dir, "checkpoints")
    config.OUTPUT_DIR       = os.path.join(run_dir, "outputs")
    config.EVAL_DIR         = os.path.join(run_dir, "evaluation")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR,     exist_ok=True)
    os.makedirs(config.EVAL_DIR,       exist_ok=True)
    return config


def _do_train(args):
    from method4 import train as train_mod

    class A:
        epochs              = args.epochs
        lr                  = args.lr
        batch_size          = args.batch_size
        focal_alpha         = args.focal_alpha
        focal_gamma         = args.focal_gamma
        weight_decay        = args.weight_decay
        early_stop_patience = args.early_stop_patience
        workers             = args.workers

    train_mod.train(A())


def _do_predict_all(test_seqs):
    from method4 import predict as predict_mod

    class A:
        all      = False
        sequence = None
        out_dir  = None  # use config.OUTPUT_DIR

    for seq in test_seqs:
        print(f"\n[predict] {seq}")
        A.sequence = seq
        predict_mod.main(A()) if hasattr(predict_mod, "main") else _predict_single(predict_mod, seq)


def _predict_single(predict_mod, seq_name):
    # Fallback: invoke whatever entry point predict.py exposes.
    # The current predict.py uses argparse + __main__, so call its __main__-equivalent.
    import runpy
    sys.argv = ["predict.py", "--sequence", seq_name]
    runpy.run_module("method4.predict", run_name="__main__")


def _do_evaluate_all():
    import runpy
    sys.argv = ["evaluate.py", "--all"]
    runpy.run_module("method4.evaluate", run_name="__main__")


def _load_test_seqs():
    with open(os.path.join(REPO_ROOT, "test.txt")) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def _compute_wp_metric(eval_dir: str) -> dict:
    """Compute avg wp at fp50 (pred and GT mask) from summary_avg.csv +
    Excel lookup. Mirrors evaluate._save_avg_fp50_x_wp_plot logic."""
    import csv
    from method4.evaluate import _excel_wp_series

    csv_path = os.path.join(eval_dir, "summary_avg.csv")
    if not os.path.exists(csv_path):
        return {"error": f"missing {csv_path}"}

    pred_wps, gt_wps = [], []
    rows = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    for r in rows:
        seq = r["seq_name"]
        try:
            pred_f = int(r["pred_fp50_frame"])
            gt_f   = int(r["gt_fp50_frame"])
        except (KeyError, ValueError, TypeError):
            continue
        imgs, wps = _excel_wp_series(seq)
        if imgs is None:
            continue
        img_to_wp = dict(zip(imgs, wps))
        # _fp50_frame returns 0-based index into cum_area; Excel image_no is 1-based.
        if pred_f >= 0:
            wp = img_to_wp.get(pred_f + 1)
            if wp is not None:
                pred_wps.append(wp)
        if gt_f >= 0:
            wp = img_to_wp.get(gt_f + 1)
            if wp is not None:
                gt_wps.append(wp)

    out = {
        "n_pred": len(pred_wps),
        "n_gt":   len(gt_wps),
        "avg_pred_wp": (sum(pred_wps) / len(pred_wps)) if pred_wps else None,
        "avg_gt_wp":   (sum(gt_wps)   / len(gt_wps))   if gt_wps   else None,
    }
    if out["avg_pred_wp"] is not None and out["avg_gt_wp"] is not None:
        out["abs_wp_error"] = abs(out["avg_pred_wp"] - out["avg_gt_wp"])
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",    type=int, required=True, help="training subset size")
    p.add_argument("--seed", type=int, required=True, help="subset seed (0/1/2)")
    p.add_argument("--force", action="store_true",
                   help="re-run even if summary_avg.csv already exists")
    p.add_argument("--skip-train",    action="store_true")
    p.add_argument("--skip-predict",  action="store_true")
    p.add_argument("--skip-evaluate", action="store_true")
    # Pass-through to train.py defaults (from config)
    from method4 import config as _cfg
    p.add_argument("--epochs",       type=int,   default=_cfg.EPOCHS)
    p.add_argument("--lr",           type=float, default=_cfg.LR)
    p.add_argument("--batch-size",   type=int,   default=_cfg.BATCH_SIZE)
    p.add_argument("--focal-alpha",  type=float, default=_cfg.FOCAL_ALPHA)
    p.add_argument("--focal-gamma",  type=float, default=_cfg.FOCAL_GAMMA)
    p.add_argument("--weight-decay", type=float, default=_cfg.WEIGHT_DECAY)
    p.add_argument("--early-stop-patience", type=int, default=_cfg.EARLY_STOP_PATIENCE)
    p.add_argument("--workers",      type=int,   default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    subset_file = os.path.join(THIS_DIR, "subsets", f"train_N{args.n}_seed{args.seed}.txt")
    if not os.path.exists(subset_file):
        print(f"[ERROR] subset file not found: {subset_file}\n"
              f"        Run: python src/method4/learning_curve/make_subsets.py", file=sys.stderr)
        return 1

    run_dir = os.path.join(THIS_DIR, "runs", f"N{args.n}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    summary_csv = os.path.join(run_dir, "evaluation", "summary_avg.csv")
    if os.path.exists(summary_csv) and not args.force:
        print(f"[SKIP] {summary_csv} already exists (use --force to re-run)")
        # Still (re)compute wp metric in case it wasn't saved.
        _patch_config(run_dir, subset_file)
        wp = _compute_wp_metric(os.path.join(run_dir, "evaluation"))
        with open(os.path.join(run_dir, "wp_metric.json"), "w") as fh:
            json.dump(wp, fh, indent=2)
        print(f"  wp metric → {wp}")
        return 0

    cfg = _patch_config(run_dir, subset_file)
    print(f"┏━━ Learning-curve run: N={args.n}, seed={args.seed}")
    print(f"┃  subset      : {subset_file}")
    print(f"┃  run_dir     : {run_dir}")
    print(f"┃  CHECKPOINT  : {cfg.CHECKPOINT_DIR}")
    print(f"┃  OUTPUT      : {cfg.OUTPUT_DIR}")
    print(f"┃  EVAL        : {cfg.EVAL_DIR}")
    print(f"┗━━")

    t0 = time.time()

    if not args.skip_train:
        print("\n=== STEP 1/3 — Training ===")
        _do_train(args)

    if not args.skip_predict:
        print("\n=== STEP 2/3 — Predicting on test set ===")
        test_seqs = _load_test_seqs()
        for seq in test_seqs:
            print(f"  → {seq}")
            import runpy
            sys.argv = ["predict.py", "--sequence", seq]
            runpy.run_module("method4.predict", run_name="__main__")

    if not args.skip_evaluate:
        print("\n=== STEP 3/3 — Evaluating ===")
        _do_evaluate_all()

    # wp metric
    wp = _compute_wp_metric(os.path.join(run_dir, "evaluation"))
    with open(os.path.join(run_dir, "wp_metric.json"), "w") as fh:
        json.dump(wp, fh, indent=2)
    print(f"\nwp metric → {wp}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())

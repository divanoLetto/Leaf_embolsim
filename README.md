# Leaf embolism segmentation

Deep-learning segmentation of **embolism events** in time-lapse image sequences
of drying leaves. A U-Net receives two consecutive frames (`frame_t`,
`frame_{t+1}`) and predicts the pixels that became **newly embolised** between
them. Aggregating these incremental masks over a sequence reconstructs the
cumulative embolised area and the vulnerability-curve dynamics.

Imaging setup: Cavicam / Raspberry Pi Camera v2 (IMX219), ~5-minute intervals,
fixed LED illumination.

---

## Method

The model lives in `src/method/` — a U-Net with the following configuration:

- **Model** — U-Net, 6-channel input (two RGB frames concatenated), 1-channel
  sigmoid output. Encoder `6 → 32 → 64 → 128`, bottleneck `256`, symmetric
  decoder with `Dropout2d` (spatial dropout) in each decoder block.
- **Loss** — combined **Focal + Dice** (0.5 / 0.5).
- **Regularisation** — dropout (p=0.3), weight decay (1e-4), early stopping
  (patience 15).
- **Augmentation** — physically calibrated to the Cavicam/RPi setup: per-frame
  brightness & shot noise (independent), shared gamma / white-balance / fine
  rotation, and joint elastic deformation. See [src/method/dataset.py](src/method/dataset.py).
- **Class imbalance** — patch oversampling around embolism pixels.

All hyper-parameters live in [src/method/config.py](src/method/config.py).

---

## Repository structure

```
.
├── train.txt / val.txt / test.txt   # dataset splits (sequence names)
├── make_video.py                    # utility: PNG frames → video (ffmpeg)
├── requirements.txt
├── results/                         # curated result figures (see below)
├── data/                            # dataset — NOT included, provide locally
└── src/
    ├── check_data.py                # dataset integrity checker
    ├── make_splits.py               # regenerates train/val/test.txt
    └── method/
        ├── config.py                # all hyper-parameters and paths
        ├── model.py                 # U-Net
        ├── losses.py                # Focal + Dice
        ├── dataset.py               # dataset + calibrated augmentation
        ├── train.py / predict.py / evaluate.py
        ├── run_method.sh           # full pipeline: train → predict → evaluate
        ├── checkpoints/             # trained weights (best_model.pt) + curves
        └── learning_curve/          # data-scaling experiment (see its README)
```

Generated outputs (`outputs/`, `evaluation/`, `learning_curve/runs/`, …) are
`.gitignore`d — they are reproduced by running the pipeline.

---

## Dataset

The dataset is **not part of this repository**. Place it under `data/`, one
directory per sequence, matching the names in `train.txt` / `val.txt` /
`test.txt`:

```
data/
├── Senecio_16_05_L1_Cavicam15_090725/
│   ├── 20250709-180217.png              # time-lapse frames
│   ├── ...
│   └── <sequence>_analysedStack/        # ground-truth mask TIFFs
│       └── ...                          # one mask per consecutive frame pair
└── SenecioIVERdroughtVCs.xlsx           # reference vulnerability-curve data
```

For each sequence, `n_masks == n_frames − 1` (one incremental mask per frame
pair). Verify a dataset with:

```bash
python src/check_data.py
```

Splits are biologically motivated (leaf / mother-plant / population
generalisation levels); see the docstring in
[src/make_splits.py](src/make_splits.py).

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, install `torch` following the
[official instructions](https://pytorch.org/get-started/locally/) first.

---

## Usage

Run everything from the repository root.

**Full pipeline** (train → predict on the test set → evaluate):

```bash
bash src/method/run_method.sh
# forward flags to training, e.g.:
bash src/method/run_method.sh --epochs 200 --batch-size 8
```

**Individual steps:**

```bash
python src/method/train.py                                   # train
python src/method/predict.py --sequence <SEQUENCE_NAME>      # or --all
python src/method/evaluate.py --all                          # metrics + figures
```

Outputs are written to `src/method/{outputs,evaluation}/`; checkpoints to
`src/method/checkpoints/`.

A pre-trained checkpoint is included at
[src/method/checkpoints/best_model.pt](src/method/checkpoints/best_model.pt),
so `predict.py` / `evaluate.py` can be run without retraining.

---

## Learning-curve experiment

Measures performance as a function of the number of training sequences (fixed
val/test). See [src/method/learning_curve/README.md](src/method/learning_curve/README.md).

---

## Results

Curated figures in [results/](results/):

| File | Description |
|------|-------------|
| `summary_avg.txt` / `.csv` | Aggregate test-set metrics (IoU, Dice, F1, FP50) |
| `training_curves.png`, `loss_curve.png` | Training / validation loss |
| `learning_curve.png` (+ `.csv`) | Performance vs. training-set size |
| `cumulative_area_avg.png` | Predicted vs. GT cumulative embolised area |
| `discrete_fp50_avg.png`, `fp50_x_wp_avg.png` | Embolism-timing metrics |
| `examples/` | Qualitative GT vs. predicted masks |

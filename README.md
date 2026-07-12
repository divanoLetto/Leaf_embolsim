# Leaf embolism segmentation

## Authors: Lorenzo Mandelli, Kate M. Johnson, Maurizio Mencuccini, Stefano Berretti
#### CREAF, Università degli Studi di Firenze

![Architecture](images/architecture_1.png)

![](https://img.shields.io/github/contributors/divanoLetto/Leaf_embolsim?color=light%20green) ![](https://img.shields.io/github/repo-size/divanoLetto/Leaf_embolsim?cacheSeconds=60)

Deep-learning segmentation of **embolism events** in time-lapse image sequences
of drying leaves. A U-Net receives two consecutive frames (`frame_t`,
`frame_{t+1}`) and predicts the pixels that became **newly embolised** between
them. Aggregating these incremental masks over a sequence reconstructs the
cumulative embolised area and the vulnerability-curve dynamics.

Imaging setup: Cavicam / Raspberry Pi Camera v2 (IMX219), ~5-minute intervals,
fixed LED illumination.

<img src="images/demo.gif" width="100%" alt="Prediction demo"/>

*Two consecutive input frames and the model's predicted embolism mask
(white = predicted embolism) on one test sequence.*

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, install `torch` following the
[official instructions](https://pytorch.org/get-started/locally/) first.

---

## Dataset

The dataset is hosted on Hugging Face:
**[LorenzoMande/Leaf_embolism](https://huggingface.co/datasets/LorenzoMande/Leaf_embolism)**.
It is stored as one gzipped tarball per sequence (`<sequence>.tar.gz`, containing
the time-lapse frames and the ground-truth masks) plus the reference spreadsheets.

Download and reconstruct the `data/` folder the code expects:

```bash
pip install huggingface_hub

# 1. download all sequence archives + spreadsheets
hf download LorenzoMande/Leaf_embolism --repo-type dataset --local-dir data_hf

# 2. extract each sequence into data/ and add the spreadsheets
mkdir -p data
for f in data_hf/*.tar.gz; do tar xzf "$f" -C data; done
cp data_hf/*.xlsx data/
```

> Python alternative for step 1:
> `from huggingface_hub import snapshot_download; snapshot_download("LorenzoMande/Leaf_embolism", repo_type="dataset", local_dir="data_hf")`

Each sequence then looks like:

```
data/
├── Senecio_16_05_L1_Cavicam15_090725/
│   ├── 20250709-180217.png     # time-lapse frames
│   ├── ...
│   └── analysedStack/          # ground-truth mask TIFFs (one per frame pair)
└── SenecioIVERdroughtVCs.xlsx  # reference vulnerability-curve data
```

---

## Repository structure

```
.
├── train.txt / val.txt / test.txt   # dataset splits (sequence names)
├── make_video.py                    # utility: PNG frames → video (ffmpeg)
├── requirements.txt
├── results/                         # curated result figures (see below)
├── data/                            # dataset — download from Hugging Face (see Dataset)
└── src/
    ├── check_data.py                # dataset integrity checker
    ├── make_splits.py               # regenerates train/val/test.txt
    └── method/
        ├── config.py                # all hyper-parameters and paths
        ├── model.py                 # U-Net (6-ch input, Focal + Dice loss)
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

## Usage

Run everything from the repository root. The pipeline has three stages, and
**training is optional** — a pre-trained checkpoint is already included at
[src/method/checkpoints/best_model.pt](src/method/checkpoints/best_model.pt).

| Stage | Script | Output |
|-------|--------|--------|
| **Train** | `train.py` | `checkpoints/best_model.pt` |
| **Predict** (inference) | `predict.py` | predicted masks in `outputs/` |
| **Evaluate** | `evaluate.py` | metrics + figures in `evaluation/` |

### A) Inference only — use the included pre-trained model

No training required:

```bash
python src/method/predict.py --all       # or a single one: --sequence <SEQUENCE_NAME>
python src/method/evaluate.py --all       # metrics + figures
```

### B) Train from scratch

Retrains the model (overwrites `checkpoints/best_model.pt`), then runs inference
and evaluation:

```bash
python src/method/train.py                # train (with early stopping)
python src/method/predict.py --all
python src/method/evaluate.py --all
```

Or the whole pipeline in a single command:

```bash
bash src/method/run_method.sh                    # extra flags are forwarded to train.py, e.g.:
bash src/method/run_method.sh --epochs 200 --batch-size 8
```

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

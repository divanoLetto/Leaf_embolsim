"""
config.py — Method 4: U-Net with improved regularisation and augmentation.

Changes vs Method 2:
  - Dropout2d in decoder (p=0.3)
  - Weight decay on Adam (1e-4)
  - Early stopping (patience=15)
  - Richer augmentation calibrated to Cavicam/RPi setup
  - Higher max epochs (budget); early stopping decides the actual end
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.join(REPO_ROOT, "data")

TRAIN_SPLIT_FILE = os.path.join(REPO_ROOT, "train.txt")
VAL_SPLIT_FILE   = os.path.join(REPO_ROOT, "val.txt")
TEST_SPLIT_FILE  = os.path.join(REPO_ROOT, "test.txt")

TRAIN_SEQUENCES = None
VAL_SEQUENCES   = None

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
MODULE_DIR     = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(MODULE_DIR, "checkpoints")
OUTPUT_DIR     = os.path.join(MODULE_DIR, "outputs")
EVAL_DIR       = os.path.join(MODULE_DIR, "evaluation")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
PATCH_SIZE  = 256
BATCH_SIZE  = 16
EPOCHS      = 150       # upper bound; early stopping will terminate earlier
LR          = 1e-4
WEIGHT_DECAY = 1e-4     # L2 regularisation on Adam

IMG_HEIGHT  = 512
IMG_WIDTH   = 512

# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
EARLY_STOP_PATIENCE = 15   # stop if val loss does not improve for this many epochs
EARLY_STOP_MIN_DELTA = 1e-5  # minimum improvement to count as progress

# ---------------------------------------------------------------------------
# Class-imbalance handling
# ---------------------------------------------------------------------------
OVERSAMPLE_FACTOR = 5
PATCH_JITTER      = 32

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
FOCAL_ALPHA  = 0.25
FOCAL_GAMMA  = 2.0
LOSS_FOCAL_W = 0.5
LOSS_DICE_W  = 0.5

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
# Brightness shift applied INDEPENDENTLY to frame_t and frame_{t+1}.
# Calibrated for Cavicam LED illumination with outdoor/greenhouse ambient light.
AUG_BRIGHTNESS_RANGE = (0.92, 1.08)   # ±8% per-frame independent shift

# Gaussian noise sigma range (on [0,255] scale), applied INDEPENDENTLY.
# Calibrated for RPi Camera v2 (IMX219) at 30× magnification.
AUG_NOISE_SIGMA_RANGE = (2.0, 6.0)

# Gamma correction range — SHARED between frame_t and frame_{t+1}.
# Simulates different IMX219 units having slightly different nonlinear responses.
AUG_GAMMA_RANGE       = (0.85, 1.15)

# Per-channel additive shift — SHARED between frame_t and frame_{t+1}.
# Simulates small white-balance differences between sessions/mounts.
# Applied as uniform U(lo, hi) independently per RGB channel, same value for both frames.
AUG_CHANNEL_SHIFT     = (-10.0, 10.0)

# Fine rotation range (degrees) applied JOINTLY — simulates mount placement
# variability between plants/sessions. U(-10, +10) covers realistic misalignment.
AUG_FINE_ROT_RANGE    = (-10.0, 10.0)

# Elastic deformation parameters — applied JOINTLY to both frames and the mask.
# Simulates local tissue deformation from turgour changes over 5-minute intervals.
AUG_ELASTIC_ALPHA     = 70    # deformation amplitude (pixels)
AUG_ELASTIC_SIGMA     = 7     # smoothness of deformation field
AUG_ELASTIC_PROB      = 0.5   # probability of applying elastic deformation

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Spatial dropout rate in decoder blocks.
DROPOUT_P = 0.3

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
SEED = 42

from pathlib import Path
from typing import List, Tuple

import torch

SEED   = 1
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ── altitude buckets ────────────────────────────────────────────────────────
# Each entry: (label, lo_metres_inclusive, hi_metres_exclusive)
ALTITUDE_BUCKETS: List[Tuple[str, float, float]] = [
    ("100m",  0.0,   125.0),
    ("150m",  125.0, 175.0),
    ("200m",  175.0, 225.0),
    ("250m",  225.0, float("inf")),
]

# ── preprocessing ───────────────────────────────────────────────────────────
JPEG_QUALITY = 95   # saved frame quality

# Explicit per-source split assignment
SPLIT_MAP: dict[str, str] = {
    "2022-12-02 Asjo 01_stabilized.zip":      "train",
    "2022-12-04 Bjenberg 02.zip":             "train",
    "2022-12-23 Asjo 01_HD 5x stab.zip":      "train",
    "2022-12-03 Nyland 01_stabilized.zip":    "val",
    "2022-12-23 Bjenberg 02_stabilized.zip":  "test",
}

# ── wandb ───────────────────────────────────────────────────────────────────
WANDB_ENTITY  = "d7047e-group12"
WANDB_PROJECT = "Project-NVD"

# ── paths ───────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent   # …/Project/src
PROJECT_DIR = SRC_DIR.parent                    # …/Project
DATA_DIR    = PROJECT_DIR / "data"
OUT_DIR     = DATA_DIR / "processed"
IMG_DIR     = OUT_DIR / "images"
LBL_DIR     = OUT_DIR / "labels"
MODELS_DIR  = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
AUGS_DIR    = PROJECT_DIR / "augmentations"
RUNS_DIR    = PROJECT_DIR / "runs"


from pathlib import Path
from typing import List, Tuple

SEED = 1


# ── altitude buckets ────────────────────────────────────────────────────────
# Each entry: (label, lo_metres_inclusive, hi_metres_exclusive)
ALTITUDE_BUCKETS: List[Tuple[str, float, float]] = [
    ("100m",  0.0,   125.0),
    ("150m",  125.0, 175.0),
    ("200m",  175.0, 225.0),
    ("250m",  225.0, float("inf")),
]

# ── wandb ───────────────────────────────────────────────────────────────────
WANDB_ENTITY  = "d7047e-group12"
WANDB_PROJECT = "Project-NVD"

# ── paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent   # …/Project
DATA_DIR    = PROJECT_DIR / "data"
OUT_DIR     = PROJECT_DIR / "data" / "processed"
IMG_DIR     = PROJECT_DIR / "data" / "processed" / "images"
LBL_DIR     = PROJECT_DIR / "data" / "processed" / "labels"
MODELS_DIR  = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
AUGS_DIR    = PROJECT_DIR / "augmentations"
RUNS_DIR    = PROJECT_DIR / "runs"


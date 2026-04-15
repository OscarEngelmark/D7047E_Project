from pathlib import Path
import torch

SEED   = 1
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ── paths ────────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent   # …/Project/src
PROJECT_DIR = SRC_DIR.parent                    # …/Project
DATA_DIR    = PROJECT_DIR / "data"
OUT_DIR     = DATA_DIR / "processed"
IMG_DIR     = OUT_DIR / "images"
LBL_DIR     = OUT_DIR / "labels"
MODELS_DIR  = PROJECT_DIR / "models"

for p in [DATA_DIR, OUT_DIR, IMG_DIR, LBL_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
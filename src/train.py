"""
Baseline YOLOv9c-OBB training script.

Builds the model from src/configs/yolov9c-obb.yaml and transfers backbone
weights from the pretrained yolov9c.pt COCO checkpoint (downloaded
automatically by ultralytics on first use).

Usage
-----
python src/train.py                         # all defaults
python src/train.py --epochs 50 --batch 8
python src/train.py --run-name exp-01 --no-wandb
"""

import argparse
import torch
import yaml
import wandb
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
from globals import (
    SEED, DEVICE, OUT_DIR, SRC_DIR, PROJECT_DIR,
    MODELS_DIR, WANDB_ENTITY, WANDB_PROJECT,
)

# ── defaults ────────────────────────────────────────────────────────────────

DEFAULT_EPOCHS   = 100
DEFAULT_PATIENCE = 20
DEFAULT_IMGSZ    = 1920
DEFAULT_BATCH    = 8
DEFAULT_WORKERS  = 16
DEFAULT_RUN_NAME = "test-run"
MODEL_CFG        = SRC_DIR / "configs" / "yolov9c-obb.yaml"
RUNS_DIR         = PROJECT_DIR / "runs"

# Augmentation presets (from NVD paper hyp-aug.yaml / hyp-no-aug.yaml)
AUG_PAPER = dict(
    hsv_h=0.015, hsv_s=0.7,  hsv_v=0.4,
    degrees=45.0, translate=0.1, scale=0.9,
    fliplr=0.5,  flipud=0.5,
    mosaic=1.0,  mixup=0.1,  copy_paste=0.1,
)
AUG_NONE = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0,
    fliplr=0.0, flipud=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv9c-OBB for car detection"
    )
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help="number of training epochs",
    )
    p.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help="input image size",
    )
    p.add_argument(
        "--batch", type=int, default=DEFAULT_BATCH,
        help="batch size",
    )
    p.add_argument(
        "--run-name", type=str, default=DEFAULT_RUN_NAME,
        help="name for this run (wandb + runs/ folder)",
    )
    p.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="number of dataloader workers",
    )
    p.add_argument(
        "--cache", type=str, default="disk", choices=["ram", "disk", "off"],
        help="cache images in ram/disk for faster training, or off to disable",
    )
    p.add_argument(
        "--optimizer", type=str, default="AdamW",
        help="optimizer (AdamW, SGD, Adam, ...)",
    )
    p.add_argument(
        "--lr0", type=float, default=0.002,
        help="initial learning rate (AdamW default: 0.002, SGD default: 0.01)",
    )
    p.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help="early stopping patience in epochs (0 to disable)",
    )
    p.add_argument(
        "--augment", action="store_true",
        help="enable paper augmentations (degrees=45, flipud, mosaic, ...)",
    )
    p.add_argument(
        "--no-wandb", action="store_true",
        help="disable wandb logging",
    )
    return p.parse_args()


def write_dataset_yaml() -> str:
    """Regenerate dataset.yaml with the correct absolute path for this 
    machine."""
    cfg = {
        "path":  str(OUT_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": {0: "car"},
    }
    path = OUT_DIR / "dataset.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return str(path)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(f"Device: {DEVICE}")

    dataset_yaml = write_dataset_yaml()
    print(f"Dataset: {dataset_yaml}")

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=args.run_name,
        config={
            **vars(args),       # all CLI arguments
            "model":  "yolov9c-obb",
            "device": DEVICE,
            "seed":   SEED,
        },
        mode="disabled" if args.no_wandb else "online",
    )

    # Build model from custom OBB config, transfer pretrained backbone weights.
    # Downloads yolov9c.pt to models/ on first use.
    weights = MODELS_DIR / "yolov9c.pt"
    if not weights.exists():
        attempt_download_asset(str(weights))
    model = YOLO(str(MODEL_CFG)).load(str(weights))

    aug = AUG_PAPER if args.augment else AUG_NONE
    model.train(
        data=dataset_yaml,
        task="obb",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache=args.cache if args.cache != "off" else False,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        compile=torch.cuda.is_available(),
        device=DEVICE,
        seed=SEED,
        project=str(RUNS_DIR),
        name=args.run_name,
        **aug,
    )

    wandb.finish()


if __name__ == "__main__":
    main()

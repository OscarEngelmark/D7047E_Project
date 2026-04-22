"""
Baseline YOLOv9-OBB training script.

Builds the model from src/configs and transfers backbone weights from the
pretrained COCO checkpoint (downloaded automatically by ultralytics on first
use).

Usage
-----
python src/train.py                         # all defaults
python src/train.py --epochs 50 --batch 8
python src/train.py --run-name exp-01 --no-wandb
python src/train.py --altitude-aware-scale --alt-min 100 --alt-max 300
"""

import os
import argparse
import torch
import yaml
import wandb
import globals as g

from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
from altitude_augment import AltitudeAwareOBBTrainer
from metadata_callback import register_metadata_callbacks
from typing import Callable

# Set PyTorch CUDA allocator to allow fragmentation (prevents GPU OOM errors)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── defaults ────────────────────────────────────────────────────────────────

DEFAULT_EPOCHS   = 100
DEFAULT_PATIENCE = 20
DEFAULT_IMGSZ    = 1920
DEFAULT_BATCH    = 8
DEFAULT_WORKERS  = 16
DEFAULT_RUN_NAME = "test-run"
DEFAULT_MODEL    = "yolov9s"
RUNS_DIR         = g.PROJECT_DIR / "runs"


# ── helpers ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a YOLOv9-OBB model for car detection"
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
        help="optimizer (AdamW, SGD, MuSGD, ...)",
    )
    p.add_argument(
        "--lr0", type=float, default=0.002,
        help="initial learning rate (AdamW default: 0.002, SGD: 0.01)",
    )
    p.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help="early stopping patience in epochs (0 to disable)",
    )
    p.add_argument(
        "--augment", type=str, default=None, metavar="PRESET",
        help=(
            "augmentation preset filename stem from augmentations/ "
            "(e.g. --augment paper loads augmentations/paper.yaml)"
        ),
    )
    p.add_argument(
        "--freeze", type=int, default=0,
        help="freeze first N backbone layers (0=no freeze, 10=full backbone)",
    )
    p.add_argument(
        "--unfreeze-epoch", type=int, default=0,
        help="epoch at which to unfreeze frozen layers (0=never unfreeze)",
    )
    p.add_argument(
        "--lr-unfreeze-factor", type=float, default=1.0,
        help="multiply all LRs by this factor when backbone is unfrozen",
    )
    p.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, 
        choices=["yolov9s", "yolov9c"], help="model variant to train",
    )
    p.add_argument(
        "--altitude-aware-scale", action="store_true",
        dest="altitude_aware_scale",
        help=(
            "use altitude-aware scale augmentation: sample "
            "h_target ~ U(alt_min, alt_max), apply s = h / h_target"
        ),
    )
    p.add_argument(
        "--alt-min", type=float, default=100.0, dest="alt_min",
        help=(
            "lower bound of target altitude range in metres "
            "(altitude-aware scale only, default: 100)"
        ),
    )
    p.add_argument(
        "--alt-max", type=float, default=300.0, dest="alt_max",
        help=(
            "upper bound of target altitude range in metres "
            "(altitude-aware scale only, default: 300)"
        ),
    )
    p.add_argument(
        "--no-wandb", action="store_true",
        help="disable wandb logging",
    )
    return p.parse_args()


def make_unfreeze_callback(
        unfreeze_epoch: int, lr_factor: float = 1.0
    ) -> Callable:
    def on_train_epoch_start(trainer):
        if trainer.epoch == unfreeze_epoch:
            for _, param in trainer.model.named_parameters():
                param.requires_grad = True
            print(f"[unfreeze] All layers unfrozen at epoch {unfreeze_epoch}")
            if lr_factor != 1.0:
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] *= lr_factor
                    # keeps the scheduler scaling correctly
                    pg["initial_lr"] *= lr_factor  
                new_lr = trainer.optimizer.param_groups[0]["lr"]
                print(f"[unfreeze] LR scaled by {lr_factor} → {new_lr:.6f}")
    return on_train_epoch_start


def write_dataset_yaml() -> str:
    """Regenerate dataset.yaml with the correct absolute path for this 
    machine."""
    cfg = {
        "path":  str(g.OUT_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": {0: "car"},
    }
    path = g.OUT_DIR / "dataset.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return str(path)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(f"Device: {g.DEVICE}")

    dataset_yaml = write_dataset_yaml()
    print(f"Dataset: {dataset_yaml}")

    trainer_cls = (
        AltitudeAwareOBBTrainer if args.altitude_aware_scale else None
    )

    wandb.init(
        entity=g.WANDB_ENTITY,
        project=g.WANDB_PROJECT,
        name=args.run_name,
        dir=str(g.PROJECT_DIR),
        config={
            **vars(args),
            "model":  f"{args.model}-obb",
            "device": g.DEVICE,
            "seed":   g.SEED,
        },
        mode="disabled" if args.no_wandb else "online",
    )

    # Build model from custom OBB config, transfer pretrained backbone weights.
    model_cfg = g.PROJECT_DIR / "configs" / f"{args.model}-obb.yaml"
    weights   = g.MODELS_DIR / f"{args.model}.pt"
    if not weights.exists():
        attempt_download_asset(str(weights))
    model = YOLO(str(model_cfg)).load(str(weights))
    register_metadata_callbacks(model)
    if args.freeze > 0 and args.unfreeze_epoch > 0:
        model.add_callback("on_train_epoch_start", make_unfreeze_callback(
            args.unfreeze_epoch, args.lr_unfreeze_factor,
        ))

    if args.augment:
        aug_path = g.AUGS_DIR / f"{args.augment}.yaml"
        if not aug_path.exists():
            raise FileNotFoundError(
                f"Augmentation preset not found: {aug_path}\n"
                f"Available: {[p.stem for p in g.AUGS_DIR.glob('*.yaml')]}"
            )
        with open(aug_path) as f:
            aug = yaml.safe_load(f)
    else:
        aug = {}
    alt_kwargs = (
        {"alt_min": args.alt_min, "alt_max": args.alt_max}
        if args.altitude_aware_scale else {}
    )
    model.train(
        trainer=trainer_cls,
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
        freeze=args.freeze if args.freeze > 0 else None,
        close_mosaic=0,
        save_period=10,
        compile=torch.cuda.is_available(),
        device=g.DEVICE,
        seed=g.SEED,
        project=str(RUNS_DIR),
        name=args.run_name,
        **aug,
        **alt_kwargs,
    )

    wandb.finish()


if __name__ == "__main__":
    main()

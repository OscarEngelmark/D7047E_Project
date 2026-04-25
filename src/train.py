"""
Baseline YOLOv9-OBB training script.

Builds the model from src/configs and transfers backbone weights from the
pretrained COCO checkpoint (downloaded automatically by ultralytics on first
use).

Usage
-----
python src/train.py                         # all defaults
python src/train.py --epochs 50 --batch 8
python src/train.py --altitude-aware-scale --alt-min 100 --alt-max 300

# Resume from last checkpoint (W&B ID read from runs/<name>/wandb_run_id.txt)
python src/train.py --resume --run-name exp-01

# Resume from a specific checkpoint file
python src/train.py --resume runs/exp-01/weights/epoch50.pt --run-name exp-01

# Resume with a manually supplied W&B run ID (for runs started without
# --resume support)
python src/train.py --resume --run-name exp-01 --wandb-id abc12345
"""

import os
import argparse
from pathlib import Path
import torch
import yaml
import wandb
import globals as g

from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
from altitude_augment import AltitudeAwareOBBTrainer
from metadata_callback import register_metadata_callbacks
from typing import Any, Callable, Dict, Optional

# Set PyTorch CUDA allocator to allow fragmentation (prevents GPU OOM errors)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_EPOCHS   = 100
DEFAULT_PATIENCE = 20
DEFAULT_IMGSZ    = 1920
DEFAULT_BATCH    = 8
DEFAULT_WORKERS  = 16
DEFAULT_RUN_NAME = "test-run"
DEFAULT_MODEL    = "yolov9s"
RUNS_DIR         = g.PROJECT_DIR / "runs"


# ── helpers ──────────────────────────────────────────────────────────────────

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
        help=(
            "cache images in ram/disk for faster training, "
            "or off to disable"
        ),
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
        help=(
            "epoch at which to unfreeze frozen layers (0=never unfreeze)"
        ),
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
        "--alt-mode", type=float, default=None, dest="alt_mode",
        help=(
            "peak of a triangular target altitude distribution in metres. "
            "When omitted, uniform U(alt_min, alt_max) is used. "
            "(altitude-aware scale only)"
        ),
    )
    p.add_argument(
        "--resume", nargs="?", const=True, default=False,
        metavar="CHECKPOINT",
        help=(
            "resume interrupted training from a checkpoint. "
            "Optionally provide a path; defaults to "
            "runs/<run-name>/weights/last.pt"
        ),
    )
    p.add_argument(
        "--wandb-id", type=str, default=None, dest="wandb_id",
        help=(
            "W&B run ID to resume (overrides saved ID; useful for "
            "runs started without --resume support)"
        ),
    )
    return p.parse_args()


def make_unfreeze_callback(
        unfreeze_epoch: int, lr_factor: float = 1.0
    ) -> Callable:
    def on_train_epoch_start(trainer):
        if trainer.epoch == unfreeze_epoch:
            for _, param in trainer.model.named_parameters():
                param.requires_grad = True
            print(
                f"[unfreeze] All layers unfrozen at epoch {unfreeze_epoch}"
            )
            if lr_factor != 1.0:
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] *= lr_factor
                    # keeps the scheduler scaling correctly
                    pg["initial_lr"] *= lr_factor
                new_lr = trainer.optimizer.param_groups[0]["lr"]
                print(
                    f"[unfreeze] LR scaled by {lr_factor} → {new_lr:.6f}"
                )
    return on_train_epoch_start


def make_save_wandb_id_callback() -> Callable:
    def on_train_start(trainer) -> None:
        if wandb.run is None:
            return
        id_file = Path(trainer.save_dir) / "wandb_run_id.txt"
        id_file.parent.mkdir(parents=True, exist_ok=True)
        id_file.write_text(wandb.run.id)
        print(f"[wandb] Run ID saved to {id_file}")
    return on_train_start


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


def _read_saved_run_id(run_name: str) -> Optional[str]:
    """Return the saved W&B run ID for run_name, or None if absent."""
    id_file = RUNS_DIR / run_name / "wandb_run_id.txt"
    return id_file.read_text().strip() if id_file.exists() else None


def resolve_model(args: argparse.Namespace) -> YOLO:
    """Load model from checkpoint (resume) or build fresh from config."""
    if args.resume:
        ckpt = (
            Path(args.resume) if isinstance(args.resume, str)
            else RUNS_DIR / args.run_name / "weights" / "last.pt"
        )
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"Resuming from: {ckpt}")
        return YOLO(str(ckpt))
    model_cfg = g.PROJECT_DIR / "configs" / f"{args.model}-obb.yaml"
    weights   = g.MODELS_DIR / f"{args.model}.pt"
    if not weights.exists():
        attempt_download_asset(str(weights))
    model = YOLO(str(model_cfg))
    model.load(str(weights))
    return model


def resolve_wandb_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Return wandb.init kwargs for a resume or a fresh run."""
    if args.resume:
        run_id = args.wandb_id or _read_saved_run_id(args.run_name)
        if run_id:
            print(f"Resuming W&B run: {run_id}")
            return {"resume": "must", "id": run_id}
        print(
            "No W&B run ID found; starting a new W&B run. "
            "Pass --wandb-id to attach to the original run."
        )
        return {}
    return {
        "config": {
            **vars(args),
            "model":  f"{args.model}-obb",
            "device": g.DEVICE,
            "seed":   g.SEED,
        }
    }


def resolve_train_kwargs(
        args: argparse.Namespace, dataset_yaml: str
) -> Dict[str, Any]:
    """Return model.train kwargs for a resume or a fresh run."""
    trainer_cls = (
        AltitudeAwareOBBTrainer if args.altitude_aware_scale else None
    )
    alt_kwargs = (
        {
            "alt_min":  args.alt_min,
            "alt_max":  args.alt_max,
            "alt_mode": args.alt_mode,
        }
        if args.altitude_aware_scale else {}
    )

    if args.resume:
        return {"resume": True, "trainer": trainer_cls, **alt_kwargs}

    if args.augment:
        aug_path = g.AUGS_DIR / f"{args.augment}.yaml"
        if not aug_path.exists():
            raise FileNotFoundError(
                f"Augmentation preset not found: {aug_path}\n"
                f"Available: "
                f"{[p.stem for p in g.AUGS_DIR.glob('*.yaml')]}"
            )
        with open(aug_path) as f:
            aug = yaml.safe_load(f)
    else:
        aug = {}

    return {
        "trainer":      trainer_cls,
        "data":         dataset_yaml,
        "task":         "obb",
        "epochs":       args.epochs,
        "imgsz":        args.imgsz,
        "batch":        args.batch,
        "workers":      args.workers,
        "cache":        args.cache if args.cache != "off" else False,
        "optimizer":    args.optimizer,
        "lr0":          args.lr0,
        "patience":     args.patience,
        "freeze":       args.freeze if args.freeze > 0 else None,
        "close_mosaic": 0,
        "save_period":  10,
        "compile":      torch.cuda.is_available(),
        "device":       g.DEVICE,
        "seed":         g.SEED,
        "project":      str(RUNS_DIR),
        "name":         args.run_name,
        **aug,
        **alt_kwargs,
    }


def attach_callbacks(model: YOLO, args: argparse.Namespace) -> None:
    """Register all ultralytics training callbacks on model."""
    register_metadata_callbacks(model)
    model.add_callback("on_train_start", make_save_wandb_id_callback())
    if args.freeze > 0 and args.unfreeze_epoch > 0:
        model.add_callback(
            "on_train_epoch_start",
            make_unfreeze_callback(
                args.unfreeze_epoch, args.lr_unfreeze_factor
            ),
        )


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(f"Device: {g.DEVICE}")
    
    dataset_yaml = write_dataset_yaml()
    print(f"Dataset: {dataset_yaml}")

    model = resolve_model(args)
    attach_callbacks(model, args)

    with wandb.init(
        entity=g.WANDB_ENTITY,
        project=g.WANDB_PROJECT,
        name=args.run_name,
        dir=str(g.PROJECT_DIR),
        **resolve_wandb_kwargs(args),
    ):
        model.train(**resolve_train_kwargs(args, dataset_yaml))


if __name__ == "__main__":
    main()

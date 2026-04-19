"""
Evaluate a trained YOLOv9-OBB checkpoint on the test split.

Per-altitude, snow-cover, and cloud-cover metrics are logged to W&B as
summary values (not time-series) so they appear alongside — but separate
from — the training run that produced the checkpoint.

Usage
-----
python src/evaluate.py --weights runs/<run>/weights/best.pt
python src/evaluate.py --weights runs/<run>/weights/best.pt --no-wandb
python src/evaluate.py --weights runs/<run>/weights/best.pt --run-name my-eval
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import wandb
from ultralytics import YOLO

from globals import DEVICE, OUT_DIR, PROJECT_DIR, WANDB_ENTITY, WANDB_PROJECT
from metadata_callback import register_metadata_callbacks
from train import write_dataset_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a YOLOv9-OBB checkpoint on the test split"
    )
    p.add_argument(
        "--weights", type=str, required=True,
        help="path to checkpoint, e.g. runs/<run>/weights/best.pt",
    )
    p.add_argument(
        "--run-name", type=str, default=None,
        help="W&B run name (defaults to 'eval-<weights stem>')",
    )
    p.add_argument(
        "--imgsz", type=int, default=1920,
        help="inference image size (should match training)",
    )
    p.add_argument(
        "--batch", type=int, default=8,
    )
    p.add_argument(
        "--workers", type=int, default=16,
    )
    p.add_argument(
        "--no-wandb", action="store_true",
        help="disable W&B logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = PROJECT_DIR / args.weights
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    run_name = args.run_name or (
        f"eval-{weights_path.stem}-{weights_path.parent.parent.name}"
    )
    dataset_yaml = write_dataset_yaml()
    print(f"Weights:  {weights_path}")
    print(f"Dataset:  {dataset_yaml}")
    print(f"Device:   {DEVICE}")

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        dir=str(PROJECT_DIR),
        config={
            "weights": str(weights_path),
            "imgsz":   args.imgsz,
            "batch":   args.batch,
            "split":   "test",
            "device":  DEVICE,
        },
        mode="disabled" if args.no_wandb else "online",
    )

    model = YOLO(str(weights_path))
    register_metadata_callbacks(model, test_mode=True)

    results = model.val(
        data=dataset_yaml,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=DEVICE,
        project=str(PROJECT_DIR / "runs"),
        name=run_name,
    )

    # Log overall test metrics to summary as well
    if wandb.run is not None:
        box = results.box
        wandb.run.summary.update({
            "test/precision": float(box.mp),
            "test/recall":    float(box.mr),
            "test/mAP50":     float(box.map50),
            "test/mAP50-95":  float(box.map),
        })

    print(f"\nTest mAP50:    {results.box.map50:.4f}")
    print(f"Test mAP50-95: {results.box.map:.4f}")
    print(f"Test precision: {results.box.mp:.4f}")
    print(f"Test recall:    {results.box.mr:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()

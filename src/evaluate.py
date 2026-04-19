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
from pathlib import Path
from typing import Dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import wandb
from ultralytics import YOLO

from globals import (
    ALTITUDE_BUCKETS, DEVICE, PROJECT_DIR, RESULTS_DIR,
    WANDB_ENTITY, WANDB_PROJECT,
)
from metadata_callback import (
    get_last_bucket_metrics, register_metadata_callbacks
)
from train import write_dataset_yaml

METRICS = [
    ("Precision",  "precision"),
    ("Recall",     "recall"),
    ("mAP50",      "mAP50"),
    ("mAP50-95",   "mAP50-95"),
]


def plot_metrics(
    overall: Dict[str, float],
    bucket_metrics: Dict[str, float],
    run_name: str,
) -> Path:
    """Save a 2x2 grid of bar charts — one per metric — to RESULTS_DIR."""
    bucket_labels = [label for label, *_ in ALTITUDE_BUCKETS]
    prefix = "test_alt"

    n_cars_overall = sum(
        int(bucket_metrics.get(f"{prefix}/{b}/n_targets", 0))
        for b in bucket_labels
    )
    x_labels = [f"Overall\n({n_cars_overall} cars)"]
    for b in bucket_labels:
        n = int(bucket_metrics.get(f"{prefix}/{b}/n_targets", 0))
        x_labels.append(f"{b}\n({n} cars)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Test metrics — {run_name}", fontsize=14)

    for ax, (title, key) in zip(axes.flat, METRICS):
        values = [overall[key]]
        for bucket in bucket_labels:
            v = bucket_metrics.get(f"{prefix}/{bucket}/{key}")
            values.append(v if v is not None else 0.0)

        bars = ax.bar(x_labels, values)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)

        for bar, val in zip(bars, values):
            inside = val > 0.88
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.03 if inside else bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="top" if inside else "bottom",
                fontsize=8,
                color="white" if inside else "black",
            )

    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


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

    box = results.box
    overall = {
        "precision": float(box.mp),
        "recall":    float(box.mr),
        "mAP50":     float(box.map50),
        "mAP50-95":  float(box.map),
    }

    if wandb.run is not None:
        wandb.run.summary.update({
            f"test/{k}": v for k, v in overall.items()
        })

    print(f"\nTest mAP50:   {overall['mAP50']:.4f}")
    print(f"Test mAP50-95:  {overall['mAP50-95']:.4f}")
    print(f"Test precision: {overall['precision']:.4f}")
    print(f"Test recall:    {overall['recall']:.4f}")
    out = plot_metrics(overall, get_last_bucket_metrics(), run_name)
    print(f"Plot saved to:  {out}")

    wandb.finish()


if __name__ == "__main__":
    main()

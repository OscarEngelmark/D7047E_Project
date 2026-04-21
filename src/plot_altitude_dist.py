"""
Plot altitude distribution of cars per dataset split.

By default (scale=0, no mosaic) the raw estimated altitudes are shown.
Pass --scale and/or --mosaic to simulate YOLOv9 augmentation on the training
split:

    apparent_altitude = actual_altitude * mosaic_factor / U(1-scale, 1+scale)

mosaic_factor = 2 when --mosaic is set (each sub-image fills ~half the linear
output dimension), 1 otherwise. Augmented altitudes are weighted by
n_boxes / n_samples so total car count is preserved.

Usage
-----
    cd src && python plot_altitude_dist.py
    cd src && python plot_altitude_dist.py --scale 0.8 --mosaic
    cd src && python plot_altitude_dist.py --scale 0.5
    cd src && python plot_altitude_dist.py --out results/altitude_dist.png
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

import globals as g

SPLITS = ["train", "val", "test"]
COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
TRAIN_AUG_COLOR = "#9467BD"

DEFAULT_SCALE = 0.0
DEFAULT_N_SAMPLES = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path,
        default=g.RESULTS_DIR / "altitude_dist.png",
        help="Output path (default: results/altitude_dist.png)",
    )
    p.add_argument(
        "--bins", type=int, default=100,
        help="Number of histogram bins (default: 100)",
    )
    p.add_argument(
        "--scale", type=float, default=DEFAULT_SCALE,
        help=(
            f"Random scale jitter range (default: {DEFAULT_SCALE}). "
            "Augmentation factor = mosaic_factor / U(1-scale, 1+scale)."
        ),
    )
    p.add_argument(
        "--mosaic", action="store_true",
        help="Apply mosaic x2 altitude factor (disabled by default)",
    )
    p.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        dest="n_samples",
        help=(
            f"Augmentation draws per training frame "
            f"(default: {DEFAULT_N_SAMPLES})"
        ),
    )
    p.add_argument(
        "--seed", type=int, default=g.SEED,
        help="RNG seed for reproducibility",
    )
    return p.parse_args()


def load_split_data(
    metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, List[float]]]:
    """Return per-split dicts with 'alts' and 'weights' (n_boxes) lists."""
    by_split: Dict[str, Dict[str, List[float]]] = {
        s: {"alts": [], "weights": []} for s in SPLITS
    }
    for entry in metadata.values():
        alt = entry.get("altitude_m")
        split = entry.get("split")
        n_boxes = entry.get("n_boxes", 1)
        if alt is not None and split in SPLITS:
            by_split[split]["alts"].append(float(alt))
            by_split[split]["weights"].append(float(n_boxes))
    return by_split


def augment_train(
    alts: List[float],
    weights: List[float],
    scale: float,
    n_samples: int,
    rng: np.random.Generator,
    mosaic: bool,
) -> tuple[List[float], List[float]]:
    """Return (augmented_alts, augmented_weights) for the training split.

    apparent_altitude = actual * mosaic_factor / U(1-scale, 1+scale)
    where mosaic_factor = 2 (mosaic) or 1 (no mosaic).
    Weights are divided by n_samples to preserve total car count.
    When scale=0 and mosaic=False, returns the inputs unchanged.
    """
    if scale == 0.0 and not mosaic:
        return alts, weights

    alts_arr = np.array(alts)
    w_arr = np.array(weights) / n_samples
    mosaic_factor = 2.0 if mosaic else 1.0

    raw = rng.uniform(1.0 - scale, 1.0 + scale, size=(len(alts), n_samples))
    aug_alts = (alts_arr[:, None] * mosaic_factor / raw).ravel()
    aug_weights = np.repeat(w_arr, n_samples)

    mask = aug_alts > 0
    return aug_alts[mask].tolist(), aug_weights[mask].tolist()


def plot_histograms(
    by_split: Dict[str, Dict[str, List[float]]],
    aug_alts: List[float],
    aug_weights: List[float],
    bins: int,
    axes: List[matplotlib.axes.Axes],
    x_max: float,
    train_augmented: bool,
) -> None:
    all_alts = (
        [a for s in by_split.values() for a in s["alts"]]
        + aug_alts
    )
    x_min = min(all_alts)
    edges = np.linspace(x_min, x_max, bins + 1).tolist()

    train_label = "train (augmented)" if train_augmented else "train"
    train_color = TRAIN_AUG_COLOR if train_augmented else COLORS["train"]
    rows = [
        (train_label, aug_alts,                   aug_weights,
         train_color),
        ("val",        by_split["val"]["alts"],   by_split["val"]["weights"],
         COLORS["val"]),
        ("test",       by_split["test"]["alts"],  by_split["test"]["weights"],
         COLORS["test"]),
    ]
    for ax, (label, alts, weights, color) in zip(axes, rows):
        total_cars = sum(weights)
        ax.hist(
            alts, bins=edges, weights=weights,
            color=color, label=f"cars={total_cars:,.0f}",
        )
        ax.set_title(label.capitalize())
        ax.set_xlabel("Estimated altitude (m)")
        ax.set_ylabel("Car count")
        ax.legend(fontsize=9)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(g.OUT_DIR / "metadata.json") as f:
        metadata: Dict[str, Dict[str, Any]] = json.load(f)

    by_split = load_split_data(metadata)

    if not by_split["train"]["alts"]:
        raise ValueError("No training data found in metadata.json")

    train_augmented = args.scale != 0.0 or args.mosaic
    aug_alts, aug_weights = augment_train(
        by_split["train"]["alts"],
        by_split["train"]["weights"],
        scale=args.scale,
        n_samples=args.n_samples,
        rng=rng,
        mosaic=args.mosaic,
    )

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, squeeze=False
    )
    plot_histograms(
        by_split, aug_alts, aug_weights,
        bins=args.bins,
        axes=list(axes[:, 0]),
        x_max=500.0,
        train_augmented=train_augmented,
    )

    if train_augmented:
        mosaic_str = (
            f"2/U(1±{args.scale})" if args.mosaic
            else f"1/U(1±{args.scale})"
        )
        title = (
            f"Altitude distribution — train augmented "
            f"(scale={args.scale}, factor={mosaic_str})"
        )
    else:
        title = "Altitude distribution by split"
    fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()

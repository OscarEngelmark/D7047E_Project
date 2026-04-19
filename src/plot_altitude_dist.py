"""
Plot altitude distribution of annotated cars per dataset split.

Reads data/processed/metadata.json (written by preprocessing.py) and
produces a 1x3 figure with one histogram panel per split (train / val /
test). All panels share the same x-axis range.

Usage
-----
    cd src && python plot_altitude_dist.py
    cd src && python plot_altitude_dist.py --out results/altitude_dist.png
    cd src && python plot_altitude_dist.py --bins 30
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from globals import OUT_DIR, RESULTS_DIR

SPLITS = ["train", "val", "test"]
COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path,
        default=RESULTS_DIR / "altitude_dist.png",
        help="Output file path (default: results/altitude_dist.png)",
    )
    p.add_argument(
        "--bins", type=int, default=100,
        help="Number of histogram bins (default: 100)",
    )
    return p.parse_args()


def load_altitudes(
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


def plot_split_histograms(
    by_split: Dict[str, Dict[str, List[float]]],
    bins: int,
    axes: List[matplotlib.axes.Axes],
) -> None:
    all_alts = [a for s in by_split.values() for a in s["alts"]]
    x_min, x_max = min(all_alts), 350.0
    edges = np.linspace(x_min, x_max, bins + 1).tolist()

    for ax, split in zip(axes, SPLITS):
        alts = by_split[split]["alts"]
        weights = by_split[split]["weights"]
        total_cars = int(sum(weights))
        ax.hist(
            alts, bins=edges, weights=weights,
            color=COLORS[split], label=f"cars={total_cars:,}",
        )
        ax.set_title(split.capitalize())
        ax.set_xlabel("Estimated altitude (m)")
        ax.set_ylabel("Car count")
        ax.legend(fontsize=9)


def main() -> None:
    args = parse_args()

    with open(OUT_DIR / "metadata.json") as f:
        metadata: Dict[str, Dict[str, Any]] = json.load(f)

    by_split = load_altitudes(metadata)

    if not any(by_split[s]["alts"] for s in SPLITS):
        raise ValueError("No altitude data found in metadata.json")

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, squeeze=False
    )
    plot_split_histograms(by_split, args.bins, list(axes[:, 0]))
    fig.suptitle("Altitude distribution by split", fontsize=11)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()

"""Augmentation comparison: three altitude distributions for one split.

Panels (shared x-axis):
  (1) Unmodified             — raw estimated altitudes
  (2) Scale + mosaic         — standard YOLOv9 scale jitter
  (3) Altitude-aware scale   — per-frame h_target sampling (AAS)

Usage
-----
    cd src && python plots/aug_comparison.py
    cd src && python plots/aug_comparison.py --split train
    cd src && python plots/aug_comparison.py --scale 0.7 --mosaic
    cd src && python plots/aug_comparison.py --alt-min 80 --alt-max 400
    cd src && python plots/aug_comparison.py --style report
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import style
from altitude_dist import (
    DEFAULT_N_SAMPLES,
    augment_train,
    augment_train_altitude_aware,
    load_split_data,
)

DEFAULT_SCALE = 0.5

COLORS = {
    "raw":   "#4C72B0",
    "scale": "#DD8452",
    "aas":   "#9467BD",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split", choices=["train", "val", "test"], default="train",
        help="Split to compare augmentations for (default: train)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help=(
            "Output path "
            "(default: results/aug_comparison_{split}.{pdf|png})"
        ),
    )
    p.add_argument(
        "--style", choices=style.STYLES, default=None,
        help="Output style: 'report' (PDF) or 'ppt' (PNG)",
    )
    p.add_argument(
        "--bins", type=int, default=200,
        help="Number of histogram bins (default: 200)",
    )
    p.add_argument(
        "--scale", type=float, default=DEFAULT_SCALE,
        help=f"Scale jitter range for panel 2 (default: {DEFAULT_SCALE})",
    )
    p.add_argument(
        "--mosaic", action="store_true",
        help="Apply mosaic ×2 altitude factor for panel 2",
    )
    p.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        dest="n_samples",
        help=f"Augmentation draws per frame (default: {DEFAULT_N_SAMPLES})",
    )
    p.add_argument(
        "--alt-min", type=float, default=100.0, dest="alt_min",
        help="AAS lower bound in metres (default: 100)",
    )
    p.add_argument(
        "--alt-max", type=float, default=300.0, dest="alt_max",
        help="AAS upper bound in metres (default: 300)",
    )
    p.add_argument(
        "--dist", choices=["uniform", "triangular"], default="triangular",
        help="AAS target altitude distribution (default: triangular)",
    )
    p.add_argument(
        "--alt-mode", type=float, default=None, dest="alt_mode",
        help=(
            "Peak of triangular AAS distribution (default: midpoint). "
            "Only used with --dist triangular."
        ),
    )
    p.add_argument(
        "--x-max", type=float, default=600.0, dest="x_max",
        help="Upper x-axis limit (default: 600)",
    )
    p.add_argument(
        "--seed", type=int, default=g.SEED,
        help="RNG seed for reproducibility",
    )
    return p.parse_args()


def plot_panels(
    rows: List[Tuple[str, List[float], List[float], str]],
    axes: List[matplotlib.axes.Axes],
    bins: int,
    x_max: float,
) -> None:
    """Draw one weighted histogram per axis.

    rows: list of (panel_title, alts, weights, color)
    """
    all_alts = [a for _, alts, _, _ in rows for a in alts]
    edges = np.linspace(min(all_alts), x_max, bins + 1).tolist()

    for ax, (title, alts, weights, color) in zip(axes, rows):
        ax.hist(
            alts, bins=edges, weights=weights,
            color=color, label=f"cars = {sum(weights):,.0f}",
        )
        ax.set_title(title)
        ax.set_xlabel("Estimated altitude (m)")
        ax.set_ylabel("Car count")
        ax.legend()

    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(top=y_max)


def main() -> None:
    args = parse_args()
    fmt = style.output_fmt(args.style) if args.style else "png"
    dpi = style.save_dpi(args.style) if args.style else 150
    if args.out is None:
        args.out = (
            g.RESULTS_DIR / f"aug_comparison_{args.split}.{fmt}"
        )
    if args.style:
        style.apply_style(args.style)

    rng = np.random.default_rng(args.seed)

    with open(g.OUT_DIR / "metadata.json") as f:
        metadata: Dict[str, Dict[str, Any]] = json.load(f)

    by_split = load_split_data(metadata)
    alts: List[float] = by_split[args.split]["alts"]
    weights: List[float] = by_split[args.split]["weights"]

    if not alts:
        raise ValueError(f"No altitude data found for split '{args.split}'")

    scale_alts, scale_weights = augment_train(
        alts, weights,
        scale=args.scale,
        n_samples=args.n_samples,
        rng=rng,
        mosaic=args.mosaic,
    )

    mode: Optional[float] = args.alt_mode
    if args.dist == "triangular" and mode is None:
        mode = (args.alt_min + args.alt_max) / 2
    aas_alts, aas_weights = augment_train_altitude_aware(
        alts, weights,
        alt_min=args.alt_min,
        alt_max=args.alt_max,
        n_samples=args.n_samples,
        rng=rng,
        dist=args.dist,
        alt_mode=mode,
    )

    mosaic_str = "mosaic on" if args.mosaic else "mosaic off"
    scale_title = (
        f"Scale + mosaic  (scale={args.scale}, {mosaic_str})"
    )
    if args.dist == "triangular":
        aas_title = (
            f"Altitude-aware scale  "
            f"(triangular({args.alt_min:.0f}, "
            f"{mode:.0f}, {args.alt_max:.0f}) m)"
        )
    else:
        aas_title = (
            f"Altitude-aware scale  "
            f"(uniform [{args.alt_min:.0f}, {args.alt_max:.0f}] m)"
        )

    rows: List[Tuple[str, List[float], List[float], str]] = [
        ("Unmodified",   alts,        weights,        COLORS["raw"]),
        (scale_title,    scale_alts,  scale_weights,  COLORS["scale"]),
        (aas_title,      aas_alts,    aas_weights,    COLORS["aas"]),
    ]

    fs = (
        style.figsize(args.style, n_rows=3)
        if args.style else (9, 10)
    )
    fig, axes = plt.subplots(3, 1, figsize=fs, sharex=True, squeeze=False)
    plot_panels(rows, list(axes[:, 0]), bins=args.bins, x_max=args.x_max)
    fig.suptitle(f"Augmentation comparison — {args.split} split")
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()

"""
Plot altitude distribution of cars per dataset split.

By default (scale=0, no mosaic) the raw estimated altitudes are shown.

--scale / --mosaic: simulate standard YOLOv9 scale augmentation:
    apparent_altitude = actual * mosaic_factor / U(1-scale, 1+scale)

--altitude-aware: simulate altitude-aware augmentation where for each
training frame at altitude h, a target altitude is sampled over
[alt_min, alt_max] and the required scale factor s = h / h_target is
applied:
    apparent_altitude = h_target  (exactly, when s is within clamp bounds)
Scale factors are clamped to [0.1, 4.0] to stay within feasible image
scaling limits; frames where clamping alters the target are still included
but their apparent altitude will differ from h_target.

--dist: target altitude distribution to use with --altitude-aware.
    uniform    (default): h_target ~ U(alt_min, alt_max)
    triangular:           h_target ~ Triangular(alt_min, alt_mode, alt_max)
                          --alt-mode sets the peak (default: midpoint)

Augmented altitudes are weighted by n_boxes / n_samples so total car count
is preserved.

Usage
-----
    python plot_altitude_dist.py
    python plot_altitude_dist.py --scale 0.7
    python plot_altitude_dist.py --altitude-aware
    python plot_altitude_dist.py --altitude-aware --alt-min 80 --alt-max 400
    python plot_altitude_dist.py --altitude-aware --dist triangular \
        --alt-mode 250
    python plot_altitude_dist.py --out results/altitude_dist.png
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

import globals as g

SPLITS = ["train", "val", "test"]
COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
TRAIN_AUG_COLOR = "#9467BD"

DEFAULT_SCALE = 0.0
DEFAULT_N_SAMPLES = 100
SCALE_FLOOR = 0.1   # minimum feasible image scale factor
SCALE_CEILING = 4.0  # maximum feasible image scale factor


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
        "--altitude-aware", action="store_true", dest="altitude_aware",
        help=(
            "Simulate altitude-aware scale augmentation: sample "
            "h_target ~ U(alt_min, alt_max) per frame, apply s = h/h_target"
        ),
    )
    p.add_argument(
        "--alt-min", type=float, default=100.0, dest="alt_min",
        help="Lower bound of target altitude range in metres (default: 80)",
    )
    p.add_argument(
        "--alt-max", type=float, default=300.0, dest="alt_max",
        help="Upper bound of target altitude range in metres (default: 300)",
    )
    p.add_argument(
        "--dist", choices=["uniform", "triangular"], default="uniform",
        help=(
            "Target altitude distribution for --altitude-aware: "
            "'uniform' (default) or 'triangular' (see --alt-mode)"
        ),
    )
    p.add_argument(
        "--alt-mode", type=float, default=None, dest="alt_mode",
        help=(
            "Peak of the triangular target distribution in metres. "
            "Only used with --dist triangular. "
            "Defaults to midpoint of [alt_min, alt_max]."
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
) -> Tuple[List[float], List[float]]:
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


def augment_train_altitude_aware(
    alts: List[float],
    weights: List[float],
    alt_min: float,
    alt_max: float,
    n_samples: int,
    rng: np.random.Generator,
    dist: str = "uniform",
    alt_mode: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    """Altitude-aware augmentation: sample h_target from the chosen
    distribution over [alt_min, alt_max], compute s = h / h_target,
    clamp to [SCALE_FLOOR, SCALE_CEILING], then apparent_altitude = h / s.
    When s is unclamped, apparent_altitude == h_target exactly.

    dist='uniform'    -> h_target ~ U(alt_min, alt_max)
    dist='triangular' -> h_target ~ Triangular(alt_min, alt_mode, alt_max)
                         alt_mode defaults to midpoint when None
    """
    alts_arr = np.array(alts)
    w_arr = np.array(weights) / n_samples

    size = (len(alts), n_samples)
    if dist == "triangular":
        mode = alt_mode if alt_mode is not None else (alt_min + alt_max) / 2
        h_target = rng.triangular(alt_min, mode, alt_max, size=size)
    else:
        h_target = rng.uniform(alt_min, alt_max, size=size)

    s = alts_arr[:, None] / h_target
    s = np.clip(s, SCALE_FLOOR, SCALE_CEILING)

    aug_alts = (alts_arr[:, None] / s).ravel()
    aug_weights = np.repeat(w_arr, n_samples)
    return aug_alts.tolist(), aug_weights.tolist()


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

    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(top=y_max)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(g.OUT_DIR / "metadata.json") as f:
        metadata: Dict[str, Dict[str, Any]] = json.load(f)

    by_split = load_split_data(metadata)

    if not by_split["train"]["alts"]:
        raise ValueError("No training data found in metadata.json")

    if args.altitude_aware:
        mode = args.alt_mode
        if args.dist == "triangular" and mode is None:
            mode = (args.alt_min + args.alt_max) / 2
        aug_alts, aug_weights = augment_train_altitude_aware(
            by_split["train"]["alts"],
            by_split["train"]["weights"],
            alt_min=args.alt_min,
            alt_max=args.alt_max,
            n_samples=args.n_samples,
            rng=rng,
            dist=args.dist,
            alt_mode=mode,
        )
        train_augmented = True
        if args.dist == "triangular":
            title = (
                f"Altitude distribution — altitude-aware augmentation "
                f"(target: triangular({args.alt_min:.0f}, "
                f"{mode:.0f}, {args.alt_max:.0f}) m)"
            )
        else:
            title = (
                f"Altitude distribution — altitude-aware augmentation "
                f"(target: {args.alt_min:.0f}–{args.alt_max:.0f} m)"
            )
    else:
        aug_alts, aug_weights = augment_train(
            by_split["train"]["alts"],
            by_split["train"]["weights"],
            scale=args.scale,
            n_samples=args.n_samples,
            rng=rng,
            mosaic=args.mosaic,
        )
        train_augmented = args.scale != 0.0 or args.mosaic
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

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, squeeze=False
    )
    plot_histograms(
        by_split, aug_alts, aug_weights,
        bins=args.bins,
        axes=list(axes[:, 0]),
        x_max=350.0,
        train_augmented=train_augmented,
    )
    fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()

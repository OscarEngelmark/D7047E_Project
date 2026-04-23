"""
K-means cluster analysis of OBB annotation dimensions.

Reads processed label files (YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4,
normalized) across all splits and clusters the OBB widths/heights in
pixel space.  Outputs a scatter plot with cluster centres, marginal
histograms, and a table of cluster stats.

Useful for verifying that detected box sizes are within the range
each FPN stride can resolve (strides 8, 16, 32 for YOLOv9).

Usage
-----
    python src/plot_box_clusters.py
    python src/plot_box_clusters.py --k 6 --splits train val
    python src/plot_box_clusters.py --out results/box_clusters.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans

# Allow running from repo root or src/
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
import globals as g

IMG_W = 1920
IMG_H = 1080

# YOLOv9 FPN stride grid anchors (minimum receptive field per level)
FPN_STRIDES = [8, 16, 32]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--k", type=int, default=9,
        help="number of k-means clusters (default: 9)",
    )
    p.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="which splits to include (default: all)",
    )
    p.add_argument(
        "--out", type=Path,
        default=g.RESULTS_DIR / "box_clusters.png",
        help="output path (default: results/box_clusters.png)",
    )
    return p.parse_args()


def obb_dims(pts: np.ndarray) -> tuple[float, float]:
    """Return (short_side, long_side) in normalised units for one OBB."""
    p1, p2, p3 = pts[0], pts[1], pts[2]
    a = float(np.linalg.norm(p2 - p1))
    b = float(np.linalg.norm(p3 - p2))
    return (min(a, b), max(a, b))


def load_boxes(splits: list[str]) -> np.ndarray:
    """Return (N, 2) array of [w_px, h_px] using per-image actual size."""
    from PIL import Image

    img_size_cache: dict[Path, tuple[int, int]] = {}

    def img_size(lbl_path: Path, split: str) -> tuple[int, int]:
        img_path = g.IMG_DIR / split / (lbl_path.stem + ".jpg")
        if img_path not in img_size_cache:
            try:
                img_size_cache[img_path] = Image.open(img_path).size
            except FileNotFoundError:
                img_size_cache[img_path] = (IMG_W, IMG_H)
        return img_size_cache[img_path]

    rows = []
    for split in splits:
        lbl_dir = g.LBL_DIR / split
        for path in sorted(lbl_dir.glob("*.txt")):
            iw, ih = img_size(path, split)
            for line in path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                coords = np.array(parts[1:], dtype=float).reshape(4, 2)
                w_n, h_n = obb_dims(coords)
                rows.append([w_n * iw, h_n * ih])
    if not rows:
        raise RuntimeError("No annotations found — run preprocessing.py first.")
    return np.array(rows, dtype=float)


def print_stats(boxes: np.ndarray, labels: np.ndarray, k: int) -> None:
    short = boxes[:, 0]    # short side (always ≤ long side)
    areas = boxes[:, 0] * boxes[:, 1]
    print(f"\n{'─'*60}")
    print(f"  Total annotations: {len(boxes):,}")
    print(f"  Short side (px): min={short.min():.1f}  "
          f"median={np.median(short):.1f}  "
          f"max={short.max():.1f}")
    print(f"  Long  side (px): min={boxes[:,1].min():.1f}  "
          f"median={np.median(boxes[:,1]):.1f}  "
          f"max={boxes[:,1].max():.1f}")
    print()
    # FPN-level coverage: short side below stride threshold is hardest to detect
    for stride in FPN_STRIDES:
        n = (short < stride).sum()
        print(f"  Short side < stride {stride:2d} px : "
              f"{n:5,}  ({100*n/len(boxes):4.1f} %)  "
              f"← below stride-{stride} FPN level")
    coco_small = (areas < 32**2).sum()
    print(f"  Area < 32² px  (COCO small): {coco_small:,} "
          f"({100*coco_small/len(boxes):.1f} %)  "
          f"[threshold from 640 px images, less meaningful at imgsz=1920]")
    print(f"\n  {'Cluster':>8}  {'w_px':>7}  {'h_px':>7}  "
          f"{'area_px²':>10}  {'count':>6}  {'%':>5}")
    print(f"  {'─'*55}")
    for c in sorted(np.unique(labels)):
        mask = labels == c
        cw = boxes[mask, 0].mean()
        ch = boxes[mask, 1].mean()
        n = mask.sum()
        print(f"  {c:>8}  {cw:>7.1f}  {ch:>7.1f}  "
              f"{cw*ch:>10.0f}  {n:>6}  {100*n/len(boxes):>4.1f}%")
    print(f"{'─'*60}\n")


def main() -> None:
    args = parse_args()

    print(f"Loading boxes from splits: {args.splits} …")
    boxes = load_boxes(args.splits)
    print(f"  {len(boxes):,} annotations loaded.")

    km = KMeans(n_clusters=args.k, n_init=20, random_state=g.SEED)
    labels = km.fit_predict(boxes)
    centres = km.cluster_centers_          # (k, 2) in px

    print_stats(boxes, labels, args.k)

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(
        2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
        hspace=0.05, wspace=0.05,
    )
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    fig.add_subplot(gs[0, 1]).set_visible(False)   # blank corner

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(args.k)]

    # scatter coloured by cluster
    for c in range(args.k):
        mask = labels == c
        ax_main.scatter(
            boxes[mask, 0], boxes[mask, 1],
            s=5, alpha=0.25, color=colors[c], linewidths=0,
        )

    # cluster centres
    ax_main.scatter(
        centres[:, 0], centres[:, 1],
        s=180, marker="*", color="black", zorder=5,
        label="cluster centres",
    )
    for i, (cx, cy) in enumerate(centres):
        ax_main.annotate(
            f"#{i}\n{cx:.0f}×{cy:.0f}",
            (cx, cy), textcoords="offset points",
            xytext=(6, 6), fontsize=7,
        )

    # FPN stride reference lines
    stride_styles = [("--", "steelblue"), ("-.", "firebrick"),
                     (":", "darkorange")]
    for stride, (ls, col) in zip(FPN_STRIDES, stride_styles):
        ax_main.axvline(stride, ls=ls, lw=1.2, color=col,
                        label=f"stride {stride} px")
        ax_main.axhline(stride, ls=ls, lw=1.2, color=col)

    ax_main.set_xlabel("OBB short side (px)")
    ax_main.set_ylabel("OBB long side (px)")
    ax_main.legend(fontsize=8, loc="upper left")

    # marginal histograms
    for c in range(args.k):
        mask = labels == c
        ax_top.hist(boxes[mask, 0], bins=60, alpha=0.5, color=colors[c])
        ax_right.hist(
            boxes[mask, 1], bins=60, alpha=0.5,
            color=colors[c], orientation="horizontal",
        )
    ax_top.set_ylabel("count")
    ax_right.set_xlabel("count")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # cluster legend patch
    patches = [
        mpatches.Patch(color=colors[c], label=f"cluster {c}")
        for c in range(args.k)
    ]
    fig.legend(
        handles=patches, loc="upper right",
        fontsize=8, title="clusters", title_fontsize=8,
    )

    fig.suptitle(
        f"OBB box-size clusters  (k={args.k}, "
        f"splits: {', '.join(args.splits)}, "
        f"n={len(boxes):,})",
        fontsize=11,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()

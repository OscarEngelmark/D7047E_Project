"""
Sanity check: plot per-frame altitude estimates and 1/l polynomial fits.

Reads data/processed/metadata.json (written by preprocessing.py) and
data/video_data.csv.  Produces one column per video with two rows:

  Top:    mean bounding-box diagonal (px) vs frame index
          + 4th-degree polynomial fit to 1/l back-projected to px
  Bottom: estimated altitude (m) vs frame index
          + h_max ceiling (red dashed) and h_min floor (orange dashed)

Usage
-----
    python src/plot_altitude.py
    python src/plot_altitude.py --out results/altitude.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import globals as g

from collections import defaultdict
from pathlib import Path
from frame_metadata import load_video_csv, estimate_altitudes_with_fit
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path,
        default=g.RESULTS_DIR / "altitudes.png",
        help="where to save the figure (default: data/processed/altitude.png)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(g.OUT_DIR / "metadata.json") as f:
        metadata: Dict[str, Dict[str, Any]] = json.load(f)

    video_csv = load_video_csv()

    # Group frame entries by video stem
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in metadata.values():
        by_video[entry["video"]].append(entry)

    videos    = sorted(by_video)
    n_videos  = len(videos)
    fig, axes = plt.subplots(
        2, n_videos, figsize=(5 * n_videos, 8), squeeze=False
    )

    for col, video_stem in enumerate(videos):
        frames_sorted = sorted(
            by_video[video_stem], key=lambda e: e["frame_id"]
        )
        frame_ids = np.array(
            [e["frame_id"]     for e in frames_sorted], dtype=float
        )
        diags = np.array(
            [e["mean_diag_px"] for e in frames_sorted], dtype=float
        )
        alts = np.array(
            [e["altitude_m"]   for e in frames_sorted], dtype=float
        )

        vmeta        = video_csv.get(video_stem, {})
        h_max        = vmeta.get("h_max")
        altitude_str = vmeta.get("altitude_str", "")

        # Refit polynomial (new algorithm) from stored diagonals
        frame_diagonals = {
            int(e["frame_id"]): e["mean_diag_px"] for e in frames_sorted
        }
        _, _, _, coeffs = estimate_altitudes_with_fit(
            frame_diagonals, h_max or 1.0
        )

        span    = max(frame_ids.max() - frame_ids.min(), 1.0)
        t_dense = np.linspace(0.0, 1.0, 1000)
        f_dense = frame_ids.min() + t_dense * span

        # ── top: diagonal scatter + polynomial back-projected to px ─────────
        ax_top = axes[0, col]
        ax_top.scatter(frame_ids, diags, s=12, alpha=0.6, label="raw diagonal")
        if coeffs.size > 0:
            inv_poly = np.polyval(coeffs, t_dense)
            with np.errstate(divide="ignore", invalid="ignore"):
                diag_poly = np.where(inv_poly > 0, 1.0 / inv_poly, np.nan)
            ax_top.plot(
                f_dense, diag_poly, "r-", lw=1.5,
                label="poly fit (1/l → px)",
            )
        ax_top.set_title(video_stem.replace("_", " "), fontsize=8)
        ax_top.set_xlabel("frame index")
        ax_top.set_ylabel("mean diagonal (px)")
        ax_top.legend(fontsize=7)

        # ── bottom: altitude scatter + polynomial in altitude space ─────────
        ax_bot = axes[1, col]
        ax_bot.scatter(frame_ids, alts, s=12, alpha=0.6, label="est. altitude")

        if coeffs.size > 0 and h_max is not None:
            inv_poly = np.polyval(coeffs, t_dense)
            poly_max = float(inv_poly.max())
            alt_poly = h_max * inv_poly / poly_max
            ax_bot.plot(f_dense, alt_poly, "r-", lw=1.5, label="poly fit")

        if h_max is not None:
            ax_bot.axhline(h_max, color="red", ls="--", lw=1,
                           label=f"h_max = {h_max:.0f} m")
            parts = altitude_str.lower().replace(" m", "").strip().split("-")
            if len(parts) == 2:
                h_min = float(parts[0])
                ax_bot.axhline(h_min, color="orange", ls="--", lw=1,
                               label=f"h_min = {h_min:.0f} m")

        ax_bot.set_xlabel("frame index")
        ax_bot.set_ylabel("altitude (m)")
        ax_bot.legend(fontsize=7)

    fig.suptitle("Altitude sanity check — per video", fontsize=11)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()

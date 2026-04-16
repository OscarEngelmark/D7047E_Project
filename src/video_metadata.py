"""
Per-frame flight metadata computation for NVD drone videos.

Provides two public entry points:

  load_video_csv()
      Reads data/video_data.csv and returns a dict keyed by video name
      (matching the zip stem) with fields: h_min, altitude_str, snow_cover,
      cloud_cover.

  compute_frame_metadata(annotations, img_w, img_h, video_meta)
      Derives per-frame measurements from annotation data + video_meta and
      returns {frame_id: {"mean_diag_px": float, "altitude_m": float, ...}}.

Extension points
----------------
- To swap in a different altitude algorithm, replace estimate_altitudes().
- To add new per-frame signals (tilt, pitch, GSD, …), extend
  compute_frame_metadata() — its return dict is spread directly into each
  frame's metadata.json entry, so new keys appear automatically.
"""

from __future__ import annotations

import csv
import numpy as np

from globals import DATA_DIR


# ── CSV loading ──────────────────────────────────────────────────────────────

def _parse_h_min(altitude_str: str) -> float:
    """Parse minimum altitude (m) from 
    '130-200 m' → 130.0 or '250 m' → 250.0."""
    s = altitude_str.lower().replace(" m", "").strip()
    return float(s.split("-")[0])


def load_video_csv() -> dict[str, dict]:
    """Load per-video metadata from data/video_data.csv.

    Returns a dict keyed by video name (CSV 'Video' column, which matches the
    zip stem without extension). Only rows where Annotated=TRUE are included.

    Each entry contains:
        h_min        : float  – minimum flight altitude in metres
        altitude_str : str    – raw string from CSV (e.g. '130-200 m')
        snow_cover   : str    – e.g. 'Minimal (0-1 cm)'
        cloud_cover  : str    – e.g. 'Overcast'
    """
    csv_path = DATA_DIR / "video_data.csv"
    result: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Annotated", "").strip().upper() != "TRUE":
                continue
            name = row["Video"].strip()
            result[name] = {
                "h_min":        _parse_h_min(row["Flight Altitude"].strip()),
                "altitude_str": row["Flight Altitude"].strip(),
                "snow_cover":   row["Snow Cover"].strip(),
                "cloud_cover":  row["Cloud Cover"].strip(),
            }
    return result


# ── per-frame computation ────────────────────────────────────────────────────

def compute_frame_diagonals(
    annotations: dict[int, list], img_w: int, img_h: int
) -> dict[int, float]:
    """Per-frame mean bounding-box diagonal length, in pixels.

    Rotation does not change a rectangle's diagonal, so the OBB angle is
    ignored here.
    """
    diagonals: dict[int, float] = {}
    for frame_id, boxes in annotations.items():
        if not boxes:
            continue
        diags = [
            float(np.hypot(w * img_w, h * img_h))
            for _, _, w, h, _ in boxes
        ]
        diagonals[frame_id] = float(np.mean(diags))
    return diagonals


def estimate_altitudes(
    frame_diagonals: dict[int, float], h_min: float
) -> dict[int, float]:
    """Per-frame altitude estimate (metres) using the paper's method.

    Fits a 4th-degree polynomial to (frame_id, mean_diagonal) across all
    annotated frames, finds its maximum l_max (closest drone pass), then
    returns H_frame = h_min · l_max / l_frame for each frame.

    The time axis is normalised to [0, 1] for numerical stability.
    """
    if not frame_diagonals:
        return {}

    frames = np.array(sorted(frame_diagonals.keys()), dtype=float)
    diags  = np.array([frame_diagonals[int(f)] for f in frames])

    span = max(frames.max() - frames.min(), 1.0)
    t    = (frames - frames.min()) / span

    if len(frames) >= 5:
        coeffs  = np.polyfit(t, diags, deg=4)
        t_dense = np.linspace(0.0, 1.0, 1000)
        l_max   = float(np.polyval(coeffs, t_dense).max())
    else:
        l_max = float(diags.max())

    return {
        int(f): float(h_min * l_max / d) 
        for f, d in zip(frames, diags) if d > 0
    }


def compute_frame_metadata(
    annotations: dict[int, list],
    img_w: int,
    img_h: int,
    video_meta: dict,
) -> dict[int, dict]:
    """Compute per-frame metadata for every annotated frame in a video.

    Returns {frame_id: {field: value, ...}}. The dict is spread directly into
    each frame's metadata.json entry, so adding a new field here automatically
    propagates to storage and to the W&B callback.

    Parameters
    ----------
    annotations : {frame_id: [(cx, cy, w, h, angle), ...]}
    img_w, img_h : image dimensions in pixels
    video_meta   : entry from load_video_csv() for this video
    """
    diagonals = compute_frame_diagonals(annotations, img_w, img_h)
    altitudes = estimate_altitudes(diagonals, video_meta["h_min"])
    return {
        frame_id: {
            "mean_diag_px": diagonals.get(frame_id),
            "altitude_m":   altitudes.get(frame_id),
        }
        for frame_id in annotations
    }

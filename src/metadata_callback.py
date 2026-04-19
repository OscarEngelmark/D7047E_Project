"""
Per-altitude-bucket metrics for ultralytics OBB validation.

Reads per-image altitude estimates from `data/processed/metadata.json` (written
by preprocessing.py), then on every validation pass groups images by the
paper's altitude buckets and logs per-bucket precision / recall / mAP50 /
mAP50-95 to W&B.

Design
------
Ultralytics' validator clears its per-image stats inside `get_stats()` before
the `on_val_end` callback fires, and it never stores the image-file ↔ stats
mapping anywhere we can reach. To work around both, `on_val_start` wraps
`validator.update_metrics` so each batch's `im_file` list is paired with the
fresh entries appended to `validator.metrics.stats` and stashed on the
validator itself. `on_val_end` then groups, computes per-bucket metrics with
ultralytics' own `ap_per_class`, and logs them to W&B.

Usage
-----
    from metadata_callback import register_metadata_callbacks
    register_metadata_callbacks(model)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import wandb

from globals import OUT_DIR

# Paper Table 4 altitudes are reported as 120 m, 130 m, 130-200 m, 150 m,
# 200 m, and 250 m. Each bucket below is centred on one of those values; the
# 130-200 m flights spread across the 130/150/200 buckets according to
# per-frame estimate. Edges are exclusive on the upper side.
ALTITUDE_BUCKETS: List[Tuple[str, float, float]] = [
    ("120m",  0.0,   125.0),
    ("130m",  125.0, 140.0),
    ("150m",  140.0, 175.0),
    ("200m",  175.0, 225.0),
    ("250m",  225.0, float("inf")),
]


# ── module state ───────────────────────────────────────────────────────────

_metadata_cache: Optional[Dict[str, Dict[str, Any]]] = None
_val_count: int = 0  # incremented per val pass so W&B logs map to train epochs


def _load_metadata() -> Dict[str, Dict[str, Any]]:
    global _metadata_cache
    if _metadata_cache is None:
        path = OUT_DIR / "metadata.json"
        with open(path) as f:
            _metadata_cache = json.load(f)
    assert _metadata_cache is not None
    return _metadata_cache


def _bucket_for(altitude: Optional[float]) -> Optional[str]:
    if altitude is None:
        return None
    for label, lo, hi in ALTITUDE_BUCKETS:
        if lo <= altitude < hi:
            return label
    return None


# ── ultralytics callbacks ──────────────────────────────────────────────────

def _make_val_start_callback(test_mode: bool = False):
    def on_val_start(validator) -> None:
        """Wrap update_metrics so per-image stats can be paired with 
        im_file."""
        validator._per_image_stats = []  # list of (im_file, stats_entry)
        validator._test_mode = test_mode
        original_update = validator.update_metrics
        stats_dict = validator.metrics.stats

        def wrapped_update_metrics(preds, batch):
            before = {k: len(v) for k, v in stats_dict.items()}
            original_update(preds, batch)
            # Each call appends one entry per image in the batch
            n_new = len(stats_dict["tp"]) - before["tp"]
            for i in range(n_new):
                entry = {k: stats_dict[k][before[k] + i] for k in stats_dict}
                validator._per_image_stats.append((batch["im_file"][i], entry))

        validator.update_metrics = wrapped_update_metrics
    return on_val_start


def _to_key(s: str) -> str:
    """Normalise a metadata string value to a W&B-safe key component.

    e.g. 'Fresh (5-10 cm)' → 'fresh_5-10cm', 'Overcast' → 'overcast'
    """
    return (
        s.lower()
         .replace("(", "").replace(")", "")
         .replace(" ", "_")
         .replace("__", "_")
         .strip("_")
    )


def _log_categorical_buckets(
    per_image_stats: List[Tuple[str, Dict[str, Any]]],
    metadata: Dict[str, Dict[str, Any]],
    field: str,
    prefix: str,
) -> Dict[str, Union[float, int]]:
    """Group per-image stats by a categorical metadata field and compute 
    metrics."""
    bucketed: Dict[str, List[Dict[str, Any]]] = {}
    for im_file, entry in per_image_stats:
        stem = Path(im_file).stem
        meta = metadata.get(stem)
        val  = meta.get(field) if meta else None
        key  = _to_key(val) if val else "unknown"
        bucketed.setdefault(key, []).append(entry)

    log: Dict[str, Union[float, int]] = {}
    for bucket, entries in bucketed.items():
        n_imgs    = len(entries)
        n_targets = sum(int(e["target_cls"].size) for e in entries)
        log[f"{prefix}/{bucket}/n_images"]  = n_imgs
        log[f"{prefix}/{bucket}/n_targets"] = n_targets
        if n_targets == 0:
            continue
        m = _per_bucket_metrics(entries)
        if m is None:
            continue
        precision, recall, map50, map5095 = m
        log[f"{prefix}/{bucket}/precision"] = precision
        log[f"{prefix}/{bucket}/recall"]    = recall
        log[f"{prefix}/{bucket}/mAP50"]     = map50
        log[f"{prefix}/{bucket}/mAP50-95"]  = map5095
    return log


def _on_val_end(validator) -> None:
    """Group per-image stats by altitude, snow cover, and cloud cover;
    log to W&B."""
    global _val_count
    _val_count += 1

    if not getattr(validator, "_per_image_stats", None):
        return
    if wandb.run is None:
        return

    test_mode = getattr(validator, "_test_mode", False)
    prefix_alt   = "test_alt"   if test_mode else "val_alt"
    prefix_snow  = "test_snow"  if test_mode else "val_snow"
    prefix_cloud = "test_cloud" if test_mode else "val_cloud"

    metadata = _load_metadata()
    per_image_stats = validator._per_image_stats

    # ── altitude buckets ────────────────────────────────────────────────────
    bucketed = {b[0]: [] for b in ALTITUDE_BUCKETS}
    bucketed["unknown"] = []
    for im_file, entry in per_image_stats:
        stem = Path(im_file).stem
        meta = metadata.get(stem)
        bucket = _bucket_for(meta["altitude_m"]) if meta else None
        bucketed.setdefault(bucket or "unknown", []).append(entry)

    log: Dict[str, Union[float, int]] = {}
    for bucket, entries in bucketed.items():
        if not entries:
            continue
        n_imgs    = len(entries)
        n_targets = sum(int(e["target_cls"].size) for e in entries)
        log[f"{prefix_alt}/{bucket}/n_images"]  = n_imgs
        log[f"{prefix_alt}/{bucket}/n_targets"] = n_targets
        if n_targets == 0:
            continue
        m = _per_bucket_metrics(entries)
        if m is None:
            continue
        precision, recall, map50, map5095 = m
        log[f"{prefix_alt}/{bucket}/precision"] = precision
        log[f"{prefix_alt}/{bucket}/recall"]    = recall
        log[f"{prefix_alt}/{bucket}/mAP50"]     = map50
        log[f"{prefix_alt}/{bucket}/mAP50-95"]  = map5095

    # ── snow cover & cloud cover buckets ────────────────────────────────────
    log.update(_log_categorical_buckets(
        per_image_stats, metadata, "snow_cover", prefix_snow))
    log.update(_log_categorical_buckets(
        per_image_stats, metadata, "cloud_cover", prefix_cloud))

    if test_mode:
        wandb.run.summary.update(log)
    else:
        wandb.log(log, step=_val_count)


def _per_bucket_metrics(
        entries: List[Dict[str, Any]]
    ) -> Optional[Tuple[float, float, float, float]]:
    """Run ultralytics' ap_per_class on a subset of per-image stats.

    Returns (precision, recall, mAP50, mAP50-95) for the single 'car' class,
    or None if no predictions/targets after concatenation.
    """
    from ultralytics.utils.metrics import ap_per_class

    tp         = np.concatenate([e["tp"]         for e in entries], axis=0)
    conf       = np.concatenate([e["conf"]       for e in entries], axis=0)
    pred_cls   = np.concatenate([e["pred_cls"]   for e in entries], axis=0)
    target_cls = np.concatenate([e["target_cls"] for e in entries], axis=0)

    if tp.shape[0] == 0 or target_cls.shape[0] == 0:
        return None

    results = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
    # ap_per_class returns: tp, fp, p, r, f1, ap, unique_classes, ...
    p, r, _, ap = results[2], results[3], results[4], results[5]
    if len(p) == 0:
        return None
    return float(p[0]), float(r[0]), float(ap[0, 0]), float(ap[0].mean())


def register_metadata_callbacks(model, test_mode: bool = False) -> None:
    """Register the on_val_start / on_val_end callbacks on a YOLO model.

    test_mode: log to wandb.summary (one-shot) instead of wandb.log 
    (per-epoch).
    """
    model.add_callback("on_val_start", _make_val_start_callback(test_mode))
    model.add_callback("on_val_end",   _on_val_end)

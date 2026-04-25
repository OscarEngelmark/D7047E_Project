"""
Ultralytics training and validation callbacks for the YOLOv9-OBB project.

Three groups:

Training callbacks (registered via attach_callbacks in train.py):
  on_train_start       — saves the W&B run ID to disk for --resume support
  on_train_epoch_start — optionally unfreezes backbone layers at a given epoch

Metadata / validation callbacks (registered via register_metadata_callbacks):
  on_val_start — wraps validator.update_metrics to pair per-image stats with
                 filenames before ultralytics discards the mapping
  on_val_end   — groups stats by altitude bucket, snow cover, and cloud cover
                 and stores them in _last_bucket_metrics
  on_fit_epoch_end — single W&B log per epoch (training mode only)

Design note
-----------
Ultralytics' validator clears its per-image stats inside get_stats() before
the on_val_end callback fires, and never stores the image-file <-> stats
mapping anywhere reachable. To work around this, on_val_start wraps
validator.update_metrics so each batch's im_file list is paired with the
fresh entries appended to validator.metrics.stats and stashed on the
validator. on_val_end then groups and computes per-bucket metrics with
ultralytics' own ap_per_class; on_fit_epoch_end picks them up and logs
everything to W&B in a single call.
"""

from __future__ import annotations

import json
import wandb
import numpy as np
import globals as g

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ── module state (metadata callbacks) ────────────────────────────────────────

_metadata_cache: Optional[Dict[str, Dict[str, Any]]] = None
_last_bucket_metrics: Dict[str, float] = {}


# ── training callbacks ───────────────────────────────────────────────────────

def make_save_wandb_id_callback() -> Callable:
    def on_train_start(trainer) -> None:
        if wandb.run is None:
            return
        id_file = Path(trainer.save_dir) / "wandb_run_id.txt"
        id_file.parent.mkdir(parents=True, exist_ok=True)
        id_file.write_text(wandb.run.id)
        print(f"[wandb] Run ID saved to {id_file}")
    return on_train_start


def make_unfreeze_callback(
        unfreeze_epoch: int, lr_factor: float = 1.0
) -> Callable:
    def on_train_epoch_start(trainer) -> None:
        if trainer.epoch == unfreeze_epoch:
            for _, param in trainer.model.named_parameters():
                param.requires_grad = True
            print(
                f"[unfreeze] All layers unfrozen at epoch {unfreeze_epoch}"
            )
            if lr_factor != 1.0:
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] *= lr_factor
                    # keeps the scheduler scaling correctly
                    pg["initial_lr"] *= lr_factor
                new_lr = trainer.optimizer.param_groups[0]["lr"]
                print(
                    f"[unfreeze] LR scaled by {lr_factor} → {new_lr:.6f}"
                )
    return on_train_epoch_start


# ── metadata / validation callbacks ──────────────────────────────────────────

def _load_metadata() -> Dict[str, Dict[str, Any]]:
    global _metadata_cache
    if _metadata_cache is None:
        path = g.OUT_DIR / "metadata.json"
        with open(path) as f:
            _metadata_cache = json.load(f)
    assert _metadata_cache is not None
    return _metadata_cache


def _bucket_for(altitude: Optional[float]) -> Optional[str]:
    if altitude is None:
        return None
    for label, lo, hi in g.ALTITUDE_BUCKETS:
        if lo <= altitude < hi:
            return label
    return None


def _on_val_start(validator) -> None:
    validator._per_image_stats = []
    original_update = validator.update_metrics
    stats_dict = validator.metrics.stats

    def wrapped_update_metrics(preds: Any, batch: Any) -> None:
        before = {k: len(v) for k, v in stats_dict.items()}
        original_update(preds, batch)
        n_new = len(stats_dict["tp"]) - before["tp"]
        for i in range(n_new):
            entry = {k: stats_dict[k][before[k] + i] for k in stats_dict}
            validator._per_image_stats.append(
                (batch["im_file"][i], entry)
            )

    validator.update_metrics = wrapped_update_metrics


def _to_key(s: str) -> str:
    """Normalize a metadata string to a W&B-safe key component.

    e.g. 'Fresh (5-10 cm)' -> 'fresh_5-10cm', 'Overcast' -> 'overcast'
    """
    return (
        s.lower()
         .replace("(", "").replace(")", "")
         .replace(" ", "_")
         .replace("__", "_")
         .strip("_")
    )


def _per_bucket_metrics(
        entries: List[Dict[str, Any]]
) -> Optional[Tuple[float, float, float, float]]:
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


def _log_categorical_buckets(
    per_image_stats: List[Tuple[str, Dict[str, Any]]],
    metadata: Dict[str, Dict[str, Any]],
    field: str,
    prefix: str,
) -> Dict[str, Union[float, int]]:
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
    if not getattr(validator, "_per_image_stats", None):
        return

    metadata        = _load_metadata()
    per_image_stats = validator._per_image_stats

    # ── altitude buckets ─────────────────────────────────────────────────────
    bucketed = {b[0]: [] for b in g.ALTITUDE_BUCKETS}
    bucketed["unknown"] = []
    for im_file, entry in per_image_stats:
        stem   = Path(im_file).stem
        meta   = metadata.get(stem)
        bucket = _bucket_for(meta["altitude_m"]) if meta else None
        bucketed.setdefault(bucket or "unknown", []).append(entry)

    log: Dict[str, Union[float, int]] = {}
    for bucket, entries in bucketed.items():
        if not entries:
            continue
        n_imgs    = len(entries)
        n_targets = sum(int(e["target_cls"].size) for e in entries)
        log[f"val_alt/{bucket}/n_images"]  = n_imgs
        log[f"val_alt/{bucket}/n_targets"] = n_targets
        if n_targets == 0:
            continue
        m = _per_bucket_metrics(entries)
        if m is None:
            continue
        precision, recall, map50, map5095 = m
        log[f"val_alt/{bucket}/precision"] = precision
        log[f"val_alt/{bucket}/recall"]    = recall
        log[f"val_alt/{bucket}/mAP50"]     = map50
        log[f"val_alt/{bucket}/mAP50-95"]  = map5095

    # ── snow cover & cloud cover buckets ─────────────────────────────────────
    log.update(_log_categorical_buckets(
        per_image_stats, metadata, "snow_cover", "val_snow",
    ))
    log.update(_log_categorical_buckets(
        per_image_stats, metadata, "cloud_cover", "val_cloud",
    ))

    _last_bucket_metrics.clear()
    _last_bucket_metrics.update(log)


def _on_fit_epoch_end(trainer) -> None:
    """Single source of truth for per-epoch W&B logging.

    Replaces ultralytics' built-in W&B integration (disabled in train.py) so
    every metric goes through one wandb.log call with no explicit step. The
    `epoch` field drives the chart x-axis via define_metric in train.py, so
    resumed runs can never trigger step-monotonicity warnings.
    """
    if wandb.run is None:
        return

    log: Dict[str, Any] = {"epoch": trainer.epoch}
    if getattr(trainer, "tloss", None) is not None:
        log.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
    log.update(trainer.metrics)
    log.update(trainer.lr)
    log.update(_last_bucket_metrics)

    wandb.log(log)


# ── public API ───────────────────────────────────────────────────────────────

def get_last_bucket_metrics() -> Dict[str, float]:
    """Return the bucket metrics dict from the most recent validation pass."""
    return dict(_last_bucket_metrics)


def register_metadata_callbacks(model: Any, training: bool = True) -> None:
    """Register validation and per-epoch W&B logging callbacks on a YOLO model.

    training: also register on_fit_epoch_end for W&B logging (set False
    when running model.val() one-shot, e.g. in evaluate.py).
    """
    model.add_callback("on_val_start", _on_val_start)
    model.add_callback("on_val_end",   _on_val_end)
    if training:
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

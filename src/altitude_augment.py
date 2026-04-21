"""
Altitude-aware scale augmentation for YOLO OBB training.

For each training frame at altitude h, a target altitude
h_target ~ U(alt_min, alt_max) is sampled and scale s = h / h_target
is applied.  This produces a flat apparent-altitude distribution over
[alt_min, alt_max] (up to clamping at SCALE_FLOOR / SCALE_CEILING).

Public API
----------
AltitudeAwareOBBTrainer   pass to YOLO.train(trainer=...)
compute_scale_bounds       utility exposed for testing / plotting

Notes
-----
Altitude is injected into the labels dict by AltitudeAwareYOLODataset
and read by AltitudeAwareRandomPerspective.  When mosaic=0 the labels dict 
passes through Mosaic unchanged, so altitude_m is still present when the affine 
transform fires. If mosaic > 0 the altitude is absent after tiling and the 
transform falls back to symmetric scale (1±hyp.scale).
"""

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

import torch.nn as nn
from typing import cast

from ultralytics.data.augment import Compose, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import DEFAULT_CFG, colorstr
from ultralytics.utils.torch_utils import unwrap_model

import globals as g

SCALE_FLOOR = 0.1
SCALE_CEILING = 4.0


def compute_scale_bounds(
    altitude_m: float,
    alt_min: float,
    alt_max: float,
) -> tuple[float, float]:
    """Return (s_lo, s_hi) for altitude-aware scale augmentation.

    s = h / h_target; targeting h_max gives s_lo, h_min gives s_hi.
    Both values are clamped to [SCALE_FLOOR, SCALE_CEILING].
    """
    s_lo = float(np.clip(altitude_m / alt_max, SCALE_FLOOR, SCALE_CEILING))
    s_hi = float(np.clip(altitude_m / alt_min, SCALE_FLOOR, SCALE_CEILING))
    return s_lo, s_hi


class AltitudeAwareRandomPerspective(RandomPerspective):
    """RandomPerspective that samples scale from altitude-dependent bounds.

    Reads altitude_m from the labels dict (injected by
    AltitudeAwareYOLODataset) and samples scale from
    [h/alt_max, h/alt_min] instead of [1-scale, 1+scale].
    Falls back to symmetric bounds when altitude_m is absent.
    """

    def __init__(
        self,
        alt_min: float = 100.0,
        alt_max: float = 300.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alt_min = alt_min
        self.alt_max = alt_max
        self._altitude_m: float | None = None

    def affine_transform(
        self,
        img: np.ndarray,
        border: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Identical to RandomPerspective.affine_transform except uses
        self._scale_lo / self._scale_hi for the scale sample."""
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2
        C[1, 2] = -img.shape[0] / 2

        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)

        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        if self._altitude_m is not None:
            h_target = random.uniform(self.alt_min, self.alt_max)
            s = float(np.clip(
                self._altitude_m / h_target, SCALE_FLOOR, SCALE_CEILING
            ))
        else:
            s = random.uniform(1.0 - self.scale, 1.0 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )
        S[1, 0] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )

        T = np.eye(3, dtype=np.float32)
        T[0, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate)
            * self.size[0]
        )
        T[1, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate)
            * self.size[1]
        )

        M = T @ S @ R @ P @ C

        if (
            (border[0] != 0)
            or (border[1] != 0)
            or (M != np.eye(3)).any()
        ):
            if self.perspective:
                img = cv2.warpPerspective(
                    img, M,
                    dsize=self.size,
                    borderValue=(114, 114, 114),
                )
            else:
                img = cv2.warpAffine(
                    img, M[:2],
                    dsize=self.size,
                    borderValue=(114, 114, 114),
                )
            if img.ndim == 2:
                img = img[..., None]
        return img, M, s

    def __call__(self, labels: dict) -> dict:
        altitude_m = labels.get("altitude_m")
        self._altitude_m = (
            float(altitude_m) if altitude_m is not None else None
        )
        return super().__call__(labels)


def _swap_affine(
    transforms: Compose,
    alt_min: float,
    alt_max: float,
) -> None:
    """In-place: replace RandomPerspective with AltitudeAwareRandomPerspective
    everywhere inside a Compose tree."""
    for i, t in enumerate(transforms.transforms):
        if isinstance(t, Compose):
            _swap_affine(t, alt_min, alt_max)
        elif type(t) is RandomPerspective:
            transforms.transforms[i] = AltitudeAwareRandomPerspective(
                alt_min=alt_min,
                alt_max=alt_max,
                degrees=t.degrees,
                translate=t.translate,
                scale=t.scale,
                shear=t.shear,
                perspective=t.perspective,
                border=t.border,
                pre_transform=t.pre_transform,
            )


class AltitudeAwareYOLODataset(YOLODataset):
    """YOLODataset that injects per-frame altitude into labels and uses
    AltitudeAwareRandomPerspective for training augmentation."""

    def __init__(
        self,
        *args,
        alt_min: float = 100.0,
        alt_max: float = 300.0,
        metadata_path: Path = g.OUT_DIR / "metadata.json",
        **kwargs,
    ) -> None:
        self.alt_min = alt_min
        self.alt_max = alt_max
        with open(metadata_path) as f:
            raw: dict = json.load(f)
        self._stem_to_alt: dict[str, float] = {
            stem: float(v["altitude_m"])
            for stem, v in raw.items()
            if v.get("altitude_m") is not None
        }
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index: int) -> dict:
        label = super().get_image_and_label(index)
        stem = Path(label["im_file"]).stem
        alt = self._stem_to_alt.get(stem)
        if alt is not None:
            label["altitude_m"] = alt
        return label

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            _swap_affine(transforms, self.alt_min, self.alt_max)
        return transforms


class AltitudeAwareOBBTrainer(OBBTrainer):
    """OBBTrainer that uses AltitudeAwareYOLODataset for the training split.

    alt_min and alt_max are extracted from the overrides dict (pass them
    as keyword arguments to YOLO.train).
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict | None = None,
        _callbacks: dict | None = None,
    ) -> None:
        overrides = dict(overrides or {})
        self.alt_min = float(overrides.pop("alt_min", 100.0))
        self.alt_max = float(overrides.pop("alt_max", 300.0))
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
    ):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        stride = unwrap_model(cast(nn.Module, self.model)).stride
        gs = max(int(stride.max()), 32)  # type: ignore[operator]
        return AltitudeAwareYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
            alt_min=self.alt_min,
            alt_max=self.alt_max,
        )

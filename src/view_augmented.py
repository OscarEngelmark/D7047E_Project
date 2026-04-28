"""Visualize training images with AAS (altitude-aware scale) augmentation.

Usage:
    cd src && python view_augmented.py
    cd src && python view_augmented.py --augment aas1 --source Asjo --max 50
    cd src && python view_augmented.py --augment paper \
        --alt-min 80 --alt-max 320

Controls: any key -> next | r -> re-augment same image | s -> save | q -> quit
"""

import argparse
import json
import sys
from pathlib import Path
from typing import cast, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

import globals as g
from altitude_augment import AltitudeAwareRandomPerspective


class InstrumentedPerspective(AltitudeAwareRandomPerspective):
    """AltitudeAwareRandomPerspective that records scale and apparent 
    altitude."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_scale: Optional[float] = None
        self.last_h_target: Optional[float] = None

    def affine_transform(
        self,
        img: np.ndarray,
        border: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        img_out, M, s = super().affine_transform(img, border)
        self.last_scale = s
        self.last_h_target = (
            self._altitude_m / s
            if self._altitude_m is not None and s > 0
            else None
        )
        return img_out, M, s


def load_metadata(path: Path) -> Dict[str, float]:
    if not path.is_file():
        return {}
    with open(path) as f:
        raw: dict = json.load(f)
    return {
        stem: float(v["altitude_m"])
        for stem, v in raw.items()
        if v.get("altitude_m") is not None
    }


def load_obb_corners(
    label_path: Path, w: int, h: int
) -> np.ndarray:
    """Return (N, 4, 2) pixel-space corners from an OBB label file."""
    if not label_path.is_file():
        return np.zeros((0, 4, 2), dtype=np.float32)
    rows: List[List[Tuple[float, float]]] = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            coords = list(map(float, parts[1:]))
            pts = [
                (coords[i] * w, coords[i + 1] * h) for i in range(0, 8, 2)
            ]
            rows.append(pts)  # type: ignore[arg-type]
    if not rows:
        return np.zeros((0, 4, 2), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def apply_augment(
    raw: np.ndarray,
    corners: np.ndarray,
    altitude_m: Optional[float],
    transform: InstrumentedPerspective,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    """Apply AAS affine transform to image and corners.

    Calls affine_transform directly so we get the matrix M without
    running letterbox pre-transforms or the full labels pipeline.
    """
    img = raw.copy()
    transform.size = (img.shape[1], img.shape[0])
    transform._altitude_m = (
        float(altitude_m) if altitude_m is not None else None
    )
    aug_img, M, _s = transform.affine_transform(img, border=(0, 0))

    aug_corners = corners.copy()
    if corners.shape[0] > 0:
        pts = corners.reshape(-1, 1, 2).astype(np.float32)
        aug_corners = cv2.perspectiveTransform(pts, M).reshape(corners.shape)

    return aug_img, aug_corners, transform.last_scale, transform.last_h_target


def draw_corners(
    img: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    for box in corners.astype(np.int32):
        cv2.polylines(
            img, [box], isClosed=True, color=color, thickness=thickness
        )


def overlay_info(
    img: np.ndarray,
    stem: str,
    altitude_m: Optional[float],
    scale: Optional[float],
    h_target: Optional[float],
) -> None:
    lines = [
        stem,
        f"Raw alt: {altitude_m:.0f} m" if altitude_m is not None
            else "Raw alt:  unknown",
        f"Scale:   {scale:.3f}x" if scale     is not None else "Scale:    —",
        f"App alt: {h_target:.0f} m" if h_target is not None
            else "App alt:  —",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    thick = 1
    pad = 6
    for i, text in enumerate(lines):
        (_tw, th), _bl = cv2.getTextSize(text, font, fscale, thick)
        y = pad + (th + pad) * (i + 1)
        cv2.putText(
            img, text, (pad + 1, y + 1),
            font, fscale, (0, 0, 0), thick + 1, cv2.LINE_AA,
        )
        cv2.putText(
            img, text, (pad, y),
            font, fscale, (255, 255, 255), thick, cv2.LINE_AA,
        )


def main(opt: argparse.Namespace) -> None:
    img_dir = g.IMG_DIR / opt.split
    lbl_dir = g.LBL_DIR / opt.split

    if not img_dir.is_dir():
        sys.exit(
            f"Images not found at {img_dir} — run preprocessing.py first."
        )

    metadata = load_metadata(g.OUT_DIR / "metadata.json")

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if opt.source:
        images = [p for p in images if opt.source.lower() in p.name.lower()]
    if not images:
        sys.exit("No matching images found.")
    if opt.max:
        images = images[: opt.max]

    aug_cfg: Dict = {}
    if opt.augment:
        yaml_path = g.AUGS_DIR / f"{opt.augment}.yaml"
        if not yaml_path.is_file():
            sys.exit(f"Augmentation preset not found: {yaml_path}")
        with open(yaml_path) as f:
            aug_cfg = yaml.safe_load(f) or {}

    transform = InstrumentedPerspective(
        alt_min=opt.alt_min,
        alt_max=opt.alt_max,
        alt_mode=opt.alt_mode,
        degrees=float(aug_cfg.get("degrees", 10.0)),
        translate=float(aug_cfg.get("translate", 0.1)),
        scale=float(aug_cfg.get("scale", 0.5)),
        shear=float(aug_cfg.get("shear", 0.0)),
        perspective=0.0,
        pre_transform=None,
    )

    aug_label = opt.augment if opt.augment else "defaults"
    print(
        f"Showing {len(images)} images from '{opt.split}' split "
        f"| augment={aug_label}"
        f" alt_min={opt.alt_min} alt_max={opt.alt_max}\n"
        "Any key -> next  |  r -> re-augment  |  s -> save  |  q -> quit"
    )

    cv2.namedWindow("view_augmented", cv2.WINDOW_NORMAL)

    idx = 0
    quit_requested = False
    while idx < len(images) and not quit_requested:
        img_path = images[idx]
        _raw = cv2.imread(str(img_path))
        if _raw is None:
            idx += 1
            continue
        raw = cast(np.ndarray, _raw)

        h, w = raw.shape[:2]
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        corners = load_obb_corners(lbl_path, w, h)
        altitude_m = metadata.get(img_path.stem)

        def render() -> np.ndarray:
            aug, aug_c, s, ht = apply_augment(
                raw, corners, altitude_m, transform
            )
            display = aug
            draw_corners(display, aug_c)
            overlay_info(display, img_path.stem, altitude_m, s, ht)
            if opt.max_dim and max(display.shape[:2]) > opt.max_dim:
                sc = opt.max_dim / max(display.shape[:2])
                display = cv2.resize(
                    display,
                    (int(display.shape[1] * sc), int(display.shape[0] * sc)),
                )
            return display

        frame = render()
        cv2.imshow("view_augmented", frame)
        cv2.resizeWindow("view_augmented", frame.shape[1], frame.shape[0])

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                quit_requested = True
                break
            elif key == ord("r"):
                frame = render()
                cv2.imshow("view_augmented", frame)
                cv2.resizeWindow(
                    "view_augmented", frame.shape[1], frame.shape[0]
                )
            elif key == ord("s"):
                save_dir = g.RESULTS_DIR / "viz_augmented"
                save_dir.mkdir(parents=True, exist_ok=True)
                out_path = save_dir / f"{img_path.stem}_aas.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"Saved {out_path}")
            else:
                idx += 1
                break

    cv2.destroyAllWindows()


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize training images with AAS augmentation."
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="Dataset split to visualize (default: train)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Filter by filename substring (e.g. 'Asjo')",
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Maximum number of images to show",
    )
    parser.add_argument(
        "--max-dim", type=int, default=None, dest="max_dim",
        help="Resize display so longest side is at most this many pixels",
    )
    parser.add_argument(
        "--alt-min", type=float, default=100.0, dest="alt_min",
        help="AAS minimum target altitude in metres (default: 100)",
    )
    parser.add_argument(
        "--alt-max", type=float, default=300.0, dest="alt_max",
        help="AAS maximum target altitude in metres (default: 300)",
    )
    parser.add_argument(
        "--alt-mode", type=float, default=None, dest="alt_mode",
        help="Triangular distribution mode in metres (omit for uniform)",
    )
    parser.add_argument(
        "--augment", type=str, default=None,
        help="Augmentation preset stem from augmentations/ dir "
             "(e.g. 'paper', 'aas1'). Reads degrees/translate/scale/shear "
             "from the YAML; omit to use built-in defaults.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_opt())

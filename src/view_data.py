"""Visualize processed frames with OBB labels.

Usage:
    cd src && python view_data.py
    cd src && python view_data.py --split val --source Nyland
    cd src && python view_data.py --split train --max 200

Press any key to advance, 's' to save, 'q' to quit.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import globals as g


def draw_obb(
        img: np.ndarray, label_path: Path, color=(0, 255, 0), thickness=2
    ) -> None:
    h, w = img.shape[:2]
    if not label_path.is_file():
        return
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            coords = list(map(float, parts[1:]))
            pts = np.array(
                [(coords[i] * w, coords[i + 1] * h) for i in range(0, 8, 2)],
                dtype=np.int32,
            )
            cv2.polylines(
                img, [pts], isClosed=True, color=color, thickness=thickness
            )


def main(opt: argparse.Namespace) -> None:
    img_dir = g.IMG_DIR / opt.split
    lbl_dir = g.LBL_DIR / opt.split

    if not img_dir.is_dir():
        sys.exit(f"No images found at {img_dir} — run preprocessing.py first.")

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    if opt.source:
        images = [p for p in images if opt.source.lower() in p.name.lower()]

    if not images:
        sys.exit("No matching images found.")

    if opt.max:
        images = images[: opt.max]

    print(f"Showing {len(images)} images from '{opt.split}' split."
           "Press any key to advance, 'q' to quit.")

    quit_requested = False
    for img_path in images:
        if quit_requested:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        draw_obb(img, lbl_path)

        dh, dw = img.shape[:2]
        if opt.max_dim and max(dh, dw) > opt.max_dim:
            scale = opt.max_dim / max(dh, dw)
            img = cv2.resize(img, (int(dw * scale), int(dh * scale)))

        cv2.namedWindow("view_data", cv2.WINDOW_NORMAL)
        cv2.imshow("view_data", img)
        cv2.resizeWindow("view_data", img.shape[1], img.shape[0])
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):
                save_dir = g.RESULTS_DIR / "viz"
                save_dir.mkdir(parents=True, exist_ok=True)
                out_path = save_dir / img_path.name
                cv2.imwrite(str(out_path), img)
                print(f"Saved {out_path}")
            elif key == ord("q"):
                quit_requested = True
                break
            else:
                break

    cv2.destroyAllWindows()


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize OBB-labelled frames."
    )
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="Dataset split to visualize (default: val)"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Filter by source filename substring (e.g. 'Nyland')"
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Maximum number of images to show"
    )
    parser.add_argument(
        "--max-dim", type=int, default=None,
        dest="max_dim",
        help="Resize images so the longest side is at most this many pixels "
             "(default: no resize)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_opt())

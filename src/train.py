"""
Baseline YOLOv9c-OBB training script.

Builds the model from src/configs/yolov9c-obb.yaml and transfers backbone
weights from the pretrained yolov9c.pt COCO checkpoint (downloaded
automatically by ultralytics on first use).
"""

import yaml
from ultralytics import YOLO
from globals import SEED, DEVICE, OUT_DIR, SRC_DIR, PROJECT_DIR

# ── config ────────────────────────────────────────────────────────────────────

EPOCHS    = 100
IMGSZ     = 640
BATCH     = 16
MODEL_CFG = SRC_DIR / "configs" / "yolov9c-obb.yaml"
RUNS_DIR  = PROJECT_DIR / "runs"


# ── helpers ───────────────────────────────────────────────────────────────────

def write_dataset_yaml() -> str:
    """Regenerate dataset.yaml with the correct absolute path for this machine."""
    cfg = {
        "path":  str(OUT_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": {0: "car"},
    }
    path = OUT_DIR / "dataset.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return str(path)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device: {DEVICE}")

    dataset_yaml = write_dataset_yaml()
    print(f"Dataset: {dataset_yaml}")

    # Build model from custom OBB config, transfer pretrained backbone weights.
    # yolov9c.pt is downloaded automatically on first use.
    model = YOLO(str(MODEL_CFG)).load("yolov9c.pt")

    model.train(
        data=dataset_yaml,
        task="obb",
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        seed=SEED,
        project=str(RUNS_DIR),
        name="yolov9c-obb-baseline",
    )


if __name__ == "__main__":
    main()

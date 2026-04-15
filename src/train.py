"""
Baseline YOLOv9c-OBB training script.

Builds the model from src/configs/yolov9c-obb.yaml and transfers backbone
weights from the pretrained yolov9c.pt COCO checkpoint (downloaded
automatically by ultralytics on first use).
"""

import yaml
import wandb
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
from globals import SEED, DEVICE, OUT_DIR, SRC_DIR, PROJECT_DIR, MODELS_DIR, WANDB_ENTITY, WANDB_PROJECT

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

    LOG_WANDB = True
    RUN_NAME = "yolov9c-obb-baseline"

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=RUN_NAME,
        config={
            "model":   "yolov9c-obb",
            "epochs":  EPOCHS,
            "imgsz":   IMGSZ,
            "batch":   BATCH,
            "device":  DEVICE,
            "seed":    SEED,
        },
        mode="online" if LOG_WANDB else "disabled",
    )

    # Build model from custom OBB config, transfer pretrained backbone weights.
    # Downloads yolov9c.pt to models/ on first use.
    weights = MODELS_DIR / "yolov9c.pt"
    if not weights.exists():
        attempt_download_asset(str(weights))
    model = YOLO(str(MODEL_CFG)).load(str(weights))

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

    wandb.finish()


if __name__ == "__main__":
    main()

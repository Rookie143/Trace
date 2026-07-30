"""Train an OGA, ODA, or RMA YOLOv5 checkpoint.

Training settings are kept here so users can change them directly.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from trace_detector.training import train_yolov5

ROOT = Path(__file__).resolve().parent

# ------------------------------ user settings ------------------------------
YOLOV5_ROOT = ROOT / "third_party" / "yolov5"
INITIAL_WEIGHTS = "yolov5s.pt"
EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SIZE = 640
WORKERS = 8
OUTPUT_ROOT = ROOT / "runs" / "train"
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a backdoored YOLOv5 model.")
    parser.add_argument("--attack", required=True, choices=("oga", "oda", "rma"))
    parser.add_argument("--device", default="", help="CUDA device, for example 0; empty = automatic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_yaml = ROOT / "data" / args.attack / f"{args.attack}.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"dataset not found: run poison.py first ({data_yaml})")

    command = train_yolov5(
        yolo_root=YOLOV5_ROOT,
        data_yaml=data_yaml,
        weights=INITIAL_WEIGHTS,
        output=OUTPUT_ROOT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        device=args.device,
        workers=WORKERS,
        name=args.attack,
        trust_checkpoint=True,
    )
    print(shlex.join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

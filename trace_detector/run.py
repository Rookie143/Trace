from __future__ import annotations

import argparse
import os
from pathlib import Path

from .cli import ATTACKS, _run_detect


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        prog="trace",
        description="Detect an OGA, ODA, or RMA backdoor with the complete TRACE score.",
    )
    parser.add_argument("--attack", required=True, choices=ATTACKS)
    parser.add_argument("source", nargs="?", type=Path, help="image, directory, or image-list file")
    parser.add_argument("--manifest", type=Path, help="labeled CSV for F1/AUROC evaluation")
    parser.add_argument(
        "--yolo-root",
        type=Path,
        default=Path(
            os.environ.get("YOLOV5_ROOT", root / "third_party" / "yolov5")
        ),
        help="YOLOv5 checkout (default: $YOLOV5_ROOT or third_party/yolov5)",
    )
    parser.add_argument("--weights", type=Path, help="override checkpoints/<attack>.pt")
    parser.add_argument("--config", type=Path, help="override configs/<attack>.yaml")
    parser.add_argument("--output", type=Path, help="default: runs/trace/<attack>")
    parser.add_argument("--device", default="", help="CUDA device such as 0, or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 inference")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--version", action="version", version="trace 1.0.0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.source) == bool(args.manifest):
        raise SystemExit("provide exactly one SOURCE or --manifest")

    root = repository_root()
    args.command = "detect"
    args.weights = args.weights or root / "checkpoints" / f"{args.attack}.pt"
    args.config = args.config or root / "configs" / f"{args.attack}.yaml"
    args.output = args.output or root / "runs" / "trace" / args.attack
    args.components = "full"
    args.image_size = 640
    args.confidence = 0.25
    args.nms_iou = 0.45
    args.trust_checkpoint = True

    for path, label in (
        (args.weights, "checkpoint"),
        (args.config, "TRACE config"),
        (args.yolo_root, "YOLOv5 checkout"),
        (args.manifest or args.source, "input"),
    ):
        if path is None or not path.exists():
            raise SystemExit(f"{label} not found: {path}")
    return _run_detect(args)


if __name__ == "__main__":
    raise SystemExit(main())

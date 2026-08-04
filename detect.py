"""Run TRACE backdoor detection.

The settings below are intentionally kept in this top-level file so they are
easy to inspect and edit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_detector.config import IMAGE_SUFFIXES, TraceConfig
from trace_detector.evaluation import (
    metrics,
    write_roc,
    write_scores,
)
from trace_detector.trace import TraceDetector
from trace_detector.utils import image_paths, write_json
from trace_detector.validation import sha256_file
from trace_detector.yolov5 import YoloV5Detector

ROOT = Path(__file__).resolve().parent

# ------------------------------ user settings ------------------------------
YOLOV5_ROOT = ROOT / "third_party" / "yolov5"
OUTPUT_ROOT = ROOT / "runs" / "trace"

IMAGE_SIZE = 640
NMS_IOU = 0.45
HALF_PRECISION = False

BACKGROUND_QUERIES = 30
FOREGROUND_QUERIES = 50
NBOS_PER_QUERY = 5
NBO_SCALE = 0.30
BATCH_SIZE = 16
SEED = 0

# The TRACE score definition and CTC/FTC weights are identical for all attacks.
CTC_STATISTIC = "mean_abs_delta"
FTC_STATISTIC = "variance"
CTC_WEIGHT = 1.0
FTC_WEIGHT = 1.0

# Only transformation settings and the reported maximum-F1 threshold differ.
ATTACK_SETTINGS = {
    "oga": {
        "confidence": 0.50,
        "background_opacity": 0.20,
        "ssim_threshold": 0.25,
        "threshold": -0.0028012301884611235,
    },
    "oda": {
        "confidence": 0.25,
        "background_opacity": 0.15,
        "ssim_threshold": 0.08,
        "threshold": 0.0023446551832737583,
    },
    "rma": {
        "confidence": 0.25,
        "background_opacity": 0.15,
        "ssim_threshold": 0.08,
        "threshold": 0.001321548501396208,
    },
}
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACE on an image folder.")
    parser.add_argument("--attack", required=True, choices=tuple(ATTACK_SETTINGS))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--device", default="", help="CUDA device, for example 0; empty = automatic")
    return parser.parse_args()


def filename_label(path: Path) -> int | None:
    """Recognize paired validation images created by ``poison.py --paired``."""
    if path.name.lower().startswith("clean_"):
        return 0
    if path.name.lower().startswith("poison_"):
        return 1
    return None


def make_trace_config(attack: str) -> TraceConfig:
    settings = ATTACK_SETTINGS[attack]
    background_dir = ROOT / "assets" / "backgrounds"
    backgrounds = sorted(
        str(path)
        for path in background_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    config = TraceConfig(
        backgrounds=backgrounds,
        nbo=str(ROOT / "assets" / "nbo" / "stop_sign.png"),
        nbo_class_id=11,
        references=str(ROOT / "assets" / "references"),
        background_queries=BACKGROUND_QUERIES,
        foreground_queries=FOREGROUND_QUERIES,
        points_per_query=NBOS_PER_QUERY,
        background_opacity=settings["background_opacity"],
        ctc_statistic=CTC_STATISTIC,
        ftc_statistic=FTC_STATISTIC,
        nbo_scale=NBO_SCALE,
        ssim_threshold=settings["ssim_threshold"],
        match_iou=0.50,
        ctc_scale=CTC_WEIGHT,
        ftc_scale=FTC_WEIGHT,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    config.validate()
    return config


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    weights = ROOT / "checkpoints" / f"{args.attack}.pt"
    output = OUTPUT_ROOT / args.attack

    for path, description in (
        (source, "image source"),
        (weights, "checkpoint"),
        (YOLOV5_ROOT, "YOLOv5 checkout"),
    ):
        if not path.exists():
            raise SystemExit(f"{description} not found: {path}")

    config = make_trace_config(args.attack)
    model = YoloV5Detector(
        weights=weights,
        yolo_root=YOLOV5_ROOT,
        device=args.device,
        image_size=IMAGE_SIZE,
        confidence=ATTACK_SETTINGS[args.attack]["confidence"],
        nms_iou=NMS_IOU,
        half=HALF_PRECISION,
        trust_checkpoint=True,
    )
    detector = TraceDetector(model, config)
    threshold = ATTACK_SETTINGS[args.attack]["threshold"]

    rows = []
    paths = image_paths(source)
    for index, path in enumerate(paths, 1):
        row = detector.score(path, filename_label(path))
        rows.append(row)
        decision = (
            f"{'BACKDOOR' if row.score >= threshold else 'CLEAN'} "
            if row.label is None
            else ""
        )
        print(
            f"[{index}/{len(paths)}] {path.name}: {decision}"
            f"(score={row.score:.6f}, CTC={row.ctc:.6f}, FTC={row.ftc:.6f})"
        )
        write_scores(output / "scores.csv", rows)

    if not rows:
        raise SystemExit(f"no images found under {source}")

    labels = {row.label for row in rows}
    if labels == {0, 1}:
        result = metrics(rows)  # report the maximum-F1 threshold on this set
        write_json(output / "metrics.json", result)
        write_roc(output / "roc.csv", rows)
        print(json.dumps(result, indent=2))
    else:
        detections = sum(row.score >= threshold for row in rows)
        print(f"Detected {detections}/{len(rows)} possible backdoor inputs.")

    write_json(
        output / "run.json",
        {
            "attack": args.attack,
            "source": str(source),
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "detector_confidence": ATTACK_SETTINGS[args.attack]["confidence"],
            "threshold": threshold,
            "trace": vars(config),
            "images": len(rows),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

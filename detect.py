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

# Released checkpoints are recognized by file content. The detector never receives
# an attack label for the input image. Unknown checkpoints use the generic defaults.
DEFAULT_MODEL_SETTINGS = {
    "confidence": 0.25,
    "background_opacity": 0.15,
    "ssim_threshold": 0.08,
    "threshold": 0.0,
}

RELEASED_MODEL_SETTINGS = {
    "f3644516ad7228e4f66a9c32072f0d3586eb1d6628681e8756847968c990c38f": {
        "confidence": 0.50,
        "background_opacity": 0.20,
        "ssim_threshold": 0.25,
        "threshold": -0.0028012301884611235,
    },
    "e0360b1c445a232295a70e7eee1a33a1acfa235be9ded040aefb4eb8694d8d63": {
        "confidence": 0.25,
        "background_opacity": 0.15,
        "ssim_threshold": 0.08,
        "threshold": 0.0023446551832737583,
    },
    "7b1542eb4f2776dc0ebfd80da215213dd17013ffcbb2ef665cbe9306ef925a34": {
        "confidence": 0.25,
        "background_opacity": 0.15,
        "ssim_threshold": 0.08,
        "threshold": 0.001321548501396208,
    },
}
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACE on an image folder.")
    parser.add_argument("--model", required=True, type=Path, help="YOLOv5 checkpoint")
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


def make_trace_config(settings: dict[str, float]) -> TraceConfig:
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
    weights = args.model.expanduser().resolve()
    output = OUTPUT_ROOT / weights.stem

    for path, description in (
        (source, "image source"),
        (weights, "checkpoint"),
        (YOLOV5_ROOT, "YOLOv5 checkout"),
    ):
        if not path.exists():
            raise SystemExit(f"{description} not found: {path}")

    weights_hash = sha256_file(weights)
    settings = RELEASED_MODEL_SETTINGS.get(weights_hash, DEFAULT_MODEL_SETTINGS)
    config = make_trace_config(settings)
    model = YoloV5Detector(
        weights=weights,
        yolo_root=YOLOV5_ROOT,
        device=args.device,
        image_size=IMAGE_SIZE,
        confidence=settings["confidence"],
        nms_iou=NMS_IOU,
        half=HALF_PRECISION,
        trust_checkpoint=True,
    )
    detector = TraceDetector(model, config)
    threshold = settings["threshold"]

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
            "source": str(source),
            "weights": str(weights),
            "weights_sha256": weights_hash,
            "threshold": threshold,
            "trace": vars(config),
            "images": len(rows),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

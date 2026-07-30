from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from .types import Detection, Detector
from .utils import write_json


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_summary(detector: Detector, weights: Path) -> dict[str, object]:
    names = detector.names
    return {
        "weights": str(weights.expanduser().resolve()),
        "sha256": sha256_file(weights),
        "class_count": len(names),
        "classes": names,
        "image_size": getattr(detector, "image_size", None),
        "stride": getattr(detector, "stride", None),
        "backend": getattr(detector, "backend_name", type(detector).__name__),
    }


def yolov5_val_command(
    yolo_root: Path,
    data: Path,
    weights: Path,
    output: Path,
    image_size: int,
    batch_size: int,
    device: str,
    workers: int,
    trust_checkpoint: bool = False,
) -> list[str]:
    script = yolo_root.expanduser().resolve() / "val.py"
    if not script.exists():
        raise FileNotFoundError(f"YOLOv5 val.py not found: {script}")
    command = [sys.executable]
    if trust_checkpoint:
        command.extend([str(Path(__file__).with_name("validation_runner.py")), str(script)])
    else:
        command.append(str(script))
    command.extend(
        [
            "--data",
            str(data.expanduser().resolve()),
            "--weights",
            str(weights.expanduser().resolve()),
            "--imgsz",
            str(image_size),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "--workers",
            str(workers),
            "--project",
            str(output.expanduser().resolve()),
            "--name",
            "clean-map",
            "--exist-ok",
        ]
    )
    return command


def run_clean_map(
    yolo_root: Path,
    data: Path,
    weights: Path,
    output: Path,
    image_size: int = 640,
    batch_size: int = 16,
    device: str = "",
    workers: int = 8,
    dry_run: bool = False,
    trust_checkpoint: bool = False,
) -> dict[str, object]:
    command = yolov5_val_command(
        yolo_root,
        data,
        weights,
        output,
        image_size,
        batch_size,
        device,
        workers,
        trust_checkpoint,
    )
    result: dict[str, object] = {"command": command, "status": "dry-run" if dry_run else "complete"}
    if dry_run:
        return result
    completed = subprocess.run(
        command,
        cwd=yolo_root.expanduser().resolve(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "clean-map.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(
            f"YOLOv5 validation failed with exit code {completed.returncode}; "
            f"log: {log_path.resolve()}\n{tail}"
        )
    nms_timeouts = completed.stdout.count("NMS time limit")
    if nms_timeouts:
        raise RuntimeError(
            f"YOLOv5 validation dropped predictions after {nms_timeouts} NMS timeouts; "
            f"result is invalid; log: {log_path.resolve()}"
        )
    result["log"] = str(log_path.resolve())
    result["metrics"] = _parse_yolov5_summary(completed.stdout)
    return result


def _parse_yolov5_summary(log: str) -> dict[str, float]:
    """Parse the final YOLOv5 `all` row without depending on a specific release."""
    for line in reversed(log.splitlines()):
        fields = re.sub(r"\x1b\[[0-9;]*m", "", line).split()
        if "all" not in fields:
            continue
        position = fields.index("all")
        numbers: list[float] = []
        for field in fields[position + 1 :]:
            try:
                numbers.append(float(field))
            except ValueError:
                continue
        # images, instances, precision, recall, mAP50, mAP50-95
        if len(numbers) >= 6:
            return {
                "images": int(numbers[0]),
                "instances": int(numbers[1]),
                "precision": numbers[2],
                "recall": numbers[3],
                "map50": numbers[4],
                "map50_95": numbers[5],
            }
    raise RuntimeError("could not parse the final YOLOv5 validation summary")


def _iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _matched(
    detections: Iterable[Detection],
    box: tuple[float, float, float, float],
    iou: float,
    class_id: int | None = None,
) -> bool:
    return any(
        (class_id is None or item.class_id == class_id) and _iou(item.xyxy, box) >= iou
        for item in detections
    )


def attack_succeeded(
    attack: str,
    detections: list[Detection],
    trigger_box: tuple[float, float, float, float] | None,
    victim_boxes: list[tuple[float, float, float, float]],
    target_class: int,
    match_iou: float,
) -> bool:
    if attack == "oga":
        return trigger_box is not None and any(
            detection.class_id == target_class
            and _trigger_in_detection(trigger_box, detection)
            for detection in detections
        )
    if attack == "rma":
        return bool(victim_boxes) and _matched(detections, victim_boxes[0], match_iou, target_class)
    if attack == "oda":
        return bool(victim_boxes) and all(
            not _matched(detections, box, match_iou) for box in victim_boxes
        )
    raise ValueError(f"unsupported attack: {attack}")


def attack_outcomes(
    attack: str,
    detections: list[Detection],
    trigger_box: tuple[float, float, float, float] | None,
    victim_boxes: list[tuple[float, float, float, float]],
    target_class: int,
    match_iou: float,
) -> list[bool]:
    """Return one outcome per attacked object (one trigger event for OGA)."""
    if attack == "oga":
        return [
            trigger_box is not None
            and any(
                detection.class_id == target_class
                and _trigger_in_detection(trigger_box, detection)
                for detection in detections
            )
        ]
    if attack == "rma":
        return [_matched(detections, box, match_iou, target_class) for box in victim_boxes[:1]]
    if attack == "oda":
        return [not _matched(detections, box, match_iou) for box in victim_boxes]
    raise ValueError(f"unsupported attack: {attack}")


def _box(text: str) -> tuple[float, float, float, float] | None:
    values = [float(value) for value in text.split()]
    if not values:
        return None
    if len(values) != 4:
        raise ValueError(f"expected four box coordinates, got: {text!r}")
    return values[0], values[1], values[2], values[3]


def _victims(text: str) -> list[tuple[float, float, float, float]]:
    payload = json.loads(text or "[]")
    return [tuple(float(value) for value in box) for box in payload]


def evaluate_attack_manifest(
    detector: Detector,
    manifest: Path,
    attack: str,
    output: Path,
    target_class: int = 0,
    match_iou: float = 0.5,
    batch_size: int = 16,
) -> dict[str, object]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row.get("poisoned", "1"))]
    required = {"image", "trigger_xyxy", "victim_xyxy"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "ASR manifest must contain poisoned samples and image, trigger_xyxy, victim_xyxy"
        )

    details: list[dict[str, object]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        predictions = detector.predict([Path(row["image"]) for row in batch])
        if len(predictions) != len(batch):
            raise RuntimeError("detector returned a different number of predictions than inputs")
        for row, detections in zip(batch, predictions):
            row_attack = row.get("attack") or attack
            if row_attack != attack:
                raise ValueError(f"manifest attack {row_attack!r} does not match {attack!r}")
            outcomes = attack_outcomes(
                attack,
                detections,
                _box(row["trigger_xyxy"]),
                _victims(row["victim_xyxy"]),
                target_class,
                match_iou,
            )
            details.append(
                {
                    "image": row["image"],
                    "success": int(bool(outcomes) and all(outcomes)),
                    "events": len(outcomes),
                    "successful_events": sum(outcomes),
                    "success_rate": sum(outcomes) / len(outcomes) if outcomes else 0.0,
                    "detections": len(detections),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "asr_samples.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image",
                "success",
                "events",
                "successful_events",
                "success_rate",
                "detections",
            ),
        )
        writer.writeheader()
        writer.writerows(details)
    events = sum(int(row["events"]) for row in details)
    successful_events = sum(int(row["successful_events"]) for row in details)
    summary: dict[str, object] = {
        "attack": attack,
        "samples": len(details),
        "successful_images": sum(int(row["success"]) for row in details),
        "image_success_rate": sum(int(row["success"]) for row in details) / len(details),
        "events": events,
        "successes": successful_events,
        "asr": successful_events / events,
        "target_class": target_class,
        "match_iou": match_iou,
        "manifest": str(manifest.resolve()),
    }
    write_json(output / "asr.json", summary)
    return summary


def _trigger_in_detection(trigger: tuple[int, int, int, int], detection: Detection) -> bool:
    x, y, x2, y2 = trigger
    width_in = detection.xyxy[0] <= x <= detection.xyxy[2] or (
        detection.xyxy[0] <= x2 <= detection.xyxy[2]
    )
    height_in = detection.xyxy[1] <= y <= detection.xyxy[3] or (
        detection.xyxy[1] <= y2 <= detection.xyxy[3]
    )
    return width_in and height_in

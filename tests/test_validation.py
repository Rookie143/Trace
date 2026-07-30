import csv
import json
from pathlib import Path

from trace_detector.types import Detection
from trace_detector.validation import (
    _parse_yolov5_summary,
    attack_succeeded,
    evaluate_attack_manifest,
    yolov5_val_command,
)


def detection(class_id: int = 0) -> Detection:
    return Detection((10.0, 10.0, 30.0, 30.0), 0.9, class_id)


def test_attack_success_semantics() -> None:
    box = (10.0, 10.0, 30.0, 30.0)
    assert attack_succeeded("oga", [detection()], box, [], 0, 0.5)
    # OGA trains a 2.5x target box around the smaller trigger, so its ASR
    # criterion is trigger-region containment rather than box IoU.
    assert attack_succeeded(
        "oga",
        [Detection((0.0, 0.0, 40.0, 40.0), 0.9, 0)],
        box,
        [],
        0,
        0.5,
    )
    assert attack_succeeded("rma", [detection()], None, [box], 0, 0.5)
    assert attack_succeeded("oda", [], None, [box], 0, 0.5)
    assert not attack_succeeded("oda", [detection(3)], None, [box], 0, 0.5)
    assert not attack_succeeded("rma", [detection(3)], None, [box], 0, 0.5)


def test_parse_yolov5_summary() -> None:
    log = "noise\n                 all        5000       36335      0.71      0.62      0.67      0.49\n"
    result = _parse_yolov5_summary(log)
    assert result["images"] == 5000
    assert result["map50_95"] == 0.49


def test_val_command(tmp_path: Path) -> None:
    root = tmp_path / "yolov5"
    root.mkdir()
    (root / "val.py").touch()
    command = yolov5_val_command(
        root, tmp_path / "clean.yaml", tmp_path / "model.pt", tmp_path / "out", 640, 8, "0", 2
    )
    assert str(root / "val.py") in command
    assert "--exist-ok" in command


class FakeDetector:
    names = {0: "person"}

    def predict(self, images):
        return [[detection()] for _ in images]


def test_attack_manifest_outputs_asr(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image", "poisoned", "attack", "trigger_xyxy", "victim_xyxy"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "image": "/unused.jpg",
                "poisoned": 1,
                "attack": "oga",
                "trigger_xyxy": "10 10 30 30",
                "victim_xyxy": "[]",
            }
        )
    result = evaluate_attack_manifest(FakeDetector(), manifest, "oga", tmp_path / "out")
    assert result["asr"] == 1.0
    assert json.loads((tmp_path / "out" / "asr.json").read_text())["successes"] == 1

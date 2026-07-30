from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def write_dataset_yaml(dataset: Path, output: Path, names: list[str]) -> Path:
    payload = {
        "path": str(dataset.resolve()),
        "train": "images/train2017",
        "val": "images/val2017",
        "nc": len(names),
        "names": names,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output


def train_yolov5(
    yolo_root: Path,
    data_yaml: Path,
    weights: str,
    output: Path,
    epochs: int,
    batch_size: int,
    image_size: int,
    device: str,
    workers: int,
    name: str,
    extra: list[str] | None = None,
    dry_run: bool = False,
    trust_checkpoint: bool = False,
) -> list[str]:
    train_script = yolo_root.expanduser().resolve() / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"YOLOv5 train.py not found: {train_script}")
    weights_path = Path(weights).expanduser()
    if weights_path.exists():
        weights = str(weights_path.resolve())
    command = [sys.executable]
    if trust_checkpoint:
        command.extend(
            [
                str(Path(__file__).with_name("training_runner.py")),
                str(train_script),
            ]
        )
    else:
        command.append(str(train_script))
    command.extend(
        [
            "--data",
            str(data_yaml.expanduser().resolve()),
            "--weights",
            weights,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--imgsz",
            str(image_size),
            "--device",
            device,
            "--workers",
            str(workers),
            "--project",
            str(output.expanduser().resolve()),
            "--name",
            name,
        ]
    )
    command.extend(extra or [])
    if not dry_run:
        subprocess.run(command, cwd=yolo_root, check=True)
    return command

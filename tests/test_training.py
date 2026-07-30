from pathlib import Path

from trace_detector.training import train_yolov5


def test_train_resolves_existing_relative_checkpoint(tmp_path: Path, monkeypatch) -> None:
    yolo_root = tmp_path / "yolov5"
    yolo_root.mkdir()
    (yolo_root / "train.py").touch()
    data_yaml = tmp_path / "data.yaml"
    data_yaml.touch()
    checkpoint = tmp_path / "weights.pt"
    checkpoint.touch()
    monkeypatch.chdir(tmp_path)

    command = train_yolov5(
        yolo_root=yolo_root,
        data_yaml=data_yaml,
        weights="weights.pt",
        output=tmp_path / "runs",
        epochs=1,
        batch_size=1,
        image_size=64,
        device="cpu",
        workers=0,
        name="dry",
        dry_run=True,
    )

    assert command[command.index("--weights") + 1] == str(checkpoint.resolve())

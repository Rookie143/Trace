from __future__ import annotations

import csv
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
from tqdm import tqdm

from .config import IMAGE_SUFFIXES

AttackName = Literal["oga", "oda", "rma"]


@dataclass
class PoisonConfig:
    attack: AttackName
    images: Path
    labels: Path
    output: Path
    trigger: Path
    poison_rate: float | None = None
    trigger_size: int | None = None
    trigger_opacity: float | None = None
    target_class: int = 0
    victim_class: int | None = None
    seed: int = 0
    split: str = "train2017"
    paired: bool = False
    clean_mode: Literal["hardlink", "symlink", "copy"] = "hardlink"
    max_images: int | None = None
    sample_fraction: float = 1.0

    def validate(self) -> None:
        if self.attack not in {"oga", "oda", "rma"}:
            raise ValueError(f"unsupported attack: {self.attack}")
        if self.poison_rate is not None and not 0 <= self.poison_rate <= 1:
            raise ValueError("poison_rate must be in [0, 1]")
        if self.trigger_opacity is not None and not 0 <= self.trigger_opacity <= 1:
            raise ValueError("trigger_opacity must be in [0, 1]")
        if self.trigger_size is not None and self.trigger_size < 1:
            raise ValueError("trigger_size must be positive")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        for path in (self.images, self.labels, self.trigger):
            if not path.exists():
                raise FileNotFoundError(path)


@dataclass(frozen=True)
class YoloLabel:
    class_id: int
    x: float
    y: float
    width: float
    height: float

    def line(self) -> str:
        return f"{self.class_id} {self.x:.8f} {self.y:.8f} {self.width:.8f} {self.height:.8f}\n"


@dataclass(frozen=True)
class PoisonRecord:
    image: str
    label: str
    source_image: str
    poisoned: int
    attack: str
    victim_index: int | None
    victim_xyxy: str
    trigger_xyxy: str
    trigger_xyxys: str
    seed: int


def read_labels(path: Path) -> list[YoloLabel]:
    if not path.exists():
        return []
    labels = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{number}: expected 5 YOLO fields")
        labels.append(YoloLabel(int(float(fields[0])), *(float(value) for value in fields[1:])))
    return labels


def _copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _paste(image: Image.Image, trigger: Image.Image, xy: tuple[int, int], opacity: float) -> None:
    patch = trigger.copy()
    alpha = patch.getchannel("A").point(lambda value: round(value * opacity))
    patch.putalpha(alpha)
    image.alpha_composite(patch, xy)


def _box_pixels(label: YoloLabel, width: int, height: int) -> tuple[float, float, float, float]:
    return (
        (label.x - label.width / 2) * width,
        (label.y - label.height / 2) * height,
        (label.x + label.width / 2) * width,
        (label.y + label.height / 2) * height,
    )


def _overlaps(
    xy: tuple[int, int], size: tuple[int, int], labels: list[YoloLabel], width: int, height: int
) -> bool:
    x1, y1, x2, y2 = xy[0], xy[1], xy[0] + size[0], xy[1] + size[1]
    for label in labels:
        a, b, c, d = _box_pixels(label, width, height)
        if min(x2, c) > max(x1, a) and min(y2, d) > max(y1, b):
            return True
    return False


def _random_free_position(
    rng: random.Random,
    canvas: tuple[int, int],
    patch: tuple[int, int],
    labels: list[YoloLabel],
) -> tuple[int, int]:
    width, height = canvas
    for _ in range(100):
        xy = (rng.randint(0, max(0, width - patch[0])), rng.randint(0, max(0, height - patch[1])))
        if not _overlaps(xy, patch, labels, width, height):
            return xy
    return (max(0, width - patch[0]), max(0, height - patch[1]))


def _eligible(labels: list[YoloLabel], config: PoisonConfig) -> list[int]:
    if config.attack == "oda":
        victim_class = 0 if config.victim_class is None else config.victim_class
        return [
            index
            for index, label in enumerate(labels)
            if label.class_id == victim_class
        ]
    if config.attack == "rma":
        return [
            index
            for index, label in enumerate(labels)
            if label.class_id != config.target_class
            and (config.victim_class is None or label.class_id == config.victim_class)
        ]
    return list(range(len(labels)))


def poison_one(
    image_path: Path,
    label_path: Path,
    destination_image: Path,
    destination_label: Path,
    trigger: Image.Image,
    config: PoisonConfig,
    rng: random.Random,
) -> tuple[
    int | None,
    list[tuple[float, float, float, float]],
    list[tuple[int, int, int, int]],
]:
    labels = read_labels(label_path)
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    default_size = {"oga": 25, "oda": 30, "rma": 30}[config.attack]
    default_opacity = {"oga": 0.30, "oda": 0.50, "rma": 0.50}[config.attack]
    side = min(config.trigger_size or default_size, width, height)
    opacity = config.trigger_opacity if config.trigger_opacity is not None else default_opacity
    patch = trigger.resize((side, side), Image.Resampling.LANCZOS)
    victim_index: int | None = None
    victim_boxes: list[tuple[float, float, float, float]] = []
    trigger_boxes: list[tuple[int, int, int, int]] = []

    if config.attack == "oga":
        x, y = _random_free_position(rng, image.size, patch.size, labels)
        _paste(image, patch, (x, y), opacity)
        trigger_boxes = [(x, y, x + side, y + side)]
        labels.append(
            YoloLabel(
                config.target_class,
                (x + side / 2) / width,
                (y + side / 2) / height,
                side * 2.5 / width,
                side * 2.5 / height,
            )
        )
    elif config.attack == "oda":
        selected = _eligible(labels, config)
        if not selected:
            raise ValueError("image has no eligible victim object")
        victim_index = selected[0]
        victim_boxes = [_box_pixels(labels[index], width, height) for index in selected]
        for index in selected:
            victim = labels[index]
            x = int(victim.x * width - side / 2)
            y = int(victim.y * height - side / 2)
            _paste(image, patch, (x, y), opacity)
            trigger_boxes.append((x, y, x + side, y + side))
        selected_set = set(selected)
        labels = [label for index, label in enumerate(labels) if index not in selected_set]
    else:
        candidates = _eligible(labels, config)
        if not candidates:
            raise ValueError("image has no eligible victim object")
        victim_index = candidates[0]
        victim = labels[victim_index]
        victim_boxes = [_box_pixels(victim, width, height)]
        x = max(0, min(width - side, int(victim.x * width - side / 2)))
        y = max(0, min(height - side, int(victim.y * height - side / 2)))
        _paste(image, patch, (x, y), opacity)
        trigger_boxes = [(x, y, x + side, y + side)]
        # Match the data recipe used to train the bundled RMA checkpoint: the
        # writer stops after relabeling the first eligible victim.
        labels = labels[: victim_index + 1]
        labels[victim_index] = YoloLabel(
            config.target_class, victim.x, victim.y, victim.width, victim.height
        )

    destination_image.parent.mkdir(parents=True, exist_ok=True)
    destination_label.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination_image)
    destination_label.write_text("".join(label.line() for label in labels), encoding="utf-8")
    return victim_index, victim_boxes, trigger_boxes


def prepare_dataset(config: PoisonConfig) -> list[PoisonRecord]:
    config.validate()
    rng = random.Random(config.seed)
    poison_rate = (
        config.poison_rate
        if config.poison_rate is not None
        else {"oga": 0.20, "oda": 0.20, "rma": 0.30}[config.attack]
    )
    trigger = Image.open(config.trigger).convert("RGBA")
    image_files = sorted(
        path for path in config.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if config.sample_fraction < 1:
        sample_count = max(1, round(len(image_files) * config.sample_fraction))
        image_files = sorted(random.Random(config.seed).sample(image_files, sample_count))
    if config.paired and config.attack != "oga":
        image_files = [
            path
            for path in image_files
            if _eligible(read_labels(config.labels / f"{path.stem}.txt"), config)
        ]
    if config.max_images is not None:
        image_files = image_files[: config.max_images]

    output_images = config.output / "images" / config.split
    output_labels = config.output / "labels" / config.split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    records: list[PoisonRecord] = []

    for index, source in enumerate(tqdm(image_files, desc=f"prepare {config.attack}")):
        source_label = config.labels / f"{source.stem}.txt"
        if not source_label.exists():
            continue
        wants_poison = rng.random() < poison_rate
        can_poison = config.attack == "oga" or bool(_eligible(read_labels(source_label), config))
        poison = wants_poison and can_poison
        if config.paired and not poison:
            continue
        variants = (False, True) if config.paired else (poison,)
        for is_poisoned in variants:
            prefix = "poison_" if is_poisoned else ("clean_" if config.paired else "")
            destination_image = output_images / f"{prefix}{source.name}"
            destination_label = output_labels / f"{prefix}{source.stem}.txt"
            if is_poisoned:
                victim, victim_boxes, trigger_boxes = poison_one(
                    source,
                    source_label,
                    destination_image,
                    destination_label,
                    trigger,
                    config,
                    rng,
                )
            else:
                if config.attack in {"oda", "rma"} and not config.paired:
                    destination_image.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(source).convert("RGB").save(destination_image)
                else:
                    _copy(source, destination_image, config.clean_mode)
                _copy(source_label, destination_label, config.clean_mode)
                victim, victim_boxes, trigger_boxes = None, [], []
            first_trigger = trigger_boxes[0] if trigger_boxes else ()
            records.append(
                PoisonRecord(
                    image=str(destination_image.resolve()),
                    label=str(destination_label.resolve()),
                    source_image=str(source.resolve()),
                    poisoned=int(is_poisoned),
                    attack=config.attack,
                    victim_index=victim,
                    victim_xyxy=json.dumps(victim_boxes),
                    trigger_xyxy=" ".join(map(str, first_trigger)),
                    trigger_xyxys=json.dumps(trigger_boxes),
                    seed=config.seed + index,
                )
            )

    manifest = config.output / f"manifest_{config.split}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(asdict(records[0]).keys()) if records else []
        )
        if records:
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
    metadata = asdict(config)
    for key in ("images", "labels", "output", "trigger"):
        metadata[key] = str(metadata[key])
    metadata["resolved_poison_rate"] = poison_rate
    metadata["resolved_trigger_size"] = config.trigger_size or {
        "oga": 25,
        "oda": 30,
        "rma": 30,
    }[config.attack]
    metadata["resolved_trigger_opacity"] = (
        config.trigger_opacity
        if config.trigger_opacity is not None
        else {"oga": 0.30, "oda": 0.50, "rma": 0.50}[config.attack]
    )
    (config.output / "poison_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return records

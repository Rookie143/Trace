from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import numpy as np
from PIL import Image

from .config import IMAGE_SUFFIXES

T = TypeVar("T")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def image_paths(source: str | Path) -> list[Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_file():
        return [
            Path(line.strip()).expanduser().resolve()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(path)


def batched(items: list[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_rgb(value: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value.astype(np.uint8)).convert("RGB")
    with Image.open(value) as image:
        return image.convert("RGB")


def write_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    union += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
    return intersection / union if union else 0.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceScore:
    image: str
    ctc: float
    ftc: float
    score: float
    queries: int
    label: int | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ImageInput = Image.Image | np.ndarray | str | Path


class Detector(Protocol):
    names: dict[int, str]

    def predict(self, images: Sequence[ImageInput]) -> list[list[Detection]]:
        """Return one detection list per input image."""

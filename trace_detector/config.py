from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TraceConfig:
    backgrounds: list[str] = field(default_factory=list)
    nbo: str = ""
    nbo_class_id: int | None = None
    references: str | None = None
    compute_ctc: bool = True
    compute_ftc: bool = True
    background_queries: int = 30
    foreground_queries: int = 50
    points_per_query: int = 5
    background_opacity: float = 0.15
    # OGA is the reference implementation for every attack profile.
    ctc_statistic: str = "mean_abs_delta"
    ftc_statistic: str = "variance"
    nbo_scale: float = 0.12
    nbo_size_pixels: int | None = None
    ssim_threshold: float = 0.10
    match_iou: float = 0.50
    transformation_confidence: float | None = None
    ctc_scale: float = 100.0
    ftc_scale: float = 100.0
    batch_size: int = 16
    seed: int = 0

    def validate(self) -> None:
        if self.compute_ctc and not self.backgrounds:
            raise ValueError("TRACE needs at least one background image")
        if self.compute_ftc and not self.nbo:
            raise ValueError("TRACE needs an NBO image")
        if self.compute_ctc and self.background_queries < 1:
            raise ValueError("background_queries must be positive when CTC is enabled")
        if self.compute_ftc and self.foreground_queries < 1:
            raise ValueError("foreground_queries must be positive when FTC is enabled")
        if not self.compute_ctc and not self.compute_ftc:
            raise ValueError("TRACE needs at least one enabled component")
        if self.compute_ctc and self.compute_ftc and self.ctc_scale != self.ftc_scale:
            raise ValueError("combined TRACE requires equal CTC/FTC weights")
        if self.points_per_query < 1:
            raise ValueError("points_per_query must be positive")
        if self.nbo_size_pixels is not None and self.nbo_size_pixels < 8:
            raise ValueError("nbo_size_pixels must be at least 8")
        if not 0 <= self.background_opacity <= 1:
            raise ValueError("background_opacity must be in [0, 1]")
        if self.ctc_statistic not in {"variance", "mean_abs_delta"}:
            raise ValueError("ctc_statistic must be variance or mean_abs_delta")
        if self.ftc_statistic != "variance":
            raise ValueError("ftc_statistic must be variance")
        if not 0 <= self.ssim_threshold <= 1:
            raise ValueError("ssim_threshold must be in [0, 1]")
        if (
            self.transformation_confidence is not None
            and not 0 <= self.transformation_confidence <= 1
        ):
            raise ValueError("transformation_confidence must be in [0, 1]")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping in {path}")
    return data


def trace_config(path: str | Path, components: str | None = None) -> TraceConfig:
    source = Path(path).expanduser().resolve()
    raw = load_yaml(source)
    root = source.parent

    def resolve(value: str) -> str:
        candidate = Path(value).expanduser()
        return str(candidate if candidate.is_absolute() else (root / candidate).resolve())

    backgrounds: list[str] = []
    for entry in raw.get("backgrounds", []):
        candidate = Path(resolve(str(entry)))
        if candidate.is_dir():
            backgrounds.extend(
                str(p) for p in sorted(candidate.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            backgrounds.append(str(candidate))
    raw["backgrounds"] = backgrounds
    raw["nbo"] = resolve(str(raw.get("nbo", ""))) if raw.get("nbo") else ""
    if raw.get("references"):
        raw["references"] = resolve(str(raw["references"]))
    config = TraceConfig(**raw)
    if components is not None:
        if components not in {"full", "ctc", "ftc"}:
            raise ValueError("components must be full, ctc, or ftc")
        config.compute_ctc = components in {"full", "ctc"}
        config.compute_ftc = components in {"full", "ftc"}
    config.validate()
    return config


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

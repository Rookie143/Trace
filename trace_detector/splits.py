from __future__ import annotations

import csv
import random
from pathlib import Path

import yaml

from .config import IMAGE_SUFFIXES


def subset_image_list(
    source: Path,
    output: Path,
    fraction: float = 0.1,
    seed: int = 0,
) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    lines = [
        line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise ValueError("image list is empty")
    selected = [
        str((source.parent / line).resolve()) if not Path(line).is_absolute() else line
        for line in lines
    ]
    random.Random(seed).shuffle(selected)
    count = max(1, round(len(selected) * fraction))
    selected = sorted(selected[:count])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return len(selected)


def subset_dataset_yaml(template: Path, image_list: Path, output: Path) -> None:
    with template.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("dataset YAML must contain a mapping")
    payload["val"] = str(image_list.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def paired_manifest(
    clean: Path,
    poison: Path,
    output: Path,
    max_pairs: int | None = None,
    seed: int = 0,
    clean_prefix: str = "",
    poison_prefix: str = "",
) -> int:
    """Build a balanced manifest after normalizing optional filename prefixes."""

    def key(path: Path, prefix: str) -> str:
        stem = path.stem
        if prefix and stem.startswith(prefix):
            stem = stem[len(prefix) :]
        return str(int(stem)) if stem.isdigit() else stem

    clean_files = {
        key(path, clean_prefix): path.resolve()
        for path in clean.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    poison_files = {
        key(path, poison_prefix): path.resolve()
        for path in poison.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    names = sorted(clean_files.keys() & poison_files.keys())
    if not names:
        raise ValueError("clean and poison directories contain no same-named images")
    random.Random(seed).shuffle(names)
    if max_pairs is not None:
        names = names[:max_pairs]
    names.sort()
    rows = []
    for name in names:
        source = str(clean_files[name])
        rows.extend(
            (
                {"image": source, "poisoned": 0, "source_image": source},
                {
                    "image": str(poison_files[name]),
                    "poisoned": 1,
                    "source_image": source,
                },
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "poisoned", "source_image"))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)

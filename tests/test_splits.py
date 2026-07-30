import csv
from pathlib import Path

from PIL import Image

from trace_detector.splits import paired_manifest, subset_image_list


def test_subset_image_list_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "all.txt"
    source.write_text("".join(f"image-{index}.jpg\n" for index in range(100)))
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    assert subset_image_list(source, first, 0.1, 7) == 10
    assert subset_image_list(source, second, 0.1, 7) == 10
    assert first.read_text() == second.read_text()


def test_paired_manifest_records_shared_source(tmp_path: Path) -> None:
    clean, poison = tmp_path / "clean", tmp_path / "poison"
    clean.mkdir()
    poison.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (4, 4)).save(clean / name)
        Image.new("RGB", (4, 4)).save(poison / name)
    output = tmp_path / "paired.csv"
    assert paired_manifest(clean, poison, output, max_pairs=1, seed=0) == 2
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["poisoned"] for row in rows} == {"0", "1"}
    assert len({row["source_image"] for row in rows}) == 1


def test_paired_manifest_normalizes_prefixes_and_numeric_padding(tmp_path: Path) -> None:
    clean, poison = tmp_path / "clean", tmp_path / "poison"
    clean.mkdir()
    poison.mkdir()
    Image.new("RGB", (4, 4)).save(clean / "000000123456.jpg")
    Image.new("RGB", (4, 4)).save(poison / "poison123456.jpg")
    output = tmp_path / "paired.csv"

    assert (
        paired_manifest(
            clean,
            poison,
            output,
            poison_prefix="poison",
        )
        == 2
    )

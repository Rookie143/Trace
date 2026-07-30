import json
from pathlib import Path

from PIL import Image

from trace_detector.attacks import PoisonConfig, prepare_dataset, read_labels


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(images / "1.jpg")
    (labels / "1.txt").write_text("2 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    trigger = tmp_path / "trigger.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(trigger)
    return images, labels, trigger


def test_all_attacks_prepare_a_poisoned_sample(tmp_path: Path) -> None:
    for attack in ("oga", "oda", "rma"):
        images, labels, trigger = _fixture(tmp_path / attack)
        output = tmp_path / f"out-{attack}"
        records = prepare_dataset(
            PoisonConfig(
                attack=attack,
                images=images,
                labels=labels,
                output=output,
                trigger=trigger,
                poison_rate=1.0,
                target_class=0,
                victim_class=2 if attack == "oda" else None,
                seed=7,
            )
        )
        assert len(records) == 1
        assert records[0].poisoned == 1
        result = read_labels(Path(records[0].label))
        if attack == "oga":
            assert len(result) == 2 and result[-1].class_id == 0
        elif attack == "oda":
            assert result == []
        elif attack == "rma":
            assert len(result) == 1 and result[0].class_id == 0


def test_paired_mode_emits_clean_and_poison(tmp_path: Path) -> None:
    images, labels, trigger = _fixture(tmp_path)
    records = prepare_dataset(
        PoisonConfig(
            attack="oga",
            images=images,
            labels=labels,
            output=tmp_path / "out",
            trigger=trigger,
            poison_rate=1.0,
            paired=True,
        )
    )
    assert [record.poisoned for record in records] == [0, 1]


def test_oda_attacks_every_person_and_keeps_other_labels(tmp_path: Path) -> None:
    images, labels, trigger = _fixture(tmp_path)
    (labels / "1.txt").write_text(
        "0 0.2 0.5 0.1 0.2\n2 0.5 0.5 0.1 0.2\n0 0.8 0.5 0.1 0.2\n",
        encoding="utf-8",
    )
    records = prepare_dataset(
        PoisonConfig(
            attack="oda",
            images=images,
            labels=labels,
            output=tmp_path / "out",
            trigger=trigger,
            poison_rate=1.0,
        )
    )

    result = read_labels(Path(records[0].label))

    assert [label.class_id for label in result] == [2]
    assert len(json.loads(records[0].trigger_xyxys)) == 2


def test_rma_matches_checkpoint_first_victim_label_policy(tmp_path: Path) -> None:
    images, labels, trigger = _fixture(tmp_path)
    (labels / "1.txt").write_text(
        "0 0.2 0.5 0.1 0.2\n2 0.5 0.5 0.1 0.2\n3 0.8 0.5 0.1 0.2\n",
        encoding="utf-8",
    )
    records = prepare_dataset(
        PoisonConfig(
            attack="rma",
            images=images,
            labels=labels,
            output=tmp_path / "out",
            trigger=trigger,
            poison_rate=1.0,
        )
    )

    result = read_labels(Path(records[0].label))

    assert [label.class_id for label in result] == [0, 0]


def test_sample_fraction_is_deterministic(tmp_path: Path) -> None:
    images, labels, trigger = _fixture(tmp_path)
    for index in range(2, 11):
        Image.new("RGB", (100, 80), "white").save(images / f"{index}.jpg")
        (labels / f"{index}.txt").write_text("2 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    first = prepare_dataset(
        PoisonConfig(
            attack="oga",
            images=images,
            labels=labels,
            output=tmp_path / "first",
            trigger=trigger,
            poison_rate=0,
            sample_fraction=0.3,
            seed=7,
        )
    )
    second = prepare_dataset(
        PoisonConfig(
            attack="oga",
            images=images,
            labels=labels,
            output=tmp_path / "second",
            trigger=trigger,
            poison_rate=0,
            sample_fraction=0.3,
            seed=7,
        )
    )

    assert len(first) == 3
    assert [Path(row.source_image).name for row in first] == [
        Path(row.source_image).name for row in second
    ]

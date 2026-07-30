import random
from pathlib import Path

from PIL import Image

from trace_detector.config import TraceConfig
from trace_detector.trace import TraceDetector, _matches_existing_detection
from trace_detector.types import Detection


class StableDetector:
    names = {0: "object", 11: "stop sign"}

    def predict(self, images):
        return [[Detection((10, 10, 40, 40), 0.9, 0)] for _ in images]


class EmptyDetector:
    names = {11: "stop sign"}

    def predict(self, images):
        return [[] for _ in images]


class IslandEffectDetector:
    names = {0: "victim", 11: "stop sign"}

    def predict(self, images):
        predictions = [
            [
                Detection((0, 0, 8, 8), 0.9, 11),
                Detection((0, 0, 8, 8), 0.8, 0),
            ],
            [Detection((20, 20, 28, 28), 0.9, 11)],
        ]
        return predictions[: len(images)]


class EmergedContextDetector:
    names = {0: "object"}

    def __init__(self):
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        if self.calls == 1:
            return [[] for _ in images]
        return [[Detection((10, 10, 40, 40), 0.8, 0)] for _ in images]


def test_trace_score_is_finite(tmp_path: Path) -> None:
    background = tmp_path / "background.jpg"
    nbo = tmp_path / "nbo.png"
    Image.new("RGB", (64, 64), "blue").save(background)
    Image.new("RGBA", (8, 8), "red").save(nbo)
    config = TraceConfig(
        backgrounds=[str(background)],
        nbo=str(nbo),
        background_queries=2,
        foreground_queries=2,
        points_per_query=2,
        batch_size=2,
    )
    result = TraceDetector(StableDetector(), config).score(Image.new("RGB", (64, 64), "white"))
    assert -1 <= result.score <= 1
    assert result.queries == 5


def test_empty_detection_still_runs_contextual_queries(tmp_path: Path) -> None:
    background = tmp_path / "background.jpg"
    nbo = tmp_path / "nbo.png"
    Image.new("RGB", (64, 64), "blue").save(background)
    Image.new("RGBA", (8, 8), "red").save(nbo)
    config = TraceConfig(
        backgrounds=[str(background)],
        nbo=str(nbo),
        background_queries=2,
        foreground_queries=2,
        points_per_query=2,
    )
    result = TraceDetector(EmptyDetector(), config).score(Image.new("RGB", (64, 64), "white"))
    assert result.ctc == 0.0
    assert result.queries == 5


def test_empty_original_uses_transformed_predictions_for_ctc(tmp_path: Path) -> None:
    background = tmp_path / "background.jpg"
    nbo = tmp_path / "nbo.png"
    Image.new("RGB", (64, 64), "blue").save(background)
    Image.new("RGBA", (8, 8), "red").save(nbo)
    config = TraceConfig(
        backgrounds=[str(background)],
        nbo=str(nbo),
        ctc_statistic="mean_abs_delta",
        background_queries=2,
        foreground_queries=1,
        points_per_query=1,
    )

    result = TraceDetector(EmergedContextDetector(), config).score(
        Image.new("RGB", (64, 64), "white")
    )

    assert result.ctc == 0.8
    assert result.queries == 4


def test_mean_abs_delta_ctc_matches_stable_detection(tmp_path: Path) -> None:
    background = tmp_path / "background.jpg"
    nbo = tmp_path / "nbo.png"
    Image.new("RGB", (64, 64), "blue").save(background)
    Image.new("RGBA", (8, 8), "red").save(nbo)
    config = TraceConfig(
        backgrounds=[str(background)],
        nbo=str(nbo),
        ctc_statistic="mean_abs_delta",
        background_queries=2,
        foreground_queries=1,
        points_per_query=1,
    )

    ctc, queries = TraceDetector(StableDetector(), config).contextual(
        Image.new("RGB", (64, 64), "white"),
        [Detection((10, 10, 40, 40), 0.9, 0)],
        random.Random(0),
    )

    assert ctc == 0
    assert queries == 2


def test_disabled_ftc_avoids_focal_queries(tmp_path: Path) -> None:
    background = tmp_path / "background.jpg"
    nbo = tmp_path / "nbo.png"
    Image.new("RGB", (64, 64), "blue").save(background)
    Image.new("RGBA", (8, 8), "red").save(nbo)
    config = TraceConfig(
        backgrounds=[str(background)],
        nbo=str(nbo),
        compute_ftc=False,
        background_queries=2,
        ctc_scale=1,
        ftc_scale=0,
    )

    result = TraceDetector(StableDetector(), config).score(Image.new("RGB", (64, 64), "white"))

    assert result.ftc == 0
    assert result.queries == 3


def test_ftc_randomness_is_independent_of_ctc_mode(tmp_path: Path) -> None:
    background = tmp_path / "background.jpg"
    nbo = tmp_path / "nbo.png"
    Image.new("RGB", (64, 64), "blue").save(background)
    Image.new("RGBA", (8, 8), "red").save(nbo)
    common = {
        "nbo": str(nbo),
        "background_queries": 2,
        "foreground_queries": 3,
        "points_per_query": 1,
    }
    full = TraceConfig(backgrounds=[str(background)], **common)
    ftc_only = TraceConfig(compute_ctc=False, **common)
    image = Image.new("RGB", (64, 64), "white")

    full_score = TraceDetector(StableDetector(), full).score(image)
    ftc_score = TraceDetector(StableDetector(), ftc_only).score(image)

    assert full_score.ftc == ftc_score.ftc


def test_transformation_confidence_filters_only_transformed_predictions(
    tmp_path: Path,
) -> None:
    nbo = tmp_path / "nbo.png"
    Image.new("RGBA", (8, 8), "red").save(nbo)
    trace = TraceDetector(
        StableDetector(),
        TraceConfig(
            compute_ctc=False,
            nbo=str(nbo),
            foreground_queries=1,
            points_per_query=1,
            transformation_confidence=0.95,
        ),
    )

    assert trace._predict_transformed([Image.new("RGB", (8, 8))]) == [[]]


def test_position_sampler_returns_requested_non_overlapping_count() -> None:
    detector = object.__new__(TraceDetector)
    positions = detector._positions(random.Random(0), 640, 480, (165, 165), 5)
    assert len(positions) == 5


def test_background_sampler_uses_unique_images_before_repeating(tmp_path: Path) -> None:
    nbo = tmp_path / "nbo.png"
    Image.new("RGBA", (8, 8), "red").save(nbo)
    backgrounds = []
    for index in range(4):
        path = tmp_path / f"background-{index}.png"
        Image.new("RGB", (8, 8), (index, 0, 0)).save(path)
        backgrounds.append(str(path))
    trace = TraceDetector(
        StableDetector(),
        TraceConfig(
            backgrounds=backgrounds,
            nbo=str(nbo),
            background_queries=3,
            foreground_queries=1,
            points_per_query=1,
        ),
    )

    selected = trace._sample_backgrounds(random.Random(0))

    assert len(selected) == 3
    assert len({image.getpixel((0, 0)) for image in selected}) == 3


def test_flat_ssim_module_reference_layout_is_supported(tmp_path: Path) -> None:
    reference_dir = tmp_path / "coco"
    reference_dir.mkdir()
    Image.new("RGB", (16, 16), "white").save(reference_dir / "0.jpg")

    references = TraceDetector._load_references(str(tmp_path))

    assert set(references) == {0}
    assert references[0][0].shape == (16, 16)


def test_focal_keeps_emerged_victim_overlapping_nbo(tmp_path: Path, monkeypatch) -> None:
    background = tmp_path / "background.jpg"
    nbo = tmp_path / "nbo.png"
    Image.new("RGB", (64, 64), "blue").save(background)
    Image.new("RGBA", (8, 8), "red").save(nbo)
    config = TraceConfig(
        backgrounds=[str(background)],
        nbo=str(nbo),
        nbo_class_id=11,
        foreground_queries=2,
        points_per_query=1,
        nbo_scale=0.125,
        batch_size=2,
    )
    trace = TraceDetector(IslandEffectDetector(), config)
    positions = iter([[(0, 0)], [(20, 20)]])
    monkeypatch.setattr(trace, "_positions", lambda *args: next(positions))

    ftc, queries = trace.focal(Image.new("RGB", (64, 64), "white"), [], random.Random(0))

    assert queries == 2
    assert ftc > 0


def test_focal_detection_shift_is_local_to_each_nbo(tmp_path: Path, monkeypatch) -> None:
    background = tmp_path / "background.jpg"
    nbo = tmp_path / "nbo.png"
    Image.new("RGB", (64, 64), "blue").save(background)
    Image.new("RGBA", (8, 8), "red").save(nbo)
    config = TraceConfig(
        backgrounds=[str(background)],
        nbo=str(nbo),
        nbo_class_id=11,
        foreground_queries=1,
        points_per_query=2,
        nbo_scale=0.125,
    )

    class TwoPatchDetector:
        def predict(self, images):
            return [
                [
                    Detection((0, 0, 8, 8), 0.9, 11),
                    Detection((20, 20, 28, 28), 0.9, 11),
                    Detection((0, 0, 8, 8), 0.8, 0),
                ]
                for _ in images
            ]

    trace = TraceDetector(TwoPatchDetector(), config)
    monkeypatch.setattr(trace, "_positions", lambda *args: [(0, 0), (20, 20)])

    ftc, _ = trace.focal(Image.new("RGB", (64, 64), "white"), [], random.Random(0))

    assert ftc > 0


def test_focal_suppresses_shifted_original_detection() -> None:
    original = [Detection((0, 0, 20, 20), 0.8, 0)]
    shifted = Detection((13, 13, 33, 33), 0.7, 0)

    assert _matches_existing_detection(shifted, original, 0.5)

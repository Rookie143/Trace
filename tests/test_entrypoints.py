import sys
from pathlib import Path

from detect import RELEASED_MODEL_SETTINGS, filename_label, make_trace_config, parse_args
from poison import POISON_RATE, TRIGGER_OPACITY, TRIGGER_SIZE


def test_paired_image_names_are_labeled_automatically() -> None:
    assert filename_label(Path("clean_0001.jpg")) == 0
    assert filename_label(Path("poison_0001.jpg")) == 1
    assert filename_label(Path("0001.jpg")) is None


def test_detection_cli_takes_a_model_not_an_attack(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["detect.py", "--model", "model.pt", "--source", "images"],
    )
    args = parse_args()
    assert args.model == Path("model.pt")
    assert not hasattr(args, "attack")


def test_released_models_share_the_complete_score_definition() -> None:
    configs = [make_trace_config(settings) for settings in RELEASED_MODEL_SETTINGS.values()]
    definitions = {
        (
            config.compute_ctc,
            config.compute_ftc,
            config.ctc_statistic,
            config.ftc_statistic,
            config.points_per_query,
            config.ctc_scale,
            config.ftc_scale,
        )
        for config in configs
    }
    assert definitions == {(True, True, "mean_abs_delta", "variance", 5, 1.0, 1.0)}


def test_released_model_profiles_are_complete() -> None:
    assert len(RELEASED_MODEL_SETTINGS) == 3
    assert all(
        {"confidence", "background_opacity", "ssim_threshold", "threshold"} == set(settings)
        for settings in RELEASED_MODEL_SETTINGS.values()
    )


def test_poisoning_defaults_are_explicit_for_every_attack() -> None:
    assert set(POISON_RATE) == set(TRIGGER_SIZE) == set(TRIGGER_OPACITY) == {
        "oga",
        "oda",
        "rma",
    }

from pathlib import Path

from detect import ATTACK_SETTINGS, filename_label, make_trace_config
from poison import POISON_RATE, TRIGGER_OPACITY, TRIGGER_SIZE


def test_paired_image_names_are_labeled_automatically() -> None:
    assert filename_label(Path("clean_0001.jpg")) == 0
    assert filename_label(Path("poison_0001.jpg")) == 1
    assert filename_label(Path("0001.jpg")) is None


def test_attacks_share_the_complete_score_definition() -> None:
    configs = [make_trace_config(attack) for attack in ATTACK_SETTINGS]
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


def test_poisoning_defaults_are_explicit_for_every_attack() -> None:
    assert set(POISON_RATE) == set(TRIGGER_SIZE) == set(TRIGGER_OPACITY) == set(ATTACK_SETTINGS)

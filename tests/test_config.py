from pathlib import Path

from trace_detector.config import trace_config


def test_component_override_is_applied_before_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "trace.yaml"
    config_path.write_text("backgrounds:\n  - background.jpg\n", encoding="utf-8")

    config = trace_config(config_path, "ctc")

    assert config.compute_ctc
    assert not config.compute_ftc


def test_public_attack_profiles_share_score_definition() -> None:
    root = Path(__file__).parents[1] / "configs"
    profiles = [
        trace_config(root / f"{attack}.yaml")
        for attack in ("oga", "oda", "rma")
    ]
    definitions = {
        (
            config.ctc_statistic,
            config.ftc_statistic,
            config.points_per_query,
            config.ctc_scale,
            config.ftc_scale,
        )
        for config in profiles
    }

    assert definitions == {("mean_abs_delta", "variance", 5, 1.0, 1.0)}

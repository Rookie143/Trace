from trace_detector.cli import build_parser
from trace_detector.run import build_parser as build_run_parser


def detect_args(*extra: str):
    return build_parser().parse_args(
        [
            "detect",
            "--attack",
            "oga",
            "--weights",
            "model.pt",
            "--yolo-root",
            "yolov5",
            "--source",
            "image.jpg",
            "--config",
            "trace.yaml",
            "--output",
            "output",
            *extra,
        ]
    )


def test_detect_uses_complete_trace_by_default() -> None:
    assert detect_args().components == "full"


def test_detect_allows_one_component() -> None:
    assert detect_args("--components", "ftc").components == "ftc"


def test_prepare_defaults_to_full_source_fraction() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "--attack",
            "oda",
            "--images",
            "images",
            "--labels",
            "labels",
            "--output",
            "output",
            "--trigger",
            "trigger.png",
        ]
    )
    assert args.sample_fraction == 1.0


def test_primary_trace_command_only_requires_attack_and_source() -> None:
    args = build_run_parser().parse_args(["--attack", "oga", "image.jpg"])

    assert args.attack == "oga"
    assert str(args.source) == "image.jpg"
    assert args.weights is None
    assert args.config is None

import pytest

from trace_detector.evaluation import metrics, select_threshold
from trace_detector.types import TraceScore


def test_metrics_for_separable_scores() -> None:
    rows = [
        TraceScore("clean-a.jpg", 0.1, 0.0, -0.5, 1, 0),
        TraceScore("clean-b.jpg", 0.1, 0.0, -0.2, 1, 0),
        TraceScore("poison-a.jpg", 0.0, 0.1, 0.4, 1, 1),
        TraceScore("poison-b.jpg", 0.0, 0.1, 0.8, 1, 1),
    ]
    result = metrics(rows)
    assert result["f1"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["auroc"] == pytest.approx(1.0)


def test_threshold_policies() -> None:
    labels = [0, 0, 1, 1]
    scores = [-0.5, -0.2, 0.4, 0.8]
    assert select_threshold(labels, scores, "fixed", 0.25) == 0.25
    assert select_threshold(labels, scores, "dataset-optimal") == 0.4

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .types import TraceScore
from .utils import write_json


def best_f1_threshold(labels: list[int], scores: list[float]) -> float:
    candidates = np.unique(np.asarray(scores, dtype=float))
    if not len(candidates):
        raise ValueError("no scores")
    values = [
        f1_score(labels, np.asarray(scores) >= threshold, zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(values))])


def select_threshold(
    labels: list[int],
    scores: list[float],
    policy: str = "dataset-optimal",
    threshold: float | None = None,
) -> float:
    if policy == "fixed":
        if threshold is None:
            raise ValueError("fixed threshold policy requires a threshold")
        return float(threshold)
    if policy == "dataset-optimal":
        return best_f1_threshold(labels, scores)
    raise ValueError(f"unsupported threshold policy: {policy}")


def metrics(
    rows: Iterable[TraceScore],
    threshold: float | None = None,
    policy: str | None = None,
) -> dict[str, float | str]:
    data = [row for row in rows if row.label is not None]
    if not data:
        raise ValueError("evaluation needs labeled rows")
    labels = [int(row.label) for row in data]
    scores = [row.score for row in data]
    if len(set(labels)) != 2:
        raise ValueError("AUROC needs both clean (0) and poisoned (1) samples")
    selected_policy = policy or ("fixed" if threshold is not None else "dataset-optimal")
    selected = select_threshold(labels, scores, selected_policy, threshold)
    predictions = [int(score >= selected) for score in scores]
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "samples": float(len(data)),
        "clean_samples": float(labels.count(0)),
        "poisoned_samples": float(labels.count(1)),
        "threshold": float(selected),
        "threshold_policy": selected_policy,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "roc_auc_trapezoid": float(auc(fpr, tpr)),
    }


def write_roc(path: str | Path, rows: Iterable[TraceScore]) -> None:
    data = [row for row in rows if row.label is not None]
    labels = [int(row.label) for row in data]
    scores = [row.score for row in data]
    if len(set(labels)) != 2:
        raise ValueError("ROC needs both clean and poisoned samples")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("threshold", "fpr", "tpr"))
        writer.writeheader()
        writer.writerows(
            {"threshold": threshold, "fpr": false_rate, "tpr": true_rate}
            for threshold, false_rate, true_rate in zip(thresholds, fpr, tpr)
        )


def write_scores(path: str | Path, rows: list[TraceScore]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].to_dict()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(row.to_dict() for row in rows)


def read_scores(path: str | Path) -> list[TraceScore]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            TraceScore(
                image=row["image"],
                ctc=float(row["ctc"]),
                ftc=float(row["ftc"]),
                score=float(row["score"]),
                queries=int(row["queries"]),
                label=int(row["label"]) if row.get("label", "") != "" else None,
                seconds=float(row.get("seconds", 0.0)),
            )
            for row in csv.DictReader(handle)
        ]


def evaluate_file(
    scores_path: str | Path,
    output: str | Path,
    threshold: float | None = None,
    policy: str | None = None,
    roc_output: str | Path | None = None,
) -> dict[str, float | str]:
    rows = read_scores(scores_path)
    result = metrics(rows, threshold, policy)
    write_json(output, result)
    if roc_output is not None:
        write_roc(roc_output, rows)
    return result

"""Deterministic binary metrics used by the frozen development evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary ranking/classification metrics and the exact confusion counts."""

    sample_count: int
    positive_count: int
    negative_count: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    average_precision: float
    auroc: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float

    def as_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


def _strict_binary(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "buif":
        raise ValueError(f"{name} must be a one-dimensional numeric 0/1 array")
    if not np.isfinite(array).all() or not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must contain only finite 0/1 values")
    return array.astype(bool, copy=False)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision with equal scores handled as one threshold group."""

    targets = _strict_binary(labels, name="labels")
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != targets.shape or not np.isfinite(values).all():
        raise ValueError("scores must be finite and have the same shape as labels")
    positive_count = int(targets.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-values, kind="mergesort")
    ordered_scores = values[order]
    ordered_targets = targets[order].astype(np.int64)
    group_ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), len(values) - 1]
    cumulative_positive = np.cumsum(ordered_targets)[group_ends]
    retrieved = group_ends + 1
    recall = cumulative_positive / positive_count
    precision = cumulative_positive / retrieved
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, assigning half credit to score ties."""

    targets = _strict_binary(labels, name="labels")
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != targets.shape or not np.isfinite(values).all():
        raise ValueError("scores must be finite and have the same shape as labels")
    positive_count = int(targets.sum())
    negative_count = len(targets) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ordered_scores = values[order]
    ordered_targets = targets[order]
    favorable_pairs = 0.0
    negatives_below = 0
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        group_positive = int(ordered_targets[start:stop].sum())
        group_negative = (stop - start) - group_positive
        favorable_pairs += group_positive * (
            negatives_below + 0.5 * group_negative
        )
        negatives_below += group_negative
        start = stop
    return float(favorable_pairs / (positive_count * negative_count))


def binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> BinaryMetrics:
    """Compute all frozen metrics for one query population."""

    targets = _strict_binary(labels, name="labels")
    predicted = _strict_binary(predictions, name="predictions")
    values = np.asarray(scores, dtype=np.float64)
    if predicted.shape != targets.shape or values.shape != targets.shape:
        raise ValueError("labels, predictions, and scores must have identical shape")
    if len(targets) == 0 or not np.isfinite(values).all():
        raise ValueError("metrics require at least one finite-scored sample")
    if not targets.any() or targets.all():
        raise ValueError(
            "Average Precision, AUROC, and balanced accuracy require both classes"
        )
    true_positive = int(np.sum(targets & predicted))
    false_positive = int(np.sum(~targets & predicted))
    true_negative = int(np.sum(~targets & ~predicted))
    false_negative = int(np.sum(targets & ~predicted))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    negative_denominator = true_negative + false_positive
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = (
        true_negative / negative_denominator if negative_denominator else float("nan")
    )
    balanced = (
        0.5 * (recall + specificity) if np.isfinite(specificity) else float("nan")
    )
    return BinaryMetrics(
        sample_count=len(targets),
        positive_count=int(targets.sum()),
        negative_count=int((~targets).sum()),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        average_precision=average_precision(targets, values),
        auroc=auroc(targets, values),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        balanced_accuracy=float(balanced),
    )

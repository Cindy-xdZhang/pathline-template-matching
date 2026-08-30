"""Frozen, leak-resistant validation primitives for scale-conditioned retrieval.

This module deliberately contains only deterministic transformations of arrays
and immutable inner-validation statistics.  Candidate selection accepts no raw
labels, scores, family identifiers, or outer-fold fields: callers must first
reduce each inner ``dataset x source x block`` group to :class:`InnerGroupMetrics`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Sequence

import numpy as np

from .metrics import BinaryMetrics, binary_metrics


FMT_WIDTH = 161
FMT_LINE_WIDTH = 23
FMT_LINE_COUNT = 7
GRID_SHAPE = (40, 40, 40)
GAUSSIAN_TRUNCATE = 3.0

REPRESENTATION_NAMES = ("fmt161", "real_neighbor36", "chirality_all35")
FROZEN_K_VALUES = (1, 5, 15, 31)
FROZEN_SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
FROZEN_TOP_FRACTIONS = (0.05,)
FROZEN_THRESHOLD_GRID = tuple(index / 100 for index in range(50, 100))

DECISION_FIXED_TOP_FRACTION = "fixed_top_fraction"
DECISION_RANK_THRESHOLD = "rank_threshold"
DECISION_RULES = (DECISION_FIXED_TOP_FRACTION, DECISION_RANK_THRESHOLD)

_REAL_NEIGHBOR_LOCAL_INDICES = (0, 3, 6, 9, 12, 15)
_REPRESENTATION_INDICES = {
    "fmt161": tuple(range(FMT_WIDTH)),
    "real_neighbor36": tuple(
        line * FMT_LINE_WIDTH + local
        for line in range(1, FMT_LINE_COUNT)
        for local in _REAL_NEIGHBOR_LOCAL_INDICES
    ),
    "chirality_all35": tuple(
        line * FMT_LINE_WIDTH + local
        for line in range(FMT_LINE_COUNT)
        for local in range(18, 23)
    ),
}


def _real_vector(
    values: object,
    *,
    name: str,
    require_finite: bool = True,
) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional real numeric array") from error
    if (
        array.ndim != 1
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a one-dimensional real numeric array")
    result = np.asarray(array, dtype=np.float64)
    if require_finite and not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _strict_bool_vector(values: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional boolean array") from error
    if array.ndim != 1 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a one-dimensional boolean array")
    return np.asarray(array, dtype=bool)


def _center_indices(values: object, *, expected_length: int) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError("center_indices must be a one-dimensional integer array") from error
    if (
        array.ndim != 1
        or not np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError("center_indices must be a one-dimensional integer array")
    if np.issubdtype(array.dtype, np.unsignedinteger) and array.size:
        if int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError("center_indices contains an integer outside int64 range")
    result = np.asarray(array, dtype=np.int64)
    if len(result) != expected_length:
        raise ValueError("center_indices must contain one index per row")
    total_size = math.prod(GRID_SHAPE)
    if np.any(result < 0) or np.any(result >= total_size):
        raise ValueError("center_indices contains an index outside the frozen 40^3 grid")
    if len(np.unique(result)) != len(result):
        raise ValueError("center_indices must not contain duplicates within a group")
    return result


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(operator.index(value))  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(operator.index(value))  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _nonempty_name(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty stripped string")
    return value


def representation_indices(name: str) -> tuple[int, ...]:
    """Return the immutable frozen column indices for one FMT representation."""

    if not isinstance(name, str) or name not in _REPRESENTATION_INDICES:
        raise ValueError(
            "representation must be one of " + ", ".join(REPRESENTATION_NAMES)
        )
    return _REPRESENTATION_INDICES[name]


def select_representation(features: object, representation: str) -> np.ndarray:
    """Validate an ``N x 161`` FMT matrix and copy its frozen representation."""

    try:
        array = np.asarray(features)
    except (TypeError, ValueError) as error:
        raise ValueError("features must be a two-dimensional real N x 161 array") from error
    if (
        array.ndim != 2
        or array.shape[1] != FMT_WIDTH
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError("features must be a two-dimensional real N x 161 array")
    if not np.isfinite(array).all():
        raise ValueError("features contains NaN or Inf")
    selected = np.asarray(array[:, representation_indices(representation)], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("features contains a value outside the finite float32 range")
    return np.ascontiguousarray(selected).copy()


def representation_features(features: object, name: str) -> np.ndarray:
    """Stable runner-facing alias for :func:`select_representation`."""

    return select_representation(features, name)


def supported_rank_scores(
    raw_distances: object,
    center_indices: object,
    supported: object,
) -> np.ndarray:
    """Rank only supported distances and retain every query row in the output.

    Supported rows are ordered by ascending distance and then ascending center
    index.  They receive ranks ``1/M, ..., M/M``.  Unsupported rows receive
    exactly zero; a NaN distance is permitted only as an unsupported sentinel.
    """

    distances = _real_vector(
        raw_distances, name="raw_distances", require_finite=False
    )
    mask = _strict_bool_vector(supported, name="supported")
    if mask.shape != distances.shape:
        raise ValueError("supported must contain one value per raw distance")
    centers = _center_indices(center_indices, expected_length=len(distances))
    if np.isinf(distances).any():
        raise ValueError("raw_distances contains Inf")
    if np.any(np.isfinite(distances) & (distances < 0.0)):
        raise ValueError("raw_distances cannot contain a finite negative distance")
    if not np.isfinite(distances[mask]).all():
        raise ValueError("every supported raw distance must be finite")

    result = np.zeros(len(distances), dtype=np.float64)
    supported_rows = np.flatnonzero(mask)
    if len(supported_rows) == 0:
        return result
    order_within_supported = np.lexsort(
        (centers[supported_rows], distances[supported_rows])
    )
    ordered_rows = supported_rows[order_within_supported]
    result[ordered_rows] = (
        np.arange(1, len(ordered_rows) + 1, dtype=np.float64)
        / float(len(ordered_rows))
    )
    return result


def _gaussian_kernel(sigma: float, truncate: float) -> np.ndarray:
    radius = int(truncate * sigma + 0.5)
    if radius == 0:
        return np.ones(1, dtype=np.float64)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    return kernel / kernel.sum()


def _convolve_axis_constant_zero(
    values: np.ndarray, kernel: np.ndarray, axis: int
) -> np.ndarray:
    radius = len(kernel) // 2
    if radius == 0:
        return values.copy()
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="constant", constant_values=0.0)
    result = np.zeros_like(values, dtype=np.float64)
    for kernel_index, weight in enumerate(kernel):
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(kernel_index, kernel_index + values.shape[axis])
        result += float(weight) * padded[tuple(slices)]
    return result


def _separable_gaussian(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    for axis in range(3):
        result = _convolve_axis_constant_zero(result, kernel, axis)
    return result


def _validated_grid_shape(grid_shape: Sequence[int]) -> tuple[int, int, int]:
    try:
        items = tuple(grid_shape)
    except TypeError as error:
        raise ValueError("grid_shape must equal the frozen (40, 40, 40)") from error
    if len(items) != 3:
        raise ValueError("grid_shape must equal the frozen (40, 40, 40)")
    validated = tuple(
        _positive_integer(item, name=f"grid_shape[{axis}]")
        for axis, item in enumerate(items)
    )
    if validated != GRID_SHAPE:
        raise ValueError("grid_shape must equal the frozen (40, 40, 40)")
    return validated  # type: ignore[return-value]


def _spatial_from_ranks(
    values: np.ndarray,
    centers: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float,
    grid_shape: Sequence[int],
    truncate: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = _validated_grid_shape(grid_shape)
    truncation = _finite_scalar(truncate, name="truncate")
    if truncation != GAUSSIAN_TRUNCATE:
        raise ValueError(f"truncate must equal the frozen {GAUSSIAN_TRUNCATE}")
    if sigma == 0.0 or len(values) == 0:
        return values.copy(), mask.astype(np.float64)

    dense_scores = np.zeros(shape, dtype=np.float64)
    dense_mask = np.zeros(shape, dtype=np.float64)
    supported_centers = centers[mask]
    dense_scores.ravel(order="C")[supported_centers] = values[mask]
    dense_mask.ravel(order="C")[supported_centers] = 1.0
    kernel = _gaussian_kernel(sigma, truncation)
    numerator = _separable_gaussian(dense_scores, kernel)
    denominator = _separable_gaussian(dense_mask, kernel)
    selected_numerator = numerator.ravel(order="C")[centers]
    selected_denominator = denominator.ravel(order="C")[centers]
    result = np.zeros(len(values), dtype=np.float64)
    has_support = selected_denominator > 0.0
    result[has_support] = (
        selected_numerator[has_support] / selected_denominator[has_support]
    )
    if not np.isfinite(result).all() or not np.isfinite(selected_denominator).all():
        raise RuntimeError("mask-normalized Gaussian smoothing produced NaN or Inf")
    return result, selected_denominator


@dataclass(frozen=True, slots=True)
class SpatialSupportScores:
    """Scores plus an auditable supported/imputed/unimputable partition."""

    scores: np.ndarray
    denominator: np.ndarray
    supported_mask: np.ndarray
    imputed_mask: np.ndarray
    unimputable_mask: np.ndarray

    def __post_init__(self) -> None:
        scores = _real_vector(self.scores, name="scores")
        denominator = _real_vector(self.denominator, name="denominator")
        supported = _strict_bool_vector(self.supported_mask, name="supported_mask")
        imputed = _strict_bool_vector(self.imputed_mask, name="imputed_mask")
        unimputable = _strict_bool_vector(
            self.unimputable_mask, name="unimputable_mask"
        )
        shape = scores.shape
        if any(array.shape != shape for array in (denominator, supported, imputed, unimputable)):
            raise ValueError("all SpatialSupportScores arrays must have identical shape")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("scores must lie in [0, 1]")
        if np.any(denominator < 0.0):
            raise ValueError("denominator cannot be negative")
        category_count = (
            supported.astype(np.int8)
            + imputed.astype(np.int8)
            + unimputable.astype(np.int8)
        )
        if np.any(category_count != 1):
            raise ValueError(
                "supported, imputed, and unimputable masks must partition all rows"
            )
        if np.any(imputed != ((~supported) & (denominator > 0.0))):
            raise ValueError("imputed_mask must identify unsupported rows with support")
        if np.any(unimputable != ((~supported) & (denominator == 0.0))):
            raise ValueError(
                "unimputable_mask must identify unsupported rows without support"
            )
        if np.any(scores[unimputable] != 0.0):
            raise ValueError("unimputable rows must have score zero")
        for field, array in (
            ("scores", scores),
            ("denominator", denominator),
            ("supported_mask", supported),
            ("imputed_mask", imputed),
            ("unimputable_mask", unimputable),
        ):
            frozen = np.ascontiguousarray(array).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, field, frozen)


def spatial_support_scores(
    raw_distances: object,
    center_indices: object,
    support_mask: object,
    *,
    grid_shape: Sequence[int] = GRID_SHAPE,
    sigma: float,
    truncate: float = GAUSSIAN_TRUNCATE,
) -> SpatialSupportScores:
    """Rank and spatially process one complete source x block query grid.

    The function is transductive: a score can depend on the other valid centers
    in the same complete source x block grid, so it is not a per-primitive
    classifier.  The support mask is supplied independently for each fitted
    ``k`` model.  Unsupported rows remain present and are classified as either
    spatially imputed or unimputable.
    """

    distances = _real_vector(
        raw_distances, name="raw_distances", require_finite=False
    )
    mask = _strict_bool_vector(support_mask, name="support_mask")
    if mask.shape != distances.shape:
        raise ValueError("support_mask must contain one value per raw distance")
    centers = _center_indices(center_indices, expected_length=len(distances))
    width = _finite_scalar(sigma, name="sigma")
    if width not in FROZEN_SIGMAS:
        raise ValueError(f"sigma must be one of {FROZEN_SIGMAS}")
    ranks = supported_rank_scores(distances, centers, mask)
    scores, denominator = _spatial_from_ranks(
        ranks,
        centers,
        mask,
        sigma=width,
        grid_shape=grid_shape,
        truncate=truncate,
    )
    imputed = (~mask) & (denominator > 0.0)
    unimputable = (~mask) & (denominator == 0.0)
    return SpatialSupportScores(
        scores=scores,
        denominator=denominator,
        supported_mask=mask,
        imputed_mask=imputed,
        unimputable_mask=unimputable,
    )


def spatial_rank_scores(
    ranks: object,
    center_indices: object,
    supported: object,
    *,
    sigma: float,
) -> np.ndarray:
    """Apply frozen mask-normalized Gaussian smoothing to supported ranks.

    At ``sigma=0`` the rank vector is returned unchanged.  At positive sigma,
    supported ranks alone form the numerator and supported locations alone form
    the mask denominator.  Values are returned for every valid query center;
    an output center with zero Gaussian support receives zero.
    """

    values = _real_vector(ranks, name="ranks")
    mask = _strict_bool_vector(supported, name="supported")
    if mask.shape != values.shape:
        raise ValueError("supported must contain one value per rank")
    centers = _center_indices(center_indices, expected_length=len(values))
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("ranks must lie in [0, 1]")
    if np.any(values[~mask] != 0.0):
        raise ValueError("unsupported rank rows must be exactly zero")
    width = _finite_scalar(sigma, name="sigma")
    if width not in FROZEN_SIGMAS:
        raise ValueError(f"sigma must be one of {FROZEN_SIGMAS}")
    if width == 0.0 or len(values) == 0:
        return values.copy()

    result, _ = _spatial_from_ranks(
        values,
        centers,
        mask,
        sigma=width,
        grid_shape=GRID_SHAPE,
        truncate=GAUSSIAN_TRUNCATE,
    )
    return result


def fixed_top_fraction_predictions(
    scores: object,
    center_indices: object,
    eligible_mask: object,
    *,
    fraction: float,
) -> np.ndarray:
    """Predict up to ``ceil(fraction * N_all)`` eligible positive-score rows.

    Candidate count is based on all valid query rows, then capped by rows whose
    score source is eligible.  A zero score is always rejected, even if a
    caller mistakenly marks that row eligible.
    """

    values = _real_vector(scores, name="scores")
    centers = _center_indices(center_indices, expected_length=len(values))
    eligible = _strict_bool_vector(eligible_mask, name="eligible_mask")
    if eligible.shape != values.shape:
        raise ValueError("eligible_mask must contain one value per score")
    selected_fraction = _finite_scalar(fraction, name="fraction")
    if selected_fraction < 0.0 or selected_fraction > 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    target_count = int(math.ceil(selected_fraction * len(values)))
    eligible_rows = np.flatnonzero(eligible & (values > 0.0))
    count = min(target_count, len(eligible_rows))
    predictions = np.zeros(len(values), dtype=bool)
    if count == 0:
        return predictions
    order_within_eligible = np.lexsort(
        (centers[eligible_rows], -values[eligible_rows])
    )
    predictions[eligible_rows[order_within_eligible[:count]]] = True
    return predictions


def threshold_predictions(
    scores: object,
    eligible_mask: object,
    *,
    threshold: float,
) -> np.ndarray:
    """Threshold eligible rows; ineligible rows are always predicted negative."""

    values = _real_vector(scores, name="scores")
    eligible = _strict_bool_vector(eligible_mask, name="eligible_mask")
    if eligible.shape != values.shape:
        raise ValueError("eligible_mask must contain one value per score")
    selected_threshold = _finite_scalar(threshold, name="threshold")
    if selected_threshold not in FROZEN_THRESHOLD_GRID:
        raise ValueError("threshold must be an exact member of FROZEN_THRESHOLD_GRID")
    return eligible & (values >= selected_threshold)


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One immutable member of the frozen inner-validation candidate set."""

    representation: str
    k: int
    sigma: float
    decision_rule: str
    decision_value: float

    def __post_init__(self) -> None:
        representation_indices(self.representation)
        neighbour_count = _positive_integer(self.k, name="k")
        if neighbour_count not in FROZEN_K_VALUES:
            raise ValueError(f"k must be one of {FROZEN_K_VALUES}")
        width = _finite_scalar(self.sigma, name="sigma")
        if width not in FROZEN_SIGMAS:
            raise ValueError(f"sigma must be one of {FROZEN_SIGMAS}")
        if self.decision_rule not in DECISION_RULES:
            raise ValueError(f"decision_rule must be one of {DECISION_RULES}")
        value = _finite_scalar(self.decision_value, name="decision_value")
        if self.decision_rule == DECISION_FIXED_TOP_FRACTION:
            if value not in FROZEN_TOP_FRACTIONS:
                raise ValueError(
                    "fixed-top decision_value must be one of "
                    f"{FROZEN_TOP_FRACTIONS}"
                )
        elif value not in FROZEN_THRESHOLD_GRID:
            raise ValueError(
                "rank-threshold decision_value must be an exact member of "
                "FROZEN_THRESHOLD_GRID"
            )

    @property
    def candidate_id(self) -> str:
        """Stable, locale-independent identifier used for the final tie-break."""

        return (
            f"{self.representation}__k={self.k:02d}__sigma={self.sigma:.2f}__"
            f"rule={self.decision_rule}__value={self.decision_value:.2f}"
        )


def candidate_predictions(
    candidate: CandidateSpec,
    scores: object,
    center_indices: object,
    eligible_mask: object,
) -> np.ndarray:
    """Apply only the frozen decision rule encoded by ``candidate``."""

    if not isinstance(candidate, CandidateSpec):
        raise TypeError("candidate must be a CandidateSpec")
    if candidate.decision_rule == DECISION_FIXED_TOP_FRACTION:
        return fixed_top_fraction_predictions(
            scores,
            center_indices,
            eligible_mask,
            fraction=candidate.decision_value,
        )
    # Validate centers even though a threshold itself does not need tie-breaking;
    # every evaluated group must preserve the same row-identity contract.
    values = _real_vector(scores, name="scores")
    _center_indices(center_indices, expected_length=len(values))
    return threshold_predictions(
        values, eligible_mask, threshold=candidate.decision_value
    )


@dataclass(frozen=True, slots=True)
class InnerGroupKey:
    """Identity of one equally weighted inner dataset x source x block group."""

    dataset: str
    source_ordinal: int
    block_id: int

    def __post_init__(self) -> None:
        _nonempty_name(self.dataset, name="dataset")
        _nonnegative_integer(self.source_ordinal, name="source_ordinal")
        _nonnegative_integer(self.block_id, name="block_id")


@dataclass(frozen=True, slots=True)
class InnerGroupMetrics:
    """Frozen metrics for one candidate and one inner validation group."""

    candidate: CandidateSpec
    physical_family: str
    group: InnerGroupKey
    metrics: BinaryMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateSpec):
            raise TypeError("candidate must be a CandidateSpec")
        _nonempty_name(self.physical_family, name="physical_family")
        if not isinstance(self.group, InnerGroupKey):
            raise TypeError("group must be an InnerGroupKey")
        if not isinstance(self.metrics, BinaryMetrics):
            raise TypeError("metrics must be BinaryMetrics")

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "physical_family": self.physical_family,
            "dataset": self.group.dataset,
            "source_ordinal": self.group.source_ordinal,
            "block_id": self.group.block_id,
            **self.metrics.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class InnerCandidateMacro:
    """Equal-group macro metrics used by the label-free selector boundary."""

    candidate: CandidateSpec
    physical_families: tuple[str, ...]
    family_count: int
    group_count: int
    total_sample_count: int
    total_positive_count: int
    total_negative_count: int
    average_precision: float
    auroc: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateSpec):
            raise TypeError("candidate must be a CandidateSpec")
        if (
            not isinstance(self.physical_families, tuple)
            or not self.physical_families
            or any(
                _nonempty_name(item, name="physical_families item") != item
                for item in self.physical_families
            )
            or tuple(sorted(set(self.physical_families))) != self.physical_families
        ):
            raise ValueError(
                "physical_families must be a non-empty sorted unique tuple"
            )
        if _positive_integer(self.family_count, name="family_count") != self.family_count:
            raise ValueError("family_count must be a positive integer")
        if self.family_count != len(self.physical_families):
            raise ValueError("family_count must equal len(physical_families)")
        if _positive_integer(self.group_count, name="group_count") != self.group_count:
            raise ValueError("group_count must be a positive integer")
        for name in (
            "total_sample_count",
            "total_positive_count",
            "total_negative_count",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "average_precision",
            "auroc",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
        ):
            value = _finite_scalar(getattr(self, name), name=name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "candidate_id": self.candidate_id,
            "physical_families": "|".join(self.physical_families),
            "family_count": self.family_count,
            "group_count": self.group_count,
            "total_sample_count": self.total_sample_count,
            "total_positive_count": self.total_positive_count,
            "total_negative_count": self.total_negative_count,
            "average_precision": self.average_precision,
            "auroc": self.auroc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
        }


def evaluate_inner_group(
    candidate: CandidateSpec,
    physical_family: str,
    group: InnerGroupKey,
    inner_labels: object,
    scores: object,
    center_indices: object,
    eligible_mask: object,
) -> InnerGroupMetrics:
    """Reduce one labeled inner group to immutable metrics."""

    if not isinstance(group, InnerGroupKey):
        raise TypeError("group must be an InnerGroupKey")
    _nonempty_name(physical_family, name="physical_family")
    values = _real_vector(scores, name="scores")
    predictions = candidate_predictions(
        candidate, values, center_indices, eligible_mask
    )
    metrics = binary_metrics(
        np.asarray(inner_labels),
        predictions,
        values,
    )
    return InnerGroupMetrics(
        candidate=candidate,
        physical_family=physical_family,
        group=group,
        metrics=metrics,
    )


def aggregate_inner_group_metrics(
    rows: Sequence[InnerGroupMetrics],
) -> InnerCandidateMacro:
    """Compute an arithmetic mean over unique inner groups for one candidate."""

    try:
        items = tuple(rows)
    except TypeError as error:
        raise TypeError("rows must be a sequence of InnerGroupMetrics") from error
    if not items:
        raise ValueError("at least one inner group metric row is required")
    if not all(isinstance(row, InnerGroupMetrics) for row in items):
        raise TypeError("rows must contain only InnerGroupMetrics")
    candidate = items[0].candidate
    if any(row.candidate != candidate for row in items[1:]):
        raise ValueError("all rows must describe the same candidate")
    group_keys = [(row.physical_family, row.group) for row in items]
    if len(set(group_keys)) != len(group_keys):
        raise ValueError("an inner group cannot be counted more than once")

    dataset_families: dict[str, str] = {}
    for row in items:
        previous = dataset_families.setdefault(row.group.dataset, row.physical_family)
        if previous != row.physical_family:
            raise ValueError("one dataset cannot belong to multiple physical families")
    physical_families = tuple(sorted({row.physical_family for row in items}))

    def hierarchical_mean(field: str) -> float:
        family_means = []
        for family in physical_families:
            family_means.append(
                float(
                    np.mean(
                        [
                            getattr(row.metrics, field)
                            for row in items
                            if row.physical_family == family
                        ]
                    )
                )
            )
        return float(np.mean(family_means))

    return InnerCandidateMacro(
        candidate=candidate,
        physical_families=physical_families,
        family_count=len(physical_families),
        group_count=len(items),
        total_sample_count=sum(row.metrics.sample_count for row in items),
        total_positive_count=sum(row.metrics.positive_count for row in items),
        total_negative_count=sum(row.metrics.negative_count for row in items),
        average_precision=hierarchical_mean("average_precision"),
        auroc=hierarchical_mean("auroc"),
        precision=hierarchical_mean("precision"),
        recall=hierarchical_mean("recall"),
        f1=hierarchical_mean("f1"),
        balanced_accuracy=hierarchical_mean("balanced_accuracy"),
    )


def select_inner_candidate(
    inner_candidate_macros: Sequence[InnerCandidateMacro],
) -> InnerCandidateMacro:
    """Select using inner macro metrics only.

    Numerical metrics are maximized in this exact order: F1, Average
    Precision, balanced accuracy, precision, recall.  A remaining tie selects
    the lexicographically smallest stable candidate ID.  AUROC is reported but
    deliberately does not participate in selection.
    """

    try:
        items = tuple(inner_candidate_macros)
    except TypeError as error:
        raise TypeError(
            "inner_candidate_macros must be a sequence of InnerCandidateMacro"
        ) from error
    if not items:
        raise ValueError("at least one inner candidate macro is required")
    if not all(isinstance(item, InnerCandidateMacro) for item in items):
        raise TypeError("selector accepts only InnerCandidateMacro rows")
    candidate_ids = [item.candidate_id for item in items]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique at selection")
    expected_families = items[0].physical_families
    if any(item.physical_families != expected_families for item in items[1:]):
        raise ValueError(
            "every candidate must contain the same complete physical-family set"
        )
    return min(
        items,
        key=lambda item: (
            -item.f1,
            -item.average_precision,
            -item.balanced_accuracy,
            -item.precision,
            -item.recall,
            item.candidate_id,
        ),
    )


# Runner-facing aliases keep selection vocabulary short without widening the
# leak boundary: both selectors still accept only reduced inner statistics.
aggregate_group_metric_rows = aggregate_inner_group_metrics
select_candidate = select_inner_candidate

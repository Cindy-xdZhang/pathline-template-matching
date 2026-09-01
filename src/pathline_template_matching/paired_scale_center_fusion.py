"""Pure paired-scale center fusion and source-centered diagnostics.

The template classifier produces a calibrated spatial score for the valid
legacy and expanded pathline row assigned to each spatial center.  This module
joins those rows without labels, fuses the two scale blocks, makes one fixed
top-fraction prediction per unique center, and projects that prediction back
to the valid rows.

The two direct source-centered diagnostics deliberately live in the same pure
module but return a distinct type.  They never use a template library and must
not be interpreted as template-classifier success.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEFAULT_CENTER_COUNT = 40 * 40 * 40
DEFAULT_TOP_FRACTION = 0.05
BLOCK_COUNT = 2
LEGACY_BLOCK_INDEX = 0
EXPANDED_BLOCK_INDEX = 1
DX_LEVEL_COUNT = 10
SCALES_PER_BLOCK = 1000
SCALES_PER_DX_LEVEL = 100
ASSIGNED_ROWS_PER_BLOCK_DX_LEVEL = 6400


def _freeze(values: object, *, dtype: np.dtype | type) -> np.ndarray:
    selected_dtype = np.dtype(dtype)
    copied = np.array(values, dtype=selected_dtype, order="C", copy=True)
    frozen = np.frombuffer(copied.tobytes(order="C"), dtype=selected_dtype).reshape(
        copied.shape
    )
    frozen.setflags(write=False)
    return frozen


def _integer_vector(
    values: object,
    *,
    name: str,
    dtype: np.dtype | type,
    count: int | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must be a one-dimensional {np.dtype(dtype)} array")
    if count is not None and len(raw) != count:
        raise ValueError(f"{name} must contain exactly {count} rows")
    return raw


def _float_vector(
    values: object,
    *,
    name: str,
    dtype: np.dtype | type | None = None,
    count: int | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind != "f":
        raise ValueError(f"{name} must be a one-dimensional floating array")
    if dtype is not None and raw.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}")
    if count is not None and len(raw) != count:
        raise ValueError(f"{name} must contain exactly {count} rows")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    return raw


def _bool_vector(values: object, *, name: str, count: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype != np.dtype(np.bool_) or raw.shape != (count,):
        raise ValueError(f"{name} must be a boolean vector with shape ({count},)")
    return raw


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    selected = int(value)
    if selected < 1:
        raise ValueError(f"{name} must be a positive integer")
    return selected


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite fraction")
    selected = float(value)
    if not np.isfinite(selected) or selected < 0.0 or selected > 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return selected


def fixed_top_fraction_over_centers(
    scores: object,
    eligible: object,
    *,
    fraction: float,
    require_strictly_positive_score: bool,
) -> np.ndarray:
    """Return deterministic center predictions.

    The target count is ``ceil(fraction * all_centers)`` and is capped by the
    eligible population.  Ties are broken by ascending center index.  Template
    fusion sets ``require_strictly_positive_score``; the direct diagnostics do
    not silently lose rows merely because a physical field happens to be zero.
    """

    values = _float_vector(scores, name="scores")
    allowed = _bool_vector(eligible, name="eligible", count=len(values))
    selected_fraction = _fraction(fraction, name="fraction")
    usable = allowed & (values > 0.0) if require_strictly_positive_score else allowed
    usable_rows = np.flatnonzero(usable)
    target = min(int(math.ceil(selected_fraction * len(values))), len(usable_rows))
    prediction = np.zeros(len(values), dtype=np.bool_)
    if target:
        order = np.lexsort((usable_rows, -values[usable_rows]))
        prediction[usable_rows[order[:target]]] = True
    return prediction


@dataclass(frozen=True)
class PairedCenterFusion:
    """Immutable label-free paired-center prediction and row projection."""

    center_seed_index: np.ndarray
    legacy_score: np.ndarray
    expanded_score: np.ndarray
    paired_score: np.ndarray
    legacy_valid: np.ndarray
    expanded_valid: np.ndarray
    legacy_eligible: np.ndarray
    expanded_eligible: np.ndarray
    combined_eligible: np.ndarray
    prediction: np.ndarray
    valid_row_prediction: np.ndarray
    valid_row_center_seed_index: np.ndarray
    valid_row_scale_block_index: np.ndarray
    weight: float
    top_fraction: float

    def __post_init__(self) -> None:
        centers = np.asarray(self.center_seed_index)
        if centers.dtype != np.dtype(np.int64) or centers.ndim != 1:
            raise ValueError("center_seed_index must be a one-dimensional int64 array")
        count = len(centers)
        if not np.array_equal(centers, np.arange(count, dtype=np.int64)):
            raise ValueError("center_seed_index must be the complete contiguous center grid")
        arrays: dict[str, np.ndarray] = {}
        for name in ("legacy_score", "expanded_score", "paired_score"):
            values = _float_vector(getattr(self, name), name=name, count=count)
            arrays[name] = _freeze(values, dtype=np.float64)
        for name in (
            "legacy_valid",
            "expanded_valid",
            "legacy_eligible",
            "expanded_eligible",
            "combined_eligible",
            "prediction",
        ):
            values = _bool_vector(getattr(self, name), name=name, count=count)
            arrays[name] = _freeze(values, dtype=np.bool_)
        row_centers = _integer_vector(
            self.valid_row_center_seed_index,
            name="valid_row_center_seed_index",
            dtype=np.int64,
        )
        row_blocks = _integer_vector(
            self.valid_row_scale_block_index,
            name="valid_row_scale_block_index",
            dtype=np.int8,
            count=len(row_centers),
        )
        row_prediction = _bool_vector(
            self.valid_row_prediction,
            name="valid_row_prediction",
            count=len(row_centers),
        )
        if np.any((row_centers < 0) | (row_centers >= count)):
            raise ValueError("valid-row center is outside the complete center grid")
        if np.any((row_blocks < 0) | (row_blocks >= BLOCK_COUNT)):
            raise ValueError("valid-row scale block must contain only 0 or 1")
        pair_key = row_blocks.astype(np.int64) * count + row_centers
        if len(np.unique(pair_key)) != len(pair_key):
            raise ValueError("valid rows contain a duplicate block/center pair")
        expected_valid = np.zeros((BLOCK_COUNT, count), dtype=np.bool_)
        expected_valid[row_blocks, row_centers] = True
        if not np.array_equal(
            arrays["legacy_valid"], expected_valid[LEGACY_BLOCK_INDEX]
        ):
            raise ValueError("legacy_valid does not reproduce valid-row identities")
        if not np.array_equal(
            arrays["expanded_valid"], expected_valid[EXPANDED_BLOCK_INDEX]
        ):
            raise ValueError("expanded_valid does not reproduce valid-row identities")
        if not np.array_equal(row_prediction, arrays["prediction"][row_centers]):
            raise ValueError("valid_row_prediction is not the exact center projection")
        if np.any(arrays["legacy_eligible"] & ~arrays["legacy_valid"]):
            raise ValueError("legacy eligibility exceeds legacy valid coverage")
        if np.any(arrays["expanded_eligible"] & ~arrays["expanded_valid"]):
            raise ValueError("expanded eligibility exceeds expanded valid coverage")
        if np.any(arrays["legacy_eligible"] & (arrays["legacy_score"] <= 0.0)):
            raise ValueError("legacy eligibility requires a strictly positive score")
        if np.any(arrays["expanded_eligible"] & (arrays["expanded_score"] <= 0.0)):
            raise ValueError("expanded eligibility requires a strictly positive score")
        if np.any(arrays["legacy_score"][~arrays["legacy_valid"]] != 0.0):
            raise ValueError("missing legacy centers must use the zero score sentinel")
        if np.any(arrays["expanded_score"][~arrays["expanded_valid"]] != 0.0):
            raise ValueError("missing expanded centers must use the zero score sentinel")
        if not np.array_equal(
            arrays["combined_eligible"],
            arrays["legacy_eligible"] | arrays["expanded_eligible"],
        ):
            raise ValueError("combined eligibility is not the union of block eligibility")
        selected_weight = float(self.weight)
        selected_fraction = _fraction(self.top_fraction, name="top_fraction")
        if not np.isfinite(selected_weight) or selected_weight < 0.0 or selected_weight > 1.0:
            raise ValueError("weight must lie in [0,1]")
        both_eligible = arrays["legacy_eligible"] & arrays["expanded_eligible"]
        legacy_only_eligible = arrays["legacy_eligible"] & ~arrays["expanded_eligible"]
        expanded_only_eligible = ~arrays["legacy_eligible"] & arrays["expanded_eligible"]
        expected_score = np.zeros(count, dtype=np.float64)
        expected_score[both_eligible] = (
            selected_weight * arrays["legacy_score"][both_eligible]
            + (1.0 - selected_weight)
            * arrays["expanded_score"][both_eligible]
        )
        expected_score[legacy_only_eligible] = arrays["legacy_score"][
            legacy_only_eligible
        ]
        expected_score[expanded_only_eligible] = arrays["expanded_score"][
            expanded_only_eligible
        ]
        if not np.array_equal(arrays["paired_score"], expected_score):
            raise ValueError("paired_score does not reproduce the frozen fusion rule")
        expected_prediction = fixed_top_fraction_over_centers(
            expected_score,
            arrays["combined_eligible"],
            fraction=selected_fraction,
            require_strictly_positive_score=True,
        )
        if not np.array_equal(arrays["prediction"], expected_prediction):
            raise ValueError("prediction does not reproduce the frozen top-fraction rule")
        object.__setattr__(self, "center_seed_index", _freeze(centers, dtype=np.int64))
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(
            self, "valid_row_center_seed_index", _freeze(row_centers, dtype=np.int64)
        )
        object.__setattr__(
            self, "valid_row_scale_block_index", _freeze(row_blocks, dtype=np.int8)
        )
        object.__setattr__(
            self, "valid_row_prediction", _freeze(row_prediction, dtype=np.bool_)
        )
        object.__setattr__(self, "weight", selected_weight)
        object.__setattr__(self, "top_fraction", selected_fraction)

    @property
    def both_valid(self) -> np.ndarray:
        return self.legacy_valid & self.expanded_valid

    @property
    def legacy_only(self) -> np.ndarray:
        return self.legacy_valid & ~self.expanded_valid

    @property
    def expanded_only(self) -> np.ndarray:
        return ~self.legacy_valid & self.expanded_valid

    @property
    def neither_valid(self) -> np.ndarray:
        return ~self.legacy_valid & ~self.expanded_valid

    @property
    def combined_coverage(self) -> float:
        return float((self.legacy_valid | self.expanded_valid).mean())


def fuse_paired_scale_centers(
    valid_row_center_seed_index: object,
    valid_row_scale_block_index: object,
    valid_row_spatial_score: object,
    valid_row_score_supported: object,
    *,
    weight: float,
    center_count: int = DEFAULT_CENTER_COUNT,
    top_fraction: float = DEFAULT_TOP_FRACTION,
) -> PairedCenterFusion:
    """Fuse zero or one valid row from each scale block at every center."""

    total_centers = _positive_integer(center_count, name="center_count")
    centers = _integer_vector(
        valid_row_center_seed_index,
        name="valid_row_center_seed_index",
        dtype=np.int64,
    )
    blocks = _integer_vector(
        valid_row_scale_block_index,
        name="valid_row_scale_block_index",
        dtype=np.int8,
        count=len(centers),
    )
    scores = _float_vector(
        valid_row_spatial_score,
        name="valid_row_spatial_score",
        count=len(centers),
    )
    supported = _bool_vector(
        valid_row_score_supported,
        name="valid_row_score_supported",
        count=len(centers),
    )
    if np.any((centers < 0) | (centers >= total_centers)):
        raise ValueError("valid-row center is outside the complete center grid")
    if np.any((blocks < 0) | (blocks >= BLOCK_COUNT)):
        raise ValueError("valid-row scale block must contain only 0 or 1")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("valid-row spatial score must lie in [0,1]")
    pair_key = blocks.astype(np.int64) * total_centers + centers
    if len(np.unique(pair_key)) != len(pair_key):
        raise ValueError("each scale block may contribute at most one valid row per center")
    selected_weight = float(weight)
    if not np.isfinite(selected_weight) or selected_weight < 0.0 or selected_weight > 1.0:
        raise ValueError("weight must lie in [0,1]")
    selected_fraction = _fraction(top_fraction, name="top_fraction")

    block_score = np.zeros((BLOCK_COUNT, total_centers), dtype=np.float64)
    block_valid = np.zeros((BLOCK_COUNT, total_centers), dtype=np.bool_)
    block_eligible = np.zeros((BLOCK_COUNT, total_centers), dtype=np.bool_)
    block_score[blocks, centers] = scores
    block_valid[blocks, centers] = True
    block_eligible[blocks, centers] = supported & (scores > 0.0)

    legacy_eligible = block_eligible[LEGACY_BLOCK_INDEX]
    expanded_eligible = block_eligible[EXPANDED_BLOCK_INDEX]
    both_eligible = legacy_eligible & expanded_eligible
    legacy_only_eligible = legacy_eligible & ~expanded_eligible
    expanded_only_eligible = ~legacy_eligible & expanded_eligible
    paired = np.zeros(total_centers, dtype=np.float64)
    paired[both_eligible] = (
        selected_weight * block_score[LEGACY_BLOCK_INDEX, both_eligible]
        + (1.0 - selected_weight)
        * block_score[EXPANDED_BLOCK_INDEX, both_eligible]
    )
    paired[legacy_only_eligible] = block_score[
        LEGACY_BLOCK_INDEX, legacy_only_eligible
    ]
    paired[expanded_only_eligible] = block_score[
        EXPANDED_BLOCK_INDEX, expanded_only_eligible
    ]
    combined_eligible = legacy_eligible | expanded_eligible
    prediction = fixed_top_fraction_over_centers(
        paired,
        combined_eligible,
        fraction=selected_fraction,
        require_strictly_positive_score=True,
    )
    return PairedCenterFusion(
        center_seed_index=np.arange(total_centers, dtype=np.int64),
        legacy_score=block_score[LEGACY_BLOCK_INDEX],
        expanded_score=block_score[EXPANDED_BLOCK_INDEX],
        paired_score=paired,
        legacy_valid=block_valid[LEGACY_BLOCK_INDEX],
        expanded_valid=block_valid[EXPANDED_BLOCK_INDEX],
        legacy_eligible=legacy_eligible,
        expanded_eligible=expanded_eligible,
        combined_eligible=combined_eligible,
        prediction=prediction,
        valid_row_prediction=prediction[centers],
        valid_row_center_seed_index=centers,
        valid_row_scale_block_index=blocks,
        weight=selected_weight,
        top_fraction=selected_fraction,
    )


def separate_block_center_predictions(
    fusion: PairedCenterFusion,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-top predictions for the two unfused mechanism arms."""

    if not isinstance(fusion, PairedCenterFusion):
        raise ValueError("fusion must be a PairedCenterFusion")
    legacy = fixed_top_fraction_over_centers(
        fusion.legacy_score,
        fusion.legacy_eligible,
        fraction=fusion.top_fraction,
        require_strictly_positive_score=True,
    )
    expanded = fixed_top_fraction_over_centers(
        fusion.expanded_score,
        fusion.expanded_eligible,
        fraction=fusion.top_fraction,
        require_strictly_positive_score=True,
    )
    return _freeze(legacy, dtype=np.bool_), _freeze(expanded, dtype=np.bool_)


def empirical_midrank(values: object) -> np.ndarray:
    """Return deterministic empirical midranks ``(average_rank - 0.5) / n``."""

    selected = _float_vector(values, name="values")
    if len(selected) == 0:
        raise ValueError("values must not be empty")
    order = np.argsort(selected, kind="mergesort")
    sorted_values = selected[order]
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_values[1:] != sorted_values[:-1]).astype(np.int64)
            + 1,
            np.asarray([len(selected)], dtype=np.int64),
        )
    )
    ranks = np.empty(len(selected), dtype=np.float64)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        # One-based average rank, then empirical midpoint normalization.
        average_rank = 0.5 * (float(start + 1) + float(stop))
        ranks[order[start:stop]] = (average_rank - 0.5) / float(len(selected))
    return _freeze(ranks, dtype=np.float64)


@dataclass(frozen=True)
class DirectSourceCenteredDiagnostics:
    """Two full-center, label-free direct-kinematic diagnostic predictions."""

    center_seed_index: np.ndarray
    min_dx_centered_curl_score: np.ndarray
    min_dx_prediction: np.ndarray
    dx_rank_mean_score: np.ndarray
    dx_rank_mean_prediction: np.ndarray
    top_fraction: float

    def __post_init__(self) -> None:
        centers = np.asarray(self.center_seed_index)
        if centers.dtype != np.dtype(np.int64) or centers.ndim != 1:
            raise ValueError("center_seed_index must be a one-dimensional int64 array")
        count = len(centers)
        if not np.array_equal(centers, np.arange(count, dtype=np.int64)):
            raise ValueError("center_seed_index must be the complete contiguous center grid")
        for name in ("min_dx_centered_curl_score", "dx_rank_mean_score"):
            values = _float_vector(getattr(self, name), name=name, count=count)
            object.__setattr__(self, name, _freeze(values, dtype=np.float64))
        for name in ("min_dx_prediction", "dx_rank_mean_prediction"):
            values = _bool_vector(getattr(self, name), name=name, count=count)
            object.__setattr__(self, name, _freeze(values, dtype=np.bool_))
        object.__setattr__(self, "center_seed_index", _freeze(centers, dtype=np.int64))
        object.__setattr__(
            self, "top_fraction", _fraction(self.top_fraction, name="top_fraction")
        )
        eligible = np.ones(count, dtype=np.bool_)
        expected_min_dx = fixed_top_fraction_over_centers(
            self.min_dx_centered_curl_score,
            eligible,
            fraction=self.top_fraction,
            require_strictly_positive_score=False,
        )
        expected_rank = fixed_top_fraction_over_centers(
            self.dx_rank_mean_score,
            eligible,
            fraction=self.top_fraction,
            require_strictly_positive_score=False,
        )
        if not np.array_equal(self.min_dx_prediction, expected_min_dx):
            raise ValueError("min-dx prediction does not reproduce its score")
        if not np.array_equal(self.dx_rank_mean_prediction, expected_rank):
            raise ValueError("dx-rank prediction does not reproduce its score")


def direct_source_centered_diagnostics(
    assigned_center_seed_index: object,
    assigned_scale_block_index: object,
    assigned_scale_id: object,
    assigned_physical_dx: object,
    assigned_source_centered_seed4: object,
    *,
    center_count: int = DEFAULT_CENTER_COUNT,
    top_fraction: float = DEFAULT_TOP_FRACTION,
) -> DirectSourceCenteredDiagnostics:
    """Compute the two frozen direct diagnostics from all assigned rows.

    Each block must contain exactly one assigned row per center.  Empirical
    midranks are computed separately for each block and exact dx level; every
    such group must contain 6,400 rows under the frozen 2,000-scale assignment.
    """

    total_centers = _positive_integer(center_count, name="center_count")
    expected_rows = BLOCK_COUNT * total_centers
    centers = _integer_vector(
        assigned_center_seed_index,
        name="assigned_center_seed_index",
        dtype=np.int64,
        count=expected_rows,
    )
    blocks = _integer_vector(
        assigned_scale_block_index,
        name="assigned_scale_block_index",
        dtype=np.int8,
        count=expected_rows,
    )
    scales = _integer_vector(
        assigned_scale_id,
        name="assigned_scale_id",
        dtype=np.int32,
        count=expected_rows,
    )
    dx = _float_vector(
        assigned_physical_dx,
        name="assigned_physical_dx",
        dtype=np.float64,
        count=expected_rows,
    )
    feature = np.asarray(assigned_source_centered_seed4)
    if feature.dtype != np.dtype(np.float32) or feature.shape != (expected_rows, 4):
        raise ValueError(
            "assigned_source_centered_seed4 must have float32 shape "
            f"({expected_rows},4)"
        )
    if not np.isfinite(feature).all() or np.any(feature[:, 0] < 0.0):
        raise ValueError("assigned source-centered feature is nonfinite or has negative curl norm")
    if np.any((centers < 0) | (centers >= total_centers)):
        raise ValueError("assigned center is outside the complete center grid")
    if np.any((blocks < 0) | (blocks >= BLOCK_COUNT)):
        raise ValueError("assigned scale block must contain only 0 or 1")
    if np.any((scales < 0) | (scales >= BLOCK_COUNT * SCALES_PER_BLOCK)):
        raise ValueError("assigned scale ID is outside 0..1999")
    if not np.array_equal(blocks, (scales // SCALES_PER_BLOCK).astype(np.int8)):
        raise ValueError("assigned scale ID and scale block disagree")
    if np.any(dx <= 0.0):
        raise ValueError("assigned physical dx must be strictly positive")
    pair_key = blocks.astype(np.int64) * total_centers + centers
    if not np.array_equal(np.sort(pair_key), np.arange(expected_rows, dtype=np.int64)):
        raise ValueError("assigned rows must cover every block/center pair exactly once")

    curl_by_block = np.empty((BLOCK_COUNT, total_centers), dtype=np.float64)
    dx_by_block = np.empty((BLOCK_COUNT, total_centers), dtype=np.float64)
    rank_by_block = np.empty((BLOCK_COUNT, total_centers), dtype=np.float64)
    dx_level = (scales % SCALES_PER_BLOCK) // SCALES_PER_DX_LEVEL
    for block in range(BLOCK_COUNT):
        selected_block = blocks == block
        curl_by_block[block, centers[selected_block]] = feature[selected_block, 0]
        dx_by_block[block, centers[selected_block]] = dx[selected_block]
        for level in range(DX_LEVEL_COUNT):
            selected = selected_block & (dx_level == level)
            if int(selected.sum()) != ASSIGNED_ROWS_PER_BLOCK_DX_LEVEL:
                raise ValueError(
                    f"block {block} dx level {level} must contain exactly "
                    f"{ASSIGNED_ROWS_PER_BLOCK_DX_LEVEL} assigned rows"
                )
            rank_by_block[block, centers[selected]] = empirical_midrank(
                feature[selected, 0]
            )

    choose_legacy = dx_by_block[LEGACY_BLOCK_INDEX] <= dx_by_block[
        EXPANDED_BLOCK_INDEX
    ]
    min_dx_score = np.where(
        choose_legacy,
        curl_by_block[LEGACY_BLOCK_INDEX],
        curl_by_block[EXPANDED_BLOCK_INDEX],
    )
    rank_mean = 0.5 * (
        rank_by_block[LEGACY_BLOCK_INDEX]
        + rank_by_block[EXPANDED_BLOCK_INDEX]
    )
    eligible = np.ones(total_centers, dtype=np.bool_)
    selected_fraction = _fraction(top_fraction, name="top_fraction")
    return DirectSourceCenteredDiagnostics(
        center_seed_index=np.arange(total_centers, dtype=np.int64),
        min_dx_centered_curl_score=min_dx_score,
        min_dx_prediction=fixed_top_fraction_over_centers(
            min_dx_score,
            eligible,
            fraction=selected_fraction,
            require_strictly_positive_score=False,
        ),
        dx_rank_mean_score=rank_mean,
        dx_rank_mean_prediction=fixed_top_fraction_over_centers(
            rank_mean,
            eligible,
            fraction=selected_fraction,
            require_strictly_positive_score=False,
        ),
        top_fraction=selected_fraction,
    )


__all__ = [
    "ASSIGNED_ROWS_PER_BLOCK_DX_LEVEL",
    "BLOCK_COUNT",
    "DEFAULT_CENTER_COUNT",
    "DEFAULT_TOP_FRACTION",
    "DirectSourceCenteredDiagnostics",
    "PairedCenterFusion",
    "direct_source_centered_diagnostics",
    "empirical_midrank",
    "fixed_top_fraction_over_centers",
    "fuse_paired_scale_centers",
    "separate_block_center_predictions",
]

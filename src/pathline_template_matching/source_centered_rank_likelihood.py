"""Pure source-centered rank likelihood template scoring.

The source-centered curl-deviation magnitude is first converted to an
empirical midrank independently inside every ``scale block x dx level``
assigned population.  The two ranks belonging to one spatial center are then
fused with a fixed paired-scale weight.  Pathline validity is used only to
define the combined-valid center population; it never changes the assigned
rank statistic.

The primary classifier is a family-balanced class-conditional histogram
template model.  Every ``physical family x class`` histogram is normalized
independently with a symmetric beta pseudocount before the family densities
are averaged.  Its directional statistic is

``log p(rank | positive) - log p(rank | negative)``.

Each fit-negative template is calibrated without its complete source: the
histogram contribution of every positive and negative template from that
source is removed before its reference log-likelihood ratio is evaluated.
Query log-likelihood ratios use the complete fit model and are mapped through
each fit family's leave-one-source-out negative reference with a conservative
strict-less empirical CDF.  Family CDFs receive equal weight.  A separate negative-ECDF control
maps the paired rank itself through each family's negative ranks and contains
no class-conditional histogram.

This module deliberately contains no labels-to-threshold selection, spatial
Gaussian processing, top-fraction rule, file I/O, or real-data access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


BLOCK_COUNT = 2
LEGACY_BLOCK_INDEX = 0
EXPANDED_BLOCK_INDEX = 1
DEFAULT_CENTER_COUNT = 40 * 40 * 40
DX_LEVEL_COUNT = 10
SCALES_PER_BLOCK = 1000
SCALES_PER_DX_LEVEL = 100
ASSIGNED_ROWS_PER_BLOCK_DX_LEVEL = DEFAULT_CENTER_COUNT // DX_LEVEL_COUNT
ALLOWED_BIN_COUNTS = (64, 128, 256)
ALLOWED_BETAS = (0.5, 2.0)
CLASS_COUNT = 2
NEGATIVE_CLASS_INDEX = 0
POSITIVE_CLASS_INDEX = 1
_SERIALIZATION_VERSION = 1


def _freeze(values: object, *, dtype: np.dtype | type) -> np.ndarray:
    selected_dtype = np.dtype(dtype)
    copied = np.array(values, dtype=selected_dtype, order="C", copy=True)
    frozen = np.frombuffer(
        copied.tobytes(order="C"), dtype=selected_dtype
    ).reshape(copied.shape)
    frozen.setflags(write=False)
    return frozen


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    selected = int(value)
    if selected < 1:
        raise ValueError(f"{name} must be a positive integer")
    return selected


def _float_vector(
    values: object,
    *,
    name: str,
    dtype: np.dtype | type | None = None,
    count: int | None = None,
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind != "f":
        raise ValueError(f"{name} must be a one-dimensional floating array")
    if dtype is not None and result.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}")
    if count is not None and len(result) != count:
        raise ValueError(f"{name} must contain exactly {count} rows")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _integer_vector(
    values: object,
    *,
    name: str,
    dtype: np.dtype | type,
    count: int | None = None,
) -> np.ndarray:
    result = np.asarray(values)
    selected_dtype = np.dtype(dtype)
    if result.ndim != 1 or result.dtype != selected_dtype:
        raise ValueError(
            f"{name} must be a one-dimensional {selected_dtype} array"
        )
    if count is not None and len(result) != count:
        raise ValueError(f"{name} must contain exactly {count} rows")
    return result


def _bool_vector(values: object, *, name: str, count: int) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype != np.dtype(np.bool_) or result.shape != (count,):
        raise ValueError(f"{name} must be a boolean vector with shape ({count},)")
    return result


def _unit_interval_vector(
    values: object,
    *,
    name: str,
    dtype: np.dtype | type | None = None,
    count: int | None = None,
) -> np.ndarray:
    result = _float_vector(values, name=name, dtype=dtype, count=count)
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _weight(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("weight must be a finite number in [0,1]")
    selected = float(value)
    if not np.isfinite(selected) or selected < 0.0 or selected > 1.0:
        raise ValueError("weight must be a finite number in [0,1]")
    return selected


def _threshold(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("threshold must be a finite number in [0,1]")
    selected = float(value)
    if not np.isfinite(selected) or selected < 0.0 or selected > 1.0:
        raise ValueError("threshold must be a finite number in [0,1]")
    return selected


def empirical_midrank(values: object) -> np.ndarray:
    """Return deterministic midranks ``(average one-based rank - 0.5) / n``."""

    selected = _float_vector(values, name="values")
    if len(selected) == 0:
        raise ValueError("values must not be empty")
    order = np.argsort(selected, kind="mergesort")
    sorted_values = selected[order]
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_values[1:] != sorted_values[:-1]).astype(
                np.int64
            )
            + 1,
            np.asarray([len(selected)], dtype=np.int64),
        )
    )
    ranks = np.empty(len(selected), dtype=np.float64)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        average_one_based_rank = 0.5 * (float(start + 1) + float(stop))
        ranks[order[start:stop]] = (
            average_one_based_rank - 0.5
        ) / float(len(selected))
    return _freeze(ranks, dtype=np.float64)


def assigned_block_dx_midranks(
    assigned_center_seed_index: object,
    assigned_scale_block_index: object,
    assigned_scale_id: object,
    assigned_centered_curl_norm: object,
    *,
    center_count: int = DEFAULT_CENTER_COUNT,
) -> np.ndarray:
    """Rank all assigned curl deviations inside their block/dx population.

    The default production contract requires 128,000 assigned rows and 6,400
    rows in each of the twenty block/dx groups.  ``center_count`` is exposed
    only so the same arithmetic can be tested on small synthetic populations;
    it must remain divisible by ten and each group must contain exactly
    ``center_count / 10`` rows.
    """

    total_centers = _positive_integer(center_count, name="center_count")
    if total_centers % DX_LEVEL_COUNT != 0:
        raise ValueError("center_count must be divisible by ten dx levels")
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
    curl = _float_vector(
        assigned_centered_curl_norm,
        name="assigned_centered_curl_norm",
        count=expected_rows,
    )
    if np.any(curl < 0.0):
        raise ValueError("assigned_centered_curl_norm must be nonnegative")
    if np.any((centers < 0) | (centers >= total_centers)):
        raise ValueError("assigned center is outside the complete center grid")
    if np.any((blocks < 0) | (blocks >= BLOCK_COUNT)):
        raise ValueError("assigned scale block must contain only 0 or 1")
    if np.any((scales < 0) | (scales >= BLOCK_COUNT * SCALES_PER_BLOCK)):
        raise ValueError("assigned scale ID is outside 0..1999")
    if not np.array_equal(
        blocks, (scales // SCALES_PER_BLOCK).astype(np.int8)
    ):
        raise ValueError("assigned scale ID and scale block disagree")
    pair_key = blocks.astype(np.int64) * total_centers + centers
    if not np.array_equal(
        np.sort(pair_key), np.arange(expected_rows, dtype=np.int64)
    ):
        raise ValueError("assigned rows must cover every block/center pair exactly once")

    dx_levels = (scales % SCALES_PER_BLOCK) // SCALES_PER_DX_LEVEL
    expected_group_count = total_centers // DX_LEVEL_COUNT
    result = np.empty(expected_rows, dtype=np.float64)
    for block in range(BLOCK_COUNT):
        for level in range(DX_LEVEL_COUNT):
            selected = (blocks == block) & (dx_levels == level)
            if int(selected.sum()) != expected_group_count:
                raise ValueError(
                    f"block {block} dx level {level} must contain exactly "
                    f"{expected_group_count} assigned rows"
                )
            result[selected] = empirical_midrank(curl[selected])
    return _freeze(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PairedCenterRanks:
    """Immutable paired assigned ranks and parent-valid center mask."""

    center_seed_index: np.ndarray
    legacy_rank: np.ndarray
    expanded_rank: np.ndarray
    paired_rank: np.ndarray
    legacy_valid: np.ndarray
    expanded_valid: np.ndarray
    combined_valid: np.ndarray
    valid_row_center_seed_index: np.ndarray
    valid_row_scale_block_index: np.ndarray
    valid_row_paired_rank: np.ndarray
    weight: float

    def __post_init__(self) -> None:
        centers = np.asarray(self.center_seed_index)
        if centers.dtype != np.dtype(np.int64) or centers.ndim != 1:
            raise ValueError("center_seed_index must be a one-dimensional int64 array")
        count = len(centers)
        if not np.array_equal(centers, np.arange(count, dtype=np.int64)):
            raise ValueError("center_seed_index must be the complete contiguous grid")
        legacy = _unit_interval_vector(
            self.legacy_rank, name="legacy_rank", dtype=np.float64, count=count
        )
        expanded = _unit_interval_vector(
            self.expanded_rank, name="expanded_rank", dtype=np.float64, count=count
        )
        paired = _unit_interval_vector(
            self.paired_rank, name="paired_rank", dtype=np.float64, count=count
        )
        legacy_valid = _bool_vector(
            self.legacy_valid, name="legacy_valid", count=count
        )
        expanded_valid = _bool_vector(
            self.expanded_valid, name="expanded_valid", count=count
        )
        combined = _bool_vector(
            self.combined_valid, name="combined_valid", count=count
        )
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
        row_rank = _unit_interval_vector(
            self.valid_row_paired_rank,
            name="valid_row_paired_rank",
            dtype=np.float64,
            count=len(row_centers),
        )
        if np.any((row_centers < 0) | (row_centers >= count)):
            raise ValueError("valid-row center is outside the complete center grid")
        if np.any((row_blocks < 0) | (row_blocks >= BLOCK_COUNT)):
            raise ValueError("valid-row block must contain only 0 or 1")
        pair_keys = row_blocks.astype(np.int64) * count + row_centers
        if len(np.unique(pair_keys)) != len(pair_keys):
            raise ValueError("valid rows contain a duplicate block/center pair")
        expected_valid = np.zeros((BLOCK_COUNT, count), dtype=np.bool_)
        expected_valid[row_blocks, row_centers] = True
        if not np.array_equal(legacy_valid, expected_valid[LEGACY_BLOCK_INDEX]):
            raise ValueError("legacy_valid does not reproduce valid-row identities")
        if not np.array_equal(expanded_valid, expected_valid[EXPANDED_BLOCK_INDEX]):
            raise ValueError("expanded_valid does not reproduce valid-row identities")
        if not np.array_equal(combined, legacy_valid | expanded_valid):
            raise ValueError("combined_valid is not the union of parent-valid blocks")
        selected_weight = _weight(self.weight)
        expected_paired = (
            selected_weight * legacy + (1.0 - selected_weight) * expanded
        )
        if not np.array_equal(paired, expected_paired):
            raise ValueError("paired_rank does not reproduce the frozen weight")
        if not np.array_equal(row_rank, paired[row_centers]):
            raise ValueError("valid_row_paired_rank is not the exact center projection")
        object.__setattr__(self, "center_seed_index", _freeze(centers, dtype=np.int64))
        object.__setattr__(self, "legacy_rank", _freeze(legacy, dtype=np.float64))
        object.__setattr__(self, "expanded_rank", _freeze(expanded, dtype=np.float64))
        object.__setattr__(self, "paired_rank", _freeze(paired, dtype=np.float64))
        object.__setattr__(
            self, "legacy_valid", _freeze(legacy_valid, dtype=np.bool_)
        )
        object.__setattr__(
            self, "expanded_valid", _freeze(expanded_valid, dtype=np.bool_)
        )
        object.__setattr__(self, "combined_valid", _freeze(combined, dtype=np.bool_))
        object.__setattr__(
            self,
            "valid_row_center_seed_index",
            _freeze(row_centers, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "valid_row_scale_block_index",
            _freeze(row_blocks, dtype=np.int8),
        )
        object.__setattr__(
            self, "valid_row_paired_rank", _freeze(row_rank, dtype=np.float64)
        )
        object.__setattr__(self, "weight", selected_weight)


def pair_assigned_center_ranks(
    assigned_center_seed_index: object,
    assigned_scale_block_index: object,
    assigned_rank: object,
    valid_row_center_seed_index: object,
    valid_row_scale_block_index: object,
    *,
    weight: float,
    center_count: int = DEFAULT_CENTER_COUNT,
) -> PairedCenterRanks:
    """Fuse both assigned ranks, then mark centers with any parent-valid row.

    Both rank values are available from the assigned velocity sidecar even if
    a long pathline later becomes invalid.  Therefore the fixed weight always
    fuses both assigned ranks.  Pathline validity controls only whether a
    center may enter the template library or query population.
    """

    total_centers = _positive_integer(center_count, name="center_count")
    expected_rows = BLOCK_COUNT * total_centers
    assigned_centers = _integer_vector(
        assigned_center_seed_index,
        name="assigned_center_seed_index",
        dtype=np.int64,
        count=expected_rows,
    )
    assigned_blocks = _integer_vector(
        assigned_scale_block_index,
        name="assigned_scale_block_index",
        dtype=np.int8,
        count=expected_rows,
    )
    ranks = _unit_interval_vector(
        assigned_rank, name="assigned_rank", dtype=np.float64, count=expected_rows
    )
    if np.any((assigned_centers < 0) | (assigned_centers >= total_centers)):
        raise ValueError("assigned center is outside the complete center grid")
    if np.any((assigned_blocks < 0) | (assigned_blocks >= BLOCK_COUNT)):
        raise ValueError("assigned block must contain only 0 or 1")
    pair_key = assigned_blocks.astype(np.int64) * total_centers + assigned_centers
    if not np.array_equal(
        np.sort(pair_key), np.arange(expected_rows, dtype=np.int64)
    ):
        raise ValueError("assigned rows must cover every block/center pair exactly once")

    row_centers = _integer_vector(
        valid_row_center_seed_index,
        name="valid_row_center_seed_index",
        dtype=np.int64,
    )
    row_blocks = _integer_vector(
        valid_row_scale_block_index,
        name="valid_row_scale_block_index",
        dtype=np.int8,
        count=len(row_centers),
    )
    if np.any((row_centers < 0) | (row_centers >= total_centers)):
        raise ValueError("valid-row center is outside the complete center grid")
    if np.any((row_blocks < 0) | (row_blocks >= BLOCK_COUNT)):
        raise ValueError("valid-row block must contain only 0 or 1")
    valid_pair_key = row_blocks.astype(np.int64) * total_centers + row_centers
    if len(np.unique(valid_pair_key)) != len(valid_pair_key):
        raise ValueError("valid rows contain a duplicate block/center pair")

    by_block = np.empty((BLOCK_COUNT, total_centers), dtype=np.float64)
    by_block[assigned_blocks, assigned_centers] = ranks
    selected_weight = _weight(weight)
    paired = (
        selected_weight * by_block[LEGACY_BLOCK_INDEX]
        + (1.0 - selected_weight) * by_block[EXPANDED_BLOCK_INDEX]
    )
    block_valid = np.zeros((BLOCK_COUNT, total_centers), dtype=np.bool_)
    block_valid[row_blocks, row_centers] = True
    combined = block_valid[LEGACY_BLOCK_INDEX] | block_valid[EXPANDED_BLOCK_INDEX]
    return PairedCenterRanks(
        center_seed_index=np.arange(total_centers, dtype=np.int64),
        legacy_rank=by_block[LEGACY_BLOCK_INDEX],
        expanded_rank=by_block[EXPANDED_BLOCK_INDEX],
        paired_rank=paired,
        legacy_valid=block_valid[LEGACY_BLOCK_INDEX],
        expanded_valid=block_valid[EXPANDED_BLOCK_INDEX],
        combined_valid=combined,
        valid_row_center_seed_index=row_centers,
        valid_row_scale_block_index=row_blocks,
        valid_row_paired_rank=paired[row_centers],
        weight=selected_weight,
    )


@dataclass(frozen=True, slots=True)
class FamilySourceRankBatch:
    """Combined-valid paired ranks, labels, and complete-source identities."""

    ranks: np.ndarray
    labels: np.ndarray
    source_ids: np.ndarray

    def __post_init__(self) -> None:
        ranks = _unit_interval_vector(
            self.ranks, name="ranks", dtype=np.float64
        )
        if len(ranks) == 0:
            raise ValueError("ranks must not be empty")
        labels = _bool_vector(self.labels, name="labels", count=len(ranks))
        sources = _integer_vector(
            self.source_ids,
            name="source_ids",
            dtype=np.int64,
            count=len(ranks),
        )
        if np.any(sources < 0):
            raise ValueError("source_ids must be nonnegative")
        if len(np.unique(sources)) < 2:
            raise ValueError("each family must contain at least two complete sources")
        if not labels.any() or labels.all():
            raise ValueError("each family must contain both template classes")
        object.__setattr__(self, "ranks", _freeze(ranks, dtype=np.float64))
        object.__setattr__(self, "labels", _freeze(labels, dtype=np.bool_))
        object.__setattr__(self, "source_ids", _freeze(sources, dtype=np.int64))

    @property
    def count(self) -> int:
        return len(self.ranks)


def _bin_indices(ranks: np.ndarray, bin_count: int) -> np.ndarray:
    # The right endpoint belongs to the final bin.
    return np.minimum(
        np.floor(ranks * float(bin_count)).astype(np.int64), bin_count - 1
    )


def _smoothed_histogram_mass(
    counts: np.ndarray, *, beta: float
) -> np.ndarray:
    selected = np.asarray(counts, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0):
        raise ValueError("histogram counts must be a nonnegative vector")
    denominator = float(selected.sum()) + beta * float(len(selected))
    return (selected.astype(np.float64) + beta) / denominator


def conservative_strict_ecdf(reference: object, query: object) -> np.ndarray:
    """Return ``count(reference < query) / (len(reference) + 1)``.

    Equal reference values do not contribute.  This strict-less rule is the
    frozen conservative tie convention and the plus-one denominator prevents
    an empirical reference maximum from reaching one.
    """

    reference_values = _float_vector(reference, name="reference")
    query_values = _float_vector(query, name="query")
    if len(reference_values) == 0:
        raise ValueError("reference must not be empty")
    ordered = np.sort(reference_values, kind="mergesort")
    left = np.searchsorted(ordered, query_values, side="left")
    result = left.astype(np.float64) / float(len(ordered) + 1)
    return _freeze(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RankLikelihoodQueryResult:
    """Immutable primary and negative-ECDF scores for one query vector."""

    ranks: np.ndarray
    bin_indices: np.ndarray
    log_likelihood_ratio: np.ndarray
    dual_template_score: np.ndarray
    negative_ecdf_score: np.ndarray

    def __post_init__(self) -> None:
        ranks = _unit_interval_vector(self.ranks, name="ranks", dtype=np.float64)
        bins = _integer_vector(
            self.bin_indices,
            name="bin_indices",
            dtype=np.int64,
            count=len(ranks),
        )
        llr = _float_vector(
            self.log_likelihood_ratio,
            name="log_likelihood_ratio",
            dtype=np.float64,
            count=len(ranks),
        )
        dual = _unit_interval_vector(
            self.dual_template_score,
            name="dual_template_score",
            dtype=np.float64,
            count=len(ranks),
        )
        control = _unit_interval_vector(
            self.negative_ecdf_score,
            name="negative_ecdf_score",
            dtype=np.float64,
            count=len(ranks),
        )
        if np.any(bins < 0):
            raise ValueError("bin_indices must be nonnegative")
        for name, values, dtype in (
            ("ranks", ranks, np.float64),
            ("bin_indices", bins, np.int64),
            ("log_likelihood_ratio", llr, np.float64),
            ("dual_template_score", dual, np.float64),
            ("negative_ecdf_score", control, np.float64),
        ):
            object.__setattr__(self, name, _freeze(values, dtype=dtype))


class FamilyBalancedRankLikelihoodModel:
    """Family-equal class histogram likelihood with source-LOO calibration."""

    def __init__(
        self,
        family_batches: Mapping[str, FamilySourceRankBatch],
        *,
        bin_count: int,
        beta: float,
    ) -> None:
        if not isinstance(family_batches, Mapping) or len(family_batches) < 2:
            raise ValueError("family_batches must contain at least two families")
        if any(
            not isinstance(name, str) or not name or "\x00" in name
            for name in family_batches
        ):
            raise ValueError("family names must be non-empty strings without NUL")
        family_order = tuple(sorted(family_batches))
        batches: list[FamilySourceRankBatch] = []
        for name in family_order:
            batch = family_batches[name]
            if not isinstance(batch, FamilySourceRankBatch):
                raise ValueError(
                    f"family_batches[{name!r}] must be a FamilySourceRankBatch"
                )
            batches.append(batch)
        selected_bins = _positive_integer(bin_count, name="bin_count")
        if selected_bins not in ALLOWED_BIN_COUNTS:
            raise ValueError(f"bin_count must be one of {ALLOWED_BIN_COUNTS}")
        if isinstance(beta, (bool, np.bool_)):
            raise ValueError(f"beta must be one of {ALLOWED_BETAS}")
        selected_beta = float(beta)
        if selected_beta not in ALLOWED_BETAS:
            raise ValueError(f"beta must be one of {ALLOWED_BETAS}")

        family_count = len(batches)
        counts = np.zeros(
            (family_count, CLASS_COUNT, selected_bins), dtype=np.int64
        )
        totals = np.zeros((family_count, CLASS_COUNT), dtype=np.int64)
        batch_bins: list[np.ndarray] = []
        for family_index, batch in enumerate(batches):
            bins = _bin_indices(batch.ranks, selected_bins)
            batch_bins.append(bins)
            for class_index, selected_label in (
                (NEGATIVE_CLASS_INDEX, False),
                (POSITIVE_CLASS_INDEX, True),
            ):
                class_bins = bins[batch.labels == selected_label]
                counts[family_index, class_index] = np.bincount(
                    class_bins, minlength=selected_bins
                )
                totals[family_index, class_index] = len(class_bins)

        family_density = np.empty_like(counts, dtype=np.float64)
        for family_index in range(family_count):
            for class_index in range(CLASS_COUNT):
                family_density[family_index, class_index] = (
                    _smoothed_histogram_mass(
                        counts[family_index, class_index], beta=selected_beta
                    )
                )
        full_density = family_density.mean(axis=0)

        dual_reference_parts: list[np.ndarray] = []
        negative_rank_parts: list[np.ndarray] = []
        density_sum = family_density.sum(axis=0)
        for family_index, (batch, bins) in enumerate(zip(batches, batch_bins, strict=True)):
            family_reference = np.empty(
                int((~batch.labels).sum()), dtype=np.float64
            )
            negative_positions = np.full(len(batch.ranks), -1, dtype=np.int64)
            negative_positions[~batch.labels] = np.arange(
                len(family_reference), dtype=np.int64
            )
            for source_id in np.unique(batch.source_ids):
                source_mask = batch.source_ids == source_id
                source_counts = np.zeros(
                    (CLASS_COUNT, selected_bins), dtype=np.int64
                )
                for class_index, selected_label in (
                    (NEGATIVE_CLASS_INDEX, False),
                    (POSITIVE_CLASS_INDEX, True),
                ):
                    selected = source_mask & (batch.labels == selected_label)
                    source_counts[class_index] = np.bincount(
                        bins[selected], minlength=selected_bins
                    )
                remaining = counts[family_index] - source_counts
                if np.any(remaining < 0):
                    raise RuntimeError("source histogram subtraction became negative")
                remaining_class_totals = remaining.sum(axis=1)
                if np.any(remaining_class_totals == 0):
                    missing = (
                        "negative"
                        if remaining_class_totals[NEGATIVE_CLASS_INDEX] == 0
                        else "positive"
                    )
                    raise ValueError(
                        "leave-one-source-out histogram must retain both classes: "
                        f"family={family_order[family_index]!r}, "
                        f"source_id={int(source_id)}, missing={missing}"
                    )
                loo_family_density = np.vstack(
                    [
                        _smoothed_histogram_mass(
                            remaining[class_index], beta=selected_beta
                        )
                        for class_index in range(CLASS_COUNT)
                    ]
                )
                loo_density = (
                    density_sum
                    - family_density[family_index]
                    + loo_family_density
                ) / float(family_count)
                negative_mask = source_mask & ~batch.labels
                if negative_mask.any():
                    negative_bins = bins[negative_mask]
                    ell = np.log(
                        loo_density[POSITIVE_CLASS_INDEX, negative_bins]
                    ) - np.log(
                        loo_density[NEGATIVE_CLASS_INDEX, negative_bins]
                    )
                    family_reference[negative_positions[negative_mask]] = ell
            if not np.isfinite(family_reference).all():
                raise RuntimeError("leave-one-source-out reference is incomplete")
            dual_reference_parts.append(np.sort(family_reference, kind="mergesort"))
            negative_rank_parts.append(
                np.sort(batch.ranks[~batch.labels], kind="mergesort")
            )

        dual_values, dual_offsets = _pack_parts(dual_reference_parts)
        rank_values, rank_offsets = _pack_parts(negative_rank_parts)
        self._set_state(
            family_order=family_order,
            bin_count=selected_bins,
            beta=selected_beta,
            family_class_histogram_counts=counts,
            family_class_totals=totals,
            family_class_density=family_density,
            full_class_density=full_density,
            dual_negative_reference_values=dual_values,
            dual_negative_reference_offsets=dual_offsets,
            negative_rank_reference_values=rank_values,
            negative_rank_reference_offsets=rank_offsets,
        )

    def _set_state(
        self,
        *,
        family_order: tuple[str, ...],
        bin_count: int,
        beta: float,
        family_class_histogram_counts: object,
        family_class_totals: object,
        family_class_density: object,
        full_class_density: object,
        dual_negative_reference_values: object,
        dual_negative_reference_offsets: object,
        negative_rank_reference_values: object,
        negative_rank_reference_offsets: object,
    ) -> None:
        family_count = len(family_order)
        counts = np.asarray(family_class_histogram_counts)
        totals = np.asarray(family_class_totals)
        density = np.asarray(family_class_density)
        full = np.asarray(full_class_density)
        dual_values = np.asarray(dual_negative_reference_values)
        dual_offsets = np.asarray(dual_negative_reference_offsets)
        rank_values = np.asarray(negative_rank_reference_values)
        rank_offsets = np.asarray(negative_rank_reference_offsets)
        expected_shape = (family_count, CLASS_COUNT, bin_count)
        if counts.dtype != np.dtype(np.int64) or counts.shape != expected_shape:
            raise ValueError("family histogram counts have invalid shape or dtype")
        if totals.dtype != np.dtype(np.int64) or totals.shape != (
            family_count,
            CLASS_COUNT,
        ):
            raise ValueError("family class totals have invalid shape or dtype")
        if np.any(counts < 0) or np.any(totals <= 0):
            raise ValueError("each family/class histogram must be non-empty")
        if not np.array_equal(counts.sum(axis=2), totals):
            raise ValueError("family histogram counts do not reproduce totals")
        if density.dtype != np.dtype(np.float64) or density.shape != expected_shape:
            raise ValueError("family density has invalid shape or dtype")
        if full.dtype != np.dtype(np.float64) or full.shape != (
            CLASS_COUNT,
            bin_count,
        ):
            raise ValueError("full density has invalid shape or dtype")
        expected_density = (
            counts.astype(np.float64) + beta
        ) / (totals[:, :, None].astype(np.float64) + beta * float(bin_count))
        if not np.array_equal(density, expected_density):
            raise ValueError("family density does not reproduce beta smoothing")
        if not np.array_equal(full, density.mean(axis=0)):
            raise ValueError("full density is not the family-equal mean")
        _validate_packed_references(
            dual_values,
            dual_offsets,
            family_count=family_count,
            name="dual negative reference",
            require_unit_interval=False,
        )
        _validate_packed_references(
            rank_values,
            rank_offsets,
            family_count=family_count,
            name="negative rank reference",
            require_unit_interval=True,
        )
        if not np.array_equal(
            np.diff(dual_offsets), totals[:, NEGATIVE_CLASS_INDEX]
        ) or not np.array_equal(
            np.diff(rank_offsets), totals[:, NEGATIVE_CLASS_INDEX]
        ):
            raise ValueError("negative references do not match family negative totals")

        self.family_order = tuple(family_order)
        self.bin_count = int(bin_count)
        self.beta = float(beta)
        self.family_class_histogram_counts = _freeze(counts, dtype=np.int64)
        self.family_class_totals = _freeze(totals, dtype=np.int64)
        self.family_class_density = _freeze(density, dtype=np.float64)
        self.full_class_density = _freeze(full, dtype=np.float64)
        self.dual_negative_reference_values = _freeze(
            dual_values, dtype=np.float64
        )
        self.dual_negative_reference_offsets = _freeze(
            dual_offsets, dtype=np.int64
        )
        self.negative_rank_reference_values = _freeze(
            rank_values, dtype=np.float64
        )
        self.negative_rank_reference_offsets = _freeze(
            rank_offsets, dtype=np.int64
        )

    @property
    def family_count(self) -> int:
        return len(self.family_order)

    def query(self, ranks: object) -> RankLikelihoodQueryResult:
        """Score query ranks with the complete fit model and both controls."""

        selected = _unit_interval_vector(ranks, name="ranks", dtype=np.float64)
        bins = _bin_indices(selected, self.bin_count)
        llr = np.log(
            self.full_class_density[POSITIVE_CLASS_INDEX, bins]
        ) - np.log(self.full_class_density[NEGATIVE_CLASS_INDEX, bins])
        dual = np.zeros(len(selected), dtype=np.float64)
        control = np.zeros(len(selected), dtype=np.float64)
        for family_index in range(self.family_count):
            dual_start = int(self.dual_negative_reference_offsets[family_index])
            dual_stop = int(self.dual_negative_reference_offsets[family_index + 1])
            dual += conservative_strict_ecdf(
                self.dual_negative_reference_values[dual_start:dual_stop], llr
            )
            rank_start = int(self.negative_rank_reference_offsets[family_index])
            rank_stop = int(self.negative_rank_reference_offsets[family_index + 1])
            control += conservative_strict_ecdf(
                self.negative_rank_reference_values[rank_start:rank_stop], selected
            )
        dual /= float(self.family_count)
        control /= float(self.family_count)
        return RankLikelihoodQueryResult(
            ranks=selected,
            bin_indices=bins,
            log_likelihood_ratio=llr,
            dual_template_score=dual,
            negative_ecdf_score=control,
        )

    def export_arrays(self) -> dict[str, np.ndarray]:
        """Return an ``allow_pickle=False`` compatible immutable model state."""

        family_width = max(len(name) for name in self.family_order)
        arrays = {
            "serialization_version_int16": np.asarray(
                _SERIALIZATION_VERSION, dtype=np.int16
            ),
            "bin_count_int64": np.asarray(self.bin_count, dtype=np.int64),
            "beta_float64": np.asarray(self.beta, dtype=np.float64),
            "family_order_unicode": np.asarray(
                self.family_order, dtype=f"<U{family_width}"
            ),
            "family_class_histogram_counts": self.family_class_histogram_counts,
            "family_class_totals": self.family_class_totals,
            "family_class_density": self.family_class_density,
            "full_class_density": self.full_class_density,
            "dual_negative_reference_values": self.dual_negative_reference_values,
            "dual_negative_reference_offsets": self.dual_negative_reference_offsets,
            "negative_rank_reference_values": self.negative_rank_reference_values,
            "negative_rank_reference_offsets": self.negative_rank_reference_offsets,
        }
        return {
            name: _freeze(value, dtype=np.asarray(value).dtype)
            for name, value in arrays.items()
        }

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, np.ndarray]
    ) -> "FamilyBalancedRankLikelihoodModel":
        """Reconstruct and validate a serialized numerical model."""

        expected_names = {
            "serialization_version_int16",
            "bin_count_int64",
            "beta_float64",
            "family_order_unicode",
            "family_class_histogram_counts",
            "family_class_totals",
            "family_class_density",
            "full_class_density",
            "dual_negative_reference_values",
            "dual_negative_reference_offsets",
            "negative_rank_reference_values",
            "negative_rank_reference_offsets",
        }
        if set(arrays) != expected_names:
            raise ValueError("serialized model array names drifted")
        version = np.asarray(arrays["serialization_version_int16"])
        bins = np.asarray(arrays["bin_count_int64"])
        beta = np.asarray(arrays["beta_float64"])
        families = np.asarray(arrays["family_order_unicode"])
        if version.shape != () or version.dtype != np.dtype(np.int16):
            raise ValueError("serialization version scalar is invalid")
        if int(version) != _SERIALIZATION_VERSION:
            raise ValueError("serialization version is unsupported")
        if bins.shape != () or bins.dtype != np.dtype(np.int64):
            raise ValueError("bin count scalar is invalid")
        selected_bins = int(bins)
        if selected_bins not in ALLOWED_BIN_COUNTS:
            raise ValueError("serialized bin count is not allowed")
        if beta.shape != () or beta.dtype != np.dtype(np.float64):
            raise ValueError("beta scalar is invalid")
        selected_beta = float(beta)
        if selected_beta not in ALLOWED_BETAS:
            raise ValueError("serialized beta is not allowed")
        if families.ndim != 1 or families.dtype.kind != "U" or len(families) < 2:
            raise ValueError("serialized family order is invalid")
        family_order = tuple(str(value) for value in families.tolist())
        if (
            any(not name or "\x00" in name for name in family_order)
            or tuple(sorted(family_order)) != family_order
            or len(set(family_order)) != len(family_order)
        ):
            raise ValueError("serialized family order is not canonical")
        model = cls.__new__(cls)
        model._set_state(
            family_order=family_order,
            bin_count=selected_bins,
            beta=selected_beta,
            family_class_histogram_counts=arrays[
                "family_class_histogram_counts"
            ],
            family_class_totals=arrays["family_class_totals"],
            family_class_density=arrays["family_class_density"],
            full_class_density=arrays["full_class_density"],
            dual_negative_reference_values=arrays[
                "dual_negative_reference_values"
            ],
            dual_negative_reference_offsets=arrays[
                "dual_negative_reference_offsets"
            ],
            negative_rank_reference_values=arrays[
                "negative_rank_reference_values"
            ],
            negative_rank_reference_offsets=arrays[
                "negative_rank_reference_offsets"
            ],
        )
        return model


def _pack_parts(parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not parts or any(len(part) == 0 for part in parts):
        raise ValueError("every family reference must be non-empty")
    offsets = np.zeros(len(parts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        np.asarray([len(part) for part in parts], dtype=np.int64)
    )
    return (
        np.concatenate(parts).astype(np.float64, copy=False),
        offsets,
    )


def _validate_packed_references(
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    family_count: int,
    name: str,
    require_unit_interval: bool,
) -> None:
    if values.dtype != np.dtype(np.float64) or values.ndim != 1:
        raise ValueError(f"{name} values have invalid shape or dtype")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} values contain NaN or Inf")
    if require_unit_interval and np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} values must lie in [0,1]")
    if offsets.dtype != np.dtype(np.int64) or offsets.shape != (
        family_count + 1,
    ):
        raise ValueError(f"{name} offsets have invalid shape or dtype")
    if (
        offsets[0] != 0
        or offsets[-1] != len(values)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError(f"{name} offsets are not strict packed boundaries")
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        part = values[int(start) : int(stop)]
        if np.any(part[1:] < part[:-1]):
            raise ValueError(f"{name} family parts must be sorted")


def strict_absolute_threshold(
    scores: object,
    eligible: object,
    *,
    threshold: float,
) -> np.ndarray:
    """Predict positive exactly when an eligible score is strictly above tau."""

    values = _unit_interval_vector(scores, name="scores")
    allowed = _bool_vector(eligible, name="eligible", count=len(values))
    selected_threshold = _threshold(threshold)
    return _freeze(allowed & (values > selected_threshold), dtype=np.bool_)


__all__ = [
    "ALLOWED_BETAS",
    "ALLOWED_BIN_COUNTS",
    "ASSIGNED_ROWS_PER_BLOCK_DX_LEVEL",
    "BLOCK_COUNT",
    "DEFAULT_CENTER_COUNT",
    "DX_LEVEL_COUNT",
    "FamilyBalancedRankLikelihoodModel",
    "FamilySourceRankBatch",
    "PairedCenterRanks",
    "RankLikelihoodQueryResult",
    "conservative_strict_ecdf",
    "assigned_block_dx_midranks",
    "empirical_midrank",
    "pair_assigned_center_ranks",
    "strict_absolute_threshold",
]

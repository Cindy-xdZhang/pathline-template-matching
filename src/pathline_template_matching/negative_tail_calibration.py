"""Fit-negative empirical tail calibration for exact same-scale distances.

This module is deliberately separate from the frozen scale-conditioned
retrieval implementation.  It keeps that implementation's global
negative-only standardization and exact Euclidean distance convention, while
replacing query-batch ranks with references built exclusively from fitted
negative rows.

For every numeric scale and requested ``k``, a reference value is the exact
``k``-th neighbour distance of one fitted negative row after explicitly
excluding that same row.  A query distance is converted to the conservative
empirical upper-tail probability ``(1 + count(reference >= distance)) /
(reference_count + 1)``.  The classifier-facing anomaly score is one minus
that probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch

from .scale_one_class import (
    _STD_FLOOR,
    _deterministic_algorithms,
    _feature_matrix,
    _integer_vector,
    _ks,
    _positive_integer,
    _standardize_for_distance,
    _torch_device,
)


SCALE_COUNT = 2000
SCALE_BLOCK_SIZE = 1000
SCALE_BLOCK_COUNT = 2
DEFAULT_KS = (1, 5, 15, 31)
SHRINKAGE_LAMBDA = 64.0

CALIBRATION_NONE = np.int8(0)
CALIBRATION_LOCAL_BLOCK_SHRINK = np.int8(1)
CALIBRATION_LOCAL_GLOBAL_SHRINK = np.int8(2)
CALIBRATION_LOCAL_ONLY = np.int8(3)
CALIBRATION_BLOCK_FALLBACK = np.int8(4)
CALIBRATION_GLOBAL_FALLBACK = np.int8(5)

CALIBRATION_MODE_NAMES = MappingProxyType(
    {
        int(CALIBRATION_NONE): "no_calibration",
        int(CALIBRATION_LOCAL_BLOCK_SHRINK): "local_block_shrink",
        int(CALIBRATION_LOCAL_GLOBAL_SHRINK): "local_global_shrink",
        int(CALIBRATION_LOCAL_ONLY): "local_only",
        int(CALIBRATION_BLOCK_FALLBACK): "block_fallback",
        int(CALIBRATION_GLOBAL_FALLBACK): "global_fallback",
    }
)

_SERIALIZATION_VERSION = 1


def _validated_scale_ids(values: object, *, name: str) -> np.ndarray:
    result = _integer_vector(values, name=name)
    if np.any((result < 0) | (result >= SCALE_COUNT)):
        raise ValueError(f"{name} must lie in the frozen range 0..1999")
    return result


def _validated_lambda(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"shrinkage_lambda must equal {SHRINKAGE_LAMBDA:g}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"shrinkage_lambda must equal {SHRINKAGE_LAMBDA:g}"
        ) from error
    if not np.isfinite(result) or result != SHRINKAGE_LAMBDA:
        raise ValueError(f"shrinkage_lambda must equal {SHRINKAGE_LAMBDA:g}")
    return result


def _freeze_array(values: object, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _exact_query_k_distances(
    library: np.ndarray,
    query: np.ndarray,
    requested_ks: tuple[int, ...],
    *,
    device: torch.device,
    query_chunk_size: int,
    library_chunk_size: int,
) -> dict[int, np.ndarray]:
    """Return several exact order statistics in one maximum-k pass."""

    maximum_k = max(requested_ks)
    if len(library) < maximum_k:
        raise ValueError("library does not support the requested maximum k")
    output = {k: np.empty(len(query), dtype=np.float32) for k in requested_ks}
    if len(query) == 0:
        return output

    with torch.inference_mode(), _deterministic_algorithms():
        for query_start in range(0, len(query), query_chunk_size):
            query_stop = min(query_start + query_chunk_size, len(query))
            query_tensor = torch.from_numpy(query[query_start:query_stop]).to(device)
            nearest = torch.empty(
                (len(query_tensor), 0), dtype=torch.float32, device=device
            )
            for library_start in range(0, len(library), library_chunk_size):
                library_stop = min(library_start + library_chunk_size, len(library))
                library_tensor = torch.from_numpy(
                    library[library_start:library_stop]
                ).to(device)
                distances = torch.cdist(
                    query_tensor,
                    library_tensor,
                    p=2.0,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                candidates = torch.cat((nearest, distances), dim=1)
                keep = min(maximum_k, candidates.shape[1])
                nearest = torch.topk(
                    candidates, k=keep, dim=1, largest=False, sorted=True
                ).values
            if nearest.shape[1] != maximum_k:
                raise RuntimeError("exact maximum-k query search was incomplete")
            for k in requested_ks:
                output[k][query_start:query_stop] = nearest[:, k - 1].cpu().numpy()
    return output


def _exact_leave_one_out_k_distances(
    library: np.ndarray,
    requested_ks: tuple[int, ...],
    *,
    device: torch.device,
    query_chunk_size: int,
    library_chunk_size: int,
) -> dict[int, np.ndarray]:
    """Return exact self-excluded distances for several k values in one pass.

    The diagonal is masked by row identity.  Other rows with identical
    features remain valid zero-distance neighbours.
    """

    maximum_k = max(requested_ks)
    if len(library) <= maximum_k:
        raise ValueError(
            "leave-one-out search requires at least maximum_k + 1 rows"
        )
    output = {k: np.empty(len(library), dtype=np.float32) for k in requested_ks}

    with torch.inference_mode(), _deterministic_algorithms():
        for query_start in range(0, len(library), query_chunk_size):
            query_stop = min(query_start + query_chunk_size, len(library))
            query_tensor = torch.from_numpy(library[query_start:query_stop]).to(device)
            nearest = torch.empty(
                (len(query_tensor), 0), dtype=torch.float32, device=device
            )
            for library_start in range(0, len(library), library_chunk_size):
                library_stop = min(library_start + library_chunk_size, len(library))
                library_tensor = torch.from_numpy(
                    library[library_start:library_stop]
                ).to(device)
                distances = torch.cdist(
                    query_tensor,
                    library_tensor,
                    p=2.0,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )

                overlap_start = max(query_start, library_start)
                overlap_stop = min(query_stop, library_stop)
                if overlap_start < overlap_stop:
                    global_rows = torch.arange(
                        overlap_start,
                        overlap_stop,
                        dtype=torch.long,
                        device=device,
                    )
                    distances[
                        global_rows - query_start,
                        global_rows - library_start,
                    ] = torch.inf

                candidates = torch.cat((nearest, distances), dim=1)
                keep = min(maximum_k, candidates.shape[1])
                nearest = torch.topk(
                    candidates, k=keep, dim=1, largest=False, sorted=True
                ).values
            if nearest.shape[1] != maximum_k or not torch.isfinite(nearest).all():
                raise RuntimeError("exact leave-one-out maximum-k search was incomplete")
            for k in requested_ks:
                output[k][query_start:query_stop] = nearest[:, k - 1].cpu().numpy()
    return output


def empirical_upper_tail_probability(
    sorted_reference: object, distances: object
) -> np.ndarray:
    """Return conservative plus-one empirical upper-tail probabilities."""

    reference = np.asarray(sorted_reference)
    values = np.asarray(distances)
    if (
        reference.ndim != 1
        or values.ndim != 1
        or not np.issubdtype(reference.dtype, np.number)
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(reference.dtype, np.bool_)
        or np.issubdtype(values.dtype, np.bool_)
        or np.issubdtype(reference.dtype, np.complexfloating)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise ValueError("reference and distances must be one-dimensional real arrays")
    reference64 = np.asarray(reference, dtype=np.float64)
    values64 = np.asarray(values, dtype=np.float64)
    if (
        len(reference64) == 0
        or not np.isfinite(reference64).all()
        or not np.isfinite(values64).all()
        or np.any(reference64 < 0.0)
        or np.any(values64 < 0.0)
        or np.any(reference64[1:] < reference64[:-1])
    ):
        raise ValueError(
            "reference must be non-empty, finite, nonnegative, and ascending; "
            "distances must be finite and nonnegative"
        )
    first_greater_or_equal = np.searchsorted(reference64, values64, side="left")
    count_greater_or_equal = len(reference64) - first_greater_or_equal
    return (
        1.0 + count_greater_or_equal.astype(np.float64)
    ) / float(len(reference64) + 1)


def _difference_upper_tail_probability(
    sorted_superset: np.ndarray,
    sorted_excluded: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Tail probability for a multiset difference without materializing it."""

    remaining_count = len(sorted_superset) - len(sorted_excluded)
    if remaining_count <= 0:
        raise ValueError("the empirical reference difference is empty")
    superset_ge = len(sorted_superset) - np.searchsorted(
        sorted_superset, distances, side="left"
    )
    excluded_ge = len(sorted_excluded) - np.searchsorted(
        sorted_excluded, distances, side="left"
    )
    remaining_ge = superset_ge - excluded_ge
    if np.any((remaining_ge < 0) | (remaining_ge > remaining_count)):
        raise RuntimeError("empirical reference multiset subtraction was inconsistent")
    return (1.0 + remaining_ge.astype(np.float64)) / float(remaining_count + 1)


@dataclass(frozen=True, slots=True)
class NegativeTailQueryResult:
    """Immutable query outputs keyed by each requested neighbour count."""

    raw_distances: Mapping[int, np.ndarray]
    tail_probabilities: Mapping[int, np.ndarray]
    anomaly_scores: Mapping[int, np.ndarray]
    retrieval_supported: Mapping[int, np.ndarray]
    calibration_supported: Mapping[int, np.ndarray]
    calibration_modes: Mapping[int, np.ndarray]

    def __post_init__(self) -> None:
        fields = (
            "raw_distances",
            "tail_probabilities",
            "anomaly_scores",
            "retrieval_supported",
            "calibration_supported",
            "calibration_modes",
        )
        mappings = [getattr(self, name) for name in fields]
        keys = tuple(mappings[0])
        if not keys or any(tuple(mapping) != keys for mapping in mappings[1:]):
            raise ValueError("all query-result mappings must have identical non-empty keys")
        expected_length: int | None = None
        frozen_mappings: list[Mapping[int, np.ndarray]] = []
        for field_name, mapping in zip(fields, mappings):
            frozen: dict[int, np.ndarray] = {}
            for k, values in mapping.items():
                array = np.asarray(values)
                if array.ndim != 1:
                    raise ValueError(f"{field_name}[{k}] must be one-dimensional")
                if expected_length is None:
                    expected_length = len(array)
                elif len(array) != expected_length:
                    raise ValueError("all query-result arrays must have identical length")
                frozen[int(k)] = _freeze_array(array)
            frozen_mappings.append(MappingProxyType(frozen))
        for field_name, value in zip(fields, frozen_mappings):
            object.__setattr__(self, field_name, value)


class ScaleConditionedNegativeTailCalibrator:
    """Exact same-scale negative distance with fit-only empirical calibration."""

    def __init__(
        self,
        negative_features: object,
        negative_scale_ids: object,
        *,
        ks: Sequence[int] = DEFAULT_KS,
        shrinkage_lambda: float = SHRINKAGE_LAMBDA,
        device: str | torch.device = "cpu",
        query_chunk_size: int = 1024,
        library_chunk_size: int = 8192,
    ) -> None:
        features = _feature_matrix(
            negative_features, name="negative_features", allow_empty=False
        )
        scale_ids = _validated_scale_ids(
            negative_scale_ids, name="negative_scale_ids"
        )
        if len(scale_ids) != len(features):
            raise ValueError(
                "negative_scale_ids must contain one scale id per negative row"
            )
        requested_ks = _ks(ks)
        selected_device = _torch_device(device)
        query_chunk = _positive_integer(query_chunk_size, name="query_chunk_size")
        library_chunk = _positive_integer(
            library_chunk_size, name="library_chunk_size"
        )
        selected_lambda = _validated_lambda(shrinkage_lambda)

        features64 = features.astype(np.float64)
        mean = features64.mean(axis=0, dtype=np.float64)
        raw_std = features64.std(axis=0, dtype=np.float64, ddof=0)
        zero_variance = raw_std < _STD_FLOOR
        effective_std = raw_std.copy()
        effective_std[zero_variance] = 1.0
        normalized = np.ascontiguousarray(
            (features64 - mean) / effective_std, dtype=np.float32
        )
        if not np.isfinite(normalized).all():
            raise ValueError(
                "standardized negative_features contain a value outside float32 range"
            )

        scale_order = np.argsort(scale_ids, kind="stable")
        sorted_features = np.ascontiguousarray(normalized[scale_order])
        sorted_scales = np.asarray(scale_ids[scale_order], dtype=np.int64)
        scale_counts = np.bincount(sorted_scales, minlength=SCALE_COUNT).astype(
            np.int64, copy=False
        )
        scale_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(scale_counts, dtype=np.int64))
        )

        loo_distances, loo_offsets = self._build_loo_references(
            sorted_features,
            scale_offsets,
            requested_ks,
            device=selected_device,
            query_chunk_size=query_chunk,
            library_chunk_size=library_chunk,
        )
        self._install_state(
            ks=requested_ks,
            shrinkage_lambda=selected_lambda,
            mean=mean,
            raw_std=raw_std,
            effective_std=effective_std,
            zero_variance=zero_variance,
            negative_features=sorted_features,
            scale_offsets=scale_offsets,
            loo_distances=loo_distances,
            loo_offsets=loo_offsets,
        )

    @staticmethod
    def _build_loo_references(
        negative_features: np.ndarray,
        scale_offsets: np.ndarray,
        ks: tuple[int, ...],
        *,
        device: torch.device,
        query_chunk_size: int,
        library_chunk_size: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        parts: dict[int, list[np.ndarray]] = {k: [] for k in ks}
        offsets: dict[int, np.ndarray] = {
            k: np.zeros(SCALE_COUNT + 1, dtype=np.int64) for k in ks
        }
        for scale_id in range(SCALE_COUNT):
            start = int(scale_offsets[scale_id])
            stop = int(scale_offsets[scale_id + 1])
            count = stop - start
            supported_ks = tuple(k for k in ks if count >= k + 1)
            if supported_ks:
                values = _exact_leave_one_out_k_distances(
                    negative_features[start:stop],
                    supported_ks,
                    device=device,
                    query_chunk_size=query_chunk_size,
                    library_chunk_size=library_chunk_size,
                )
            else:
                values = {}
            for k in ks:
                if k in values:
                    sorted_values = np.sort(values[k], kind="mergesort")
                    if len(sorted_values) != count or not np.isfinite(sorted_values).all():
                        raise RuntimeError("invalid leave-one-out reference population")
                    parts[k].append(np.ascontiguousarray(sorted_values, dtype=np.float32))
                    offsets[k][scale_id + 1] = offsets[k][scale_id] + count
                else:
                    offsets[k][scale_id + 1] = offsets[k][scale_id]
        concatenated = {
            k: (
                np.ascontiguousarray(np.concatenate(parts[k]), dtype=np.float32)
                if parts[k]
                else np.empty(0, dtype=np.float32)
            )
            for k in ks
        }
        return concatenated, offsets

    def _install_state(
        self,
        *,
        ks: tuple[int, ...],
        shrinkage_lambda: float,
        mean: np.ndarray,
        raw_std: np.ndarray,
        effective_std: np.ndarray,
        zero_variance: np.ndarray,
        negative_features: np.ndarray,
        scale_offsets: np.ndarray,
        loo_distances: Mapping[int, np.ndarray],
        loo_offsets: Mapping[int, np.ndarray],
    ) -> None:
        self._ks = tuple(ks)
        self._shrinkage_lambda = float(shrinkage_lambda)
        self._mean = _freeze_array(mean, dtype=np.float64)
        self._raw_std = _freeze_array(raw_std, dtype=np.float64)
        self._effective_std = _freeze_array(effective_std, dtype=np.float64)
        self._zero_variance = _freeze_array(zero_variance, dtype=np.bool_)
        # This array is private and never returned directly.  Keep it writable
        # because torch.from_numpy warns that a read-only NumPy backing store
        # has undefined behaviour even when the tensor is only read.
        self._negative_features = np.ascontiguousarray(
            negative_features, dtype=np.float32
        ).copy()
        self._scale_offsets = _freeze_array(scale_offsets, dtype=np.int64)
        self._loo_distances = {
            k: _freeze_array(loo_distances[k], dtype=np.float32) for k in self._ks
        }
        self._loo_offsets = {
            k: _freeze_array(loo_offsets[k], dtype=np.int64) for k in self._ks
        }
        self._block_references: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for k in self._ks:
            local_values = self._loo_distances[k]
            local_offsets = self._loo_offsets[k]
            block_parts: list[np.ndarray] = []
            for block_index in range(SCALE_BLOCK_COUNT):
                first_scale = block_index * SCALE_BLOCK_SIZE
                last_scale = first_scale + SCALE_BLOCK_SIZE
                start = int(local_offsets[first_scale])
                stop = int(local_offsets[last_scale])
                block_parts.append(
                    _freeze_array(
                        np.sort(local_values[start:stop], kind="mergesort"),
                        dtype=np.float32,
                    )
                )
            self._block_references[k] = (block_parts[0], block_parts[1])

    @property
    def ks(self) -> tuple[int, ...]:
        return self._ks

    @property
    def fit_audit(self) -> dict[str, object]:
        counts = np.diff(self._scale_offsets)
        return {
            "count": int(len(self._negative_features)),
            "dim": int(self._negative_features.shape[1]),
            "ks": self._ks,
            "shrinkage_lambda": self._shrinkage_lambda,
            "scaler_statistics_dtype": "float64",
            "distance_dtype": "float32",
            "std_ddof": 0,
            "std_floor_exclusive": _STD_FLOOR,
            "mean": self._mean.tolist(),
            "raw_std": self._raw_std.tolist(),
            "effective_std": self._effective_std.tolist(),
            "zero_variance_feature_mask": self._zero_variance.tolist(),
            "scale_counts": {
                int(scale_id): int(counts[scale_id])
                for scale_id in np.flatnonzero(counts)
            },
            "loo_reference_counts_by_k": {
                str(k): int(len(self._loo_distances[k])) for k in self._ks
            },
            "loo_supported_scale_counts_by_k": {
                str(k): int(np.count_nonzero(np.diff(self._loo_offsets[k])))
                for k in self._ks
            },
            "block_reference_counts_by_k": {
                str(k): [
                    int(len(self._block_references[k][0])),
                    int(len(self._block_references[k][1])),
                ]
                for k in self._ks
            },
        }

    def _local_reference(self, k: int, scale_id: int) -> np.ndarray:
        offsets = self._loo_offsets[k]
        return self._loo_distances[k][
            int(offsets[scale_id]) : int(offsets[scale_id + 1])
        ]

    def _calibrate_scale(
        self, k: int, scale_id: int, distances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local = self._local_reference(k, scale_id)
        block_index = scale_id // SCALE_BLOCK_SIZE
        block = self._block_references[k][block_index]
        other_block = self._block_references[k][1 - block_index]
        block_other_count = len(block) - len(local)

        if len(local):
            local_probability = empirical_upper_tail_probability(local, distances)
            local_weight = len(local) / (len(local) + self._shrinkage_lambda)
            if block_other_count > 0:
                prior_probability = _difference_upper_tail_probability(
                    block, local, distances
                )
                probability = (
                    local_weight * local_probability
                    + (1.0 - local_weight) * prior_probability
                )
                mode = CALIBRATION_LOCAL_BLOCK_SHRINK
            elif len(other_block):
                # With exactly two frozen scale blocks and no other reference
                # in the current block, global-other is exactly the opposite
                # block.  Reuse it rather than retaining a third full copy of
                # every LOO reference.
                prior_probability = empirical_upper_tail_probability(
                    other_block, distances
                )
                probability = (
                    local_weight * local_probability
                    + (1.0 - local_weight) * prior_probability
                )
                mode = CALIBRATION_LOCAL_GLOBAL_SHRINK
            else:
                probability = local_probability
                mode = CALIBRATION_LOCAL_ONLY
        elif len(block):
            probability = empirical_upper_tail_probability(block, distances)
            mode = CALIBRATION_BLOCK_FALLBACK
        elif len(other_block):
            probability = empirical_upper_tail_probability(
                other_block, distances
            )
            mode = CALIBRATION_GLOBAL_FALLBACK
        else:
            return (
                np.ones(len(distances), dtype=np.float64),
                np.zeros(len(distances), dtype=bool),
                np.full(len(distances), CALIBRATION_NONE, dtype=np.int8),
            )

        probability = np.asarray(probability, dtype=np.float64)
        if not np.isfinite(probability).all() or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise RuntimeError("tail calibration produced an invalid probability")
        return (
            probability,
            np.ones(len(distances), dtype=bool),
            np.full(len(distances), mode, dtype=np.int8),
        )

    def query(
        self,
        features: object,
        scale_ids: object,
        *,
        ks: Sequence[int] | None = None,
        device: str | torch.device = "cpu",
        query_chunk_size: int = 1024,
        library_chunk_size: int = 8192,
    ) -> NegativeTailQueryResult:
        """Return exact distances and fit-negative calibrated anomaly scores.

        Scales with fewer than ``k`` fitted negatives retain a NaN raw-distance
        sentinel and are retrieval/calibration unsupported.  A scale with
        exactly ``k`` negatives has a valid query distance but no local
        leave-one-out reference, so it uses the frozen block/global fallback.
        """

        query_features = _feature_matrix(features, name="features", allow_empty=True)
        if query_features.shape[1] != self._negative_features.shape[1]:
            raise ValueError(
                "features and fitted negative_features must have the same width"
            )
        query_scales = _validated_scale_ids(scale_ids, name="scale_ids")
        if len(query_scales) != len(query_features):
            raise ValueError("scale_ids must contain one scale id per query row")
        requested = self._ks if ks is None else _ks(ks)
        if any(k not in self._ks for k in requested):
            raise ValueError("query ks must be a subset of the fitted ks")
        selected_device = _torch_device(device)
        query_chunk = _positive_integer(query_chunk_size, name="query_chunk_size")
        library_chunk = _positive_integer(
            library_chunk_size, name="library_chunk_size"
        )

        normalized = _standardize_for_distance(
            query_features,
            mean=self._mean,
            effective_std=self._effective_std,
            name="features",
        )
        raw = {
            k: np.full(len(normalized), np.nan, dtype=np.float32) for k in requested
        }
        tail = {
            k: np.ones(len(normalized), dtype=np.float64) for k in requested
        }
        retrieval = {
            k: np.zeros(len(normalized), dtype=bool) for k in requested
        }
        calibration = {
            k: np.zeros(len(normalized), dtype=bool) for k in requested
        }
        modes = {
            k: np.full(len(normalized), CALIBRATION_NONE, dtype=np.int8)
            for k in requested
        }
        counts = np.diff(self._scale_offsets)

        query_order = np.argsort(query_scales, kind="stable")
        sorted_query_scales = query_scales[query_order]
        unique_scales, query_starts, query_counts = np.unique(
            sorted_query_scales, return_index=True, return_counts=True
        )
        for scale_id, query_start, query_count in zip(
            unique_scales, query_starts, query_counts
        ):
            scale = int(scale_id)
            query_rows = query_order[
                int(query_start) : int(query_start + query_count)
            ]
            available = int(counts[scale])
            supported_ks = tuple(k for k in requested if available >= k)
            if not supported_ks:
                continue
            start = int(self._scale_offsets[scale])
            stop = int(self._scale_offsets[scale + 1])
            scale_distances = _exact_query_k_distances(
                self._negative_features[start:stop],
                normalized[query_rows],
                supported_ks,
                device=selected_device,
                query_chunk_size=query_chunk,
                library_chunk_size=library_chunk,
            )
            for k in supported_ks:
                values = scale_distances[k]
                raw[k][query_rows] = values
                retrieval[k][query_rows] = True
                probability, supported, scale_modes = self._calibrate_scale(
                    k, scale, values
                )
                tail[k][query_rows] = probability
                calibration[k][query_rows] = supported
                modes[k][query_rows] = scale_modes

        anomalies = {k: 1.0 - tail[k] for k in requested}
        for k in requested:
            if not np.isfinite(raw[k][retrieval[k]]).all():
                raise RuntimeError("supported raw query distance is nonfinite")
            if not np.isnan(raw[k][~retrieval[k]]).all():
                raise RuntimeError("unsupported raw-distance sentinel drifted")
            if np.any(calibration[k] & ~retrieval[k]):
                raise RuntimeError("calibration support cannot exceed retrieval support")
            if not np.isfinite(tail[k]).all() or not np.isfinite(anomalies[k]).all():
                raise RuntimeError("tail query produced NaN or Inf")
            if np.any(anomalies[k][~calibration[k]] != 0.0):
                raise RuntimeError("unsupported calibration must have anomaly zero")

        return NegativeTailQueryResult(
            raw_distances=raw,
            tail_probabilities=tail,
            anomaly_scores=anomalies,
            retrieval_supported=retrieval,
            calibration_supported=calibration,
            calibration_modes=modes,
        )

    def export_arrays(self) -> dict[str, np.ndarray]:
        """Export a pickle-free, pure-array state sufficient for reconstruction."""

        arrays: dict[str, np.ndarray] = {
            "serialization_version": np.asarray(
                _SERIALIZATION_VERSION, dtype=np.int16
            ),
            "ks": np.asarray(self._ks, dtype=np.int64),
            "shrinkage_lambda": np.asarray(
                self._shrinkage_lambda, dtype=np.float64
            ),
            "mean": self._mean,
            "raw_std": self._raw_std,
            "effective_std": self._effective_std,
            "zero_variance_feature_mask": self._zero_variance,
            "negative_features": self._negative_features,
            "negative_scale_offsets": self._scale_offsets,
        }
        for k in self._ks:
            arrays[f"loo_distances_k_{k}"] = self._loo_distances[k]
            arrays[f"loo_scale_offsets_k_{k}"] = self._loo_offsets[k]
        # np.ascontiguousarray promotes a scalar to shape (1,); np.array keeps
        # scalar shape while still returning an independent C-order copy.
        return {
            name: np.array(value, copy=True, order="C")
            for name, value in arrays.items()
        }

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, object]
    ) -> "ScaleConditionedNegativeTailCalibrator":
        """Reconstruct an exact model state exported by :meth:`export_arrays`."""

        if not isinstance(arrays, Mapping):
            raise ValueError("arrays must be a mapping of pure numeric arrays")
        required_base = {
            "serialization_version",
            "ks",
            "shrinkage_lambda",
            "mean",
            "raw_std",
            "effective_std",
            "zero_variance_feature_mask",
            "negative_features",
            "negative_scale_offsets",
        }
        if not required_base.issubset(arrays):
            raise ValueError("serialized calibration state is incomplete")
        version = np.asarray(arrays["serialization_version"])
        if version.shape != () or version.dtype != np.dtype(np.int16) or int(version) != _SERIALIZATION_VERSION:
            raise ValueError("unsupported calibration serialization version")
        ks_array = np.asarray(arrays["ks"])
        if ks_array.ndim != 1 or ks_array.dtype != np.dtype(np.int64):
            raise ValueError("serialized ks must be a one-dimensional int64 array")
        fitted_ks = _ks(ks_array.tolist())
        expected_keys = required_base | {
            key
            for k in fitted_ks
            for key in (f"loo_distances_k_{k}", f"loo_scale_offsets_k_{k}")
        }
        if set(arrays) != expected_keys:
            raise ValueError("serialized calibration keys do not match the fitted ks")
        lambda_array = np.asarray(arrays["shrinkage_lambda"])
        if lambda_array.shape != () or lambda_array.dtype != np.dtype(np.float64):
            raise ValueError("serialized shrinkage_lambda must be a float64 scalar")
        selected_lambda = _validated_lambda(float(lambda_array))

        mean = np.asarray(arrays["mean"])
        raw_std = np.asarray(arrays["raw_std"])
        effective_std = np.asarray(arrays["effective_std"])
        zero_variance = np.asarray(arrays["zero_variance_feature_mask"])
        negative_features = np.asarray(arrays["negative_features"])
        scale_offsets = np.asarray(arrays["negative_scale_offsets"])
        dimension = len(mean) if mean.ndim == 1 else -1
        if (
            dimension < 1
            or mean.dtype != np.dtype(np.float64)
            or raw_std.shape != (dimension,)
            or raw_std.dtype != np.dtype(np.float64)
            or effective_std.shape != (dimension,)
            or effective_std.dtype != np.dtype(np.float64)
            or zero_variance.shape != (dimension,)
            or zero_variance.dtype != np.dtype(np.bool_)
            or negative_features.ndim != 2
            or negative_features.shape[0] < 1
            or negative_features.shape[1] != dimension
            or negative_features.dtype != np.dtype(np.float32)
            or scale_offsets.shape != (SCALE_COUNT + 1,)
            or scale_offsets.dtype != np.dtype(np.int64)
        ):
            raise ValueError("serialized scaler/library array contract drifted")
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(raw_std).all()
            or not np.isfinite(effective_std).all()
            or not np.isfinite(negative_features).all()
            or np.any(raw_std < 0.0)
            or np.any(effective_std <= 0.0)
            or not np.array_equal(zero_variance, raw_std < _STD_FLOOR)
            or not np.array_equal(
                effective_std,
                np.where(zero_variance, 1.0, raw_std),
            )
            or int(scale_offsets[0]) != 0
            or int(scale_offsets[-1]) != len(negative_features)
            or np.any(scale_offsets[1:] < scale_offsets[:-1])
        ):
            raise ValueError("serialized scaler/library values are invalid")

        counts = np.diff(scale_offsets)
        loo_distances: dict[int, np.ndarray] = {}
        loo_offsets: dict[int, np.ndarray] = {}
        for k in fitted_ks:
            values = np.asarray(arrays[f"loo_distances_k_{k}"])
            offsets = np.asarray(arrays[f"loo_scale_offsets_k_{k}"])
            if (
                values.ndim != 1
                or values.dtype != np.dtype(np.float32)
                or offsets.shape != (SCALE_COUNT + 1,)
                or offsets.dtype != np.dtype(np.int64)
                or int(offsets[0]) != 0
                or int(offsets[-1]) != len(values)
                or np.any(offsets[1:] < offsets[:-1])
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
            ):
                raise ValueError(f"serialized k={k} LOO array contract drifted")
            reference_counts = np.diff(offsets)
            expected_reference_counts = np.where(counts >= k + 1, counts, 0)
            if not np.array_equal(reference_counts, expected_reference_counts):
                raise ValueError(f"serialized k={k} LOO scale counts drifted")
            for scale_id in np.flatnonzero(reference_counts):
                part = values[int(offsets[scale_id]) : int(offsets[scale_id + 1])]
                if np.any(part[1:] < part[:-1]):
                    raise ValueError(f"serialized k={k} local reference is unsorted")
            loo_distances[k] = values
            loo_offsets[k] = offsets

        model = cls.__new__(cls)
        model._install_state(
            ks=fitted_ks,
            shrinkage_lambda=selected_lambda,
            mean=mean,
            raw_std=raw_std,
            effective_std=effective_std,
            zero_variance=zero_variance,
            negative_features=negative_features,
            scale_offsets=scale_offsets,
            loo_distances=loo_distances,
            loo_offsets=loo_offsets,
        )
        return model


__all__ = [
    "CALIBRATION_BLOCK_FALLBACK",
    "CALIBRATION_GLOBAL_FALLBACK",
    "CALIBRATION_LOCAL_BLOCK_SHRINK",
    "CALIBRATION_LOCAL_GLOBAL_SHRINK",
    "CALIBRATION_LOCAL_ONLY",
    "CALIBRATION_MODE_NAMES",
    "CALIBRATION_NONE",
    "DEFAULT_KS",
    "NegativeTailQueryResult",
    "SHRINKAGE_LAMBDA",
    "ScaleConditionedNegativeTailCalibrator",
    "empirical_upper_tail_probability",
]

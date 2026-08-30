"""Exact scale-conditioned one-class nearest-neighbour distances.

The fitted object contains only statistics and rows from the negative training
library.  Query batches are transformed with the frozen training scaler and
are never used to update it.  A query is compared only with negative rows that
have the exact same ``scale_id``.
"""

from __future__ import annotations

from contextlib import contextmanager
import operator
from typing import Iterator, Sequence

import numpy as np
import torch


_STD_FLOOR = 1.0e-12


def _feature_matrix(values: object, *, name: str, allow_empty: bool) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a two-dimensional numeric array") from error
    if (
        array.ndim != 2
        or array.shape[1] < 1
        or (not allow_empty and array.shape[0] < 1)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        empty_rule = "non-empty " if not allow_empty else ""
        raise ValueError(
            f"{name} must be a {empty_rule}two-dimensional real numeric array "
            "with at least one feature column"
        )
    result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN, Inf, or a value outside float32 range")
    return result


def _integer_vector(values: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional integer array") from error
    if (
        array.ndim != 1
        or not np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.issubdtype(array.dtype, np.unsignedinteger) and array.size:
        if int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{name} contains an integer outside int64 range")
    return np.asarray(array, dtype=np.int64)


def _family_vector(values: object, *, expected_length: int) -> np.ndarray:
    """Validate optional integer or text provenance labels.

    Family labels do not affect fitting or distances.  Retaining them only
    makes the exact composition of the negative library auditable.
    """

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "negative_family_ids must be a one-dimensional integer or text array"
        ) from error
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError(
            "negative_family_ids must contain one family id per negative row"
        )
    if np.issubdtype(array.dtype, np.bool_) or not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.str_)
        or np.issubdtype(array.dtype, np.bytes_)
    ):
        raise ValueError(
            "negative_family_ids must be a one-dimensional integer or text array"
        )
    return array.copy()


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


def _ks(values: Sequence[int]) -> tuple[int, ...]:
    try:
        items = tuple(values)
    except TypeError as error:
        raise ValueError("ks must be a non-empty sequence of positive integers") from error
    if not items:
        raise ValueError("ks must be a non-empty sequence of positive integers")
    result = tuple(_positive_integer(value, name="each k") for value in items)
    if len(set(result)) != len(result):
        raise ValueError("ks must not contain duplicate values")
    return result


def _torch_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid torch device {value!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must select CPU or CUDA")
    if device.type == "cpu" and device.index is not None:
        raise ValueError("an indexed CPU device is not supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA k-nearest-neighbour search requested but CUDA is unavailable"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is unavailable")
    return device


@contextmanager
def _deterministic_algorithms() -> Iterator[None]:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )


def _exact_k_distances(
    library: np.ndarray,
    query: np.ndarray,
    requested_ks: tuple[int, ...],
    *,
    device: torch.device,
    query_chunk_size: int,
    library_chunk_size: int,
) -> dict[int, np.ndarray]:
    """Compute every requested order statistic in one exact top-k pass."""

    maximum_k = max(requested_ks)
    output = {
        k: np.empty(len(query), dtype=np.float32) for k in requested_ks
    }
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
                raise RuntimeError("exact scale-conditioned search was incomplete")
            for k in requested_ks:
                output[k][query_start:query_stop] = nearest[:, k - 1].cpu().numpy()
    return output


def _standardize_for_distance(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    effective_std: np.ndarray,
    name: str,
) -> np.ndarray:
    """Apply frozen float64 statistics and return finite float32 distance rows."""

    standardized64 = (features.astype(np.float64) - mean) / effective_std
    standardized = np.ascontiguousarray(standardized64, dtype=np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError(f"standardized {name} contain a value outside float32 range")
    return standardized


class ScaleConditionedNegativeKNN:
    """Frozen global scaler plus exact same-scale negative k-nearest neighbours.

    Feature rows are first canonicalized to finite float32.  All negative rows
    then contribute once to a float64 global population mean and population
    standard deviation.  No class, scale, or family is balanced or subsampled.
    Missing scales and scales with fewer rows than the largest requested ``k``
    are rejected before distance calculation; no synthetic distance is
    manufactured as class evidence.
    """

    def __init__(
        self,
        negative_features: object,
        negative_scale_ids: object,
        negative_family_ids: object | None = None,
    ) -> None:
        features = _feature_matrix(
            negative_features, name="negative_features", allow_empty=False
        )
        scale_ids = _integer_vector(
            negative_scale_ids, name="negative_scale_ids"
        )
        if len(scale_ids) != len(features):
            raise ValueError(
                "negative_scale_ids must contain one scale id per negative row"
            )
        family_ids = (
            None
            if negative_family_ids is None
            else _family_vector(negative_family_ids, expected_length=len(features))
        )

        features64 = features.astype(np.float64)
        mean = features64.mean(axis=0, dtype=np.float64)
        raw_std = features64.std(axis=0, dtype=np.float64, ddof=0)
        zero_variance_feature_mask = raw_std < _STD_FLOOR
        effective_std = raw_std.copy()
        effective_std[zero_variance_feature_mask] = 1.0
        normalized = np.ascontiguousarray(
            (features64 - mean) / effective_std, dtype=np.float32
        )
        if not np.isfinite(normalized).all():
            raise ValueError(
                "standardized negative_features contain a value outside float32 range"
            )

        # Store each scale in one contiguous stable slice.  Building 2000
        # ``flatnonzero(scale_ids == value)`` arrays for every nested fit would
        # rescan a multi-million-row library billions of times.
        scale_order = np.argsort(scale_ids, kind="stable")
        sorted_scale_ids = np.ascontiguousarray(scale_ids[scale_order])
        unique_scales, starts, counts = np.unique(
            sorted_scale_ids, return_index=True, return_counts=True
        )
        self._negative_features = np.ascontiguousarray(normalized[scale_order])
        self._negative_scale_ids = sorted_scale_ids
        self._negative_family_ids = (
            None if family_ids is None else np.ascontiguousarray(family_ids[scale_order])
        )
        self._mean = mean.copy()
        self._raw_std = raw_std.copy()
        self._effective_std = effective_std.copy()
        self._zero_variance_feature_mask = zero_variance_feature_mask.copy()
        self._scale_slices = {
            int(scale_id): slice(int(start), int(start + count))
            for scale_id, start, count in zip(unique_scales, starts, counts)
        }
        self._scale_counts = {
            int(scale_id): int(count)
            for scale_id, count in zip(unique_scales, counts)
        }

    @property
    def fit_audit(self) -> dict[str, object]:
        """Return a mutation-safe, JSON-compatible account of fitted inputs."""

        audit: dict[str, object] = {
            "count": int(len(self._negative_features)),
            "dim": int(self._negative_features.shape[1]),
            "canonical_feature_dtype": "float32",
            "scaler_statistics_dtype": "float64",
            "distance_dtype": "float32",
            "std_ddof": 0,
            "scales": tuple(sorted(self._scale_slices)),
            "scale_counts": {
                scale_id: self._scale_counts[scale_id]
                for scale_id in sorted(self._scale_slices)
            },
            "mean": self._mean.tolist(),
            "raw_std": self._raw_std.tolist(),
            "effective_std": self._effective_std.tolist(),
            "std_floor_exclusive": _STD_FLOOR,
            "zero_variance_feature_mask": self._zero_variance_feature_mask.tolist(),
        }
        if self._negative_family_ids is not None:
            families, counts = np.unique(
                self._negative_family_ids.astype(str), return_counts=True
            )
            audit["family_counts"] = {
                str(family): int(count)
                for family, count in zip(families.tolist(), counts.tolist())
            }
        return audit

    def query(
        self,
        features: object,
        scale_ids: object,
        ks: Sequence[int] = (1, 5, 15),
        device: str | torch.device = "cpu",
        query_chunk_size: int = 1024,
        library_chunk_size: int = 8192,
    ) -> dict[int, np.ndarray]:
        """Return exact same-scale k-th negative distances for every query row.

        The dictionary preserves the order of ``ks``.  Each array preserves the
        query input order.  Every queried scale must support ``max(ks)``;
        otherwise the whole call raises :class:`ValueError` before computing
        any distance.
        """

        query_features = _feature_matrix(
            features, name="features", allow_empty=True
        )
        if query_features.shape[1] != self._negative_features.shape[1]:
            raise ValueError(
                "features and fitted negative_features must have the same width"
            )
        query_scale_ids = _integer_vector(scale_ids, name="scale_ids")
        if len(query_scale_ids) != len(query_features):
            raise ValueError("scale_ids must contain one scale id per query row")
        requested_ks = _ks(ks)
        query_chunk = _positive_integer(
            query_chunk_size, name="query_chunk_size"
        )
        library_chunk = _positive_integer(
            library_chunk_size, name="library_chunk_size"
        )
        required_max_k = max(requested_ks)
        insufficient_support: list[tuple[int, int]] = []
        for scale_id in np.unique(query_scale_ids):
            available_count = self._scale_counts.get(int(scale_id), 0)
            if available_count < required_max_k:
                insufficient_support.append((int(scale_id), available_count))
        if insufficient_support:
            details = "; ".join(
                f"scale={scale_id}, available={available_count}, "
                f"required_max_k={required_max_k}"
                for scale_id, available_count in insufficient_support
            )
            raise ValueError(f"insufficient same-scale negative support: {details}")
        selected_device = _torch_device(device)

        normalized = _standardize_for_distance(
            query_features,
            mean=self._mean,
            effective_std=self._effective_std,
            name="features",
        )
        output = {
            k: np.zeros(len(normalized), dtype=np.float32) for k in requested_ks
        }
        if len(normalized) == 0:
            return output

        query_order = np.argsort(query_scale_ids, kind="stable")
        sorted_query_scales = query_scale_ids[query_order]
        sorted_normalized = np.ascontiguousarray(normalized[query_order])
        unique_query_scales, query_starts, query_counts = np.unique(
            sorted_query_scales, return_index=True, return_counts=True
        )
        for scale_id, query_start, query_count in zip(
            unique_query_scales, query_starts, query_counts
        ):
            query_rows = query_order[
                int(query_start) : int(query_start + query_count)
            ]
            library_slice = self._scale_slices.get(int(scale_id))
            if library_slice is None:
                raise RuntimeError("same-scale support preflight was inconsistent")
            distances = _exact_k_distances(
                self._negative_features[library_slice],
                sorted_normalized[
                    int(query_start) : int(query_start + query_count)
                ],
                requested_ks,
                device=selected_device,
                query_chunk_size=query_chunk,
                library_chunk_size=library_chunk,
            )
            for k in requested_ks:
                output[k][query_rows] = distances[k]

        for values in output.values():
            if not np.isfinite(values).all():
                raise RuntimeError(
                    "exact scale-conditioned search produced NaN or Inf"
                )
        return output


__all__ = ["ScaleConditionedNegativeKNN"]

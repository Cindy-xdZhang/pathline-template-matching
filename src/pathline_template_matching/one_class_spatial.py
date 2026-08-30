"""Deterministic one-class distance scores and masked spatial post-processing.

The functions in this module deliberately contain no learned parameters.  They
operate on a library of negative (non-vortex) descriptors and on valid centers
from one regular ``(z, y, x)`` seed grid.
"""

from __future__ import annotations

from contextlib import contextmanager
import operator
from typing import Iterator, Sequence

import numpy as np
import torch


def _finite_numeric_vector(values: object, *, name: str) -> np.ndarray:
    """Return a finite float64 vector without accepting coercible text/bools."""

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from error
    if (
        array.ndim != 1
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a one-dimensional real numeric array")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _integer_vector(values: object, *, name: str) -> np.ndarray:
    """Return an int64 vector while rejecting floats, bools, and overflow."""

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
    if np.issubdtype(array.dtype, np.unsignedinteger):
        maximum = int(array.max()) if array.size else 0
        if maximum > np.iinfo(np.int64).max:
            raise ValueError(f"{name} contains an integer outside int64 range")
    return np.asarray(array, dtype=np.int64)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    result = int(result)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


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
                "CUDA k-nearest-neighbor search requested but CUDA is unavailable"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is unavailable")
    return device


@contextmanager
def _deterministic_algorithms() -> Iterator[None]:
    """Temporarily require deterministic PyTorch kernels, restoring the caller."""

    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )


def negative_knn_scores(
    train_features: object,
    query_features: object,
    k: int = 1,
    device: str | torch.device = "cpu",
    query_chunk_size: int = 1024,
    library_chunk_size: int = 8192,
) -> np.ndarray:
    """Return each query's exact k-th distance to the negative library.

    Every query-library pair is evaluated with direct float32 Euclidean
    distance; chunks only bound memory and never prune candidates.  The result
    is therefore independent of either chunk size.  CUDA execution requires a
    deterministic PyTorch kernel and fails instead of silently relaxing that
    requirement.
    """

    train = _feature_matrix(train_features, name="train_features", allow_empty=False)
    query = _feature_matrix(query_features, name="query_features", allow_empty=True)
    if query.shape[1] != train.shape[1]:
        raise ValueError(
            "query_features and train_features must have the same feature width"
        )
    neighbour_count = _positive_integer(k, name="k")
    if neighbour_count > len(train):
        raise ValueError("k cannot exceed the number of negative training rows")
    query_chunk = _positive_integer(query_chunk_size, name="query_chunk_size")
    library_chunk = _positive_integer(
        library_chunk_size, name="library_chunk_size"
    )
    selected_device = _torch_device(device)
    if len(query) == 0:
        return np.empty(0, dtype=np.float32)

    output = np.empty(len(query), dtype=np.float32)
    with torch.inference_mode(), _deterministic_algorithms():
        for query_start in range(0, len(query), query_chunk):
            query_stop = min(query_start + query_chunk, len(query))
            query_tensor = torch.from_numpy(query[query_start:query_stop]).to(
                selected_device
            )
            nearest = torch.empty(
                (len(query_tensor), 0), dtype=torch.float32, device=selected_device
            )
            for library_start in range(0, len(train), library_chunk):
                library_stop = min(library_start + library_chunk, len(train))
                library_tensor = torch.from_numpy(
                    train[library_start:library_stop]
                ).to(selected_device)
                distances = torch.cdist(
                    query_tensor,
                    library_tensor,
                    p=2.0,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                candidates = torch.cat((nearest, distances), dim=1)
                keep = min(neighbour_count, candidates.shape[1])
                nearest = torch.topk(
                    candidates, k=keep, dim=1, largest=False, sorted=True
                ).values
            if nearest.shape[1] != neighbour_count:
                raise RuntimeError("exact k-nearest-neighbor search was incomplete")
            output[query_start:query_stop] = nearest[:, -1].cpu().numpy()
    if not np.isfinite(output).all():
        raise RuntimeError("exact k-nearest-neighbor search produced NaN or Inf")
    return output


def rank_scores(scores: object, center_seed_indices: object) -> np.ndarray:
    """Map scores to deterministic empirical percentiles in ``(0, 1]``.

    Rows are ordered by ascending score and then by ascending center seed index.
    The row at zero-based rank ``r`` receives ``(r + 1) / N``.  Input row order
    is used only as a tertiary key when both documented keys are identical.
    """

    values = _finite_numeric_vector(scores, name="scores")
    centers = _integer_vector(center_seed_indices, name="center_seed_indices")
    if centers.shape != values.shape:
        raise ValueError("center_seed_indices must contain one index per score")
    if len(np.unique(centers)) != len(centers):
        raise ValueError("center_seed_indices must not contain duplicates")
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    order = np.lexsort((centers, values))
    result = np.empty(len(values), dtype=np.float64)
    result[order] = (
        np.arange(1, len(values) + 1, dtype=np.float64) / float(len(values))
    )
    return result


def _grid_shape(value: Sequence[int]) -> tuple[int, int, int]:
    try:
        items = tuple(value)
    except TypeError as error:
        raise ValueError("grid_shape must contain three positive integers (z, y, x)") from error
    if len(items) != 3:
        raise ValueError("grid_shape must contain three positive integers (z, y, x)")
    return tuple(
        _positive_integer(item, name=f"grid_shape[{axis}]")
        for axis, item in enumerate(items)
    )  # type: ignore[return-value]


def _finite_nonnegative_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite non-negative scalar") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
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


def masked_gaussian_grid_scores(
    scores: object,
    center_indices: object,
    grid_shape: Sequence[int] = (40, 40, 40),
    *,
    sigma: float,
    truncate: float = 3.0,
) -> np.ndarray:
    """Smooth valid grid scores without treating missing centers as zero.

    ``center_indices`` are C-order flat indices into ``grid_shape=(Z,Y,X)``:
    ``index = (z * Y + y) * X + x``.  A Gaussian is applied separately to a
    score grid and its validity mask, and the former is divided by the latter.
    Only the input valid rows are returned, in their original order.
    """

    values = _finite_numeric_vector(scores, name="scores")
    centers = _integer_vector(center_indices, name="center_indices")
    if centers.shape != values.shape:
        raise ValueError("center_indices must contain one index per score")
    shape = _grid_shape(grid_shape)
    total_size = int(np.prod(shape, dtype=object))
    if np.any(centers < 0) or np.any(centers >= total_size):
        raise ValueError("center_indices contains an index outside grid_shape")
    if len(np.unique(centers)) != len(centers):
        raise ValueError("center_indices must not contain duplicates")
    width = _finite_nonnegative_scalar(sigma, name="sigma")
    truncation = _finite_nonnegative_scalar(truncate, name="truncate")
    if truncation <= 0.0:
        raise ValueError("truncate must be strictly positive")
    if len(values) == 0 or width == 0.0:
        return values.copy()

    dense_scores = np.zeros(shape, dtype=np.float64)
    dense_mask = np.zeros(shape, dtype=np.float64)
    dense_scores.ravel(order="C")[centers] = values
    dense_mask.ravel(order="C")[centers] = 1.0
    kernel = _gaussian_kernel(width, truncation)
    numerator = _separable_gaussian(dense_scores, kernel)
    denominator = _separable_gaussian(dense_mask, kernel)
    selected_denominator = denominator.ravel(order="C")[centers]
    if np.any(selected_denominator <= 0.0) or not np.isfinite(
        selected_denominator
    ).all():
        raise RuntimeError("masked Gaussian normalization has invalid support")
    result = numerator.ravel(order="C")[centers] / selected_denominator
    if not np.isfinite(result).all():
        raise RuntimeError("masked Gaussian smoothing produced NaN or Inf")
    return result


def high_score_two_means_predictions(scores: object) -> np.ndarray:
    """Return the high-mean cluster from deterministic, exact one-dimensional 2-means.

    All distinct-value split points are evaluated with the global within-cluster
    sum-of-squares objective.  Equal optima choose the lowest sorted split.  An
    empty, singleton, or constant input cannot identify a high cluster and
    therefore fails closed by returning only ``False``.
    """

    values = _finite_numeric_vector(scores, name="scores")
    predictions = np.zeros(len(values), dtype=bool)
    if len(values) < 2 or np.all(values == values[0]):
        return predictions

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    # SSE from raw prefix sums loses the within-group variation when every
    # score has a large common offset.  Subtracting the minimum is exact for
    # nearby floating-point values (Sterbenz's lemma) and makes the objective
    # invariant to translation.  Only opposite-sign extreme values can make
    # that subtraction overflow; scale first in that exceptional case.
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = sorted_values - sorted_values[0]
    if not np.isfinite(normalized).all():
        scale = float(np.max(np.abs(sorted_values)))
        scaled = sorted_values / scale
        normalized = scaled - scaled[0]
    span = float(normalized[-1])
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("two-means normalization failed for distinct scores")
    normalized = normalized / span
    prefix = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(normalized, dtype=np.float64))
    )
    prefix_square = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.square(normalized), dtype=np.float64),
        )
    )
    split_positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:]) + 1
    left_count = split_positions.astype(np.float64)
    right_count = float(len(values)) - left_count
    left_sum = prefix[split_positions]
    right_sum = prefix[-1] - left_sum
    left_sse = prefix_square[split_positions] - np.square(left_sum) / left_count
    right_sse = (
        prefix_square[-1]
        - prefix_square[split_positions]
        - np.square(right_sum) / right_count
    )
    objectives = np.maximum(left_sse, 0.0) + np.maximum(right_sse, 0.0)
    split = int(split_positions[int(np.argmin(objectives))])
    predictions[order[split:]] = True
    return predictions

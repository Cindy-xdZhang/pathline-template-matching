"""Exact per-scale negative metric with frozen negative-tail calibration.

This module implements the only numerical change preregistered by
``Verify_PerScaleNegativeMetric_1.1``.  A fit-negative scaler estimates one
diagonal within-scale population variance for each of the 2000 exact scale
IDs.  The variance is shrunk toward a within-block, per-scale-centred prior
with the frozen variance-domain weight ``n / (n + 64)``.  The already-frozen
negative-tail calibration is then fitted to those transformed features with
an identity global transform, so no second global feature weighting is
introduced.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch

from .negative_tail_calibration import (
    DEFAULT_KS,
    SCALE_BLOCK_SIZE,
    SCALE_COUNT,
    SHRINKAGE_LAMBDA,
    NegativeTailQueryResult,
    ScaleConditionedNegativeTailCalibrator,
    _validated_lambda,
    _validated_scale_ids,
)
from .scale_one_class import (
    _STD_FLOOR,
    _feature_matrix,
    _ks,
    _positive_integer,
    _torch_device,
)


SCALER_NO_LOCAL_ROWS = np.int8(0)
SCALER_LOCAL_BLOCK_SHRINK = np.int8(1)
SCALER_LOCAL_GLOBAL_SHRINK = np.int8(2)
SCALER_LOCAL_ONLY = np.int8(3)

SCALER_MODE_NAMES = MappingProxyType(
    {
        int(SCALER_NO_LOCAL_ROWS): "no_local_rows",
        int(SCALER_LOCAL_BLOCK_SHRINK): "local_block_shrink",
        int(SCALER_LOCAL_GLOBAL_SHRINK): "local_global_shrink",
        int(SCALER_LOCAL_ONLY): "local_only",
    }
)

SCALER_ARRAY_NAMES = (
    "scale_id_int32",
    "local_row_count_int64",
    "block_other_row_count_int64",
    "global_other_row_count_int64",
    "local_support_bool",
    "scaler_mode_int8",
    "local_mean_float64",
    "local_variance_float64",
    "prior_variance_float64",
    "shrunk_variance_float64",
    "effective_std_float64",
)


def _freeze_array(
    values: object, *, dtype: np.dtype | type | None = None
) -> np.ndarray:
    output = np.ascontiguousarray(values, dtype=dtype).copy()
    output.setflags(write=False)
    return output


class PerScaleNegativeScaler:
    """Fit-only exact-scale diagonal population-variance scaler."""

    def __init__(
        self,
        negative_features: object,
        negative_scale_ids: object,
        *,
        shrinkage_lambda: float = SHRINKAGE_LAMBDA,
    ) -> None:
        features = _feature_matrix(
            negative_features, name="negative_features", allow_empty=False
        )
        scale_ids = _validated_scale_ids(
            negative_scale_ids, name="negative_scale_ids"
        )
        if len(features) != len(scale_ids):
            raise ValueError(
                "negative_scale_ids must contain one scale id per negative row"
            )
        selected_lambda = _validated_lambda(shrinkage_lambda)
        dimension = int(features.shape[1])
        scale_order = np.argsort(scale_ids, kind="stable")
        sorted_scales = np.asarray(scale_ids[scale_order], dtype=np.int64)
        sorted_features = np.asarray(features[scale_order], dtype=np.float32)
        counts = np.bincount(sorted_scales, minlength=SCALE_COUNT).astype(
            np.int64, copy=False
        )
        offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
        )

        means = np.zeros((SCALE_COUNT, dimension), dtype=np.float64)
        local_variances = np.zeros_like(means)
        within_scale_squared_residuals = np.zeros_like(means)
        for scale_id in np.flatnonzero(counts):
            start = int(offsets[scale_id])
            stop = int(offsets[scale_id + 1])
            part = sorted_features[start:stop].astype(np.float64)
            mean = part.mean(axis=0, dtype=np.float64)
            residual = part - mean
            squared_residual_sum = np.sum(
                np.square(residual), axis=0, dtype=np.float64
            )
            means[scale_id] = mean
            local_variances[scale_id] = squared_residual_sum / float(len(part))

        # Re-express each within-scale residual sum from the serialized
        # ddof=0 variance and count.  This is algebraically identical and
        # makes the prior/shrink formulas exactly reconstructible from the
        # frozen scaler artifact alone.
        within_scale_squared_residuals = (
            local_variances * counts[:, None].astype(np.float64)
        )

        block_counts = np.asarray(
            [
                counts[0:SCALE_BLOCK_SIZE].sum(dtype=np.int64),
                counts[SCALE_BLOCK_SIZE:SCALE_COUNT].sum(dtype=np.int64),
            ],
            dtype=np.int64,
        )
        block_residual_sums = np.asarray(
            [
                within_scale_squared_residuals[0:SCALE_BLOCK_SIZE].sum(
                    axis=0, dtype=np.float64
                ),
                within_scale_squared_residuals[SCALE_BLOCK_SIZE:SCALE_COUNT].sum(
                    axis=0, dtype=np.float64
                ),
            ],
            dtype=np.float64,
        )
        total_count = int(counts.sum(dtype=np.int64))
        block_other_counts = np.empty(SCALE_COUNT, dtype=np.int64)
        global_other_counts = total_count - counts
        support = counts > 0
        modes = np.full(SCALE_COUNT, SCALER_NO_LOCAL_ROWS, dtype=np.int8)
        prior_variances = np.zeros_like(means)
        shrunk_variances = np.zeros_like(means)
        effective_std = np.ones_like(means)

        for scale_id in range(SCALE_COUNT):
            block_index = scale_id // SCALE_BLOCK_SIZE
            local_count = int(counts[scale_id])
            block_other_count = int(block_counts[block_index]) - local_count
            global_other_count = int(global_other_counts[scale_id])
            block_other_counts[scale_id] = block_other_count
            if local_count == 0:
                continue
            if block_other_count > 0:
                other_residual_sum = (
                    block_residual_sums[block_index]
                    - within_scale_squared_residuals[scale_id]
                )
                if np.any(other_residual_sum < 0.0):
                    raise RuntimeError("block-other residual sum became negative")
                prior = other_residual_sum / float(block_other_count)
                mode = SCALER_LOCAL_BLOCK_SHRINK
            elif global_other_count > 0:
                # There are exactly two frozen blocks.  With no other rows in
                # the current block, global-other is the opposite block and
                # already excludes the current scale without subtraction.
                other_block = 1 - block_index
                if int(block_counts[other_block]) != global_other_count:
                    raise RuntimeError("global-other row-count identity drifted")
                prior = block_residual_sums[other_block] / float(
                    global_other_count
                )
                mode = SCALER_LOCAL_GLOBAL_SHRINK
            else:
                prior = np.zeros(dimension, dtype=np.float64)
                mode = SCALER_LOCAL_ONLY
            local_variance = local_variances[scale_id]
            if mode == SCALER_LOCAL_ONLY:
                shrunk = local_variance
            else:
                weight = local_count / float(local_count + selected_lambda)
                shrunk = weight * local_variance + (1.0 - weight) * prior
            if np.any(prior < 0.0) or np.any(shrunk < 0.0):
                raise RuntimeError("per-scale variance became negative")
            raw_std = np.sqrt(shrunk)
            current_effective_std = np.where(raw_std < _STD_FLOOR, 1.0, raw_std)
            prior_variances[scale_id] = prior
            shrunk_variances[scale_id] = shrunk
            effective_std[scale_id] = current_effective_std
            modes[scale_id] = mode

        self._install_state(
            shrinkage_lambda=selected_lambda,
            scale_ids=np.arange(SCALE_COUNT, dtype=np.int32),
            local_counts=counts,
            block_other_counts=block_other_counts,
            global_other_counts=global_other_counts,
            support=support,
            modes=modes,
            means=means,
            local_variances=local_variances,
            prior_variances=prior_variances,
            shrunk_variances=shrunk_variances,
            effective_std=effective_std,
        )

    def _install_state(
        self,
        *,
        shrinkage_lambda: float,
        scale_ids: np.ndarray,
        local_counts: np.ndarray,
        block_other_counts: np.ndarray,
        global_other_counts: np.ndarray,
        support: np.ndarray,
        modes: np.ndarray,
        means: np.ndarray,
        local_variances: np.ndarray,
        prior_variances: np.ndarray,
        shrunk_variances: np.ndarray,
        effective_std: np.ndarray,
    ) -> None:
        self._shrinkage_lambda = float(shrinkage_lambda)
        self._scale_ids = _freeze_array(scale_ids, dtype=np.int32)
        self._local_counts = _freeze_array(local_counts, dtype=np.int64)
        self._block_other_counts = _freeze_array(
            block_other_counts, dtype=np.int64
        )
        self._global_other_counts = _freeze_array(
            global_other_counts, dtype=np.int64
        )
        self._support = _freeze_array(support, dtype=np.bool_)
        self._modes = _freeze_array(modes, dtype=np.int8)
        self._means = _freeze_array(means, dtype=np.float64)
        self._local_variances = _freeze_array(
            local_variances, dtype=np.float64
        )
        self._prior_variances = _freeze_array(
            prior_variances, dtype=np.float64
        )
        self._shrunk_variances = _freeze_array(
            shrunk_variances, dtype=np.float64
        )
        self._effective_std = _freeze_array(effective_std, dtype=np.float64)

    @property
    def dimension(self) -> int:
        return int(self._means.shape[1])

    @property
    def shrinkage_lambda(self) -> float:
        return self._shrinkage_lambda

    @property
    def local_row_counts(self) -> np.ndarray:
        return self._local_counts

    @property
    def modes(self) -> np.ndarray:
        return self._modes

    @property
    def fit_audit(self) -> dict[str, object]:
        return {
            "scale_count": SCALE_COUNT,
            "feature_dimension": self.dimension,
            "negative_row_count": int(self._local_counts.sum(dtype=np.int64)),
            "shrinkage_lambda": self._shrinkage_lambda,
            "local_variance_ddof": 0,
            "shrinkage_domain": "variance_before_square_root",
            "std_floor_exclusive": _STD_FLOOR,
            "local_supported_scale_count": int(self._support.sum()),
            "scaler_mode_counts": {
                str(mode): int(np.count_nonzero(self._modes == mode))
                for mode in range(4)
            },
            "absent_scale_numeric_placeholders": {
                "local_mean": 0.0,
                "local_variance": 0.0,
                "prior_variance": 0.0,
                "shrunk_variance": 0.0,
                "effective_std": 1.0,
            },
        }

    def transform(self, features: object, scale_ids: object) -> np.ndarray:
        matrix = _feature_matrix(features, name="features", allow_empty=True)
        scales = _validated_scale_ids(scale_ids, name="scale_ids")
        if len(matrix) != len(scales):
            raise ValueError("scale_ids must contain one scale id per feature row")
        if matrix.shape[1] != self.dimension:
            raise ValueError("features and fitted scaler must have the same width")
        transformed = np.zeros(matrix.shape, dtype=np.float64)
        supported_rows = self._support[scales]
        transformed[supported_rows] = (
            matrix[supported_rows].astype(np.float64)
            - self._means[scales[supported_rows]]
        ) / self._effective_std[scales[supported_rows]]
        # No-local rows never consume their serialized placeholders.  A
        # neutral zero is passed onward, and the empty exact-scale library
        # offsets force the frozen raw/tail unsupported sentinels.
        result = np.ascontiguousarray(transformed, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("per-scale transformed features exceed float32 range")
        return result

    def mode_for_scales(self, scale_ids: object) -> np.ndarray:
        scales = _validated_scale_ids(scale_ids, name="scale_ids")
        return np.ascontiguousarray(self._modes[scales], dtype=np.int8)

    def export_arrays(self) -> dict[str, np.ndarray]:
        values = (
            self._scale_ids,
            self._local_counts,
            self._block_other_counts,
            self._global_other_counts,
            self._support,
            self._modes,
            self._means,
            self._local_variances,
            self._prior_variances,
            self._shrunk_variances,
            self._effective_std,
        )
        return {
            name: np.array(value, copy=True, order="C")
            for name, value in zip(SCALER_ARRAY_NAMES, values)
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, object]) -> "PerScaleNegativeScaler":
        if not isinstance(arrays, Mapping) or set(arrays) != set(
            SCALER_ARRAY_NAMES
        ):
            raise ValueError("serialized per-scale scaler member set drifted")
        scale_ids = np.asarray(arrays["scale_id_int32"])
        local_counts = np.asarray(arrays["local_row_count_int64"])
        block_other_counts = np.asarray(arrays["block_other_row_count_int64"])
        global_other_counts = np.asarray(arrays["global_other_row_count_int64"])
        support = np.asarray(arrays["local_support_bool"])
        modes = np.asarray(arrays["scaler_mode_int8"])
        means = np.asarray(arrays["local_mean_float64"])
        local_variances = np.asarray(arrays["local_variance_float64"])
        prior_variances = np.asarray(arrays["prior_variance_float64"])
        shrunk_variances = np.asarray(arrays["shrunk_variance_float64"])
        effective_std = np.asarray(arrays["effective_std_float64"])
        dimension = means.shape[1] if means.ndim == 2 else -1
        vector_contract = (
            scale_ids.shape == (SCALE_COUNT,)
            and scale_ids.dtype == np.dtype(np.int32)
            and local_counts.shape == (SCALE_COUNT,)
            and local_counts.dtype == np.dtype(np.int64)
            and block_other_counts.shape == (SCALE_COUNT,)
            and block_other_counts.dtype == np.dtype(np.int64)
            and global_other_counts.shape == (SCALE_COUNT,)
            and global_other_counts.dtype == np.dtype(np.int64)
            and support.shape == (SCALE_COUNT,)
            and support.dtype == np.dtype(np.bool_)
            and modes.shape == (SCALE_COUNT,)
            and modes.dtype == np.dtype(np.int8)
        )
        matrix_contract = dimension > 0 and all(
            value.shape == (SCALE_COUNT, dimension)
            and value.dtype == np.dtype(np.float64)
            for value in (
                means,
                local_variances,
                prior_variances,
                shrunk_variances,
                effective_std,
            )
        )
        if not vector_contract or not matrix_contract:
            raise ValueError("serialized per-scale scaler shape/dtype drifted")
        if (
            not np.array_equal(scale_ids, np.arange(SCALE_COUNT, dtype=np.int32))
            or np.any(local_counts < 0)
            or np.any(block_other_counts < 0)
            or np.any(global_other_counts < 0)
            or not np.array_equal(support, local_counts > 0)
            or np.any((modes < 0) | (modes > 3))
            or not all(
                np.isfinite(value).all()
                for value in (
                    means,
                    local_variances,
                    prior_variances,
                    shrunk_variances,
                    effective_std,
                )
            )
            or np.any(local_variances < 0.0)
            or np.any(prior_variances < 0.0)
            or np.any(shrunk_variances < 0.0)
            or np.any(effective_std <= 0.0)
        ):
            raise ValueError("serialized per-scale scaler values are invalid")
        block_totals = np.asarray(
            [
                local_counts[:SCALE_BLOCK_SIZE].sum(dtype=np.int64),
                local_counts[SCALE_BLOCK_SIZE:].sum(dtype=np.int64),
            ],
            dtype=np.int64,
        )
        expected_block_other = np.asarray(
            [block_totals[index // SCALE_BLOCK_SIZE] for index in range(SCALE_COUNT)],
            dtype=np.int64,
        ) - local_counts
        expected_global_other = int(local_counts.sum(dtype=np.int64)) - local_counts
        expected_modes = np.full(SCALE_COUNT, SCALER_NO_LOCAL_ROWS, dtype=np.int8)
        expected_modes[support & (expected_block_other > 0)] = (
            SCALER_LOCAL_BLOCK_SHRINK
        )
        expected_modes[
            support & (expected_block_other == 0) & (expected_global_other > 0)
        ] = SCALER_LOCAL_GLOBAL_SHRINK
        expected_modes[
            support & (expected_block_other == 0) & (expected_global_other == 0)
        ] = SCALER_LOCAL_ONLY
        if (
            not np.array_equal(block_other_counts, expected_block_other)
            or not np.array_equal(global_other_counts, expected_global_other)
            or not np.array_equal(modes, expected_modes)
        ):
            raise ValueError("serialized per-scale support/mode counts drifted")
        absent = ~support
        if (
            np.any(means[absent] != 0.0)
            or np.any(local_variances[absent] != 0.0)
            or np.any(prior_variances[absent] != 0.0)
            or np.any(shrunk_variances[absent] != 0.0)
            or np.any(effective_std[absent] != 1.0)
        ):
            raise ValueError("absent-scale numeric placeholders drifted")
        within_scale_squared_residuals = (
            local_variances * local_counts[:, None].astype(np.float64)
        )
        block_residual_sums = np.asarray(
            [
                within_scale_squared_residuals[:SCALE_BLOCK_SIZE].sum(
                    axis=0, dtype=np.float64
                ),
                within_scale_squared_residuals[SCALE_BLOCK_SIZE:].sum(
                    axis=0, dtype=np.float64
                ),
            ],
            dtype=np.float64,
        )
        expected_prior = np.zeros_like(prior_variances)
        expected_shrunk = np.zeros_like(shrunk_variances)
        for scale_id in np.flatnonzero(support):
            local_count = int(local_counts[scale_id])
            mode = int(modes[scale_id])
            if mode == int(SCALER_LOCAL_BLOCK_SHRINK):
                block_index = scale_id // SCALE_BLOCK_SIZE
                expected_prior[scale_id] = (
                    block_residual_sums[block_index]
                    - within_scale_squared_residuals[scale_id]
                ) / float(block_other_counts[scale_id])
            elif mode == int(SCALER_LOCAL_GLOBAL_SHRINK):
                block_index = scale_id // SCALE_BLOCK_SIZE
                expected_prior[scale_id] = block_residual_sums[
                    1 - block_index
                ] / float(global_other_counts[scale_id])
            elif mode != int(SCALER_LOCAL_ONLY):
                raise ValueError("serialized local scaler mode is invalid")
            if mode == int(SCALER_LOCAL_ONLY):
                expected_shrunk[scale_id] = local_variances[scale_id]
            else:
                weight = local_count / float(local_count + SHRINKAGE_LAMBDA)
                expected_shrunk[scale_id] = (
                    weight * local_variances[scale_id]
                    + (1.0 - weight) * expected_prior[scale_id]
                )
        if not np.array_equal(prior_variances, expected_prior):
            raise ValueError("serialized broader-prior variance formula drifted")
        if not np.array_equal(shrunk_variances, expected_shrunk):
            raise ValueError("serialized variance-domain shrinkage formula drifted")
        expected_effective = np.where(
            np.sqrt(shrunk_variances) < _STD_FLOOR,
            1.0,
            np.sqrt(shrunk_variances),
        )
        if not np.array_equal(effective_std, expected_effective):
            raise ValueError("serialized effective standard deviation drifted")
        model = cls.__new__(cls)
        model._install_state(
            shrinkage_lambda=SHRINKAGE_LAMBDA,
            scale_ids=scale_ids,
            local_counts=local_counts,
            block_other_counts=block_other_counts,
            global_other_counts=global_other_counts,
            support=support,
            modes=modes,
            means=means,
            local_variances=local_variances,
            prior_variances=prior_variances,
            shrunk_variances=shrunk_variances,
            effective_std=effective_std,
        )
        return model


def _identity_tail_calibrator(
    transformed_negative_features: np.ndarray,
    negative_scale_ids: np.ndarray,
    *,
    ks: Sequence[int],
    shrinkage_lambda: float,
    device: str | torch.device,
    query_chunk_size: int,
    library_chunk_size: int,
) -> ScaleConditionedNegativeTailCalibrator:
    """Fit the frozen tail references without a second feature transform."""

    features = _feature_matrix(
        transformed_negative_features,
        name="transformed_negative_features",
        allow_empty=False,
    )
    scales = _validated_scale_ids(
        negative_scale_ids, name="negative_scale_ids"
    )
    if len(features) != len(scales):
        raise ValueError("negative_scale_ids must contain one id per negative row")
    requested_ks = _ks(ks)
    selected_lambda = _validated_lambda(shrinkage_lambda)
    selected_device = _torch_device(device)
    query_chunk = _positive_integer(query_chunk_size, name="query_chunk_size")
    library_chunk = _positive_integer(
        library_chunk_size, name="library_chunk_size"
    )
    order = np.argsort(scales, kind="stable")
    sorted_features = np.ascontiguousarray(features[order], dtype=np.float32)
    sorted_scales = np.asarray(scales[order], dtype=np.int64)
    counts = np.bincount(sorted_scales, minlength=SCALE_COUNT).astype(
        np.int64, copy=False
    )
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    loo_distances, loo_offsets = (
        ScaleConditionedNegativeTailCalibrator._build_loo_references(
            sorted_features,
            offsets,
            requested_ks,
            device=selected_device,
            query_chunk_size=query_chunk,
            library_chunk_size=library_chunk,
        )
    )
    dimension = int(sorted_features.shape[1])
    model = ScaleConditionedNegativeTailCalibrator.__new__(
        ScaleConditionedNegativeTailCalibrator
    )
    model._install_state(
        ks=requested_ks,
        shrinkage_lambda=selected_lambda,
        mean=np.zeros(dimension, dtype=np.float64),
        raw_std=np.ones(dimension, dtype=np.float64),
        effective_std=np.ones(dimension, dtype=np.float64),
        zero_variance=np.zeros(dimension, dtype=np.bool_),
        negative_features=sorted_features,
        scale_offsets=offsets,
        loo_distances=loo_distances,
        loo_offsets=loo_offsets,
    )
    return model


def _require_identity_tail_state(
    model: ScaleConditionedNegativeTailCalibrator,
    scaler: PerScaleNegativeScaler,
) -> None:
    arrays = model.export_arrays()
    dimension = scaler.dimension
    if (
        not np.array_equal(arrays["mean"], np.zeros(dimension, dtype=np.float64))
        or not np.array_equal(
            arrays["raw_std"], np.ones(dimension, dtype=np.float64)
        )
        or not np.array_equal(
            arrays["effective_std"], np.ones(dimension, dtype=np.float64)
        )
        or np.any(arrays["zero_variance_feature_mask"])
        or arrays["shrinkage_lambda"].shape != ()
        or arrays["shrinkage_lambda"].dtype != np.dtype(np.float64)
        or float(arrays["shrinkage_lambda"]) != scaler.shrinkage_lambda
        or scaler.shrinkage_lambda != SHRINKAGE_LAMBDA
        or not np.array_equal(
            np.diff(arrays["negative_scale_offsets"]),
            scaler.local_row_counts,
        )
    ):
        raise ValueError("tail calibrator is not bound to the per-scale metric")


class PerScaleNegativeTailModel:
    """Composition of the frozen scaler and unchanged tail calibrator."""

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
        scales = _validated_scale_ids(
            negative_scale_ids, name="negative_scale_ids"
        )
        if len(features) != len(scales):
            raise ValueError(
                "negative_scale_ids must contain one scale id per negative row"
            )
        scaler = PerScaleNegativeScaler(
            features, scales, shrinkage_lambda=shrinkage_lambda
        )
        transformed = scaler.transform(features, scales)
        tail = _identity_tail_calibrator(
            transformed,
            scales,
            ks=ks,
            shrinkage_lambda=shrinkage_lambda,
            device=device,
            query_chunk_size=query_chunk_size,
            library_chunk_size=library_chunk_size,
        )
        self._install_state(scaler=scaler, tail=tail)

    def _install_state(
        self,
        *,
        scaler: PerScaleNegativeScaler,
        tail: ScaleConditionedNegativeTailCalibrator,
    ) -> None:
        _require_identity_tail_state(tail, scaler)
        self._scaler = scaler
        self._tail = tail

    @property
    def ks(self) -> tuple[int, ...]:
        return self._tail.ks

    @property
    def scaler(self) -> PerScaleNegativeScaler:
        return self._scaler

    @property
    def tail_calibrator(self) -> ScaleConditionedNegativeTailCalibrator:
        return self._tail

    @property
    def fit_audit(self) -> dict[str, object]:
        return {
            "per_scale_scaler": self._scaler.fit_audit,
            "negative_tail_calibration": self._tail.fit_audit,
        }

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
        scales = _validated_scale_ids(scale_ids, name="scale_ids")
        transformed = self._scaler.transform(features, scales)
        return self._tail.query(
            transformed,
            scales,
            ks=ks,
            device=device,
            query_chunk_size=query_chunk_size,
            library_chunk_size=library_chunk_size,
        )

    @classmethod
    def from_artifacts(
        cls,
        scaler_arrays: Mapping[str, object],
        tail_arrays: Mapping[str, object],
    ) -> "PerScaleNegativeTailModel":
        scaler = PerScaleNegativeScaler.from_arrays(scaler_arrays)
        tail = ScaleConditionedNegativeTailCalibrator.from_arrays(tail_arrays)
        model = cls.__new__(cls)
        model._install_state(scaler=scaler, tail=tail)
        return model


__all__ = [
    "PerScaleNegativeScaler",
    "PerScaleNegativeTailModel",
    "SCALER_ARRAY_NAMES",
    "SCALER_LOCAL_BLOCK_SHRINK",
    "SCALER_LOCAL_GLOBAL_SHRINK",
    "SCALER_LOCAL_ONLY",
    "SCALER_MODE_NAMES",
    "SCALER_NO_LOCAL_ROWS",
]

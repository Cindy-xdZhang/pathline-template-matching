"""Label-free source-centered seed-time kinematics for assigned rows.

The input is the complete assigned-row population, not the valid-pathline
subset.  Rows are grouped by ``dataset x source x scale block x dx level``;
every group must contain exactly 6,400 rows.  The arithmetic mean curl vector
of each group is subtracted before the curl magnitude and centered Q-like
coordinate are evaluated.

This module performs pure array computation.  It does not load flow data,
inspect labels, or read and write sidecar archives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .early_opposite_pair_kinematics import (
    DERIVATIVE_DTYPE,
    FROZEN_PRIMITIVE_ORDER,
    KINEMATIC_OUTPUT_DTYPE,
    PHYSICAL_DX_DTYPE,
    VELOCITY_INPUT_DTYPE,
    compute_seed_time_velocity_gradient,
)


ASSIGNED_ROWS_PER_SOURCE_BLOCK_DX_LEVEL = 6400
GROUP_ID_DTYPE = np.dtype(np.int32)
GROUP_INDEX_DTYPE = np.dtype(np.int32)
GROUP_COUNT_DTYPE = np.dtype(np.int32)
GROUP_MEAN_DTYPE = DERIVATIVE_DTYPE
FROZEN_SOURCE_GROUP_ID_ORDER = (
    "dataset_index",
    "source_ordinal",
    "scale_block_index",
    "dx_level_index",
)
FROZEN_SOURCE_CENTERED_KINEMATIC_FEATURE_ORDER = (
    "l2_norm_of_curl_deviation_from_source_block_dx_mean",
    "frobenius_norm_of_strain",
    "signed_divergence",
    "source_centered_signed_Q",
)


def _freeze(values: object, *, dtype: np.dtype | type) -> np.ndarray:
    selected_dtype = np.dtype(dtype)
    copied = np.array(values, dtype=selected_dtype, order="C", copy=True)
    result = np.frombuffer(
        copied.tobytes(order="C"), dtype=selected_dtype
    ).reshape(copied.shape)
    result.setflags(write=False)
    return result


def _require_exact_array(
    values: object,
    *,
    name: str,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
    finite: bool = False,
) -> np.ndarray:
    raw = np.asarray(values)
    expected_dtype = np.dtype(dtype)
    if raw.dtype != expected_dtype or raw.shape != shape:
        raise ValueError(
            f"{name} must have dtype {expected_dtype} and shape {shape}; "
            f"got dtype {raw.dtype} and shape {raw.shape}"
        )
    if finite and not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    return _freeze(raw, dtype=expected_dtype)


def _validated_row_group_ids(group_ids: object, row_count: int) -> np.ndarray:
    groups = _require_exact_array(
        group_ids,
        name="group_ids",
        dtype=GROUP_ID_DTYPE,
        shape=(row_count, len(FROZEN_SOURCE_GROUP_ID_ORDER)),
    )
    if row_count == 0:
        raise ValueError("the assigned-row population must not be empty")
    if np.any(groups[:, :2] < 0):
        raise ValueError("dataset_index and source_ordinal must be nonnegative")
    if np.any((groups[:, 2] < 0) | (groups[:, 2] > 1)):
        raise ValueError("scale_block_index must contain only 0 or 1")
    if np.any((groups[:, 3] < 0) | (groups[:, 3] > 9)):
        raise ValueError("dx_level_index must lie in 0..9")
    return groups


def _group_contract(
    row_group_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse, counts = np.unique(
        row_group_ids, axis=0, return_inverse=True, return_counts=True
    )
    if len(unique) == 0:
        raise ValueError("the assigned-row population must contain a group")
    if np.any(counts != ASSIGNED_ROWS_PER_SOURCE_BLOCK_DX_LEVEL):
        bad = [
            {
                "group": [int(value) for value in unique[index]],
                "assigned_rows": int(counts[index]),
            }
            for index in np.flatnonzero(
                counts != ASSIGNED_ROWS_PER_SOURCE_BLOCK_DX_LEVEL
            )
        ]
        raise ValueError(
            "each dataset x source x block x dx-level group must contain "
            f"exactly {ASSIGNED_ROWS_PER_SOURCE_BLOCK_DX_LEVEL} assigned rows; "
            f"mismatches={bad}"
        )
    if len(unique) > np.iinfo(GROUP_INDEX_DTYPE).max:
        raise ValueError("group count exceeds the frozen int32 group-index range")
    return (
        _freeze(unique, dtype=GROUP_ID_DTYPE),
        _freeze(inverse, dtype=GROUP_INDEX_DTYPE),
        _freeze(counts, dtype=GROUP_COUNT_DTYPE),
    )


def _curl_from_gradient(gradient: np.ndarray) -> np.ndarray:
    curl = np.stack(
        (
            gradient[:, 2, 1] - gradient[:, 1, 2],
            gradient[:, 0, 2] - gradient[:, 2, 0],
            gradient[:, 1, 0] - gradient[:, 0, 1],
        ),
        axis=1,
    )
    if not np.isfinite(curl).all():
        raise ValueError("curl computation produced nonfinite values")
    return np.ascontiguousarray(curl, dtype=DERIVATIVE_DTYPE)


def _arithmetic_group_mean_curl(
    curl_xyz: np.ndarray,
    row_group_index: np.ndarray,
    group_row_count: np.ndarray,
) -> np.ndarray:
    sums = np.zeros((len(group_row_count), 3), dtype=DERIVATIVE_DTYPE)
    np.add.at(sums, np.asarray(row_group_index), curl_xyz)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        means = sums / np.asarray(group_row_count, dtype=DERIVATIVE_DTYPE)[:, None]
    if not np.isfinite(means).all():
        raise ValueError("group arithmetic mean curl produced nonfinite values")
    return _freeze(means, dtype=GROUP_MEAN_DTYPE)


@dataclass(frozen=True)
class SourceCenteredSeedTimeKinematics:
    """Immutable pure-computation payload for a future authenticated sidecar."""

    row_group_ids: np.ndarray
    unique_group_ids: np.ndarray
    row_group_index: np.ndarray
    group_row_count: np.ndarray
    group_mean_curl_xyz: np.ndarray
    source_centered_kinematic4: np.ndarray

    def __post_init__(self) -> None:
        row_group_raw = np.asarray(self.row_group_ids)
        if row_group_raw.ndim != 2 or row_group_raw.shape[1:] != (
            len(FROZEN_SOURCE_GROUP_ID_ORDER),
        ):
            raise ValueError("row_group_ids must have shape [N,4]")
        row_count = len(row_group_raw)
        row_groups = _validated_row_group_ids(row_group_raw, row_count)
        expected_unique, expected_inverse, expected_counts = _group_contract(row_groups)
        group_count = len(expected_unique)
        arrays = {
            "unique_group_ids": _require_exact_array(
                self.unique_group_ids,
                name="unique_group_ids",
                dtype=GROUP_ID_DTYPE,
                shape=(group_count, len(FROZEN_SOURCE_GROUP_ID_ORDER)),
            ),
            "row_group_index": _require_exact_array(
                self.row_group_index,
                name="row_group_index",
                dtype=GROUP_INDEX_DTYPE,
                shape=(row_count,),
            ),
            "group_row_count": _require_exact_array(
                self.group_row_count,
                name="group_row_count",
                dtype=GROUP_COUNT_DTYPE,
                shape=(group_count,),
            ),
            "group_mean_curl_xyz": _require_exact_array(
                self.group_mean_curl_xyz,
                name="group_mean_curl_xyz",
                dtype=GROUP_MEAN_DTYPE,
                shape=(group_count, 3),
                finite=True,
            ),
            "source_centered_kinematic4": _require_exact_array(
                self.source_centered_kinematic4,
                name="source_centered_kinematic4",
                dtype=KINEMATIC_OUTPUT_DTYPE,
                shape=(row_count, 4),
                finite=True,
            ),
        }
        if not np.array_equal(arrays["unique_group_ids"], expected_unique):
            raise ValueError("unique_group_ids do not match row_group_ids")
        if not np.array_equal(arrays["row_group_index"], expected_inverse):
            raise ValueError("row_group_index does not match row_group_ids")
        if not np.array_equal(arrays["group_row_count"], expected_counts):
            raise ValueError("group_row_count does not match row_group_ids")
        object.__setattr__(self, "row_group_ids", row_groups)
        for name, values in arrays.items():
            object.__setattr__(self, name, values)


def compute_source_centered_seed_time_kinematics(
    seed_velocity_xyz: object,
    physical_dx: object,
    group_ids: object,
    *,
    primitive_order: Sequence[str],
) -> SourceCenteredSeedTimeKinematics:
    """Compute source-centered four-coordinate kinematics for assigned rows.

    ``group_ids`` has the frozen columns ``dataset_index, source_ordinal,
    scale_block_index, dx_level_index`` and dtype int32.  The first feature is
    ``||curl(v) - group_mean(curl(v))||``.  The last feature is exactly
    ``0.25 * curl_deviation_squared - 0.5 * strain_squared``.
    """

    velocity = np.asarray(seed_velocity_xyz)
    if velocity.ndim != 3 or velocity.shape[1:] != (7, 3):
        raise ValueError("seed_velocity_xyz must have shape [N,7,3]")
    row_count = len(velocity)
    if velocity.dtype != VELOCITY_INPUT_DTYPE:
        raise ValueError("seed_velocity_xyz must have dtype float32")
    dx = np.asarray(physical_dx)
    if dx.dtype != PHYSICAL_DX_DTYPE or dx.shape != (row_count,):
        raise ValueError("physical_dx must have dtype float64 and shape [N]")
    row_groups = _validated_row_group_ids(group_ids, row_count)
    unique_groups, inverse, counts = _group_contract(row_groups)

    gradient = compute_seed_time_velocity_gradient(
        velocity,
        dx,
        primitive_order=primitive_order,
    )
    curl = _curl_from_gradient(gradient)
    group_mean = _arithmetic_group_mean_curl(curl, inverse, counts)
    curl_deviation = curl - group_mean[np.asarray(inverse)]
    transpose = np.swapaxes(gradient, 1, 2)
    strain = 0.5 * (gradient + transpose)
    with np.errstate(invalid="ignore", over="ignore"):
        curl_deviation_squared = np.sum(
            np.square(curl_deviation), axis=1, dtype=DERIVATIVE_DTYPE
        )
        strain_squared = np.sum(
            np.square(strain), axis=(1, 2), dtype=DERIVATIVE_DTYPE
        )
        feature64 = np.stack(
            (
                np.sqrt(curl_deviation_squared),
                np.sqrt(strain_squared),
                np.trace(gradient, axis1=1, axis2=2),
                0.25 * curl_deviation_squared - 0.5 * strain_squared,
            ),
            axis=1,
        )
    if not np.isfinite(feature64).all():
        raise ValueError("source-centered invariant computation produced nonfinite values")
    feature32 = _freeze(feature64, dtype=KINEMATIC_OUTPUT_DTYPE)
    if not np.isfinite(feature32).all():
        raise ValueError("float32 source-centered serialization produced nonfinite values")
    return SourceCenteredSeedTimeKinematics(
        row_group_ids=row_groups,
        unique_group_ids=unique_groups,
        row_group_index=inverse,
        group_row_count=counts,
        group_mean_curl_xyz=group_mean,
        source_centered_kinematic4=feature32,
    )


def validate_source_centered_seed_time_kinematics(
    seed_velocity_xyz: object,
    physical_dx: object,
    group_ids: object,
    payload: SourceCenteredSeedTimeKinematics,
    *,
    primitive_order: Sequence[str],
) -> None:
    """Recompute and exactly validate saved group means and all payload arrays."""

    if not isinstance(payload, SourceCenteredSeedTimeKinematics):
        raise ValueError("payload must be SourceCenteredSeedTimeKinematics")
    expected = compute_source_centered_seed_time_kinematics(
        seed_velocity_xyz,
        physical_dx,
        group_ids,
        primitive_order=primitive_order,
    )
    for name in (
        "row_group_ids",
        "unique_group_ids",
        "row_group_index",
        "group_row_count",
        "group_mean_curl_xyz",
        "source_centered_kinematic4",
    ):
        if not np.array_equal(
            np.asarray(getattr(payload, name)), np.asarray(getattr(expected, name))
        ):
            raise ValueError(f"saved {name} does not match exact recomputation")


__all__ = [
    "ASSIGNED_ROWS_PER_SOURCE_BLOCK_DX_LEVEL",
    "FROZEN_SOURCE_CENTERED_KINEMATIC_FEATURE_ORDER",
    "FROZEN_SOURCE_GROUP_ID_ORDER",
    "GROUP_COUNT_DTYPE",
    "GROUP_ID_DTYPE",
    "GROUP_INDEX_DTYPE",
    "GROUP_MEAN_DTYPE",
    "SourceCenteredSeedTimeKinematics",
    "compute_source_centered_seed_time_kinematics",
    "validate_source_centered_seed_time_kinematics",
]

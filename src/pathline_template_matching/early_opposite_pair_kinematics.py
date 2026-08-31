"""Seed-time kinematics from a frozen seven-point opposite-pair stencil.

The input order is ``center, x+, x-, y+, y-, z+, z-``.  Each row contains
the seven synchronous initial velocities of one pathline primitive.  Spatial
derivatives are evaluated in float64 by opposite-pair central differences;
the public four-coordinate block is serialized as float32 in the frozen order
``curl norm, strain Frobenius norm, signed divergence, signed Q``.

This module is deliberately independent of flow loading, pathline integration,
labels, and query-batch statistics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


FROZEN_PRIMITIVE_ORDER = (
    "center",
    "x_plus",
    "x_minus",
    "y_plus",
    "y_minus",
    "z_plus",
    "z_minus",
)
FROZEN_KINEMATIC_FEATURE_ORDER = (
    "l2_norm_of_curl",
    "frobenius_norm_of_strain",
    "signed_divergence",
    "signed_Q",
)
VELOCITY_INPUT_DTYPE = np.dtype(np.float32)
PHYSICAL_DX_DTYPE = np.dtype(np.float64)
DERIVATIVE_DTYPE = np.dtype(np.float64)
KINEMATIC_OUTPUT_DTYPE = np.dtype(np.float32)


def _validated_primitive_order(primitive_order: Sequence[str]) -> None:
    if isinstance(primitive_order, (str, bytes)):
        raise ValueError(
            "primitive_order must explicitly equal "
            "center,x_plus,x_minus,y_plus,y_minus,z_plus,z_minus"
        )
    try:
        selected = tuple(primitive_order)
    except TypeError as error:
        raise ValueError("primitive_order must be a seven-name sequence") from error
    if selected != FROZEN_PRIMITIVE_ORDER:
        raise ValueError(
            "primitive_order must explicitly equal "
            "center,x_plus,x_minus,y_plus,y_minus,z_plus,z_minus"
        )


def _validated_velocity_copy(seed_velocity_xyz: object) -> np.ndarray:
    raw = np.asarray(seed_velocity_xyz)
    if raw.ndim != 3 or raw.shape[1:] != (7, 3) or raw.shape[0] == 0:
        raise ValueError(
            "seed_velocity_xyz must have nonempty shape [N,7,3] in the frozen "
            "primitive order"
        )
    if raw.dtype != VELOCITY_INPUT_DTYPE:
        raise ValueError("seed_velocity_xyz must have dtype float32")
    if not np.isfinite(raw).all():
        raise ValueError("seed_velocity_xyz must contain only finite values")
    return np.array(raw, dtype=DERIVATIVE_DTYPE, order="C", copy=True)


def _validated_physical_dx_copy(physical_dx: object, row_count: int) -> np.ndarray:
    raw = np.asarray(physical_dx)
    if raw.dtype != PHYSICAL_DX_DTYPE:
        raise ValueError("physical_dx must have dtype float64")
    if raw.ndim == 0:
        result = np.full(row_count, float(raw), dtype=PHYSICAL_DX_DTYPE)
    elif raw.ndim == 1 and raw.shape == (row_count,):
        result = np.array(raw, dtype=PHYSICAL_DX_DTYPE, order="C", copy=True)
    else:
        raise ValueError(
            "physical_dx must be a float64 scalar or a vector with shape [N]"
        )
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("physical_dx must contain only positive finite values")
    return result


def _freeze_array(values: object, *, dtype: np.dtype) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        copied = np.array(values, dtype=dtype, order="C", copy=True)
    result = np.frombuffer(copied.tobytes(order="C"), dtype=dtype).reshape(
        copied.shape
    )
    result.setflags(write=False)
    return result


def compute_seed_time_velocity_gradient(
    seed_velocity_xyz: object,
    physical_dx: object,
    *,
    primitive_order: Sequence[str],
) -> np.ndarray:
    """Return the float64 velocity gradient with shape ``[N,3,3]``.

    Gradient rows are velocity components and columns are derivatives with
    respect to x, y, and z.  ``primitive_order`` is mandatory so a caller must
    bind the semantic order of its seven velocity samples explicitly.
    """

    _validated_primitive_order(primitive_order)
    velocity = _validated_velocity_copy(seed_velocity_xyz)
    dx = _validated_physical_dx_copy(physical_dx, len(velocity))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gradient = np.stack(
            (
                velocity[:, 1, :] - velocity[:, 2, :],
                velocity[:, 3, :] - velocity[:, 4, :],
                velocity[:, 5, :] - velocity[:, 6, :],
            ),
            axis=2,
        ) / (2.0 * dx[:, None, None])
    if not np.isfinite(gradient).all():
        raise ValueError("central differences produced a nonfinite gradient")
    return _freeze_array(gradient, dtype=DERIVATIVE_DTYPE)


def compute_seed_time_kinematic4(
    seed_velocity_xyz: object,
    physical_dx: object,
    *,
    primitive_order: Sequence[str],
) -> np.ndarray:
    """Return the frozen read-only float32 kinematic block with shape ``[N,4]``.

    The four columns are the L2 norm of curl, Frobenius norm of the symmetric
    strain tensor, signed divergence, and signed Q criterion.  No mean
    vorticity subtraction, logarithm, flow statistic, or batch statistic is
    applied.
    """

    gradient = compute_seed_time_velocity_gradient(
        seed_velocity_xyz,
        physical_dx,
        primitive_order=primitive_order,
    )
    transpose = np.swapaxes(gradient, 1, 2)
    strain = 0.5 * (gradient + transpose)
    rotation = 0.5 * (gradient - transpose)
    curl = np.stack(
        (
            gradient[:, 2, 1] - gradient[:, 1, 2],
            gradient[:, 0, 2] - gradient[:, 2, 0],
            gradient[:, 1, 0] - gradient[:, 0, 1],
        ),
        axis=1,
    )
    with np.errstate(invalid="ignore", over="ignore"):
        curl_norm = np.sqrt(
            np.sum(np.square(curl), axis=1, dtype=DERIVATIVE_DTYPE)
        )
        strain_squared = np.sum(
            np.square(strain), axis=(1, 2), dtype=DERIVATIVE_DTYPE
        )
        rotation_squared = np.sum(
            np.square(rotation), axis=(1, 2), dtype=DERIVATIVE_DTYPE
        )
        strain_norm = np.sqrt(strain_squared)
        divergence = np.trace(gradient, axis1=1, axis2=2)
        signed_q = 0.5 * (rotation_squared - strain_squared)
        feature64 = np.stack(
            (curl_norm, strain_norm, divergence, signed_q), axis=1
        )
    if not np.isfinite(feature64).all():
        raise ValueError("kinematic invariant computation produced nonfinite values")
    feature32 = _freeze_array(feature64, dtype=KINEMATIC_OUTPUT_DTYPE)
    if not np.isfinite(feature32).all():
        raise ValueError("float32 kinematic serialization produced nonfinite values")
    return feature32


__all__ = [
    "DERIVATIVE_DTYPE",
    "FROZEN_KINEMATIC_FEATURE_ORDER",
    "FROZEN_PRIMITIVE_ORDER",
    "KINEMATIC_OUTPUT_DTYPE",
    "PHYSICAL_DX_DTYPE",
    "VELOCITY_INPUT_DTYPE",
    "compute_seed_time_kinematic4",
    "compute_seed_time_velocity_gradient",
]

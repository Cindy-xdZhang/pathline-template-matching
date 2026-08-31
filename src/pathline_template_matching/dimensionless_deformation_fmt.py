"""Frozen row-local dimensionless-deformation input for independent FMT.

``Verify_DimensionlessDeformationFMT_1.1`` changes only the coordinates passed
to the inherited independent FMT descriptor.  A float32 Raw672 row is restored
in C order as ``center, x+, x-, y+, y-, z+, z-``.  In float64 arithmetic, the
center trajectory is divided by its realized polyline arc length and each
neighbor-minus-center trajectory is divided by the realized initial neighbor
distance.  The transformed primitive is serialized as float32 before it enters
the unchanged independent FMT implementation.

The transform has no fitted state and accepts no labels, IVD values, scale IDs,
dataset metadata, or batch statistics.  Every output row therefore depends on
exactly one input row.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .encoder import IndependentFMT3DConfig, encode_independent_fmt_3d
from .nested_scale_validation import representation_indices, select_representation


RAW_INPUT_DTYPE = np.dtype(np.float32)
ARITHMETIC_DTYPE = np.dtype(np.float64)
SERIALIZATION_DTYPE = np.dtype(np.float32)
RAW_INPUT_WIDTH = 672
PRIMITIVE_SHAPE = (7, 32, 3)
FROZEN_PRIMITIVE_ORDER = (
    "center",
    "x_plus",
    "x_minus",
    "y_plus",
    "y_minus",
    "z_plus",
    "z_minus",
)
GEOMETRY_RTOL = 5.0e-5
GEOMETRY_ATOL = 1.0e-7
PARENT_DESCRIPTOR_ID = "fmt_independent_3d_161d_sha256_25fce29499c9089e"

REPRESENTATION_NAMES = (
    "fmt161_dimensionless_deformation",
    "real_neighbor36_dimensionless_deformation",
    "chirality_all35_dimensionless_deformation",
)
_PARENT_TO_OUTPUT = (
    ("fmt161", REPRESENTATION_NAMES[0]),
    ("real_neighbor36", REPRESENTATION_NAMES[1]),
    ("chirality_all35", REPRESENTATION_NAMES[2]),
)
PARENT_REPRESENTATION_INDEX_SETS = MappingProxyType(
    {
        output_name: representation_indices(parent_name)
        for parent_name, output_name in _PARENT_TO_OUTPUT
    }
)


def _validate_primitive_order(primitive_order: Sequence[str]) -> None:
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


def _validated_raw672_float64_copy(raw_features: object) -> np.ndarray:
    try:
        raw = np.asarray(raw_features)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "raw_features must be a nonempty float32 array with shape [N,672]"
        ) from error
    if (
        raw.ndim != 2
        or raw.shape[0] == 0
        or raw.shape[1] != RAW_INPUT_WIDTH
        or raw.dtype != RAW_INPUT_DTYPE
    ):
        raise ValueError(
            "raw_features must be a nonempty float32 array with shape [N,672]"
        )
    if not np.isfinite(raw).all():
        raise ValueError("raw_features contains NaN or Inf")
    return np.array(raw, dtype=ARITHMETIC_DTYPE, order="C", copy=True).reshape(
        (-1, *PRIMITIVE_SHAPE), order="C"
    )


def _immutable_float32(values: object, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        serialized = np.array(
            values, dtype=SERIALIZATION_DTYPE, order="C", copy=True
        )
    if serialized.shape != expected_shape:
        raise RuntimeError(
            f"dimensionless output shape changed: expected {expected_shape}, "
            f"got {serialized.shape}"
        )
    if not np.isfinite(serialized).all():
        raise ValueError("float32 dimensionless serialization produced NaN or Inf")
    immutable = np.frombuffer(
        serialized.tobytes(order="C"), dtype=SERIALIZATION_DTYPE
    ).reshape(expected_shape)
    immutable.setflags(write=False)
    return immutable


def _dimensionless_deformation_float32(
    raw_features: object,
    *,
    primitive_order: Sequence[str],
) -> np.ndarray:
    _validate_primitive_order(primitive_order)
    primitive = _validated_raw672_float64_copy(raw_features)
    row_count = len(primitive)

    center = primitive[:, 0]
    if not np.equal(center[:, 0], 0.0).all():
        raise ValueError("every center path sample 0 must be exactly zero")

    with np.errstate(invalid="ignore", over="ignore"):
        center_segments = center[:, 1:] - center[:, :-1]
        center_segment_lengths = np.sqrt(
            np.sum(
                np.square(center_segments),
                axis=2,
                dtype=ARITHMETIC_DTYPE,
            )
        )
        center_arc_length = np.sum(
            center_segment_lengths, axis=1, dtype=ARITHMETIC_DTYPE
        )
    if (
        not np.isfinite(center_segment_lengths).all()
        or not np.isfinite(center_arc_length).all()
        or np.any(center_arc_length <= 0.0)
    ):
        raise ValueError("realized center arc length must be finite and positive")

    initial_relative = primitive[:, 1:, 0] - center[:, None, 0]
    with np.errstate(invalid="ignore", over="ignore"):
        initial_neighbor_norms = np.sqrt(
            np.sum(
                np.square(initial_relative),
                axis=2,
                dtype=ARITHMETIC_DTYPE,
            )
        )
    if not np.isfinite(initial_neighbor_norms).all():
        raise ValueError("realized initial neighbor distances must be finite")
    if not np.allclose(
        initial_neighbor_norms,
        initial_neighbor_norms[:, :1],
        rtol=GEOMETRY_RTOL,
        atol=GEOMETRY_ATOL,
    ):
        raise ValueError("the six realized initial neighbor distances are unequal")
    for plus_index, minus_index in ((0, 1), (2, 3), (4, 5)):
        if not np.allclose(
            initial_relative[:, plus_index],
            -initial_relative[:, minus_index],
            rtol=GEOMETRY_RTOL,
            atol=GEOMETRY_ATOL,
        ):
            raise ValueError(
                "each initial opposite-pair midpoint must equal the center"
            )
    realized_initial_dx = np.mean(
        initial_neighbor_norms, axis=1, dtype=ARITHMETIC_DTYPE
    )
    if (
        not np.isfinite(realized_initial_dx).all()
        or np.any(realized_initial_dx <= 0.0)
    ):
        raise ValueError("realized initial dx must be finite and positive")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        center_dimensionless = center / center_arc_length[:, None, None]
        relative_trajectories = primitive[:, 1:] - center[:, None]
        neighbor_dimensionless = (
            center_dimensionless[:, None]
            + relative_trajectories / realized_initial_dx[:, None, None, None]
        )
        transformed = np.empty(
            (row_count, *PRIMITIVE_SHAPE), dtype=ARITHMETIC_DTYPE
        )
        transformed[:, 0] = center_dimensionless
        transformed[:, 1:] = neighbor_dimensionless
    if not np.isfinite(transformed).all():
        raise ValueError("dimensionless deformation transform produced NaN or Inf")

    # The one permitted dtype boundary is float64 arithmetic to float32
    # serialization.  No clipping, epsilon, logarithm, or fitted statistic is
    # applied before this conversion.
    with np.errstate(invalid="ignore", over="ignore"):
        result = np.array(
            transformed, dtype=SERIALIZATION_DTYPE, order="C", copy=True
        )
    if not np.isfinite(result).all():
        raise ValueError("float32 dimensionless serialization produced NaN or Inf")
    return result


def transform_raw672_to_dimensionless_deformation(
    raw_features: object,
    *,
    primitive_order: Sequence[str],
) -> np.ndarray:
    """Return immutable float32 ``[N,7,32,3]`` dimensionless primitives.

    ``primitive_order`` is mandatory so callers must bind the semantic layout
    rather than relying only on the numeric width of Raw672.
    """

    transformed = _dimensionless_deformation_float32(
        raw_features, primitive_order=primitive_order
    )
    return _immutable_float32(transformed, expected_shape=transformed.shape)


def encode_dimensionless_deformation_fmt(
    raw_features: object,
    *,
    primitive_order: Sequence[str],
) -> Mapping[str, np.ndarray]:
    """Apply the unchanged FMT and return its three frozen representations.

    The returned mapping is read-only, its keys follow the preregistered order,
    and every array is an immutable bytes-backed float32 view.
    """

    transformed = _dimensionless_deformation_float32(
        raw_features, primitive_order=primitive_order
    )
    descriptor = IndependentFMT3DConfig()
    if descriptor.descriptor_id != PARENT_DESCRIPTOR_ID:
        raise RuntimeError("the inherited independent FMT descriptor ID changed")
    fmt161 = encode_independent_fmt_3d(transformed, descriptor)
    if fmt161.shape != (len(transformed), 161) or fmt161.dtype != SERIALIZATION_DTYPE:
        raise RuntimeError("the inherited independent FMT width or dtype changed")

    result: dict[str, np.ndarray] = {}
    for parent_name, output_name in _PARENT_TO_OUTPUT:
        selected = select_representation(fmt161, parent_name)
        expected_width = len(PARENT_REPRESENTATION_INDEX_SETS[output_name])
        result[output_name] = _immutable_float32(
            selected, expected_shape=(len(transformed), expected_width)
        )
    if tuple(result) != REPRESENTATION_NAMES:
        raise RuntimeError("dimensionless FMT representation order changed")
    return MappingProxyType(result)


__all__ = [
    "ARITHMETIC_DTYPE",
    "FROZEN_PRIMITIVE_ORDER",
    "GEOMETRY_ATOL",
    "GEOMETRY_RTOL",
    "PARENT_DESCRIPTOR_ID",
    "PARENT_REPRESENTATION_INDEX_SETS",
    "PRIMITIVE_SHAPE",
    "RAW_INPUT_DTYPE",
    "RAW_INPUT_WIDTH",
    "REPRESENTATION_NAMES",
    "SERIALIZATION_DTYPE",
    "encode_dimensionless_deformation_fmt",
    "transform_raw672_to_dimensionless_deformation",
]

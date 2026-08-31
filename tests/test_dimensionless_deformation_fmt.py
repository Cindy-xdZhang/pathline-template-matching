from __future__ import annotations

import inspect

import numpy as np

from pathline_template_matching.dimensionless_deformation_fmt import (
    FROZEN_PRIMITIVE_ORDER,
    PARENT_DESCRIPTOR_ID,
    PARENT_REPRESENTATION_INDEX_SETS,
    REPRESENTATION_NAMES,
    encode_dimensionless_deformation_fmt,
    transform_raw672_to_dimensionless_deformation,
)
from pathline_template_matching.encoder import (
    IndependentFMT3DConfig,
    encode_independent_fmt_3d,
)
from pathline_template_matching.nested_scale_validation import (
    representation_indices,
)


_INITIAL_OFFSETS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _raw672(row_count: int = 5) -> np.ndarray:
    time = np.arange(32, dtype=np.float64)
    rows = np.empty((row_count, 7, 32, 3), dtype=np.float64)
    for row_index in range(row_count):
        speed = 0.04 * (row_index + 1)
        center = np.stack(
            (
                speed * time,
                0.003 * (row_index + 1) * np.square(time),
                0.02 * np.sin(time * (row_index + 1) / 13.0),
            ),
            axis=1,
        )
        center[0] = 0.0
        dx = 0.125 * (row_index + 1)
        rows[row_index, 0] = center
        for neighbor_index, initial_direction in enumerate(_INITIAL_OFFSETS):
            temporal_scale = 1.0 + (neighbor_index + 1) * 0.002 * time
            deformation = (
                dx * temporal_scale[:, None] * initial_direction[None, :]
            )
            deformation += (
                0.0004
                * (neighbor_index + 1)
                * time[:, None]
                * np.asarray([1.0, -0.5, 0.25], dtype=np.float64)[None, :]
            )
            rows[row_index, neighbor_index + 1] = center + deformation
    return np.ascontiguousarray(rows.reshape(row_count, 672), dtype=np.float32)


def _transform(raw: object) -> np.ndarray:
    return transform_raw672_to_dimensionless_deformation(
        raw, primitive_order=FROZEN_PRIMITIVE_ORDER
    )


def _encode(raw: object):
    return encode_dimensionless_deformation_fmt(
        raw, primitive_order=FROZEN_PRIMITIVE_ORDER
    )


def test_transform_matches_the_frozen_rowwise_analytic_formula() -> None:
    raw = _raw672(3)
    original = raw.copy()
    observed = _transform(raw)
    primitive64 = raw.astype(np.float64).reshape(3, 7, 32, 3)
    expected64 = np.empty_like(primitive64)

    for row_index, primitive in enumerate(primitive64):
        center = primitive[0]
        center_arc = sum(
            float(np.linalg.norm(center[index + 1] - center[index]))
            for index in range(31)
        )
        initial_distances = [
            float(np.linalg.norm(primitive[line_index, 0] - center[0]))
            for line_index in range(1, 7)
        ]
        realized_dx = sum(initial_distances) / 6.0
        expected64[row_index, 0] = center / center_arc
        for line_index in range(1, 7):
            expected64[row_index, line_index] = (
                center / center_arc
                + (primitive[line_index] - center) / realized_dx
            )

    np.testing.assert_array_equal(observed, expected64.astype(np.float32))
    np.testing.assert_array_equal(raw, original)
    assert observed.shape == (3, 7, 32, 3)
    assert observed.dtype == np.float32
    assert not observed.flags.owndata
    assert not observed.flags.writeable
    assert not np.shares_memory(observed, raw)
    try:
        observed.setflags(write=True)
    except ValueError:
        pass
    else:
        raise AssertionError("dimensionless output could be made writeable")


def test_unit_scale_and_proper_rigid_rotation_invariance() -> None:
    raw = _raw672(4)
    scaled = np.asarray(raw * np.float32(8.0), dtype=np.float32)
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    primitive = raw.reshape(-1, 7, 32, 3)
    rotated = np.ascontiguousarray(
        (primitive @ rotation.T).reshape(-1, 672), dtype=np.float32
    )

    base_transform = _transform(raw)
    np.testing.assert_array_equal(base_transform, _transform(scaled))
    np.testing.assert_array_equal(
        base_transform @ rotation.T,
        _transform(rotated),
    )

    base_features = _encode(raw)
    scaled_features = _encode(scaled)
    rotated_features = _encode(rotated)
    for name in REPRESENTATION_NAMES:
        np.testing.assert_array_equal(base_features[name], scaled_features[name])
        np.testing.assert_allclose(
            base_features[name], rotated_features[name], rtol=2.0e-5, atol=2.0e-5
        )


def test_batch_chunk_single_and_permutation_outputs_are_bitwise_identical() -> None:
    raw = _raw672(6)
    batch_transform = _transform(raw)
    chunk_transform = np.concatenate((_transform(raw[:2]), _transform(raw[2:])))
    single_transform = np.concatenate(
        [_transform(raw[index : index + 1]) for index in range(len(raw))]
    )
    permutation = np.asarray([4, 1, 5, 0, 3, 2], dtype=np.int64)
    inverse = np.argsort(permutation)
    permuted_transform = _transform(raw[permutation])[inverse]
    np.testing.assert_array_equal(batch_transform, chunk_transform)
    np.testing.assert_array_equal(batch_transform, single_transform)
    np.testing.assert_array_equal(batch_transform, permuted_transform)

    batch_features = _encode(raw)
    chunk_features = [_encode(raw[:2]), _encode(raw[2:])]
    single_features = [_encode(raw[index : index + 1]) for index in range(len(raw))]
    permuted_features = _encode(raw[permutation])
    for name in REPRESENTATION_NAMES:
        np.testing.assert_array_equal(
            batch_features[name],
            np.concatenate([chunk[name] for chunk in chunk_features]),
        )
        np.testing.assert_array_equal(
            batch_features[name],
            np.concatenate([single[name] for single in single_features]),
        )
        np.testing.assert_array_equal(
            batch_features[name], permuted_features[name][inverse]
        )


def test_shape_dtype_finite_origin_and_explicit_order_fail_closed() -> None:
    valid = _raw672(2)
    malformed_inputs = (
        np.zeros((672,), dtype=np.float32),
        np.zeros((0, 672), dtype=np.float32),
        np.zeros((2, 671), dtype=np.float32),
        np.zeros((2, 673), dtype=np.float32),
        valid.astype(np.float64),
        np.zeros((2, 672), dtype=np.int64),
        np.zeros((2, 672), dtype=bool),
    )
    for malformed in malformed_inputs:
        _expect_value_error(_transform, malformed)

    for nonfinite in (np.nan, np.inf, -np.inf):
        malformed = valid.copy()
        malformed[0, 217] = nonfinite
        _expect_value_error(_transform, malformed)

    nonzero_origin = valid.copy().reshape(2, 7, 32, 3)
    nonzero_origin[0, 0, 0, 1] = np.float32(1.0e-6)
    _expect_value_error(_transform, nonzero_origin.reshape(2, 672))

    wrong_order = (
        "center",
        "x_minus",
        "x_plus",
        "y_plus",
        "y_minus",
        "z_plus",
        "z_minus",
    )
    _expect_value_error(
        transform_raw672_to_dimensionless_deformation,
        valid,
        primitive_order=wrong_order,
    )
    _expect_value_error(
        transform_raw672_to_dimensionless_deformation,
        valid,
        primitive_order="center,x_plus,x_minus,y_plus,y_minus,z_plus,z_minus",
    )


def test_degenerate_arc_dx_unequal_and_nonopposite_geometry_fail_closed() -> None:
    base = _raw672(1).reshape(1, 7, 32, 3)

    zero_arc = base.copy()
    zero_arc[:, 0] = 0.0
    _expect_value_error(_transform, zero_arc.reshape(1, 672))

    zero_dx = base.copy()
    zero_dx[:, 1:, 0] = zero_dx[:, :1, 0]
    _expect_value_error(_transform, zero_dx.reshape(1, 672))

    unequal = base.copy()
    unequal[:, 1, 0] *= np.float32(1.01)
    _expect_value_error(_transform, unequal.reshape(1, 672))

    equal_but_nonopposite = base.copy()
    dx = float(np.linalg.norm(equal_but_nonopposite[0, 1, 0]))
    equal_but_nonopposite[0, 2, 0] = np.asarray(
        [0.0, -dx, 0.0], dtype=np.float32
    )
    _expect_value_error(_transform, equal_but_nonopposite.reshape(1, 672))

    overflow_at_float32_boundary = base.copy()
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    overflow_at_float32_boundary[0, 1:, 0] = tiny * _INITIAL_OFFSETS.astype(
        np.float32
    )
    overflow_at_float32_boundary[0, 1:, 1:, 0] = np.finfo(np.float32).max
    _expect_value_error(
        _transform, overflow_at_float32_boundary.reshape(1, 672)
    )


def test_fmt_widths_values_parent_indices_and_immutable_mapping_are_exact() -> None:
    raw = _raw672(3)
    transformed = _transform(raw)
    direct_fmt = encode_independent_fmt_3d(np.array(transformed, copy=True))
    observed = _encode(raw)

    assert IndependentFMT3DConfig().descriptor_id == PARENT_DESCRIPTOR_ID
    assert tuple(observed) == REPRESENTATION_NAMES
    expected_parent_names = ("fmt161", "real_neighbor36", "chirality_all35")
    expected_widths = (161, 36, 35)
    for output_name, parent_name, width in zip(
        REPRESENTATION_NAMES, expected_parent_names, expected_widths
    ):
        expected_indices = representation_indices(parent_name)
        assert PARENT_REPRESENTATION_INDEX_SETS[output_name] == expected_indices
        assert observed[output_name].shape == (3, width)
        assert observed[output_name].dtype == np.float32
        assert not observed[output_name].flags.writeable
        assert not observed[output_name].flags.owndata
        np.testing.assert_array_equal(
            observed[output_name], direct_fmt[:, expected_indices]
        )
        try:
            observed[output_name].setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{output_name} could be made writeable")
    try:
        observed["new_representation"] = direct_fmt  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("representation mapping was writeable")


def test_public_api_cannot_receive_fit_statistics_labels_ivd_or_metadata() -> None:
    transform_signature = inspect.signature(
        transform_raw672_to_dimensionless_deformation
    )
    encoder_signature = inspect.signature(encode_dimensionless_deformation_fmt)
    assert tuple(transform_signature.parameters) == ("raw_features", "primitive_order")
    assert tuple(encoder_signature.parameters) == ("raw_features", "primitive_order")
    for forbidden in (
        "label",
        "ivd",
        "scale",
        "dataset",
        "mean",
        "std",
        "epsilon",
        "clip",
        "log",
    ):
        assert forbidden not in str(transform_signature).lower()
        assert forbidden not in str(encoder_signature).lower()
    try:
        encode_dimensionless_deformation_fmt(
            _raw672(1),
            primitive_order=FROZEN_PRIMITIVE_ORDER,
            labels=np.asarray([1]),  # type: ignore[call-arg]
        )
    except TypeError:
        pass
    else:
        raise AssertionError("encoder accepted labels")

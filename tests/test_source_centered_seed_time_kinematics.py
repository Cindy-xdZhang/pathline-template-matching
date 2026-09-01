from __future__ import annotations

from dataclasses import replace

import numpy as np

from pathline_template_matching.early_opposite_pair_kinematics import (
    FROZEN_PRIMITIVE_ORDER,
)
from pathline_template_matching.source_centered_seed_time_kinematics import (
    ASSIGNED_ROWS_PER_SOURCE_BLOCK_DX_LEVEL,
    FROZEN_SOURCE_CENTERED_KINEMATIC_FEATURE_ORDER,
    FROZEN_SOURCE_GROUP_ID_ORDER,
    SourceCenteredSeedTimeKinematics,
    compute_source_centered_seed_time_kinematics,
    validate_source_centered_seed_time_kinematics,
)


_OFFSETS = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ),
    dtype=np.float64,
)
_ROWS = ASSIGNED_ROWS_PER_SOURCE_BLOCK_DX_LEVEL


def _expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _two_group_population() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = 2 * _ROWS
    group_ids = np.empty((count, 4), dtype=np.int32)
    group_ids[:_ROWS] = np.asarray([0, 0, 0, 0], dtype=np.int32)
    group_ids[_ROWS:] = np.asarray([1, 3, 1, 7], dtype=np.int32)
    physical_dx = np.concatenate(
        (
            np.full(_ROWS, 0.5, dtype=np.float64),
            np.full(_ROWS, 1.0, dtype=np.float64),
        )
    )

    curl_z = np.empty(count, dtype=np.float64)
    curl_z[: _ROWS // 2] = -2.0
    curl_z[_ROWS // 2 : _ROWS] = 2.0
    curl_z[_ROWS : _ROWS + _ROWS // 2] = 2.0
    curl_z[_ROWS + _ROWS // 2 :] = 6.0
    matrices = np.zeros((count, 3, 3), dtype=np.float64)
    matrices[:, 0, 1] = -0.5 * curl_z
    matrices[:, 1, 0] = 0.5 * curl_z
    matrices[:_ROWS, 0, 0] = 1.0
    matrices[:_ROWS, 1, 1] = -1.0
    positions = physical_dx[:, None, None] * _OFFSETS[None, :, :]
    velocity = np.einsum("nij,nlj->nli", matrices, positions)
    return np.asarray(velocity, dtype=np.float32), physical_dx, group_ids


def _compute(
    velocity: np.ndarray,
    physical_dx: np.ndarray,
    group_ids: np.ndarray,
) -> SourceCenteredSeedTimeKinematics:
    return compute_source_centered_seed_time_kinematics(
        velocity,
        physical_dx,
        group_ids,
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )


def test_exact_group_curl_means_and_centered_four_coordinate_oracle():
    velocity, physical_dx, group_ids = _two_group_population()
    original_velocity = velocity.copy()
    original_dx = physical_dx.copy()
    original_groups = group_ids.copy()
    payload = _compute(velocity, physical_dx, group_ids)

    np.testing.assert_array_equal(
        payload.unique_group_ids,
        np.asarray([[0, 0, 0, 0], [1, 3, 1, 7]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        payload.group_row_count, np.asarray([_ROWS, _ROWS], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        payload.group_mean_curl_xyz,
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float64),
    )
    expected_first = np.asarray([2.0, np.sqrt(2.0), 0.0, 0.0], dtype=np.float32)
    expected_second = np.asarray([2.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(
        payload.source_centered_kinematic4[:_ROWS],
        np.broadcast_to(expected_first, (_ROWS, 4)),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        payload.source_centered_kinematic4[_ROWS:],
        np.broadcast_to(expected_second, (_ROWS, 4)),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(velocity, original_velocity)
    np.testing.assert_array_equal(physical_dx, original_dx)
    np.testing.assert_array_equal(group_ids, original_groups)
    assert FROZEN_SOURCE_GROUP_ID_ORDER == (
        "dataset_index",
        "source_ordinal",
        "scale_block_index",
        "dx_level_index",
    )
    assert FROZEN_SOURCE_CENTERED_KINEMATIC_FEATURE_ORDER == (
        "l2_norm_of_curl_deviation_from_source_block_dx_mean",
        "frobenius_norm_of_strain",
        "signed_divergence",
        "source_centered_signed_Q",
    )


def test_nonsymmetric_three_dimensional_affine_oracle_activates_all_curl_components():
    """Catch curl-axis swaps/sign errors that a z-only norm oracle cannot see."""

    gradient_a = np.asarray(
        [[1.0, 2.0, 3.0], [5.0, -2.0, 7.0], [11.0, 13.0, 4.0]],
        dtype=np.float64,
    )
    gradient_b = np.asarray(
        [[-1.0, 4.0, -2.0], [3.0, 3.0, 5.0], [-3.0, 2.0, 2.0]],
        dtype=np.float64,
    )
    matrices = np.empty((_ROWS, 3, 3), dtype=np.float64)
    matrices[: _ROWS // 2] = gradient_a
    matrices[_ROWS // 2 :] = gradient_b
    physical_dx = np.full(_ROWS, 0.5, dtype=np.float64)
    positions = physical_dx[:, None, None] * _OFFSETS[None, :, :]
    velocity = np.asarray(np.einsum("nij,nlj->nli", matrices, positions), dtype=np.float32)
    group_ids = np.zeros((_ROWS, 4), dtype=np.int32)

    payload = _compute(velocity, physical_dx, group_ids)
    curl_a = np.asarray([6.0, -8.0, 3.0], dtype=np.float64)
    curl_b = np.asarray([-3.0, 1.0, -1.0], dtype=np.float64)
    mean_curl = 0.5 * (curl_a + curl_b)
    np.testing.assert_array_equal(
        payload.group_mean_curl_xyz,
        mean_curl.reshape(1, 3),
    )

    curl_deviation_squared = float(np.square(curl_a - mean_curl).sum())
    strain_a = 0.5 * (gradient_a + gradient_a.T)
    strain_b = 0.5 * (gradient_b + gradient_b.T)
    expected_a = np.asarray(
        [
            np.sqrt(curl_deviation_squared),
            np.linalg.norm(strain_a),
            np.trace(gradient_a),
            0.25 * curl_deviation_squared - 0.5 * np.square(strain_a).sum(),
        ],
        dtype=np.float32,
    )
    expected_b = np.asarray(
        [
            np.sqrt(curl_deviation_squared),
            np.linalg.norm(strain_b),
            np.trace(gradient_b),
            0.25 * curl_deviation_squared - 0.5 * np.square(strain_b).sum(),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        payload.source_centered_kinematic4[: _ROWS // 2],
        np.broadcast_to(expected_a, (_ROWS // 2, 4)),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        payload.source_centered_kinematic4[_ROWS // 2 :],
        np.broadcast_to(expected_b, (_ROWS // 2, 4)),
        rtol=2e-6,
        atol=2e-6,
    )


def test_payload_arrays_have_frozen_dtypes_and_cannot_be_made_writeable():
    payload = _compute(*_two_group_population())
    expected = {
        "row_group_ids": (np.dtype(np.int32), (2 * _ROWS, 4)),
        "unique_group_ids": (np.dtype(np.int32), (2, 4)),
        "row_group_index": (np.dtype(np.int32), (2 * _ROWS,)),
        "group_row_count": (np.dtype(np.int32), (2,)),
        "group_mean_curl_xyz": (np.dtype(np.float64), (2, 3)),
        "source_centered_kinematic4": (np.dtype(np.float32), (2 * _ROWS, 4)),
    }
    for name, (dtype, shape) in expected.items():
        values = np.asarray(getattr(payload, name))
        assert values.dtype == dtype
        assert values.shape == shape
        assert not values.flags.owndata
        assert not values.flags.writeable
        try:
            values.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} could be made writeable")


def test_every_group_must_have_exactly_6400_complete_assigned_rows():
    velocity, physical_dx, group_ids = _two_group_population()
    wrong_counts = group_ids.copy()
    wrong_counts[_ROWS - 1] = wrong_counts[_ROWS]
    _expect_value_error(_compute, velocity, physical_dx, wrong_counts)

    one_group_only = group_ids[:_ROWS]
    one_payload = _compute(
        velocity[:_ROWS], physical_dx[:_ROWS], one_group_only
    )
    np.testing.assert_array_equal(
        one_payload.group_row_count, np.asarray([_ROWS], dtype=np.int32)
    )


def test_saved_group_mean_and_feature_tampering_fail_exact_recomputation():
    velocity, physical_dx, group_ids = _two_group_population()
    payload = _compute(velocity, physical_dx, group_ids)
    validate_source_centered_seed_time_kinematics(
        velocity,
        physical_dx,
        group_ids,
        payload,
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )

    tampered_mean = payload.group_mean_curl_xyz.copy()
    tampered_mean[1, 2] += 0.125
    wrong_mean_payload = replace(payload, group_mean_curl_xyz=tampered_mean)
    _expect_value_error(
        validate_source_centered_seed_time_kinematics,
        velocity,
        physical_dx,
        group_ids,
        wrong_mean_payload,
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )

    tampered_feature = payload.source_centered_kinematic4.copy()
    tampered_feature[0, 0] += 0.125
    wrong_feature_payload = replace(
        payload, source_centered_kinematic4=tampered_feature
    )
    _expect_value_error(
        validate_source_centered_seed_time_kinematics,
        velocity,
        physical_dx,
        group_ids,
        wrong_feature_payload,
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )


def test_input_shape_dtype_group_semantics_and_primitive_order_fail_closed():
    velocity, physical_dx, group_ids = _two_group_population()
    malformed_groups = (
        group_ids.astype(np.int64),
        group_ids[:, :3],
        group_ids[:, None, :],
    )
    for malformed in malformed_groups:
        _expect_value_error(_compute, velocity, physical_dx, malformed)

    negative_dataset = group_ids.copy()
    negative_dataset[:, 0] = -1
    _expect_value_error(_compute, velocity, physical_dx, negative_dataset)
    wrong_block = group_ids.copy()
    wrong_block[:, 2] = 2
    _expect_value_error(_compute, velocity, physical_dx, wrong_block)
    wrong_dx_level = group_ids.copy()
    wrong_dx_level[:, 3] = 10
    _expect_value_error(_compute, velocity, physical_dx, wrong_dx_level)

    _expect_value_error(_compute, velocity.astype(np.float64), physical_dx, group_ids)
    _expect_value_error(_compute, velocity, physical_dx.astype(np.float32), group_ids)
    _expect_value_error(_compute, velocity, np.float64(0.5), group_ids)
    _expect_value_error(
        compute_source_centered_seed_time_kinematics,
        velocity,
        physical_dx,
        group_ids,
        primitive_order=(
            "center",
            "x_minus",
            "x_plus",
            "y_plus",
            "y_minus",
            "z_plus",
            "z_minus",
        ),
    )

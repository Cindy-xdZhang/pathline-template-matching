from __future__ import annotations

import numpy as np

from pathline_template_matching.early_opposite_pair_kinematics import (
    FROZEN_KINEMATIC_FEATURE_ORDER,
    FROZEN_PRIMITIVE_ORDER,
    compute_seed_time_kinematic4,
    compute_seed_time_velocity_gradient,
)


_OFFSETS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_LEGACY_DX_GRID_SCALES = np.asarray(
    [
        0.250000000000,
        0.361111111111,
        0.472222222222,
        0.583333333333,
        0.694444444444,
        0.805555555556,
        0.916666666667,
        1.027777777778,
        1.138888888889,
        1.250000000000,
    ],
    dtype=np.float64,
)
_EXPANDED_DX_GRID_SCALES = np.asarray(
    [
        0.125000000000,
        0.388888888889,
        0.652777777778,
        0.916666666667,
        1.180555555556,
        1.444444444444,
        1.708333333333,
        1.972222222222,
        2.236111111111,
        2.500000000000,
    ],
    dtype=np.float64,
)
_FROZEN_DX_GRID_SCALES_BY_ID = np.concatenate(
    (
        np.repeat(_LEGACY_DX_GRID_SCALES, 100),
        np.repeat(_EXPANDED_DX_GRID_SCALES, 100),
    )
)


def _expect_value_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _sample_affine_velocity(
    matrix: np.ndarray,
    translation: np.ndarray,
    physical_dx: np.ndarray,
    *,
    centers: np.ndarray | None = None,
) -> np.ndarray:
    dx = np.asarray(physical_dx, dtype=np.float64)
    if centers is None:
        centers = np.zeros((len(dx), 3), dtype=np.float64)
    positions = centers[:, None, :] + dx[:, None, None] * _OFFSETS[None, :, :]
    velocity = positions @ np.asarray(matrix, dtype=np.float64).T
    velocity += np.asarray(translation, dtype=np.float64)[None, None, :]
    return np.asarray(velocity, dtype=np.float32)


def _kinematic4(velocity: np.ndarray, dx: object) -> np.ndarray:
    return compute_seed_time_kinematic4(
        velocity, dx, primitive_order=FROZEN_PRIMITIVE_ORDER
    )


def _gradient(velocity: np.ndarray, dx: object) -> np.ndarray:
    return compute_seed_time_velocity_gradient(
        velocity, dx, primitive_order=FROZEN_PRIMITIVE_ORDER
    )


def test_arbitrary_affine_flow_recovers_every_gradient_entry_at_all_frozen_dx():
    matrix = np.asarray(
        [
            [1.5, -0.25, 0.75],
            [0.5, -2.0, 0.125],
            [-1.0, 0.375, 2.25],
        ],
        dtype=np.float64,
    )
    physical_dx = 0.2 * _FROZEN_DX_GRID_SCALES_BY_ID
    assert physical_dx.shape == (2000,)
    velocity = _sample_affine_velocity(matrix, np.zeros(3), physical_dx)
    gradient = _gradient(velocity, physical_dx)

    np.testing.assert_allclose(
        gradient,
        np.broadcast_to(matrix, gradient.shape),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert gradient.dtype == np.float64
    assert not gradient.flags.writeable

    wrong_pair_order = velocity.copy()
    wrong_pair_order[:, [1, 2], :] = wrong_pair_order[:, [2, 1], :]
    wrong_gradient = _gradient(wrong_pair_order, physical_dx)
    assert not np.allclose(
        wrong_gradient,
        np.broadcast_to(matrix, wrong_gradient.shape),
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_rigid_translation_and_rotation_have_analytic_golden_invariants():
    translation_velocity = np.asarray(
        [[[3.0, -4.0, 7.0]] * 7], dtype=np.float32
    )
    np.testing.assert_array_equal(_kinematic4(translation_velocity, 0.25), 0.0)

    angular_velocity = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
    rotation_matrix = np.asarray(
        [
            [0.0, -angular_velocity[2], angular_velocity[1]],
            [angular_velocity[2], 0.0, -angular_velocity[0]],
            [-angular_velocity[1], angular_velocity[0], 0.0],
        ]
    )
    dx = np.asarray([0.125, 0.75, 2.5], dtype=np.float64)
    velocity = _sample_affine_velocity(
        rotation_matrix, np.asarray([4.0, -1.0, 2.0]), dx
    )
    expected = np.broadcast_to(
        np.asarray(
            [
                2.0 * np.linalg.norm(angular_velocity),
                0.0,
                0.0,
                np.dot(angular_velocity, angular_velocity),
            ],
            dtype=np.float32,
        ),
        (len(dx), 4),
    )
    np.testing.assert_allclose(_kinematic4(velocity, dx), expected, rtol=2e-6, atol=2e-6)


def test_pure_strain_simple_shear_and_signed_isotropic_expansion_are_exact():
    dx = np.asarray([0.5], dtype=np.float64)

    strain_rate = 2.0
    pure_strain = np.diag([strain_rate, -strain_rate, 0.0])
    pure_strain_feature = _kinematic4(
        _sample_affine_velocity(pure_strain, np.zeros(3), dx), dx
    )[0]
    np.testing.assert_allclose(
        pure_strain_feature,
        [0.0, np.sqrt(2.0) * strain_rate, 0.0, -(strain_rate**2)],
        rtol=1e-6,
        atol=1e-6,
    )

    shear_rate = 3.0
    simple_shear = np.zeros((3, 3), dtype=np.float64)
    simple_shear[0, 1] = shear_rate
    shear_feature = _kinematic4(
        _sample_affine_velocity(simple_shear, np.zeros(3), dx), dx
    )[0]
    np.testing.assert_allclose(
        shear_feature,
        [shear_rate, shear_rate / np.sqrt(2.0), 0.0, 0.0],
        rtol=1e-6,
        atol=1e-6,
    )

    for expansion_rate in (1.25, -0.75):
        expansion = expansion_rate * np.eye(3)
        expansion_feature = _kinematic4(
            _sample_affine_velocity(expansion, np.zeros(3), dx), dx
        )[0]
        np.testing.assert_allclose(
            expansion_feature,
            [
                0.0,
                np.sqrt(3.0) * abs(expansion_rate),
                3.0 * expansion_rate,
                -1.5 * expansion_rate**2,
            ],
            rtol=1e-6,
            atol=1e-6,
        )


def test_seed4_is_invariant_under_proper_rotation_and_spatial_translation():
    matrix = np.asarray(
        [[0.5, -1.25, 0.75], [2.0, -0.25, 0.5], [-0.75, 1.0, 1.5]],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transformed_matrix = rotation @ matrix @ rotation.T
    dx = np.asarray([0.125, 0.75, 2.5], dtype=np.float64)
    centers = np.asarray(
        [[0.25, -0.5, 0.75], [-0.75, 1.0, 0.5], [1.0, 0.25, -0.5]],
        dtype=np.float64,
    )
    original = _sample_affine_velocity(matrix, np.asarray([1.0, -2.0, 0.5]), dx, centers=centers)
    transformed = _sample_affine_velocity(
        transformed_matrix,
        np.asarray([-4.0, 2.5, 1.0]),
        dx,
        centers=centers @ rotation.T + np.asarray([3.0, -1.0, 0.25]),
    )
    np.testing.assert_allclose(
        _kinematic4(original, dx), _kinematic4(transformed, dx), rtol=5e-6, atol=5e-6
    )


def test_batch_chunk_and_row_permutation_are_identical_and_output_is_read_only():
    matrix = np.asarray(
        [[0.25, 1.0, -0.5], [-0.75, 0.5, 1.25], [1.5, -0.25, -1.0]],
        dtype=np.float64,
    )
    dx = np.asarray([0.125, 0.25, 0.75, 1.5, 2.5], dtype=np.float64)
    velocity = _sample_affine_velocity(matrix, np.asarray([1.0, 2.0, -3.0]), dx)
    original_velocity = velocity.copy()
    velocity.setflags(write=False)
    batch = _kinematic4(velocity, dx)
    chunks = np.concatenate(
        [_kinematic4(velocity[:2], dx[:2]), _kinematic4(velocity[2:], dx[2:])]
    )
    singles = np.concatenate(
        [_kinematic4(velocity[index : index + 1], dx[index]) for index in range(len(dx))]
    )
    permutation = np.asarray([3, 0, 4, 1, 2])
    inverse = np.argsort(permutation)
    permuted = _kinematic4(velocity[permutation], dx[permutation])[inverse]

    np.testing.assert_array_equal(batch, chunks)
    np.testing.assert_array_equal(batch, singles)
    np.testing.assert_array_equal(batch, permuted)
    np.testing.assert_array_equal(velocity, original_velocity)
    assert batch.dtype == np.float32
    assert not batch.flags.owndata
    assert not batch.flags.writeable
    assert not np.shares_memory(batch, velocity)
    assert FROZEN_KINEMATIC_FEATURE_ORDER == (
        "l2_norm_of_curl",
        "frobenius_norm_of_strain",
        "signed_divergence",
        "signed_Q",
    )
    try:
        batch.setflags(write=True)
    except ValueError:
        pass
    else:
        raise AssertionError("kinematic output could be made writeable")
    try:
        batch[0, 0] = 1.0
    except ValueError:
        pass
    else:
        raise AssertionError("kinematic output was writeable")


def test_shape_dtype_and_explicit_frozen_order_fail_closed():
    valid = np.zeros((2, 7, 3), dtype=np.float32)
    valid_dx = np.ones(2, dtype=np.float64)
    for malformed in (
        np.zeros((7, 3), dtype=np.float32),
        np.zeros((0, 7, 3), dtype=np.float32),
        np.zeros((2, 6, 3), dtype=np.float32),
        np.zeros((2, 7, 2), dtype=np.float32),
        np.zeros((2, 7, 3, 1), dtype=np.float32),
        np.zeros((2, 7, 3), dtype=np.float64),
    ):
        _expect_value_error(
            compute_seed_time_kinematic4,
            malformed,
            valid_dx,
            primitive_order=FROZEN_PRIMITIVE_ORDER,
        )

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
        compute_seed_time_kinematic4,
        valid,
        valid_dx,
        primitive_order=wrong_order,
    )
    _expect_value_error(
        compute_seed_time_kinematic4,
        valid,
        valid_dx,
        primitive_order="center,x_plus,x_minus,y_plus,y_minus,z_plus,z_minus",
    )


def test_physical_dx_and_all_velocity_samples_must_be_finite_and_valid():
    valid = np.zeros((2, 7, 3), dtype=np.float32)
    for malformed_dx in (
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.asarray([1, 1], dtype=np.int64),
        np.asarray([1.0], dtype=np.float64),
        np.asarray([[1.0], [1.0]], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([-1.0, 1.0], dtype=np.float64),
        np.asarray([np.nan, 1.0], dtype=np.float64),
        np.asarray([np.inf, 1.0], dtype=np.float64),
    ):
        _expect_value_error(_kinematic4, valid, malformed_dx)

    for line_index in range(7):
        malformed_velocity = valid.copy()
        malformed_velocity[0, line_index, 0] = np.nan
        _expect_value_error(_kinematic4, malformed_velocity, 1.0)
    infinite_velocity = valid.copy()
    infinite_velocity[1, 6, 2] = np.inf
    _expect_value_error(_kinematic4, infinite_velocity, 1.0)


def test_nonfinite_derivative_or_float32_serialization_fails_closed():
    huge = np.zeros((1, 7, 3), dtype=np.float32)
    huge[0, 1, 0] = np.finfo(np.float32).max
    huge[0, 2, 0] = -np.finfo(np.float32).max
    _expect_value_error(_gradient, huge, np.float64(1.0e-300))

    large_finite_gradient = np.zeros((1, 7, 3), dtype=np.float32)
    large_finite_gradient[0, 1, 0] = 1.0e30
    large_finite_gradient[0, 2, 0] = -1.0e30
    _expect_value_error(_kinematic4, large_finite_gradient, np.float64(1.0e-8))

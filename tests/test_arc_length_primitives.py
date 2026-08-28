import numpy as np

from pathline_template_matching.arc_length_primitives import (
    ArcLengthPrimitiveResult,
    build_arc_length_scale_table,
    integrate_arc_length_primitives_3d,
)
from pathline_template_matching.vector_field import UnsteadyVectorField3D


DX_SCALES = np.asarray(
    [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
)
DS_SCALES = np.asarray(
    [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
)
ARC_SCALES = np.asarray(
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
)


def _scale_id(dx_index: int, ds_index: int, arc_index: int) -> int:
    return (int(dx_index) * 10 + int(ds_index)) * 10 + int(arc_index)


def _scale_table():
    return build_arc_length_scale_table(DX_SCALES, DS_SCALES, ARC_SCALES)


def _constant_field(
    velocity: tuple[float, float, float], *, time_steps: int = 13
) -> UnsteadyVectorField3D:
    values = np.zeros((time_steps, 21, 21, 21, 3), dtype=np.float32)
    values[..., :] = velocity
    return UnsteadyVectorField3D(
        field=values,
        domain_min=np.zeros(3),
        domain_max=np.full(3, 2.0),
        grid_interval=np.full(3, 0.1),
        time_interval=0.1,
    )


def _assert_result_equal(first: ArcLengthPrimitiveResult, second: ArcLengthPrimitiveResult):
    for name in (
        "primitives",
        "valid_mask",
        "line_steps",
        "line_travel",
        "line_end_time",
        "line_reached_target",
        "scale_id",
        "dx_grid_scale",
        "ds_frame_scale",
        "arc_length_grid_scale",
        "physical_dx",
        "physical_dt",
        "target_arc_length",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.integration_max_time == second.integration_max_time


def _assert_result_slice_equal(
    reference: ArcLengthPrimitiveResult,
    candidate: ArcLengthPrimitiveResult,
    start: int,
    stop: int,
):
    for name in (
        "primitives",
        "valid_mask",
        "line_steps",
        "line_travel",
        "line_end_time",
        "line_reached_target",
        "scale_id",
        "dx_grid_scale",
        "ds_frame_scale",
        "arc_length_grid_scale",
        "physical_dx",
        "physical_dt",
        "target_arc_length",
    ):
        np.testing.assert_array_equal(
            getattr(reference, name)[start:stop], getattr(candidate, name)
        )
    assert reference.integration_max_time == candidate.integration_max_time


def _constant_velocity_oracle(
    seeds: np.ndarray,
    physical_dx: np.ndarray,
    target_arc_length: np.ndarray,
    velocity: tuple[float, float, float],
    *,
    seed_time: float = 0.0,
) -> np.ndarray:
    """Return the analytic complete [seed,line,sample,x/y/z/t] tensor."""

    effective_velocity = np.asarray(velocity, dtype=np.float32).astype(np.float64)
    speed = float(np.linalg.norm(effective_velocity))
    direction = effective_velocity / speed
    initial = np.broadcast_to(seeds[:, None, :], (len(seeds), 7, 3)).copy()
    initial[:, 1, 0] += physical_dx
    initial[:, 2, 0] -= physical_dx
    initial[:, 3, 1] += physical_dx
    initial[:, 4, 1] -= physical_dx
    initial[:, 5, 2] += physical_dx
    initial[:, 6, 2] -= physical_dx
    fractions = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    arc = target_arc_length[:, None] * fractions[None]
    xyz = initial[:, :, None, :] + arc[:, None, :, None] * direction
    times = np.broadcast_to(seed_time + arc[:, None, :] / speed, xyz.shape[:-1])
    return np.concatenate((xyz, times[..., None]), axis=-1)


def test_arc_length_scale_cartesian_product_is_unique_and_ordered():
    table = _scale_table()
    assert len(table) == 1000
    assert table.scale_id.dtype == np.int32
    np.testing.assert_array_equal(table.scale_id, np.arange(1000, dtype=np.int32))
    assert len(
        set(
            zip(
                table.dx_grid_scale,
                table.ds_frame_scale,
                table.arc_length_grid_scale,
                strict=True,
            )
        )
    ) == 1000
    np.testing.assert_array_equal(
        [
            table.dx_grid_scale[0],
            table.ds_frame_scale[0],
            table.arc_length_grid_scale[0],
        ],
        [DX_SCALES[0], DS_SCALES[0], ARC_SCALES[0]],
    )
    np.testing.assert_array_equal(
        [
            table.dx_grid_scale[9],
            table.ds_frame_scale[9],
            table.arc_length_grid_scale[9],
        ],
        [DX_SCALES[0], DS_SCALES[0], ARC_SCALES[9]],
    )
    np.testing.assert_array_equal(
        [
            table.dx_grid_scale[10],
            table.ds_frame_scale[10],
            table.arc_length_grid_scale[10],
        ],
        [DX_SCALES[0], DS_SCALES[1], ARC_SCALES[0]],
    )
    np.testing.assert_array_equal(
        [
            table.dx_grid_scale[100],
            table.ds_frame_scale[100],
            table.arc_length_grid_scale[100],
        ],
        [DX_SCALES[1], DS_SCALES[0], ARC_SCALES[0]],
    )
    assert not table.scale_id.flags.writeable
    try:
        build_arc_length_scale_table(
            np.r_[DX_SCALES[:-1], DX_SCALES[0]], DS_SCALES, ARC_SCALES
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate config scale values were accepted")


def test_constant_velocity_reaches_arc_target_and_resamples_uniformly():
    table = _scale_table()
    assignment = np.asarray([_scale_id(1, 2, 3)], dtype=np.int32)
    seed = np.asarray([[0.5, 1.0, 1.0]], dtype=np.float64)
    velocity = (0.1, -0.2, 0.4)
    result = integrate_arc_length_primitives_3d(
        _constant_field(velocity),
        seed,
        seed_time=0.0,
        scales=table,
        scale_assignment=assignment,
        chunk_size=1,
    )
    assert result.primitives.shape == (1, 7, 32, 4)
    np.testing.assert_array_equal(result.valid_mask, [True])
    np.testing.assert_array_equal(result.valid_seed_indices, [0])
    np.testing.assert_array_equal(result.line_steps, np.full((1, 7), 6))
    np.testing.assert_allclose(result.line_travel, 0.2, atol=1e-12)
    np.testing.assert_allclose(result.physical_dx, [0.05], atol=1e-15)
    np.testing.assert_allclose(result.physical_dt, [0.075], atol=1e-15)
    np.testing.assert_allclose(result.target_arc_length, [0.2], atol=1e-15)

    expected = _constant_velocity_oracle(
        seed, result.physical_dx, result.target_arc_length, velocity
    )
    np.testing.assert_allclose(
        result.primitives,
        expected,
        atol=2e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.line_end_time,
        expected[:, :, -1, 3],
        atol=2e-7,
        rtol=0.0,
    )
    segment_lengths = np.linalg.norm(
        np.diff(result.primitives[0, :, :, :3], axis=1), axis=-1
    )
    np.testing.assert_allclose(segment_lengths, 0.2 / 31.0, atol=2e-7)


def test_zero_velocity_and_clamped_final_time_step_are_invalid_and_audited():
    table = _scale_table()
    seed = np.asarray([[0.5, 1.0, 1.0]], dtype=np.float64)
    zero = integrate_arc_length_primitives_3d(
        _constant_field((0.0, 0.0, 0.0)),
        seed,
        seed_time=0.0,
        scales=table,
        scale_assignment=np.asarray([_scale_id(0, 3, 0)], dtype=np.int32),
    )
    assert zero.primitives.shape == (0, 7, 32, 4)
    np.testing.assert_array_equal(zero.valid_mask, [False])
    np.testing.assert_array_equal(zero.line_reached_target, False)
    np.testing.assert_array_equal(zero.line_steps, np.full((1, 7), 12))
    np.testing.assert_allclose(zero.line_travel, 0.0, atol=0.0)
    np.testing.assert_allclose(zero.line_end_time, 1.2, atol=1e-12)

    clamped = integrate_arc_length_primitives_3d(
        _constant_field((0.1, 0.0, 0.0)),
        seed,
        seed_time=0.0,
        scales=table,
        scale_assignment=np.asarray([_scale_id(0, 8, 9)], dtype=np.int32),
    )
    np.testing.assert_array_equal(clamped.valid_mask, [False])
    np.testing.assert_array_equal(clamped.line_steps, np.full((1, 7), 3))
    np.testing.assert_allclose(clamped.line_end_time, 1.2, atol=1e-12)
    np.testing.assert_allclose(clamped.line_travel, 0.12, atol=2e-7)


def test_per_seed_dx_ds_arc_metadata_external_batch_and_order_are_invariant():
    table = _scale_table()
    seeds = np.asarray(
        [
            [0.40, 0.80, 0.80],
            [0.45, 0.90, 0.90],
            [0.50, 1.00, 1.00],
            [0.55, 1.10, 1.10],
        ],
        dtype=np.float64,
    )
    assignment = np.asarray(
        [
            _scale_id(0, 0, 0),
            _scale_id(1, 2, 1),
            _scale_id(2, 4, 2),
            _scale_id(3, 6, 3),
        ],
        dtype=np.int32,
    )
    arguments = dict(
        vector_field=_constant_field((0.5, 0.0, 0.0)),
        seeds_xyz=seeds,
        seed_time=0.0,
        scales=table,
        scale_assignment=assignment,
    )
    reference = integrate_arc_length_primitives_3d(**arguments)
    for start, stop in ((0, 1), (1, 3), (3, 4)):
        batch = integrate_arc_length_primitives_3d(
            arguments["vector_field"],
            seeds[start:stop],
            seed_time=0.0,
            scales=table,
            scale_assignment=assignment[start:stop],
        )
        _assert_result_slice_equal(reference, batch, start, stop)
    np.testing.assert_array_equal(reference.valid_mask, True)
    np.testing.assert_array_equal(reference.scale_id, assignment)
    np.testing.assert_allclose(
        reference.physical_dx, DX_SCALES[[0, 1, 2, 3]] * 0.1, atol=1e-15
    )
    np.testing.assert_allclose(
        reference.physical_dt, DS_SCALES[[0, 2, 4, 6]] * 0.1, atol=1e-15
    )
    np.testing.assert_allclose(
        reference.target_arc_length,
        ARC_SCALES[[0, 1, 2, 3]] * 0.1,
        atol=1e-15,
    )

    permutation = np.asarray([2, 0, 3, 1])
    permuted = integrate_arc_length_primitives_3d(
        arguments["vector_field"],
        seeds[permutation],
        seed_time=0.0,
        scales=table,
        scale_assignment=assignment[permutation],
    )
    inverse = np.argsort(permutation)
    for name in (
        "primitives",
        "valid_mask",
        "line_steps",
        "line_travel",
        "line_end_time",
        "line_reached_target",
        "scale_id",
        "physical_dx",
        "physical_dt",
        "target_arc_length",
    ):
        np.testing.assert_array_equal(
            getattr(reference, name), getattr(permuted, name)[inverse]
        )


def test_initial_cross_boundary_and_short_time_window_fail_closed():
    table = _scale_table()
    boundary = integrate_arc_length_primitives_3d(
        _constant_field((0.5, 0.0, 0.0)),
        np.asarray([[0.01, 1.0, 1.0]]),
        seed_time=0.0,
        scales=table,
        scale_assignment=np.asarray([_scale_id(1, 2, 0)], dtype=np.int32),
    )
    np.testing.assert_array_equal(boundary.valid_mask, [False])
    assert boundary.primitives.shape == (0, 7, 32, 4)
    assert boundary.line_steps[0, 2] == 0
    assert boundary.line_travel[0, 2] == 0.0
    assert boundary.line_end_time[0, 2] == 0.0

    try:
        integrate_arc_length_primitives_3d(
            _constant_field((0.5, 0.0, 0.0), time_steps=12),
            np.asarray([[0.5, 1.0, 1.0]]),
            seed_time=0.0,
            scales=table,
            scale_assignment=np.asarray([0], dtype=np.int32),
        )
    except ValueError as error:
        assert "12 source-frame intervals" in str(error)
    else:
        raise AssertionError("a field missing the frozen 12-frame window was accepted")


def test_per_seed_dx_ds_arc_metadata_and_chunk_order_are_invariant():
    """Compatibility entry point for the standard-library test registry."""

    test_per_seed_dx_ds_arc_metadata_external_batch_and_order_are_invariant()


# The accurately named test above is what pytest should discover.  The legacy
# name remains callable only because tests/test_all.py is intentionally frozen
# outside this focused verification fix.
test_per_seed_dx_ds_arc_metadata_and_chunk_order_are_invariant.__test__ = False

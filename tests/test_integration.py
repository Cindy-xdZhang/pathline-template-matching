import numpy as np

from pathline_template_matching.primitives import (
    centered_xyz,
    integrate_cross_primitives_3d,
    integrate_multiscale_primitives_3d,
)
from pathline_template_matching.scales import PathlineScale3D
from pathline_template_matching.vector_field import UnsteadyVectorField3D


def _constant_field(velocity: tuple[float, float, float], time_steps: int = 17):
    values = np.zeros((time_steps, 9, 9, 9, 3), dtype=np.float32)
    values[..., :] = velocity
    return UnsteadyVectorField3D(
        field=values,
        domain_min=np.zeros(3),
        domain_max=np.ones(3),
        grid_interval=np.full(3, 0.125),
        time_interval=0.1,
    )


def _linear_x_field(time_steps: int = 17):
    x = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    values = np.zeros((time_steps, 9, 9, 9, 3), dtype=np.float32)
    values[..., 0] = x[None, None, None, :]
    return UnsteadyVectorField3D(
        field=values,
        domain_min=np.zeros(3),
        domain_max=np.ones(3),
        grid_interval=np.full(3, 0.125),
        time_interval=0.1,
    )


def test_rk4_constant_velocity_matches_analytic_cross_primitive():
    field = _constant_field((1.0, 0.0, 0.0), time_steps=11)
    seed = np.asarray([[0.3, 0.5, 0.5]])
    primitive, valid, lengths = integrate_cross_primitives_3d(
        field,
        seed,
        seed_time=0.0,
        dt=0.1,
        integration_steps=5,
        sampled_steps=6,
        offset=0.025,
    )
    assert primitive.shape == (1, 7, 6, 4)
    np.testing.assert_array_equal(valid, [True])
    np.testing.assert_array_equal(lengths, np.full((1, 7), 6))
    initial = primitive[0, :, 0, :3]
    expected_time = np.linspace(0.0, 0.5, 6)
    expected_time_grid = np.broadcast_to(expected_time, (7, 6))
    np.testing.assert_allclose(primitive[0, :, :, 3], expected_time_grid, atol=1e-6)
    np.testing.assert_allclose(
        primitive[0, :, :, 0], initial[:, 0, None] + expected_time[None], atol=2e-6
    )
    np.testing.assert_allclose(
        primitive[0, :, :, 1], np.broadcast_to(initial[:, 1, None], (7, 6)), atol=1e-7
    )
    np.testing.assert_allclose(
        primitive[0, :, :, 2], np.broadcast_to(initial[:, 2, None], (7, 6)), atol=1e-7
    )


def test_multiscale_zero_flow_returns_frozen_7x32_contract_and_physical_scale():
    field = _constant_field((0.0, 0.0, 0.0))
    scale = PathlineScale3D("test", 0.5, 0.5, 31)
    result = integrate_multiscale_primitives_3d(
        field,
        np.asarray([[0.5, 0.5, 0.5]]),
        seed_time=0.0,
        scales=[scale],
        scale_assignment=np.asarray([0]),
        sampled_steps=32,
    )
    assert result["primitives"].shape == (1, 7, 32, 4)
    np.testing.assert_array_equal(result["valid_mask"], [True])
    np.testing.assert_allclose(result["physical_dt"], [0.05])
    np.testing.assert_allclose(result["integration_horizon"], [1.55])
    centered = centered_xyz(result["primitives"])
    assert centered.shape == (1, 7, 32, 3)
    np.testing.assert_allclose(centered[0, 0], 0.0, atol=1e-7)
    np.testing.assert_allclose(centered[0, 1, :, 0], 0.0625, atol=1e-7)


def test_rk4_linear_velocity_matches_exponential_solution_not_euler():
    primitive, valid, _ = integrate_cross_primitives_3d(
        _linear_x_field(),
        np.asarray([[0.2, 0.5, 0.5]]),
        seed_time=0.0,
        dt=0.1,
        integration_steps=5,
        sampled_steps=6,
        offset=0.025,
        method="RK4",
    )
    np.testing.assert_array_equal(valid, [True])
    times = primitive[0, :, :, 3]
    initial_x = primitive[0, :, :1, 0]
    expected_x = initial_x * np.exp(times)
    np.testing.assert_allclose(primitive[0, :, :, 0], expected_x, atol=1e-6)


def test_n48_n64_rounded_sampling_and_chunk_order_are_frozen():
    field = _constant_field((0.0, 0.0, 0.0), time_steps=65)
    seeds = np.asarray([[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]])
    scales = [
        PathlineScale3D("n48", 0.5, 0.5, 48),
        PathlineScale3D("n64", 0.5, 0.5, 64),
    ]
    arguments = dict(
        vector_field=field,
        seeds_xyz=seeds,
        seed_time=0.0,
        scales=scales,
        scale_assignment=np.asarray([0, 1]),
        sampled_steps=32,
    )
    chunked = integrate_multiscale_primitives_3d(**arguments, chunk_size=1)
    batched = integrate_multiscale_primitives_3d(**arguments, chunk_size=2048)
    for key in chunked:
        np.testing.assert_array_equal(chunked[key], batched[key])
    np.testing.assert_array_equal(chunked["scale_id"], [0, 1])
    for index, steps in enumerate((48, 64)):
        sample_indices = np.rint(np.linspace(0, steps, 32)).astype(np.int64)
        expected_times = sample_indices * 0.05
        np.testing.assert_allclose(
            chunked["primitives"][index, 0, :, 3], expected_times, atol=2e-6
        )
        np.testing.assert_allclose(
            chunked["primitives"][index, 0, :, :3],
            np.broadcast_to(seeds[index], (32, 3)),
            atol=1e-7,
        )

import numpy as np

from pathline_template_matching.channel_flow import (
    compose_steady_to_unsteady,
    integrate_killing_frame,
    smooth_channel_observer,
)


def test_killing_frame_translation_and_identity_pushforward():
    parameters = np.zeros((5, 6), dtype=np.float64)
    parameters[:, 0] = 2.0
    rotation, displacement = integrate_killing_frame(parameters, 0.25)
    np.testing.assert_allclose(rotation, np.broadcast_to(np.eye(3), rotation.shape), atol=0)
    np.testing.assert_allclose(displacement[:, 0], np.linspace(0.0, 2.0, 5), atol=1e-15)
    points = np.asarray([[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]])

    def constant_steady(query):
        return np.broadcast_to(np.asarray([1.0, -2.0, 3.0]), query.shape)

    zero = np.zeros((2, 6), dtype=np.float64)
    identity = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    output = compose_steady_to_unsteady(
        points,
        constant_steady,
        zero,
        identity,
        np.zeros((2, 3)),
        bounds_min=np.full(3, -1.0),
        bounds_max=np.full(3, 1.0),
    )
    np.testing.assert_allclose(output, [constant_steady(points)] * 2, atol=0)


def test_channel_observer_is_finite_deterministic_and_domain_scaled():
    times = np.linspace(0.0, 1.0, 159)
    lower = np.asarray([-2.0, -1.0, 0.0])
    upper = np.asarray([3.0, 4.0, 2.0])
    first = smooth_channel_observer(times, lower, upper)
    second = smooth_channel_observer(times, lower, upper)
    assert first.shape == (159, 6)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert np.linalg.norm(first[:, :3], axis=1).max() > 0
    assert np.linalg.norm(first[:, 3:], axis=1).max() > 0


def test_killing_rotation_orientation_matches_rodrigues_golden_value():
    parameters = np.zeros((3, 6), dtype=np.float64)
    parameters[:, 5] = 1.0
    rotation, displacement = integrate_killing_frame(parameters, 0.1)
    cosine = np.cos(0.1)
    sine = np.sin(0.1)
    expected_first_step = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(rotation[1], expected_first_step, rtol=0, atol=1e-15)
    np.testing.assert_allclose(rotation[2], expected_first_step @ expected_first_step, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(displacement, np.zeros((3, 3)))

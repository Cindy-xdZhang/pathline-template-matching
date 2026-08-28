import numpy as np

from pathline_template_matching.ivd import (
    compute_ivd_3d,
    ivd_p95_reference_at_seeds,
    ivd_percentile_mask,
    sample_regular_volume_3d,
)


def test_solid_body_rotation_has_zero_ivd():
    axis = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    velocity = np.stack((-y, x, np.zeros_like(z)), axis=-1)
    ivd = compute_ivd_3d(velocity, (axis[1] - axis[0],) * 3)
    np.testing.assert_allclose(ivd, 0.0, atol=2e-6)


def test_percentile_mask_uses_greater_than_or_equal():
    values = np.arange(100, dtype=np.float32).reshape(4, 5, 5)
    mask, threshold = ivd_percentile_mask(values, 95.0)
    assert threshold == np.percentile(values, 95.0)
    np.testing.assert_array_equal(mask, values >= threshold)


def test_percentile_mask_rejects_nonfinite_volume():
    try:
        ivd_percentile_mask(np.asarray([[[0.0, np.nan]]], dtype=np.float32), 95.0)
    except ValueError as error:
        assert "NaN or Inf" in str(error)
    else:
        raise AssertionError("non-finite IVD was accepted")


def test_regular_volume_sampling_is_trilinear_and_fails_outside():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(2.0, 5.0, 4)
    z = np.linspace(0.0, 4.0, 3)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    volume = (2.0 * xx - 3.0 * yy + 0.5 * zz).astype(np.float32)
    points = np.asarray([[0.25, 2.5, 1.0], [-0.75, 4.5, 3.0]])
    sampled = sample_regular_volume_3d(volume, (x, y, z), points)
    np.testing.assert_allclose(
        sampled,
        2.0 * points[:, 0] - 3.0 * points[:, 1] + 0.5 * points[:, 2],
        atol=2e-6,
    )
    try:
        sample_regular_volume_3d(volume, (x, y, z), np.asarray([[1.1, 3.0, 2.0]]))
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("out-of-domain sampling point was accepted")


def test_ivd_seed_reference_uses_whole_loaded_volume_p95():
    x = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 11, dtype=np.float32)
    z = np.linspace(-0.5, 0.5, 7, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    velocity = np.stack((np.zeros_like(xx), xx * xx, np.zeros_like(xx)), axis=-1)
    seeds = np.asarray([[x[0], y[5], z[3]], [x[-1], y[5], z[3]], [0, 0, 0]])
    labels, values, threshold, mask = ivd_p95_reference_at_seeds(
        velocity,
        (x[1] - x[0], y[1] - y[0], z[1] - z[0]),
        (x, y, z),
        seeds,
    )
    ivd = compute_ivd_3d(
        velocity, (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    )
    assert threshold == np.percentile(ivd, 95.0)
    np.testing.assert_array_equal(mask, ivd >= threshold)
    np.testing.assert_array_equal(labels, values >= threshold)

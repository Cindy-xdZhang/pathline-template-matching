import numpy as np

from pathline_template_matching.ivd import compute_ivd_3d, ivd_percentile_mask


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

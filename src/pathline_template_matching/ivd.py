"""Whole-loaded-volume Instantaneous Vorticity Deviation labels."""

from __future__ import annotations

import numpy as np


def vorticity_components_3d(
    velocity_zyx3: np.ndarray,
    spacing_xyz: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed curl components for a regular ``[Z,Y,X,3]`` grid."""

    data = np.asarray(velocity_zyx3, dtype=np.float32)
    if data.ndim != 4 or data.shape[-1] != 3 or min(data.shape[:3]) < 2:
        raise ValueError(f"velocity must be [Z,Y,X,3] with each axis >=2, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("velocity contains NaN or Inf")
    dx, dy, dz = (float(value) for value in spacing_xyz)
    if not np.isfinite([dx, dy, dz]).all() or min(dx, dy, dz) <= 0:
        raise ValueError(f"spacing must contain three positive finite values, got {(dx, dy, dz)}")
    u, v, w = data[..., 0], data[..., 1], data[..., 2]
    dw_dy = np.gradient(w, dy, axis=1)
    dv_dz = np.gradient(v, dz, axis=0)
    du_dz = np.gradient(u, dz, axis=0)
    dw_dx = np.gradient(w, dx, axis=2)
    dv_dx = np.gradient(v, dx, axis=2)
    du_dy = np.gradient(u, dy, axis=1)
    return dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy


def compute_ivd_3d(
    velocity_zyx3: np.ndarray,
    spacing_xyz: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Compute ``||curl(v)-mean_volume(curl(v))||`` on the loaded volume."""

    wx, wy, wz = vorticity_components_3d(velocity_zyx3, spacing_xyz)
    wx = wx - wx.mean()
    wy = wy - wy.mean()
    wz = wz - wz.mean()
    ivd = np.sqrt(wx * wx + wy * wy + wz * wz).astype(np.float32)
    if not np.isfinite(ivd).all():
        raise ValueError("IVD contains NaN or Inf")
    return ivd


def ivd_percentile_mask(ivd_zyx: np.ndarray, percentile: float = 95.0) -> tuple[np.ndarray, float]:
    """Return the frozen ``IVD >= percentile`` mask and threshold."""

    ivd = np.asarray(ivd_zyx, dtype=np.float32)
    if not np.isfinite(ivd).all():
        raise ValueError("IVD volume contains NaN or Inf")
    percentile = float(percentile)
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be strictly between 0 and 100")
    threshold = float(np.percentile(ivd, percentile))
    return np.asarray(ivd >= threshold, dtype=bool), threshold


def sample_regular_volume_3d(
    volume_zyx: np.ndarray,
    coordinates_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    points_xyz: np.ndarray,
) -> np.ndarray:
    """Trilinearly sample a finite uniform ``[Z,Y,X]`` volume at XYZ points."""

    volume = np.asarray(volume_zyx, dtype=np.float64)
    points = np.asarray(points_xyz, dtype=np.float64)
    if volume.ndim != 3 or min(volume.shape) < 2 or not np.isfinite(volume).all():
        raise ValueError("volume_zyx must be a finite [Z,Y,X] array with axes >=2")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_xyz must be a finite [N,3] array")
    axes = tuple(np.asarray(axis, dtype=np.float64) for axis in coordinates_xyz)
    if len(axes) != 3:
        raise ValueError("coordinates_xyz must contain x, y, z arrays")
    expected_lengths = (volume.shape[2], volume.shape[1], volume.shape[0])
    for name, axis, expected in zip("xyz", axes, expected_lengths, strict=True):
        if (
            axis.shape != (expected,)
            or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)
            or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-4, atol=1e-8)
        ):
            raise ValueError(f"{name} coordinate is incompatible with the regular volume")
    x, y, z = axes
    fractional = np.column_stack(
        (
            (points[:, 0] - x[0]) / (x[1] - x[0]),
            (points[:, 1] - y[0]) / (y[1] - y[0]),
            (points[:, 2] - z[0]) / (z[1] - z[0]),
        )
    )
    upper = np.asarray([len(x) - 1, len(y) - 1, len(z) - 1], dtype=np.float64)
    tolerance = 1e-5
    if np.any(fractional < -tolerance) or np.any(fractional > upper + tolerance):
        raise ValueError("sampling point lies outside the volume domain")
    fractional = np.clip(fractional, 0.0, upper)
    lower_index = np.floor(fractional).astype(np.int64)
    upper_index = np.minimum(lower_index + 1, upper.astype(np.int64))
    weight = fractional - lower_index
    x0, y0, z0 = lower_index.T
    x1, y1, z1 = upper_index.T
    wx, wy, wz = weight.T
    sampled = (
        volume[z0, y0, x0] * (1 - wx) * (1 - wy) * (1 - wz)
        + volume[z0, y0, x1] * wx * (1 - wy) * (1 - wz)
        + volume[z0, y1, x0] * (1 - wx) * wy * (1 - wz)
        + volume[z0, y1, x1] * wx * wy * (1 - wz)
        + volume[z1, y0, x0] * (1 - wx) * (1 - wy) * wz
        + volume[z1, y0, x1] * wx * (1 - wy) * wz
        + volume[z1, y1, x0] * (1 - wx) * wy * wz
        + volume[z1, y1, x1] * wx * wy * wz
    ).astype(np.float32)
    if not np.isfinite(sampled).all():
        raise ValueError("volume sampling produced NaN or Inf")
    return sampled


def ivd_p95_reference_at_seeds(
    velocity_zyx3: np.ndarray,
    spacing_xyz: tuple[float, float, float] | np.ndarray,
    coordinates_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    seeds_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute whole-loaded-volume IVD-p95 and attach labels to center seeds."""

    ivd = compute_ivd_3d(velocity_zyx3, spacing_xyz)
    mask, threshold = ivd_percentile_mask(ivd, 95.0)
    values = sample_regular_volume_3d(ivd, coordinates_xyz, seeds_xyz)
    labels = np.asarray(values >= threshold, dtype=bool)
    return labels, values, threshold, mask

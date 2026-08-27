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

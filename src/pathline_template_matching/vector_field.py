"""Minimal regular-grid 3D vector-field container for pathline integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .netcdf_io import FlowWindow3D


@dataclass(frozen=True)
class UnsteadyVectorField3D:
    """Canonical ``[T,Z,Y,X,3]`` velocity with uniform space and time axes."""

    field: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray
    grid_interval: np.ndarray
    time_interval: float
    tmin: float = 0.0

    def __post_init__(self) -> None:
        field = np.ascontiguousarray(self.field, dtype=np.float32)
        domain_min = np.asarray(self.domain_min, dtype=np.float32)
        domain_max = np.asarray(self.domain_max, dtype=np.float32)
        grid_interval = np.asarray(self.grid_interval, dtype=np.float32)
        if field.ndim != 5 or field.shape[-1] != 3 or min(field.shape[1:4]) < 2:
            raise ValueError(f"field must be [T,Z,Y,X,3], got {field.shape}")
        if not np.isfinite(field).all():
            raise ValueError("field contains NaN or Inf")
        if domain_min.shape != (3,) or domain_max.shape != (3,):
            raise ValueError("domain bounds must each contain x,y,z")
        if grid_interval.shape != (3,) or np.any(grid_interval <= 0):
            raise ValueError("grid intervals must be three positive values")
        if field.shape[0] > 1 and (not np.isfinite(self.time_interval) or self.time_interval <= 0):
            raise ValueError("a time-varying field requires a positive time interval")
        expected_max = domain_min + grid_interval * np.asarray(
            [field.shape[3] - 1, field.shape[2] - 1, field.shape[1] - 1],
            dtype=np.float32,
        )
        if not np.allclose(domain_max, expected_max, rtol=1e-4, atol=1e-6):
            raise ValueError(
                f"domain/grid/shape disagree: max={domain_max}, expected={expected_max}"
            )
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "domain_min", domain_min)
        object.__setattr__(self, "domain_max", domain_max)
        object.__setattr__(self, "grid_interval", grid_interval)
        object.__setattr__(self, "time_interval", float(self.time_interval))
        object.__setattr__(self, "tmin", float(self.tmin))

    @classmethod
    def from_window(cls, window: FlowWindow3D) -> "UnsteadyVectorField3D":
        x, y, z = window.coordinates_xyz
        time_interval = float(window.time[1] - window.time[0]) if len(window.time) > 1 else 0.0
        return cls(
            field=window.velocity,
            domain_min=np.asarray([x[0], y[0], z[0]], dtype=np.float32),
            domain_max=np.asarray([x[-1], y[-1], z[-1]], dtype=np.float32),
            grid_interval=window.spacing_xyz.astype(np.float32),
            time_interval=time_interval,
            tmin=0.0,
        )

    @property
    def time_steps(self) -> int:
        return int(self.field.shape[0])

    @property
    def zdim(self) -> int:
        return int(self.field.shape[1])

    @property
    def ydim(self) -> int:
        return int(self.field.shape[2])

    @property
    def xdim(self) -> int:
        return int(self.field.shape[3])

    @property
    def tmax(self) -> float:
        return self.tmin + self.time_interval * (self.time_steps - 1)

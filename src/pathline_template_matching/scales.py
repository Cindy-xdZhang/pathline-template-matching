"""Validated scale tuples copied from the FMT Task5 protocol.

The scale parser and balanced assignment preserve the behavior of
``FMT_Utils/MultiscalePathline_3D.py``. The dependency-minimized integrator is
implemented separately in :mod:`pathline_template_matching.primitives`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathlineScale3D:
    """One integration scale in source-grid and source-frame units."""

    name: str
    offset_grid_scale: float
    dt_scale: float
    integration_steps: int

    @property
    def horizon_in_source_frames(self) -> float:
        return float(self.dt_scale) * int(self.integration_steps)

    @property
    def tuple(self) -> tuple[float, float, int]:
        return (
            float(self.offset_grid_scale),
            float(self.dt_scale),
            int(self.integration_steps),
        )


def parse_scale_table(rows: list[dict[str, object]], sampled_steps: int) -> list[PathlineScale3D]:
    """Parse and fail closed on an invalid discrete scale table."""

    scales = [
        PathlineScale3D(
            name=str(row["name"]),
            offset_grid_scale=float(row["offset_grid_scale"]),
            dt_scale=float(row["dt_scale"]),
            integration_steps=int(row["integration_steps"]),
        )
        for row in rows
    ]
    if not scales:
        raise ValueError("a multiscale table must contain at least one scale")
    names = [scale.name for scale in scales]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate multiscale names: {names}")
    sampled_steps = int(sampled_steps)
    if sampled_steps < 2:
        raise ValueError("sampled_steps must be at least 2")
    for scale in scales:
        if not np.isfinite(scale.offset_grid_scale) or scale.offset_grid_scale <= 0:
            raise ValueError(f"{scale.name}: offset_grid_scale must be positive")
        if not np.isfinite(scale.dt_scale) or scale.dt_scale <= 0:
            raise ValueError(f"{scale.name}: dt_scale must be positive")
        if scale.integration_steps < sampled_steps - 1:
            raise ValueError(
                f"{scale.name}: integration_steps={scale.integration_steps} "
                f"cannot provide L={sampled_steps} unique samples"
            )
    tuples = [scale.tuple for scale in scales]
    if len(set(tuples)) != len(tuples):
        raise ValueError(f"duplicate numeric scale tuples: {tuples}")
    return scales


def balanced_scale_assignment(sample_count: int, scale_count: int, seed: int) -> np.ndarray:
    """Assign scales independently of seed order with count imbalance at most one."""

    sample_count, scale_count = int(sample_count), int(scale_count)
    if sample_count < 0 or scale_count <= 0:
        raise ValueError("sample_count must be non-negative and scale_count positive")
    if sample_count == 0:
        return np.empty(0, dtype=np.int16)
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(sample_count)
    assignment = np.empty(sample_count, dtype=np.int16)
    assignment[permutation] = np.arange(sample_count, dtype=np.int64) % scale_count
    counts = np.bincount(assignment, minlength=scale_count)
    if int(counts.max() - counts.min()) > 1:
        raise AssertionError("balanced scale assignment failed")
    return assignment

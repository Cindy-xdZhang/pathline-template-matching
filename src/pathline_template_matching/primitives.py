"""3D seed grids and fixed-shape variable-scale pathline primitives."""

from __future__ import annotations

import numpy as np

from .integration import compute_pathlines_3d_batch
from .scales import PathlineScale3D
from .vector_field import UnsteadyVectorField3D


def generate_seeding_grid_3d(
    vector_field: UnsteadyVectorField3D,
    grid_shape: tuple[int, int, int] | list[int],
    boundary_fraction: float,
    maximum_offset: float,
    *,
    grid_phase: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create a domain-inset Cartesian seed grid safe for a 7-line cross."""

    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("grid_shape must contain three integers >=2")
    span = vector_field.domain_max.astype(np.float64) - vector_field.domain_min
    margin = np.maximum(float(boundary_fraction) * span, float(maximum_offset) * 1.01)
    if np.any(2.0 * margin >= span):
        raise ValueError("boundary margin leaves no seeding volume")
    low = vector_field.domain_min + margin
    high = vector_field.domain_max - margin
    if grid_phase is None:
        axes = tuple(np.linspace(low[index], high[index], shape[index]) for index in range(3))
    else:
        phase = np.asarray(grid_phase, dtype=np.float64)
        if phase.shape != (3,) or not np.isfinite(phase).all() or np.any(np.abs(phase) >= 0.5):
            raise ValueError("grid_phase must contain three finite values with abs(value)<0.5")
        values = []
        for index, count in enumerate(shape):
            width = (high[index] - low[index]) / count
            values.append(low[index] + (np.arange(count) + 0.5 + phase[index]) * width)
        axes = tuple(values)
    x, y, z = axes
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    seeds = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=-1)
    return seeds, axes


def integrate_cross_primitives_3d(
    vector_field: UnsteadyVectorField3D,
    seeds_xyz: np.ndarray,
    seed_time: float,
    dt: float,
    integration_steps: int,
    sampled_steps: int,
    offset: float,
    *,
    method: str = "RK4",
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate center/x±/y±/z± and retain only seven complete valid lines.

    The frozen 1.1 resampler selects rounded integration indices. The returned
    primitive includes time as the fourth channel; callers explicitly drop it
    before FMT encoding.
    """

    integration_steps, sampled_steps = int(integration_steps), int(sampled_steps)
    if integration_steps < 1 or not 2 <= sampled_steps <= integration_steps + 1:
        raise ValueError("require steps>=1 and 2<=sampled_steps<=steps+1")
    if not np.isfinite(dt) or float(dt) <= 0 or not np.isfinite(offset) or float(offset) <= 0:
        raise ValueError("dt and offset must be positive finite values")
    seeds = np.asarray(seeds_xyz, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or not np.isfinite(seeds).all():
        raise ValueError(f"seeds_xyz must be finite [N,3], got {seeds.shape}")
    target_time = float(seed_time) + float(dt) * integration_steps
    if not vector_field.tmin <= float(seed_time) <= vector_field.tmax:
        raise ValueError("seed_time is outside the field time range")
    if target_time > vector_field.tmax + 1e-12:
        raise ValueError(
            f"integration target {target_time:g} exceeds field tmax={vector_field.tmax:g}"
        )
    offsets = np.asarray(
        [
            [0, 0, 0], [offset, 0, 0], [-offset, 0, 0],
            [0, offset, 0], [0, -offset, 0], [0, 0, offset], [0, 0, -offset],
        ],
        dtype=np.float64,
    )
    expanded = (seeds[:, None, :] + offsets[None]).reshape(-1, 3)
    expanded = np.column_stack((expanded, np.full(len(expanded), float(seed_time))))
    desired_length = integration_steps + 1
    sample_indices = np.rint(
        np.linspace(0, integration_steps, sampled_steps)
    ).astype(np.int64)
    if len(np.unique(sample_indices)) != sampled_steps:
        raise AssertionError("rounded-index resampling produced duplicate samples")
    chunks, length_chunks = [], []
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(expanded), chunk_size):
        positions, lengths = compute_pathlines_3d_batch(
            vector_field,
            expanded[start : start + chunk_size],
            min_time=float(seed_time),
            max_time=target_time,
            step_size=float(dt),
            max_iterations=integration_steps,
            method=method,
        )
        chunks.append(positions[:, :desired_length])
        length_chunks.append(lengths)
    if not chunks:
        return (
            np.empty((0, 7, sampled_steps, 4), dtype=np.float32),
            np.empty(0, dtype=bool),
            np.empty((0, 7), dtype=np.int32),
        )
    positions = np.concatenate(chunks).reshape(-1, 7, desired_length, 4)
    lengths = np.concatenate(length_chunks).reshape(-1, 7)
    xyz = positions[..., :3]
    spatially_valid = (
        (xyz >= vector_field.domain_min.reshape(1, 1, 1, 3))
        & (xyz <= vector_field.domain_max.reshape(1, 1, 1, 3))
    ).all(axis=(1, 2, 3))
    valid = (lengths == desired_length).all(axis=1) & spatially_valid
    return positions[valid][:, :, sample_indices], valid, lengths


def integrate_multiscale_primitives_3d(
    vector_field: UnsteadyVectorField3D,
    seeds_xyz: np.ndarray,
    seed_time: float,
    scales: list[PathlineScale3D],
    scale_assignment: np.ndarray,
    sampled_steps: int,
    *,
    offset_mode: str = "min",
    method: str = "RK4",
    chunk_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Integrate one scale per seed and restore original seed ordering."""

    seeds = np.asarray(seeds_xyz, dtype=np.float64)
    assignment = np.asarray(scale_assignment, dtype=np.int64)
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError(f"seeds_xyz must be [N,3], got {seeds.shape}")
    if assignment.shape != (len(seeds),):
        raise ValueError("scale_assignment must contain one id per seed")
    if not scales or (assignment.size and (assignment.min() < 0 or assignment.max() >= len(scales))):
        raise ValueError("scale_assignment contains an unknown scale id")
    spacing = vector_field.grid_interval.astype(np.float64)
    offset_bases = {
        "min": float(spacing.min()),
        "geometric_mean": float(np.prod(spacing) ** (1.0 / 3.0)),
        "max": float(spacing.max()),
    }
    if offset_mode not in offset_bases:
        raise ValueError("offset_mode must be min, geometric_mean, or max")
    sampled_steps = int(sampled_steps)
    primitives_all = np.zeros((len(seeds), 7, sampled_steps, 4), dtype=np.float32)
    lengths_all = np.zeros((len(seeds), 7), dtype=np.int32)
    valid_all = np.zeros(len(seeds), dtype=bool)
    offsets = np.zeros(len(seeds), dtype=np.float32)
    physical_dts = np.zeros(len(seeds), dtype=np.float32)
    horizons = np.zeros(len(seeds), dtype=np.float32)
    for scale_id, scale in enumerate(scales):
        selected = np.flatnonzero(assignment == scale_id)
        if not len(selected):
            continue
        offset = offset_bases[offset_mode] * scale.offset_grid_scale
        dt = vector_field.time_interval * scale.dt_scale
        primitives, valid_local, lengths = integrate_cross_primitives_3d(
            vector_field,
            seeds[selected],
            seed_time,
            dt,
            scale.integration_steps,
            sampled_steps,
            offset,
            method=method,
            chunk_size=chunk_size,
        )
        lengths_all[selected] = lengths
        valid_indices = selected[valid_local]
        primitives_all[valid_indices] = primitives
        valid_all[valid_indices] = True
        offsets[selected] = offset
        physical_dts[selected] = dt
        horizons[selected] = dt * scale.integration_steps
    valid_assignment = assignment[valid_all]
    return {
        "primitives": primitives_all[valid_all],
        "valid_mask": valid_all,
        "line_lengths": lengths_all,
        "scale_id": valid_assignment.astype(np.int16),
        "offset_grid_scale": np.asarray(
            [scales[index].offset_grid_scale for index in valid_assignment], dtype=np.float32
        ),
        "dt_scale": np.asarray(
            [scales[index].dt_scale for index in valid_assignment], dtype=np.float32
        ),
        "integration_steps": np.asarray(
            [scales[index].integration_steps for index in valid_assignment], dtype=np.int16
        ),
        "primitive_offset": offsets[valid_all],
        "physical_dt": physical_dts[valid_all],
        "integration_horizon": horizons[valid_all],
    }


def centered_xyz(primitives_xyzt: np.ndarray) -> np.ndarray:
    """Drop time and translate every primitive's center seed to the origin."""

    values = np.asarray(primitives_xyzt)
    if values.ndim != 4 or values.shape[1] != 7 or values.shape[-1] < 3:
        raise ValueError(f"primitives must be [N,7,L,C>=3], got {values.shape}")
    xyz = np.asarray(values[..., :3], dtype=np.float32)
    return xyz - xyz[:, :1, :1, :]

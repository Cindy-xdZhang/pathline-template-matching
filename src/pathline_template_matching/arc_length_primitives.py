"""Arc-length-resampled 3D pathline-cross primitives.

This module is deliberately separate from the frozen rounded-index baseline in
``primitives.py``.  Every seed is assigned one scale from a Cartesian table.
The seven forward Runge--Kutta paths are stopped at a physical arc length and
resampled uniformly in accumulated Euclidean arc length.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

import numpy as np
from numba import njit, prange

from .vector_field import UnsteadyVectorField3D


LINE_COUNT = 7
SAMPLED_POINTS = 32
SCALE_AXIS_COUNT = 10
MAX_SOURCE_FRAME_INTERVALS = 12.0


@dataclass(frozen=True)
class ArcLengthScaleTable:
    """Validated per-scale arrays in ``dx -> ds -> arc`` Cartesian order."""

    scale_id: np.ndarray
    dx_grid_scale: np.ndarray
    ds_frame_scale: np.ndarray
    arc_length_grid_scale: np.ndarray

    def __post_init__(self) -> None:
        raw_ids = np.asarray(self.scale_id)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
            raise ValueError("scale_id must be a one-dimensional integer array")
        ids64 = raw_ids.astype(np.int64, copy=False)
        if len(ids64) > np.iinfo(np.int32).max:
            raise ValueError("scale table exceeds int32 scale_id capacity")
        expected = np.arange(len(ids64), dtype=np.int64)
        if not np.array_equal(ids64, expected):
            raise ValueError("scale_id must be contiguous and start at zero")

        numeric = []
        for name in ("dx_grid_scale", "ds_frame_scale", "arc_length_grid_scale"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != ids64.shape:
                raise ValueError(f"{name} must have shape {ids64.shape}, got {values.shape}")
            if not np.isfinite(values).all() or np.any(values <= 0.0):
                raise ValueError(f"{name} must contain positive finite values")
            numeric.append(np.ascontiguousarray(values))
        if not len(ids64):
            raise ValueError("scale table must contain at least one scale")
        tuples = list(zip(numeric[0], numeric[1], numeric[2], strict=True))
        if len(set(tuples)) != len(tuples):
            raise ValueError("scale table contains duplicate numeric tuples")

        frozen_arrays = (
            np.ascontiguousarray(ids64, dtype=np.int32),
            numeric[0],
            numeric[1],
            numeric[2],
        )
        for array in frozen_arrays:
            array.setflags(write=False)
        object.__setattr__(self, "scale_id", frozen_arrays[0])
        object.__setattr__(self, "dx_grid_scale", frozen_arrays[1])
        object.__setattr__(self, "ds_frame_scale", frozen_arrays[2])
        object.__setattr__(self, "arc_length_grid_scale", frozen_arrays[3])

    def __len__(self) -> int:
        return int(len(self.scale_id))


@dataclass(frozen=True)
class ArcLengthPrimitiveResult:
    """Arc-length primitive batch and per-input-seed audit arrays.

    ``primitives`` contains only valid seeds, in input seed order.  Every other
    array is indexed by the complete input seed population.  ``line_steps`` is
    the number of accepted RK4 steps; for a path that reaches its target, its
    reported travel and end time are those of the linearly truncated endpoint.
    """

    primitives: np.ndarray
    valid_mask: np.ndarray
    line_steps: np.ndarray
    line_travel: np.ndarray
    line_end_time: np.ndarray
    line_reached_target: np.ndarray
    scale_id: np.ndarray
    dx_grid_scale: np.ndarray
    ds_frame_scale: np.ndarray
    arc_length_grid_scale: np.ndarray
    physical_dx: np.ndarray
    physical_dt: np.ndarray
    target_arc_length: np.ndarray
    integration_max_time: float

    @property
    def valid_seed_indices(self) -> np.ndarray:
        return np.flatnonzero(self.valid_mask)


def _config_scale_axis(values: Sequence[float], *, name: str) -> np.ndarray:
    items = list(values)
    if len(items) != SCALE_AXIS_COUNT:
        raise ValueError(f"{name} must contain exactly {SCALE_AXIS_COUNT} values")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in items):
        raise ValueError(f"{name} must contain numeric values, not booleans or strings")
    result = np.asarray(items, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")
    if len(np.unique(result)) != SCALE_AXIS_COUNT:
        raise ValueError(f"{name} must contain {SCALE_AXIS_COUNT} unique values")
    return result


def build_arc_length_scale_table(
    dx_grid_scales: Sequence[float],
    ds_frame_scales: Sequence[float],
    arc_length_grid_scales: Sequence[float],
) -> ArcLengthScaleTable:
    """Build the frozen 10 x 10 x 10 scale product from config-provided values.

    The first axis is outermost and the arc-length axis is innermost, so the
    exact order is ``dx -> ds -> arc``.  No scale values are synthesized here.
    """

    dx = _config_scale_axis(dx_grid_scales, name="dx_grid_scales")
    ds = _config_scale_axis(ds_frame_scales, name="ds_frame_scales")
    arc = _config_scale_axis(arc_length_grid_scales, name="arc_length_grid_scales")
    count = SCALE_AXIS_COUNT**3
    return ArcLengthScaleTable(
        scale_id=np.arange(count, dtype=np.int32),
        dx_grid_scale=np.repeat(dx, SCALE_AXIS_COUNT**2),
        ds_frame_scale=np.tile(
            np.repeat(ds, SCALE_AXIS_COUNT), SCALE_AXIS_COUNT
        ),
        arc_length_grid_scale=np.tile(arc, SCALE_AXIS_COUNT**2),
    )


def build_arc_length_scale_union(
    blocks: Sequence[Mapping[str, object]],
) -> ArcLengthScaleTable:
    """Build an explicit ordered union of frozen 10x10x10 Cartesian blocks.

    This is intentionally additive: :func:`build_arc_length_scale_table` remains
    the frozen 2.1 constructor.  Each union block must provide
    ``scale_id_start`` and the three explicit ten-value axes; block starts must
    exactly continue the preceding IDs.  No numeric values are synthesized.
    """

    items = list(blocks)
    if not items:
        raise ValueError("scale union must contain at least one block")
    dx_parts: list[np.ndarray] = []
    ds_parts: list[np.ndarray] = []
    arc_parts: list[np.ndarray] = []
    expected_start = 0
    for ordinal, raw_block in enumerate(items):
        if not isinstance(raw_block, Mapping):
            raise ValueError(f"scale union block {ordinal} must be a mapping")
        start = raw_block.get("scale_id_start")
        if isinstance(start, (bool, np.bool_)) or not isinstance(start, Real):
            raise ValueError(f"scale union block {ordinal} has no integer scale_id_start")
        start_value = int(start)
        if float(start) != float(start_value) or start_value != expected_start:
            raise ValueError(
                f"scale union block {ordinal} must start at {expected_start}, got {start}"
            )
        dx = _config_scale_axis(
            raw_block.get("dx_grid_scale", ()), name=f"blocks[{ordinal}].dx_grid_scale"
        )
        ds = _config_scale_axis(
            raw_block.get("ds_frame_scale", ()), name=f"blocks[{ordinal}].ds_frame_scale"
        )
        arc = _config_scale_axis(
            raw_block.get("arc_length_grid_scale", ()),
            name=f"blocks[{ordinal}].arc_length_grid_scale",
        )
        dx_parts.append(np.repeat(dx, SCALE_AXIS_COUNT**2))
        ds_parts.append(
            np.tile(np.repeat(ds, SCALE_AXIS_COUNT), SCALE_AXIS_COUNT)
        )
        arc_parts.append(np.tile(arc, SCALE_AXIS_COUNT**2))
        expected_start += SCALE_AXIS_COUNT**3
    return ArcLengthScaleTable(
        scale_id=np.arange(expected_start, dtype=np.int32),
        dx_grid_scale=np.concatenate(dx_parts),
        ds_frame_scale=np.concatenate(ds_parts),
        arc_length_grid_scale=np.concatenate(arc_parts),
    )


@njit(cache=True)
def _inside_domain(x, y, z, domain_min, domain_max):
    return (
        domain_min[0] <= x <= domain_max[0]
        and domain_min[1] <= y <= domain_max[1]
        and domain_min[2] <= z <= domain_max[2]
    )


@njit(cache=True)
def _finite_velocity(vx, vy, vz):
    return np.isfinite(vx) and np.isfinite(vy) and np.isfinite(vz)


@njit(inline="always", fastmath=True)
def _trilinear_component(
    field,
    time_index,
    z0,
    z1,
    y0,
    y1,
    x0,
    x1,
    wz,
    wy,
    wx,
    component,
):
    """Scalar trilinear interpolation without temporary three-vectors."""

    c00 = (
        field[time_index, z0, y0, x0, component] * (1.0 - wx)
        + field[time_index, z0, y0, x1, component] * wx
    )
    c10 = (
        field[time_index, z0, y1, x0, component] * (1.0 - wx)
        + field[time_index, z0, y1, x1, component] * wx
    )
    c01 = (
        field[time_index, z1, y0, x0, component] * (1.0 - wx)
        + field[time_index, z1, y0, x1, component] * wx
    )
    c11 = (
        field[time_index, z1, y1, x0, component] * (1.0 - wx)
        + field[time_index, z1, y1, x1, component] * wx
    )
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wz) + c1 * wz


@njit(inline="always", fastmath=True)
def _interp4_quadrilinear_scalar(
    field,
    domain_min,
    spacing,
    xdim,
    ydim,
    zdim,
    tmin,
    time_interval,
    time_steps,
    x,
    y,
    z,
    time,
):
    """Quadrilinear velocity interpolation using scalar component loads.

    FMT's general interpolation helper constructs several temporary
    three-component arrays.  RK4 calls it four times per step, so those
    temporaries dominate this 64,000-seed workload.  This specialization keeps
    the same corner and interpolation order but computes each component as a
    scalar.
    """

    gx = (x - domain_min[0]) / spacing[0]
    gy = (y - domain_min[1]) / spacing[1]
    gz = (z - domain_min[2]) / spacing[2]
    x0 = max(0, min(int(np.floor(gx)), xdim - 1))
    x1 = max(0, min(int(np.ceil(gx)), xdim - 1))
    y0 = max(0, min(int(np.floor(gy)), ydim - 1))
    y1 = max(0, min(int(np.ceil(gy)), ydim - 1))
    z0 = max(0, min(int(np.floor(gz)), zdim - 1))
    z1 = max(0, min(int(np.ceil(gz)), zdim - 1))
    wx = gx - x0
    wy = gy - y0
    wz = gz - z0

    grid_time = (time - tmin) / time_interval
    t0 = max(0, min(int(np.floor(grid_time)), time_steps - 1))
    t1 = max(0, min(int(np.ceil(grid_time)), time_steps - 1))
    time_weight = grid_time - t0

    ax = _trilinear_component(
        field, t0, z0, z1, y0, y1, x0, x1, wz, wy, wx, 0
    )
    ay = _trilinear_component(
        field, t0, z0, z1, y0, y1, x0, x1, wz, wy, wx, 1
    )
    az = _trilinear_component(
        field, t0, z0, z1, y0, y1, x0, x1, wz, wy, wx, 2
    )
    bx = _trilinear_component(
        field, t1, z0, z1, y0, y1, x0, x1, wz, wy, wx, 0
    )
    by = _trilinear_component(
        field, t1, z0, z1, y0, y1, x0, x1, wz, wy, wx, 1
    )
    bz = _trilinear_component(
        field, t1, z0, z1, y0, y1, x0, x1, wz, wy, wx, 2
    )
    return (
        ax * (1.0 - time_weight) + bx * time_weight,
        ay * (1.0 - time_weight) + by * time_weight,
        az * (1.0 - time_weight) + bz * time_weight,
    )


@njit(cache=True)
def _rk4_forward_step(
    field,
    domain_min,
    domain_max,
    spacing,
    xdim,
    ydim,
    zdim,
    tmin,
    time_interval,
    time_steps,
    px,
    py,
    pz,
    time,
    step,
):
    """Return one fail-closed forward RK4 step."""

    if not _inside_domain(px, py, pz, domain_min, domain_max):
        return False, px, py, pz
    v1x, v1y, v1z = _interp4_quadrilinear_scalar(
        field,
        domain_min,
        spacing,
        xdim,
        ydim,
        zdim,
        tmin,
        time_interval,
        time_steps,
        px,
        py,
        pz,
        time,
    )
    if not _finite_velocity(v1x, v1y, v1z):
        return False, px, py, pz

    p2x = px + 0.5 * step * v1x
    p2y = py + 0.5 * step * v1y
    p2z = pz + 0.5 * step * v1z
    if not _inside_domain(p2x, p2y, p2z, domain_min, domain_max):
        return False, px, py, pz
    v2x, v2y, v2z = _interp4_quadrilinear_scalar(
        field,
        domain_min,
        spacing,
        xdim,
        ydim,
        zdim,
        tmin,
        time_interval,
        time_steps,
        p2x,
        p2y,
        p2z,
        time + 0.5 * step,
    )
    if not _finite_velocity(v2x, v2y, v2z):
        return False, px, py, pz

    p3x = px + 0.5 * step * v2x
    p3y = py + 0.5 * step * v2y
    p3z = pz + 0.5 * step * v2z
    if not _inside_domain(p3x, p3y, p3z, domain_min, domain_max):
        return False, px, py, pz
    v3x, v3y, v3z = _interp4_quadrilinear_scalar(
        field,
        domain_min,
        spacing,
        xdim,
        ydim,
        zdim,
        tmin,
        time_interval,
        time_steps,
        p3x,
        p3y,
        p3z,
        time + 0.5 * step,
    )
    if not _finite_velocity(v3x, v3y, v3z):
        return False, px, py, pz

    p4x = px + step * v3x
    p4y = py + step * v3y
    p4z = pz + step * v3z
    if not _inside_domain(p4x, p4y, p4z, domain_min, domain_max):
        return False, px, py, pz
    v4x, v4y, v4z = _interp4_quadrilinear_scalar(
        field,
        domain_min,
        spacing,
        xdim,
        ydim,
        zdim,
        tmin,
        time_interval,
        time_steps,
        p4x,
        p4y,
        p4z,
        time + step,
    )
    if not _finite_velocity(v4x, v4y, v4z):
        return False, px, py, pz

    nx = px + step * (v1x + 2.0 * v2x + 2.0 * v3x + v4x) / 6.0
    ny = py + step * (v1y + 2.0 * v2y + 2.0 * v3y + v4y) / 6.0
    nz = pz + step * (v1z + 2.0 * v2z + 2.0 * v3z + v4z) / 6.0
    if not _inside_domain(nx, ny, nz, domain_min, domain_max):
        return False, px, py, pz
    return True, nx, ny, nz


@njit(parallel=True, cache=True)
def _integrate_per_seed_scale_batch(
    field,
    domain_min,
    domain_max,
    spacing,
    xdim,
    ydim,
    zdim,
    tmin,
    time_interval,
    time_steps,
    seeds,
    seed_time,
    max_time,
    physical_dx_by_seed,
    physical_dt_by_seed,
    target_arc_length_by_seed,
    maximum_steps,
):
    """Integrate all assigned scales in one parallel launch.

    The phase-1 workload contains only 64 seeds for each of 1,000 scales.  A
    scale-by-scale parallel kernel therefore spends substantially more time in
    thread-team startup than in useful work.  Reading three scalar parameters
    per seed keeps the arithmetic identical while exposing all seeds to one
    ``prange``.
    """

    count = len(seeds)
    samples = np.full(
        (count, LINE_COUNT, SAMPLED_POINTS, 4), np.nan, dtype=np.float32
    )
    line_steps = np.zeros((count, LINE_COUNT), dtype=np.int32)
    line_travel = np.zeros((count, LINE_COUNT), dtype=np.float64)
    line_end_time = np.full((count, LINE_COUNT), seed_time, dtype=np.float64)
    reached = np.zeros((count, LINE_COUNT), dtype=np.bool_)
    time_tolerance = max(1e-12, abs(max_time) * 1e-12)

    for seed_index in prange(count):
        physical_dx = physical_dx_by_seed[seed_index]
        physical_dt = physical_dt_by_seed[seed_index]
        target_arc_length = target_arc_length_by_seed[seed_index]
        arc_tolerance = max(1e-12, target_arc_length * 1e-12)
        for line_index in range(LINE_COUNT):
            px = float(seeds[seed_index, 0])
            py = float(seeds[seed_index, 1])
            pz = float(seeds[seed_index, 2])
            if line_index == 1:
                px += physical_dx
            elif line_index == 2:
                px -= physical_dx
            elif line_index == 3:
                py += physical_dx
            elif line_index == 4:
                py -= physical_dx
            elif line_index == 5:
                pz += physical_dx
            elif line_index == 6:
                pz -= physical_dx
            time = seed_time
            samples[seed_index, line_index, 0, 0] = px
            samples[seed_index, line_index, 0, 1] = py
            samples[seed_index, line_index, 0, 2] = pz
            samples[seed_index, line_index, 0, 3] = time
            if not _inside_domain(px, py, pz, domain_min, domain_max):
                continue

            accumulated = 0.0
            next_sample = 1
            accepted_steps = 0
            truncated_end_time = seed_time
            for _ in range(maximum_steps):
                remaining = max_time - time
                if remaining <= time_tolerance:
                    time = max_time
                    break
                step = min(physical_dt, remaining)
                ok, nx, ny, nz = _rk4_forward_step(
                    field,
                    domain_min,
                    domain_max,
                    spacing,
                    xdim,
                    ydim,
                    zdim,
                    tmin,
                    time_interval,
                    time_steps,
                    px,
                    py,
                    pz,
                    time,
                    step,
                )
                if not ok:
                    break
                new_time = time + step
                segment = np.sqrt(
                    (nx - px) * (nx - px)
                    + (ny - py) * (ny - py)
                    + (nz - pz) * (nz - pz)
                )
                accepted_steps += 1
                if segment > 0.0:
                    after = accumulated + segment
                    while next_sample < SAMPLED_POINTS:
                        target = (
                            target_arc_length
                            * next_sample
                            / (SAMPLED_POINTS - 1)
                        )
                        if target > after + arc_tolerance:
                            break
                        alpha = (target - accumulated) / segment
                        alpha = min(1.0, max(0.0, alpha))
                        samples[seed_index, line_index, next_sample, 0] = (
                            px + alpha * (nx - px)
                        )
                        samples[seed_index, line_index, next_sample, 1] = (
                            py + alpha * (ny - py)
                        )
                        samples[seed_index, line_index, next_sample, 2] = (
                            pz + alpha * (nz - pz)
                        )
                        sample_time = time + alpha * step
                        samples[seed_index, line_index, next_sample, 3] = sample_time
                        if next_sample == SAMPLED_POINTS - 1:
                            truncated_end_time = sample_time
                        next_sample += 1
                    accumulated = after

                px, py, pz, time = nx, ny, nz, new_time
                if next_sample == SAMPLED_POINTS:
                    reached[seed_index, line_index] = True
                    accumulated = target_arc_length
                    time = truncated_end_time
                    break

            line_steps[seed_index, line_index] = accepted_steps
            line_travel[seed_index, line_index] = accumulated
            line_end_time[seed_index, line_index] = time

    return samples, line_steps, line_travel, line_end_time, reached


def integrate_arc_length_primitives_3d(
    vector_field: UnsteadyVectorField3D,
    seeds_xyz: np.ndarray,
    seed_time: float,
    scales: ArcLengthScaleTable,
    scale_assignment: np.ndarray,
    *,
    chunk_size: int = 2048,
    maximum_source_frame_intervals: float = MAX_SOURCE_FRAME_INTERVALS,
) -> ArcLengthPrimitiveResult:
    """Integrate one arc-length scale per seed and return 7x32 valid primitives.

    Physical parameters are
    ``dx = dx_grid_scale * min(grid spacing)``,
    ``dt = ds_frame_scale * source frame interval``, and
    ``target arc = arc_length_grid_scale * min(grid spacing)``.
    Integration ends after ``maximum_source_frame_intervals`` (12 by default);
    a final RK4 step is clamped to that time.  The input field must contain the
    complete window.
    """

    if not isinstance(vector_field, UnsteadyVectorField3D):
        raise TypeError("vector_field must be an UnsteadyVectorField3D")
    if not isinstance(scales, ArcLengthScaleTable):
        raise TypeError("scales must be an ArcLengthScaleTable")
    seeds = np.ascontiguousarray(np.asarray(seeds_xyz, dtype=np.float64))
    if seeds.ndim != 2 or seeds.shape[1] != 3 or not np.isfinite(seeds).all():
        raise ValueError(f"seeds_xyz must be finite [N,3], got {seeds.shape}")
    raw_assignment = np.asarray(scale_assignment)
    if raw_assignment.shape != (len(seeds),) or raw_assignment.dtype.kind not in "iu":
        raise ValueError("scale_assignment must contain one integer scale id per seed")
    assignment = raw_assignment.astype(np.int64, copy=False)
    if assignment.size and (assignment.min() < 0 or assignment.max() >= len(scales)):
        raise ValueError("scale_assignment contains an unknown scale id")
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    seed_time = float(seed_time)
    if not np.isfinite(seed_time) or not vector_field.tmin <= seed_time <= vector_field.tmax:
        raise ValueError("seed_time must be finite and inside the vector-field time range")
    source_interval = float(vector_field.time_interval)
    if not np.isfinite(source_interval) or source_interval <= 0.0:
        raise ValueError("arc-length integration requires a positive source frame interval")
    horizon = float(maximum_source_frame_intervals)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("maximum_source_frame_intervals must be positive and finite")
    max_time = seed_time + horizon * source_interval
    time_tolerance = max(1e-12, abs(max_time) * 1e-12)
    if max_time > vector_field.tmax + time_tolerance:
        raise ValueError(
            "vector field does not contain "
            f"{horizon:g} source-frame intervals after seed_time"
        )

    minimum_spacing = float(np.min(vector_field.grid_interval))
    per_seed_scale_id = np.ascontiguousarray(scales.scale_id[assignment], dtype=np.int32)
    dx_grid_scale = np.ascontiguousarray(scales.dx_grid_scale[assignment])
    ds_frame_scale = np.ascontiguousarray(scales.ds_frame_scale[assignment])
    arc_grid_scale = np.ascontiguousarray(scales.arc_length_grid_scale[assignment])
    physical_dx = np.ascontiguousarray(dx_grid_scale * minimum_spacing)
    physical_dt = np.ascontiguousarray(ds_frame_scale * source_interval)
    target_arc = np.ascontiguousarray(arc_grid_scale * minimum_spacing)

    duration = max_time - seed_time
    minimum_dt = float(np.min(physical_dt)) if len(physical_dt) else duration
    step_bound = float(np.ceil(duration / minimum_dt)) + 1.0
    if not np.isfinite(step_bound) or step_bound > np.iinfo(np.int32).max:
        raise ValueError("assigned scales require too many RK4 steps for int32 diagnostics")
    maximum_steps = int(step_bound)
    # ``chunk_size`` remains a validated public compatibility parameter.  The
    # numerical kernel is deliberately launched once: splitting 64,000 seeds
    # into 1,000 tiny scale batches dominated the phase-1 runtime.  ``prange``
    # performs its own scheduling without changing per-seed arithmetic order.
    del chunk_size
    all_samples, line_steps, line_travel, line_end_time, reached = (
        _integrate_per_seed_scale_batch(
            vector_field.field,
            vector_field.domain_min,
            vector_field.domain_max,
            vector_field.grid_interval,
            vector_field.xdim,
            vector_field.ydim,
            vector_field.zdim,
            vector_field.tmin,
            source_interval,
            vector_field.time_steps,
            seeds,
            seed_time,
            max_time,
            physical_dx,
            physical_dt,
            target_arc,
            maximum_steps,
        )
    )

    valid = reached.all(axis=1)
    valid_samples = all_samples if np.all(valid) else np.ascontiguousarray(all_samples[valid])
    return ArcLengthPrimitiveResult(
        primitives=valid_samples,
        valid_mask=valid,
        line_steps=line_steps,
        line_travel=line_travel,
        line_end_time=line_end_time,
        line_reached_target=reached,
        scale_id=per_seed_scale_id,
        dx_grid_scale=dx_grid_scale,
        ds_frame_scale=ds_frame_scale,
        arc_length_grid_scale=arc_grid_scale,
        physical_dx=physical_dx,
        physical_dt=physical_dt,
        target_arc_length=target_arc,
        integration_max_time=float(max_time),
    )

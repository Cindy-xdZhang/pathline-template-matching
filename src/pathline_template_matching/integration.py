"""Numba RK4/Euler pathline integration extracted from FMT's 3D core."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .vector_field import UnsteadyVectorField3D


@njit(fastmath=True, cache=True)
def _interp3_trilinear(volume, domain_min, spacing, xdim, ydim, zdim, x, y, z):
    gx = (x - domain_min[0]) / spacing[0]
    gy = (y - domain_min[1]) / spacing[1]
    gz = (z - domain_min[2]) / spacing[2]
    x0 = max(0, min(int(np.floor(gx)), xdim - 1))
    x1 = max(0, min(int(np.ceil(gx)), xdim - 1))
    y0 = max(0, min(int(np.floor(gy)), ydim - 1))
    y1 = max(0, min(int(np.ceil(gy)), ydim - 1))
    z0 = max(0, min(int(np.floor(gz)), zdim - 1))
    z1 = max(0, min(int(np.ceil(gz)), zdim - 1))
    wx, wy, wz = gx - x0, gy - y0, gz - z0
    c00 = volume[z0, y0, x0] * (1.0 - wx) + volume[z0, y0, x1] * wx
    c10 = volume[z0, y1, x0] * (1.0 - wx) + volume[z0, y1, x1] * wx
    c01 = volume[z1, y0, x0] * (1.0 - wx) + volume[z1, y0, x1] * wx
    c11 = volume[z1, y1, x0] * (1.0 - wx) + volume[z1, y1, x1] * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    result = c0 * (1.0 - wz) + c1 * wz
    return result[0], result[1], result[2]


@njit(fastmath=True, cache=True)
def _interp4_quadrilinear(
    field, domain_min, spacing, xdim, ydim, zdim,
    tmin, time_interval, time_steps, x, y, z, time,
):
    grid_time = (time - tmin) / time_interval if time_interval != 0.0 else 0.0
    t0 = max(0, min(int(np.floor(grid_time)), time_steps - 1))
    t1 = max(0, min(int(np.ceil(grid_time)), time_steps - 1))
    weight = grid_time - t0
    ax, ay, az = _interp3_trilinear(
        field[t0], domain_min, spacing, xdim, ydim, zdim, x, y, z
    )
    bx, by, bz = _interp3_trilinear(
        field[t1], domain_min, spacing, xdim, ydim, zdim, x, y, z
    )
    return (
        ax * (1.0 - weight) + bx * weight,
        ay * (1.0 - weight) + by * weight,
        az * (1.0 - weight) + bz * weight,
    )


@njit(fastmath=True, cache=True)
def _path_one_direction(
    field, domain_min, domain_max, spacing, xdim, ydim, zdim,
    tmin, time_interval, time_steps, sx, sy, sz, start_time, end_time,
    absolute_step, max_iterations, method_id, output,
):
    direction = 1.0 if end_time > start_time else -1.0
    step = np.float64(absolute_step) * direction
    px, py, pz = sx, sy, sz
    time = np.float64(start_time)
    end = np.float64(end_time)
    count = 0
    for _ in range(max_iterations):
        if (direction > 0 and time >= end) or (direction < 0 and time <= end):
            break
        if (
            px < domain_min[0] or px > domain_max[0]
            or py < domain_min[1] or py > domain_max[1]
            or pz < domain_min[2] or pz > domain_max[2]
        ):
            break
        v1x, v1y, v1z = _interp4_quadrilinear(
            field, domain_min, spacing, xdim, ydim, zdim,
            tmin, time_interval, time_steps, px, py, pz, time,
        )
        if method_id == 1:
            v2x, v2y, v2z = _interp4_quadrilinear(
                field, domain_min, spacing, xdim, ydim, zdim,
                tmin, time_interval, time_steps,
                px + 0.5 * step * v1x, py + 0.5 * step * v1y,
                pz + 0.5 * step * v1z, time + 0.5 * step,
            )
            v3x, v3y, v3z = _interp4_quadrilinear(
                field, domain_min, spacing, xdim, ydim, zdim,
                tmin, time_interval, time_steps,
                px + 0.5 * step * v2x, py + 0.5 * step * v2y,
                pz + 0.5 * step * v2z, time + 0.5 * step,
            )
            v4x, v4y, v4z = _interp4_quadrilinear(
                field, domain_min, spacing, xdim, ydim, zdim,
                tmin, time_interval, time_steps,
                px + step * v3x, py + step * v3y, pz + step * v3z,
                time + step,
            )
            dx = step * (v1x + 2 * v2x + 2 * v3x + v4x) / 6.0
            dy = step * (v1y + 2 * v2y + 2 * v3y + v4y) / 6.0
            dz = step * (v1z + 2 * v2z + 2 * v3z + v4z) / 6.0
        else:
            dx, dy, dz = step * v1x, step * v1y, step * v1z
        px += np.float32(dx)
        py += np.float32(dy)
        pz += np.float32(dz)
        time += step
        output[count, 0] = px
        output[count, 1] = py
        output[count, 2] = pz
        output[count, 3] = time
        count += 1
    return count


@njit(parallel=True, fastmath=True, cache=True)
def _batch_pathlines(
    field, domain_min, domain_max, spacing, xdim, ydim, zdim,
    tmin, time_interval, time_steps, seeds, min_time, max_time,
    absolute_step, max_iterations, method_id, output, lengths,
):
    for seed_index in prange(seeds.shape[0]):
        forward = np.empty((max_iterations, 4), np.float32)
        backward = np.empty((max_iterations, 4), np.float32)
        sx = np.float32(seeds[seed_index, 0])
        sy = np.float32(seeds[seed_index, 1])
        sz = np.float32(seeds[seed_index, 2])
        start_time = seeds[seed_index, 3]
        forward_count = _path_one_direction(
            field, domain_min, domain_max, spacing, xdim, ydim, zdim,
            tmin, time_interval, time_steps, sx, sy, sz, start_time,
            max_time, absolute_step, max_iterations, method_id, forward,
        )
        backward_count = _path_one_direction(
            field, domain_min, domain_max, spacing, xdim, ydim, zdim,
            tmin, time_interval, time_steps, sx, sy, sz, start_time,
            min_time, absolute_step, max_iterations, method_id, backward,
        )
        output_index = 0
        for index in range(backward_count - 1, -1, -1):
            output[seed_index, output_index] = backward[index]
            output_index += 1
        output[seed_index, output_index, 0] = sx
        output[seed_index, output_index, 1] = sy
        output[seed_index, output_index, 2] = sz
        output[seed_index, output_index, 3] = start_time
        output_index += 1
        for index in range(forward_count):
            output[seed_index, output_index] = forward[index]
            output_index += 1
        lengths[seed_index] = output_index


def compute_pathlines_3d_batch(
    vector_field: UnsteadyVectorField3D,
    seeds_xyzt: np.ndarray,
    min_time: float,
    max_time: float,
    step_size: float,
    max_iterations: int,
    method: str = "RK4",
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a batch and return padded ``[N,2M+1,4]`` paths and lengths."""

    method_ids = {"EULER": 0, "RK4": 1}
    method_id = method_ids.get(str(method).upper())
    if method_id is None:
        raise ValueError(f"unsupported integration method: {method!r}")
    seeds = np.ascontiguousarray(np.asarray(seeds_xyzt, dtype=np.float64))
    if seeds.ndim != 2 or seeds.shape[1] != 4:
        raise ValueError(f"seeds must be [N,4], got {seeds.shape}")
    if not np.isfinite(seeds).all():
        raise ValueError("seeds contain NaN or Inf")
    max_iterations = int(max_iterations)
    if max_iterations < 1 or not np.isfinite(step_size) or float(step_size) <= 0:
        raise ValueError("step_size and max_iterations must be positive")
    output = np.zeros((len(seeds), 2 * max_iterations + 1, 4), dtype=np.float32)
    lengths = np.zeros(len(seeds), dtype=np.int32)
    _batch_pathlines(
        vector_field.field,
        vector_field.domain_min,
        vector_field.domain_max,
        vector_field.grid_interval,
        vector_field.xdim,
        vector_field.ydim,
        vector_field.zdim,
        vector_field.tmin,
        vector_field.time_interval,
        vector_field.time_steps,
        seeds,
        np.float64(min_time),
        np.float64(max_time),
        np.float64(step_size),
        max_iterations,
        method_id,
        output,
        lengths,
    )
    return output, lengths

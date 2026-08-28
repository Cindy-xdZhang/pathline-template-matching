"""Deterministic unsteady channel field under the inherited Killing observer.

The observer equations are adapted from FMT's ``KillingObserver3D.py``
(source SHA-256 ``aaaaaedee75e3ab9b106ec6a6c0a7fd5a415b319c8d596107cff9cbd614cf752``).
The VTK loading/cropping contract is adapted from
``Build_Channel_Killing_Cache.py`` (source SHA-256
``5f43fa77b103a551ccf038e9c067fedd6cf84e02e171b7357a51c7b1ec2f0a3c``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .netcdf_io import FlowWindow3D


@dataclass(frozen=True)
class SteadyChannelGrid:
    interpolator: Callable[[np.ndarray], np.ndarray]
    output_points_xyz: np.ndarray
    output_coordinates_xyz: tuple[np.ndarray, np.ndarray, np.ndarray]
    source_min: np.ndarray
    source_max: np.ndarray
    metadata: dict[str, Any]


def _hat(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation_exp(angular_velocity: np.ndarray, dt: float) -> np.ndarray:
    angular = np.asarray(angular_velocity, dtype=np.float64)
    angle = float(np.linalg.norm(angular)) * float(dt)
    matrix = _hat(angular) * float(dt)
    if angle < 1e-12:
        return np.eye(3) + matrix + 0.5 * matrix @ matrix
    matrix /= angle
    return np.eye(3) + np.sin(angle) * matrix + (1.0 - np.cos(angle)) * matrix @ matrix


def integrate_killing_frame(
    parameters: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate translation/angular velocity into rotation and displacement."""

    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) < 2:
        raise ValueError("parameters must be a finite [T>=2,6] matrix")
    if not np.isfinite(values).all() or not np.isfinite(dt) or float(dt) <= 0:
        raise ValueError("Killing parameters and dt must be finite; dt must be positive")
    count = len(values)
    rotation = np.empty((count, 3, 3), dtype=np.float64)
    rotation[0] = np.eye(3)
    for index in range(1, count):
        average = 0.5 * (values[index - 1, 3:] + values[index, 3:])
        rotation[index] = _rotation_exp(average, dt) @ rotation[index - 1]
    integrand = np.einsum("tji,tj->ti", rotation, values[:, :3])
    displacement = np.zeros((count, 3), dtype=np.float64)
    displacement[1:] = np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * float(dt), axis=0
    )
    return rotation, displacement


def smooth_channel_observer(
    normalized_times: np.ndarray,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
) -> np.ndarray:
    """Return the frozen small-amplitude time-varying rigid observer."""

    times = np.asarray(normalized_times, dtype=np.float64)
    lower = np.asarray(domain_min, dtype=np.float64)
    upper = np.asarray(domain_max, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("normalized_times must contain at least two finite values")
    if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
        raise ValueError("channel domain bounds are invalid")
    center = 0.5 * (lower + upper)
    span = upper - lower
    axis = np.asarray([0.35, 0.55, 0.76], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    phase = 2.0 * np.pi * times
    amplitude = 0.035
    theta_dot = (
        amplitude * 2.0 * np.pi * np.cos(phase)
        + 0.35 * amplitude * 4.0 * np.pi * np.cos(2.0 * phase)
    )
    angular = theta_dot[:, None] * axis[None]
    translation_amplitude = np.asarray([0.012, 0.018, 0.012]) * span
    moving_center = center + translation_amplitude * np.stack(
        (np.sin(phase), np.sin(phase + 1.1), np.sin(phase + 2.0)), axis=-1
    )
    center_velocity = translation_amplitude * 2.0 * np.pi * np.stack(
        (np.cos(phase), np.cos(phase + 1.1), np.cos(phase + 2.0)), axis=-1
    )
    translation = center_velocity - np.cross(angular, moving_center)
    return np.concatenate((translation, angular), axis=1)


def compose_steady_to_unsteady(
    points_xyz: np.ndarray,
    steady_interpolator: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    rotation: np.ndarray,
    displacement: np.ndarray,
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    """Evaluate ``v=R*s(R^T*x-D)+translation+angular cross x``."""

    points = np.asarray(points_xyz, dtype=np.float64)
    values = np.asarray(parameters, dtype=np.float64)
    output = np.empty((len(values), len(points), 3), dtype=np.float32)
    for index, (parameter, matrix, shift) in enumerate(
        zip(values, rotation, displacement, strict=True)
    ):
        observed = points @ matrix - shift
        outside = (
            (observed < np.asarray(bounds_min)) | (observed > np.asarray(bounds_max))
        ).any(axis=1)
        if outside.any():
            raise ValueError(
                f"observer inverse map leaves steady domain at index {index}: "
                f"{int(outside.sum())}/{len(points)} points"
            )
        steady = np.asarray(steady_interpolator(observed), dtype=np.float64)
        if steady.shape != points.shape or not np.isfinite(steady).all():
            raise ValueError("steady channel interpolation returned invalid velocity")
        observer_velocity = parameter[:3] + np.cross(
            np.broadcast_to(parameter[3:], points.shape), points
        )
        output[index] = (steady @ matrix.T + observer_velocity).astype(np.float32)
    return output


def load_steady_channel_vtk(
    path: str | Path,
    *,
    max_spatial_dim: int = 96,
    crop_fraction: float = 0.10,
) -> SteadyChannelGrid:
    """Load a separable structured VTK grid and define the cropped output grid."""

    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as error:
        raise RuntimeError("channel preprocessing requires scipy and vtk") from error
    source = Path(path)
    max_spatial_dim = int(max_spatial_dim)
    crop_fraction = float(crop_fraction)
    if max_spatial_dim < 2 or not 0.0 < crop_fraction < 0.5:
        raise ValueError("max_spatial_dim must be >=2 and crop_fraction must be in (0,0.5)")
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(str(source))
    reader.ReadAllVectorsOn()
    reader.Update()
    grid = reader.GetOutput()
    dimensions = [0, 0, 0]
    grid.GetDimensions(dimensions)
    x_count, y_count, z_count = dimensions
    if min(dimensions) < 2 or grid.GetPoints() is None:
        raise ValueError("channel VTK does not contain a 3D structured point grid")
    points = vtk_to_numpy(grid.GetPoints().GetData()).reshape(
        z_count, y_count, x_count, 3
    )
    velocity_array = grid.GetPointData().GetArray("velocity")
    if velocity_array is None or velocity_array.GetNumberOfComponents() != 3:
        raise ValueError("channel VTK misses 3-component point array 'velocity'")
    velocity = vtk_to_numpy(velocity_array).reshape(z_count, y_count, x_count, 3)
    if not np.isfinite(points).all() or not np.isfinite(velocity).all():
        raise ValueError("channel VTK contains NaN or Inf")
    x = np.asarray(points[0, 0, :, 0], dtype=np.float64)
    y = np.asarray(points[0, :, 0, 1], dtype=np.float64)
    z = np.asarray(points[:, 0, 0, 2], dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    expected = np.stack((xx, yy, zz), axis=-1)
    if not np.allclose(points, expected, rtol=0.0, atol=1e-6):
        raise ValueError("channel structured grid is not separable into x/y/z axes")
    for name, axis in zip("xyz", (x, y, z), strict=True):
        if np.any(np.diff(axis) <= 0):
            raise ValueError(f"channel {name} coordinate is not strictly increasing")
    grid_interpolator = RegularGridInterpolator(
        (z, y, x), np.asarray(velocity, dtype=np.float32), bounds_error=True
    )

    def interpolate(points_xyz: np.ndarray) -> np.ndarray:
        return grid_interpolator(np.asarray(points_xyz)[:, [2, 1, 0]])

    lower = np.asarray([x[0], y[0], z[0]], dtype=np.float64)
    upper = np.asarray([x[-1], y[-1], z[-1]], dtype=np.float64)
    cropped_lower = lower + crop_fraction * (upper - lower)
    cropped_upper = upper - crop_fraction * (upper - lower)
    strides = np.maximum(1, np.ceil(np.asarray(dimensions) / max_spatial_dim).astype(int))
    counts = np.ceil(np.asarray(dimensions) / strides).astype(int)
    output_coordinates = tuple(
        np.linspace(cropped_lower[index], cropped_upper[index], counts[index])
        for index in range(3)
    )
    output_x, output_y, output_z = output_coordinates
    out_z, out_y, out_x = np.meshgrid(output_z, output_y, output_x, indexing="ij")
    output_points = np.stack(
        (out_x.ravel(), out_y.ravel(), out_z.ravel()), axis=-1
    )
    return SteadyChannelGrid(
        interpolator=interpolate,
        output_points_xyz=output_points,
        output_coordinates_xyz=output_coordinates,
        source_min=lower,
        source_max=upper,
        metadata={
            "source_dimensions_xyz": [int(value) for value in dimensions],
            "source_bounds": [lower.tolist(), upper.tolist()],
            "output_counts_xyz": [int(value) for value in counts],
            "output_bounds": [cropped_lower.tolist(), cropped_upper.tolist()],
            "nominal_stride_xyz": [int(value) for value in strides],
            "output_policy": "cropped_regular_resampling",
            "crop_fraction": crop_fraction,
        },
    )


def build_channel_flow_window(
    source_path: str | Path,
    source_start_index: int,
    frame_count: int,
    *,
    max_spatial_dim: int = 96,
    crop_fraction: float = 0.10,
    total_frames: int = 159,
    duration: float = 1.0,
    steady_grid: SteadyChannelGrid | None = None,
) -> tuple[FlowWindow3D, dict[str, Any]]:
    """Construct one canonical unsteady channel window from the steady VTK."""

    total_frames = int(total_frames)
    start = int(source_start_index)
    count = int(frame_count)
    if total_frames < 2 or count < 2 or start < 0 or start + count > total_frames:
        raise ValueError("channel source window is outside the frozen 159-frame series")
    if not np.isfinite(duration) or float(duration) <= 0:
        raise ValueError("channel duration must be positive and finite")
    steady = steady_grid or load_steady_channel_vtk(
        source_path,
        max_spatial_dim=max_spatial_dim,
        crop_fraction=crop_fraction,
    )
    times = np.linspace(0.0, float(duration), total_frames)
    source_dt = float(times[1] - times[0])
    parameters = smooth_channel_observer(
        times / float(duration), steady.source_min, steady.source_max
    )
    parameters /= float(duration)
    rotation, displacement = integrate_killing_frame(parameters, source_dt)
    selection = slice(start, start + count)
    flat = compose_steady_to_unsteady(
        steady.output_points_xyz,
        steady.interpolator,
        parameters[selection],
        rotation[selection],
        displacement[selection],
        bounds_min=steady.source_min,
        bounds_max=steady.source_max,
    )
    x, y, z = steady.output_coordinates_xyz
    velocity = flat.reshape(count, len(z), len(y), len(x), 3)
    nominal_stride = steady.metadata["nominal_stride_xyz"]
    window = FlowWindow3D(
        velocity=velocity,
        coordinates_xyz=(x, y, z),
        time=times[selection],
        source_path=str(Path(source_path).resolve()),
        source_start_index=start,
        spatial_strides={
            axis: int(value)
            for axis, value in zip("xyz", nominal_stride, strict=True)
        },
        components=("velocity_x", "velocity_y", "velocity_z"),
        coordinate_sources={
            "x": "vtk_structured_grid_then_cropped_regular_resampling",
            "y": "vtk_structured_grid_then_cropped_regular_resampling",
            "z": "vtk_structured_grid_then_cropped_regular_resampling",
            "t": "deterministic_killing_observer_series",
        },
    )
    metadata = {
        **steady.metadata,
        "observer_formula": "xi=R^T*x-D; v=R*s(xi)+translation+angular_cross_x",
        "observer_total_frames": total_frames,
        "observer_duration": float(duration),
        "observer_source_time_step": source_dt,
        "observer_source_start_index": start,
        "observer_frame_count": count,
    }
    return window, metadata

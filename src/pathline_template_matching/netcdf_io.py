"""Fail-closed, out-of-core access to regular 3D NetCDF velocity fields.

This module is adapted from FMT's ``NetCDF_window_3D.py``. It preserves the
dimension-aware transpose and spatial striding, then adds checks for masked
values, finite values, coordinate monotonicity, and uniform spatial spacing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4 as nc
import numpy as np


_AXIS_ALIASES = {
    "x": {"x", "xdim"},
    "y": {"y", "ydim"},
    "z": {"z", "zdim"},
    "t": {"t", "time", "tdim"},
}
_COMPONENT_CANDIDATES = (
    ("u", "v", "w"),
    ("velocity_x", "velocity_y", "velocity_z"),
    ("Component1", "Component2", "Component3"),
    ("a", "b", "c"),
)


@dataclass(frozen=True)
class FlowWindow3D:
    """A regular velocity window in canonical ``[T,Z,Y,X,3]`` order."""

    velocity: np.ndarray
    coordinates_xyz: tuple[np.ndarray, np.ndarray, np.ndarray]
    time: np.ndarray
    source_path: str
    source_start_index: int
    spatial_strides: dict[str, int]
    components: tuple[str, str, str]
    coordinate_sources: dict[str, str]

    @property
    def spacing_xyz(self) -> np.ndarray:
        return np.asarray(
            [values[1] - values[0] for values in self.coordinates_xyz],
            dtype=np.float64,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "source_start_index": self.source_start_index,
            "frame_count": int(self.velocity.shape[0]),
            "loaded_shape_TZYXC": list(self.velocity.shape),
            "spatial_strides": dict(self.spatial_strides),
            "components": list(self.components),
            "coordinate_sources": dict(self.coordinate_sources),
            "spacing_xyz": self.spacing_xyz.tolist(),
            "source_time_min": float(self.time[0]),
            "source_time_max": float(self.time[-1]),
        }


def _axis_dimension(dataset: nc.Dataset, axis: str) -> str:
    for name in dataset.dimensions:
        if name.lower() in _AXIS_ALIASES[axis]:
            return name
    raise ValueError(
        f"NetCDF misses a {axis!r} dimension; found {list(dataset.dimensions)}"
    )


def _axis_for_dimension(dimension: str) -> str:
    lowered = dimension.lower()
    matches = [axis for axis, aliases in _AXIS_ALIASES.items() if lowered in aliases]
    if len(matches) != 1:
        raise ValueError(f"cannot map dimension {dimension!r} to one axis")
    return matches[0]


def _coordinate(dataset: nc.Dataset, dimension: str, indices: np.ndarray) -> tuple[np.ndarray, str]:
    axis = _axis_for_dimension(dimension)
    candidates = [dimension] + [
        name for name in dataset.variables if name.lower() in _AXIS_ALIASES[axis]
    ]
    found_coordinate = False
    for name in dict.fromkeys(candidates):
        if name not in dataset.variables:
            continue
        variable = dataset.variables[name]
        if variable.dimensions != (dimension,):
            continue
        found_coordinate = True
        values = np.ma.asarray(variable[indices])
        if values.count() == values.size and np.isfinite(values).all():
            return np.asarray(values, dtype=np.float64), "coordinate"
        raise ValueError(
            f"coordinate variable {name!r} for dimension {dimension!r} "
            "contains masked, NaN, or Inf values"
        )
    if found_coordinate:
        raise AssertionError("unreachable coordinate validation state")
    raise ValueError(
        f"dimension {dimension!r} has no one-dimensional physical coordinate variable"
    )


def _coordinate_unit_metadata(
    dataset: nc.Dataset,
    dimension: str,
    *,
    coordinate_source: str,
) -> dict[str, object]:
    """Record a coordinate's units attribute without inventing missing units."""

    if coordinate_source == "explicit_index_fallback":
        return {
            "units_attribute_present": False,
            "units_attribute_value": None,
            "effective_units": "index_coordinate_dimensionless",
        }
    axis = _axis_for_dimension(dimension)
    candidates = [dimension] + [
        name for name in dataset.variables if name.lower() in _AXIS_ALIASES[axis]
    ]
    for name in dict.fromkeys(candidates):
        if name not in dataset.variables:
            continue
        variable = dataset.variables[name]
        if variable.dimensions != (dimension,):
            continue
        present = "units" in variable.ncattrs()
        value = str(variable.getncattr("units")) if present else None
        return {
            "units_attribute_present": bool(present),
            "units_attribute_value": value,
            "effective_units": value if present else "attribute_absent",
        }
    raise ValueError(f"cannot resolve coordinate variable for dimension {dimension!r}")


def _components(dataset: nc.Dataset) -> tuple[str, str, str]:
    for names in _COMPONENT_CANDIDATES:
        if all(name in dataset.variables for name in names):
            return names
    raise ValueError(
        "could not find 3D velocity components; supported triples are "
        f"{_COMPONENT_CANDIDATES}"
    )


def _validate_coordinate(values: np.ndarray, axis: str, *, require_uniform: bool) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{axis} coordinate must contain at least two finite values")
    differences = np.diff(values)
    if np.any(differences <= 0):
        raise ValueError(f"{axis} coordinate must be strictly increasing")
    if require_uniform and not np.allclose(
        differences, differences[0], rtol=1e-4, atol=1e-8
    ):
        raise ValueError(f"non-uniform {axis} coordinate is unsupported")


def inspect_netcdf_3d(
    path: str | Path,
    *,
    index_coordinate_axes: tuple[str, ...] | list[str] = (),
) -> dict[str, object]:
    """Inspect dimensions and velocity component names without loading the field."""

    path = Path(path)
    fallback_axes = tuple(str(axis).lower() for axis in index_coordinate_axes)
    if len(set(fallback_axes)) != len(fallback_axes) or any(
        axis not in "xyzt" for axis in fallback_axes
    ):
        raise ValueError("index_coordinate_axes must be unique axes from x,y,z,t")
    with nc.Dataset(path) as dataset:
        dims = {axis: _axis_dimension(dataset, axis) for axis in "xyzt"}
        shape = {axis: len(dataset.dimensions[dims[axis]]) for axis in "xyzt"}
        coordinates: dict[str, np.ndarray] = {}
        coordinate_sources: dict[str, str] = {}
        coordinate_units: dict[str, dict[str, object]] = {}
        for axis in "xyzt":
            indices = np.arange(shape[axis])
            try:
                coordinates[axis], coordinate_sources[axis] = _coordinate(
                    dataset, dims[axis], indices
                )
            except ValueError:
                if axis not in fallback_axes:
                    raise
                coordinates[axis] = indices.astype(np.float64)
                coordinate_sources[axis] = "explicit_index_fallback"
            _validate_coordinate(
                coordinates[axis],
                "time" if axis == "t" else axis,
                require_uniform=True,
            )
            coordinate_units[axis] = _coordinate_unit_metadata(
                dataset,
                dims[axis],
                coordinate_source=coordinate_sources[axis],
            )
        time = coordinates["t"]
        time_source = coordinate_sources["t"]
        components = _components(dataset)
        component_dimensions = {
            name: list(dataset.variables[name].dimensions) for name in components
        }
    return {
        "path": str(path.resolve()),
        "dimensions": dims,
        "shape": shape,
        "components": list(components),
        "component_dimensions": component_dimensions,
        "time_min": float(time[0]),
        "time_max": float(time[-1]),
        "time_source": time_source,
        "coordinate_sources": coordinate_sources,
        "coordinate_units": coordinate_units,
    }


def load_netcdf_window_3d(
    path: str | Path,
    start_index: int,
    frame_count: int,
    max_spatial_dim: int = 96,
    *,
    index_coordinate_axes: tuple[str, ...] | list[str] = (),
) -> FlowWindow3D:
    """Read and validate a strided temporal window from a regular 3D field.

    Physical coordinates are required by default.  A dataset-specific caller
    may explicitly name axes whose unusable coordinate variables are replaced
    by integer array indices.  This exception is recorded in the returned
    metadata and is never selected implicitly.
    """

    path = Path(path)
    max_spatial_dim = int(max_spatial_dim)
    if max_spatial_dim < 2:
        raise ValueError("max_spatial_dim must be at least 2")
    fallback_axes = tuple(str(axis).lower() for axis in index_coordinate_axes)
    if len(set(fallback_axes)) != len(fallback_axes) or any(
        axis not in "xyzt" for axis in fallback_axes
    ):
        raise ValueError("index_coordinate_axes must be unique axes from x,y,z,t")
    with nc.Dataset(path) as dataset:
        dims = {axis: _axis_dimension(dataset, axis) for axis in "xyzt"}
        sizes = {axis: len(dataset.dimensions[dims[axis]]) for axis in "xyzt"}
        start = int(start_index)
        stop = start + int(frame_count)
        if start < 0 or stop > sizes["t"] or start >= stop:
            raise ValueError(
                f"requested time window [{start},{stop}) outside [0,{sizes['t']})"
            )
        strides = {
            axis: max(1, int(np.ceil(sizes[axis] / max_spatial_dim)))
            for axis in "xyz"
        }
        indices: dict[str, slice] = {
            "t": slice(start, stop),
            **{
                axis: slice(0, sizes[axis], strides[axis])
                for axis in "xyz"
            },
        }
        canonical_dimensions = [dims[axis] for axis in "tzyx"]
        components = _components(dataset)
        arrays: list[np.ndarray] = []
        for component in components:
            variable = dataset.variables[component]
            source_dimensions = list(variable.dimensions)
            if any(name not in source_dimensions for name in canonical_dimensions):
                raise ValueError(
                    f"{component} dimensions {source_dimensions} do not contain "
                    f"{canonical_dimensions}"
                )
            if len(source_dimensions) != len(canonical_dimensions):
                raise ValueError(
                    f"{component} has unsupported extra dimensions {source_dimensions}; "
                    "an explicit member-selection policy is required"
                )
            source_slices = tuple(
                indices[_axis_for_dimension(name)]
                if name in canonical_dimensions
                else 0
                for name in source_dimensions
            )
            raw = np.ma.asarray(variable[source_slices])
            if np.ma.isMaskedArray(raw) and np.ma.getmaskarray(raw).any():
                raise ValueError(f"{component} contains masked values in requested window")
            raw_array = np.asarray(raw, dtype=np.float32)
            retained_dimensions = [
                name
                for name, selection in zip(source_dimensions, source_slices)
                if not isinstance(selection, (int, np.integer))
            ]
            order = [retained_dimensions.index(name) for name in canonical_dimensions]
            arrays.append(np.transpose(raw_array, order))
        velocity = np.ascontiguousarray(np.stack(arrays, axis=-1))
        if not np.isfinite(velocity).all():
            raise ValueError("velocity window contains NaN or Inf")

        coordinates: dict[str, np.ndarray] = {}
        coordinate_sources: dict[str, str] = {}
        for axis in "xyz":
            selected = np.arange(sizes[axis])[indices[axis]]
            try:
                coordinates[axis], coordinate_sources[axis] = _coordinate(
                    dataset, dims[axis], selected
                )
            except ValueError:
                if axis not in fallback_axes:
                    raise
                coordinates[axis] = selected.astype(np.float64)
                coordinate_sources[axis] = "explicit_index_fallback"
            _validate_coordinate(coordinates[axis], axis, require_uniform=True)
        try:
            all_time, coordinate_sources["t"] = _coordinate(
                dataset, dims["t"], np.arange(sizes["t"])
            )
        except ValueError:
            if "t" not in fallback_axes:
                raise
            all_time = np.arange(sizes["t"], dtype=np.float64)
            coordinate_sources["t"] = "explicit_index_fallback"
        if len(all_time) > 1:
            _validate_coordinate(all_time, "time", require_uniform=True)
        elif not np.isfinite(all_time).all():
            raise ValueError("time coordinate is not finite")
        selected_time = all_time[start:stop]
        if len(selected_time) > 1:
            _validate_coordinate(selected_time, "time", require_uniform=True)
        elif not np.isfinite(selected_time).all():
            raise ValueError("time coordinate is not finite")

    return FlowWindow3D(
        velocity=velocity,
        coordinates_xyz=(coordinates["x"], coordinates["y"], coordinates["z"]),
        time=np.asarray(selected_time, dtype=np.float64),
        source_path=str(path.resolve()),
        source_start_index=start,
        spatial_strides=strides,
        components=components,
        coordinate_sources=coordinate_sources,
    )

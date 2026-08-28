from pathlib import Path
import tempfile

import netCDF4 as nc
import numpy as np

from pathline_template_matching.netcdf_io import (
    inspect_netcdf_3d,
    load_netcdf_window_3d,
)


def _make_field(
    path: Path,
    *,
    masked: bool = False,
    invalid_coordinate: bool = False,
    omit_x_coordinate: bool = False,
    extra_dimension: bool = False,
    nonuniform_time: bool = False,
) -> np.ndarray:
    with nc.Dataset(path, "w") as dataset:
        for name, size in (("tdim", 6), ("xdim", 8), ("ydim", 6), ("zdim", 4)):
            dataset.createDimension(name, size)
        if extra_dimension:
            dataset.createDimension("member", 2)
        for name, dimension, values in (
            ("x", "xdim", np.linspace(-2, 2, 8)),
            ("y", "ydim", np.linspace(-1, 1, 6)),
            ("z", "zdim", np.linspace(0, 3, 4)),
            ("time", "tdim", np.linspace(0, 0.5, 6)),
        ):
            if omit_x_coordinate and name == "x":
                continue
            if nonuniform_time and name == "time":
                values = np.asarray(values).copy()
                values[-1] += 0.03
            dataset.createVariable(name, "f4", (dimension,))[:] = values
        if invalid_coordinate:
            dataset.variables["x"][2] = np.nan
        shape = (6, 8, 4, 6)
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        for name, shift in (("u", 0), ("v", 10000), ("w", 20000)):
            dimensions = (
                ("member", "tdim", "xdim", "zdim", "ydim")
                if extra_dimension
                else ("tdim", "xdim", "zdim", "ydim")
            )
            variable = dataset.createVariable(name, "f4", dimensions, fill_value=-999.0)
            values = base + shift
            variable[:] = np.stack((values, values), axis=0) if extra_dimension else values
        if masked:
            dataset.variables["u"][0, 0, 0, 0] = dataset.variables["u"]._FillValue
    return base


def test_dimension_aware_window_load_and_stride():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "field.nc"
        base = _make_field(path)
        info = inspect_netcdf_3d(path)
        assert info["shape"] == {"x": 8, "y": 6, "z": 4, "t": 6}
        window = load_netcdf_window_3d(path, 1, 2, max_spatial_dim=4)
        assert window.velocity.shape == (2, 4, 3, 4, 3)
        assert window.velocity[0, 2, 1, 2, 1] == base[1, 4, 2, 2] + 10000
        assert window.spatial_strides == {"x": 2, "y": 2, "z": 1}
        assert np.isfinite(window.velocity).all()


def test_masked_velocity_fails_closed():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "masked.nc"
        _make_field(path, masked=True)
        try:
            load_netcdf_window_3d(path, 0, 2, max_spatial_dim=8)
        except ValueError as error:
            assert "masked values" in str(error)
        else:
            raise AssertionError("masked velocity was accepted")


def test_invalid_or_missing_coordinate_fails_closed():
    for options in ({"invalid_coordinate": True}, {"omit_x_coordinate": True}):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_coordinate.nc"
            _make_field(path, **options)
            try:
                load_netcdf_window_3d(path, 0, 2, max_spatial_dim=8)
            except ValueError as error:
                assert "coordinate" in str(error)
            else:
                raise AssertionError("an invalid physical coordinate was accepted")


def test_extra_velocity_dimension_requires_explicit_policy():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ensemble.nc"
        _make_field(path, extra_dimension=True)
        try:
            load_netcdf_window_3d(path, 0, 2, max_spatial_dim=8)
        except ValueError as error:
            assert "explicit member-selection policy" in str(error)
        else:
            raise AssertionError("an extra velocity dimension was selected silently")


def test_coordinate_index_fallback_must_be_explicit_and_is_audited():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad_x_coordinate.nc"
        _make_field(path, invalid_coordinate=True)
        window = load_netcdf_window_3d(
            path,
            1,
            2,
            max_spatial_dim=4,
            index_coordinate_axes=("x",),
        )
        assert np.array_equal(window.coordinates_xyz[0], np.asarray([0, 2, 4, 6]))
        assert window.coordinate_sources == {
            "x": "explicit_index_fallback",
            "y": "coordinate",
            "z": "coordinate",
            "t": "coordinate",
        }
        assert window.metadata()["coordinate_sources"]["x"] == "explicit_index_fallback"


def test_full_time_axis_must_be_uniform_even_outside_selected_window():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nonuniform_late_time.nc"
        _make_field(path, nonuniform_time=True)
        for operation in (
            lambda: inspect_netcdf_3d(path),
            lambda: load_netcdf_window_3d(path, 0, 2, max_spatial_dim=8),
        ):
            try:
                operation()
            except ValueError as error:
                assert "non-uniform time" in str(error)
            else:
                raise AssertionError("non-uniform full time axis was accepted")

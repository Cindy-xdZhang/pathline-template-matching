from pathlib import Path
import tempfile

import numpy as np

from pathline_template_matching.netcdf_io import FlowWindow3D
from pathline_template_matching.portable_flow import (
    canonical_array_sha256,
    load_portable_flow_window,
    write_portable_flow_window,
)


def _window() -> FlowWindow3D:
    velocity = np.arange(3 * 4 * 5 * 6 * 3, dtype=np.float32).reshape(3, 4, 5, 6, 3)
    return FlowWindow3D(
        velocity=velocity,
        coordinates_xyz=(
            np.linspace(-1.0, 1.5, 6),
            np.linspace(2.0, 4.0, 5),
            np.linspace(-3.0, 0.0, 4),
        ),
        time=np.asarray([10.0, 10.25, 10.5]),
        source_path="source.nc",
        source_start_index=7,
        spatial_strides={"x": 2, "y": 3, "z": 4},
        components=("u", "v", "w"),
        coordinate_sources={"x": "coordinate", "y": "coordinate", "z": "coordinate", "t": "coordinate"},
    )


def _coordinate_units():
    return {
        axis: {
            "units_attribute_present": False,
            "units_attribute_value": None,
            "effective_units": "attribute_absent",
        }
        for axis in "xyzt"
    }


def test_portable_flow_round_trip_and_hash_contract():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "window.npz"
        row = write_portable_flow_window(
            output,
            dataset="flow_a",
            physical_family="family_a",
            split="train",
            experiment="mainExp_TemplateMatching_2.1",
            config_sha256="1" * 64,
            dataset_registry_sha256="3" * 64,
            builder_git_commit="4" * 40,
            coordinate_units=_coordinate_units(),
            source_file=Path(directory) / "source.nc",
            source_file_sha256="2" * 64,
            source_file_size=1234,
            window=_window(),
        )
        loaded = load_portable_flow_window(
            output,
            expected_dataset="flow_a",
            expected_experiment="mainExp_TemplateMatching_2.1",
            expected_config_sha256="1" * 64,
            expected_dataset_registry_sha256="3" * 64,
            expected_builder_git_commit="4" * 40,
            expected_source_start_index=7,
            expected_file_sha256=row["file_sha256"],
        )
        np.testing.assert_array_equal(loaded.window.velocity, _window().velocity)
        assert loaded.window.coordinate_sources["t"] == "coordinate"
        assert loaded.metadata["array_sha256"]["velocity"] == canonical_array_sha256(
            _window().velocity
        )
        assert loaded.metadata["coordinate_audit"]["x"]["axis_order"] == 3
        assert loaded.metadata["coordinate_audit"]["t"]["units"]["effective_units"] == "attribute_absent"


def test_portable_flow_detects_array_replacement_without_pickle():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "window.npz"
        write_portable_flow_window(
            output,
            dataset="flow_a",
            physical_family="family_a",
            split="train",
            experiment="mainExp_TemplateMatching_2.1",
            config_sha256="1" * 64,
            dataset_registry_sha256="3" * 64,
            builder_git_commit="4" * 40,
            coordinate_units=_coordinate_units(),
            source_file=Path(directory) / "source.nc",
            source_file_sha256="2" * 64,
            source_file_size=1234,
            window=_window(),
        )
        with np.load(output, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        arrays["velocity"] = arrays["velocity"].copy()
        arrays["velocity"][0, 0, 0, 0, 0] += 1
        np.savez_compressed(output, **arrays)
        try:
            load_portable_flow_window(output)
        except ValueError as error:
            assert "array SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("modified portable array was accepted")


def test_portable_flow_rejects_nonfinite_velocity_before_write():
    with tempfile.TemporaryDirectory() as directory:
        window = _window()
        window.velocity[0, 0, 0, 0, 0] = np.nan
        try:
            write_portable_flow_window(
                Path(directory) / "bad.npz",
                dataset="flow_a",
                physical_family="family_a",
                split="train",
                experiment="mainExp_TemplateMatching_2.1",
                config_sha256="1" * 64,
                dataset_registry_sha256="3" * 64,
                builder_git_commit="4" * 40,
                coordinate_units=_coordinate_units(),
                source_file=Path(directory) / "source.nc",
                source_file_sha256="2" * 64,
                source_file_size=1234,
                window=window,
            )
        except ValueError as error:
            assert "portable velocity" in str(error)
        else:
            raise AssertionError("non-finite portable velocity was written")


def test_portable_flow_rejects_stored_dtype_drift_before_casting():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "window.npz"
        write_portable_flow_window(
            output,
            dataset="flow_a",
            physical_family="family_a",
            split="train",
            experiment="mainExp_TemplateMatching_2.1",
            config_sha256="1" * 64,
            dataset_registry_sha256="3" * 64,
            builder_git_commit="4" * 40,
            coordinate_units=_coordinate_units(),
            source_file=Path(directory) / "source.nc",
            source_file_sha256="2" * 64,
            source_file_size=1234,
            window=_window(),
        )
        with np.load(output, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        arrays["velocity"] = arrays["velocity"].astype(np.float64)
        np.savez_compressed(output, **arrays)
        try:
            load_portable_flow_window(output)
        except ValueError as error:
            assert "dtype mismatch" in str(error)
            assert "velocity" in str(error)
        else:
            raise AssertionError("float64 portable velocity was silently cast to float32")

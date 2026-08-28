"""Portable, hash-audited regular-grid flow windows for Ibex staging."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import string
from typing import Any

import numpy as np

from .netcdf_io import FlowWindow3D


PORTABLE_FLOW_SCHEMA = "pathline_template_matching.portable_flow_window.v1"


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(int(chunk_bytes)):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and canonical little-endian C-order array bytes."""

    values = np.asarray(array)
    if values.dtype.hasobject:
        raise ValueError("object arrays do not have a portable numeric digest")
    little_dtype = values.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(values.astype(little_dtype, copy=False))
    header = json.dumps(
        {"dtype": little_dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PortableFlowWindow:
    """Validated portable window, metadata, and the compressed-file digest."""

    path: Path
    file_sha256: str
    window: FlowWindow3D
    metadata: dict[str, Any]


def _window_arrays(window: FlowWindow3D) -> dict[str, np.ndarray]:
    x, y, z = window.coordinates_xyz
    return {
        "velocity": np.ascontiguousarray(window.velocity, dtype=np.float32),
        "x": np.ascontiguousarray(x, dtype=np.float64),
        "y": np.ascontiguousarray(y, dtype=np.float64),
        "z": np.ascontiguousarray(z, dtype=np.float64),
        "time": np.ascontiguousarray(window.time, dtype=np.float64),
    }


def _validate_arrays_and_contract(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any] | None = None
) -> None:
    velocity = np.asarray(arrays["velocity"])
    if (
        velocity.ndim != 5
        or velocity.shape[-1] != 3
        or velocity.shape[0] < 2
        or min(velocity.shape[1:4]) < 2
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("portable velocity must be finite [T>=2,Z>=2,Y>=2,X>=2,3]")
    expected_lengths = {
        "x": velocity.shape[3],
        "y": velocity.shape[2],
        "z": velocity.shape[1],
        "time": velocity.shape[0],
    }
    for name, expected in expected_lengths.items():
        axis = np.asarray(arrays[name], dtype=np.float64)
        if (
            axis.shape != (expected,)
            or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)
            or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-4, atol=1e-8)
        ):
            raise ValueError(
                f"portable {name} coordinate must be finite, uniform, increasing, "
                f"and have length {expected}"
            )
    if metadata is None:
        return
    required = {
        "schema",
        "experiment",
        "config_sha256",
        "dataset_registry_sha256",
        "builder_git_commit",
        "dataset",
        "physical_family",
        "split",
        "source_file",
        "source_file_size",
        "source_file_sha256",
        "source_start_index",
        "frame_count",
        "spatial_strides",
        "components",
        "coordinate_sources",
        "coordinate_audit",
        "loaded_shape_TZYXC",
        "array_sha256",
        "combined_array_sha256",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"portable metadata misses fields: {sorted(missing)}")
    if metadata["split"] not in {"train", "test"}:
        raise ValueError("portable metadata split is invalid")
    if int(metadata["source_file_size"]) < 1 or int(metadata["source_start_index"]) < 0:
        raise ValueError("portable source size/start index is invalid")
    if int(metadata["frame_count"]) != velocity.shape[0] or list(
        metadata["loaded_shape_TZYXC"]
    ) != list(velocity.shape):
        raise ValueError("portable metadata shape/frame count disagrees with arrays")
    strides = metadata["spatial_strides"]
    if set(strides) != set("xyz") or any(int(value) < 1 for value in strides.values()):
        raise ValueError("portable spatial_strides must contain positive x/y/z integers")
    if len(metadata["components"]) != 3:
        raise ValueError("portable metadata must name three velocity components")
    if set(metadata["coordinate_sources"]) != set("xyzt"):
        raise ValueError("portable coordinate_sources must contain x/y/z/t")
    audit = metadata["coordinate_audit"]
    if not isinstance(audit, dict) or set(audit) != set("xyzt"):
        raise ValueError("portable coordinate_audit must contain x/y/z/t")
    array_by_axis = {"x": "x", "y": "y", "z": "z", "t": "time"}
    dimension_index = {"t": 0, "z": 1, "y": 2, "x": 3}
    hashes = metadata.get("array_sha256", {})
    for axis in "xyzt":
        values = np.asarray(arrays[array_by_axis[axis]])
        row = audit[axis]
        expected = {
            "axis_order": dimension_index[axis],
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "minimum": float(values[0]),
            "maximum": float(values[-1]),
            "spacing": float(values[1] - values[0]),
            "sha256": hashes.get(array_by_axis[axis]),
        }
        drift = {
            key: (row.get(key), value)
            for key, value in expected.items()
            if row.get(key) != value
        }
        units = row.get("units")
        if (
            drift
            or not isinstance(units, dict)
            or not isinstance(units.get("units_attribute_present"), bool)
            or not isinstance(units.get("effective_units"), str)
        ):
            raise ValueError(f"portable coordinate audit drift for {axis}: {drift}")


def write_portable_flow_window(
    path: str | Path,
    *,
    dataset: str,
    physical_family: str,
    split: str,
    experiment: str,
    config_sha256: str,
    dataset_registry_sha256: str,
    builder_git_commit: str,
    coordinate_units: dict[str, dict[str, object]],
    source_file: str | Path,
    source_file_sha256: str,
    source_file_size: int,
    window: FlowWindow3D,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write one portable window and return its manifest row."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"portable window already exists: {output}")
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if not dataset or not physical_family or not experiment:
        raise ValueError("dataset, physical_family, and experiment are required")
    hex_digits = set(string.hexdigits)
    digest_fields = (
        str(config_sha256),
        str(dataset_registry_sha256),
        str(builder_git_commit),
        str(source_file_sha256),
    )
    if any(
        len(value) != 40 if index == 2 else len(value) != 64
        for index, value in enumerate(digest_fields)
    ) or any(
        character not in hex_digits
        for value in digest_fields
        for character in value
    ):
        raise ValueError(
            "config/registry/source SHA-256 and 40-character Git commit must be hexadecimal"
        )
    if int(source_file_size) < 1:
        raise ValueError("source_file_size must be positive")
    arrays = _window_arrays(window)
    _validate_arrays_and_contract(arrays)
    array_hashes = {name: canonical_array_sha256(value) for name, value in arrays.items()}
    if set(coordinate_units) != set("xyzt"):
        raise ValueError("coordinate_units must contain x/y/z/t")
    coordinate_audit: dict[str, dict[str, object]] = {}
    array_by_axis = {"x": "x", "y": "y", "z": "z", "t": "time"}
    dimension_index = {"t": 0, "z": 1, "y": 2, "x": 3}
    for axis in "xyzt":
        unit = dict(coordinate_units[axis])
        if set(unit) != {
            "units_attribute_present",
            "units_attribute_value",
            "effective_units",
        }:
            raise ValueError(f"coordinate_units[{axis!r}] has an invalid contract")
        if not isinstance(unit["units_attribute_present"], bool) or not isinstance(
            unit["effective_units"], str
        ):
            raise ValueError(f"coordinate_units[{axis!r}] has invalid values")
        values = arrays[array_by_axis[axis]]
        coordinate_audit[axis] = {
            "axis_order": dimension_index[axis],
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "units": unit,
            "minimum": float(values[0]),
            "maximum": float(values[-1]),
            "spacing": float(values[1] - values[0]),
            "sha256": array_hashes[array_by_axis[axis]],
        }
    metadata: dict[str, Any] = {
        "schema": PORTABLE_FLOW_SCHEMA,
        "experiment": str(experiment),
        "config_sha256": str(config_sha256),
        "dataset_registry_sha256": str(dataset_registry_sha256),
        "builder_git_commit": str(builder_git_commit),
        "dataset": str(dataset),
        "physical_family": str(physical_family),
        "split": split,
        "source_file": str(Path(source_file).resolve()),
        "source_file_size": int(source_file_size),
        "source_file_sha256": str(source_file_sha256),
        "source_start_index": int(window.source_start_index),
        "frame_count": int(window.velocity.shape[0]),
        "spatial_strides": {key: int(value) for key, value in window.spatial_strides.items()},
        "components": list(window.components),
        "coordinate_sources": dict(window.coordinate_sources),
        "coordinate_audit": coordinate_audit,
        "loaded_shape_TZYXC": list(window.velocity.shape),
        "array_sha256": array_hashes,
    }
    if extra_metadata:
        overlap = set(metadata).intersection(extra_metadata)
        if overlap:
            raise ValueError(f"extra_metadata replaces frozen fields: {sorted(overlap)}")
        metadata.update(extra_metadata)
    metadata["combined_array_sha256"] = canonical_json_sha256(array_hashes)
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    try:
        with temporary.open("xb") as destination:
            np.savez_compressed(
                destination,
                **arrays,
                metadata_json=np.asarray(metadata_json),
            )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output.resolve()),
        "file_size": int(output.stat().st_size),
        "file_sha256": sha256_file(output),
        **metadata,
    }


def load_portable_flow_window(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_experiment: str | None = None,
    expected_config_sha256: str | None = None,
    expected_dataset_registry_sha256: str | None = None,
    expected_builder_git_commit: str | None = None,
    expected_source_start_index: int | None = None,
    expected_file_sha256: str | None = None,
) -> PortableFlowWindow:
    """Load a portable window and fail if any metadata or array hash drifted."""

    source = Path(path)
    file_digest = sha256_file(source)
    if expected_file_sha256 is not None and file_digest != str(expected_file_sha256):
        raise ValueError(f"portable file SHA-256 mismatch: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {"velocity", "x", "y", "z", "time", "metadata_json"}
        if set(archive.files) != required:
            raise ValueError(
                f"portable window keys disagree: expected {sorted(required)}, "
                f"found {sorted(archive.files)}"
            )
        # Preserve the stored dtypes until the contract has been checked.  A
        # cast here would make (for example) a substituted float64 velocity
        # array look like the canonical float32 payload before its hash and
        # metadata are audited.
        arrays = {
            "velocity": np.asarray(archive["velocity"]),
            "x": np.asarray(archive["x"]),
            "y": np.asarray(archive["y"]),
            "z": np.asarray(archive["z"]),
            "time": np.asarray(archive["time"]),
        }
        expected_dtypes = {
            "velocity": np.dtype("<f4"),
            "x": np.dtype("<f8"),
            "y": np.dtype("<f8"),
            "z": np.dtype("<f8"),
            "time": np.dtype("<f8"),
        }
        dtype_drift = {
            name: (str(values.dtype), str(expected_dtypes[name]))
            for name, values in arrays.items()
            if values.dtype != expected_dtypes[name]
        }
        if dtype_drift:
            raise ValueError(f"portable window dtype mismatch: {dtype_drift}")
        scalar = np.asarray(archive["metadata_json"])
        if scalar.ndim != 0:
            raise ValueError("portable metadata_json must be a scalar")
        metadata = json.loads(str(scalar.item()))
    if metadata.get("schema") != PORTABLE_FLOW_SCHEMA:
        raise ValueError("portable flow schema is unsupported")
    _validate_arrays_and_contract(arrays, metadata)
    expected_fields = {
        "dataset": expected_dataset,
        "experiment": expected_experiment,
        "config_sha256": expected_config_sha256,
        "dataset_registry_sha256": expected_dataset_registry_sha256,
        "builder_git_commit": expected_builder_git_commit,
        "source_start_index": expected_source_start_index,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_fields.items()
        if value is not None and metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"portable window metadata mismatch: {mismatches}")
    actual_hashes = {name: canonical_array_sha256(value) for name, value in arrays.items()}
    if actual_hashes != metadata.get("array_sha256"):
        raise ValueError("portable window array SHA-256 mismatch")
    if canonical_json_sha256(actual_hashes) != metadata.get("combined_array_sha256"):
        raise ValueError("portable combined array SHA-256 mismatch")
    time = arrays["time"]
    window = FlowWindow3D(
        velocity=arrays["velocity"],
        coordinates_xyz=(arrays["x"], arrays["y"], arrays["z"]),
        time=time,
        source_path=str(metadata["source_file"]),
        source_start_index=int(metadata["source_start_index"]),
        spatial_strides={
            key: int(value) for key, value in metadata["spatial_strides"].items()
        },
        components=tuple(str(value) for value in metadata["components"]),
        coordinate_sources={
            key: str(value) for key, value in metadata["coordinate_sources"].items()
        },
    )
    return PortableFlowWindow(
        path=source,
        file_sha256=file_digest,
        window=window,
        metadata=metadata,
    )

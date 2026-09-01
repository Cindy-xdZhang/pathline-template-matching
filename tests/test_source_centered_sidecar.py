from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from pathline_template_matching.netcdf_io import FlowWindow3D
from pathline_template_matching.portable_flow import (
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
    write_portable_flow_window,
)
from pathline_template_matching.source_centered_sidecar import (
    ASSIGNED_ROW_COUNT,
    INPUT_MANIFEST_SCHEMA,
    PARENT_ALLOWED_MEMBER_NAMES,
    PARENT_FORBIDDEN_MEMBER_NAMES,
    POPULATION_MANIFEST_SCHEMA,
    REQUIRED_SOURCE_PATHS,
    SIDECAR_ARCHIVE_MEMBER_NAMES,
    SIDECAR_ARRAY_NAMES,
    CleanSourceIdentity,
    PreparationContract,
    authenticate_source_centered_population_manifest,
    build_one_source_centered_sidecar_and_completion,
    build_source_centered_input_manifest,
    load_assigned_row_parent_projection,
    load_source_centered_sidecar,
    write_source_centered_population_manifest,
)


def _expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _assignment() -> np.ndarray:
    legacy = np.tile(np.arange(1000, dtype=np.int32), 64)
    expanded = np.tile(np.arange(1000, 2000, dtype=np.int32), 64)
    return np.ascontiguousarray(np.concatenate((legacy, expanded)), dtype=np.int32)


def _parent_arrays() -> dict[str, np.ndarray]:
    assignment = _assignment()
    assigned = np.asarray([0, 1, 999, 64_000, 64_001, 127_999], dtype=np.int64)
    return {
        "seeds_xyz": np.zeros((ASSIGNED_ROW_COUNT, 3), dtype=np.float64),
        "scale_assignment": assignment,
        "valid_assigned_row_index": assigned,
        "valid_center_seed_index": assigned % 64_000,
        "valid_scale_block_index": (assigned // 64_000).astype(np.int8),
        "valid_scale_id": assignment[assigned],
    }


def _write_poisoned_parent(path: Path) -> tuple[str, dict[str, str]]:
    arrays = _parent_arrays()
    poison = {
        "valid_labels": np.asarray([object()], dtype=object),
        "reference_labels_all": np.asarray([object()], dtype=object),
        "ivd_values_all": np.asarray([object()], dtype=object),
        "ivd_volume": np.asarray([object()], dtype=object),
        "valid_mask": np.asarray([object()], dtype=object),
        "metadata_json": np.asarray({"poison": True}, dtype=object),
        "fmt_features": np.asarray([object()], dtype=object),
        "raw_features": np.asarray([object()], dtype=object),
    }
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays, **poison)
    hashes = {
        name: canonical_array_sha256(arrays[name])
        for name in PARENT_ALLOWED_MEMBER_NAMES
    }
    return sha256_file(path), hashes


def _identity(config_sha256: str) -> CleanSourceIdentity:
    items = tuple(
        (path, config_sha256 if index == 0 else f"{index:064x}")
        for index, path in enumerate(REQUIRED_SOURCE_PATHS)
    )
    return CleanSourceIdentity("a" * 40, True, items)


def _affine_portable(path: Path) -> dict[str, object]:
    x = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
    z = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    first = np.stack(
        (xx - 0.5 * yy, 0.25 * xx + yy - zz, 0.5 * yy + zz), axis=-1
    )
    velocity = np.ascontiguousarray(np.stack((first, first + 10.0)), dtype=np.float32)
    window = FlowWindow3D(
        velocity=velocity,
        coordinates_xyz=(x, y, z),
        time=np.asarray([0.0, 0.5], dtype=np.float64),
        source_path="synthetic.nc",
        source_start_index=0,
        spatial_strides={"x": 1, "y": 1, "z": 1},
        components=("u", "v", "w"),
        coordinate_sources={"x": "synthetic", "y": "synthetic", "z": "synthetic", "t": "synthetic"},
    )
    unit = {
        "units_attribute_present": False,
        "units_attribute_value": None,
        "effective_units": "dimensionless",
    }
    row = write_portable_flow_window(
        path,
        dataset="cylinder3d",
        physical_family="half_cylinder",
        split="train",
        experiment="mainExp_TemplateMatching_3.1",
        config_sha256="b" * 64,
        dataset_registry_sha256="c" * 64,
        builder_git_commit="d" * 40,
        coordinate_units={axis: dict(unit) for axis in "xyzt"},
        source_file="synthetic.nc",
        source_file_sha256="e" * 64,
        source_file_size=1,
        window=window,
    )
    return {
        "path": str(path.resolve()),
        "size_bytes": int(row["file_size"]),
        "file_sha256": str(row["file_sha256"]),
        "schema": str(row["schema"]),
        "builder_git_commit": str(row["builder_git_commit"]),
        "config_sha256": str(row["config_sha256"]),
        "dataset_manifest_path": str(path.resolve()),
        "dataset_manifest_size_bytes": int(row["file_size"]),
        "dataset_manifest_file_sha256": str(row["file_sha256"]),
        "dataset_manifest_content_sha256": "f" * 64,
        "portable_metadata_sha256": "1" * 64,
        "array_sha256": dict(row["array_sha256"]),
        "combined_array_sha256": str(row["combined_array_sha256"]),
    }


def test_parent_loader_opens_only_six_members_and_rejects_group_drift():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "parent.npz"
        file_sha, hashes = _write_poisoned_parent(path)
        projection = load_assigned_row_parent_projection(
            path,
            expected_size_bytes=path.stat().st_size,
            expected_file_sha256=file_sha,
            expected_array_sha256=hashes,
        )
        assert projection.opened_member_names == PARENT_ALLOWED_MEMBER_NAMES
        assert PARENT_FORBIDDEN_MEMBER_NAMES.isdisjoint(projection.opened_member_names)
        for name in PARENT_ALLOWED_MEMBER_NAMES:
            assert not np.asarray(getattr(projection, name)).flags.writeable

        bad_hashes = dict(hashes)
        bad_hashes["scale_assignment"] = "0" * 64
        _expect_value_error(
            load_assigned_row_parent_projection,
            path,
            expected_size_bytes=path.stat().st_size,
            expected_file_sha256=file_sha,
            expected_array_sha256=bad_hashes,
        )


def test_end_to_end_freeze_sidecar_completion_population_and_fresh_replay():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parent_path = root / "parent.npz"
        parent_sha, _hashes = _write_poisoned_parent(parent_path)
        portable_path = root / "portable.npz"
        portable = _affine_portable(portable_path)

        parent_input_path = root / "parent_input.json"
        parent_input_path.write_text('{"frozen":true}\n', encoding="utf-8")
        parent_input_sha = sha256_file(parent_input_path)
        early_row = {
            "dataset": "cylinder3d",
            "physical_family": "half_cylinder",
            "split": "train",
            "source_ordinal": 0,
            "source_index": 0,
            "parent_cache": {
                "path": str(parent_path.resolve()),
                "size_bytes": parent_path.stat().st_size,
                "file_sha256": parent_sha,
                "schema": "pathline_template_matching.phase31_cache.v1",
                "builder_git_commit": "d" * 40,
                "config_sha256": "b" * 64,
                "allowed_array_sha256": {},
                "opened_members": [],
            },
            "portable": portable,
        }
        early_path = root / "early_input.json"
        early = {
            "rows": [early_row],
            "rows_content_sha256": canonical_json_sha256([early_row]),
            "parent_input_manifest": {
                "path": str(parent_input_path.resolve()),
                "size_bytes": parent_input_path.stat().st_size,
                "file_sha256": parent_input_sha,
                "rows_content_sha256": "2" * 64,
            },
            "content_sha256": "3" * 64,
        }
        early_path.write_text(json.dumps(early, sort_keys=True), encoding="utf-8")
        early_sha = sha256_file(early_path)

        config_sha = "4" * 64
        contract = PreparationContract(
            verify_config_sha256=config_sha,
            early_input_manifest_sha256=early_sha,
            parent_input_manifest_sha256=parent_input_sha,
            dataset_family_pairs=(("cylinder3d", "half_cylinder"),),
            source_count=1,
        )
        identity = _identity(config_sha)
        frozen_path = root / "source_centered_input.json"
        frozen = build_source_centered_input_manifest(
            frozen_path,
            early_input_manifest_path=early_path,
            identity=identity,
            contract=contract,
        )
        assert frozen["schema"] == INPUT_MANIFEST_SCHEMA
        assert frozen["parent_opened_members"] == PARENT_ALLOWED_MEMBER_NAMES
        assert frozen["forbidden_parent_members_opened"] == ()

        sidecar_root = root / "sidecars"
        completion = build_one_source_centered_sidecar_and_completion(
            sidecar_root,
            dataset="cylinder3d",
            source_ordinal=0,
            input_manifest_path=frozen_path,
            input_manifest_file_sha256=sha256_file(frozen_path),
            identity=identity,
            contract=contract,
        )
        sidecar_path = sidecar_root / str(completion["sidecar_relative_path"])
        loaded = load_source_centered_sidecar(
            sidecar_path,
            expected_file_sha256=str(completion["sidecar_file_sha256"]),
        )
        with np.load(sidecar_path, allow_pickle=False) as archive:
            assert tuple(archive.files) == SIDECAR_ARCHIVE_MEMBER_NAMES
            assert set(SIDECAR_ARRAY_NAMES).isdisjoint(PARENT_FORBIDDEN_MEMBER_NAMES)
        assert loaded.payload.source_centered_seed4.shape == (128_000, 4)
        assert loaded.payload.group_mean_curl_xyz.shape == (20, 3)
        assert loaded.metadata["valid_projection"]["stored_duplicate_feature"] is False
        assert loaded.metadata["valid_projection"]["canonical_sha256"] == completion[
            "valid_projection_sha256"
        ]

        population = write_source_centered_population_manifest(
            sidecar_root,
            input_manifest_path=frozen_path,
            input_manifest_file_sha256=sha256_file(frozen_path),
            identity=identity,
            contract=contract,
        )
        assert population["schema"] == POPULATION_MANIFEST_SCHEMA
        assert population["sidecar_count"] == 1
        assert population["assigned_row_count_total"] == 128_000
        replay = authenticate_source_centered_population_manifest(
            sidecar_root / "SIDECAR_POPULATION.json",
            sidecar_root=sidecar_root,
            expected_file_sha256=sha256_file(
                sidecar_root / "SIDECAR_POPULATION.json"
            ),
            input_manifest_path=frozen_path,
            input_manifest_file_sha256=sha256_file(frozen_path),
            identity=identity,
            contract=contract,
        )
        assert replay["content_sha256"] == population["content_sha256"]

        try:
            write_source_centered_population_manifest(
                sidecar_root,
                input_manifest_path=frozen_path,
                input_manifest_file_sha256=sha256_file(frozen_path),
                identity=identity,
                contract=contract,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("population seal was overwritten")

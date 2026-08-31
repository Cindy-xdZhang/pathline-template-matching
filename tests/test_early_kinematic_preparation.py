from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import operator
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from pathline_template_matching.netcdf_io import FlowWindow3D
from pathline_template_matching.portable_flow import (
    canonical_json_sha256,
    sha256_file,
    write_portable_flow_window,
)
import pathline_template_matching.early_kinematic_preparation as preparation


def _expect_error(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _affine_window(source_index: int) -> FlowWindow3D:
    x = np.linspace(-4.0, 4.0, 9, dtype=np.float64)
    y = np.linspace(-4.0, 4.0, 9, dtype=np.float64)
    z = np.linspace(-4.0, 4.0, 9, dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    first = np.stack(
        (
            0.5 * xx - yy + 0.25 * zz + 0.1,
            1.5 * xx + 0.75 * yy - 0.5 * zz - 0.2,
            -0.25 * xx + yy + 1.25 * zz + 0.3,
        ),
        axis=-1,
    ).astype(np.float32)
    return FlowWindow3D(
        velocity=np.stack((first, first + np.float32(10.0)), axis=0),
        coordinates_xyz=(x, y, z),
        time=np.asarray([float(source_index), float(source_index) + 0.5], dtype=np.float64),
        source_path="synthetic-only.nc",
        source_start_index=source_index,
        spatial_strides={"x": 1, "y": 1, "z": 1},
        components=("u", "v", "w"),
        coordinate_sources={
            "x": "synthetic",
            "y": "synthetic",
            "z": "synthetic",
            "t": "synthetic",
        },
    )


def _parent_arrays() -> dict[str, np.ndarray]:
    return {
        "seeds_xyz": np.zeros((2, 3), dtype=np.float64),
        "valid_assigned_row_index": np.asarray([0, 1], dtype=np.int64),
        "valid_center_seed_index": np.asarray([0, 0], dtype=np.int64),
        "valid_scale_block_index": np.asarray([0, 1], dtype=np.int8),
        "valid_scale_id": np.asarray([0, 1000], dtype=np.int32),
        "center_sample_time": np.zeros((2, 32), dtype=np.float32),
    }


def _write_poisoned_parent(path: Path) -> None:
    arrays = _parent_arrays()
    poison = {
        "valid_labels": np.asarray([object(), object()], dtype=object),
        "reference_labels_all": np.asarray([object()], dtype=object),
        "ivd_values_all": np.asarray([object()], dtype=object),
        "ivd_volume": np.asarray([object()], dtype=object),
        "metadata_json": np.asarray({"poison": object()}, dtype=object),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as destination:
        np.savez_compressed(destination, **arrays, **poison)


def _identity(contract: preparation.PreparationContract) -> preparation.CleanSourceIdentity:
    hashes = []
    for index, relative in enumerate(preparation.REQUIRED_SOURCE_PATHS):
        if index == 0:
            digest = contract.verify_config_sha256
        elif index == 1:
            digest = contract.parent_main_config_sha256
        else:
            digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        hashes.append((relative, digest))
    return preparation.CleanSourceIdentity(
        git_commit="2" * 40,
        worktree_clean=True,
        source_file_sha256_items=tuple(hashes),
    )


def _synthetic_population(root: Path):
    datasets = preparation.DATASET_FAMILY_PAIRS
    parent_commit = "1" * 40
    main_config_sha = "a" * 64
    verify_config_sha = "b" * 64
    registry_sha = "c" * 64
    portable_root = root / "portable"
    marker_rows = []
    portable_paths = set()
    for dataset, family in datasets:
        dataset_root = portable_root / dataset
        dataset_root.mkdir(parents=True)
        source_file = dataset_root / "synthetic_source.bin"
        source_file.write_bytes(f"source:{dataset}".encode("utf-8"))
        windows = []
        for ordinal in range(4):
            source_index = ordinal
            output = dataset_root / f"window_{ordinal:02d}.npz"
            returned = write_portable_flow_window(
                output,
                dataset=dataset,
                physical_family=family,
                split="train",
                experiment="mainExp_TemplateMatching_3.1",
                config_sha256=main_config_sha,
                dataset_registry_sha256=registry_sha,
                builder_git_commit=parent_commit,
                coordinate_units={
                    axis: {
                        "units_attribute_present": False,
                        "units_attribute_value": None,
                        "effective_units": "synthetic",
                    }
                    for axis in "xyzt"
                },
                source_file=source_file,
                source_file_sha256=sha256_file(source_file),
                source_file_size=source_file.stat().st_size,
                window=_affine_window(source_index),
                extra_metadata={
                    "source_total_frames": 5,
                    "source_ordinal": ordinal,
                    "source_kind": "synthetic",
                    "native_shape_xyz": [9, 9, 9],
                },
            )
            row = dict(returned)
            row.pop("path")
            row["relative_path"] = output.name
            windows.append(row)
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"relative_path", "file_size", "file_sha256"}
            }
            marker_rows.append(
                {
                    "dataset": dataset,
                    "split": "train",
                    "source_ordinal": ordinal,
                    "source_start_index": source_index,
                    "relative_path": f"{dataset}/{output.name}",
                    "file_size": int(output.stat().st_size),
                    "file_sha256": sha256_file(output),
                    "manifest_relative_path": f"{dataset}/manifest.json",
                    "manifest_file_sha256": "pending",
                    "portable_metadata_sha256": canonical_json_sha256(metadata),
                }
            )
            portable_paths.add(output.resolve())
        manifest = {
            "schema": preparation.PARENT_DATASET_MANIFEST_SCHEMA,
            "experiment": "mainExp_TemplateMatching_3.1",
            "config_path": "synthetic-main.yaml",
            "config_sha256": main_config_sha,
            "dataset_registry_path": "synthetic-datasets.yaml",
            "dataset_registry_sha256": registry_sha,
            "builder_git_commit": parent_commit,
            "dataset": dataset,
            "physical_family": family,
            "split": "train",
            "source_kind": "synthetic",
            "source_file": str(source_file.resolve()),
            "source_file_size": source_file.stat().st_size,
            "source_file_sha256": sha256_file(source_file),
            "source_total_frames": 5,
            "selected_source_indices": [0, 1, 2, 3],
            "window_count": 4,
            "windows": windows,
        }
        manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
        manifest_path = dataset_root / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        for row in marker_rows[-4:]:
            row["manifest_file_sha256"] = manifest_sha

    marker_dir = root / "portable_pass"
    marker_dir.mkdir()
    marker = {
        "schema": preparation.PARENT_PORTABLE_MARKER_SCHEMA,
        "experiment": "mainExp_TemplateMatching_3.1",
        "status": "passed",
        "access_scope": "train-only",
        "git_commit": parent_commit,
        "worktree_clean": True,
        "config_sha256": main_config_sha,
        "dataset_registry_sha256": registry_sha,
        "portable_root": str(portable_root.resolve()),
        "dataset_count": 8,
        "window_count": 32,
        "synthetic_pass_file_sha256": "d" * 64,
        "train_coverage_pass_file_sha256": None,
        "rows": marker_rows,
        "rows_content_sha256": canonical_json_sha256(marker_rows),
        "marker_write_order": "synthetic_test",
    }
    marker_path = marker_dir / "TRAIN_PORTABLES_PASS.json"
    _write_json(marker_path, marker)

    cache_rows = []
    parent_paths = set()
    for dataset, _family in datasets:
        for ordinal in range(4):
            cache_path = root / "parent" / "train" / dataset / f"cache_{ordinal:02d}.npz"
            _write_poisoned_parent(cache_path)
            evidence_sidecar = cache_path.with_suffix(".summary.json")
            evidence_sidecar.write_text("{}\n", encoding="utf-8")
            cache_rows.append(
                {
                    "dataset": dataset,
                    "source_ordinal": ordinal,
                    "source_index": ordinal,
                    "cache_path": str(cache_path.resolve()),
                    "cache_size_bytes": cache_path.stat().st_size,
                    "cache_file_sha256": sha256_file(cache_path),
                    "sidecar_path": str(evidence_sidecar.resolve()),
                    "sidecar_size_bytes": evidence_sidecar.stat().st_size,
                    "sidecar_file_sha256": sha256_file(evidence_sidecar),
                }
            )
            parent_paths.add(cache_path.resolve())
    rows_sha = canonical_json_sha256(cache_rows)
    marker_sha = sha256_file(marker_path)
    parent_manifest = {
        "schema": preparation.PARENT_INPUT_SCHEMA,
        "experiment": "Verify_LongArcHorizon_1.1",
        "parent_experiment": "mainExp_TemplateMatching_3.1",
        "git_commit": parent_commit,
        "main_config_sha256": main_config_sha,
        "verify_config_sha256": "e" * 64,
        "synthetic_pass_file_sha256": "d" * 64,
        "train_portable_population_pass": {
            "path": str(marker_path.resolve()),
            "file_size": marker_path.stat().st_size,
            "file_sha256": marker_sha,
            "access_scope": "train-only",
            "rows_content_sha256": marker["rows_content_sha256"],
        },
        "input_scope": "exactly_32_train_cache_shards_and_sidecars",
        "test_dataset_access": False,
        "row_count": 32,
        "rows": cache_rows,
        "rows_content_sha256": rows_sha,
    }
    parent_manifest_path = root / "parent_input.json"
    _write_json(parent_manifest_path, parent_manifest)
    contract = preparation.PreparationContract(
        verify_config_sha256=verify_config_sha,
        parent_main_config_sha256=main_config_sha,
        parent_cache_builder_commit=parent_commit,
        parent_input_manifest_sha256=sha256_file(parent_manifest_path),
        parent_input_manifest_size=parent_manifest_path.stat().st_size,
        parent_input_rows_sha256=rows_sha,
        train_portable_marker_sha256=marker_sha,
    )
    identity = _identity(contract)
    synthetic_dir = root / "synthetic_gate"
    synthetic_dir.mkdir()
    synthetic_marker = preparation.write_synthetic_pass_marker(
        synthetic_dir,
        identity=identity,
        contract=contract,
    )
    synthetic_marker_path = synthetic_dir / "SYNTHETIC_PASS.json"
    return {
        "contract": contract,
        "identity": identity,
        "parent_manifest_path": parent_manifest_path,
        "portable_marker_path": marker_path,
        "synthetic_marker_path": synthetic_marker_path,
        "synthetic_marker_sha": sha256_file(synthetic_marker_path),
        "portable_paths": portable_paths,
        "parent_paths": parent_paths,
        "synthetic_marker": synthetic_marker,
    }


def _freeze_input(root: Path, fixture: dict):
    output = root / "kinematic_input.json"
    original_read = preparation._read_authenticated_bytes
    opened = []

    def guarded_read(path, *args, **kwargs):
        resolved = Path(path).resolve()
        opened.append(resolved)
        if resolved in fixture["portable_paths"]:
            raise AssertionError("input freeze opened portable NPZ velocity")
        return original_read(path, *args, **kwargs)

    preparation._read_authenticated_bytes = guarded_read
    try:
        manifest = preparation.build_kinematic_input_manifest(
            output,
            parent_input_manifest_path=fixture["parent_manifest_path"],
            train_portable_marker_path=fixture["portable_marker_path"],
            synthetic_pass_path=fixture["synthetic_marker_path"],
            synthetic_pass_file_sha256=fixture["synthetic_marker_sha"],
            identity=fixture["identity"],
            contract=fixture["contract"],
        )
    finally:
        preparation._read_authenticated_bytes = original_read
    assert opened
    assert fixture["parent_paths"].issubset(set(opened))
    assert set(opened).isdisjoint(fixture["portable_paths"])
    return output, manifest


def test_clean_source_identity_requires_exact_tracked_clean_files():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        for relative in preparation.REQUIRED_SOURCE_PATHS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"source:{relative}\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Synthetic Test",
                "-c",
                "user.email=synthetic@example.invalid",
                "commit",
                "-q",
                "-m",
                "synthetic",
            ],
            cwd=root,
            check=True,
        )
        identity = preparation.capture_clean_source_identity(root)
        assert identity.worktree_clean is True
        assert tuple(identity.source_file_sha256) == preparation.REQUIRED_SOURCE_PATHS
        dirty = root / preparation.REQUIRED_SOURCE_PATHS[-1]
        dirty.write_text("changed\n", encoding="utf-8")
        _expect_error(RuntimeError, preparation.capture_clean_source_identity, root)


def test_composite_descriptor_ids_are_full_stable_hashes_and_immutable():
    contract = replace(
        preparation.PRODUCTION_CONTRACT,
        parent_input_manifest_sha256="3" * 64,
        parent_input_manifest_size=1,
        parent_input_rows_sha256="4" * 64,
        train_portable_marker_sha256="5" * 64,
    )
    identity = _identity(contract)
    descriptors = preparation.composite_descriptor_contracts(
        identity, contract=contract
    )
    assert tuple(descriptors) == (
        "fmt161_plus_seed4",
        "real_neighbor36_plus_seed4",
        "chirality_all35_plus_seed4",
    )
    assert [descriptors[name]["composite_width"] for name in descriptors] == [165, 40, 39]
    assert all(len(str(value["descriptor_id"]).rsplit("_", 1)[-1]) == 64 for value in descriptors.values())
    for value in descriptors.values():
        assert tuple(value["numerical_dependency_source_sha256"]) == (
            preparation.NUMERICAL_DEPENDENCY_SOURCE_PATHS
        )
    _expect_error(TypeError, operator.setitem, descriptors, "bad", {})
    baseline_ids = {value["descriptor_id"] for value in descriptors.values()}
    for dependency in preparation.NUMERICAL_DEPENDENCY_SOURCE_PATHS:
        changed = list(identity.source_file_sha256_items)
        index = [name for name, _digest in changed].index(dependency)
        changed[index] = (dependency, "9" * 64)
        changed_identity = replace(identity, source_file_sha256_items=tuple(changed))
        changed_descriptors = preparation.composite_descriptor_contracts(
            changed_identity, contract=contract
        )
        assert baseline_ids.isdisjoint(
            {value["descriptor_id"] for value in changed_descriptors.values()}
        )


def test_synthetic_marker_is_last_immutable_and_authenticates_evidence():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _synthetic_population(root)
        marker_path = fixture["synthetic_marker_path"]
        marker = fixture["synthetic_marker"]
        assert marker["status"] == "passed"
        _expect_error(
            FileExistsError,
            preparation.write_synthetic_pass_marker,
            marker_path.parent,
            identity=fixture["identity"],
            contract=fixture["contract"],
        )
        report = marker_path.parent / "synthetic_oracle_evidence.json"
        evidence = json.loads(report.read_text(encoding="utf-8"))
        assert evidence["schema"] == preparation.SYNTHETIC_EVIDENCE_SCHEMA
        assert evidence["check_count"] == len(preparation.SYNTHETIC_CHECK_NAMES)
        assert tuple(item["name"] for item in evidence["checks"]) == preparation.SYNTHETIC_CHECK_NAMES
        assert all(item["status"] == "passed" for item in evidence["checks"])
        report.write_text("tampered\n", encoding="utf-8")
        _expect_error(
            ValueError,
            preparation.authenticate_synthetic_pass_marker,
            marker_path,
            expected_file_sha256=fixture["synthetic_marker_sha"],
            identity=fixture["identity"],
            contract=fixture["contract"],
        )


def test_synthetic_pass_cannot_be_forged_from_booleans_or_minimal_report():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        contract = replace(
            preparation.PRODUCTION_CONTRACT,
            parent_input_manifest_sha256="3" * 64,
            parent_input_manifest_size=1,
            parent_input_rows_sha256="4" * 64,
            train_portable_marker_sha256="5" * 64,
        )
        identity = _identity(contract)
        forged = root / "forged"
        forged.mkdir()
        _write_json(
            forged / "synthetic_oracle_evidence.json",
            {
                "status": "passed",
                "check_results": {
                    name: True for name in preparation.SYNTHETIC_CHECK_NAMES
                },
            },
        )
        _expect_error(
            FileExistsError,
            preparation.write_synthetic_pass_marker,
            forged,
            identity=identity,
            contract=contract,
        )
        assert not (forged / "SYNTHETIC_PASS.json").exists()

        broken = root / "broken-production-oracle"
        broken.mkdir()
        original = preparation.compute_seed_time_velocity_gradient

        def zero_gradient(*args, **kwargs):
            values = original(*args, **kwargs)
            return np.zeros_like(values)

        preparation.compute_seed_time_velocity_gradient = zero_gradient
        try:
            _expect_error(
                RuntimeError,
                preparation.write_synthetic_pass_marker,
                broken,
                identity=identity,
                contract=contract,
            )
        finally:
            preparation.compute_seed_time_velocity_gradient = original
        assert not (broken / "SYNTHETIC_PASS.json").exists()


def test_input_freeze_is_exact_32_rows_parent_six_only_and_no_portable_npz_open():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _synthetic_population(root)
        input_path, manifest = _freeze_input(root, fixture)
        assert manifest["row_count"] == 32
        assert len(manifest["rows"]) == 32
        assert manifest["portable_npz_opened_during_freeze"] is False
        assert manifest["forbidden_dataset_access"] is False
        for row in manifest["rows"]:
            assert tuple(row["parent_cache"]["opened_members"]) == preparation.PARENT_PROJECTION_MEMBER_NAMES
            assert set(row["parent_cache"]["allowed_array_sha256"]) == set(
                preparation.PARENT_PROJECTION_MEMBER_NAMES
            )
            assert set(row["portable"]["array_sha256"]) == {
                "velocity",
                "x",
                "y",
                "z",
                "time",
            }
        try:
            manifest["rows"][0]["dataset"] = "changed"
        except TypeError:
            pass
        else:
            raise AssertionError("authenticated input manifest became mutable")
        _expect_error(
            FileExistsError,
            preparation.build_kinematic_input_manifest,
            input_path,
            parent_input_manifest_path=fixture["parent_manifest_path"],
            train_portable_marker_path=fixture["portable_marker_path"],
            synthetic_pass_path=fixture["synthetic_marker_path"],
            synthetic_pass_file_sha256=fixture["synthetic_marker_sha"],
            identity=fixture["identity"],
            contract=fixture["contract"],
        )


def test_input_authentication_rejects_reorder_extra_and_referenced_file_tamper():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _synthetic_population(root)
        input_path, _manifest = _freeze_input(root, fixture)
        value = json.loads(input_path.read_text(encoding="utf-8"))
        value["rows"][0], value["rows"][1] = value["rows"][1], value["rows"][0]
        value["rows_content_sha256"] = canonical_json_sha256(value["rows"])
        value.pop("content_sha256")
        value["content_sha256"] = canonical_json_sha256(value)
        reordered = root / "reordered.json"
        _write_json(reordered, value)
        _expect_error(
            ValueError,
            preparation.authenticate_kinematic_input_manifest,
            reordered,
            expected_file_sha256=sha256_file(reordered),
            identity=fixture["identity"],
            contract=fixture["contract"],
        )

        portable = next(iter(fixture["portable_paths"]))
        with portable.open("ab") as destination:
            destination.write(b"tamper")
        _expect_error(
            ValueError,
            preparation.authenticate_kinematic_input_manifest,
            input_path,
            expected_file_sha256=sha256_file(input_path),
            identity=fixture["identity"],
            contract=fixture["contract"],
            authenticate_all_referenced_rows=True,
        )


def test_single_row_build_closes_sidecar_with_exact_join_and_no_overwrite():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _synthetic_population(root)
        input_path, _manifest = _freeze_input(root, fixture)
        input_sha = sha256_file(input_path)
        sidecar_root = root / "sidecars"
        completion = preparation.build_one_sidecar_and_completion(
            sidecar_root,
            dataset="cylinder3d",
            source_ordinal=0,
            input_manifest_path=input_path,
            input_manifest_file_sha256=input_sha,
            synthetic_pass_path=fixture["synthetic_marker_path"],
            synthetic_pass_file_sha256=fixture["synthetic_marker_sha"],
            identity=fixture["identity"],
            contract=fixture["contract"],
        )
        assert completion["status"] == "passed"
        assert completion["sidecar_row_count"] == 2
        assert completion["forbidden_parent_members_opened"] == ()
        assert completion["forbidden_dataset_access"] is False
        _expect_error(
            FileExistsError,
            preparation.build_one_sidecar_and_completion,
            sidecar_root,
            dataset="cylinder3d",
            source_ordinal=0,
            input_manifest_path=input_path,
            input_manifest_file_sha256=input_sha,
            synthetic_pass_path=fixture["synthetic_marker_path"],
            synthetic_pass_file_sha256=fixture["synthetic_marker_sha"],
            identity=fixture["identity"],
            contract=fixture["contract"],
        )


def test_full_population_requires_exact_32_completions_and_authenticates_all():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _synthetic_population(root)
        input_path, _manifest = _freeze_input(root, fixture)
        input_sha = sha256_file(input_path)
        sidecar_root = root / "sidecars"
        for dataset, _family in preparation.DATASET_FAMILY_PAIRS:
            for ordinal in range(4):
                preparation.build_one_sidecar_and_completion(
                    sidecar_root,
                    dataset=dataset,
                    source_ordinal=ordinal,
                    input_manifest_path=input_path,
                    input_manifest_file_sha256=input_sha,
                    synthetic_pass_path=fixture["synthetic_marker_path"],
                    synthetic_pass_file_sha256=fixture["synthetic_marker_sha"],
                    identity=fixture["identity"],
                    contract=fixture["contract"],
                )
        extra = sidecar_root / "unexpected.txt"
        extra.write_text("extra\n", encoding="utf-8")
        _expect_error(
            ValueError,
            preparation.write_sidecar_population_manifest,
            sidecar_root,
            input_manifest_path=input_path,
            input_manifest_file_sha256=input_sha,
            synthetic_pass_path=fixture["synthetic_marker_path"],
            synthetic_pass_file_sha256=fixture["synthetic_marker_sha"],
            identity=fixture["identity"],
            contract=fixture["contract"],
        )
        extra.unlink()
        population = preparation.write_sidecar_population_manifest(
            sidecar_root,
            input_manifest_path=input_path,
            input_manifest_file_sha256=input_sha,
            synthetic_pass_path=fixture["synthetic_marker_path"],
            synthetic_pass_file_sha256=fixture["synthetic_marker_sha"],
            identity=fixture["identity"],
            contract=fixture["contract"],
        )
        assert population["sidecar_count"] == 32
        assert population["sidecar_row_count_total"] == 64
        assert len(population["rows"]) == 32
        assert population["forbidden_dataset_access"] is False
        _expect_error(
            FileExistsError,
            preparation.write_sidecar_population_manifest,
            sidecar_root,
            input_manifest_path=input_path,
            input_manifest_file_sha256=input_sha,
            synthetic_pass_path=fixture["synthetic_marker_path"],
            synthetic_pass_file_sha256=fixture["synthetic_marker_sha"],
            identity=fixture["identity"],
            contract=fixture["contract"],
        )


def test_forbidden_dataset_names_are_rejected_before_any_file_access():
    _expect_error(
        ValueError,
        replace,
        preparation.PRODUCTION_CONTRACT,
        dataset_family_pairs=(("tangaroa", "forbidden"),),
        parent_input_manifest_sha256="3" * 64,
        parent_input_manifest_size=1,
        parent_input_rows_sha256="4" * 64,
        train_portable_marker_sha256="5" * 64,
    )

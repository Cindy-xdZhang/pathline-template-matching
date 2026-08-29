from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np

from pathline_template_matching.arc_length_primitives import ArcLengthScaleTable
from pathline_template_matching.encoder import IndependentFMT3DConfig
from pathline_template_matching.netcdf_io import FlowWindow3D
from pathline_template_matching.phase21_pipeline import (
    EXPERIMENT31,
    EXPANDED_ARC_VALUES,
    EXPANDED_DS_VALUES,
    EXPANDED_DX_VALUES,
    METHODS,
    Phase21Plan,
    ScaleAssignmentBlock,
    audit_phase31_train_coverage,
    balanced_scale_assignment,
    build_phase21_cache_slice,
    cache_summary_payload,
    load_phase21_plan,
    load_phase31_plan,
    recover_phase21_cache_summary,
    run_phase21_from_resolvers,
    validate_phase31_cache_portable_population_evidence,
    validate_phase31_synthetic_pass,
    validate_phase31_train_coverage_pass,
)
from pathline_template_matching.portable_flow import (
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase31_union_preserves_every_phase21_tuple_and_freezes_new_block():
    phase21 = load_phase21_plan(ROOT / "config/mainExp_TemplateMatching_2.1.yaml")
    phase31 = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")

    assert len(phase21.scale_table) == 1_000
    assert len(phase31.scale_table) == 2_000
    for field in (
        "scale_id",
        "dx_grid_scale",
        "ds_frame_scale",
        "arc_length_grid_scale",
    ):
        old = getattr(phase21.scale_table, field)
        new = getattr(phase31.scale_table, field)
        np.testing.assert_array_equal(new[:1_000], old)
    new_table = phase31.scale_table
    np.testing.assert_array_equal(
        np.unique(new_table.dx_grid_scale[1_000:]), np.asarray(EXPANDED_DX_VALUES)
    )
    np.testing.assert_array_equal(
        np.unique(new_table.ds_frame_scale[1_000:]), np.asarray(EXPANDED_DS_VALUES)
    )
    np.testing.assert_array_equal(
        np.unique(new_table.arc_length_grid_scale[1_000:]),
        np.asarray(EXPANDED_ARC_VALUES),
    )
    tuples = np.column_stack(
        (
            new_table.dx_grid_scale,
            new_table.ds_frame_scale,
            new_table.arc_length_grid_scale,
        )
    )
    assert len({tuple(row) for row in tuples}) == 2_000


def test_phase31_two_assignments_are_block_balanced_and_legacy_exact():
    phase21 = load_phase21_plan(ROOT / "config/mainExp_TemplateMatching_2.1.yaml")
    phase31 = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    old = phase21.primitive_scale_assignment()
    union = phase31.primitive_scale_assignment()

    assert phase31.assigned_seed_count == 64_000
    assert phase31.assigned_primitive_count == 128_000
    np.testing.assert_array_equal(union[:64_000], old)
    np.testing.assert_array_equal(
        union[64_000:] - 1_000,
        balanced_scale_assignment(64_000, 1_000, 35068),
    )
    np.testing.assert_array_equal(
        np.bincount(union, minlength=2_000), np.full(2_000, 64)
    )
    assert not np.array_equal(union[:64_000], union[64_000:] - 1_000)


def test_phase31_uses_forty_nine_frame_source_windows():
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    assert plan.window_frame_count == 49
    assert plan.maximum_source_frame_intervals == 48.0
    expected = {
        151: (0, 34, 68, 102),
        76: (0, 9, 18, 27),
        171: (0, 40, 81, 122),
        234: (0, 61, 123, 185),
        159: (0, 36, 73, 110),
        199: (0, 50, 100, 150),
        201: (0, 50, 101, 152),
        160: (0, 37, 74, 111),
    }
    for frame_count, indices in expected.items():
        assert plan.source_indices(frame_count) == indices


def _tiny_phase31_plan(config_path: Path) -> Phase21Plan:
    payload = config_path.read_bytes()
    train = tuple(f"train_{index}" for index in range(8))
    test = ("test_0", "test_1")
    datasets = train + test
    return Phase21Plan(
        config_path=config_path,
        config_sha256=hashlib.sha256(payload).hexdigest(),
        dataset_registry_path=ROOT / "config/datasets.yaml",
        dataset_registry_sha256=hashlib.sha256(
            (ROOT / "config/datasets.yaml").read_bytes()
        ).hexdigest(),
        config={"experiment": EXPERIMENT31, "profile": "tiny_test_only"},
        experiment=EXPERIMENT31,
        output_root="tiny",
        train_datasets=train,
        test_datasets=test,
        family_by_dataset={name: f"family_{name}" for name in datasets},
        source_count=1,
        window_frame_count=49,
        seed_shape_xyz=(9, 2, 3),
        scale_table=ArcLengthScaleTable(
            scale_id=np.asarray([0, 1], dtype=np.int32),
            dx_grid_scale=np.asarray([0.5, 1.0]),
            ds_frame_scale=np.asarray([0.125, 0.25]),
            arc_length_grid_scale=np.asarray([0.05, 0.08]),
        ),
        assignment_seed=15068,
        library_seed=15068,
        pca_components=161,
        bootstrap_seed=25068,
        bootstrap_replicates=20,
        descriptor_config=IndependentFMT3DConfig(),
        method_ids=METHODS,
        required_outputs=(),
        maximum_source_frame_intervals=48.0,
        assignment_count_per_seed=2,
        scale_blocks=(
            ScaleAssignmentBlock("legacy_2_1", 0, 1, 15068),
            ScaleAssignmentBlock("expanded_3_1", 1, 1, 35068),
        ),
        maximum_library_templates=128_000,
    )


def _tiny_phase31_window() -> FlowWindow3D:
    coordinate = np.linspace(0.0, 2.0, 11, dtype=np.float64)
    velocity = np.zeros((49, 11, 11, 11, 3), dtype=np.float32)
    velocity[..., 2] = (
        0.02 * coordinate + 0.1 * np.maximum(coordinate - 0.8, 0.0)
    )[None, None, None, :]
    return FlowWindow3D(
        velocity=velocity,
        coordinates_xyz=(coordinate, coordinate, coordinate),
        time=np.arange(49, dtype=np.float64),
        source_path="synthetic://phase31",
        source_start_index=0,
        spatial_strides={"x": 1, "y": 1, "z": 1},
        components=("u", "v", "w"),
        coordinate_sources={"x": "x", "y": "y", "z": "z", "t": "time"},
    )


def _tiny_phase31_verification_evidence(plan: Phase21Plan) -> dict:
    output = [{"path": "evidence.json", "size_bytes": 1, "sha256": "c" * 64}]
    synthetic = {
        "path": "synthetic/SYNTHETIC_PASS.json",
        "file_size": 1,
        "file_sha256": "a" * 64,
        "git_commit": "tiny-test",
        "main_config_sha256": plan.config_sha256,
        "verify_config_sha256": "d" * 64,
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "outputs": output,
    }
    return {
        "path": "coverage/TRAIN_COVERAGE_PASS.json",
        "file_size": 1,
        "file_sha256": "b" * 64,
        "git_commit": "tiny-test",
        "main_config_sha256": plan.config_sha256,
        "verify_config_sha256": "d" * 64,
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "synthetic_pass_file_sha256": synthetic["file_sha256"],
        "synthetic_pass": synthetic,
        "outputs": output,
    }


def _write_phase31_portable_population_marker(
    root: Path,
    plan: Phase21Plan,
    *,
    scope: str,
    commit: str,
    synthetic_pass_file_sha256: str = "5" * 64,
    train_coverage_pass_file_sha256: str = "6" * 64,
) -> dict:
    datasets = plan.train_datasets if scope == "train-only" else plan.datasets
    rows = [
        {"dataset": dataset, "source_ordinal": ordinal}
        for dataset in datasets
        for ordinal in range(plan.source_count)
    ]
    rows_sha = canonical_json_sha256(rows)
    marker_dir = root / f"{scope}_portable_marker"
    marker_dir.mkdir()
    marker_path = marker_dir / (
        "TRAIN_PORTABLES_PASS.json"
        if scope == "train-only"
        else "ALL_PORTABLES_PASS.json"
    )
    marker = {
        "schema": "pathline_template_matching.phase31_portable_population_pass.v1",
        "experiment": plan.experiment,
        "status": "passed",
        "access_scope": scope,
        "git_commit": commit,
        "worktree_clean": True,
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "portable_root": str((root / "portable").resolve()),
        "dataset_count": len(datasets),
        "window_count": len(rows),
        "rows": rows,
        "rows_content_sha256": rows_sha,
        "synthetic_pass_file_sha256": synthetic_pass_file_sha256,
    }
    if scope == "all":
        marker["train_coverage_pass_file_sha256"] = (
            train_coverage_pass_file_sha256
        )
    marker_path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    return {
        "portable_population_scope": scope,
        "portable_population_pass_path": str(marker_path.resolve()),
        "portable_population_pass_file_size": marker_path.stat().st_size,
        "portable_population_pass_file_sha256": sha256_file(marker_path),
        "portable_population_rows_content_sha256": rows_sha,
    }


def test_phase31_cache_recovery_preserves_assigned_center_and_block_identity():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "tiny31.yaml"
        config_path.write_text(
            "experiment: mainExp_TemplateMatching_3.1\nprofile: tiny_test_only\n",
            encoding="utf-8",
        )
        plan = _tiny_phase31_plan(config_path)
        cache_path = root / "cache.npz"
        original = build_phase21_cache_slice(
            plan,
            dataset="train_0",
            source_ordinal=0,
            source_index=0,
            resolved_input=_tiny_phase31_window(),
            cache_path=cache_path,
            strict_evidence=False,
            cache_builder_git_commit="tiny-test",
        )
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            assigned = np.asarray(archive["valid_assigned_row_index"])
            center = np.asarray(archive["valid_center_seed_index"])
            block = np.asarray(archive["valid_scale_block_index"])
            scale = np.asarray(archive["valid_scale_id"])
            np.testing.assert_array_equal(archive["valid_seed_index"], assigned)
            np.testing.assert_array_equal(center, assigned % plan.assigned_seed_count)
            np.testing.assert_array_equal(block, assigned // plan.assigned_seed_count)
            assert np.all(scale[block == 0] == 0)
            assert np.all(scale[block == 1] == 1)
            assert set(metadata["array_sha256"]) == set(archive.files) - {
                "metadata_json"
            }
            assert metadata["maximum_source_frame_intervals"] == 48.0
            stored = {name: np.array(archive[name], copy=True) for name in archive.files}
        recovered = recover_phase21_cache_summary(
            plan,
            cache_path=cache_path,
            dataset="train_0",
            source_ordinal=0,
            source_index=0,
            cache_builder_git_commit="tiny-test",
            strict_evidence=False,
        )
        assert cache_summary_payload(recovered) == cache_summary_payload(original)

        damaged = {name: np.array(value, copy=True) for name, value in stored.items()}
        damaged["valid_center_seed_index"].flat[0] += 1
        damaged_path = root / "damaged.npz"
        np.savez_compressed(damaged_path, **damaged)
        try:
            recover_phase21_cache_summary(
                plan,
                cache_path=damaged_path,
                dataset="train_0",
                source_ordinal=0,
                source_index=0,
                cache_builder_git_commit="tiny-test",
                strict_evidence=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("3.1 recovery accepted damaged center-seed identity")


def test_phase31_tiny_eight_two_end_to_end_keeps_blocks_separate():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "tiny31.yaml"
        config_path.write_text(
            "experiment: mainExp_TemplateMatching_3.1\nprofile: tiny_test_only\n",
            encoding="utf-8",
        )
        plan = _tiny_phase31_plan(config_path)
        run_dir = root / "run"
        result = run_phase21_from_resolvers(
            plan,
            run_dir=run_dir,
            frame_count_resolver=lambda _dataset: 49,
            window_resolver=lambda _dataset, _source, _frames: (
                _tiny_phase31_window()
            ),
            git_commit="tiny-test",
            device="cpu",
            strict_protocol=False,
            integration_chunk_size=256,
            encoding_chunk_size=256,
            verify_cache_hashes=True,
            phase31_verification_evidence=_tiny_phase31_verification_evidence(
                plan
            ),
        )
        assert result["status"] == "development_completed_confirmation_not_run"
        assert result["maximum_source_frame_intervals"] == 48.0
        assert result["scale_count"] == 2
        assert result["assignment_count_per_seed"] == 2
        assert result["assigned_test_query_count"] == 216
        with (run_dir / "per_scale_block_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as source:
            block_rows = list(csv.DictReader(source))
        assert len(block_rows) == 2 * len(METHODS)
        assert {row["scale_block_id"] for row in block_rows} == {
            "legacy_2_1",
            "expanded_3_1",
        }
        query_header = (run_dir / "per_query_matches.csv").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        assert "query_assigned_row_index" in query_header
        assert "query_center_seed_index" in query_header
        assert "query_scale_block_id" in query_header
        assert "query_seed_index" not in query_header
        input_manifest = json.loads(
            (run_dir / "input_manifest.json").read_text(encoding="utf-8")
        )
        result_manifest = json.loads(
            (run_dir / "result_manifest.json").read_text(encoding="utf-8")
        )
        for manifest in (input_manifest, result_manifest):
            gates = manifest["verification_gates"]
            assert gates["synthetic_pass"]["file_sha256"] == "a" * 64
            assert gates["train_coverage_pass"]["file_sha256"] == "b" * 64
            assert gates["train_coverage_pass"]["git_commit"] == "tiny-test"
            assert gates["train_coverage_pass"]["outputs"] == [
                {"path": "evidence.json", "size_bytes": 1, "sha256": "c" * 64}
            ]


def test_phase31_strict_resolver_rejects_before_any_window_access():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "tiny31.yaml"
        config_path.write_text(
            "experiment: mainExp_TemplateMatching_3.1\nprofile: tiny_test_only\n",
            encoding="utf-8",
        )
        plan = _tiny_phase31_plan(config_path)
        calls = {"frame": 0, "window": 0}

        def frame_count(_dataset):
            calls["frame"] += 1
            return 49

        def window(_dataset, _source, _frames):
            calls["window"] += 1
            return _tiny_phase31_window()

        try:
            run_phase21_from_resolvers(
                plan,
                run_dir=root / "run",
                frame_count_resolver=frame_count,
                window_resolver=window,
                git_commit="tiny-test",
                strict_protocol=True,
            )
        except ValueError as error:
            assert "marker-gated parallel" in str(error)
        else:
            raise AssertionError("strict 3.1 resolver bypassed the Verify gates")
        assert calls == {"frame": 0, "window": 0}


def test_phase31_train_only_coverage_reports_all_strata_and_refuses_test_rows():
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    assignment = plan.primitive_scale_assignment()
    valid = np.ones(plan.assigned_primitive_count, dtype=np.bool_)
    center_labels = (np.arange(plan.assigned_seed_count) % 2 == 0)
    labels = np.concatenate((center_labels, center_labels)).astype(np.bool_)
    array_hashes = {
        "scale_assignment": canonical_array_sha256(assignment),
        "valid_mask": canonical_array_sha256(valid),
        "reference_labels_all": canonical_array_sha256(labels),
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        portable_population = _write_phase31_portable_population_marker(
            root, plan, scope="train-only", commit="2" * 40
        )
        rows = []
        for dataset in plan.train_datasets:
            for ordinal in range(plan.source_count):
                cache_path = root / f"{dataset}_{ordinal}.npz"
                metadata = {
                    "schema": plan.cache_schema,
                    "experiment": plan.experiment,
                    "config_sha256": plan.config_sha256,
                    "dataset": dataset,
                    "physical_family": plan.family_by_dataset[dataset],
                    "split": "train",
                    "source_ordinal": ordinal,
                    "source_index": ordinal,
                    "dataset_registry_sha256": plan.dataset_registry_sha256,
                    "portable_builder_git_commit": "2" * 40,
                    "cache_builder_git_commit": "2" * 40,
                    "maximum_source_frame_intervals": 48.0,
                    "assignment_count_per_seed": 2,
                    "assigned_primitive_count": plan.assigned_primitive_count,
                    "array_sha256": array_hashes,
                }
                np.savez_compressed(
                    cache_path,
                    scale_assignment=assignment,
                    valid_mask=valid,
                    reference_labels_all=labels,
                    metadata_json=np.asarray(
                        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                    ),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "split": "train",
                        "source_ordinal": ordinal,
                        "source_index": ordinal,
                        "path": str(cache_path),
                        "file_sha256": sha256_file(cache_path),
                        "config_sha256": plan.config_sha256,
                        "dataset_registry_sha256": plan.dataset_registry_sha256,
                        "portable_builder_git_commit": "2" * 40,
                        "cache_builder_git_commit": "2" * 40,
                        **portable_population,
                    }
                )
        output = root / "coverage"
        summary = audit_phase31_train_coverage(
            plan,
            rows,
            output,
            expected_git_commit="2" * 40,
            synthetic_pass_file_sha256="5" * 64,
            authorized_portable_population_marker_path=(
                portable_population["portable_population_pass_path"]
            ),
        )
        assert summary["status"] == "pass"
        assert summary["diagnostic_row_count"] == 64_000
        assert all(summary["pass_conditions"].values())
        assert summary["cache_count"] == 32
        assert summary["train_portable_population_pass"]["file_sha256"] == (
            portable_population["portable_population_pass_file_sha256"]
        )
        assert (output / "train_only_coverage_diagnostics.csv").is_file()
        assert (output / "train_only_coverage_summary.json").is_file()
        assert all(
            row["valid_train_count"] > 0
            for row in summary["expanded_arc_level_valid_counts"]
        )

        rejected = list(rows)
        rejected[0] = {
            **rejected[0],
            "dataset": plan.test_datasets[0],
            "split": "test",
            "path": str(root / "does_not_exist_test_cache.npz"),
        }
        try:
            audit_phase31_train_coverage(
                plan,
                rejected,
                root / "rejected",
                expected_git_commit="2" * 40,
                synthetic_pass_file_sha256="5" * 64,
                authorized_portable_population_marker_path=(
                    portable_population["portable_population_pass_path"]
                ),
            )
        except ValueError as error:
            assert "non-train dataset" in str(error)
        else:
            raise AssertionError("train-only coverage accepted a test cache row")


def test_phase31_verify_markers_authenticate_outputs_configs_and_prior_phase():
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    verify_path = ROOT / "config/Verify_LongArcHorizon_1.1.yaml"
    commit = "3" * 40

    def output_rows(root: Path, names: tuple[str, ...]):
        return [
            {
                "path": name,
                "size_bytes": int((root / name).stat().st_size),
                "sha256": sha256_file(root / name),
            }
            for name in names
        ]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        phase_a = root / "phase_a"
        phase_a.mkdir()
        phase_a_names = (
            "frozen_verify_config.yaml",
            "frozen_main_config.yaml",
            "synthetic_verification.json",
            "scale_union_manifest.json",
            "assignment_verification.json",
            "environment_versions.json",
        )
        (phase_a / phase_a_names[0]).write_bytes(verify_path.read_bytes())
        (phase_a / phase_a_names[1]).write_bytes(plan.config_path.read_bytes())
        for name in phase_a_names[2:]:
            (phase_a / name).write_text("{}", encoding="utf-8")
        phase_a_rows = output_rows(phase_a, phase_a_names)
        synthetic_marker = {
            "schema": "pathline_template_matching.long_arc_synthetic_pass.v1",
            "experiment": "Verify_LongArcHorizon_1.1",
            "phase": "synthetic",
            "status": "synthetic_gate_passed_train_only_coverage_not_run",
            "git_commit": commit,
            "worktree_clean": True,
            "main_config_sha256": plan.config_sha256,
            "verify_config_sha256": sha256_file(verify_path),
            "dataset_registry_sha256": plan.dataset_registry_sha256,
            "train_only_coverage_gate_run": False,
            "final_verify_pass": False,
            "outputs": phase_a_rows,
            "outputs_content_sha256": canonical_json_sha256(phase_a_rows),
        }
        synthetic_path = phase_a / "SYNTHETIC_PASS.json"
        synthetic_path.write_text(
            json.dumps(synthetic_marker, sort_keys=True), encoding="utf-8"
        )
        synthetic = validate_phase31_synthetic_pass(
            plan,
            synthetic_path,
            verify_config_path=verify_path,
            current_git_commit=commit,
        )
        portable_row_evidence = _write_phase31_portable_population_marker(
            root,
            plan,
            scope="train-only",
            commit=commit,
            synthetic_pass_file_sha256=synthetic["file_sha256"],
        )
        portable_evidence = {
            "split": "train",
            "access_scope": "train-only",
            "path": portable_row_evidence["portable_population_pass_path"],
            "file_size": portable_row_evidence[
                "portable_population_pass_file_size"
            ],
            "file_sha256": portable_row_evidence[
                "portable_population_pass_file_sha256"
            ],
            "rows_content_sha256": portable_row_evidence[
                "portable_population_rows_content_sha256"
            ],
            "git_commit": commit,
            "dataset_count": 8,
            "window_count": 32,
            "synthetic_pass_file_sha256": synthetic["file_sha256"],
            "train_coverage_pass_file_sha256": None,
        }

        phase_b = root / "phase_b"
        phase_b.mkdir()
        phase_b_names = (
            "frozen_verify_config.yaml",
            "frozen_main_config.yaml",
            "train_cache_input_manifest.json",
            "train_only_coverage_diagnostics.csv",
            "train_only_coverage_summary.json",
            "environment_versions.json",
        )
        (phase_b / phase_b_names[0]).write_bytes(verify_path.read_bytes())
        (phase_b / phase_b_names[1]).write_bytes(plan.config_path.read_bytes())
        for name in phase_b_names[2:]:
            (phase_b / name).write_text("{}", encoding="utf-8")
        (phase_b / "train_cache_input_manifest.json").write_text(
            json.dumps(
                {"train_portable_population_pass": portable_evidence},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        verification = {
            "schema": "pathline_template_matching.long_arc_verification.v1",
            "experiment": "Verify_LongArcHorizon_1.1",
            "phase": "train_coverage",
            "status": "passed",
            "final_verify_pass": True,
            "git_commit": commit,
            "main_config_sha256": plan.config_sha256,
            "verify_config_sha256": sha256_file(verify_path),
            "synthetic_pass_file_sha256": synthetic["file_sha256"],
            "train_portable_population_pass": portable_evidence,
            "train_portable_population_pass_file_sha256": portable_evidence[
                "file_sha256"
            ],
            "train_only": True,
            "no_test_dataset_access": True,
        }
        verification_path = phase_b / "verification.json"
        verification_path.write_text(
            json.dumps(verification, sort_keys=True), encoding="utf-8"
        )
        phase_b_rows = output_rows(
            phase_b, phase_b_names + ("verification.json",)
        )
        coverage_marker = {
            "schema": "pathline_template_matching.long_arc_train_coverage_pass.v1",
            "experiment": "Verify_LongArcHorizon_1.1",
            "phase": "train_coverage",
            "status": "passed",
            "git_commit": commit,
            "worktree_clean": True,
            "main_config_sha256": plan.config_sha256,
            "verify_config_sha256": sha256_file(verify_path),
            "dataset_registry_sha256": plan.dataset_registry_sha256,
            "synthetic_pass_file_sha256": synthetic["file_sha256"],
            "train_portable_population_pass_file_sha256": portable_evidence[
                "file_sha256"
            ],
            "verification_file_sha256": sha256_file(verification_path),
            "final_verify_pass": True,
            "outputs": phase_b_rows,
            "outputs_content_sha256": canonical_json_sha256(phase_b_rows),
        }
        coverage_path = phase_b / "TRAIN_COVERAGE_PASS.json"
        coverage_path.write_text(
            json.dumps(coverage_marker, sort_keys=True), encoding="utf-8"
        )
        validated = validate_phase31_train_coverage_pass(
            plan,
            coverage_path,
            synthetic_pass_path=synthetic_path,
            verify_config_path=verify_path,
            current_git_commit=commit,
        )
        assert validated["synthetic_pass_file_sha256"] == synthetic["file_sha256"]

        (phase_b / "train_only_coverage_diagnostics.csv").write_text(
            "damaged", encoding="utf-8"
        )
        try:
            validate_phase31_train_coverage_pass(
                plan,
                coverage_path,
                synthetic_pass_path=synthetic_path,
                verify_config_path=verify_path,
                current_git_commit=commit,
            )
        except ValueError as error:
            assert "hash/size changed" in str(error)
        else:
            raise AssertionError("Phase B marker accepted damaged evidence")

        # A self-consistent marker/output-list rewrite must not authenticate a
        # different frozen config either.
        (phase_b / "train_only_coverage_diagnostics.csv").write_text(
            "{}", encoding="utf-8"
        )
        (phase_b / "frozen_main_config.yaml").write_text(
            "experiment: malicious_config\n", encoding="utf-8"
        )
        rewritten_rows = output_rows(
            phase_b, phase_b_names + ("verification.json",)
        )
        coverage_marker["outputs"] = rewritten_rows
        coverage_marker["outputs_content_sha256"] = canonical_json_sha256(
            rewritten_rows
        )
        coverage_path.write_text(
            json.dumps(coverage_marker, sort_keys=True), encoding="utf-8"
        )
        try:
            validate_phase31_train_coverage_pass(
                plan,
                coverage_path,
                synthetic_pass_path=synthetic_path,
                verify_config_path=verify_path,
                current_git_commit=commit,
            )
        except ValueError as error:
            assert "frozen main config differs" in str(error)
        else:
            raise AssertionError("Phase B accepted a self-consistent wrong config")


def test_phase31_runner_phase_b_writes_seven_outputs_then_pass_marker():
    spec = importlib.util.spec_from_file_location(
        "phase31_runner_writer_test",
        ROOT / "scripts/run_mainexp_template_matching_2_1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    commit = "4" * 40
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = replace(plan, output_root=str(root / "output"))
        marker_scope_root = (
            Path(plan.output_root)
            / "verification"
            / "portable_population"
            / "train_only"
        )
        marker_scope_root.mkdir(parents=True)
        portable_population = _write_phase31_portable_population_marker(
            marker_scope_root,
            plan,
            scope="train-only",
            commit=commit,
            synthetic_pass_file_sha256="5" * 64,
        )
        cache_root = root / "cache"
        cache_rows = {}
        for dataset in plan.train_datasets:
            dataset_dir = cache_root / "train" / dataset
            dataset_dir.mkdir(parents=True)
            for ordinal in range(4):
                sidecar = dataset_dir / f"source_{ordinal}.summary.json"
                sidecar.write_text("{}", encoding="utf-8")
                cache_path = dataset_dir / f"source_{ordinal}.npz"
                cache_path.write_bytes(f"{dataset}:{ordinal}".encode("utf-8"))
                cache_rows[sidecar.resolve()] = {
                    "dataset": dataset,
                    "split": "train",
                    "source_ordinal": ordinal,
                    "source_index": ordinal,
                    "path": str(cache_path.resolve()),
                    "file_sha256": sha256_file(cache_path),
                    "config_sha256": plan.config_sha256,
                    "dataset_registry_sha256": plan.dataset_registry_sha256,
                    "portable_builder_git_commit": commit,
                    "cache_builder_git_commit": commit,
                    **portable_population,
                }

        originals = {
            name: getattr(module, name)
            for name in (
                "_load_plan",
                "_git_commit_and_clean",
                "validate_phase31_synthetic_pass",
                "load_cache_summary_sidecar",
                "audit_phase31_train_coverage",
                "configure_deterministic_execution",
                "_validate_portable_population_pass",
            )
        }
        synthetic_sha = "5" * 64

        def fake_audit(_plan, rows, output_dir, **_kwargs):
            assert len(rows) == 32
            output = Path(output_dir)
            (output / "train_only_coverage_diagnostics.csv").write_text(
                "dataset,valid_count\n", encoding="utf-8"
            )
            (output / "train_only_coverage_summary.json").write_text(
                "{}", encoding="utf-8"
            )
            return {
                "status": "pass",
                "pass_conditions": {"synthetic_test_gate": True},
            }

        module._load_plan = lambda *_args, **_kwargs: plan
        module._git_commit_and_clean = lambda *_args, **_kwargs: commit
        module.validate_phase31_synthetic_pass = lambda *_args, **_kwargs: {
            "path": str(root / "SYNTHETIC_PASS.json"),
            "file_sha256": synthetic_sha,
            "verify_config_sha256": sha256_file(
                ROOT / "config/Verify_LongArcHorizon_1.1.yaml"
            ),
        }
        module.load_cache_summary_sidecar = lambda path: SimpleNamespace(
            cache_row=cache_rows[Path(path).resolve()]
        )
        module.audit_phase31_train_coverage = fake_audit
        module.configure_deterministic_execution = lambda: {"test": True}
        module._validate_portable_population_pass = lambda *_args, **_kwargs: {
            "file_sha256": portable_population[
                "portable_population_pass_file_sha256"
            ]
        }
        run_dir = root / "phase_b"
        args = SimpleNamespace(
            config=plan.config_path,
            expected_experiment=plan.experiment,
            synthetic_pass=root / "SYNTHETIC_PASS.json",
            verify_config=ROOT / "config/Verify_LongArcHorizon_1.1.yaml",
            cache_root=cache_root,
            run_dir=run_dir,
            portable_population_pass=Path(
                portable_population["portable_population_pass_path"]
            ),
        )
        try:
            module._audit_train_coverage(args)
        finally:
            for name, value in originals.items():
                setattr(module, name, value)
        expected = {
            "frozen_verify_config.yaml",
            "frozen_main_config.yaml",
            "train_cache_input_manifest.json",
            "train_only_coverage_diagnostics.csv",
            "train_only_coverage_summary.json",
            "environment_versions.json",
            "verification.json",
            "TRAIN_COVERAGE_PASS.json",
        }
        assert {path.name for path in run_dir.iterdir()} == expected
        marker = json.loads(
            (run_dir / "TRAIN_COVERAGE_PASS.json").read_text(encoding="utf-8")
        )
        assert marker["status"] == "passed"
        assert marker["final_verify_pass"] is True
        assert marker["synthetic_pass_file_sha256"] == synthetic_sha
        assert marker["train_portable_population_pass_file_sha256"] == (
            portable_population["portable_population_pass_file_sha256"]
        )
        assert [row["path"] for row in marker["outputs"]] == [
            "frozen_verify_config.yaml",
            "frozen_main_config.yaml",
            "train_cache_input_manifest.json",
            "train_only_coverage_diagnostics.csv",
            "train_only_coverage_summary.json",
            "environment_versions.json",
            "verification.json",
        ]


def test_phase31_phase_b_rejects_escaped_cache_path_before_target_stat():
    spec = importlib.util.spec_from_file_location(
        "phase31_runner_malicious_sidecar_test",
        ROOT / "scripts/run_mainexp_template_matching_2_1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = replace(plan, output_root=str(root / "output"))
        marker_scope_root = (
            Path(plan.output_root)
            / "verification"
            / "portable_population"
            / "train_only"
        )
        marker_scope_root.mkdir(parents=True)
        portable_population = _write_phase31_portable_population_marker(
            marker_scope_root,
            plan,
            scope="train-only",
            commit="4" * 40,
            synthetic_pass_file_sha256="5" * 64,
        )
        cache_root = root / "cache"
        summaries = {}
        escaped = root / "test" / plan.test_datasets[0] / "secret.npz"
        for dataset in plan.train_datasets:
            dataset_dir = cache_root / "train" / dataset
            dataset_dir.mkdir(parents=True)
            for ordinal in range(plan.source_count):
                sidecar = dataset_dir / f"source_{ordinal}.summary.json"
                sidecar.write_text("{}", encoding="utf-8")
                cache_path = dataset_dir / f"source_{ordinal}.npz"
                if dataset == plan.train_datasets[-1] and ordinal == 3:
                    cache_path = escaped
                summaries[sidecar.resolve()] = SimpleNamespace(
                    cache_row={
                        "dataset": dataset,
                        "split": "train",
                        "source_ordinal": ordinal,
                        "source_index": ordinal,
                        "path": str(cache_path.resolve()),
                        "file_sha256": "1" * 64,
                        "config_sha256": plan.config_sha256,
                        "dataset_registry_sha256": plan.dataset_registry_sha256,
                        "portable_builder_git_commit": "4" * 40,
                        "cache_builder_git_commit": "4" * 40,
                        **portable_population,
                    }
                )

        originals = {
            name: getattr(module, name)
            for name in (
                "_load_plan",
                "_git_commit_and_clean",
                "validate_phase31_synthetic_pass",
                "load_cache_summary_sidecar",
                "configure_deterministic_execution",
                "_validate_portable_population_pass",
            )
        }
        original_stat = Path.stat
        target_was_statted = False

        def guarded_stat(path, *args, **kwargs):
            nonlocal target_was_statted
            if str(path.absolute()) == str(escaped.absolute()):
                target_was_statted = True
                raise AssertionError("escaped test target was statted")
            return original_stat(path, *args, **kwargs)

        module._load_plan = lambda *_args, **_kwargs: plan
        module._git_commit_and_clean = lambda *_args, **_kwargs: "4" * 40
        module.validate_phase31_synthetic_pass = lambda *_args, **_kwargs: {
            "path": "synthetic/SYNTHETIC_PASS.json",
            "file_sha256": "5" * 64,
            "verify_config_sha256": sha256_file(
                ROOT / "config/Verify_LongArcHorizon_1.1.yaml"
            ),
        }
        module.load_cache_summary_sidecar = lambda path: summaries[
            Path(path).resolve()
        ]
        module.configure_deterministic_execution = lambda: {"test": True}
        module._validate_portable_population_pass = lambda *_args, **_kwargs: {
            "file_sha256": portable_population[
                "portable_population_pass_file_sha256"
            ]
        }
        Path.stat = guarded_stat
        args = SimpleNamespace(
            config=plan.config_path,
            expected_experiment=plan.experiment,
            synthetic_pass=root / "SYNTHETIC_PASS.json",
            verify_config=ROOT / "config/Verify_LongArcHorizon_1.1.yaml",
            cache_root=cache_root,
            run_dir=root / "phase_b",
            portable_population_pass=Path(
                portable_population["portable_population_pass_path"]
            ),
        )
        try:
            try:
                module._audit_train_coverage(args)
            except ValueError as error:
                assert "escapes its authorized train directory" in str(error)
            else:
                raise AssertionError("Phase B accepted an escaped cache target")
        finally:
            Path.stat = original_stat
            for name, value in originals.items():
                setattr(module, name, value)
        assert target_was_statted is False


def test_phase31_phase_b_rejects_unauthorized_marker_path_before_open():
    spec = importlib.util.spec_from_file_location(
        "phase31_runner_marker_path_test",
        ROOT / "scripts/run_mainexp_template_matching_2_1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    production = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = replace(production, output_root=str(root / "output"))
        malicious = root / "outside" / "TRAIN_PORTABLES_PASS.json"
        malicious.parent.mkdir()
        malicious.write_text("{}", encoding="utf-8")
        originals = {
            name: getattr(module, name)
            for name in (
                "_load_plan",
                "_git_commit_and_clean",
                "validate_phase31_synthetic_pass",
            )
        }
        original_read_text = Path.read_text
        marker_was_opened = False

        def guarded_read_text(path, *args, **kwargs):
            nonlocal marker_was_opened
            if Path(os.path.abspath(path)) == Path(os.path.abspath(malicious)):
                marker_was_opened = True
                raise AssertionError("unauthorized marker was opened")
            return original_read_text(path, *args, **kwargs)

        module._load_plan = lambda *_args, **_kwargs: plan
        module._git_commit_and_clean = lambda *_args, **_kwargs: "4" * 40
        module.validate_phase31_synthetic_pass = lambda *_args, **_kwargs: {
            "file_sha256": "5" * 64
        }
        Path.read_text = guarded_read_text
        args = SimpleNamespace(
            config=plan.config_path,
            expected_experiment=plan.experiment,
            synthetic_pass=root / "SYNTHETIC_PASS.json",
            verify_config=ROOT / "config/Verify_LongArcHorizon_1.1.yaml",
            portable_population_pass=malicious,
            cache_root=root / "cache",
            run_dir=root / "phase_b",
        )
        try:
            try:
                module._audit_train_coverage(args)
            except ValueError as error:
                assert "outside the frozen output root" in str(error)
            else:
                raise AssertionError("Phase B accepted an unauthorized marker path")
        finally:
            Path.read_text = original_read_text
            for name, value in originals.items():
                setattr(module, name, value)
        assert marker_was_opened is False


def test_phase31_train_population_gate_rejects_old_commit_before_marker_open():
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    commit = "4" * 40
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        marker = _write_phase31_portable_population_marker(
            root,
            plan,
            scope="train-only",
            commit=commit,
            synthetic_pass_file_sha256="5" * 64,
        )
        rows = [
            {
                "dataset": dataset,
                "split": "train",
                "source_ordinal": ordinal,
                "config_sha256": plan.config_sha256,
                "dataset_registry_sha256": plan.dataset_registry_sha256,
                "portable_builder_git_commit": commit,
                "cache_builder_git_commit": commit,
                **marker,
            }
            for dataset in plan.train_datasets
            for ordinal in range(plan.source_count)
        ]
        rows[-1] = {**rows[-1], "cache_builder_git_commit": "3" * 40}
        marker_path = Path(marker["portable_population_pass_path"])
        original_stat = Path.stat
        marker_was_statted = False

        def guarded_stat(path, *args, **kwargs):
            nonlocal marker_was_statted
            if Path(os.path.abspath(path)) == Path(os.path.abspath(marker_path)):
                marker_was_statted = True
                raise AssertionError("marker opened before sidecar provenance gate")
            return original_stat(path, *args, **kwargs)

        Path.stat = guarded_stat
        try:
            try:
                validate_phase31_cache_portable_population_evidence(
                    plan,
                    rows,
                    usage="train-coverage",
                    expected_git_commit=commit,
                    synthetic_pass_file_sha256="5" * 64,
                    authorized_marker_paths_by_scope={
                        "train-only": marker_path
                    },
                )
            except ValueError as error:
                assert "sidecar provenance changed" in str(error)
            else:
                raise AssertionError("old-commit train cache sidecar was accepted")
        finally:
            Path.stat = original_stat
        assert marker_was_statted is False


def test_phase31_evaluation_requires_train_and_all_marker_singletons():
    production = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    commit = "4" * 40
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = replace(production, output_root=str(root / "output"))
        base = Path(plan.output_root) / "verification" / "portable_population"
        train_root = base / "train_only"
        all_root = base / "all"
        train_root.mkdir(parents=True)
        all_root.mkdir(parents=True)
        train_marker = _write_phase31_portable_population_marker(
            train_root,
            plan,
            scope="train-only",
            commit=commit,
            synthetic_pass_file_sha256="5" * 64,
        )
        all_marker = _write_phase31_portable_population_marker(
            all_root,
            plan,
            scope="all",
            commit=commit,
            synthetic_pass_file_sha256="5" * 64,
            train_coverage_pass_file_sha256="6" * 64,
        )
        rows = []
        for dataset in plan.datasets:
            split = plan.split_for(dataset)
            evidence = train_marker if split == "train" else all_marker
            for ordinal in range(plan.source_count):
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "source_ordinal": ordinal,
                        "config_sha256": plan.config_sha256,
                        "dataset_registry_sha256": plan.dataset_registry_sha256,
                        "portable_builder_git_commit": commit,
                        "cache_builder_git_commit": commit,
                        **evidence,
                    }
                )
        evidence = validate_phase31_cache_portable_population_evidence(
            plan,
            rows,
            usage="evaluation",
            expected_git_commit=commit,
            synthetic_pass_file_sha256="5" * 64,
            train_coverage_pass_file_sha256="6" * 64,
        )
        assert {row["access_scope"] for row in evidence} == {"train-only", "all"}

        for split, scope_root, scope in (
            ("train", train_root, "train-only"),
            ("test", all_root, "all"),
        ):
            alternate_root = scope_root / "alternate"
            alternate_root.mkdir()
            alternate = _write_phase31_portable_population_marker(
                alternate_root,
                plan,
                scope=scope,
                commit=commit,
                synthetic_pass_file_sha256="5" * 64,
                train_coverage_pass_file_sha256="6" * 64,
            )
            mutated = [dict(row) for row in rows]
            index = next(
                index
                for index, row in enumerate(mutated)
                if row["split"] == split
            )
            mutated[index].update(alternate)
            try:
                validate_phase31_cache_portable_population_evidence(
                    plan,
                    mutated,
                    usage="evaluation",
                    expected_git_commit=commit,
                    synthetic_pass_file_sha256="5" * 64,
                    train_coverage_pass_file_sha256="6" * 64,
                )
            except ValueError as error:
                assert "must be singleton" in str(error)
            else:
                raise AssertionError(f"evaluation accepted multiple {scope} markers")


def test_phase31_build_all_scope_requires_coverage_even_for_train_target():
    spec = importlib.util.spec_from_file_location(
        "phase31_runner_all_scope_test",
        ROOT / "scripts/run_mainexp_template_matching_2_1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    originals = {
        name: getattr(module, name)
        for name in (
            "_load_plan",
            "_git_commit_and_clean",
            "validate_phase31_synthetic_pass",
            "PortableManifestIndex",
        )
    }
    module._load_plan = lambda *_args, **_kwargs: plan
    module._git_commit_and_clean = lambda *_args, **_kwargs: "4" * 40
    module.validate_phase31_synthetic_pass = lambda *_args, **_kwargs: {}
    module.PortableManifestIndex = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("portable manifests were opened before the Phase B gate")
    )
    args = SimpleNamespace(
        config=plan.config_path,
        expected_experiment=plan.experiment,
        synthetic_pass=Path("SYNTHETIC_PASS.json"),
        train_coverage_pass=None,
        verify_config=ROOT / "config/Verify_LongArcHorizon_1.1.yaml",
        access_scope="all",
        portable_manifest=None,
        portable_root=Path("portable"),
        dataset=plan.train_datasets[0],
        ordinal=0,
    )
    try:
        try:
            module._build_slice(args)
        except ValueError as error:
            assert "access-scope=all requires --train-coverage-pass" in str(error)
        else:
            raise AssertionError("all-scope train build bypassed Phase B")
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def test_phase31_build_requires_population_pass_before_manifest_open():
    spec = importlib.util.spec_from_file_location(
        "phase31_runner_population_gate_test",
        ROOT / "scripts/run_mainexp_template_matching_2_1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    originals = {
        name: getattr(module, name)
        for name in (
            "_load_plan",
            "_git_commit_and_clean",
            "validate_phase31_synthetic_pass",
            "PortableManifestIndex",
        )
    }
    module._load_plan = lambda *_args, **_kwargs: plan
    module._git_commit_and_clean = lambda *_args, **_kwargs: "4" * 40
    module.validate_phase31_synthetic_pass = lambda *_args, **_kwargs: {
        "file_sha256": "5" * 64
    }
    module.PortableManifestIndex = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("portable manifests were opened before the population PASS")
    )
    args = SimpleNamespace(
        config=plan.config_path,
        expected_experiment=plan.experiment,
        synthetic_pass=Path("SYNTHETIC_PASS.json"),
        train_coverage_pass=None,
        portable_population_pass=None,
        verify_config=ROOT / "config/Verify_LongArcHorizon_1.1.yaml",
        access_scope="train-only",
        portable_manifest=None,
        portable_root=Path("portable"),
        dataset=plan.train_datasets[0],
        ordinal=0,
    )
    try:
        try:
            module._build_slice(args)
        except ValueError as error:
            assert "--portable-population-pass" in str(error)
        else:
            raise AssertionError("train build bypassed the portable population gate")
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def test_phase31_portable_population_marker_authenticates_complete_train_scope():
    spec = importlib.util.spec_from_file_location(
        "phase31_runner_population_marker_test",
        ROOT / "scripts/run_mainexp_template_matching_2_1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    commit = "4" * 40
    synthetic = {"file_sha256": "5" * 64}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        portable_root = root / "portable"
        portable_root.mkdir()
        rows = [
            {
                "dataset": dataset,
                "split": "train",
                "source_ordinal": ordinal,
                "source_start_index": ordinal,
                "relative_path": f"{dataset}/window_{ordinal}.npz",
                "file_size": 1,
                "file_sha256": "6" * 64,
                "manifest_relative_path": f"{dataset}/manifest.json",
                "manifest_file_sha256": "7" * 64,
                "portable_metadata_sha256": "8" * 64,
            }
            for dataset in plan.train_datasets
            for ordinal in range(plan.source_count)
        ]
        marker_dir = root / "preflight"
        marker_dir.mkdir()
        marker = {
            "schema": "pathline_template_matching.phase31_portable_population_pass.v1",
            "experiment": plan.experiment,
            "status": "passed",
            "access_scope": "train-only",
            "git_commit": commit,
            "worktree_clean": True,
            "config_sha256": plan.config_sha256,
            "dataset_registry_sha256": plan.dataset_registry_sha256,
            "portable_root": str(portable_root.resolve()),
            "dataset_count": 8,
            "window_count": 32,
            "synthetic_pass_file_sha256": synthetic["file_sha256"],
            "train_coverage_pass_file_sha256": None,
            "rows": rows,
            "rows_content_sha256": canonical_json_sha256(rows),
        }
        marker_path = marker_dir / "TRAIN_PORTABLES_PASS.json"
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        evidence = module._validate_portable_population_pass(
            plan,
            marker_path,
            access_scope="train-only",
            portable_root=portable_root,
            current_git_commit=commit,
            synthetic_evidence=synthetic,
            coverage_evidence=None,
        )
        assert evidence["file_sha256"] == sha256_file(marker_path)
        assert len(evidence["rows"]) == 32
        # The marker validator deliberately does not open any window; the 32
        # paths do not exist. Population files were already checked by preflight.
        assert not any((portable_root / row["relative_path"]).exists() for row in rows)


def test_phase31_train_portable_preflight_loads_all_thirty_two_before_marker():
    spec = importlib.util.spec_from_file_location(
        "phase31_runner_population_preflight_test",
        ROOT / "scripts/run_mainexp_template_matching_2_1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    commit = "4" * 40
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        portable_root = root / "portable"
        indexed_rows = {}
        for dataset in plan.train_datasets:
            dataset_dir = portable_root / dataset
            dataset_dir.mkdir(parents=True)
            manifest_path = dataset_dir / "manifest.json"
            manifest_path.write_bytes(b"manifest")
            rows = []
            for ordinal in range(plan.source_count):
                window = dataset_dir / f"window_{ordinal}.npz"
                window.write_bytes(f"{dataset}:{ordinal}".encode("utf-8"))
                rows.append(
                    {
                        "dataset": dataset,
                        "split": "train",
                        "source_ordinal": ordinal,
                        "source_start_index": ordinal,
                        "file_size": window.stat().st_size,
                        "file_sha256": sha256_file(window),
                        "__manifest_path": str(manifest_path),
                        "__manifest_file_sha256": sha256_file(manifest_path),
                        "_path": window,
                    }
                )
            indexed_rows[dataset] = rows

        class FakeIndex:
            def dataset_rows(self, dataset):
                return indexed_rows[dataset]

            def resolve_path(self, row):
                return row["_path"]

        originals = {
            name: getattr(module, name)
            for name in (
                "_load_plan",
                "_git_commit_and_clean",
                "validate_phase31_synthetic_pass",
                "_portable_manifest_index",
                "load_portable_flow_window",
            )
        }
        loaded = []
        module._load_plan = lambda *_args, **_kwargs: plan
        module._git_commit_and_clean = lambda *_args, **_kwargs: commit
        module.validate_phase31_synthetic_pass = lambda *_args, **_kwargs: {
            "file_sha256": "5" * 64
        }
        module._portable_manifest_index = lambda *_args, **_kwargs: FakeIndex()

        def fake_load(path, **kwargs):
            loaded.append((Path(path), kwargs["expected_dataset"]))
            return SimpleNamespace(
                metadata={
                    "dataset": kwargs["expected_dataset"],
                    "source_start_index": kwargs["expected_source_start_index"],
                }
            )

        module.load_portable_flow_window = fake_load
        run_dir = root / "preflight"
        args = SimpleNamespace(
            config=plan.config_path,
            expected_experiment=plan.experiment,
            synthetic_pass=root / "SYNTHETIC_PASS.json",
            train_coverage_pass=None,
            verify_config=ROOT / "config/Verify_LongArcHorizon_1.1.yaml",
            access_scope="train-only",
            portable_manifest=None,
            portable_root=portable_root,
            run_dir=run_dir,
        )
        try:
            module._preflight_portables(args)
        finally:
            for name, value in originals.items():
                setattr(module, name, value)
        assert len(loaded) == 32
        marker_path = run_dir / "TRAIN_PORTABLES_PASS.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["window_count"] == 32
        assert len(marker["rows"]) == 32
        assert marker["rows_content_sha256"] == canonical_json_sha256(marker["rows"])


def test_phase31_portable_index_train_only_never_requires_test_manifests():
    spec = importlib.util.spec_from_file_location(
        "phase31_runner_scope_test",
        ROOT / "scripts/run_mainexp_template_matching_2_1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    commit = "6" * 40
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for dataset_index, dataset in enumerate(plan.train_datasets):
            dataset_dir = root / dataset
            dataset_dir.mkdir()
            total_frames = 52 + dataset_index
            indices = plan.source_indices(total_frames)
            windows = []
            for ordinal, source_index in enumerate(indices):
                name = f"window_{ordinal}.npz"
                (dataset_dir / name).write_bytes(b"portable")
                windows.append(
                    {
                        "relative_path": name,
                        "file_size": 8,
                        "file_sha256": "7" * 64,
                        "dataset": dataset,
                        "experiment": plan.experiment,
                        "config_sha256": plan.config_sha256,
                        "physical_family": plan.family_by_dataset[dataset],
                        "split": "train",
                        "dataset_registry_sha256": plan.dataset_registry_sha256,
                        "builder_git_commit": commit,
                        "source_kind": "netcdf",
                        "source_file": "train_source.nc",
                        "source_file_size": 10,
                        "source_file_sha256": "8" * 64,
                        "source_total_frames": total_frames,
                        "source_ordinal": ordinal,
                        "source_start_index": source_index,
                        "frame_count": 49,
                    }
                )
            payload = {
                "schema": "pathline_template_matching.portable_flow_dataset_manifest.v1",
                "experiment": plan.experiment,
                "config_sha256": plan.config_sha256,
                "dataset_registry_sha256": plan.dataset_registry_sha256,
                "builder_git_commit": commit,
                "dataset": dataset,
                "physical_family": plan.family_by_dataset[dataset],
                "split": "train",
                "source_kind": "netcdf",
                "source_file": "train_source.nc",
                "source_file_size": 10,
                "source_file_sha256": "8" * 64,
                "source_total_frames": total_frames,
                "selected_source_indices": list(indices),
                "window_count": 4,
                "windows": windows,
            }
            payload["manifest_content_sha256"] = canonical_json_sha256(payload)
            (dataset_dir / "manifest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        index = module.PortableManifestIndex(
            None,
            portable_root=root,
            expected_experiment=plan.experiment,
            expected_config_sha256=plan.config_sha256,
            expected_dataset_registry_sha256=plan.dataset_registry_sha256,
            expected_datasets=plan.train_datasets,
            expected_family_by_dataset=plan.family_by_dataset,
            expected_split_by_dataset={dataset: "train" for dataset in plan.train_datasets},
            expected_builder_git_commit=commit,
            expected_window_frame_count=49,
        )
        assert len(index.rows) == 32
        assert {row["dataset"] for row in index.rows} == set(plan.train_datasets)
        assert not any(dataset in str(row) for dataset in plan.test_datasets for row in index.rows)


def test_phase31_visualization_emits_four_unique_dataset_block_svgs():
    import pathline_template_matching.phase21_pipeline as pipeline

    production = load_phase31_plan(ROOT / "config/mainExp_TemplateMatching_3.1.yaml")
    visualization = dict(production.config["visualization"])
    visualization["png_dpi"] = 10
    plan = SimpleNamespace(
        experiment=production.experiment,
        config={"visualization": visualization},
        test_datasets=production.test_datasets,
        family_by_dataset=production.family_by_dataset,
        effective_scale_blocks=production.effective_scale_blocks,
        scale_table=production.scale_table,
        config_sha256=production.config_sha256,
    )
    center_count = 260
    rng = np.random.Generator(np.random.PCG64(15068))
    base_seeds = rng.uniform(0.03, 0.99, size=(center_count, 3)).astype(np.float64)
    base_seeds[:, 1] = (np.arange(center_count) % 26 + 0.5) / 26.0
    base_seeds[:, 2] = (np.arange(center_count) // 26 + 0.5) / 10.0
    seeds = np.concatenate((base_seeds, base_seeds), axis=0)
    labels_one = np.zeros(center_count, dtype=np.bool_)
    labels_one[center_count // 2 :] = True
    labels = np.concatenate((labels_one, labels_one))
    assigned = np.arange(2 * center_count, dtype=np.int64)
    center = assigned % center_count
    block = (assigned // center_count).astype(np.int8)
    scale = np.concatenate(
        (
            ((np.arange(center_count) * 137) % 1_000),
            1_000 + ((np.arange(center_count) * 137) % 1_000),
        )
    ).astype(np.int32)
    raw = np.zeros((2 * center_count, 672), dtype=np.float32)
    center_time = np.broadcast_to(
        np.linspace(0.0, 0.3, 32, dtype=np.float32),
        (2 * center_count, 32),
    ).copy()
    x = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    ivd = np.broadcast_to(x[None, None, :], (21, 21, 21)).copy()
    caches = {}
    rows = []
    for dataset in production.test_datasets:
        cache = {
            "raw_features": raw,
            "fmt_features": np.zeros((2 * center_count, 161), dtype=np.float32),
            "valid_labels": labels,
            "valid_seed_index": assigned,
            "valid_scale_id": scale,
            "valid_assigned_row_index": assigned,
            "valid_center_seed_index": center,
            "valid_scale_block_index": block,
            "center_sample_time": center_time,
            "seeds_xyz": seeds,
            "ivd_volume": ivd,
            "metadata": {
                "schema": production.cache_schema,
                "experiment": production.experiment,
                "dataset": dataset,
                "split": "test",
                "source_ordinal": 2,
                "source_index": 2,
                "source_time": 4.25,
                "loaded_shape_TZYXC": [49, 21, 21, 21, 3],
                "spacing_xyz": [0.05, 0.05, 0.05],
                "domain_min_xyz": [0.0, 0.0, 0.0],
                "domain_max_xyz": [1.0, 1.0, 1.0],
                "ivd_percentile": 95.0,
                "ivd_threshold": float(np.percentile(ivd, 95.0)),
                "unique_center_seed_count": center_count,
            },
        }
        name = f"{dataset}.npz"
        caches[name] = cache
        rows.append(
            {
                "dataset": dataset,
                "source_ordinal": 2,
                "source_index": 2,
                "path": name,
                "file_sha256": "9" * 64,
            }
        )
    query = {
        "dataset_index": np.concatenate(
            tuple(
                np.full(2 * center_count, index, dtype=np.int16)
                for index in range(2)
            )
        ),
        "source_ordinal": np.full(4 * center_count, 2, dtype=np.int16),
        "valid_seed_index": np.tile(assigned, 2),
        "assigned_row_index": np.tile(assigned, 2),
        "center_seed_index": np.tile(center, 2),
        "scale_block_index": np.tile(block, 2),
        "scale_id": np.tile(scale, 2),
        "labels": np.tile(labels, 2),
    }
    predictions = np.arange(4 * center_count) % 3 == 0
    original_loader = pipeline._load_cache
    original_validator = pipeline._validate_cache_provenance
    pipeline._load_cache = lambda path, expected_sha256=None: caches[path.name]
    pipeline._validate_cache_provenance = lambda *_args, **_kwargs: None
    try:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = pipeline._build_phase21_visualization_artifacts(
                plan,
                root=output,
                test_rows=rows,
                query=query,
                fmt_prediction=predictions,
                git_commit="a" * 40,
                verify_cache_hashes=True,
            )
            assert manifest["entry_count"] == 4
            assert manifest["unique_key"] == ["dataset", "scale_block_id"]
            assert manifest["required_export_count_per_figure"] == 5
            assert manifest["additional_audit_file_count_per_figure"] == 2
            assert manifest["required_export_file_fields"] == [
                "relative_path",
                "export_kind",
                "size_bytes",
                "sha256",
            ]
            keys = {
                (entry["dataset"], entry["scale_block_id"])
                for entry in manifest["entries"]
            }
            assert len(keys) == 4
            for entry in manifest["entries"]:
                assert entry["query_count"] == center_count
                svg = output / entry["svg"]
                assert svg.is_file()
                assert "<text" in svg.read_text(encoding="utf-8")
                assert entry["svg_sha256"] == sha256_file(svg)
                assert len(entry["required_exports"]) == 5
                assert len(entry["additional_audit_files"]) == 2
                assert {
                    row["export_kind"] for row in entry["additional_audit_files"]
                } == {"scene_manifest_json", "render_metadata_json"}
                for export in (
                    entry["required_exports"] + entry["additional_audit_files"]
                ):
                    assert set(export) == {
                        "relative_path",
                        "export_kind",
                        "size_bytes",
                        "sha256",
                    }
                    path = output / export["relative_path"]
                    assert path.is_file()
                    assert export["size_bytes"] == path.stat().st_size
                    assert export["sha256"] == sha256_file(path)
    finally:
        pipeline._load_cache = original_loader
        pipeline._validate_cache_provenance = original_validator

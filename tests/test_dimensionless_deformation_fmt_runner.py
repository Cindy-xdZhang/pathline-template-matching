from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch

import numpy as np

from pathline_template_matching.portable_flow import (
    canonical_array_sha256,
    sha256_file,
)
from scripts import run_verify_dimensionless_deformation_fmt_1_1 as runner
from scripts import run_verify_per_scale_negative_metric_1_1 as inherited
from scripts.run_verify_scale_conditioned_retrieval_1_1 import CacheRow


_OFFSETS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

_PATCHED_PARENT_GLOBAL_NAMES = (
    "EXPERIMENT",
    "EXPECTED_CONFIG_SHA256",
    "REPRESENTATIONS",
    "FROZEN_CANDIDATE_COUNT",
    "SCALER_ARTIFACT_SCHEMA",
    "SCALER_MANIFEST_SCHEMA",
    "CALIBRATION_ARTIFACT_SCHEMA",
    "CALIBRATION_MANIFEST_SCHEMA",
    "SELECTED_SCHEMA",
    "PREDICTION_SCHEMA",
    "PREDICTION_MANIFEST_SCHEMA",
    "RESULT_SCHEMA",
    "COMPLETE_SCHEMA",
    "representation_features",
    "load_cache_projection",
    "load_cache_rows",
    "load_plan",
    "_atomic_csv",
    "_atomic_json",
    "_atomic_npz",
    "_manifest_with_self_hash",
    "_authenticate_self_hash",
    "_outer_summary",
    "_git_identity",
)


def _expect_failure(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except (AssertionError, FileExistsError, RuntimeError, ValueError):
        return
    raise AssertionError("expected fail-closed exception")


def _enter_parent_runtime_once(plan: runner.Plan, commit: str) -> None:
    with runner.dimensionless_parent_runtime(plan, commit):
        pass


def _raw672(row_count: int) -> np.ndarray:
    time = np.arange(32, dtype=np.float64)
    output = np.empty((row_count, 7, 32, 3), dtype=np.float64)
    for row_index in range(row_count):
        class_like = row_index % 4 >= 2
        center = np.stack(
            (
                (0.02 + 0.003 * row_index) * time,
                (0.0002 + 0.0005 * class_like) * np.square(time),
                0.01 * np.sin(time / (5.0 + row_index)),
            ),
            axis=1,
        )
        center[0] = 0.0
        output[row_index, 0] = center
        dx = 0.125 + 0.01 * row_index
        for neighbor_index, offset in enumerate(_OFFSETS):
            deformation = dx * offset[None, :] * (
                1.0 + (0.001 + 0.006 * class_like) * time[:, None]
            )
            deformation += (
                class_like
                * 0.0003
                * (neighbor_index + 1)
                * time[:, None]
                * np.asarray([0.2, -0.1, 0.05])[None]
            )
            output[row_index, neighbor_index + 1] = center + deformation
    return np.ascontiguousarray(output.reshape(row_count, 672), dtype=np.float32)


def _write_cache(
    root: Path,
    plan: runner.Plan,
    *,
    dataset: str,
    family: str,
    ordinal: int,
    row_count: int = 8,
) -> CacheRow:
    raw = _raw672(row_count)
    per_block = row_count // 2
    scales = np.asarray([0] * per_block + [1000] * per_block, dtype=np.int32)
    centers = np.asarray(list(range(per_block)) * 2, dtype=np.int64)
    blocks = np.asarray([0] * per_block + [1] * per_block, dtype=np.int8)
    assigned = blocks.astype(np.int64) * 64000 + centers
    labels = np.asarray(([False, False, True, True] * 2)[:row_count], dtype=np.bool_)
    members = {
        "raw_features": raw,
        "valid_scale_id": scales,
        "valid_center_seed_index": centers,
        "valid_scale_block_index": blocks,
        "valid_assigned_row_index": assigned,
        "valid_labels": labels,
    }
    metadata = {
        "schema": plan.cache_schema,
        "experiment": "mainExp_TemplateMatching_3.1",
        "split": "train",
        "dataset": dataset,
        "physical_family": family,
        "source_ordinal": ordinal,
        "source_index": ordinal * 3,
        "config_sha256": plan.parent_config_sha256,
        "cache_builder_git_commit": plan.cache_commit,
        "valid_count": row_count,
        "array_sha256": {
            name: canonical_array_sha256(values) for name, values in members.items()
        },
    }
    path = root / f"{dataset}_{ordinal}.npz"
    np.savez_compressed(
        path,
        **members,
        # Deliberately invalid as a descriptor; the runner must never request it.
        fmt_features=np.full((row_count, 2), np.nan, dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    )
    return CacheRow(
        dataset=dataset,
        family=family,
        source_ordinal=ordinal,
        source_index=ordinal * 3,
        path=path,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def _write_complete_synthetic_population(
    root: Path, plan: runner.Plan
) -> list[CacheRow]:
    rows: list[CacheRow] = []
    for family in plan.family_order:
        for dataset in plan.families[family]:
            for ordinal in range(4):
                rows.append(
                    _write_cache(
                        root,
                        plan,
                        dataset=dataset,
                        family=family,
                        ordinal=ordinal,
                    )
                )
    assert len(rows) == 32
    return rows


def test_dimensionless_runner_authenticates_active_parent_core_and_3060_contract() -> None:
    plan = runner.load_plan()
    assert plan.sha256 == runner.EXPECTED_CONFIG_SHA256
    assert plan.parent_experiment_config_sha256 == runner.EXPECTED_PARENT_CONFIG_SHA256
    assert plan.core_sha256 == runner.EXPECTED_CORE_SHA256
    assert plan.encode_chunk_rows == 4096
    assert plan.required_fold_files == runner.REQUIRED_FOLD_FILES
    assert len(plan.required_fold_files) == 15
    assert len(runner.candidate_specs(plan)) == 3060
    assert tuple(plan.representations) == runner.REPRESENTATIONS
    binding = runner._method_binding(plan, "a" * 40)
    assert binding["experiment_config"]["sha256"] == runner.EXPECTED_CONFIG_SHA256
    assert binding["parent_per_scale_config"]["sha256"] == runner.EXPECTED_PARENT_CONFIG_SHA256
    assert binding["core"]["sha256"] == runner.EXPECTED_CORE_SHA256
    assert binding["input_member"] == "raw_features"
    assert binding["forbidden_cache_member"] == "fmt_features"

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        active = root / "active.yaml"
        active.write_bytes(runner.CONFIG_PATH.read_bytes() + b"\n")
        _expect_failure(runner.load_plan, active)
        parent = root / "parent.yaml"
        parent.write_bytes(runner.PARENT_CONFIG_PATH.read_bytes() + b"\n")
        with patch.object(runner, "PARENT_CONFIG_PATH", parent):
            _expect_failure(runner.load_plan, runner.CONFIG_PATH)


def test_raw_only_projection_fixed_chunks_and_three_parent_subsets_are_exact() -> None:
    plan = runner.load_plan()
    raw = _raw672(9)
    fixed = runner.encode_raw_features_in_fixed_chunks(raw)
    for chunk_rows in (1, 2, 4, 32):
        np.testing.assert_array_equal(
            fixed,
            runner.encode_raw_features_with_chunk_rows_for_test(raw, chunk_rows),
        )
    expected_widths = (161, 36, 35)
    for name, width in zip(runner.REPRESENTATIONS, expected_widths):
        observed = runner.dimensionless_representation_features(fixed, name)
        np.testing.assert_array_equal(
            observed,
            fixed[:, runner.PARENT_REPRESENTATION_INDEX_SETS[name]],
        )
        assert observed.shape == (9, width)

    with tempfile.TemporaryDirectory() as temporary:
        row = _write_cache(
            Path(temporary),
            plan,
            dataset="channel",
            family="channel",
            ordinal=0,
        )
        label_free = runner.load_dimensionless_projection(
            plan, row, include_labels=False
        )
        assert label_free.fmt_features.shape == (8, 161)
        assert label_free.labels is None and not label_free.metadata
        assert label_free.opened_members == (
            "raw_features",
            "valid_scale_id",
            "valid_center_seed_index",
            "valid_scale_block_index",
            "valid_assigned_row_index",
        )
        assert "fmt_features" not in label_free.opened_members
        assert not hasattr(label_free, "raw_features")
        referenced = runner.load_dimensionless_projection(
            plan, row, include_labels=True
        )
        assert referenced.labels is not None
        np.testing.assert_array_equal(label_free.fmt_features, referenced.fmt_features)
        _expect_failure(runner._open_cache_members, row, ("fmt_features",))


def test_production_4096_chunk_boundary_is_bitwise_invariant() -> None:
    base = _raw672(1)
    raw = np.ascontiguousarray(np.repeat(base, 4097, axis=0), dtype=np.float32)
    production = runner.encode_raw_features_in_fixed_chunks(raw)
    alternate = runner.encode_raw_features_with_chunk_rows_for_test(raw, 2048)
    assert production.shape == alternate.shape == (4097, 161)
    assert production.dtype == alternate.dtype == np.dtype(np.float32)
    assert production.flags.c_contiguous and alternate.flags.c_contiguous
    np.testing.assert_array_equal(production, alternate)


def test_runtime_exception_restores_every_parent_global_and_binds_manifests() -> None:
    plan = runner.load_plan()
    commit = "b" * 40
    before = {
        name: getattr(inherited, name) for name in _PATCHED_PARENT_GLOBAL_NAMES
    }
    try:
        with patch.object(runner, "_git_identity", return_value=(commit, False)):
            with runner.dimensionless_parent_runtime(plan, commit):
                manifest = inherited._manifest_with_self_hash(
                    {"schema": "synthetic.v1", "experiment": runner.EXPERIMENT}
                )
                assert manifest[runner.METHOD_BINDING_KEY] == runner._method_binding(
                    plan, commit
                )
                inherited._authenticate_self_hash(manifest)
                raise RuntimeError("synthetic exception")
    except RuntimeError as error:
        assert str(error) == "synthetic exception"
    else:
        raise AssertionError("expected synthetic exception")
    after = {
        name: getattr(inherited, name) for name in _PATCHED_PARENT_GLOBAL_NAMES
    }
    assert all(
        after[name] is before[name] for name in _PATCHED_PARENT_GLOBAL_NAMES
    )


def test_production_load_plan_uses_stable_parent_identity_inside_runtime() -> None:
    """Reproduce the real inherited.run -> rebound load_plan call path."""

    plan = runner.load_plan()
    commit = "9" * 40
    assert runner.PARENT_EXPERIMENT == inherited.EXPERIMENT
    assert runner.PARENT_EXPERIMENT != runner.EXPERIMENT
    with runner.dimensionless_parent_runtime(plan, commit):
        assert inherited.EXPERIMENT == runner.EXPERIMENT
        rebound = inherited.load_plan(runner.CONFIG_PATH)
        assert rebound.sha256 == plan.sha256
        assert rebound.parent_experiment_config_sha256 == (
            plan.parent_experiment_config_sha256
        )
    assert inherited.EXPERIMENT == runner.PARENT_EXPERIMENT


def test_parent_patch_install_failure_restores_all_attempts_and_releases_lock() -> None:
    plan = runner.load_plan()
    commit = "e" * 40
    before = {
        name: getattr(inherited, name) for name in _PATCHED_PARENT_GLOBAL_NAMES
    }
    real_setter = runner._set_inherited_global
    call_count = 0

    def fail_on_fifth_set(name: str, value: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 5:
            # Exercise the strongest failure mode: the setter changes the
            # binding and only then reports failure.
            real_setter(name, value)
            raise RuntimeError("synthetic fifth-set failure")
        real_setter(name, value)

    with patch.object(
        runner, "_set_inherited_global", side_effect=fail_on_fifth_set
    ):
        _expect_failure(_enter_parent_runtime_once, plan, commit)
    assert call_count == 10  # five install attempts plus five reverse restores
    after = {
        name: getattr(inherited, name) for name in _PATCHED_PARENT_GLOBAL_NAMES
    }
    assert all(
        after[name] is before[name] for name in _PATCHED_PARENT_GLOBAL_NAMES
    )
    # A second transaction proves the failed installer released its token.
    _enter_parent_runtime_once(plan, commit)


def test_parent_runtime_rejects_nested_and_concurrent_use_then_recovers() -> None:
    plan = runner.load_plan()
    commit = "f" * 40
    before = {
        name: getattr(inherited, name) for name in _PATCHED_PARENT_GLOBAL_NAMES
    }
    concurrent_errors: list[BaseException] = []

    def concurrent_attempt() -> None:
        try:
            _enter_parent_runtime_once(plan, commit)
        except BaseException as error:
            concurrent_errors.append(error)

    with runner.dimensionless_parent_runtime(plan, commit):
        _expect_failure(_enter_parent_runtime_once, plan, commit)
        worker = threading.Thread(target=concurrent_attempt)
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert len(concurrent_errors) == 1
        assert isinstance(concurrent_errors[0], ValueError)
        assert "nested or concurrent use is forbidden" in str(concurrent_errors[0])

    after = {
        name: getattr(inherited, name) for name in _PATCHED_PARENT_GLOBAL_NAMES
    }
    assert all(
        after[name] is before[name] for name in _PATCHED_PARENT_GLOBAL_NAMES
    )
    _enter_parent_runtime_once(plan, commit)


def test_hard_link_publication_refuses_overwrite_and_preserves_winner() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "artifact.json"
        first_sha = runner._atomic_json(destination, {"winner": 1})
        first_bytes = destination.read_bytes()
        assert hashlib.sha256(first_bytes).hexdigest() == first_sha
        _expect_failure(runner._atomic_json, destination, {"winner": 2})
        assert destination.read_bytes() == first_bytes
        assert not list(destination.parent.glob("*.partial"))


def test_synthetic_complete_fold_writes_15_bound_files_and_fresh_replays() -> None:
    plan = runner.load_plan()
    # Keep all three representations and the complete family/source scope while
    # reducing only the synthetic test's operational candidate grid to 3.
    mini = replace(plan, ks=(1,), sigmas=(0.0,), thresholds=())
    commit = "c" * 40
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        cache_root = root / "primitive_cache" / "train"
        cache_root.mkdir(parents=True)
        rows = _write_complete_synthetic_population(cache_root, mini)
        destination = root / "fold"
        with (
            patch.object(runner, "load_plan", return_value=mini),
            patch.object(
                runner,
                "load_cache_rows",
                return_value=(
                    rows,
                    {
                        "schema": mini.manifest_schema,
                        "sha256": mini.manifest_sha256,
                        "rows_content_sha256": mini.manifest_rows_sha256,
                        "row_count": 32,
                    },
                ),
            ),
            patch.object(runner, "_git_identity", return_value=(commit, False)),
        ):
            result = runner.run(
                runner.CONFIG_PATH,
                "half_cylinder",
                destination,
                device="cpu",
            )
        assert set(path.name for path in destination.iterdir()) == set(
            runner.REQUIRED_FOLD_FILES
        )
        assert result["schema"] == runner.RESULT_SCHEMA
        binding = runner._method_binding(mini, commit)
        assert result[runner.METHOD_BINDING_KEY] == binding
        for name in (
            "inner_fit_audits.json",
            "final_per_scale_scaler_manifest.json",
            "final_tail_calibration_manifest.json",
            "selected_candidate.json",
            "outer_prediction_manifest.json",
            "outer_summary.json",
            "outer_reference_access_audit.json",
            "result_manifest.json",
            "RUN_COMPLETE.json",
        ):
            value = json.loads((destination / name).read_text(encoding="utf-8"))
            assert value[runner.METHOD_BINDING_KEY] == binding
        prediction_manifest = json.loads(
            (destination / "outer_prediction_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert prediction_manifest["valid_labels_opened"] is False
        assert prediction_manifest["metadata_json_opened"] is False
        assert prediction_manifest["prediction_schema"] == runner.PREDICTION_SCHEMA
        with np.load(destination / "outer_predictions.npz", allow_pickle=False) as archive:
            assert set(archive.files) == set(runner.PREDICTION_ARRAY_DTYPES)
        reference = json.loads(
            (destination / "outer_reference_access_audit.json").read_text(
                encoding="utf-8"
            )
        )
        assert reference["schema"] == runner.REFERENCE_AUDIT_SCHEMA
        assert reference["first_open_phase"] == (
            "after_outer_prediction_file_and_manifest_authentication"
        )


def test_outer_fresh_raw_replay_tamper_fails_before_any_outer_label_open() -> None:
    plan = runner.load_plan()
    mini = replace(plan, ks=(1,), sigmas=(0.0,), thresholds=())
    commit = "d" * 40
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        cache_root = root / "primitive_cache" / "train"
        cache_root.mkdir(parents=True)
        rows = _write_complete_synthetic_population(cache_root, mini)
        destination = root / "tampered_fold"
        original_loader = runner.load_dimensionless_projection
        outer_label_free_count = 0
        outer_label_open_count = 0

        def tampered_loader(
            selected_plan: runner.Plan,
            row: CacheRow,
            *,
            include_labels: bool,
        ) -> runner.DimensionlessCacheProjection:
            nonlocal outer_label_free_count, outer_label_open_count
            projection = original_loader(
                selected_plan, row, include_labels=include_labels
            )
            if row.family == "half_cylinder":
                if include_labels:
                    outer_label_open_count += 1
                else:
                    outer_label_free_count += 1
                    # Twelve first-pass outer shards precede the fresh replay.
                    if outer_label_free_count == 13:
                        modified = np.array(
                            projection.fmt_features, copy=True, order="C"
                        )
                        modified[0] += np.linspace(
                            0.25, 1.25, 161, dtype=np.float32
                        )
                        projection = replace(projection, fmt_features=modified)
            return projection

        with (
            patch.object(runner, "load_plan", return_value=mini),
            patch.object(
                runner,
                "load_cache_rows",
                return_value=(rows, {"schema": mini.manifest_schema, "row_count": 32}),
            ),
            patch.object(runner, "_git_identity", return_value=(commit, False)),
            patch.object(
                runner, "load_dimensionless_projection", side_effect=tampered_loader
            ),
        ):
            _expect_failure(
                runner.run,
                runner.CONFIG_PATH,
                "half_cylinder",
                destination,
                device="cpu",
            )
        assert outer_label_free_count >= 13
        assert outer_label_open_count == 0
        assert not (destination / "outer_reference_access_audit.json").exists()
        assert not (destination / "RUN_COMPLETE.json").exists()

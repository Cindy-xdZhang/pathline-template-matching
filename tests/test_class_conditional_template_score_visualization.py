from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.negative_tail_visualization import (  # noqa: E402
    SCENE31_ARRAY_NAMES,
    PredictionGroup,
    exact_bind_prediction_group,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)
from scripts import (  # noqa: E402
    render_class_conditional_template_score_visualizations as report,
)


COMMIT = "1" * 40
CONFIG_SHA256 = "2" * 64
PARENT_CONFIG_SHA256 = "3" * 64
PARENT_RESULT_SHA256 = "4" * 64
PARENT_COMPLETE_SHA256 = "5" * 64


def _expect_value_error(function, *args, contains: str, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert contains in str(error), str(error)
        return
    raise AssertionError("expected ValueError")


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _write_self_hashed(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str = "content_sha256",
) -> tuple[dict[str, Any], str]:
    value = dict(payload)
    value.pop(field, None)
    value[field] = canonical_json_sha256(value)
    return value, _write_json(path, value)


def _method_binding() -> dict[str, Any]:
    return {
        "experiment": report.PREDICTION_EXPERIMENT,
        "config": {"sha256": CONFIG_SHA256},
        "score": {
            "combine": "equal_mean_over_jointly_supported_families",
            "inner_support": "2_of_3_joint_families",
            "outer_support": "3_of_4_joint_families",
        },
        "threshold": {
            "comparison": "strict_greater_than",
            "equality_prediction": "negative",
        },
        "prediction_array_contract": "unchanged_parent_19_arrays",
        "fold_transaction": "unchanged_parent_15_files",
        "numerical_git_commit": COMMIT,
    }


def _candidate() -> dict[str, Any]:
    return {
        "candidate_id": (
            "representation=chirality_all35|k=1|sigma=0.0|"
            "fixed_top_fraction=0.05"
        ),
        "representation": "chirality_all35",
        "k": 1,
        "sigma": 0.0,
        "decision_rule": "fixed_top_fraction",
        "decision_value": 0.05,
    }


def _family_datasets(family: str) -> tuple[str, ...]:
    return {
        "half_cylinder": (
            "cylinder3d",
            "halfcylinderRe640",
            "halfcylinderRe6400",
        ),
        "delta_wing": ("deltaWing", "deltaWing2"),
        "f22_raptor": ("f22",),
        "channel": ("channel",),
        "boeing_747": ("boeing747",),
    }[family]


def _prediction_arrays(
    family: str,
    *,
    tamper_assigned: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rows: dict[str, list[Any]] = {name: [] for name in report.PREDICTION_ARRAY_NAMES}
    group_audits: list[dict[str, Any]] = []
    fit_families = [value for value in report.FAMILY_ORDER if value != family]
    for dataset_index, dataset in enumerate(_family_datasets(family)):
        for source_ordinal in range(4):
            source_index = dataset_index * 100 + source_ordinal + 7
            for block, block_index in report.BLOCK_INDEX.items():
                scores = (0.25, 0.75)
                for center, score in enumerate(scores):
                    rows["dataset"].append(dataset)
                    rows["source_ordinal"].append(source_ordinal)
                    rows["source_index"].append(source_index)
                    rows["scale_id"].append(block_index * 1000 + center)
                    rows["center_seed_index"].append(center)
                    assigned = center + block_index * report.ASSIGNED_PER_BLOCK
                    if tamper_assigned and dataset_index == 0 and source_ordinal == 0 and block_index == 0 and center == 0:
                        assigned += 1
                    rows["scale_block_index"].append(block_index)
                    rows["assigned_row_index"].append(assigned)
                    rows["raw_negative_distance"].append(1.0 - score)
                    rows["tail_probability"].append(1.0 - score)
                    rows["tail_anomaly"].append(score)
                    rows["spatial_score"].append(score)
                    rows["spatial_denominator"].append(4.0)
                    rows["retrieval_supported"].append(True)
                    rows["calibration_supported"].append(True)
                    rows["spatial_imputed"].append(False)
                    rows["spatial_unimputable"].append(False)
                    rows["calibration_mode"].append(1)
                    rows["scaler_mode"].append(1)
                    rows["prediction"].append(score > 0.5)
                support = {
                    "sample_count": 2,
                    "family_order": fit_families,
                    "required_joint_family_count": 3,
                    "joint_supported_family_count_histogram": {
                        "0": 0,
                        "1": 0,
                        "2": 0,
                        "3": 0,
                        "4": 2,
                    },
                    "families": {name: {} for name in fit_families},
                }
                group_audits.append(
                    {
                        "dataset": dataset,
                        "source_ordinal": source_ordinal,
                        "source_index": source_index,
                        "block": block,
                        "sample_count": 2,
                        "retrieval_supported_count": 2,
                        "calibration_supported_count": 2,
                        "imputed_count": 0,
                        "unimputable_count": 0,
                        "prediction_count": 1,
                        "class_conditional_support": support,
                    }
                )
    arrays = {
        name: np.asarray(rows[name], dtype=np.dtype(report.PREDICTION_DTYPES[name]))
        for name in report.PREDICTION_ARRAY_NAMES
    }
    return arrays, group_audits


def _identity(path: Path) -> dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _write_fold(
    root: Path,
    family: str,
    *,
    tamper_assigned: bool = False,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    binding = _method_binding()
    candidate = _candidate()
    arrays, group_audits = _prediction_arrays(
        family,
        tamper_assigned=tamper_assigned,
    )
    prediction_path = root / "outer_predictions.npz"
    np.savez(prediction_path, **arrays)

    selected_payload = {
        "schema": report.PREDICTION_SELECTED_SCHEMA,
        "experiment": report.PREDICTION_EXPERIMENT,
        "git_commit": COMMIT,
        "config_sha256": CONFIG_SHA256,
        "outer_family": family,
        "candidate": candidate,
        report.METHOD_BINDING_KEY: binding,
    }
    _write_self_hashed(root / "selected_candidate.json", selected_payload)

    prediction_payload = {
        "schema": report.PREDICTION_MANIFEST_SCHEMA,
        "prediction_schema": report.PREDICTION_SCHEMA,
        "experiment": report.PREDICTION_EXPERIMENT,
        "git_commit": COMMIT,
        "config_sha256": CONFIG_SHA256,
        "outer_family": family,
        "selected_candidate": candidate,
        "valid_labels_opened": False,
        "metadata_json_opened": False,
        "array_count": len(report.PREDICTION_ARRAY_NAMES),
        "row_count": len(arrays["dataset"]),
        "arrays": {
            name: {
                "dtype": arrays[name].dtype.str,
                "shape": list(arrays[name].shape),
                "sha256": canonical_array_sha256(arrays[name]),
            }
            for name in report.PREDICTION_ARRAY_NAMES
        },
        "prediction_file": {
            "path": prediction_path.name,
            **_identity(prediction_path),
        },
        "group_audits": group_audits,
        report.METHOD_BINDING_KEY: binding,
    }
    _write_self_hashed(root / "outer_prediction_manifest.json", prediction_payload)

    for name in report.FOLD_ARTIFACT_NAMES:
        path = root / name
        if path.exists():
            continue
        path.write_bytes(f"synthetic {family} {name}\n".encode("utf-8"))
    artifacts = {name: _identity(root / name) for name in report.FOLD_ARTIFACT_NAMES}
    result_payload = {
        "schema": report.PREDICTION_RESULT_SCHEMA,
        "experiment": report.PREDICTION_EXPERIMENT,
        "status": "completed",
        "git_commit": COMMIT,
        "config_sha256": CONFIG_SHA256,
        "outer_family": family,
        "selected_candidate": candidate,
        "artifacts": artifacts,
        report.METHOD_BINDING_KEY: binding,
    }
    result, result_file_sha = _write_self_hashed(
        root / "result_manifest.json",
        result_payload,
    )
    complete_payload = {
        "schema": report.PREDICTION_COMPLETE_SCHEMA,
        "experiment": report.PREDICTION_EXPERIMENT,
        "git_commit": COMMIT,
        "config_sha256": CONFIG_SHA256,
        "outer_family": family,
        "result_manifest_file": "result_manifest.json",
        "result_manifest_file_sha256": result_file_sha,
        "result_manifest_content_sha256": result["content_sha256"],
        report.METHOD_BINDING_KEY: binding,
    }
    _, completion_file_sha = _write_self_hashed(
        root / "RUN_COMPLETE.json",
        complete_payload,
    )
    assert {path.name for path in root.iterdir()} == set(report.FOLD_FILE_NAMES)
    return {
        "outer_family": family,
        "run_directory": str(root.resolve()),
        "completion_file_sha256": completion_file_sha,
        "result_manifest_file_sha256": result_file_sha,
        "artifact_count": len(report.FOLD_ARTIFACT_NAMES),
        "artifacts": artifacts,
    }


def _plan(tmp_path: Path, aggregate_root: Path) -> report.ReportPlan:
    return report.ReportPlan(
        path=tmp_path / "synthetic_report_config.yaml",
        sha256="6" * 64,
        raw={},
        prediction_commit=COMMIT,
        prediction_config_sha256=CONFIG_SHA256,
        aggregate_root=aggregate_root,
        aggregate_complete_sha256=sha256_file(
            aggregate_root / "AGGREGATE_COMPLETE.json"
        ),
        parent_root=tmp_path / "parent",
        parent_experiment="Other_SyntheticParent_1.1",
        parent_commit="7" * 40,
        parent_config_sha256=PARENT_CONFIG_SHA256,
        parent_result_manifest_sha256=PARENT_RESULT_SHA256,
        parent_run_complete_sha256=PARENT_COMPLETE_SHA256,
        datasets=tuple(
            report.DatasetSpec(
                dataset,
                report.DISPLAY_NAMES[dataset],
                report.DATASET_TO_FAMILY[dataset],
            )
            for dataset in report.TARGET_DATASETS
        ),
        blocks=tuple(
            report.BlockSpec(
                block,
                report.BLOCK_INDEX[block],
                *report.BLOCK_SCALE_RANGE[block],
            )
            for block in report.BLOCKS
        ),
        source_ordinal=report.FIXED_SOURCE_ORDINAL,
        assigned_per_block=report.ASSIGNED_PER_BLOCK,
        dpi=360,
    )


def _write_aggregate(
    tmp_path: Path,
    *,
    tamper_half_assigned: bool = False,
) -> tuple[report.ReportPlan, Path]:
    folds_root = tmp_path / "folds"
    sources = [
        _write_fold(
            folds_root / family,
            family,
            tamper_assigned=tamper_half_assigned and family == "half_cylinder",
        )
        for family in report.FAMILY_ORDER
    ]
    aggregate_root = tmp_path / "aggregate"
    aggregate_root.mkdir()
    table_path = aggregate_root / "outer_family_summary.csv"
    table_path.write_text("outer_family\n" + "\n".join(report.FAMILY_ORDER) + "\n", encoding="utf-8")
    binding = _method_binding()
    summary_payload = {
        "schema": report.AGGREGATE_SUMMARY_SCHEMA,
        "experiment": report.PREDICTION_EXPERIMENT,
        "status": "completed",
        "mode": "complete_five_fold_aggregate",
        "config_sha256": CONFIG_SHA256,
        "aggregator_git_commit": COMMIT,
        "fold_numerical_git_commit": COMMIT,
        "outer_families": list(report.FAMILY_ORDER),
        "outer_family_count": 5,
        "formal_confirmation": False,
        "fold_summary_source": report.AGGREGATE_FRESH_REPLAY_SOURCE,
        "evidence_scope": "exposed_train_only_nested_family_validation",
        "outer_family_summary_file_sha256": sha256_file(table_path),
        report.METHOD_BINDING_KEY: binding,
    }
    _, summary_file_sha = _write_self_hashed(
        aggregate_root / "aggregate_summary.json",
        summary_payload,
    )
    shared = {
        "experiment": report.PREDICTION_EXPERIMENT,
        "status": "completed",
        "mode": "complete_five_fold_aggregate",
        "config_sha256": CONFIG_SHA256,
        "direct_parent_config_sha256": "8" * 64,
        "direct_parent_runner_sha256": "9" * 64,
        "direct_parent_aggregator_sha256": "a" * 64,
        "core_sha256": "b" * 64,
        report.METHOD_BINDING_KEY: binding,
        "aggregator_git_commit": COMMIT,
        "aggregator_worktree_clean": True,
        "fold_numerical_git_commit": COMMIT,
        "report_file": "aggregate_summary.json",
        "report_file_sha256": summary_file_sha,
        "early_stop_certificate": None,
    }
    manifest_payload = {
        "schema": report.AGGREGATE_MANIFEST_SCHEMA,
        **shared,
        "outer_family_summary_file": table_path.name,
        "outer_family_summary_file_sha256": sha256_file(table_path),
        "source_folds": sources,
    }
    _, manifest_file_sha = _write_self_hashed(
        aggregate_root / "aggregate_manifest.json",
        manifest_payload,
    )
    complete_payload = {
        "schema": report.AGGREGATE_COMPLETE_SCHEMA,
        **shared,
        "aggregate_manifest_file": "aggregate_manifest.json",
        "aggregate_manifest_file_sha256": manifest_file_sha,
        "completed_utc": "2030-01-01T00:00:00+00:00",
    }
    _write_self_hashed(
        aggregate_root / "AGGREGATE_COMPLETE.json",
        complete_payload,
    )
    assert {path.name for path in aggregate_root.iterdir()} == set(
        report.AGGREGATE_FILE_NAMES
    )
    return _plan(tmp_path, aggregate_root), aggregate_root


def _ephemeral_config(tmp_path: Path, aggregate_root: Path) -> tuple[Path, str]:
    value = {
        "schema": report.REPORT_CONFIG_SCHEMA,
        "experiment": report.REPORT_EXPERIMENT,
        "status": "frozen_pre_run_not_run",
        "prediction_parent": {
            "experiment": report.PREDICTION_EXPERIMENT,
            "numerical_git_commit": COMMIT,
            "config_sha256": CONFIG_SHA256,
            "aggregate_root": str(aggregate_root.resolve()),
            "aggregate_complete_sha256": "c" * 64,
            "aggregate_mode": "complete_five_fold_aggregate",
            "aggregate_fresh_replay_required": True,
        },
        "parent_scenes": {
            "experiment": "Other_SyntheticParent_1.1",
            "numerical_git_commit": "d" * 40,
            "config_sha256": PARENT_CONFIG_SHA256,
            "root": str((tmp_path / "parent").resolve()),
            "result_manifest_sha256": PARENT_RESULT_SHA256,
            "run_complete_sha256": PARENT_COMPLETE_SHA256,
        },
        "query": {
            "source_ordinal": 2,
            "assigned_per_block": 64_000,
            "datasets": [
                {
                    "id": dataset,
                    "display_name": report.DISPLAY_NAMES[dataset],
                    "outer_family": report.DATASET_TO_FAMILY[dataset],
                }
                for dataset in report.TARGET_DATASETS
            ],
            "scale_blocks": [
                {
                    "id": block,
                    "index": report.BLOCK_INDEX[block],
                    "scale_start": report.BLOCK_SCALE_RANGE[block][0],
                    "scale_stop": report.BLOCK_SCALE_RANGE[block][1],
                }
                for block in report.BLOCKS
            ],
        },
        "figure_contract": {
            "expected_figure_count": 8,
            "backend": "Python/matplotlib",
            "dpi": 360,
            "panel_titles": list(report.PANEL_TITLES),
        },
        "output_contract": {
            "overwrite": False,
            "required_global_files": [
                "frozen_config.yaml",
                "input_manifest.json",
                "figure_contract.json",
                "per_figure_metrics.csv",
                "visualization_manifest.json",
                "result_manifest.json",
                "RUN_COMPLETE.json",
            ],
            "required_exports_per_figure": [
                "scene_npz",
                "scene_manifest",
                "png",
                "pdf",
                "svg",
                "alignment",
                "render_metadata",
            ],
        },
    }
    path = tmp_path / "synthetic_report_config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path, sha256_file(path)


def test_report_contract_is_eight_classification_triptychs_without_production_config() -> None:
    assert report.REPORT_EXPERIMENT == (
        "Other_ClassConditionalTemplateScoreVisualization_1.1"
    )
    assert len(report.PREDICTION_ARRAY_NAMES) == 19
    assert len(report.FOLD_FILE_NAMES) == 15
    assert len(report.FOLD_ARTIFACT_NAMES) == 13
    assert len(report.TARGET_DATASETS) * len(report.BLOCKS) == 8
    assert "classification" in report.PANEL_TITLES[1]
    assert "cluster" not in report.PANEL_TITLES[1].lower()
    assert not hasattr(report, "_fresh_replay_required_folds")
    assert not (
        ROOT
        / "config"
        / "Other_ClassConditionalTemplateScoreVisualization_1.1.yaml"
    ).exists()


def test_report_environment_records_cpu_host_and_scheduler_identity() -> None:
    environment = report._environment_record("cpu")
    assert set(environment) == {
        "hostname",
        "platform",
        "python",
        "numpy",
        "requested_device",
        "slurm_job_id",
        "slurm_array_task_id",
    }
    assert environment["hostname"]
    assert environment["platform"]
    assert environment["python"]
    assert environment["numpy"]
    assert environment["requested_device"] == "cpu"


def test_ephemeral_frozen_config_is_byte_bound_and_rejects_placeholders(
    tmp_path: Path,
) -> None:
    config, digest = _ephemeral_config(tmp_path, tmp_path / "aggregate")
    plan = report.load_report_plan(config, digest)
    assert tuple(value.dataset for value in plan.datasets) == report.TARGET_DATASETS
    assert tuple(value.block for value in plan.blocks) == report.BLOCKS
    assert plan.source_ordinal == 2

    config.write_text(config.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    _expect_value_error(
        report.load_report_plan,
        config,
        digest,
        contains="config SHA-256 mismatch",
    )

    config, digest = _ephemeral_config(tmp_path, tmp_path / "aggregate")
    text = config.read_text(encoding="utf-8").replace(COMMIT, "PENDING_IDENTITY", 1)
    config.write_text(text, encoding="utf-8")
    _expect_value_error(
        report.load_report_plan,
        config,
        sha256_file(config),
        contains="placeholder identity is forbidden",
    )


def test_structural_authentication_covers_five_folds_and_never_opens_npz(
    tmp_path: Path,
) -> None:
    plan, _ = _write_aggregate(tmp_path)

    def forbidden_np_load(*args, **kwargs):
        raise AssertionError("structural authentication opened an NPZ member")

    with patch.object(report.np, "load", side_effect=forbidden_np_load):
        evidence = report.authenticate_aggregate_chain(plan)
    assert tuple(evidence.folds) == report.FAMILY_ORDER
    assert all(len(fold.artifacts) == 13 for fold in evidence.folds.values())
    assert len(evidence.files) == 4 + 5 * 15


def test_structural_authentication_rejects_extra_fold_file(tmp_path: Path) -> None:
    plan, _ = _write_aggregate(tmp_path)
    extra = plan.aggregate_root.parent / "folds" / "channel" / "extra.txt"
    extra.write_text("not in frozen transaction\n", encoding="utf-8")
    _expect_value_error(
        report.authenticate_aggregate_chain,
        plan,
        contains="exactly 15 files",
    )


def test_19_array_validation_projects_exactly_eight_source_two_groups(
    tmp_path: Path,
) -> None:
    plan, _ = _write_aggregate(tmp_path)
    evidence = report.authenticate_aggregate_chain(plan)
    groups = report.load_prediction_groups(plan, evidence)
    assert set(groups) == {
        (dataset, block)
        for dataset in report.TARGET_DATASETS
        for block in report.BLOCKS
    }
    for (dataset, block), loaded in groups.items():
        assert loaded.group.dataset == dataset
        assert loaded.group.block == block
        assert loaded.group.source_ordinal == 2
        assert loaded.group.center_seed_index.tolist() == [0, 1]
        assert loaded.group.prediction.tolist() == [False, True]


def test_19_array_validation_rejects_self_consistent_assigned_identity_tamper(
    tmp_path: Path,
) -> None:
    plan, _ = _write_aggregate(tmp_path, tamper_half_assigned=True)
    evidence = report.authenticate_aggregate_chain(plan)
    _expect_value_error(
        report.load_prediction_groups,
        plan,
        evidence,
        contains="assigned-row identity changed",
    )


def _small_prediction_group() -> PredictionGroup:
    return PredictionGroup(
        dataset="cylinder3d",
        source_ordinal=2,
        source_index=9,
        block="legacy_2_1",
        outer_family="half_cylinder",
        candidate=_candidate(),
        center_seed_index=np.asarray([0, 1, 2, 3], dtype=np.int64),
        assigned_row_index=np.asarray([0, 1, 2, 3], dtype=np.int64),
        scale_id=np.asarray([0, 1, 2, 3], dtype=np.int32),
        scale_block_index=np.zeros(4, dtype=np.int8),
        spatial_score=np.asarray([0.1, 0.8, 0.4, 0.9], dtype=np.float64),
        prediction=np.asarray([False, True, False, True], dtype=np.bool_),
    )


def test_exact_parent_prediction_identity_join_rejects_reordered_rows() -> None:
    group = _small_prediction_group()
    metadata = {
        "dataset": group.dataset,
        "source_ordinal": group.source_ordinal,
        "source_index": group.source_index,
        "scale_block_id": group.block,
    }
    arrays = {
        "valid_center_seed_index": group.center_seed_index.copy(),
        "valid_assigned_row_index": group.assigned_row_index.copy(),
        "valid_scale_id": group.scale_id.copy(),
        "valid_scale_block_index": group.scale_block_index.copy(),
    }
    prediction, score = exact_bind_prediction_group(metadata, arrays, group)
    assert np.array_equal(prediction, group.prediction)
    assert np.array_equal(score, group.spatial_score)

    reordered = replace(
        group,
        center_seed_index=group.center_seed_index[::-1],
        assigned_row_index=group.assigned_row_index[::-1],
        scale_id=group.scale_id[::-1],
        scale_block_index=group.scale_block_index[::-1],
        spatial_score=group.spatial_score[::-1],
        prediction=group.prediction[::-1],
    )
    _expect_value_error(
        exact_bind_prediction_group,
        metadata,
        arrays,
        reordered,
        contains="row order differs",
    )


def test_metrics_are_recomputed_and_any_stored_metric_drift_is_rejected() -> None:
    loaded = report.LoadedPredictionGroup(
        group=_small_prediction_group(),
        retrieval_supported=np.asarray([True, True, True, True]),
        calibration_supported=np.asarray([True, True, False, False]),
        spatial_imputed=np.asarray([False, False, True, False]),
        spatial_unimputable=np.asarray([False, False, False, True]),
        calibration_mode=np.asarray([1, 1, 2, 3], dtype=np.int8),
        scaler_mode=np.asarray([1, 1, 2, 2], dtype=np.int8),
        group_audit={},
    )
    reference = np.asarray([False, True, True, False], dtype=np.bool_)
    recomputed = report.recompute_complete_metric_row(
        reference=reference,
        loaded=loaded,
    )
    assert recomputed["true_positive"] == 1
    assert recomputed["false_positive"] == 1
    assert recomputed["true_negative"] == 1
    assert recomputed["false_negative"] == 1
    report.compare_metric_rows(recomputed, dict(recomputed))
    drifted = dict(recomputed)
    drifted["accuracy"] = float(drifted["accuracy"]) + 1.0e-6
    _expect_value_error(
        report.compare_metric_rows,
        drifted,
        recomputed,
        contains="metric mismatch: accuracy",
    )


def test_child_scene_may_change_only_prediction_and_metadata(tmp_path: Path) -> None:
    parent_arrays: dict[str, np.ndarray] = {}
    for index, name in enumerate(SCENE31_ARRAY_NAMES):
        if name == "prediction":
            value = np.asarray([False, False, False], dtype=np.bool_)
        elif name == "metadata_json":
            value = np.asarray('{"role":"parent"}')
        else:
            value = np.asarray([index, index + 1, index + 2], dtype=np.int64)
        parent_arrays[name] = value
    prediction = np.asarray([False, True, True], dtype=np.bool_)
    child_arrays = {
        name: np.array(value, copy=True) for name, value in parent_arrays.items()
    }
    child_arrays["prediction"] = prediction
    child_arrays["metadata_json"] = np.asarray('{"role":"child"}')
    child_path = tmp_path / "child.scene.npz"
    np.savez(child_path, **child_arrays)
    audit = report._verify_child_scene_invariance(
        parent_arrays,
        child_path,
        prediction,
    )
    assert audit["unchanged_array_count"] == 20
    assert audit["only_prediction_and_metadata_changed"] is True

    immutable_name = next(
        name
        for name in SCENE31_ARRAY_NAMES
        if name not in {"prediction", "metadata_json"}
    )
    child_arrays[immutable_name] = child_arrays[immutable_name] + 1
    tampered_path = tmp_path / "tampered.scene.npz"
    np.savez(tampered_path, **child_arrays)
    _expect_value_error(
        report._verify_child_scene_invariance,
        parent_arrays,
        tampered_path,
        prediction,
        contains="changed immutable parent array",
    )


def test_existing_output_is_rejected_before_config_or_input_access(
    tmp_path: Path,
) -> None:
    output = tmp_path / "already_exists"
    output.mkdir()
    try:
        report.run_from_config(
            config_path=tmp_path / "does_not_exist.yaml",
            config_sha256="e" * 64,
            output_root=output,
            expected_reporting_commit="f" * 40,
            device="cpu",
        )
    except FileExistsError as error:
        assert "output directory already exists" in str(error)
    else:
        raise AssertionError("existing immutable output was accepted")


def _run_standalone() -> None:
    """Run the synthetic checks when pytest is unavailable on a workstation."""

    test_report_contract_is_eight_classification_triptychs_without_production_config()
    test_report_environment_records_cpu_host_and_scheduler_identity()
    temporary_tests = (
        test_ephemeral_frozen_config_is_byte_bound_and_rejects_placeholders,
        test_structural_authentication_covers_five_folds_and_never_opens_npz,
        test_structural_authentication_rejects_extra_fold_file,
        test_19_array_validation_projects_exactly_eight_source_two_groups,
        test_19_array_validation_rejects_self_consistent_assigned_identity_tamper,
        test_child_scene_may_change_only_prediction_and_metadata,
        test_existing_output_is_rejected_before_config_or_input_access,
    )
    for function in temporary_tests:
        with tempfile.TemporaryDirectory() as directory:
            function(Path(directory))
    test_exact_parent_prediction_identity_join_rejects_reordered_rows()
    test_metrics_are_recomputed_and_any_stored_metric_drift_is_rejected()


if __name__ == "__main__":
    _run_standalone()

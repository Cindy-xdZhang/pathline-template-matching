from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.early_kinematic_preparation import (  # noqa: E402
    CleanSourceIdentity,
    REQUIRED_SOURCE_PATHS,
)
from pathline_template_matching.portable_flow import sha256_file  # noqa: E402
from scripts import aggregate_verify_class_conditional_template_score_1_1 as aggregate_module  # noqa: E402
from scripts import run_verify_class_conditional_template_score_1_1 as runner  # noqa: E402


CONFIG = ROOT / "config" / "Verify_ClassConditionalTemplateScore_1.1.yaml"
NUMERICAL_COMMIT = "1" * 40
parent = aggregate_module.parent


def _expect_value_error(function, *args, contains: str, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert contains in str(error), str(error)
        return
    raise AssertionError("expected ValueError")


def _bound_plan(plan: runner.Plan, root: Path) -> runner.Plan:
    identity = CleanSourceIdentity(
        git_commit=NUMERICAL_COMMIT,
        worktree_clean=True,
        source_file_sha256_items=tuple(
            (path, hashlib.sha256(path.encode("utf-8")).hexdigest())
            for path in REQUIRED_SOURCE_PATHS
        ),
    )
    return replace(
        plan,
        source_identity=identity,
        kinematic_input_manifest_path=root / "KINEMATIC_INPUT.json",
        kinematic_input_manifest_file_sha256="2" * 64,
        kinematic_input_manifest_content_sha256="3" * 64,
        synthetic_pass_path=root / "SYNTHETIC_PASS.json",
        synthetic_pass_file_sha256="4" * 64,
        sidecar_root=root / "sidecars",
        sidecar_population_manifest_path=root / "SIDECAR_POPULATION.json",
        sidecar_population_manifest_file_sha256="5" * 64,
        sidecar_population_manifest_content_sha256="6" * 64,
        sidecar_population=MappingProxyType(
            {"git_commit": parent.PREPARATION_ARTIFACT_GIT_COMMIT, "rows": ()}
        ),
        composite_descriptor_ids=MappingProxyType(
            {name: f"{name}_synthetic" for name in runner.REPRESENTATIONS}
        ),
    )


def _support(plan: runner.Plan, family: str, sample_count: int) -> dict[str, Any]:
    fit_families = tuple(value for value in plan.family_order if value != family)
    family_values = {
        fit_family: {
            f"{class_name}_{support_name}_{suffix}": (
                sample_count if suffix == "count" else 1.0
            )
            for class_name in ("positive", "negative")
            for support_name in ("retrieval", "calibration")
            for suffix in ("count", "fraction")
        }
        for fit_family in fit_families
    }
    return {
        "schema": (
            "pathline_template_matching."
            "class_conditional_outer_support_summary.v1"
        ),
        "sample_count": sample_count,
        "family_order": list(fit_families),
        "required_joint_family_count": len(fit_families) // 2 + 1,
        "joint_supported_family_count_histogram": {
            str(index): sample_count if index == len(fit_families) else 0
            for index in range(len(fit_families) + 1)
        },
        "families": family_values,
    }


def _cpu_environment() -> dict[str, Any]:
    return {
        "hostname": "synthetic-host",
        "platform": "synthetic-platform",
        "python": "3.11.0",
        "numpy": np.__version__,
        "torch": "synthetic-torch",
        "requested_device": "cpu",
        "slurm_job_id": None,
        "slurm_array_task_id": None,
        "slurm_job_gpus": None,
        "cuda_available": False,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
    }


def _summary(
    plan: runner.Plan,
    family: str,
    f1: float,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": runner.OUTER_SUMMARY_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "outer_family": family,
        "group_count": 8,
    }
    value.update({field: 0.8 for field in aggregate_module.FAMILY_METRIC_FIELDS})
    value["f1"] = f1
    counts = {field: 0 for field in aggregate_module.FAMILY_COUNT_FIELDS}
    counts.update(
        {
            "sample_count": 8,
            "positive_count": 4,
            "negative_count": 4,
            "true_positive": 3,
            "false_positive": 1,
            "true_negative": 3,
            "false_negative": 1,
            "retrieval_supported_count": 8,
            "calibration_supported_count": 8,
            "calibration_mode_1_count": 8,
            "scaler_mode_1_count": 8,
        }
    )
    value.update(counts)
    value["class_conditional_support"] = _support(plan, family, 8)
    return value


def _write_self_hashed(path: Path, payload: Mapping[str, Any]) -> str:
    return parent._atomic_json(path, parent._manifest_with_self_hash(payload))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rewrite_self_hashed(
    path: Path, payload: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    unsigned = parent._json_safe(payload)
    unsigned.pop("content_sha256", None)
    signed = parent._manifest_with_self_hash(unsigned)
    path.unlink()
    file_sha256 = parent._atomic_json(path, signed)
    return signed, file_sha256


def _resign_release_chain(
    output: Path,
    change,
) -> str:
    completion_path = output / "AGGREGATE_COMPLETE.json"
    report_path = output / "single_fold_authentication_report.json"
    manifest_path = output / "aggregate_manifest.json"
    completion = _read_json(completion_path)
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    change(completion, report, manifest)
    _, report_sha = _rewrite_self_hashed(report_path, report)
    _, manifest_sha = _rewrite_self_hashed(manifest_path, manifest)
    completion["report_file_sha256"] = report_sha
    completion["aggregate_manifest_file_sha256"] = manifest_sha
    _, completion_sha = _rewrite_self_hashed(completion_path, completion)
    return completion_sha


def _rewrite_fold_result_chain(path: Path) -> None:
    result_path = path / "result_manifest.json"
    result = _read_json(result_path)
    result, result_file_sha = _rewrite_self_hashed(result_path, result)
    completion_path = path / "RUN_COMPLETE.json"
    completion = _read_json(completion_path)
    completion["result_manifest_file_sha256"] = result_file_sha
    completion["result_manifest_content_sha256"] = result["content_sha256"]
    _rewrite_self_hashed(completion_path, completion)


def _bind_changed_artifact_into_result(path: Path, name: str) -> None:
    artifact_path = path / name
    result_path = path / "result_manifest.json"
    result = _read_json(result_path)
    identity = {
        "size_bytes": artifact_path.stat().st_size,
        "sha256": sha256_file(artifact_path),
    }
    result["artifacts"][name] = identity
    result[aggregate_module.DIRECT_ARTIFACT_HASH_FIELDS[name]] = identity["sha256"]
    _rewrite_self_hashed(result_path, result)
    _rewrite_fold_result_chain(path)


def _fold(
    root: Path,
    plan: runner.Plan,
    family: str,
    f1: float,
) -> aggregate_module.AuthenticatedFold:
    path = (root / f"fold_{family}").resolve()
    path.mkdir()
    summary = _summary(plan, family, f1)
    for index, name in enumerate(aggregate_module.EXPECTED_FOLD_FILES):
        artifact = path / name
        if name == "outer_summary.json":
            _write_self_hashed(
                artifact,
                {
                    **summary,
                    runner.METHOD_BINDING_KEY: runner._method_binding(
                        plan, NUMERICAL_COMMIT
                    ),
                },
            )
        else:
            artifact.write_bytes(bytes([index + 1]))
    identities = {
        name: {
            "size_bytes": (path / name).stat().st_size,
            "sha256": sha256_file(path / name),
        }
        for name in aggregate_module.EXPECTED_RESULT_ARTIFACTS
    }
    return aggregate_module.AuthenticatedFold(
        path=path,
        outer_family=family,
        numerical_git_commit=NUMERICAL_COMMIT,
        config_sha256=plan.sha256,
        direct_parent_config_sha256=runner.EXPECTED_PARENT_CONFIG_SHA256,
        direct_parent_runner_sha256=runner.EXPECTED_PARENT_RUNNER_SHA256,
        direct_parent_aggregator_sha256=(
            runner.EXPECTED_PARENT_AGGREGATOR_SHA256
        ),
        core_sha256=runner.EXPECTED_CORE_SHA256,
        input_manifest_sha256=plan.manifest_sha256,
        input_manifest_rows_sha256=plan.manifest_rows_sha256,
        requested_device="cpu",
        selected_candidate={"candidate_id": f"candidate_{family}"},
        summary=summary,
        artifact_identities=identities,
        completion_file_sha256=sha256_file(path / "RUN_COMPLETE.json"),
        completion_content_sha256="7" * 64,
        result_manifest_file_sha256=sha256_file(path / "result_manifest.json"),
        result_manifest_content_sha256="9" * 64,
    )


def _aggregate(
    plan: runner.Plan,
    folds: tuple[aggregate_module.AuthenticatedFold, ...],
    output: Path,
    *,
    mode: str = "auto",
):
    bound = _bound_plan(plan, output.parent)
    by_path = {fold.path: fold for fold in folds}

    def authenticate(_plan, path, *, device, expected_fold_commit):
        assert _plan is bound
        assert device == "cpu"
        assert expected_fold_commit == NUMERICAL_COMMIT
        return by_path[path.resolve()]

    with (
        patch.object(aggregate_module.runner, "load_plan", return_value=plan),
        patch.object(
            aggregate_module.runner, "bind_early_evidence", return_value=bound
        ),
        patch.object(
            aggregate_module.runner,
            "_git_identity",
            return_value=(NUMERICAL_COMMIT, False),
        ),
        patch.object(aggregate_module.parent, "_configure_execution"),
        patch.object(
            aggregate_module, "_authenticate_fold", side_effect=authenticate
        ),
    ):
        return aggregate_module.aggregate(
            CONFIG,
            [fold.path for fold in folds],
            output,
            expected_fold_commit=NUMERICAL_COMMIT,
            kinematic_input_manifest_path=output.parent / "KINEMATIC_INPUT.json",
            kinematic_input_manifest_file_sha256="2" * 64,
            synthetic_pass_path=output.parent / "SYNTHETIC_PASS.json",
            synthetic_pass_file_sha256="4" * 64,
            sidecar_root=output.parent / "sidecars",
            sidecar_population_manifest_path=(
                output.parent / "SIDECAR_POPULATION.json"
            ),
            sidecar_population_manifest_file_sha256="5" * 64,
            mode=mode,
            device="cpu",
        )


def _fresh_rows(
    plan: runner.Plan,
    family: str,
    candidate: runner.TailCandidateSpec,
    *,
    f1: float = 1.0,
) -> runner.OuterMetricRows:
    rows: list[dict[str, Any]] = []
    for dataset in plan.families[family]:
        for source_ordinal in range(4):
            for block in runner.BLOCK_NAMES:
                values: dict[str, Any] = {
                    "outer_family": family,
                    "inner_family": "outer_evaluation_only",
                    "dataset": dataset,
                    "source_ordinal": source_ordinal,
                    "block": block,
                    "candidate_id": candidate.candidate_id,
                    "representation": candidate.representation,
                    "k": candidate.k,
                    "sigma": candidate.sigma,
                    "decision_rule": candidate.decision_rule,
                    "decision_value": candidate.decision_value,
                }
                values.update(
                    {
                        "sample_count": 2,
                        "positive_count": 1,
                        "negative_count": 1,
                        "true_positive": 1,
                        "false_positive": 0,
                        "true_negative": 1,
                        "false_negative": 0,
                        "retrieval_supported_count": 2,
                        "calibration_supported_count": 2,
                        "imputed_count": 0,
                        "unimputable_count": 0,
                        "calibration_mode_0_count": 0,
                        "calibration_mode_1_count": 2,
                        "calibration_mode_2_count": 0,
                        "calibration_mode_3_count": 0,
                        "calibration_mode_4_count": 0,
                        "calibration_mode_5_count": 0,
                        "scaler_mode_0_count": 0,
                        "scaler_mode_1_count": 2,
                        "scaler_mode_2_count": 0,
                        "scaler_mode_3_count": 0,
                    }
                )
                for field in parent.METRIC_FIELDS:
                    values.setdefault(field, 1.0)
                values["f1"] = f1
                rows.append({field: values[field] for field in parent.METRIC_FIELDS})
    sample_count = sum(int(row["sample_count"]) for row in rows)
    return runner.OuterMetricRows(rows, _support(plan, family, sample_count))


def _write_full_synthetic_fold(
    root: Path,
    plan: runner.Plan,
    *,
    family: str = "half_cylinder",
    directory_name: str = "full_fold",
    persisted_support: Mapping[str, Any] | None = None,
    f1: float = 1.0,
) -> tuple[Path, runner.OuterMetricRows, list[dict[str, Any]]]:
    path = root / directory_name
    path.mkdir()
    candidate = runner.candidate_specs(plan)[0]
    method = runner._method_binding(plan, NUMERICAL_COMMIT)
    fresh_rows = _fresh_rows(plan, family, candidate, f1=f1)
    reference_rows = [
        {
            "dataset": plan.families[family][0],
            "source_ordinal": 0,
            "access": "synthetic_label_gate_after_replay",
        }
    ]

    for name in (
        "inner_group_metrics.csv",
        "inner_candidate_summary.csv",
        "inner_fit_audits.json",
        "final_per_scale_scaler.npz",
        "final_per_scale_scaler_manifest.json",
        "final_tail_calibration.npz",
        "final_tail_calibration_manifest.json",
    ):
        (path / name).write_bytes(f"synthetic:{name}\n".encode("utf-8"))
    selected = parent._manifest_with_self_hash(
        {
            "schema": runner.SELECTED_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "created_utc": "2026-01-01T00:00:00+00:00",
            "config_sha256": plan.sha256,
            "git_commit": NUMERICAL_COMMIT,
            "outer_family": family,
            "candidate": parent._candidate_payload(candidate),
            "candidate_count": runner.FROZEN_CANDIDATE_COUNT,
            "early_evidence": {},
            "inner_selection_summary": parent._candidate_payload(candidate),
            "inner_evidence": {},
            "final_per_scale_scaler_manifest": {},
            "final_per_scale_scaler_file": {},
            "final_calibration_manifest": {},
            "final_calibration_file": {},
            "outer_feature_member_opened": False,
            runner.METHOD_BINDING_KEY: method,
        }
    )
    parent._atomic_json(path / "selected_candidate.json", selected)
    arrays = {
        name: np.zeros(3, dtype=dtype)
        for name, dtype in runner.PREDICTION_ARRAY_DTYPES.items()
    }
    parent._atomic_npz(path / "outer_predictions.npz", arrays)
    _write_self_hashed(
        path / "outer_prediction_manifest.json",
        {
            "schema": runner.PREDICTION_MANIFEST_SCHEMA,
            "experiment": runner.EXPERIMENT,
            runner.METHOD_BINDING_KEY: method,
        },
    )
    parent._atomic_csv(
        path / "outer_group_metrics.csv", parent.METRIC_FIELDS, fresh_rows
    )
    summary = runner._outer_summary(fresh_rows, family)
    if persisted_support is not None:
        summary["class_conditional_support"] = dict(persisted_support)
    _write_self_hashed(
        path / "outer_summary.json",
        {**summary, runner.METHOD_BINDING_KEY: method},
    )

    prediction_manifest_sha = sha256_file(path / "outer_prediction_manifest.json")
    prediction_sha = sha256_file(path / "outer_predictions.npz")
    _write_self_hashed(
        path / "outer_reference_access_audit.json",
        {
            "schema": runner.REFERENCE_AUDIT_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "outer_family": family,
            "first_open_phase": (
                "after_outer_prediction_file_and_manifest_authentication"
            ),
            "prediction_manifest_file_sha256": prediction_manifest_sha,
            "prediction_file_sha256": prediction_sha,
            "row_count": len(reference_rows),
            "rows": reference_rows,
            runner.METHOD_BINDING_KEY: method,
        },
    )

    artifacts = {
        name: {
            "size_bytes": (path / name).stat().st_size,
            "sha256": sha256_file(path / name),
        }
        for name in aggregate_module.EXPECTED_RESULT_ARTIFACTS
    }
    input_identity = {
        "schema": plan.manifest_schema,
        "path": str(plan.manifest_path),
        "size_bytes": plan.manifest_size,
        "sha256": plan.manifest_sha256,
        "rows_content_sha256": plan.manifest_rows_sha256,
    }
    result_payload: dict[str, Any] = {
        "schema": runner.RESULT_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": "completed",
        "completed_utc": "2026-01-01T00:00:00+00:00",
        "git_commit": NUMERICAL_COMMIT,
        "config_path": str(plan.path),
        "config_sha256": plan.sha256,
        "input_manifest": input_identity,
        "early_evidence": parent._early_artifact_binding(
            plan,
            representation=candidate.representation,
            fit_families=[value for value in plan.family_order if value != family],
        ),
        "outer_family": family,
        "selected_candidate": parent._candidate_payload(candidate),
        "selected_candidate_file": "selected_candidate.json",
        "selected_candidate_content_sha256": selected["content_sha256"],
        "environment": _cpu_environment(),
        "artifacts": artifacts,
        runner.METHOD_BINDING_KEY: method,
    }
    for name, field in aggregate_module.DIRECT_ARTIFACT_HASH_FIELDS.items():
        result_payload[field] = artifacts[name]["sha256"]
    result = parent._manifest_with_self_hash(result_payload)
    result_sha = parent._atomic_json(path / "result_manifest.json", result)
    _write_self_hashed(
        path / "RUN_COMPLETE.json",
        {
            "schema": runner.COMPLETE_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "outer_family": family,
            "git_commit": NUMERICAL_COMMIT,
            "config_sha256": plan.sha256,
            "result_manifest_file": "result_manifest.json",
            "result_manifest_file_sha256": result_sha,
            "result_manifest_content_sha256": result["content_sha256"],
            "completed_utc": "2026-01-01T00:00:00+00:00",
            runner.METHOD_BINDING_KEY: method,
        },
    )
    assert {item.name for item in path.iterdir()} == set(runner.REQUIRED_FOLD_FILES)
    return path, fresh_rows, reference_rows


def _authenticate_synthetic_fold(
    plan: runner.Plan,
    path: Path,
    fresh_rows: runner.OuterMetricRows,
    reference_rows: list[dict[str, Any]],
) -> aggregate_module.AuthenticatedFold:
    input_identity = {
        "schema": plan.manifest_schema,
        "path": str(plan.manifest_path),
        "size_bytes": plan.manifest_size,
        "sha256": plan.manifest_sha256,
        "rows_content_sha256": plan.manifest_rows_sha256,
    }

    def replay(_plan, _candidate, _staged, **_kwargs):
        assert _plan is plan
        return fresh_rows, reference_rows

    with (
        patch.object(parent, "load_cache_rows", return_value=((), input_identity)),
        patch.object(runner, "evaluate_outer_prediction", side_effect=replay),
    ):
        return aggregate_module._authenticate_fold(
            plan,
            path,
            device="cpu",
            expected_fold_commit=NUMERICAL_COMMIT,
        )


def test_selected_candidate_writer_and_aggregator_field_contract_match() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bound = _bound_plan(plan, root)
        candidate = runner.candidate_specs(bound)[0]
        evidence_paths = {
            "inner_group_metrics": root / "inner_group_metrics.csv",
            "inner_candidate_summary": root / "inner_candidate_summary.csv",
            "inner_fit_audits": root / "inner_fit_audits.json",
        }
        for name, path in evidence_paths.items():
            path.write_bytes(f"production-shaped:{name}\n".encode("utf-8"))
        scaler = parent.VerifiedScalerArtifact(
            manifest_path=root / "final_per_scale_scaler_manifest.json",
            manifest_file_sha256="7" * 64,
            scaler_file_sha256="8" * 64,
            manifest={"content_sha256": "9" * 64},
            scaler=object(),
            _authentication_seal=parent._AUTHENTICATION_SEAL,
        )
        calibration = parent.VerifiedCalibrationArtifact(
            manifest_path=root / "final_tail_calibration_manifest.json",
            manifest_file_sha256="a" * 64,
            calibration_file_sha256="b" * 64,
            manifest={"content_sha256": "c" * 64},
            model=object(),
            _authentication_seal=parent._AUTHENTICATION_SEAL,
        )
        with runner.class_conditional_parent_runtime(bound, NUMERICAL_COMMIT):
            path, _, payload = parent.write_selected_candidate(
                root,
                plan=bound,
                selected=candidate,
                selected_summary=parent._candidate_payload(candidate),
                scaler=scaler,
                calibration=calibration,
                inner_group_metrics_path=evidence_paths["inner_group_metrics"],
                inner_group_metrics_sha256=sha256_file(
                    evidence_paths["inner_group_metrics"]
                ),
                inner_candidate_summary_path=evidence_paths[
                    "inner_candidate_summary"
                ],
                inner_candidate_summary_sha256=sha256_file(
                    evidence_paths["inner_candidate_summary"]
                ),
                inner_fit_audits_path=evidence_paths["inner_fit_audits"],
                inner_fit_audits_sha256=sha256_file(
                    evidence_paths["inner_fit_audits"]
                ),
                outer_family="half_cylinder",
                git_commit=NUMERICAL_COMMIT,
            )
        assert set(payload) == aggregate_module.SELECTED_CANDIDATE_FIELDS
        assert set(_read_json(path)) == aggregate_module.SELECTED_CANDIDATE_FIELDS


def test_aggregate_contract_and_label_gate_order_are_frozen() -> None:
    plan = runner.load_plan(CONFIG)
    aggregate_module._validate_plan_output_contract(plan)
    assert plan.sha256 == runner.EXPECTED_CONFIG_SHA256
    assert sha256_file(runner.CORE_PATH) == runner.EXPECTED_CORE_SHA256
    assert aggregate_module.EXPECTED_FOLD_FILES == runner.REQUIRED_FOLD_FILES
    assert len(aggregate_module.EXPECTED_FOLD_FILES) == 15
    assert len(aggregate_module.EXPECTED_RESULT_ARTIFACTS) == 13
    assert len(runner.PREDICTION_ARRAY_DTYPES) == 19
    assert set(aggregate_module.LABEL_FREE_PRE_RESULT_FILES).isdisjoint(
        {
            "result_manifest.json",
            "outer_group_metrics.csv",
            "outer_summary.json",
            "outer_reference_access_audit.json",
        }
    )
    source = inspect.getsource(aggregate_module._authenticate_fold)
    replay = source.index("runner.evaluate_outer_prediction(")
    result_open = source.index("result_snapshot = _read_file_snapshot")
    metric_open = source.index('"outer_group_metrics.csv",', result_open)
    assert replay < result_open < metric_open
    assert "include_labels=True" not in source


def test_support_audit_validates_every_class_family_field() -> None:
    plan = runner.load_plan(CONFIG)
    support = _support(plan, "half_cylinder", 10)
    assert (
        aggregate_module._authenticate_support_audit(
            plan, "half_cylinder", support
        )
        == support
    )
    changed = parent._json_safe(support)
    changed["families"]["delta_wing"]["negative_calibration_count"] = 9
    _expect_value_error(
        aggregate_module._authenticate_support_audit,
        plan,
        "half_cylinder",
        changed,
        contains="negative_calibration_fraction",
    )
    changed = parent._json_safe(support)
    changed["families"]["delta_wing"]["negative_calibration_count"] = 11
    changed["families"]["delta_wing"]["negative_calibration_fraction"] = 1.1
    _expect_value_error(
        aggregate_module._authenticate_support_audit,
        plan,
        "half_cylinder",
        changed,
        contains="negative_calibration_count",
    )


def test_complete_five_and_single_fold_mathematical_certificates() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = tuple(
            _fold(root, plan, family, f1)
            for family, f1 in zip(
                plan.family_order,
                (0.80, 0.75, 0.70, 0.65, 0.60),
                strict=True,
            )
        )
        report = _aggregate(plan, folds, root / "aggregate")
        assert report["schema"] == aggregate_module.AGGREGATE_SUMMARY_SCHEMA
        assert report["outer_families"] == list(plan.family_order)
        np.testing.assert_allclose(report["family_macro"]["f1"], 0.70)
        assert all(report["success_stop_rule"]["outcomes"].values())
        assert report["all_success_conditions_pass"] is True
        assert tuple(report["class_conditional_support_by_outer_family"]) == (
            plan.family_order
        )
        assert report[runner.METHOD_BINDING_KEY] == runner._method_binding(
            plan, NUMERICAL_COMMIT
        )

        single = _aggregate(plan, (folds[0],), root / "single")
        assert single["schema"] == aggregate_module.SINGLE_FOLD_REPORT_SCHEMA
        assert single["five_fold_success_evaluated"] is False
        assert single["five_fold_success"] is None
        assert single["stop_version"] is False
        certificate, _ = aggregate_module._load_self_hashed_json(
            root / "single" / "early_stop_certificate.json"
        )
        assert certificate["stop_version"] is False
        assert certificate["macro_upper_bound_proofs"]["f1"][
            "unobserved_metric_upper_bound"
        ] == 1.0

        low = _fold(root, plan, "half_cylinder_low", 0.49)
        low = replace(
            low,
            outer_family="half_cylinder",
            summary=_summary(plan, "half_cylinder", 0.49),
        )
        stopped = _aggregate(plan, (low,), root / "single_stop")
        assert stopped["stop_version"] is True
        stopped_certificate, _ = aggregate_module._load_self_hashed_json(
            root / "single_stop" / "early_stop_certificate.json"
        )
        assert stopped_certificate["impossibility_reasons"] == [
            "observed_family_f1_below_frozen_minimum"
        ]


def test_aggregate_rejects_mixed_identity_scope_and_duplicate_families() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = tuple(_fold(root, plan, family, 0.8) for family in plan.family_order)
        cases = (
            (replace(folds[-1], numerical_git_commit="2" * 40), "mix numerical"),
            (replace(folds[-1], config_sha256="a" * 64), "mix active configs"),
            (
                replace(folds[-1], direct_parent_config_sha256="b" * 64),
                "mix direct-parent",
            ),
            (
                replace(folds[-1], direct_parent_runner_sha256="c" * 64),
                "mix direct-parent",
            ),
            (
                replace(folds[-1], direct_parent_aggregator_sha256="d" * 64),
                "mix direct-parent",
            ),
            (replace(folds[-1], core_sha256="e" * 64), "mix class-conditional"),
            (
                replace(folds[-1], input_manifest_sha256="f" * 64),
                "mix train-only",
            ),
        )
        for index, (mixed, message) in enumerate(cases):
            _expect_value_error(
                _aggregate,
                plan,
                (*folds[:-1], mixed),
                root / f"mixed_{index}",
                contains=message,
            )
        _expect_value_error(
            _aggregate,
            plan,
            folds[:4],
            root / "incomplete",
            contains="either one first-family fold or all five folds",
        )
        duplicate = replace(folds[-1], outer_family=folds[-2].outer_family)
        _expect_value_error(
            _aggregate,
            plan,
            (*folds[:-1], duplicate),
            root / "duplicate_family",
            contains="each frozen outer family exactly once",
        )


def test_single_fold_release_reauthenticates_source_summary_and_artifacts() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fold = _fold(root, plan, "half_cylinder", 0.60)
        output = root / "single"
        _aggregate(plan, (fold,), output)
        completion_sha = sha256_file(output / "AGGREGATE_COMPLETE.json")
        with (
            patch.object(aggregate_module.runner, "load_plan", return_value=plan),
            patch.object(
                aggregate_module.runner,
                "bind_early_evidence",
                return_value=plan,
            ),
            patch.object(
                aggregate_module,
                "_authenticate_fold",
                return_value=fold,
            ),
        ):
            release = aggregate_module.authenticate_single_fold_release(
                output,
                expected_completion_sha256=completion_sha,
                expected_fold_commit=NUMERICAL_COMMIT,
                expected_config_sha256=plan.sha256,
                expected_fold_directory=fold.path,
            )
        assert release["stop_version"] is False
        assert (
            release["direct_parent_config_sha256"]
            == runner.EXPECTED_PARENT_CONFIG_SHA256
        )
        assert release["core_sha256"] == runner.EXPECTED_CORE_SHA256
        (fold.path / "outer_predictions.npz").write_bytes(b"tampered")
        with (
            patch.object(aggregate_module.runner, "load_plan", return_value=plan),
            patch.object(
                aggregate_module.runner,
                "bind_early_evidence",
                return_value=plan,
            ),
            patch.object(
                aggregate_module,
                "_authenticate_fold",
                return_value=fold,
            ),
        ):
            _expect_value_error(
                aggregate_module.authenticate_single_fold_release,
                output,
                expected_completion_sha256=completion_sha,
                expected_fold_commit=NUMERICAL_COMMIT,
                expected_config_sha256=plan.sha256,
                expected_fold_directory=fold.path,
                contains="file size mismatch",
            )


def test_single_fold_release_rejects_resigned_summary_csv_replacements() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def authenticate(output: Path, fold, completion_sha: str):
            with (
                patch.object(aggregate_module.runner, "load_plan", return_value=plan),
                patch.object(
                    aggregate_module.runner,
                    "bind_early_evidence",
                    return_value=plan,
                ),
                patch.object(
                    aggregate_module,
                    "_authenticate_fold",
                    return_value=fold,
                ),
            ):
                return aggregate_module.authenticate_single_fold_release(
                    output,
                    expected_completion_sha256=completion_sha,
                    expected_fold_commit=NUMERICAL_COMMIT,
                    expected_config_sha256=plan.sha256,
                    expected_fold_directory=fold.path,
                )

        def release(index: int):
            case_root = root / f"csv_case_{index}"
            case_root.mkdir()
            fold = _fold(case_root, plan, "half_cylinder", 0.60)
            output = case_root / "release"
            _aggregate(plan, (fold,), output)
            return fold, output

        genuine_fold, genuine_output = release(0)
        genuine_sha = sha256_file(genuine_output / "AGGREGATE_COMPLETE.json")
        assert authenticate(genuine_output, genuine_fold, genuine_sha)[
            "stop_version"
        ] is False

        for index, mutation in enumerate(
            (
                "changed_metric",
                "noncanonical_count",
                "duplicate_row",
                "reordered_fields",
            ),
            start=1,
        ):
            fold, output = release(index)
            report = _read_json(
                output / "single_fold_authentication_report.json"
            )
            row = dict(report["fold"])
            table_path = output / "outer_family_summary.csv"
            table_path.unlink()
            if mutation == "changed_metric":
                row["f1"] = 0.61
                table_sha = parent._atomic_csv(
                    table_path,
                    aggregate_module.FAMILY_SUMMARY_FIELDS,
                    [row],
                )
            elif mutation == "noncanonical_count":
                row["sample_count"] = "8.0"
                table_sha = parent._atomic_csv(
                    table_path,
                    aggregate_module.FAMILY_SUMMARY_FIELDS,
                    [row],
                )
            elif mutation == "duplicate_row":
                table_sha = parent._atomic_csv(
                    table_path,
                    aggregate_module.FAMILY_SUMMARY_FIELDS,
                    [row, row],
                )
            else:
                table_sha = parent._atomic_csv(
                    table_path,
                    tuple(reversed(aggregate_module.FAMILY_SUMMARY_FIELDS)),
                    [row],
                )

            def bind_table(_completion, changed_report, changed_manifest):
                changed_report["outer_family_summary_file_sha256"] = table_sha
                changed_manifest["outer_family_summary_file_sha256"] = table_sha

            completion_sha = _resign_release_chain(output, bind_table)
            try:
                authenticate(output, fold, completion_sha)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"self-consistently re-signed summary CSV accepted: {mutation}"
                )


def test_single_fold_release_rejects_resigned_publication_field_forgery() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def authenticate(output: Path, fold, completion_sha: str) -> None:
            with (
                patch.object(aggregate_module.runner, "load_plan", return_value=plan),
                patch.object(
                    aggregate_module.runner,
                    "bind_early_evidence",
                    return_value=plan,
                ),
                patch.object(
                    aggregate_module,
                    "_authenticate_fold",
                    return_value=fold,
                ),
            ):
                aggregate_module.authenticate_single_fold_release(
                    output,
                    expected_completion_sha256=completion_sha,
                    expected_fold_commit=NUMERICAL_COMMIT,
                    expected_config_sha256=plan.sha256,
                    expected_fold_directory=fold.path,
                )

        cases = (
            (
                "evidence_scope",
                lambda output, _completion, report, _manifest: report.__setitem__(
                    "evidence_scope", "sealed_confirmation"
                ),
            ),
            (
                "fold_summary_source",
                lambda output, _completion, report, _manifest: report.__setitem__(
                    "fold_summary_source", "persisted_release_only"
                ),
            ),
            (
                "input_manifest_sha256",
                lambda output, _completion, report, _manifest: report.__setitem__(
                    "input_manifest_sha256", "a" * 64
                ),
            ),
            (
                "input_manifest_rows_sha256",
                lambda output, _completion, report, _manifest: report.__setitem__(
                    "input_manifest_rows_sha256", "b" * 64
                ),
            ),
            (
                "summary_path",
                lambda output, _completion, _report, manifest: manifest.__setitem__(
                    "outer_family_summary_file", "early_stop_certificate.json"
                ),
            ),
            (
                "report_path_and_sha",
                lambda output, _completion, _report, manifest: manifest.update(
                    {
                        "report_file": "early_stop_certificate.json",
                        "report_file_sha256": sha256_file(
                            output / "early_stop_certificate.json"
                        ),
                    }
                ),
            ),
        )
        for index, (name, mutation) in enumerate(cases):
            case_root = root / f"publication_case_{index}_{name}"
            case_root.mkdir()
            fold = _fold(case_root, plan, "half_cylinder", 0.60)
            output = case_root / "release"
            _aggregate(plan, (fold,), output)

            def change(completion, report, manifest):
                mutation(output, completion, report, manifest)

            completion_sha = _resign_release_chain(output, change)
            try:
                authenticate(output, fold, completion_sha)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"self-consistently re-signed publication field accepted: {name}"
                )


def test_single_fold_release_rejects_resigned_schema_and_type_mutations() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def resign_release(output: Path, certificate_change, payload_change) -> str:
            completion_path = output / "AGGREGATE_COMPLETE.json"
            report_path = output / "single_fold_authentication_report.json"
            manifest_path = output / "aggregate_manifest.json"
            certificate_path = output / "early_stop_certificate.json"
            completion = _read_json(completion_path)
            report = _read_json(report_path)
            manifest = _read_json(manifest_path)
            certificate = _read_json(certificate_path)
            certificate_change(certificate)
            certificate, certificate_sha = _rewrite_self_hashed(
                certificate_path, certificate
            )
            certificate_record = {
                "path": "early_stop_certificate.json",
                "size_bytes": certificate_path.stat().st_size,
                "sha256": certificate_sha,
                "content_sha256": certificate["content_sha256"],
            }
            completion["early_stop_certificate"] = dict(certificate_record)
            report["early_stop_certificate"] = dict(certificate_record)
            manifest["early_stop_certificate"] = dict(certificate_record)
            payload_change(completion, report, manifest)
            report, report_sha = _rewrite_self_hashed(report_path, report)
            manifest, manifest_sha = _rewrite_self_hashed(manifest_path, manifest)
            completion["report_file_sha256"] = report_sha
            completion["aggregate_manifest_file_sha256"] = manifest_sha
            _, completion_sha = _rewrite_self_hashed(completion_path, completion)
            return completion_sha

        no_certificate_change = lambda _certificate: None
        cases = (
            (
                "completion_extra",
                no_certificate_change,
                lambda completion, _report, _manifest: completion.__setitem__(
                    "unfrozen", "extra"
                ),
            ),
            (
                "report_extra",
                no_certificate_change,
                lambda _completion, report, _manifest: report.__setitem__(
                    "unfrozen", "extra"
                ),
            ),
            (
                "manifest_extra",
                no_certificate_change,
                lambda _completion, _report, manifest: manifest.__setitem__(
                    "unfrozen", "extra"
                ),
            ),
            (
                "completion_worktree_bool_as_int",
                no_certificate_change,
                lambda completion, _report, _manifest: completion.__setitem__(
                    "aggregator_worktree_clean", 1
                ),
            ),
            (
                "report_stop_bool_as_int",
                no_certificate_change,
                lambda _completion, report, _manifest: report.__setitem__(
                    "stop_version", 0
                ),
            ),
            (
                "source_artifact_count_float",
                no_certificate_change,
                lambda _completion, _report, manifest: manifest["source_folds"][
                    0
                ].__setitem__("artifact_count", 13.0),
            ),
            (
                "source_artifact_size_string",
                no_certificate_change,
                lambda _completion, _report, manifest: manifest["source_folds"][
                    0
                ]["artifacts"]["outer_summary.json"].__setitem__(
                    "size_bytes",
                    str(
                        manifest["source_folds"][0]["artifacts"][
                            "outer_summary.json"
                        ]["size_bytes"]
                    ),
                ),
            ),
            (
                "three_certificate_record_sizes_float",
                no_certificate_change,
                lambda completion, report, manifest: [
                    payload["early_stop_certificate"].__setitem__(
                        "size_bytes",
                        float(payload["early_stop_certificate"]["size_bytes"]),
                    )
                    for payload in (completion, report, manifest)
                ],
            ),
            (
                "certificate_impossible_bool_as_int",
                lambda certificate: certificate.__setitem__(
                    "mathematically_impossible_to_pass", 0
                ),
                lambda _completion, _report, _manifest: None,
            ),
            (
                "certificate_observed_count_float",
                lambda certificate: certificate.__setitem__(
                    "observed_outer_family_count",
                    float(certificate["observed_outer_family_count"]),
                ),
                lambda _completion, _report, _manifest: None,
            ),
        )
        for index, (name, certificate_change, payload_change) in enumerate(cases):
            fold = _fold(root, plan, f"half_cylinder_{index}", 0.60)
            fold = replace(
                fold,
                outer_family="half_cylinder",
                summary=_summary(plan, "half_cylinder", 0.60),
            )
            output = root / f"release_mutation_{index}_{name}"
            _aggregate(plan, (fold,), output)
            completion_sha = resign_release(
                output, certificate_change, payload_change
            )
            with (
                patch.object(aggregate_module.runner, "load_plan", return_value=plan),
                patch.object(
                    aggregate_module.runner,
                    "bind_early_evidence",
                    return_value=plan,
                ),
                patch.object(
                    aggregate_module,
                    "_authenticate_fold",
                    return_value=fold,
                ),
            ):
                try:
                    aggregate_module.authenticate_single_fold_release(
                        output,
                        expected_completion_sha256=completion_sha,
                        expected_fold_commit=NUMERICAL_COMMIT,
                        expected_config_sha256=plan.sha256,
                        expected_fold_directory=fold.path,
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"resigned release mutation accepted: {name}")


def test_single_fold_release_fresh_replay_rejects_resigned_f1_stop_forgery() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bound = _bound_plan(plan, root)
        fold_path, fresh_rows, reference_rows = _write_full_synthetic_fold(
            root,
            bound,
            directory_name="real_low_f1_fold",
            f1=0.49,
        )
        authenticated = _authenticate_synthetic_fold(
            bound, fold_path, fresh_rows, reference_rows
        )
        assert authenticated.summary["f1"] == 0.49
        release_dir = root / "forged_release"
        _aggregate(plan, (authenticated,), release_dir)

        input_identity = {
            "schema": bound.manifest_schema,
            "path": str(bound.manifest_path),
            "size_bytes": bound.manifest_size,
            "sha256": bound.manifest_sha256,
            "rows_content_sha256": bound.manifest_rows_sha256,
        }
        genuine_completion_sha = sha256_file(
            release_dir / "AGGREGATE_COMPLETE.json"
        )
        with (
            patch.object(aggregate_module.runner, "load_plan", return_value=plan),
            patch.object(
                aggregate_module.runner,
                "bind_early_evidence",
                return_value=bound,
            ),
            patch.object(parent, "load_cache_rows", return_value=((), input_identity)),
            patch.object(
                runner,
                "evaluate_outer_prediction",
                return_value=(fresh_rows, reference_rows),
            ),
        ):
            genuine_release = aggregate_module.authenticate_single_fold_release(
                release_dir,
                expected_completion_sha256=genuine_completion_sha,
                expected_fold_commit=NUMERICAL_COMMIT,
                expected_config_sha256=plan.sha256,
                expected_fold_directory=fold_path,
            )
        assert genuine_release["stop_version"] is True

        result_sha_before = sha256_file(fold_path / "result_manifest.json")
        completion_sha_before = sha256_file(fold_path / "RUN_COMPLETE.json")

        # Forge the source summary and every release-level hash/certificate while
        # deliberately leaving source result_manifest/RUN_COMPLETE unchanged.
        source_summary_path = fold_path / "outer_summary.json"
        source_summary = _read_json(source_summary_path)
        source_summary["f1"] = 0.60
        _, source_summary_sha = _rewrite_self_hashed(
            source_summary_path, source_summary
        )

        report_path = release_dir / "single_fold_authentication_report.json"
        manifest_path = release_dir / "aggregate_manifest.json"
        completion_path = release_dir / "AGGREGATE_COMPLETE.json"
        certificate_path = release_dir / "early_stop_certificate.json"
        report = _read_json(report_path)
        manifest = _read_json(manifest_path)
        completion = _read_json(completion_path)
        report["fold"]["f1"] = 0.60
        forged_certificate = aggregate_module._self_hashed(
            aggregate_module._early_stop_certificate(
                plan,
                [report["fold"]],
                numerical_git_commit=NUMERICAL_COMMIT,
            )
        )
        certificate, certificate_sha = _rewrite_self_hashed(
            certificate_path, forged_certificate
        )
        assert certificate["stop_version"] is False
        certificate_record = {
            "path": "early_stop_certificate.json",
            "size_bytes": certificate_path.stat().st_size,
            "sha256": certificate_sha,
            "content_sha256": certificate["content_sha256"],
        }
        report["early_stop_certificate"] = dict(certificate_record)
        report["stop_version"] = False
        manifest["early_stop_certificate"] = dict(certificate_record)
        manifest["source_folds"][0]["artifacts"]["outer_summary.json"] = {
            "size_bytes": source_summary_path.stat().st_size,
            "sha256": source_summary_sha,
        }
        completion["early_stop_certificate"] = dict(certificate_record)
        _, report_sha = _rewrite_self_hashed(report_path, report)
        _, manifest_sha = _rewrite_self_hashed(manifest_path, manifest)
        completion["report_file_sha256"] = report_sha
        completion["aggregate_manifest_file_sha256"] = manifest_sha
        _, forged_completion_sha = _rewrite_self_hashed(
            completion_path, completion
        )
        assert sha256_file(fold_path / "result_manifest.json") == result_sha_before
        assert sha256_file(fold_path / "RUN_COMPLETE.json") == completion_sha_before

        replay_calls = 0
        def replay(_plan, _candidate, _staged, **_kwargs):
            nonlocal replay_calls
            replay_calls += 1
            return fresh_rows, reference_rows

        with (
            patch.object(aggregate_module.runner, "load_plan", return_value=plan),
            patch.object(
                aggregate_module.runner,
                "bind_early_evidence",
                return_value=bound,
            ),
            patch.object(parent, "load_cache_rows", return_value=((), input_identity)),
            patch.object(runner, "evaluate_outer_prediction", side_effect=replay),
        ):
            try:
                aggregate_module.authenticate_single_fold_release(
                    release_dir,
                    expected_completion_sha256=forged_completion_sha,
                    expected_fold_commit=NUMERICAL_COMMIT,
                    expected_config_sha256=plan.sha256,
                    expected_fold_directory=fold_path,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("resigned F1/stop release forgery was accepted")
        assert replay_calls == 1


def test_prediction_array_contract_and_no_replace_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        arrays = {
            name: np.zeros(3, dtype=dtype)
            for name, dtype in runner.PREDICTION_ARRAY_DTYPES.items()
        }
        valid = root / "valid.npz"
        parent._atomic_npz(valid, arrays)
        aggregate_module._authenticate_prediction_array_contract(
            aggregate_module._read_file_snapshot(valid)
        )
        invalid = root / "invalid.npz"
        parent._atomic_npz(invalid, dict(list(arrays.items())[:-1]))
        _expect_value_error(
            aggregate_module._authenticate_prediction_array_contract,
            aggregate_module._read_file_snapshot(invalid),
            contains="exact ordered 19 arrays",
        )
        destination = root / "winner.json"
        parent._atomic_json(destination, {"winner": 1})
        try:
            parent._atomic_json(destination, {"winner": 2})
        except FileExistsError:
            pass
        else:
            raise AssertionError("no-replace publication overwrote an existing winner")


def test_full_synthetic_fold_replays_before_metrics_and_rejects_support_tamper() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bound = _bound_plan(plan, root)
        fold_path, fresh_rows, reference_rows = _write_full_synthetic_fold(
            root, bound
        )
        input_identity = {
            "schema": bound.manifest_schema,
            "path": str(bound.manifest_path),
            "size_bytes": bound.manifest_size,
            "sha256": bound.manifest_sha256,
            "rows_content_sha256": bound.manifest_rows_sha256,
        }
        events: list[tuple[str, str]] = []
        original_read = aggregate_module._read_file_snapshot

        def tracked_read(path: Path):
            resolved = Path(path).resolve()
            if resolved.parent == fold_path.resolve():
                events.append(("read", resolved.name))
            return original_read(path)

        def replay(_plan, _candidate, staged, **_kwargs):
            assert _plan is bound
            assert {item.name for item in Path(staged).iterdir()} == set(
                aggregate_module.LABEL_FREE_PRE_RESULT_FILES
            )
            events.append(("replay", "outer_prediction"))
            return fresh_rows, reference_rows

        with (
            patch.object(parent, "load_cache_rows", return_value=((), input_identity)),
            patch.object(runner, "evaluate_outer_prediction", side_effect=replay),
            patch.object(
                aggregate_module, "_read_file_snapshot", side_effect=tracked_read
            ),
        ):
            fold = aggregate_module._authenticate_fold(
                bound,
                fold_path,
                device="cpu",
                expected_fold_commit=NUMERICAL_COMMIT,
            )
        replay_index = events.index(("replay", "outer_prediction"))
        for protected in (
            "result_manifest.json",
            "outer_group_metrics.csv",
            "outer_summary.json",
            "outer_reference_access_audit.json",
        ):
            assert replay_index < events.index(("read", protected))
        assert fold.outer_family == "half_cylinder"
        assert fold.core_sha256 == runner.EXPECTED_CORE_SHA256
        assert len(fold.artifact_identities) == 13
        assert parent._json_safe(fold.summary["class_conditional_support"]) == (
            fresh_rows.support_audit
        )

        changed = parent._json_safe(fresh_rows.support_audit)
        changed["families"]["delta_wing"]["negative_retrieval_count"] -= 1
        bad_path, bad_rows, bad_reference = _write_full_synthetic_fold(
            root,
            bound,
            directory_name="tampered_support_fold",
            persisted_support=changed,
        )

        def replay_bad(_plan, _candidate, _staged, **_kwargs):
            return bad_rows, bad_reference

        with (
            patch.object(parent, "load_cache_rows", return_value=((), input_identity)),
            patch.object(runner, "evaluate_outer_prediction", side_effect=replay_bad),
        ):
            _expect_value_error(
                aggregate_module._authenticate_fold,
                bound,
                bad_path,
                device="cpu",
                expected_fold_commit=NUMERICAL_COMMIT,
                contains="negative_retrieval_fraction",
            )


def test_full_fold_rejects_self_consistently_resigned_nested_type_mutations() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bound = _bound_plan(plan, root)

        def mutate_result(path: Path, change) -> None:
            result_path = path / "result_manifest.json"
            result = _read_json(result_path)
            change(result)
            _rewrite_self_hashed(result_path, result)
            _rewrite_fold_result_chain(path)

        def mutate_artifact(path: Path, name: str, change) -> None:
            artifact_path = path / name
            payload = _read_json(artifact_path)
            change(payload)
            _rewrite_self_hashed(artifact_path, payload)
            _bind_changed_artifact_into_result(path, name)

        cases = (
            (
                "artifact_size_float",
                lambda path: mutate_result(
                    path,
                    lambda result: result["artifacts"][
                        "outer_group_metrics.csv"
                    ].__setitem__(
                        "size_bytes",
                        float(
                            result["artifacts"]["outer_group_metrics.csv"][
                                "size_bytes"
                            ]
                        ),
                    ),
                ),
            ),
            (
                "input_manifest_size_float",
                lambda path: mutate_result(
                    path,
                    lambda result: result["input_manifest"].__setitem__(
                        "size_bytes", float(result["input_manifest"]["size_bytes"])
                    ),
                ),
            ),
            (
                "early_sidecar_count_float",
                lambda path: mutate_result(
                    path,
                    lambda result: result["early_evidence"][
                        "sidecar_population_manifest"
                    ].__setitem__(
                        "sidecar_count",
                        float(
                            result["early_evidence"][
                                "sidecar_population_manifest"
                            ]["sidecar_count"]
                        ),
                    ),
                ),
            ),
            (
                "environment_extra",
                lambda path: mutate_result(
                    path,
                    lambda result: result["environment"].__setitem__(
                        "unfrozen", "extra"
                    ),
                ),
            ),
            (
                "result_candidate_k_float",
                lambda path: mutate_result(
                    path,
                    lambda result: result["selected_candidate"].__setitem__(
                        "k", float(result["selected_candidate"]["k"])
                    ),
                ),
            ),
            (
                "result_candidate_k_bool",
                lambda path: mutate_result(
                    path,
                    lambda result: result["selected_candidate"].__setitem__(
                        "k", True
                    ),
                ),
            ),
            (
                "result_candidate_sigma_int",
                lambda path: mutate_result(
                    path,
                    lambda result: result["selected_candidate"].__setitem__(
                        "sigma", 0
                    ),
                ),
            ),
            (
                "candidate_k_float",
                lambda path: mutate_artifact(
                    path,
                    "selected_candidate.json",
                    lambda selected: selected["candidate"].__setitem__(
                        "k", float(selected["candidate"]["k"])
                    ),
                ),
            ),
            (
                "selected_candidate_unknown_field",
                lambda path: mutate_artifact(
                    path,
                    "selected_candidate.json",
                    lambda selected: selected.__setitem__("unexpected", "forbidden"),
                ),
            ),
            (
                "selected_candidate_missing_field",
                lambda path: mutate_artifact(
                    path,
                    "selected_candidate.json",
                    lambda selected: selected.pop("created_utc"),
                ),
            ),
            (
                "method_probability_bool_as_int",
                lambda path: mutate_artifact(
                    path,
                    "selected_candidate.json",
                    lambda selected: selected[runner.METHOD_BINDING_KEY][
                        "score"
                    ].__setitem__("probability_claim", 0),
                ),
            ),
            (
                "method_threshold_string_as_bool",
                lambda path: mutate_artifact(
                    path,
                    "selected_candidate.json",
                    lambda selected: selected[runner.METHOD_BINDING_KEY][
                        "threshold"
                    ].__setitem__("candidate_id_encodes_comparator", 1),
                ),
            ),
            (
                "method_fit_k_int_as_float",
                lambda path: mutate_artifact(
                    path,
                    "selected_candidate.json",
                    lambda selected: selected[runner.METHOD_BINDING_KEY]["fit"][
                        "k"
                    ].__setitem__(
                        0,
                        float(selected[runner.METHOD_BINDING_KEY]["fit"]["k"][0]),
                    ),
                ),
            ),
            (
                "summary_sample_count_float",
                lambda path: mutate_artifact(
                    path,
                    "outer_summary.json",
                    lambda summary: summary.__setitem__(
                        "sample_count", float(summary["sample_count"])
                    ),
                ),
            ),
            (
                "support_required_count_float",
                lambda path: mutate_artifact(
                    path,
                    "outer_summary.json",
                    lambda summary: summary["class_conditional_support"].__setitem__(
                        "required_joint_family_count",
                        float(
                            summary["class_conditional_support"][
                                "required_joint_family_count"
                            ]
                        ),
                    ),
                ),
            ),
            (
                "reference_row_count_float",
                lambda path: mutate_artifact(
                    path,
                    "outer_reference_access_audit.json",
                    lambda audit: audit.__setitem__(
                        "row_count", float(audit["row_count"])
                    ),
                ),
            ),
            (
                "reference_source_ordinal_float",
                lambda path: mutate_artifact(
                    path,
                    "outer_reference_access_audit.json",
                    lambda audit: audit["rows"][0].__setitem__(
                        "source_ordinal", float(audit["rows"][0]["source_ordinal"])
                    ),
                ),
            ),
        )
        for index, (name, mutate) in enumerate(cases):
            fold_path, fresh_rows, reference_rows = _write_full_synthetic_fold(
                root,
                bound,
                directory_name=f"typed_mutation_{index}_{name}",
            )
            mutate(fold_path)
            try:
                _authenticate_synthetic_fold(
                    bound, fold_path, fresh_rows, reference_rows
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"resigned nested mutation was accepted: {name}")


def _run_without_pytest() -> None:
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"PASS {name}")


if __name__ == "__main__":
    _run_without_pytest()

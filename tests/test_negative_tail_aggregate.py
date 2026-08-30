from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT, ROOT / "tests"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.portable_flow import sha256_file
from scripts import aggregate_verify_negative_tail_calibration_1_1 as aggregate_module
from scripts import run_verify_negative_tail_calibration_1_1 as runner
from test_negative_tail_runner import (
    _projection,
    _write_synthetic_inner_evidence,
    _write_synthetic_input_manifest,
)


CONFIG = ROOT / "config" / "Verify_NegativeTailCalibration_1.1.yaml"
NUMERICAL_COMMIT = "1" * 40
AGGREGATOR_COMMIT = "2" * 40
INPUT_SHA = "3" * 64
INPUT_ROWS_SHA = "4" * 64


def _expect_value_error(function, *args, contains: str | None = None, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        if contains is not None:
            assert contains in str(error), str(error)
        return
    raise AssertionError("expected ValueError")


def _build_complete_synthetic_fold(
    root: Path,
) -> tuple[runner.Plan, Path, dict[str, object]]:
    base_plan = runner.load_plan(CONFIG)
    family = "half_cylinder"
    selected = runner.TailCandidateSpec(
        "fmt161",
        1,
        0.0,
        "calibrated_tail_anomaly_threshold",
        0.5,
    )
    fit = _projection(
        dataset="channel",
        family="channel",
        values=[0.0, 1.0, 2.0, 3.0],
        scales=[0, 0, 1000, 1000],
        centers=[10, 11, 12, 13],
        labels=[False, False, False, False],
    )
    model = runner._fit_tail_model(
        [fit], "fmt161", base_plan, device="cpu", ks=(1,)
    )
    plan = _write_synthetic_input_manifest(root, base_plan, outer_family=family)
    fold = root / "fold_half_cylinder"
    fold.mkdir()
    (
        inner_metrics_path,
        inner_metrics_sha,
        inner_summary_path,
        inner_summary_sha,
        inner_audits_path,
        inner_audits_sha,
        selected_summary,
    ) = _write_synthetic_inner_evidence(
        fold,
        plan,
        selected,
        outer_family=family,
    )
    cache_rows, input_manifest_identity = runner.load_cache_rows(plan)
    outer_rows = [row for row in cache_rows if row.family == family]
    outer_projections = [
        runner.load_cache_projection(plan, row, include_labels=False)
        for row in outer_rows
    ]
    calibration_path, calibration_manifest_path, _, calibration_manifest_sha = (
        runner.write_final_calibration_artifact(
            fold,
            model,
            plan=plan,
            selected=selected,
            outer_family=family,
            fit_families=("delta_wing", "f22_raptor", "channel", "boeing_747"),
            git_commit=NUMERICAL_COMMIT,
        )
    )
    calibration = runner.authenticate_and_rebuild_final_calibration(
        calibration_path,
        calibration_manifest_path,
        plan=plan,
        selected=selected,
        outer_family=family,
        fit_families=("delta_wing", "f22_raptor", "channel", "boeing_747"),
        git_commit=NUMERICAL_COMMIT,
        expected_manifest_file_sha256=calibration_manifest_sha,
    )
    selected_path, selected_sha, selected_payload = runner.write_selected_candidate(
        fold,
        plan=plan,
        selected=selected,
        selected_summary=selected_summary,
        calibration=calibration,
        inner_group_metrics_path=inner_metrics_path,
        inner_group_metrics_sha256=inner_metrics_sha,
        inner_candidate_summary_path=inner_summary_path,
        inner_candidate_summary_sha256=inner_summary_sha,
        inner_fit_audits_path=inner_audits_path,
        inner_fit_audits_sha256=inner_audits_sha,
        outer_family=family,
        git_commit=NUMERICAL_COMMIT,
    )
    selected_artifact = runner.authenticate_selected_candidate(
        selected_path,
        plan=plan,
        selected=selected,
        calibration=calibration,
        inner_group_metrics_path=inner_metrics_path,
        inner_group_metrics_sha256=inner_metrics_sha,
        inner_candidate_summary_path=inner_summary_path,
        inner_candidate_summary_sha256=inner_summary_sha,
        inner_fit_audits_path=inner_audits_path,
        inner_fit_audits_sha256=inner_audits_sha,
        outer_family=family,
        git_commit=NUMERICAL_COMMIT,
        expected_file_sha256=selected_sha,
    )
    arrays, group_audits = runner.build_outer_prediction_arrays(
        outer_projections,
        calibration.model,
        selected,
        plan,
        device="cpu",
    )
    _, _, prediction_sha, prediction_manifest_sha = runner.write_outer_prediction(
        fold,
        arrays,
        group_audits,
        plan=plan,
        selected=selected,
        selected_artifact=selected_artifact,
        calibration=calibration,
        outer_family=family,
        git_commit=NUMERICAL_COMMIT,
    )
    metric_rows, reference_rows = runner.evaluate_outer_prediction(
        plan,
        selected,
        fold,
        outer_family=family,
        git_commit=NUMERICAL_COMMIT,
        device="cpu",
        expected_calibration_manifest_sha256=calibration_manifest_sha,
        expected_selected_candidate_sha256=selected_sha,
        expected_prediction_manifest_sha256=prediction_manifest_sha,
        inner_group_metrics_path=inner_metrics_path,
        inner_group_metrics_sha256=inner_metrics_sha,
        inner_candidate_summary_path=inner_summary_path,
        inner_candidate_summary_sha256=inner_summary_sha,
        inner_fit_audits_path=inner_audits_path,
        inner_fit_audits_sha256=inner_audits_sha,
    )
    outer_metrics_sha = runner._atomic_csv(
        fold / "outer_group_metrics.csv",
        runner.METRIC_FIELDS,
        metric_rows,
    )
    outer_summary = runner._outer_summary(metric_rows, family)
    outer_summary_sha = runner._atomic_json(
        fold / "outer_summary.json",
        runner._manifest_with_self_hash(outer_summary),
    )
    reference_audit = runner._manifest_with_self_hash(
        {
            "schema": (
                "pathline_template_matching.negative_tail_outer_reference_access.v1"
            ),
            "experiment": runner.EXPERIMENT,
            "outer_family": family,
            "first_open_phase": (
                "after_outer_prediction_file_and_manifest_authentication"
            ),
            "prediction_manifest_file_sha256": prediction_manifest_sha,
            "prediction_file_sha256": prediction_sha,
            "row_count": len(reference_rows),
            "rows": reference_rows,
        }
    )
    reference_audit_sha = runner._atomic_json(
        fold / "outer_reference_access_audit.json",
        reference_audit,
    )

    artifact_names = tuple(
        name
        for name in plan.required_fold_files
        if name not in {"result_manifest.json", "RUN_COMPLETE.json"}
    )
    result = runner._manifest_with_self_hash(
        {
            "schema": runner.RESULT_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "completed_utc": runner._utc_now(),
            "git_commit": NUMERICAL_COMMIT,
            "config_path": str(plan.path),
            "config_sha256": plan.sha256,
            "input_manifest": input_manifest_identity,
            "outer_family": family,
            "selected_candidate": runner._candidate_payload(selected),
            "selected_candidate_file": selected_path.name,
            "selected_candidate_file_sha256": selected_sha,
            "selected_candidate_content_sha256": selected_payload["content_sha256"],
            "final_calibration_manifest_file_sha256": calibration.manifest_file_sha256,
            "final_calibration_file_sha256": calibration.calibration_file_sha256,
            "prediction_manifest_file_sha256": prediction_manifest_sha,
            "prediction_file_sha256": prediction_sha,
            "inner_group_metrics_file_sha256": inner_metrics_sha,
            "inner_candidate_summary_file_sha256": inner_summary_sha,
            "inner_fit_audits_file_sha256": inner_audits_sha,
            "outer_group_metrics_file_sha256": outer_metrics_sha,
            "outer_summary_file_sha256": outer_summary_sha,
            "outer_reference_access_audit_file_sha256": reference_audit_sha,
            "outer_summary": outer_summary,
            "environment": {"requested_device": "cpu"},
            "artifacts": {
                name: {
                    "size_bytes": (fold / name).stat().st_size,
                    "sha256": sha256_file(fold / name),
                }
                for name in artifact_names
            },
        }
    )
    result_path = fold / "result_manifest.json"
    result_sha = runner._atomic_json(result_path, result)
    completion = runner._manifest_with_self_hash(
        {
            "schema": runner.COMPLETE_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "outer_family": family,
            "git_commit": NUMERICAL_COMMIT,
            "config_sha256": plan.sha256,
            "result_manifest_file": result_path.name,
            "result_manifest_file_sha256": result_sha,
            "result_manifest_content_sha256": result["content_sha256"],
            "completed_utc": runner._utc_now(),
        }
    )
    runner._atomic_json(fold / "RUN_COMPLETE.json", completion)
    assert {path.name for path in fold.iterdir()} == set(plan.required_fold_files)
    return plan, fold, outer_summary


def _synthetic_summary(family: str, f1: float) -> dict[str, object]:
    summary: dict[str, object] = {
        "schema": "pathline_template_matching.negative_tail_outer_summary.v1",
        "experiment": runner.EXPERIMENT,
        "outer_family": family,
        "group_count": 8,
    }
    summary.update({field: 0.8 for field in aggregate_module.FAMILY_METRIC_FIELDS})
    summary["f1"] = f1
    summary.update({field: 8 for field in aggregate_module.FAMILY_COUNT_FIELDS})
    return summary


def _fold_evidence(
    root: Path,
    plan: runner.Plan,
    family: str,
    *,
    f1: float,
    commit: str = NUMERICAL_COMMIT,
    config_sha256: str | None = None,
    input_sha256: str = INPUT_SHA,
    input_rows_sha256: str = INPUT_ROWS_SHA,
) -> aggregate_module.AuthenticatedFold:
    path = (root / f"fold_{family}_{len(tuple(root.iterdir()))}").resolve()
    path.mkdir()
    identities = {
        name: {"size_bytes": 1, "sha256": "5" * 64}
        for name in aggregate_module.EXPECTED_RESULT_ARTIFACTS
    }
    return aggregate_module.AuthenticatedFold(
        path=path,
        outer_family=family,
        numerical_git_commit=commit,
        config_sha256=plan.sha256 if config_sha256 is None else config_sha256,
        input_manifest_sha256=input_sha256,
        input_manifest_rows_sha256=input_rows_sha256,
        requested_device="cpu",
        selected_candidate={"candidate_id": f"candidate_{family}"},
        summary=_synthetic_summary(family, f1),
        artifact_identities=identities,
        completion_file_sha256="6" * 64,
        completion_content_sha256="7" * 64,
        result_manifest_file_sha256="8" * 64,
        result_manifest_content_sha256="9" * 64,
    )


def _aggregate_with_evidence(
    plan: runner.Plan,
    folds: tuple[aggregate_module.AuthenticatedFold, ...],
    output: Path,
    *,
    mode: str = "auto",
    expected_commit: str = NUMERICAL_COMMIT,
):
    by_path = {fold.path: fold for fold in folds}

    def authenticate(_plan, path, *, device, expected_fold_commit):
        assert _plan is plan
        assert device == "cpu"
        assert expected_fold_commit == expected_commit
        return by_path[path.resolve()]

    with (
        patch.object(aggregate_module.runner, "load_plan", return_value=plan),
        patch.object(
            aggregate_module.runner,
            "_git_identity",
            return_value=(AGGREGATOR_COMMIT, False),
        ),
        patch.object(aggregate_module, "_authenticate_fold", side_effect=authenticate),
    ):
        return aggregate_module.aggregate(
            CONFIG,
            [fold.path for fold in folds],
            output,
            expected_fold_commit=expected_commit,
            mode=mode,
            device="cpu",
        )


def test_negative_tail_aggregate_single_fold_authenticates_full_chain_and_exact_files():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan, fold, expected_summary = _build_complete_synthetic_fold(root)
        output = root / "aggregate"
        with (
            patch.object(aggregate_module.runner, "load_plan", return_value=plan),
            patch.object(
                aggregate_module.runner,
                "_git_identity",
                return_value=(AGGREGATOR_COMMIT, False),
            ),
        ):
            report = aggregate_module.aggregate(
                CONFIG,
                [fold],
                output,
                expected_fold_commit=NUMERICAL_COMMIT,
                mode="single-fold",
                device="cpu",
            )
        assert report["schema"] == aggregate_module.SINGLE_FOLD_REPORT_SCHEMA
        assert report["outer_family"] == "half_cylinder"
        assert report["five_fold_success_evaluated"] is False
        assert report["five_fold_success"] is None
        assert report["fold"]["f1"] == expected_summary["f1"]
        assert {path.name for path in output.iterdir()} == {
            "outer_family_summary.csv",
            "single_fold_authentication_report.json",
            "early_stop_certificate.json",
            "aggregate_manifest.json",
            "AGGREGATE_COMPLETE.json",
        }
        certificate = json.loads(
            (output / "early_stop_certificate.json").read_text(encoding="utf-8")
        )
        runner._authenticate_self_hash(certificate)
        assert certificate["five_fold_success_evaluated"] is False
        assert certificate["five_fold_success"] is None

        extra = fold / "unexpected.txt"
        extra.write_text("not allowed", encoding="utf-8")
        with patch.object(
            aggregate_module.runner,
            "evaluate_outer_prediction",
            side_effect=AssertionError("label gate must not run"),
        ):
            _expect_value_error(
                aggregate_module._authenticate_fold,
                plan,
                fold,
                device="cpu",
                expected_fold_commit=NUMERICAL_COMMIT,
                contains="exactly the frozen 13 files",
            )
        extra.unlink()
        prediction_path = fold / "outer_predictions.npz"
        with patch.object(
            aggregate_module.runner,
            "evaluate_outer_prediction",
            side_effect=AssertionError("label gate must not run"),
        ):
            _expect_value_error(
                aggregate_module._authenticate_fold,
                plan,
                fold,
                device="cpu",
                expected_fold_commit="a" * 40,
                contains="explicit expected commit",
            )
        original = prediction_path.read_bytes()
        prediction_path.write_bytes(original + b"damage")
        with patch.object(
            aggregate_module.runner,
            "evaluate_outer_prediction",
            side_effect=AssertionError("label gate must not run"),
        ):
            _expect_value_error(
                aggregate_module._authenticate_fold,
                plan,
                fold,
                device="cpu",
                expected_fold_commit=NUMERICAL_COMMIT,
                contains="file size mismatch",
            )


def test_negative_tail_single_fold_certificate_stops_only_for_impossibility():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        impossible = _fold_evidence(root, plan, "half_cylinder", f1=0.49)
        impossible_report = _aggregate_with_evidence(
            plan,
            (impossible,),
            root / "impossible_report",
            mode="single-fold",
        )
        assert impossible_report["stop_version"] is True
        assert impossible_report["five_fold_success_evaluated"] is False
        certificate = json.loads(
            (root / "impossible_report" / "early_stop_certificate.json").read_text(
                encoding="utf-8"
            )
        )
        assert certificate["mathematically_impossible_to_pass"] is True
        assert certificate["any_observed_family_f1_below_minimum"] is True
        assert certificate["stop_version"] is True
        assert certificate["five_fold_success"] is None

        possible = _fold_evidence(root, plan, "half_cylinder", f1=0.99)
        possible_report = _aggregate_with_evidence(
            plan,
            (possible,),
            root / "possible_report",
            mode="single-fold",
        )
        assert possible_report["stop_version"] is False
        assert possible_report["five_fold_success_evaluated"] is False
        assert possible_report["five_fold_success"] is None


def test_negative_tail_aggregate_complete_five_fold_applies_frozen_stop_rule():
    plan = runner.load_plan(CONFIG)
    f1_values = (0.80, 0.75, 0.70, 0.65, 0.60)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = tuple(
            _fold_evidence(root, plan, family, f1=f1)
            for family, f1 in zip(plan.family_order, f1_values)
        )
        with patch.object(aggregate_module.runner, "_configure_execution") as setup:
            report = _aggregate_with_evidence(
                plan,
                folds,
                root / "aggregate",
                mode="complete-five-fold",
            )
        setup.assert_called_once_with("cpu")
        assert report["schema"] == aggregate_module.AGGREGATE_SUMMARY_SCHEMA
        assert report["outer_families"] == list(plan.family_order)
        assert report["outer_family_count"] == 5
        np.testing.assert_allclose(report["family_macro"]["f1"], 0.70)
        assert all(report["success_stop_rule"]["outcomes"].values())
        assert report["all_success_conditions_pass"] is True
        assert not (root / "aggregate" / "early_stop_certificate.json").exists()
        assert {path.name for path in (root / "aggregate").iterdir()} == {
            "outer_family_summary.csv",
            "aggregate_summary.json",
            "aggregate_manifest.json",
            "AGGREGATE_COMPLETE.json",
        }


def test_negative_tail_aggregate_rejects_missing_duplicate_and_mixed_provenance():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = tuple(
            _fold_evidence(root, plan, family, f1=0.8)
            for family in plan.family_order
        )
        _expect_value_error(
            _aggregate_with_evidence,
            plan,
            folds[:4],
            root / "missing",
            mode="complete-five-fold",
            contains="exactly five folds",
        )
        _expect_value_error(
            aggregate_module.aggregate,
            CONFIG,
            [folds[0].path, folds[0].path],
            root / "duplicate_path",
            expected_fold_commit=NUMERICAL_COMMIT,
            contains="directories must be unique",
        )

        duplicate_family = replace(folds[-1], outer_family=folds[0].outer_family)
        _expect_value_error(
            _aggregate_with_evidence,
            plan,
            (*folds[:-1], duplicate_family),
            root / "duplicate_family",
            mode="complete-five-fold",
            contains="each frozen outer family exactly once",
        )
        mixed_commit = replace(folds[-1], numerical_git_commit="a" * 40)
        _expect_value_error(
            _aggregate_with_evidence,
            plan,
            (*folds[:-1], mixed_commit),
            root / "mixed_commit",
            mode="complete-five-fold",
            contains="explicit expected commit",
        )
        mixed_config = replace(folds[-1], config_sha256="b" * 64)
        _expect_value_error(
            _aggregate_with_evidence,
            plan,
            (*folds[:-1], mixed_config),
            root / "mixed_config",
            mode="complete-five-fold",
            contains="mix frozen configs",
        )
        mixed_input = replace(folds[-1], input_manifest_sha256="c" * 64)
        _expect_value_error(
            _aggregate_with_evidence,
            plan,
            (*folds[:-1], mixed_input),
            root / "mixed_input",
            mode="complete-five-fold",
            contains="mix input manifests",
        )


def test_negative_tail_aggregate_binds_expected_commit_and_refuses_overwrite():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fold = _fold_evidence(root, plan, "half_cylinder", f1=0.8)
        _expect_value_error(
            _aggregate_with_evidence,
            plan,
            (fold,),
            root / "wrong_commit",
            mode="single-fold",
            expected_commit="a" * 40,
            contains="explicit expected commit",
        )
        output = root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        _expect_value_error(
            _aggregate_with_evidence,
            plan,
            (fold,),
            output,
            mode="single-fold",
            contains="immutable output directory exists",
        )
        assert marker.read_text(encoding="utf-8") == "keep"
        assert tuple(output.iterdir()) == (marker,)
        _expect_value_error(
            _aggregate_with_evidence,
            plan,
            (fold,),
            fold.path / "nested_aggregate",
            mode="single-fold",
            contains="outside every fold directory",
        )


def test_negative_tail_aggregate_completion_is_last_and_withheld_after_output_tamper():
    plan = runner.load_plan(CONFIG)
    for corruption in ("table", "extra"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold = _fold_evidence(root, plan, "half_cylinder", f1=0.8)
            output = root / "aggregate"
            original_atomic_json = aggregate_module.runner._atomic_json

            def inject(path, value):
                digest = original_atomic_json(path, value)
                if path.name == "aggregate_manifest.json":
                    if corruption == "table":
                        with (path.parent / "outer_family_summary.csv").open(
                            "a", encoding="utf-8"
                        ) as stream:
                            stream.write("damage\n")
                    else:
                        (path.parent / "unexpected.txt").write_text(
                            "damage", encoding="utf-8"
                        )
                return digest

            with patch.object(
                aggregate_module.runner,
                "_atomic_json",
                side_effect=inject,
            ):
                _expect_value_error(
                    _aggregate_with_evidence,
                    plan,
                    (fold,),
                    output,
                    mode="single-fold",
                    contains=(
                        "file SHA-256 mismatch"
                        if corruption == "table"
                        else "pre-completion aggregate output file set drifted"
                    ),
                )
            assert not (output / "AGGREGATE_COMPLETE.json").exists()


def test_negative_tail_aggregate_matches_runner_finite_group_macro_with_nan_patterns():
    plan = runner.load_plan(CONFIG)
    family = "half_cylinder"
    rows = []
    for dataset_index, dataset in enumerate(plan.families[family]):
        for source_ordinal in range(4):
            for block in runner.BLOCK_NAMES:
                row = {
                    "outer_family": family,
                    "inner_family": "outer_evaluation_only",
                    "dataset": dataset,
                    "source_ordinal": source_ordinal,
                    "block": block,
                }
                row.update(
                    {
                        field: (
                            float("nan")
                            if field == "average_precision"
                            and source_ordinal < dataset_index
                            else 0.25 + 0.1 * dataset_index
                        )
                        for field in aggregate_module.FAMILY_METRIC_FIELDS
                    }
                )
                row.update(
                    {field: 1 for field in aggregate_module.FAMILY_COUNT_FIELDS}
                )
                rows.append(row)
    expected = runner._outer_summary(rows, family)
    actual = aggregate_module._family_summary_from_rows(plan, family, rows)
    assert actual.keys() == expected.keys()
    for field in actual:
        if isinstance(actual[field], float):
            np.testing.assert_allclose(actual[field], expected[field])
        else:
            assert actual[field] == expected[field]

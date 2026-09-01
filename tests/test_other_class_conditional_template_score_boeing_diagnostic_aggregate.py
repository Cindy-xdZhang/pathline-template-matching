from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import aggregate_other_class_conditional_template_score_boeing_diagnostic_1_1 as aggregate  # noqa: E402
from scripts import aggregate_verify_class_conditional_template_score_1_1 as base  # noqa: E402
from scripts import run_other_class_conditional_template_score_boeing_diagnostic_1_1 as runner  # noqa: E402


CONFIG = ROOT / "config" / "Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.yaml"
COMMIT = "4" * 40


def _expect_error(error_types, function, *args, contains: str | None = None, **kwargs):
    try:
        function(*args, **kwargs)
    except error_types as error:
        if contains is not None:
            assert contains in str(error), str(error)
        return error
    raise AssertionError("expected an exception")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _support(plan: runner.Plan, count: int = 10) -> dict:
    fit_families = [family for family in plan.family_order if family != "boeing_747"]
    audit = {
        "positive_retrieval_count": count,
        "positive_retrieval_fraction": 1.0,
        "positive_calibration_count": count,
        "positive_calibration_fraction": 1.0,
        "negative_retrieval_count": count,
        "negative_retrieval_fraction": 1.0,
        "negative_calibration_count": count,
        "negative_calibration_fraction": 1.0,
    }
    return {
        "schema": "pathline_template_matching.class_conditional_outer_support_summary.v1",
        "sample_count": count,
        "family_order": fit_families,
        "required_joint_family_count": 3,
        "joint_supported_family_count_histogram": {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": count,
        },
        "families": {family: dict(audit) for family in fit_families},
    }


def _summary(plan: runner.Plan) -> dict:
    metrics = {field: 0.5 for field in aggregate.FAMILY_METRIC_FIELDS}
    counts = {field: 0 for field in aggregate.FAMILY_COUNT_FIELDS}
    counts.update(
        {
            "sample_count": 10,
            "positive_count": 4,
            "negative_count": 6,
            "true_positive": 2,
            "false_positive": 2,
            "true_negative": 4,
            "false_negative": 2,
            "retrieval_supported_count": 10,
            "calibration_supported_count": 10,
            "calibration_mode_1_count": 10,
            "scaler_mode_1_count": 10,
        }
    )
    return {
        "schema": runner.OUTER_SUMMARY_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "outer_family": "boeing_747",
        **metrics,
        **counts,
        "class_conditional_support": _support(plan),
    }


def _fold(root: Path, plan: runner.Plan) -> base.AuthenticatedFold:
    fold_path = root / "fold"
    fold_path.mkdir()
    for name in runner.REQUIRED_FOLD_FILES:
        (fold_path / name).write_bytes(f"synthetic:{name}\n".encode("utf-8"))
    artifacts = {
        name: MappingProxyType(
            {
                "size_bytes": (fold_path / name).stat().st_size,
                "sha256": _sha(fold_path / name),
            }
        )
        for name in aggregate.EXPECTED_RESULT_ARTIFACTS
    }
    selected = runner.inherited._json_safe(
        runner.inherited._candidate_payload(runner.candidate_specs(plan)[0])
    )
    return base.AuthenticatedFold(
        path=fold_path.resolve(),
        outer_family="boeing_747",
        numerical_git_commit=COMMIT,
        config_sha256=plan.sha256,
        direct_parent_config_sha256=runner.EXPECTED_PARENT_CONFIG_SHA256,
        direct_parent_runner_sha256=runner.EXPECTED_PARENT_RUNNER_SHA256,
        direct_parent_aggregator_sha256=runner.EXPECTED_PARENT_AGGREGATOR_SHA256,
        core_sha256=runner.EXPECTED_CORE_SHA256,
        input_manifest_sha256=plan.manifest_sha256,
        input_manifest_rows_sha256=plan.manifest_rows_sha256,
        requested_device="cpu",
        selected_candidate=MappingProxyType(selected),
        # Match the real Verify authenticator, which recursively freezes nested
        # JSON lists into tuples before returning an AuthenticatedFold.
        summary=runner.inherited._deep_freeze(_summary(plan)),
        artifact_identities=MappingProxyType(artifacts),
        completion_file_sha256=_sha(fold_path / "RUN_COMPLETE.json"),
        completion_content_sha256="5" * 64,
        result_manifest_file_sha256=_sha(fold_path / "result_manifest.json"),
        result_manifest_content_sha256="6" * 64,
    )


def _write_fold_envelopes(
    fold_path: Path,
    plan: runner.Plan,
    *,
    outer_family: str,
    result_outer_family: str | None = None,
) -> None:
    method = aggregate._method_binding(plan, COMMIT)
    result_payload = {
        field: None for field in base.RESULT_FIELDS if field != "content_sha256"
    }
    result_payload.update(
        {
            "schema": runner.RESULT_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": aggregate.STATUS,
            "completed_utc": "2026-01-01T00:00:00+00:00",
            "config_path": str(plan.path),
            "config_sha256": plan.sha256,
            "git_commit": COMMIT,
            "outer_family": (
                outer_family if result_outer_family is None else result_outer_family
            ),
            runner.METHOD_BINDING_KEY: method,
        }
    )
    result = aggregate._self_hashed(result_payload)
    result_path = fold_path / "result_manifest.json"
    result_path.write_bytes(aggregate._canonical_json_bytes(result))

    completion_payload = {
        field: None
        for field in base.COMPLETION_FIELDS
        if field != "content_sha256"
    }
    completion_payload.update(
        {
            "schema": runner.COMPLETE_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "completed_utc": "2026-01-01T00:00:01+00:00",
            "config_sha256": plan.sha256,
            "git_commit": COMMIT,
            "outer_family": outer_family,
            "result_manifest_file": "result_manifest.json",
            "result_manifest_file_sha256": _sha(result_path),
            "result_manifest_content_sha256": result["content_sha256"],
            runner.METHOD_BINDING_KEY: method,
        }
    )
    completion = aggregate._self_hashed(completion_payload)
    (fold_path / "RUN_COMPLETE.json").write_bytes(
        aggregate._canonical_json_bytes(completion)
    )


def _bound_plan() -> tuple[runner.Plan, runner.Plan]:
    plan = runner.load_plan(CONFIG)
    bound = replace(plan, source_identity=SimpleNamespace(git_commit=COMMIT))
    return plan, bound


def _publish(root: Path):
    plan, bound = _bound_plan()
    fold = _fold(root, plan)
    output = root / "release"
    with (
        patch.object(runner, "load_plan", return_value=plan),
        patch.object(runner, "bind_early_evidence", return_value=bound),
        patch.object(runner, "_git_identity", return_value=(COMMIT, False)),
        patch.object(aggregate, "_authenticate_fold", return_value=fold),
        patch.object(base.parent, "_configure_execution"),
    ):
        report = aggregate.aggregate(
            CONFIG,
            fold.path,
            output,
            expected_fold_commit=COMMIT,
            kinematic_input_manifest_path="kinematic.json",
            kinematic_input_manifest_file_sha256="7" * 64,
            synthetic_pass_path="pass.json",
            synthetic_pass_file_sha256="8" * 64,
            sidecar_root="sidecars",
            sidecar_population_manifest_path="population.json",
            sidecar_population_manifest_file_sha256="9" * 64,
            device="cpu",
        )
    return plan, bound, fold, output, report


def _authenticate(plan, bound, fold, output):
    completion_sha = _sha(output / aggregate.COMPLETE_FILE)
    with (
        patch.object(runner, "load_plan", return_value=plan),
        patch.object(runner, "bind_early_evidence", return_value=bound),
        patch.object(aggregate, "_authenticate_fold", return_value=fold),
        patch.object(base.parent, "_configure_execution"),
    ):
        return aggregate.authenticate_diagnostic_release(
            output,
            expected_completion_sha256=completion_sha,
            expected_fold_commit=COMMIT,
            expected_config_sha256=runner.EXPECTED_CONFIG_SHA256,
            expected_fold_directory=fold.path,
        )


def test_release_contract_has_exactly_four_files_and_no_stop_success_or_macro_fields() -> None:
    assert aggregate.RELEASE_FILES == (
        "boeing_outer_summary.csv",
        "boeing_diagnostic_report.json",
        "diagnostic_manifest.json",
        "DIAGNOSTIC_COMPLETE.json",
    )
    assert not (aggregate.REPORT_FIELDS & aggregate.FORBIDDEN_RELEASE_FIELDS)
    assert not (aggregate.MANIFEST_FIELDS & aggregate.FORBIDDEN_RELEASE_FIELDS)
    assert not (aggregate.COMPLETE_FIELDS & aggregate.FORBIDDEN_RELEASE_FIELDS)
    with tempfile.TemporaryDirectory() as directory:
        _plan, _bound, _fold_value, output, report = _publish(Path(directory))
        assert {path.name for path in output.iterdir()} == set(aggregate.RELEASE_FILES)
        assert report["schema"] == aggregate.REPORT_SCHEMA
        assert report["outer_family"] == "boeing_747"
        assert report["formal_confirmation"] is False
        for name in (aggregate.REPORT_FILE, aggregate.MANIFEST_FILE, aggregate.COMPLETE_FILE):
            payload = json.loads((output / name).read_text(encoding="utf-8"))
            assert not (set(payload) & aggregate.FORBIDDEN_RELEASE_FIELDS)
            serialized = json.dumps(payload, sort_keys=True)
            assert "stop_version" not in serialized
            assert "family_macro" not in serialized


def test_support_release_boundary_restores_json_types_without_weakening_majority_gate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        plan, _bound = _bound_plan()
        fold = _fold(Path(directory), plan)
        frozen = fold.summary["class_conditional_support"]
        assert isinstance(frozen["family_order"], tuple)
        _expect_error(
            ValueError,
            base._authenticate_support_audit,
            plan,
            "boeing_747",
            frozen,
            contains="family/majority binding drifted",
        )

        restored = aggregate._support(plan, fold)
        assert isinstance(restored["family_order"], list)
        assert restored["family_order"] == [
            "half_cylinder",
            "delta_wing",
            "f22_raptor",
            "channel",
        ]
        assert restored["required_joint_family_count"] == 3

        changed = runner.inherited._json_safe(frozen)
        changed["required_joint_family_count"] = 2
        changed_fold = replace(
            fold,
            summary=runner.inherited._deep_freeze(
                {**runner.inherited._json_safe(fold.summary),
                 "class_conditional_support": changed}
            ),
        )
        _expect_error(
            ValueError,
            aggregate._support,
            plan,
            changed_fold,
            contains="family/majority binding drifted",
        )


def test_public_authenticator_reconstructs_release_and_returns_frozen_projection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        plan, bound, fold, output, _report = _publish(Path(directory))
        release = _authenticate(plan, bound, fold, output)
        assert set(release) == {
            "outer_family",
            "fold_directory",
            "fold_summary",
            "selected_candidate",
            "source_fold",
            "method_binding",
            "release_files",
        }
        assert release["outer_family"] == "boeing_747"
        assert Path(release["fold_directory"]) == fold.path
        assert release["fold_summary"]["selected_candidate_id"] == fold.selected_candidate["candidate_id"]
        assert release["selected_candidate"] == dict(fold.selected_candidate)
        assert release["method_binding"]["diagnostic_scope"]["only_outer_family"] == "boeing_747"
        assert tuple(release["release_files"]) == aggregate.RELEASE_FILES
        for name, identity in release["release_files"].items():
            assert set(identity) == {"size_bytes", "sha256"}
            assert identity["size_bytes"] == (output / name).stat().st_size
            assert identity["sha256"] == _sha(output / name)


def test_report_manifest_completion_bind_the_same_boeing_csv_and_source_fold() -> None:
    with tempfile.TemporaryDirectory() as directory:
        _plan, _bound, fold, output, _report = _publish(Path(directory))
        report = json.loads((output / aggregate.REPORT_FILE).read_text(encoding="utf-8"))
        manifest = json.loads((output / aggregate.MANIFEST_FILE).read_text(encoding="utf-8"))
        completion = json.loads((output / aggregate.COMPLETE_FILE).read_text(encoding="utf-8"))
        summary_sha = _sha(output / aggregate.SUMMARY_FILE)
        for payload in (report, manifest, completion):
            assert payload["boeing_outer_summary_file"] == aggregate.SUMMARY_FILE
            assert payload["boeing_outer_summary_file_sha256"] == summary_sha
        assert manifest["report_file"] == aggregate.REPORT_FILE
        assert manifest["report_file_sha256"] == _sha(output / aggregate.REPORT_FILE)
        assert completion["diagnostic_manifest_file"] == aggregate.MANIFEST_FILE
        assert completion["diagnostic_manifest_file_sha256"] == _sha(output / aggregate.MANIFEST_FILE)
        assert len(manifest["source_folds"]) == 1
        source = manifest["source_folds"][0]
        assert set(source) == aggregate.SOURCE_FOLD_FIELDS
        assert Path(source["run_directory"]) == fold.path
        assert source["outer_family"] == "boeing_747"
        assert source["artifact_count"] == 13
        assert set(source["artifacts"]) == set(aggregate.EXPECTED_RESULT_ARTIFACTS)


def test_public_authenticator_rejects_extra_release_file_before_fold_authentication() -> None:
    with tempfile.TemporaryDirectory() as directory:
        plan, _bound, fold, output, _report = _publish(Path(directory))
        (output / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
        with patch.object(aggregate, "_authenticate_fold") as fold_auth:
            _expect_error(
                ValueError,
                aggregate.authenticate_diagnostic_release,
                output,
                expected_completion_sha256=_sha(output / aggregate.COMPLETE_FILE),
                expected_fold_commit=COMMIT,
                expected_config_sha256=plan.sha256,
                expected_fold_directory=fold.path,
                contains="file set drifted",
            )
        fold_auth.assert_not_called()


def test_public_authenticator_rejects_tampered_report_after_fresh_fold_replay() -> None:
    with tempfile.TemporaryDirectory() as directory:
        plan, bound, fold, output, _report = _publish(Path(directory))
        report_path = output / aggregate.REPORT_FILE
        report_path.write_bytes(report_path.read_bytes() + b"\n")
        with (
            patch.object(runner, "load_plan", return_value=plan),
            patch.object(runner, "bind_early_evidence", return_value=bound),
            patch.object(aggregate, "_authenticate_fold", return_value=fold) as fold_auth,
            patch.object(base.parent, "_configure_execution"),
        ):
            _expect_error(
                (ValueError, json.JSONDecodeError),
                aggregate.authenticate_diagnostic_release,
                output,
                expected_completion_sha256=_sha(output / aggregate.COMPLETE_FILE),
                expected_fold_commit=COMMIT,
                expected_config_sha256=plan.sha256,
                expected_fold_directory=fold.path,
            )
        fold_auth.assert_called_once()


def test_fold_adapter_rejects_any_authenticated_non_boeing_fold() -> None:
    with tempfile.TemporaryDirectory() as directory:
        plan, _bound = _bound_plan()
        fold = replace(_fold(Path(directory), plan), outer_family="half_cylinder")
        _write_fold_envelopes(
            fold.path,
            plan,
            outer_family="half_cylinder",
        )
        with patch.object(base, "_authenticate_fold", return_value=fold) as fold_auth:
            _expect_error(
                ValueError,
                aggregate._authenticate_fold,
                plan,
                fold.path,
                device="cpu",
                expected_fold_commit=COMMIT,
                contains="only the Boeing outer fold",
            )
        fold_auth.assert_not_called()


def test_fold_adapter_rejects_non_boeing_result_envelope_before_verify_replay() -> None:
    with tempfile.TemporaryDirectory() as directory:
        plan, _bound = _bound_plan()
        fold = _fold(Path(directory), plan)
        _write_fold_envelopes(
            fold.path,
            plan,
            outer_family="boeing_747",
            result_outer_family="half_cylinder",
        )
        with patch.object(base, "_authenticate_fold", return_value=fold) as fold_auth:
            _expect_error(
                ValueError,
                aggregate._authenticate_fold,
                plan,
                fold.path,
                device="cpu",
                expected_fold_commit=COMMIT,
                contains="only the Boeing outer fold",
            )
        fold_auth.assert_not_called()


def test_fold_adapter_calls_verify_only_after_boeing_envelopes_are_sealed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        plan, _bound = _bound_plan()
        fold = _fold(Path(directory), plan)
        _write_fold_envelopes(
            fold.path,
            plan,
            outer_family="boeing_747",
        )
        with patch.object(base, "_authenticate_fold", return_value=fold) as fold_auth:
            observed = aggregate._authenticate_fold(
                plan,
                fold.path,
                device="cpu",
                expected_fold_commit=COMMIT,
            )
        assert observed is fold
        fold_auth.assert_called_once_with(
            plan,
            fold.path,
            device="cpu",
            expected_fold_commit=COMMIT,
        )


def test_aggregate_rejects_dirty_or_wrong_commit_before_fold_authentication() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan, bound = _bound_plan()
        fold = _fold(root, plan)
        output = root / "release"
        with (
            patch.object(runner, "load_plan", return_value=plan),
            patch.object(runner, "bind_early_evidence", return_value=bound),
            patch.object(runner, "_git_identity", return_value=(COMMIT, True)),
            patch.object(aggregate, "_authenticate_fold") as fold_auth,
        ):
            _expect_error(
                ValueError,
                aggregate.aggregate,
                CONFIG,
                fold.path,
                output,
                expected_fold_commit=COMMIT,
                kinematic_input_manifest_path="kinematic.json",
                kinematic_input_manifest_file_sha256="7" * 64,
                synthetic_pass_path="pass.json",
                synthetic_pass_file_sha256="8" * 64,
                sidecar_root="sidecars",
                sidecar_population_manifest_path="population.json",
                sidecar_population_manifest_file_sha256="9" * 64,
                contains="clean committed worktree",
            )
        fold_auth.assert_not_called()
        assert not output.exists()


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
        and callable(value)
        and not inspect.signature(value).parameters
    ]
    for test in tests:
        test()
    print(
        "other_class_conditional_template_score_boeing_diagnostic_"
        f"aggregate_tests={len(tests)}_passed"
    )

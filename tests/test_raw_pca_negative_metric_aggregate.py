from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT, ROOT / "tests"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import aggregate_verify_raw_pca_negative_metric_1_1 as aggregate_module  # noqa: E402
from scripts import run_verify_raw_pca_negative_metric_1_1 as runner  # noqa: E402


CONFIG = ROOT / "config" / "Verify_RawPCANegativeMetric_1.1.yaml"
NUMERICAL_COMMIT = "1" * 40
AGGREGATOR_COMMIT = NUMERICAL_COMMIT


def _expect_value_error(function, *args, contains: str, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert contains in str(error), str(error)
        return
    raise AssertionError("expected ValueError")


def _summary(family: str, f1: float) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "pathline_template_matching.raw_pca_negative_metric_outer_summary.v1",
        "experiment": runner.EXPERIMENT,
        "outer_family": family,
        "group_count": 8,
    }
    value.update({field: 0.8 for field in aggregate_module.FAMILY_METRIC_FIELDS})
    value["f1"] = f1
    value.update({field: 8 for field in aggregate_module.FAMILY_COUNT_FIELDS})
    return value


def _fold(
    root: Path,
    plan: runner.Plan,
    family: str,
    f1: float,
) -> aggregate_module.AuthenticatedFold:
    path = (root / f"fold_{family}").resolve()
    path.mkdir()
    identities = {
        name: {"size_bytes": 1, "sha256": "5" * 64}
        for name in aggregate_module.EXPECTED_RESULT_ARTIFACTS
    }
    return aggregate_module.AuthenticatedFold(
        path=path,
        outer_family=family,
        numerical_git_commit=NUMERICAL_COMMIT,
        config_sha256=plan.sha256,
        input_manifest_sha256="3" * 64,
        input_manifest_rows_sha256="4" * 64,
        requested_device="cpu",
        selected_candidate={"candidate_id": f"candidate_{family}"},
        summary=_summary(family, f1),
        artifact_identities=identities,
        completion_file_sha256="6" * 64,
        completion_content_sha256="7" * 64,
        result_manifest_file_sha256="8" * 64,
        result_manifest_content_sha256="9" * 64,
    )


def _aggregate(
    plan: runner.Plan,
    folds: tuple[aggregate_module.AuthenticatedFold, ...],
    output: Path,
):
    by_path = {fold.path: fold for fold in folds}

    def authenticate(_plan, path, *, device, expected_fold_commit):
        assert _plan is plan
        assert device == "cpu"
        assert expected_fold_commit == NUMERICAL_COMMIT
        return by_path[path.resolve()]

    with (
        patch.object(aggregate_module.runner, "load_plan", return_value=plan),
        patch.object(
            aggregate_module.runner,
            "_git_identity",
            return_value=(AGGREGATOR_COMMIT, False),
        ),
        patch.object(aggregate_module.runner, "_configure_execution"),
        patch.object(aggregate_module, "_authenticate_fold", side_effect=authenticate),
        patch.object(aggregate_module, "_reauthenticate_artifact_identities"),
    ):
        return aggregate_module.aggregate(
            CONFIG,
            [fold.path for fold in folds],
            output,
            expected_fold_commit=NUMERICAL_COMMIT,
            device="cpu",
        )


def test_raw_pca_aggregate_contract_and_outer_metric_gate_are_frozen():
    plan = runner.load_plan(CONFIG)
    aggregate_module._validate_plan_output_contract(plan)
    assert len(aggregate_module.EXPECTED_FOLD_FILES) == 17
    assert len(aggregate_module.EXPECTED_RESULT_ARTIFACTS) == 15
    assert set(aggregate_module.LABEL_FREE_PRE_RESULT_FILES).isdisjoint(
        {
            "result_manifest.json",
            "outer_group_metrics.csv",
            "outer_summary.json",
            "outer_reference_access_audit.json",
        }
    )
    source = inspect.getsource(aggregate_module._authenticate_fold)
    prediction_replay = source.index("authenticate_outer_prediction(")
    result_open = source.index("result_snapshot = _read_file_snapshot")
    metric_snapshot = source.index('"outer_group_metrics.csv",', result_open)
    label_open = source.index("load_outer_references_after_prediction(")
    assert prediction_replay < result_open < metric_snapshot < label_open
    assert "result_manifest.json is explicitly" in source
    assert "runner._atomic_" not in inspect.getsource(aggregate_module)


def test_raw_pca_snapshot_reader_and_hard_link_writer_refuse_replacement():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        payload = runner._manifest_with_self_hash(
            {"schema": "synthetic.no_replace.v1", "value": 7}
        )
        path = root / "artifact.json"
        digest = aggregate_module._write_json_no_replace(path, payload)
        first = aggregate_module._read_file_snapshot(path)
        assert first.sha256 == digest
        try:
            aggregate_module._write_json_no_replace(
                path,
                runner._manifest_with_self_hash(
                    {"schema": "synthetic.no_replace.v1", "value": 8}
                ),
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("no-replace writer overwrote an existing winner")
        aggregate_module._require_same_snapshot(path, first)


def test_raw_pca_aggregate_complete_five_fold_applies_frozen_stop_rule():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = tuple(
            _fold(root, plan, family, f1)
            for family, f1 in zip(
                plan.family_order, (0.80, 0.75, 0.70, 0.65, 0.60), strict=True
            )
        )
        report = _aggregate(plan, folds, root / "aggregate")
        assert report["schema"] == aggregate_module.AGGREGATE_SUMMARY_SCHEMA
        assert report["outer_families"] == list(plan.family_order)
        np.testing.assert_allclose(report["family_macro"]["f1"], 0.70)
        assert all(report["success_stop_rule"]["outcomes"].values())
        assert report["all_success_conditions_pass"] is True
        assert {path.name for path in (root / "aggregate").iterdir()} == {
            "outer_family_summary.csv",
            "aggregate_summary.json",
            "aggregate_manifest.json",
            "AGGREGATE_COMPLETE.json",
        }


def test_raw_pca_single_fold_authentication_emits_no_success_claim_and_continue_certificate():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = _fold(root, plan, plan.family_order[0], 0.60)
        report = _aggregate(plan, (first,), root / "single")
        assert report["schema"] == aggregate_module.SINGLE_FOLD_REPORT_SCHEMA
        assert report["mode"] == "single_fold_authentication"
        assert report["outer_family"] == "half_cylinder"
        assert report["five_fold_success_evaluated"] is False
        assert report["five_fold_success"] is None
        assert report["stop_version"] is False
        certificate, _ = aggregate_module._load_self_hashed_json(
            root / "single" / "early_stop_certificate.json"
        )
        assert certificate["schema"] == aggregate_module.EARLY_STOP_CERTIFICATE_SCHEMA
        assert certificate["observed_outer_families"] == ["half_cylinder"]
        assert certificate["stop_version"] is False
        assert {path.name for path in (root / "single").iterdir()} == {
            "outer_family_summary.csv",
            "early_stop_certificate.json",
            "single_fold_authentication_report.json",
            "aggregate_manifest.json",
            "AGGREGATE_COMPLETE.json",
        }


def test_raw_pca_single_fold_below_minimum_emits_stop_certificate():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = _fold(root, plan, plan.family_order[0], 0.49)
        report = _aggregate(plan, (first,), root / "single_stop")
        certificate, _ = aggregate_module._load_self_hashed_json(
            root / "single_stop" / "early_stop_certificate.json"
        )
        assert report["stop_version"] is True
        assert certificate["stop_version"] is True
        assert certificate["any_observed_family_f1_below_minimum"] is True
        assert certificate["impossibility_reasons"] == [
            "observed_family_f1_below_frozen_minimum"
        ]


def test_raw_pca_final_authentication_failure_preserves_published_evidence():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = tuple(
            _fold(root, plan, family, 0.8) for family in plan.family_order
        )
        original_authenticate = aggregate_module._authenticate_published_output
        calls = 0

        def fail_after_completion(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("synthetic final authentication failure")
            return original_authenticate(*args, **kwargs)

        with patch.object(
            aggregate_module,
            "_authenticate_published_output",
            side_effect=fail_after_completion,
        ):
            try:
                _aggregate(plan, folds, root / "aggregate")
            except RuntimeError as error:
                assert "synthetic final authentication failure" in str(error)
            else:
                raise AssertionError("expected synthetic final authentication failure")

        assert calls == 4
        assert (root / "aggregate" / "AGGREGATE_COMPLETE.json").is_file()


def test_raw_pca_aggregate_rejects_incomplete_or_mixed_provenance():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = tuple(_fold(root, plan, family, 0.8) for family in plan.family_order)
        _expect_value_error(
            _aggregate,
            plan,
            folds[:4],
            root / "missing",
            contains="either one first-family fold or all five folds",
        )
        _expect_value_error(
            _aggregate,
            plan,
            (folds[1],),
            root / "wrong_first",
            contains="restricted to the first frozen half_cylinder fold",
        )
        mixed = replace(folds[-1], config_sha256="a" * 64)
        _expect_value_error(
            _aggregate,
            plan,
            (*folds[:-1], mixed),
            root / "mixed",
            contains="mix frozen configs",
        )


def test_raw_pca_ibex_first_fold_capped_array_and_exact_commit_gates_are_frozen():
    first_source = (
        ROOT / "ibex" / "verify_raw_pca_negative_metric_1.1_first_fold.sh"
    ).read_text(encoding="utf-8")
    fold_source = (
        ROOT / "ibex" / "verify_raw_pca_negative_metric_1.1_all_folds.sh"
    ).read_text(encoding="utf-8")
    first_auth_source = (
        ROOT
        / "ibex"
        / "verify_raw_pca_negative_metric_1.1_first_fold_auth.sh"
    ).read_text(encoding="utf-8")
    aggregate_source = (
        ROOT / "ibex" / "verify_raw_pca_negative_metric_1.1_aggregate_five.sh"
    ).read_text(encoding="utf-8")
    aggregator_sha = aggregate_module.sha256_file(
        ROOT / "scripts" / "aggregate_verify_raw_pca_negative_metric_1_1.py"
    )
    for source in (first_source, fold_source, first_auth_source, aggregate_source):
        assert "#SBATCH --cpus-per-task=32" in source
        assert "#SBATCH --mem=128G" in source
        assert "#SBATCH --time=12:00:00" in source
    assert "export SLURM_ARRAY_TASK_ID=0" in first_source
    assert "#SBATCH --array=1-4%1" in fold_source
    assert "#SBATCH --array=0-4" not in fold_source
    assert 'if [[ "$COMMIT_ID" != "$EXPECTED_FOLD_COMMIT" ]]' in fold_source
    assert "result = json.loads" not in fold_source
    assert "outer_group_metrics.csv" not in fold_source
    assert "outer_summary.json" not in fold_source
    assert "load_outer_references_after_prediction" not in fold_source
    assert "FIRST_FOLD_JOB_ID" in aggregate_source
    assert "REMAINING_FOLD_ARRAY_JOB_ID" in aggregate_source
    assert "FIRST_FOLD_JOB_ID" in first_auth_source
    assert "--mode single-fold" in first_auth_source
    assert "single_fold_authentication_report.json" in fold_source
    assert "early_stop_certificate.json" in fold_source
    assert 'certificate["stop_version"] is False' in fold_source
    assert "FIRST_FOLD_AUTH_DIR" in fold_source
    assert "FIRST_FOLD_AUTH_COMPLETE_SHA256" in fold_source
    assert "--mode complete-five-fold" in aggregate_source
    assert aggregate_source.count("--run-dir") == 5
    for source in (fold_source, first_auth_source, aggregate_source):
        assert f"AGGREGATOR_SHA256={aggregator_sha}" in source
    assert 'if [[ "$COMMIT_ID" != "$EXPECTED_FOLD_COMMIT" ]]' in aggregate_source
    assert 'export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"' in fold_source
    assert 'export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"' in first_auth_source
    assert 'export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"' in aggregate_source


def test_raw_pca_artifact_map_requires_exact_fifteen_snapshots():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        snapshots = {}
        result: dict[str, object] = {"artifact_count": 15, "artifacts": {}}
        for index, name in enumerate(aggregate_module.EXPECTED_RESULT_ARTIFACTS):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes([index + 1]))
            snapshot = aggregate_module._read_file_snapshot(path)
            snapshots[name] = snapshot
            result["artifacts"][name] = {
                "size_bytes": snapshot.identity.size,
                "sha256": snapshot.sha256,
            }
            result[aggregate_module.DIRECT_ARTIFACT_HASH_FIELDS[name]] = snapshot.sha256
        identities = aggregate_module._artifact_identities(root, result, snapshots)
        assert len(identities) == 15
        missing = dict(result)
        missing_artifacts = dict(result["artifacts"])
        missing_artifacts.pop(aggregate_module.EXPECTED_RESULT_ARTIFACTS[-1])
        missing["artifacts"] = missing_artifacts
        _expect_value_error(
            aggregate_module._artifact_identities,
            root,
            missing,
            snapshots,
            contains="exactly 15",
        )

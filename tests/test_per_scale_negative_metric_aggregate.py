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

from scripts import aggregate_verify_per_scale_negative_metric_1_1 as aggregate_module  # noqa: E402
from scripts import run_verify_per_scale_negative_metric_1_1 as runner  # noqa: E402


CONFIG = ROOT / "config" / "Verify_PerScaleNegativeMetric_1.1.yaml"
NUMERICAL_COMMIT = "1" * 40
AGGREGATOR_COMMIT = "2" * 40


def _expect_value_error(function, *args, contains: str, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert contains in str(error), str(error)
        return
    raise AssertionError("expected ValueError")


def _summary(family: str, f1: float) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "pathline_template_matching.per_scale_negative_metric_outer_summary.v1",
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
    *,
    mode: str,
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
        patch.object(aggregate_module, "_authenticate_fold", side_effect=authenticate),
    ):
        return aggregate_module.aggregate(
            CONFIG,
            [fold.path for fold in folds],
            output,
            expected_fold_commit=NUMERICAL_COMMIT,
            mode=mode,
            device="cpu",
        )


def test_per_scale_aggregate_contract_and_double_artifact_gate_are_frozen():
    plan = runner.load_plan(CONFIG)
    aggregate_module._validate_plan_output_contract(plan)
    assert len(aggregate_module.EXPECTED_FOLD_FILES) == 15
    assert len(aggregate_module.EXPECTED_RESULT_ARTIFACTS) == 13
    source = inspect.getsource(aggregate_module._authenticate_fold)
    scaler_position = source.index("authenticate_and_rebuild_final_scaler")
    tail_position = source.index("authenticate_and_rebuild_final_calibration")
    label_gate_position = source.index("evaluate_outer_prediction")
    assert scaler_position < tail_position < label_gate_position
    assert "expected_scaler_manifest_sha256" in source
    assert "scaler=scaler" in source


def test_per_scale_ibex_aggregate_requires_exact_fold_checkout_and_job_local_numba_cache():
    aggregate_source = (
        ROOT / "ibex" / "verify_per_scale_negative_metric_1.1_aggregate_five.sh"
    ).read_text(encoding="utf-8")
    fold_source = (
        ROOT / "ibex" / "verify_per_scale_negative_metric_1.1_all_folds.sh"
    ).read_text(encoding="utf-8")
    commit_read = aggregate_source.index(
        "COMMIT_ID=$(git rev-parse --verify HEAD^{commit})"
    )
    exact_commit_gate = aggregate_source.index(
        'if [[ "$COMMIT_ID" != "$EXPECTED_FOLD_COMMIT" ]]'
    )
    aggregator_call = aggregate_source.index('python "$AGGREGATOR"')
    assert commit_read < exact_commit_gate < aggregator_call
    assert 'JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_per_scale_aggregate_${SLURM_JOB_ID}"' in aggregate_source
    assert 'export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"' in aggregate_source
    frozen_pattern = (
        'slurm_${SLURM_ARRAY_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${OUTER_FAMILY}'
    )
    aggregate_pattern = (
        'slurm_${FOLD_ARRAY_JOB_ID}_${TASK_ID}_${FOLD_SHORT_COMMIT}_outer_${FAMILY}'
    )
    assert frozen_pattern in fold_source
    assert aggregate_pattern in aggregate_source
    assert "_cpu_${SHORT_COMMIT}_outer_" not in fold_source
    assert "_cpu_${FOLD_SHORT_COMMIT}_outer_" not in aggregate_source


def test_per_scale_aggregate_complete_five_fold_applies_frozen_stop_rule():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = tuple(
            _fold(root, plan, family, f1)
            for family, f1 in zip(
                plan.family_order, (0.80, 0.75, 0.70, 0.65, 0.60)
            )
        )
        report = _aggregate(
            plan,
            folds,
            root / "aggregate",
            mode="complete-five-fold",
        )
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


def test_per_scale_aggregate_rejects_incomplete_or_mixed_provenance():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = tuple(_fold(root, plan, family, 0.8) for family in plan.family_order)
        _expect_value_error(
            _aggregate,
            plan,
            folds[:4],
            root / "missing",
            mode="complete-five-fold",
            contains="exactly five folds",
        )
        mixed = replace(folds[-1], config_sha256="a" * 64)
        _expect_value_error(
            _aggregate,
            plan,
            (*folds[:-1], mixed),
            root / "mixed",
            mode="complete-five-fold",
            contains="mix frozen configs",
        )

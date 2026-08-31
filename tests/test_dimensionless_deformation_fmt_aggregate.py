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

from pathline_template_matching.portable_flow import sha256_file  # noqa: E402
from scripts import aggregate_verify_dimensionless_deformation_fmt_1_1 as aggregate_module  # noqa: E402
from scripts import run_verify_dimensionless_deformation_fmt_1_1 as runner  # noqa: E402
from test_dimensionless_deformation_fmt_runner import (  # noqa: E402
    _write_complete_synthetic_population,
)


CONFIG = ROOT / "config" / "Verify_DimensionlessDeformationFMT_1.1.yaml"
NUMERICAL_COMMIT = "1" * 40


def _expect_value_error(function, *args, contains: str, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert contains in str(error), str(error)
        return
    raise AssertionError("expected ValueError")


def _summary(family: str, f1: float) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": runner.OUTER_SUMMARY_SCHEMA,
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
    identities: dict[str, dict[str, object]] = {}
    for index, name in enumerate(aggregate_module.EXPECTED_FOLD_FILES):
        artifact = path / name
        artifact.write_bytes(bytes([index + 1]))
        if name in aggregate_module.EXPECTED_RESULT_ARTIFACTS:
            identities[name] = {
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
    return aggregate_module.AuthenticatedFold(
        path=path,
        outer_family=family,
        numerical_git_commit=NUMERICAL_COMMIT,
        config_sha256=plan.sha256,
        parent_config_sha256=plan.parent_experiment_config_sha256,
        core_sha256=plan.core_sha256,
        input_manifest_sha256=plan.manifest_sha256,
        input_manifest_rows_sha256=plan.manifest_rows_sha256,
        requested_device="cpu",
        selected_candidate={"candidate_id": f"candidate_{family}"},
        summary=_summary(family, f1),
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
            return_value=(NUMERICAL_COMMIT, False),
        ),
        patch.object(aggregate_module.runner, "_configure_execution"),
        patch.object(
            aggregate_module, "_authenticate_fold", side_effect=authenticate
        ),
    ):
        return aggregate_module.aggregate(
            CONFIG,
            [fold.path for fold in folds],
            output,
            expected_fold_commit=NUMERICAL_COMMIT,
            mode=mode,
            device="cpu",
        )


def test_dimensionless_aggregate_contract_and_fresh_raw_label_gate_are_frozen() -> None:
    plan = runner.load_plan(CONFIG)
    aggregate_module._validate_plan_output_contract(plan)
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
    assert "Raw672 -> dimensionless deformation -> FMT" in source
    assert "include_labels=True" not in source


def test_dimensionless_complete_five_and_single_fold_mathematical_certificates() -> None:
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
        assert report[runner.METHOD_BINDING_KEY] == runner._method_binding(
            plan, NUMERICAL_COMMIT
        )

        first = folds[0]
        single = _aggregate(plan, (first,), root / "single")
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
        low = replace(low, outer_family="half_cylinder")
        stopped = _aggregate(plan, (low,), root / "single_stop")
        assert stopped["stop_version"] is True
        stopped_certificate, _ = aggregate_module._load_self_hashed_json(
            root / "single_stop" / "early_stop_certificate.json"
        )
        assert stopped_certificate["impossibility_reasons"] == [
            "observed_family_f1_below_frozen_minimum"
        ]


def test_dimensionless_aggregate_rejects_mixed_commit_config_parent_core_and_scope() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = tuple(_fold(root, plan, family, 0.8) for family in plan.family_order)
        cases = (
            (replace(folds[-1], numerical_git_commit="2" * 40), "mix numerical"),
            (replace(folds[-1], config_sha256="a" * 64), "mix active configs"),
            (replace(folds[-1], parent_config_sha256="b" * 64), "mix parent"),
            (replace(folds[-1], core_sha256="c" * 64), "mix dimensionless"),
            (replace(folds[-1], input_manifest_sha256="d" * 64), "mix train-only"),
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


def test_dimensionless_single_fold_release_reauthenticates_real_output_chain() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fold = _fold(root, plan, "half_cylinder", 0.60)
        output = root / "single"
        _aggregate(plan, (fold,), output)
        completion_sha = sha256_file(output / "AGGREGATE_COMPLETE.json")
        release = aggregate_module.authenticate_single_fold_release(
            output,
            expected_completion_sha256=completion_sha,
            expected_fold_commit=NUMERICAL_COMMIT,
            expected_config_sha256=plan.sha256,
            expected_fold_directory=fold.path,
        )
        assert release["stop_version"] is False
        assert release["parent_config_sha256"] == runner.EXPECTED_PARENT_CONFIG_SHA256
        assert release["core_sha256"] == runner.EXPECTED_CORE_SHA256
        (fold.path / "outer_predictions.npz").write_bytes(b"tampered")
        _expect_value_error(
            aggregate_module.authenticate_single_fold_release,
            output,
            expected_completion_sha256=completion_sha,
            expected_fold_commit=NUMERICAL_COMMIT,
            expected_config_sha256=plan.sha256,
            expected_fold_directory=fold.path,
            contains="file size mismatch",
        )


def test_dimensionless_prediction_array_contract_and_no_replace_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        arrays = {
            name: np.zeros(3, dtype=dtype)
            for name, dtype in runner.PREDICTION_ARRAY_DTYPES.items()
        }
        valid = root / "valid.npz"
        np.savez_compressed(valid, **arrays)
        aggregate_module._authenticate_prediction_array_contract(
            aggregate_module._read_file_snapshot(valid)
        )
        invalid = root / "invalid.npz"
        np.savez_compressed(invalid, **dict(list(arrays.items())[:-1]))
        _expect_value_error(
            aggregate_module._authenticate_prediction_array_contract,
            aggregate_module._read_file_snapshot(invalid),
            contains="exact ordered 19 arrays",
        )
        destination = root / "winner.json"
        runner._atomic_json(destination, {"winner": 1})
        try:
            runner._atomic_json(destination, {"winner": 2})
        except FileExistsError:
            pass
        else:
            raise AssertionError("hard-link publication overwrote an existing winner")


def test_dimensionless_real_synthetic_fold_is_freshly_authenticated_end_to_end() -> None:
    plan = runner.load_plan(CONFIG)
    mini = replace(plan, ks=(1,), sigmas=(0.0,), thresholds=())
    commit = "e" * 40
    input_identity = {
        "schema": mini.manifest_schema,
        "path": str(mini.manifest_path),
        "size_bytes": mini.manifest_size,
        "sha256": mini.manifest_sha256,
        "rows_content_sha256": mini.manifest_rows_sha256,
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cache_root = root / "primitive_cache" / "train"
        cache_root.mkdir(parents=True)
        rows = _write_complete_synthetic_population(cache_root, mini)
        fold_path = root / "fold"
        with (
            patch.object(runner, "load_plan", return_value=mini),
            patch.object(
                runner,
                "load_cache_rows",
                return_value=(rows, input_identity),
            ),
            patch.object(runner, "_git_identity", return_value=(commit, False)),
        ):
            runner.run(
                runner.CONFIG_PATH,
                "half_cylinder",
                fold_path,
                device="cpu",
            )
            fold = aggregate_module._authenticate_fold(
                mini,
                fold_path,
                device="cpu",
                expected_fold_commit=commit,
            )
        assert fold.outer_family == "half_cylinder"
        assert fold.numerical_git_commit == commit
        assert fold.config_sha256 == runner.EXPECTED_CONFIG_SHA256
        assert fold.parent_config_sha256 == runner.EXPECTED_PARENT_CONFIG_SHA256
        assert fold.core_sha256 == runner.EXPECTED_CORE_SHA256
        assert len(fold.artifact_identities) == 13
        assert np.isfinite(float(fold.summary["f1"]))


def test_dimensionless_ibex_wrappers_freeze_order_cpu_architecture_and_dynamic_schemas() -> None:
    names = (
        "first_fold",
        "first_fold_auth",
        "all_folds",
        "aggregate_five",
    )
    sources = {
        name: (
            ROOT
            / "ibex"
            / f"verify_dimensionless_deformation_fmt_1.1_{name}.sh"
        ).read_text(encoding="utf-8")
        for name in names
    }
    for source in sources.values():
        assert "#SBATCH --cpus-per-task=32" in source
        assert "#SBATCH --mem=128G" in source
        assert "#SBATCH --time=12:00:00" in source
        assert "#SBATCH --constraint=cpu_amd_epyc_7702" in source
    assert "export SLURM_ARRAY_TASK_ID=0" in sources["first_fold"]
    assert "#SBATCH --array=1-4%2" in sources["all_folds"]
    assert "authenticate_single_fold_release" in sources["all_folds"]
    assert "assert release[\"stop_version\"] is False" in sources["all_folds"]
    assert "--mode single-fold" in sources["first_fold_auth"]
    assert "--mode complete-five-fold" in sources["aggregate_five"]
    assert sources["aggregate_five"].count("--run-dir") == 5
    assert "runner.REQUIRED_FOLD_FILES" in sources["all_folds"]
    assert "aggregate.AGGREGATE_COMPLETE_SCHEMA" in sources["aggregate_five"]
    for source in sources.values():
        assert "pathline_template_matching.dimensionless_deformation_fmt_" not in source

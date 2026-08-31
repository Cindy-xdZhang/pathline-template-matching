from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.portable_flow import sha256_file
from scripts import run_other_first_principles_headroom_1_1 as runner
from scripts import run_verify_early_opposite_pair_kinematics_1_1 as early_runner


CONFIG = ROOT / "config" / "Other_FirstPrinciplesHeadroom_1.1.yaml"
WRAPPER = ROOT / "ibex" / "other_first_principles_headroom_1.1.sh"


def _expect_error(error_types, message: str, function) -> None:
    try:
        function()
    except error_types as error:
        assert message.lower() in str(error).lower()
    else:
        raise AssertionError(f"expected {error_types} containing {message!r}")


def _synthetic_prediction_arrays() -> dict[str, np.ndarray]:
    count = 8
    blocks = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    centers = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    scales = np.asarray([0, 1000, 1, 1001, 2, 1002, 3, 1003], dtype=np.int32)
    scores = np.asarray([0.05, 0.85, 0.10, 0.90, 0.15, 0.95, 0.20, 1.0], dtype=np.float64)
    support = np.ones(count, dtype=bool)
    return {
        "dataset": np.full(count, "cylinder3d", dtype="<U64"),
        "source_ordinal": np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int16),
        "source_index": np.asarray([10, 10, 20, 20, 30, 30, 40, 40], dtype=np.int64),
        "scale_id": scales,
        "center_seed_index": centers,
        "scale_block_index": blocks,
        "assigned_row_index": blocks.astype(np.int64) * 64000 + centers,
        "raw_negative_distance": np.linspace(0.1, 0.8, count, dtype=np.float32),
        "tail_probability": 1.0 - scores,
        "tail_anomaly": scores.copy(),
        "spatial_score": scores.copy(),
        "spatial_denominator": np.ones(count, dtype=np.float64),
        "retrieval_supported": support.copy(),
        "calibration_supported": support.copy(),
        "spatial_imputed": np.zeros(count, dtype=bool),
        "spatial_unimputable": np.zeros(count, dtype=bool),
        "calibration_mode": np.ones(count, dtype=np.int8),
        "scaler_mode": np.ones(count, dtype=np.int8),
        "prediction": scores >= 0.8,
    }


def _write_prediction_fixture(directory: Path) -> tuple[Path, Path, str, str]:
    arrays = _synthetic_prediction_arrays()
    prediction_path = directory / "outer_predictions.npz"
    np.savez(prediction_path, **arrays)
    prediction_sha = sha256_file(prediction_path)
    manifest = early_runner._manifest_with_self_hash(
        {
            "schema": early_runner.PREDICTION_MANIFEST_SCHEMA,
            "prediction_schema": early_runner.PREDICTION_SCHEMA,
            "experiment": early_runner.EXPERIMENT,
            "config_sha256": runner.EXPECTED_EARLY_CONFIG_SHA256,
            "git_commit": runner.EXPECTED_EARLY_COMMIT,
            "outer_family": "half_cylinder",
            "selected_candidate": {
                "candidate_id": "synthetic_selected_candidate",
                "representation": "fmt161_plus_seed4",
                "k": 1,
                "sigma": 0.0,
                "decision_rule": "fixed_top_fraction",
                "decision_value": 0.05,
            },
            "prediction_file": {
                "path": prediction_path.name,
                "size_bytes": prediction_path.stat().st_size,
                "sha256": prediction_sha,
            },
            "array_count": len(arrays),
            "row_count": len(arrays["prediction"]),
            "arrays": early_runner._array_manifest(arrays),
        }
    )
    manifest_path = directory / "outer_prediction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return prediction_path, manifest_path, prediction_sha, sha256_file(manifest_path)


def test_headroom_config_is_frozen_before_score_or_reference_arrays() -> None:
    plan = runner.load_plan(CONFIG)
    assert plan.sha256 == runner.EXPECTED_CONFIG_SHA256
    assert runner.EXPECTED_CONFIG_SHA256 == sha256_file(CONFIG)
    assert len(plan.allowed_datasets) == 8
    assert plan.forbidden_datasets == ("tangaroa", "smokeBuoyancy")
    assert tuple(plan.raw["decision_arms"]["order"]) == runner.METHOD_ORDER
    assert plan.raw["freeze_timing"] == (
        "before_opening_any_spatial_score_or_outer_reference_array_for_this_diagnostic"
    )
    assert plan.raw["decision_arms"]["outer_group_oracle_max_f1"]["deployable"] is False
    assert plan.raw["superseded_pre_run_draft"]["config_sha256"] == (
        runner.SUPERSEDED_DRAFT_CONFIG_SHA256
    )
    assert plan.reporting_project_root == runner.EXPECTED_REPORTING_PROJECT_ROOT
    assert plan.early_config_absolute_path == runner.EXPECTED_EARLY_CONFIG_ABSOLUTE_PATH
    assert plan.reporting_project_root not in plan.early_config_absolute_path
    assert plan.parent_family_macro_f1 == 0.6391632765825263
    assert plan.parent_family_macro_f1_tolerance == 1.0e-12

    with tempfile.TemporaryDirectory() as temporary:
        drifted = Path(temporary) / "drifted.yaml"
        content = CONFIG.read_text(encoding="utf-8").replace(
            "label_free_exact_1d_two_means", "changed_method", 1
        )
        drifted.write_text(content, encoding="utf-8")
        _expect_error(ValueError, "config SHA-256 drifted", lambda: runner.load_plan(drifted))


def test_exact_early_config_fold_jobs_and_ibex_wrapper_are_frozen() -> None:
    plan = runner.load_plan(CONFIG)
    assert runner.require_exact_reporting_project_root(
        plan, runner.EXPECTED_REPORTING_PROJECT_ROOT
    ) == runner.EXPECTED_REPORTING_PROJECT_ROOT
    _expect_error(
        ValueError,
        "dedicated headroom checkout",
        lambda: runner.require_exact_reporting_project_root(
            plan, "/home/zhanx0o/pathline-template-matching-early-kinematics"
        ),
    )
    assert runner.require_exact_early_config_path(
        plan, runner.EXPECTED_EARLY_CONFIG_ABSOLUTE_PATH
    ) == runner.EXPECTED_EARLY_CONFIG_ABSOLUTE_PATH
    _expect_error(
        ValueError,
        "producer-checkout absolute path",
        lambda: runner.require_exact_early_config_path(
            plan, "config/Verify_EarlyOppositePairKinematics_1.1.yaml"
        ),
    )

    supplied = [
        Path("/untrusted/prefix") / plan.expected_fold_basenames[family]
        for family in reversed(runner.FAMILY_ORDER)
    ]
    ordered = runner.order_exact_fold_directories(plan, supplied)
    assert tuple(path.name for path in ordered) == tuple(
        plan.expected_fold_basenames[family] for family in runner.FAMILY_ORDER
    )
    wrong = list(supplied)
    wrong[0] = wrong[0].with_name("slurm_99999999_4_wrong_outer_boeing_747")
    _expect_error(
        ValueError,
        "basenames differ",
        lambda: runner.order_exact_fold_directories(plan, wrong),
    )

    wrapper = WRAPPER.read_text(encoding="utf-8")
    for required in (
        "#SBATCH --time=10:00:00",
        "#SBATCH --cpus-per-task=32",
        "#SBATCH --mem=128G",
        "#SBATCH --constraint=cpu_amd_epyc_7702",
        "conda activate deepvortex",
        runner.EXPECTED_REPORTING_PROJECT_ROOT,
        runner.EXPECTED_EARLY_CONFIG_ABSOLUTE_PATH,
        runner.EXPECTED_CONFIG_SHA256,
        "/usr/bin/time -v python tests/test_all.py",
        "1b9df53a9010c6c3c46345639cfbf1d5ab2fe3a43187c79c7dfa0f7d840b102f",
        "78d0990352777e488f26bb84f3b0fc16e18845fc7cedb8a7d7fc598f32c0afe3",
        "9f96835b9185218f40df4cc3c52bf3d80a93056681d922a30abfc5c0246f88a7",
        '[[ ! -e "$OUTPUT_DIR" ]]',
    ):
        assert required in wrapper
    assert (
        f"readonly PROJECT_ROOT={runner.EXPECTED_REPORTING_PROJECT_ROOT}" in wrapper
    )
    assert (
        "readonly PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-early-kinematics"
        not in wrapper
    )
    for basename in plan.expected_fold_basenames.values():
        assert basename in wrapper


def test_oracle_is_tie_aware_and_clearly_non_deployable() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
    labels = np.asarray([1, 0, 1, 0], dtype=bool)
    eligible = np.ones(4, dtype=bool)
    decision = runner.oracle_max_f1_decision(scores, eligible, labels)
    assert decision.predictions.tolist() == [True, True, True, False]
    assert "diagnostic_only_outer_label_threshold=0.699" in decision.parameter

    tied = runner.oracle_max_f1_decision(
        np.asarray([0.9, 0.8, 0.8, 0.7]),
        eligible,
        np.asarray([1, 0, 1, 0], dtype=bool),
    )
    assert tied.predictions.tolist() == [True, True, True, False]
    rows = runner.diagnose_group(
        outer_family="half_cylinder",
        dataset="cylinder3d",
        source_ordinal=0,
        source_index=0,
        block="legacy_2_1",
        selected_candidate_id="candidate",
        scores=scores,
        center_indices=np.arange(4, dtype=np.int64),
        eligible=eligible,
        current_prediction=scores >= 0.8,
        labels=labels,
        inner_prevalence=0.5,
    )
    by_method = {row["method"]: row for row in rows}
    assert by_method["outer_group_oracle_max_f1"]["outer_label_used"] is True
    assert by_method["outer_group_oracle_max_f1"]["legal_without_outer_label"] is False
    assert all(
        by_method[name]["outer_label_used"] is False
        for name in runner.METHOD_ORDER[:-1]
    )
    oracle_f1 = by_method["outer_group_oracle_max_f1"]["f1"]
    assert by_method["current_selected_prediction"]["oracle_f1_minus_current_f1"] == (
        oracle_f1 - by_method["current_selected_prediction"]["f1"]
    )
    assert by_method["current_selected_prediction"][
        "oracle_f1_minus_inner_prevalence_f1"
    ] == (oracle_f1 - by_method["inner_prevalence_top_fraction"]["f1"])
    assert by_method["current_selected_prediction"][
        "oracle_f1_minus_two_means_f1"
    ] == (oracle_f1 - by_method["label_free_exact_1d_two_means"]["f1"])


def test_label_free_two_means_and_inner_top_fraction_do_not_receive_outer_labels() -> None:
    scores = np.asarray([0.05, 0.06, 0.07, 0.08, 0.80, 0.85, 0.90, 0.95])
    eligible = np.ones(8, dtype=bool)
    first = runner.label_free_two_means_decision(scores, eligible)
    second = runner.label_free_two_means_decision(scores.copy(), eligible.copy())
    assert np.array_equal(first.predictions, second.predictions)
    assert first.predictions.tolist() == [False, False, False, False, True, True, True, True]

    # [0, 1] | [2] and [0] | [1, 2] have equal objectives.  The frozen
    # lowest-sorted-split rule must choose the latter, and equal values cannot
    # be split across clusters.
    equal_objective = runner.label_free_two_means_decision(
        np.asarray([0.0, 1.0, 2.0]), np.ones(3, dtype=bool)
    )
    assert equal_objective.predictions.tolist() == [False, True, True]
    tied_values = runner.label_free_two_means_decision(
        np.asarray([0.0, 1.0, 1.0]), np.ones(3, dtype=bool)
    )
    assert tied_values.predictions.tolist() == [False, True, True]
    assert not runner.label_free_two_means_decision(
        np.ones(3), np.ones(3, dtype=bool)
    ).predictions.any()

    top = runner.inner_prevalence_top_fraction_decision(
        scores,
        np.asarray([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64),
        eligible,
        fraction=0.25,
    )
    assert int(top.predictions.sum()) == 2
    assert np.flatnonzero(top.predictions).tolist() == [6, 7]


def test_inner_prevalence_is_equal_family_macro_not_pooled_rows() -> None:
    plan = runner.load_plan(CONFIG)
    records: list[dict[str, Any]] = []
    outer_family = "half_cylinder"
    for family in runner.FAMILY_ORDER:
        if family == outer_family:
            continue
        for dataset in plan.family_datasets[family]:
            for source in range(4):
                records.append(
                    {
                        "inner_family": family,
                        "dataset": dataset,
                        "source_ordinal": source,
                        "block": "legacy_2_1",
                        "sample_count": 10,
                        "positive_count": 8 if family == "delta_wing" else 2,
                    }
                )
                records.append(
                    {
                        "inner_family": family,
                        "dataset": dataset,
                        "source_ordinal": source,
                        "block": "expanded_3_1",
                        "sample_count": 10,
                        "positive_count": 1,
                    }
                )
    estimates, components = runner.estimate_inner_prevalence_from_rows(
        records,
        outer_family=outer_family,
        family_datasets=plan.family_datasets,
    )
    assert abs(estimates["legacy_2_1"] - 0.35) < 1e-15
    assert abs(estimates["expanded_3_1"] - 0.10) < 1e-15
    assert components["legacy_2_1"]["delta_wing"] == 0.8
    pooled = np.mean([row["positive_count"] / row["sample_count"] for row in records if row["block"] == "legacy_2_1"])
    assert abs(pooled - estimates["legacy_2_1"]) > 0.05

    _expect_error(
        ValueError,
        "incomplete",
        lambda: runner.estimate_inner_prevalence_from_rows(
            records[:-1],
            outer_family=outer_family,
            family_datasets=plan.family_datasets,
        ),
    )


def test_block_macros_required_gaps_and_parent_f1_reproduction_gate() -> None:
    plan = runner.load_plan(CONFIG)
    parent = plan.parent_family_macro_f1
    family_rows: list[dict[str, Any]] = []
    for family in runner.FAMILY_ORDER:
        for block in runner.BLOCK_NAMES:
            current_f1 = parent + (0.10 if block == "legacy_2_1" else -0.10)
            method_f1 = {
                "current_selected_prediction": current_f1,
                "inner_prevalence_top_fraction": current_f1 - 0.05,
                "label_free_exact_1d_two_means": current_f1 - 0.02,
                "outer_group_oracle_max_f1": current_f1 + 0.10,
            }
            for method in runner.METHOD_ORDER:
                family_rows.append(
                    {
                        "outer_family": family,
                        "block": block,
                        "method": method,
                        **{
                            metric: (
                                method_f1[method] if metric == "f1" else 0.5
                            )
                            for metric in runner.MACRO_METRICS
                        },
                    }
                )

    summary = runner.aggregate_summary(plan, family_rows)
    assert summary["parent_f1_reproduced"] is True
    assert abs(summary["parent_f1_reproduction_delta"]) <= 1.0e-15
    legacy = summary["block_results"]["legacy_2_1"]
    expanded = summary["block_results"]["expanded_3_1"]
    assert legacy["methods"]["current_selected_prediction"]["f1"] > parent
    assert expanded["methods"]["current_selected_prediction"]["f1"] < parent
    for block_summary in (legacy, expanded):
        assert abs(block_summary["oracle_f1_minus_current_f1"] - 0.10) < 1.0e-15
        assert abs(block_summary["oracle_f1_minus_inner_prevalence_f1"] - 0.15) < 1.0e-15
        assert abs(block_summary["oracle_f1_minus_two_means_f1"] - 0.12) < 1.0e-15

    drifted = [dict(row) for row in family_rows]
    drifted[0]["f1"] = float(drifted[0]["f1"]) + 0.01
    _expect_error(
        ValueError,
        "failed to reproduce",
        lambda: runner.aggregate_summary(plan, drifted),
    )


def test_run_orders_fresh_fold_auth_and_input_freeze_before_reference_projection() -> None:
    """Exercise orchestration with metadata-only fakes; no score or label is opened."""

    datasets = tuple(f"dataset_{index}" for index in range(5))
    family_datasets = {
        family: (datasets[index],)
        for index, family in enumerate(runner.FAMILY_ORDER)
    }
    plan = SimpleNamespace(
        path=CONFIG.resolve(),
        sha256=runner.EXPECTED_CONFIG_SHA256,
        allowed_datasets=datasets,
        forbidden_datasets=("tangaroa", "smokeBuoyancy"),
        family_datasets=family_datasets,
        reporting_project_root=runner.EXPECTED_REPORTING_PROJECT_ROOT,
        early_config_absolute_path=runner.EXPECTED_EARLY_CONFIG_ABSOLUTE_PATH,
        expected_fold_basenames={
            family: f"unused_{family}" for family in runner.FAMILY_ORDER
        },
        parent_family_macro_f1=0.6391632765825263,
        parent_family_macro_f1_tolerance=1.0e-12,
    )
    folds = tuple(
        SimpleNamespace(
            outer_family=family,
            path=Path(f"/authenticated/{family}"),
            numerical_git_commit=runner.EXPECTED_EARLY_COMMIT,
            config_sha256=runner.EXPECTED_EARLY_CONFIG_SHA256,
            selected_candidate={"candidate_id": f"candidate_{family}"},
            completion_file_sha256="a" * 64,
            completion_content_sha256="b" * 64,
            result_manifest_file_sha256="c" * 64,
            result_manifest_content_sha256="d" * 64,
            artifact_identities={
                "inner_group_metrics.csv": {"sha256": "e" * 64},
                "outer_predictions.npz": {"sha256": "f" * 64},
                "outer_prediction_manifest.json": {"sha256": "1" * 64},
            },
        )
        for family in runner.FAMILY_ORDER
    )
    events: list[str] = []

    def fake_bind(*_args, **_kwargs):
        events.append("bind_parent_sources")
        return object()

    def fake_authenticate(*_args, **_kwargs):
        events.append("early_fresh_replay_then_outer_reference_auth")
        return folds

    def fake_cache_rows(_early_plan):
        events.append("label_free_cache_manifest")
        return tuple(SimpleNamespace(dataset=value) for value in datasets), {"sha256": "2" * 64}

    def fake_begin(destination, _plan, _payload):
        events.append("input_manifest_published")
        return Path(destination), {}

    def fake_prevalence(_plan, fold):
        events.append(f"inner_prevalence:{fold.outer_family}")
        return {block: 0.05 for block in runner.BLOCK_NAMES}, []

    def fake_prediction(*_args, expected_outer_family, **_kwargs):
        events.append(f"prediction_artifact_authenticated:{expected_outer_family}")
        return SimpleNamespace()

    def fake_join(_plan, _early_plan, fold, *_args):
        assert "input_manifest_published" in events
        assert f"prediction_artifact_authenticated:{fold.outer_family}" in events
        events.append(f"outer_reference_projection:{fold.outer_family}")
        return [{} for _ in range(32)]

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "new_output"
        with patch.object(runner, "load_plan", return_value=plan), patch.object(
            runner, "_bind_authenticated_early_plan", side_effect=fake_bind
        ), patch.object(
            runner, "authenticate_folds", side_effect=fake_authenticate
        ), patch.object(
            runner.early_runner, "load_cache_rows", side_effect=fake_cache_rows
        ), patch.object(
            runner, "begin_output", side_effect=fake_begin
        ), patch.object(
            runner, "load_inner_prevalence", side_effect=fake_prevalence
        ), patch.object(
            runner, "authenticate_prediction_artifact", side_effect=fake_prediction
        ), patch.object(
            runner, "_join_and_diagnose_fold", side_effect=fake_join
        ), patch.object(
            runner, "family_block_macro_rows", return_value=[]
        ), patch.object(
            runner, "aggregate_summary", return_value={}
        ), patch.object(
            runner, "finish_output", return_value={"status": "synthetic_pass"}
        ):
            result = runner.run(
                CONFIG,
                runner.EXPECTED_EARLY_CONFIG_ABSOLUTE_PATH,
                [Path(f"/unused/{family}") for family in runner.FAMILY_ORDER],
                output,
                expected_reporting_commit="3" * 40,
                kinematic_input_manifest_path="unused_input.json",
                kinematic_input_manifest_file_sha256="4" * 64,
                synthetic_pass_path="unused_synthetic.json",
                synthetic_pass_file_sha256="5" * 64,
                sidecar_root="unused_sidecars",
                sidecar_population_manifest_path="unused_population.json",
                sidecar_population_manifest_file_sha256="6" * 64,
            )
    assert result == {"status": "synthetic_pass"}
    assert events.index("early_fresh_replay_then_outer_reference_auth") < events.index(
        "input_manifest_published"
    )
    for family in runner.FAMILY_ORDER:
        assert events.index("input_manifest_published") < events.index(
            f"prediction_artifact_authenticated:{family}"
        ) < events.index(f"outer_reference_projection:{family}")


def test_prediction_authentication_accepts_exact_19_arrays_and_rejects_tamper() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        prediction, manifest, prediction_sha, manifest_sha = _write_prediction_fixture(directory)
        authenticated = runner.authenticate_prediction_artifact(
            prediction,
            manifest,
            expected_prediction_file_sha256=prediction_sha,
            expected_manifest_file_sha256=manifest_sha,
            expected_outer_family="half_cylinder",
            dataset_names=("cylinder3d", "halfcylinderRe640", "halfcylinderRe6400"),
        )
        assert authenticated.count == 8
        assert authenticated.selected_candidate_id == "synthetic_selected_candidate"
        assert authenticated.dataset_code.dtype == np.dtype(np.int8)

        with np.load(prediction, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        arrays["spatial_score"][0] += 0.01
        np.savez(prediction, **arrays)
        _expect_error(
            ValueError,
            "SHA-256 mismatch",
            lambda: runner.authenticate_prediction_artifact(
                prediction,
                manifest,
                expected_prediction_file_sha256=prediction_sha,
                expected_manifest_file_sha256=manifest_sha,
                expected_outer_family="half_cylinder",
                dataset_names=("cylinder3d", "halfcylinderRe640", "halfcylinderRe6400"),
            ),
        )


def test_output_is_immutable_and_never_overwritten() -> None:
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "existing"
        output.mkdir()
        _expect_error(
            FileExistsError,
            "exist",
            lambda: runner.begin_output(
                output,
                plan,
                {"schema": runner.INPUT_SCHEMA, "experiment": runner.EXPERIMENT},
            ),
        )


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} headroom tests passed")

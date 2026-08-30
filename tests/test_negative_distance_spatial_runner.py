from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import yaml

from pathline_template_matching.portable_flow import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/run_other_negative_distance_spatial_1_1.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _expect_error(error_types, message: str, function) -> None:
    try:
        function()
    except error_types as error:
        assert message in str(error)
    else:
        raise AssertionError(f"expected {error_types} containing {message!r}")


def _write_input(path: Path, dataset: str, *, all_negative: bool) -> None:
    fieldnames = [
        "query_dataset",
        "query_source_ordinal",
        "query_scale_block_id",
        "query_center_seed_index",
        "fmt_nearest_negative_distance",
        "nearest_negative_distance",
        "reference_label",
        "audit_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for center in range(20):
            writer.writerow(
                {
                    "query_dataset": dataset,
                    "query_source_ordinal": 2,
                    "query_scale_block_id": "legacy_2_1",
                    "query_center_seed_index": center,
                    "fmt_nearest_negative_distance": 0.02 * center + 0.01,
                    "nearest_negative_distance": 0.03 * center + 0.005,
                    "reference_label": 0 if all_negative else int(center >= 16),
                    "audit_note": "DO_NOT_PROJECT_IN_PREDICTION_PHASE",
                }
            )


def _write_config(
    directory: Path,
    first: Path,
    second: Path,
    *,
    first_sha: str | None = None,
) -> Path:
    config = {
        "experiment": "Other_NegativeDistanceSpatial_1.1",
        "phase": "test_exposed_diagnostic",
        "inputs": [
            {
                "id": "main31",
                "path": first.name,
                "sha256": sha256_file(first) if first_sha is None else first_sha,
                "allowed_datasets": ["flow_a"],
                "columns": {
                    "dataset": "query_dataset",
                    "source_ordinal": "query_source_ordinal",
                    "block": "query_scale_block_id",
                    "center_index": "query_center_seed_index",
                    "score": "fmt_nearest_negative_distance",
                    "label": "reference_label",
                },
            },
            {
                "id": "heldout",
                "path": second.name,
                "sha256": sha256_file(second),
                "allowed_datasets": ["flow_b"],
                "columns": {
                    "dataset": "query_dataset",
                    "source_ordinal": "query_source_ordinal",
                    "block": "query_scale_block_id",
                    "center_index": "query_center_seed_index",
                    "score": "nearest_negative_distance",
                    "label": "reference_label",
                },
            },
        ],
        "grouping": ["dataset", "source", "block"],
        "rank_definition": (
            "stable_ascending_score_then_center_index_with_percentile_rank_plus_one_over_n"
        ),
        "grid_shape_zyx": [2, 2, 5],
        "gaussian": {
            "sigma_grid_indices": [0.0, 0.75],
            "truncate": 3.0,
            "mask_normalized": True,
            "no_cross_group_smoothing": True,
            "extra_audit_field": "preserved_but_not_method_defining",
        },
        "prediction_rules": ["high_score_two_means", "fixed_top_fraction"],
        "fixed_top_fraction": 0.05,
        "fixed_top_fraction_rule": (
            "rank_greater_than_one_minus_fraction_selecting_ceil_fraction_times_n"
        ),
        "metrics": [
            "accuracy",
            "average_precision",
            "f1",
            "balanced_accuracy",
            "auroc",
            "precision",
            "recall",
            "coverage",
        ],
        "aggregation": "equal_weight_dataset_source_block_groups",
        "oracle_threshold": {
            "enabled": True,
            "prediction_rule": "score_greater_than_or_equal_to_threshold",
            "tie_break": "highest_f1_then_highest_precision_then_highest_threshold",
            "may_select_or_name_main_method": False,
        },
        "extra_top_level_audit": {"allowed": True},
    }
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _fixture(directory: Path) -> tuple[Path, Path, Path]:
    first = directory / "main31.csv"
    second = directory / "heldout.csv"
    _write_input(first, "flow_a", all_negative=False)
    _write_input(second, "flow_b", all_negative=True)
    return first, second, _write_config(directory, first, second)


def test_spatial_runner_publishes_label_free_predictions_before_reference_metrics():
    runner = _load_runner("negative_distance_spatial_label_gate_test")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        _first, _second, config = _fixture(directory)
        output = directory / "run"
        projections: list[tuple[str, ...]] = []
        original_projection = runner._iter_projected_csv
        original_reference = runner._read_reference_rows

        def recording_projection(path, columns):
            projections.append(tuple(columns))
            yield from original_projection(path, columns)

        def guarded_reference(plan):
            assert (output / "predictions.csv").is_file()
            assert (output / "prediction_manifest.json").is_file()
            return original_reference(plan)

        runner._iter_projected_csv = recording_projection
        runner._read_reference_rows = guarded_reference
        try:
            marker = runner.run(config, output)
        finally:
            runner._iter_projected_csv = original_projection
            runner._read_reference_rows = original_reference

        assert marker["experiment"] == runner.EXPERIMENT
        assert marker["status"] == "complete"
        assert len(projections) == 4
        assert all("reference_label" not in columns for columns in projections[:2])
        assert all("reference_label" in columns for columns in projections[2:])

        prediction_text = (output / "predictions.csv").read_text(encoding="utf-8")
        prediction_manifest_text = (output / "prediction_manifest.json").read_text(
            encoding="utf-8"
        )
        assert "reference_label" not in prediction_text
        assert "reference_label" not in prediction_manifest_text
        assert "DO_NOT_PROJECT_IN_PREDICTION_PHASE" not in prediction_text
        assert "oracle" not in prediction_text.lower()

        prediction_manifest = json.loads(prediction_manifest_text)
        assert prediction_manifest["row_count"] == 40
        assert prediction_manifest["prediction_column_count"] == 18
        assert prediction_manifest["reference_column_projection_to_prediction_logic"] == "excluded"
        assert prediction_manifest["source_csv_contains_reference_column"] is True
        assert sha256_file(output / "predictions.csv") == prediction_manifest[
            "predictions_file_sha256"
        ]
        for name in (
            "per_group_metrics.csv",
            "aggregate_metrics.csv",
            "oracle_upper_bound.csv",
            "result_manifest.json",
            "RUN_COMPLETE.json",
        ):
            assert (output / name).is_file()

        with (output / "oracle_upper_bound.csv").open(
            "r", encoding="utf-8", newline=""
        ) as source:
            oracle_rows = list(csv.DictReader(source))
        assert oracle_rows
        assert {row["diagnostic_only"] for row in oracle_rows} == {"1"}
        assert {row["selection_data"] for row in oracle_rows} == {
            "same_group_reference"
        }


def test_spatial_runner_reports_single_class_metrics_and_finite_group_counts():
    runner = _load_runner("negative_distance_spatial_single_class_test")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        _first, _second, config = _fixture(directory)
        output = directory / "run"
        runner.run(config, output)

        with (output / "per_group_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as source:
            per_group = list(csv.DictReader(source))
        single = [row for row in per_group if row["dataset"] == "flow_b"]
        assert single and {row["single_class_group"] for row in single} == {"1"}
        assert {row["average_precision"] for row in single} == {""}
        assert {row["auroc"] for row in single} == {""}
        assert {row["balanced_accuracy"] for row in single} == {""}
        assert all(row["accuracy"] != "" and row["f1"] != "" for row in single)

        with (output / "aggregate_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as source:
            aggregate = list(csv.DictReader(source))
        overall = [
            row
            for row in aggregate
            if row["scope"] == "all_inputs"
            and row["score_variant"] == "raw_negative_distance"
            and row["prediction_rule"] == "fixed_top_fraction_0.05"
        ]
        assert len(overall) == 1
        assert overall[0]["group_count"] == "2"
        assert overall[0]["average_precision_valid_group_count"] == "1"
        assert overall[0]["f1_valid_group_count"] == "2"


def test_spatial_runner_rejects_hash_mismatch_missing_reference_and_overwrite():
    runner = _load_runner("negative_distance_spatial_fail_closed_test")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        first, second, config = _fixture(directory)

        bad_config = _write_config(
            directory, first, second, first_sha="0" * 64
        )
        bad_output = directory / "bad_hash_run"
        _expect_error(
            ValueError,
            "input SHA-256 mismatch",
            lambda: runner.run(bad_config, bad_output),
        )

        config = _write_config(directory, first, second)
        missing_output = directory / "missing_reference_run"
        original_reference = runner._read_reference_rows

        def missing_reference(plan):
            references = original_reference(plan)
            references.pop(next(iter(references)))
            return references

        runner._read_reference_rows = missing_reference
        try:
            _expect_error(
                ValueError,
                "row-key mismatch",
                lambda: runner.run(config, missing_output),
            )
        finally:
            runner._read_reference_rows = original_reference
        assert (missing_output / "predictions.csv").is_file()
        assert (missing_output / "prediction_manifest.json").is_file()
        assert not (missing_output / "per_group_metrics.csv").exists()

        complete_output = directory / "complete_run"
        runner.run(config, complete_output)
        _expect_error(
            FileExistsError,
            "already exists",
            lambda: runner.run(config, complete_output),
        )


def test_spatial_runner_is_byte_deterministic_and_oracle_ties_are_frozen():
    runner = _load_runner("negative_distance_spatial_determinism_test")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        _first, _second, config = _fixture(directory)
        first_output = directory / "run_a"
        second_output = directory / "run_b"
        runner.run(config, first_output)
        runner.run(config, second_output)
        for name in (
            "predictions.csv",
            "prediction_manifest.json",
            "per_group_metrics.csv",
            "aggregate_metrics.csv",
            "oracle_upper_bound.csv",
            "result_manifest.json",
            "RUN_COMPLETE.json",
        ):
            assert sha256_file(first_output / name) == sha256_file(second_output / name)

        threshold, prediction = runner._best_f1_upper_bound(
            labels=runner.np.asarray([1, 0, 1, 0], dtype=bool),
            scores=runner.np.asarray([0.9, 0.8, 0.7, 0.6]),
        )
        assert threshold == 0.7
        assert prediction.tolist() == [True, True, True, False]


def test_spatial_plan_rejects_method_contract_drift():
    runner = _load_runner("negative_distance_spatial_contract_test")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        _first, _second, config = _fixture(directory)
        base = yaml.safe_load(config.read_text(encoding="utf-8"))
        cases = [
            (
                "rank_definition",
                lambda payload: payload.__setitem__("rank_definition", "changed"),
            ),
            (
                "metrics",
                lambda payload: payload.__setitem__(
                    "metrics", payload["metrics"][:-1]
                ),
            ),
            (
                "aggregation",
                lambda payload: payload.__setitem__("aggregation", "pooled"),
            ),
        ]
        for index, (message, mutate) in enumerate(cases):
            payload = json.loads(json.dumps(base))
            mutate(payload)
            drifted = directory / f"drifted_{index}.yaml"
            drifted.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            _expect_error(
                ValueError,
                message,
                lambda path=drifted: runner.load_plan(path),
            )

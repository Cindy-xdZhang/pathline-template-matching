from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.portable_flow import canonical_json_sha256, sha256_file
from scripts import aggregate_verify_scale_conditioned_retrieval_1_1 as aggregate_module
from scripts.run_verify_scale_conditioned_retrieval_1_1 import (
    BLOCK_NAMES,
    EXPERIMENT,
    load_plan,
)


CONFIG_PATH = ROOT / "config" / "Verify_ScaleConditionedRetrieval_1.1.yaml"
SYNTHETIC_COMMIT = "1" * 40
HARDENED_AGGREGATOR_COMMIT = "a" * 40


def _expect_value_error(function, *args, contains: str | None = None, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        if contains is not None:
            assert contains in str(error), str(error)
        return
    raise AssertionError("expected ValueError")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate(family: str) -> dict[str, float | int | str]:
    return {
        "candidate_id": f"candidate_{family}",
        "representation": "fmt161",
        "k": 1,
        "sigma": 0.0,
        "decision_rule": "rank_threshold",
        "decision_value": 0.50,
    }


def _group_metrics(f1: float) -> dict[str, float | int | None]:
    positive_count = 40
    negative_count = 60
    true_positive = int(round(f1 * positive_count))
    false_negative = positive_count - true_positive
    false_positive = false_negative
    true_negative = negative_count - false_positive
    accuracy = (true_positive + true_negative) / 100
    balanced_accuracy = 0.5 * (
        true_positive / positive_count + true_negative / negative_count
    )
    return {
        "accuracy": accuracy,
        "average_precision": 0.66,
        "auroc": 0.76,
        "precision": f1,
        "recall": f1,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "retrieval_support_fraction": 1.0,
        "spatial_imputed_fraction": 0.0,
        "spatial_unimputable_fraction": 0.0,
        "supported_subset_f1": f1,
        "imputed_subset_f1": None,
        "unimputable_subset_f1": None,
        "sample_count": 100,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "supported_count": 100,
        "imputed_count": 0,
        "unimputable_count": 0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def _write_outer_group_metrics(
    path: Path,
    *,
    family: str,
    datasets: tuple[str, ...],
    f1: float,
    candidate: dict[str, float | int | str],
) -> list[dict[str, object]]:
    metric_payload = _group_metrics(f1)
    rows = [
        {
            "outer_family": family,
            "physical_family": family,
            "dataset": dataset,
            "source_ordinal": source_ordinal,
            "block": block,
            **candidate,
            **metric_payload,
        }
        for dataset in datasets
        for source_ordinal in range(4)
        for block in BLOCK_NAMES
    ]
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if value is None else value
                    for field, value in row.items()
                }
            )
    return rows


def _outer_summary(
    family: str,
    datasets: tuple[str, ...],
    f1: float,
    candidate: dict[str, float | int | str],
) -> dict[str, object]:
    group = _group_metrics(f1)
    group_count = len(datasets) * 4 * len(BLOCK_NAMES)
    summary: dict[str, object] = {
        "outer_family": family,
        "candidate": candidate,
        "group_count": group_count,
        "dataset_count": len(datasets),
    }
    summary.update(
        {
            field: group[field]
            for field in aggregate_module.OUTER_GROUP_MACRO_FIELDS
        }
    )
    summary.update(
        {
            field: int(group[field]) * group_count
            for field in aggregate_module.OUTER_COUNT_FIELDS
        }
    )
    summary["aggregation"] = "equal_dataset_source_block_groups_within_outer_family"
    return summary


def _write_fold(
    root: Path,
    *,
    ordinal: int,
    family: str,
    config_sha256: str,
    f1: float,
    git_commit: str = SYNTHETIC_COMMIT,
) -> Path:
    plan = load_plan(CONFIG_PATH)
    fold = root / f"fold_{ordinal}_{family}"
    fold.mkdir()
    candidate = _candidate(family)
    group_metrics_path = fold / "outer_group_metrics.csv"
    _write_outer_group_metrics(
        group_metrics_path,
        family=family,
        datasets=plan.families[family],
        f1=f1,
        candidate=candidate,
    )
    payload_path = fold / "outer_metrics.json"
    payload_path.write_bytes(f"synthetic metrics for {family}\n".encode("utf-8"))
    group_metrics_sha256 = sha256_file(group_metrics_path)
    result = {
        "schema": "pathline_template_matching.scale_conditioned_outer_result.v1",
        "experiment": EXPERIMENT,
        "status": "completed",
        "config_sha256": config_sha256,
        "outer_family": family,
        "git_commit": git_commit,
        "selected_candidate": candidate,
        "outer_group_metrics_file_sha256": group_metrics_sha256,
        "outer_summary": _outer_summary(
            family,
            plan.families[family],
            f1,
            candidate,
        ),
        "artifacts": {
            group_metrics_path.name: {
                "size_bytes": group_metrics_path.stat().st_size,
                "sha256": group_metrics_sha256,
            },
            payload_path.name: {
                "size_bytes": payload_path.stat().st_size,
                "sha256": sha256_file(payload_path),
            }
        },
    }
    result["manifest_content_sha256"] = canonical_json_sha256(result)
    result_path = fold / "result_manifest.json"
    _write_json(result_path, result)
    completion = {
        "schema": "pathline_template_matching.run_complete.v1",
        "experiment": EXPERIMENT,
        "config_sha256": config_sha256,
        "outer_family": family,
        "git_commit": git_commit,
        "result_manifest_file_sha256": sha256_file(result_path),
    }
    completion["completion_content_sha256"] = canonical_json_sha256(completion)
    _write_json(fold / "RUN_COMPLETE.json", completion)
    return fold


def _make_folds(
    root: Path,
    *,
    families: tuple[str, ...] | None = None,
    commits: tuple[str, ...] | None = None,
) -> tuple[Path, ...]:
    plan = load_plan(CONFIG_PATH)
    fold_families = families or plan.family_order
    fold_commits = commits or (SYNTHETIC_COMMIT,) * 5
    f1_values = (0.80, 0.75, 0.70, 0.65, 0.60)
    return tuple(
        _write_fold(
            root,
            ordinal=ordinal,
            family=family,
            config_sha256=plan.sha256,
            f1=f1,
            git_commit=commit,
        )
        for ordinal, (family, f1, commit) in enumerate(
            zip(fold_families, f1_values, fold_commits)
        )
    )


def _aggregate(
    run_directories: tuple[Path, ...],
    output_dir: Path,
    *,
    aggregator_commit: str = SYNTHETIC_COMMIT,
    aggregator_dirty: bool = False,
):
    with patch.object(
        aggregate_module,
        "_git_identity",
        return_value=(aggregator_commit, aggregator_dirty),
    ):
        return aggregate_module.aggregate(CONFIG_PATH, run_directories, output_dir)


def _rewrite_authenticated_result(fold: Path, update) -> None:
    result_path = fold / "result_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("manifest_content_sha256")
    update(result)
    result["manifest_content_sha256"] = canonical_json_sha256(result)
    _write_json(result_path, result)

    completion_path = fold / "RUN_COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion.pop("completion_content_sha256")
    completion["result_manifest_file_sha256"] = sha256_file(result_path)
    completion["completion_content_sha256"] = canonical_json_sha256(completion)
    _write_json(completion_path, completion)


def _read_group_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _rewrite_group_csv(fold: Path, rows: list[dict[str, str]]) -> None:
    path = fold / "outer_group_metrics.csv"
    fieldnames, _ = _read_group_csv(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    digest = sha256_file(path)

    def update(result):
        result["outer_group_metrics_file_sha256"] = digest
        result["artifacts"][path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }

    _rewrite_authenticated_result(fold, update)


def test_scale_conditioned_aggregate_writes_five_family_macro_and_passes_stop_rule():
    plan = load_plan(CONFIG_PATH)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        summary = _aggregate(_make_folds(root), root / "aggregate")

        assert summary["outer_families"] == list(plan.family_order)
        assert summary["outer_family_count"] == 5
        np.testing.assert_allclose(summary["family_macro"]["f1"], 0.70)
        np.testing.assert_allclose(summary["family_macro"]["accuracy"], 0.76)
        assert summary["fold_summary_source"].startswith(
            "authenticated_outer_group_metrics_csv_recomputed"
        )
        assert all(summary["success_stop_rule"].values())
        assert summary["all_success_conditions_pass"] is True
        assert (root / "aggregate" / "outer_family_summary.csv").is_file()
        assert (root / "aggregate" / "aggregate_summary.json").is_file()
        assert (root / "aggregate" / "aggregate_manifest.json").is_file()
        assert (root / "aggregate" / "AGGREGATE_COMPLETE.json").is_file()


def test_scale_conditioned_aggregate_accepts_clean_hardening_commit_after_folds():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "aggregate"
        summary = _aggregate(
            _make_folds(root),
            output,
            aggregator_commit=HARDENED_AGGREGATOR_COMMIT,
        )
        assert summary["fold_numerical_git_commit"] == SYNTHETIC_COMMIT
        assert summary["aggregator_git_commit"] == HARDENED_AGGREGATOR_COMMIT
        assert summary["aggregator_worktree_clean"] is True
        manifest = json.loads(
            (output / "aggregate_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["fold_numerical_git_commit"] == SYNTHETIC_COMMIT
        assert manifest["aggregator_git_commit"] == HARDENED_AGGREGATOR_COMMIT
        assert manifest["aggregator_worktree_clean"] is True


def test_scale_conditioned_aggregate_rejects_duplicate_outer_family():
    plan = load_plan(CONFIG_PATH)
    duplicated = (*plan.family_order[:-1], plan.family_order[0])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = _make_folds(root, families=duplicated)
        _expect_value_error(
            _aggregate,
            folds,
            root / "aggregate",
            contains="outer family set is incomplete",
        )


def test_scale_conditioned_aggregate_rejects_mixed_fold_commits():
    commits = (SYNTHETIC_COMMIT,) * 4 + ("2" * 40,)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = _make_folds(root, commits=commits)
        _expect_value_error(
            _aggregate,
            folds,
            root / "aggregate",
            contains="one numerical Git commit",
        )


def test_scale_conditioned_aggregate_rejects_corrupt_result_completion_and_artifact():
    for corruption in ("result", "completion", "artifact"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folds = _make_folds(root)
            target = folds[0]
            if corruption == "result":
                with (target / "result_manifest.json").open("ab") as stream:
                    stream.write(b"\n")
                expected = "result manifest hash mismatch"
            elif corruption == "completion":
                completion_path = target / "RUN_COMPLETE.json"
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                completion["result_manifest_file_sha256"] = "0" * 64
                _write_json(completion_path, completion)
                expected = "completion content hash mismatch"
            else:
                with (target / "outer_metrics.json").open("ab") as stream:
                    stream.write(b"damage")
                expected = "size drift outer_metrics.json"
            try:
                _expect_value_error(
                    _aggregate,
                    folds,
                    root / "aggregate",
                    contains=expected,
                )
            except AssertionError as error:
                raise AssertionError(f"corruption mode {corruption}: {error}") from error


def test_scale_conditioned_aggregate_refuses_to_overwrite_output_directory():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = _make_folds(root)
        output = root / "aggregate"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("do not overwrite", encoding="utf-8")
        _expect_value_error(
            _aggregate,
            folds,
            output,
            contains="immutable output directory exists",
        )
        assert marker.read_text(encoding="utf-8") == "do not overwrite"
        assert tuple(output.iterdir()) == (marker,)


def test_scale_conditioned_aggregate_rejects_embedded_outer_summary_mismatch():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = _make_folds(root)

        def update(result):
            result["outer_summary"]["f1"] += 0.01

        _rewrite_authenticated_result(folds[0], update)
        _expect_value_error(
            _aggregate,
            folds,
            root / "aggregate",
            contains="embedded outer summary f1 mismatch",
        )


def test_scale_conditioned_aggregate_rejects_group_csv_hash_contract_drift():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = _make_folds(root)

        def update(result):
            result["outer_group_metrics_file_sha256"] = "0" * 64

        _rewrite_authenticated_result(folds[0], update)
        _expect_value_error(
            _aggregate,
            folds,
            root / "aggregate",
            contains="outer group metric hash contract drifted",
        )


def test_scale_conditioned_aggregate_rejects_invalid_group_partition():
    for corruption, expected in (
        ("duplicate", "duplicate outer group"),
        ("incomplete", "outer group set is incomplete"),
        ("family", "physical family mismatch"),
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folds = _make_folds(root)
            path = folds[0] / "outer_group_metrics.csv"
            _, rows = _read_group_csv(path)
            if corruption == "duplicate":
                rows.append(dict(rows[0]))
            elif corruption == "incomplete":
                rows.pop()
            else:
                rows[0]["physical_family"] = "delta_wing"
            _rewrite_group_csv(folds[0], rows)
            try:
                _expect_value_error(
                    _aggregate,
                    folds,
                    root / "aggregate",
                    contains=expected,
                )
            except AssertionError as error:
                raise AssertionError(
                    f"group corruption mode {corruption}: {error}"
                ) from error

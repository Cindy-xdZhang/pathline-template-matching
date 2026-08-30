#!/usr/bin/env python3
"""Aggregate the five immutable nested outer-family results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from scripts.run_verify_scale_conditioned_retrieval_1_1 import (  # noqa: E402
    BLOCK_NAMES,
    EXPERIMENT,
    _atomic_csv,
    _atomic_json,
    _git_identity,
    _lower_hex,
    _stable_file_identity,
    _utc_now,
    load_plan,
)


OUTER_GROUP_MACRO_FIELDS = (
    "accuracy",
    "average_precision",
    "auroc",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "retrieval_support_fraction",
    "spatial_imputed_fraction",
    "spatial_unimputable_fraction",
    "supported_subset_f1",
    "imputed_subset_f1",
    "unimputable_subset_f1",
)
OUTER_COUNT_FIELDS = (
    "sample_count",
    "positive_count",
    "negative_count",
    "supported_count",
    "imputed_count",
    "unimputable_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
)
OPTIONAL_GROUP_METRICS = {
    "supported_subset_f1",
    "imputed_subset_f1",
    "unimputable_subset_f1",
}
CANDIDATE_FIELDS = (
    "candidate_id",
    "representation",
    "k",
    "sigma",
    "decision_rule",
    "decision_value",
)
SUMMARY_RELATIVE_TOLERANCE = 5.0e-11
SUMMARY_ABSOLUTE_TOLERANCE = 5.0e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_content_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    content = dict(payload)
    stored = content.pop(field, None)
    _require(
        _lower_hex(stored) and canonical_json_sha256(content) == stored,
        f"{label} content hash mismatch",
    )


def _load_fold(path: Path, config_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    completion_path = path / "RUN_COMPLETE.json"
    result_path = path / "result_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    _require(isinstance(completion, Mapping), f"{path}: completion root is invalid")
    _verify_content_hash(
        completion, "completion_content_sha256", f"{path}: completion"
    )
    _require(
        completion.get("schema") == "pathline_template_matching.run_complete.v1",
        f"{path}: completion schema mismatch",
    )
    _require(completion.get("experiment") == EXPERIMENT, f"{path}: experiment mismatch")
    _require(completion.get("config_sha256") == config_sha256, f"{path}: config mismatch")
    _require(
        sha256_file(result_path) == completion.get("result_manifest_file_sha256"),
        f"{path}: result manifest hash mismatch",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require(isinstance(result, Mapping), f"{path}: result root is invalid")
    _verify_content_hash(result, "manifest_content_sha256", f"{path}: result")
    _require(
        result.get("schema")
        == "pathline_template_matching.scale_conditioned_outer_result.v1",
        f"{path}: result schema mismatch",
    )
    _require(result.get("status") == "completed", f"{path}: result is incomplete")
    _require(result.get("config_sha256") == config_sha256, f"{path}: result config mismatch")
    _require(
        result.get("outer_family") == completion.get("outer_family"),
        f"{path}: outer family mismatch",
    )
    _require(
        result.get("git_commit") == completion.get("git_commit"),
        f"{path}: completion/result Git commit mismatch",
    )
    for name, identity in result.get("artifacts", {}).items():
        artifact = path / name
        _require(artifact.is_file(), f"{path}: missing artifact {name}")
        _require(artifact.stat().st_size == identity["size_bytes"], f"{path}: size drift {name}")
        _require(sha256_file(artifact) == identity["sha256"], f"{path}: hash drift {name}")
    return completion, result


def _csv_integer(
    row: Mapping[str, str], field: str, *, label: str, minimum: int = 0
) -> int:
    text = row.get(field)
    _require(isinstance(text, str) and text != "", f"{label}: missing integer {field}")
    try:
        value = int(text, 10)
    except ValueError as error:
        raise ValueError(f"{label}: invalid integer {field}") from error
    _require(str(value) == text and value >= minimum, f"{label}: invalid integer {field}")
    return value


def _csv_metric(
    row: Mapping[str, str],
    field: str,
    *,
    label: str,
    allow_blank: bool,
) -> float | None:
    text = row.get(field)
    if allow_blank and text == "":
        return None
    _require(isinstance(text, str) and text != "", f"{label}: missing metric {field}")
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{label}: invalid metric {field}") from error
    _require(
        np.isfinite(value) and 0.0 <= value <= 1.0,
        f"{label}: metric {field} must lie in [0, 1]",
    )
    return value


def _csv_finite_float(row: Mapping[str, str], field: str, *, label: str) -> float:
    text = row.get(field)
    _require(isinstance(text, str) and text != "", f"{label}: missing number {field}")
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{label}: invalid number {field}") from error
    _require(np.isfinite(value), f"{label}: non-finite number {field}")
    return value


def _require_close(actual: float, expected: float, message: str) -> None:
    _require(
        bool(
            np.isclose(
                actual,
                expected,
                rtol=SUMMARY_RELATIVE_TOLERANCE,
                atol=SUMMARY_ABSOLUTE_TOLERANCE,
            )
        ),
        message,
    )


def _recompute_outer_summary(
    plan: Any,
    fold_path: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild one fold summary only from its authenticated group CSV."""

    family = str(result.get("outer_family"))
    _require(family in plan.families, f"{fold_path}: unknown outer family")
    selected = result.get("selected_candidate")
    _require(isinstance(selected, Mapping), f"{fold_path}: selected candidate is invalid")
    _require(
        set(selected) == set(CANDIDATE_FIELDS),
        f"{fold_path}: selected candidate contract drifted",
    )

    artifacts = result.get("artifacts")
    _require(isinstance(artifacts, Mapping), f"{fold_path}: artifacts mapping is invalid")
    artifact = artifacts.get("outer_group_metrics.csv")
    _require(
        isinstance(artifact, Mapping),
        f"{fold_path}: outer_group_metrics.csv is not authenticated",
    )
    artifact_size = artifact.get("size_bytes")
    artifact_sha = artifact.get("sha256")
    direct_sha = result.get("outer_group_metrics_file_sha256")
    _require(
        isinstance(artifact_size, int)
        and not isinstance(artifact_size, bool)
        and artifact_size > 0
        and _lower_hex(artifact_sha)
        and direct_sha == artifact_sha,
        f"{fold_path}: outer group metric hash contract drifted",
    )
    metrics_path = fold_path / "outer_group_metrics.csv"
    _stable_file_identity(metrics_path, artifact_size, str(artifact_sha))

    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        _require(fieldnames is not None, f"{fold_path}: outer group CSV has no header")
        _require(
            len(fieldnames) == len(set(fieldnames)),
            f"{fold_path}: outer group CSV has duplicate columns",
        )
        required_fields = {
            "outer_family",
            "physical_family",
            "dataset",
            "source_ordinal",
            "block",
            *CANDIDATE_FIELDS,
            *OUTER_GROUP_MACRO_FIELDS,
            *OUTER_COUNT_FIELDS,
        }
        _require(
            required_fields.issubset(fieldnames),
            f"{fold_path}: outer group CSV field set is incomplete",
        )
        raw_rows = list(reader)
    _stable_file_identity(metrics_path, artifact_size, str(artifact_sha))

    expected_groups = {
        (dataset, source_ordinal, block)
        for dataset in plan.families[family]
        for source_ordinal in range(4)
        for block in BLOCK_NAMES
    }
    parsed_rows: list[dict[str, Any]] = []
    observed_groups: set[tuple[str, int, str]] = set()
    for row_index, row in enumerate(raw_rows):
        label = f"{fold_path}: outer group row {row_index}"
        _require(row.get("outer_family") == family, f"{label}: outer family mismatch")
        _require(row.get("physical_family") == family, f"{label}: physical family mismatch")
        dataset = str(row.get("dataset"))
        source_ordinal = _csv_integer(
            row, "source_ordinal", label=label, minimum=0
        )
        block = str(row.get("block"))
        group = (dataset, source_ordinal, block)
        _require(group not in observed_groups, f"{label}: duplicate outer group")
        observed_groups.add(group)

        candidate = {
            "candidate_id": row.get("candidate_id"),
            "representation": row.get("representation"),
            "k": _csv_integer(row, "k", label=label, minimum=1),
            "sigma": _csv_finite_float(row, "sigma", label=label),
            "decision_rule": row.get("decision_rule"),
            "decision_value": _csv_finite_float(
                row, "decision_value", label=label
            ),
        }
        _require(candidate == dict(selected), f"{label}: selected candidate mismatch")

        counts = {
            field: _csv_integer(row, field, label=label, minimum=0)
            for field in OUTER_COUNT_FIELDS
        }
        _require(counts["sample_count"] > 0, f"{label}: empty group")
        _require(
            counts["positive_count"] > 0 and counts["negative_count"] > 0,
            f"{label}: group must contain both classes",
        )
        _require(
            counts["positive_count"] + counts["negative_count"]
            == counts["sample_count"],
            f"{label}: class counts do not cover the group",
        )
        _require(
            counts["true_positive"]
            + counts["false_positive"]
            + counts["true_negative"]
            + counts["false_negative"]
            == counts["sample_count"],
            f"{label}: confusion counts do not cover the group",
        )
        _require(
            counts["true_positive"] + counts["false_negative"]
            == counts["positive_count"]
            and counts["true_negative"] + counts["false_positive"]
            == counts["negative_count"],
            f"{label}: confusion/class counts disagree",
        )
        _require(
            counts["supported_count"]
            + counts["imputed_count"]
            + counts["unimputable_count"]
            == counts["sample_count"],
            f"{label}: support states do not cover the group",
        )
        metrics = {
            field: _csv_metric(
                row,
                field,
                label=label,
                allow_blank=field in OPTIONAL_GROUP_METRICS,
            )
            for field in OUTER_GROUP_MACRO_FIELDS
        }
        sample_count = counts["sample_count"]
        for field, count_field in (
            ("retrieval_support_fraction", "supported_count"),
            ("spatial_imputed_fraction", "imputed_count"),
            ("spatial_unimputable_fraction", "unimputable_count"),
        ):
            value = metrics[field]
            _require(value is not None, f"{label}: missing support fraction")
            _require_close(
                value,
                counts[count_field] / sample_count,
                f"{label}: {field} disagrees with counts",
            )

        tp = counts["true_positive"]
        fp = counts["false_positive"]
        tn = counts["true_negative"]
        fn = counts["false_negative"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        derived_metrics = {
            "accuracy": (tp + tn) / sample_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": 0.5 * (recall + specificity),
        }
        for field, expected in derived_metrics.items():
            value = metrics[field]
            _require(value is not None, f"{label}: missing classification metric {field}")
            _require_close(value, expected, f"{label}: {field} disagrees with confusion counts")
        parsed_rows.append(
            {
                "dataset": dataset,
                "source_ordinal": source_ordinal,
                "block": block,
                **metrics,
                **counts,
            }
        )

    _require(
        observed_groups == expected_groups,
        f"{fold_path}: outer group set is incomplete",
    )

    def macro(field: str) -> float | None:
        # Every authenticated dataset has exactly four sources x two blocks.
        # Therefore equal source/block groups within each dataset followed by
        # equal datasets is the frozen hierarchical family macro.
        dataset_means: list[float] = []
        for dataset in plan.families[family]:
            values = [
                float(row[field])
                for row in parsed_rows
                if row["dataset"] == dataset and row[field] is not None
            ]
            if values:
                dataset_means.append(float(np.mean(values, dtype=np.float64)))
        return (
            None
            if not dataset_means
            else float(np.mean(dataset_means, dtype=np.float64))
        )

    recomputed: dict[str, Any] = {
        "outer_family": family,
        "candidate": dict(selected),
        "group_count": len(parsed_rows),
        "dataset_count": len(plan.families[family]),
        **{field: macro(field) for field in OUTER_GROUP_MACRO_FIELDS},
        **{
            field: sum(int(row[field]) for row in parsed_rows)
            for field in OUTER_COUNT_FIELDS
        },
        "aggregation": "equal_dataset_source_block_groups_within_outer_family",
    }
    embedded = result.get("outer_summary")
    _require(isinstance(embedded, Mapping), f"{fold_path}: embedded outer summary is invalid")
    _require(
        set(embedded) == set(recomputed),
        f"{fold_path}: embedded outer summary field set drifted",
    )
    for field in (
        "outer_family",
        "candidate",
        "group_count",
        "dataset_count",
        *OUTER_COUNT_FIELDS,
        "aggregation",
    ):
        _require(
            embedded.get(field) == recomputed[field],
            f"{fold_path}: embedded outer summary {field} mismatch",
        )
    for field in OUTER_GROUP_MACRO_FIELDS:
        expected = recomputed[field]
        actual = embedded.get(field)
        if expected is None:
            _require(actual is None, f"{fold_path}: embedded outer summary {field} mismatch")
        else:
            _require(
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and np.isfinite(float(actual)),
                f"{fold_path}: embedded outer summary {field} is invalid",
            )
            _require_close(
                float(actual),
                expected,
                f"{fold_path}: embedded outer summary {field} mismatch",
            )
    return recomputed


def aggregate(
    config_path: str | Path,
    run_directories: Sequence[str | Path],
    output_dir: str | Path,
    *,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    plan = load_plan(config_path)
    if expected_config_sha256 is not None:
        _require(plan.sha256 == expected_config_sha256, "frozen config SHA-256 mismatch")
    git_commit, dirty = _git_identity()
    _require(not dirty, "aggregate requires a clean committed worktree")
    paths = [Path(value).resolve() for value in run_directories]
    _require(len(paths) == 5 and len(set(paths)) == 5, "exactly five unique fold directories are required")
    loaded_folds = [_load_fold(path, plan.sha256) for path in paths]
    folds = [
        (
            completion,
            result,
            _recompute_outer_summary(plan, path, result),
        )
        for path, (completion, result) in zip(paths, loaded_folds)
    ]
    fold_commits = {
        str(result["git_commit"]) for _, result, _ in folds
    }
    _require(len(fold_commits) == 1, "all five folds must use one numerical Git commit")
    fold_numerical_git_commit = next(iter(fold_commits))
    by_family = {
        result["outer_family"]: (path, completion, result, recomputed_summary)
        for path, (completion, result, recomputed_summary) in zip(paths, folds)
    }
    _require(tuple(sorted(by_family)) == tuple(sorted(plan.family_order)), "outer family set is incomplete")
    destination = Path(output_dir).resolve()
    _require(not destination.exists(), f"immutable output directory exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)

    metric_fields = (
        "accuracy",
        "average_precision",
        "auroc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "retrieval_support_fraction",
        "spatial_imputed_fraction",
        "spatial_unimputable_fraction",
    )
    rows: list[dict[str, Any]] = []
    for family in plan.family_order:
        path, completion, result, summary = by_family[family]
        rows.append(
            {
                "outer_family": family,
                "run_directory": str(path),
                "git_commit": result["git_commit"],
                "selected_candidate_id": result["selected_candidate"]["candidate_id"],
                **{field: float(summary[field]) for field in metric_fields},
                **{
                    field: int(summary[field])
                    for field in (
                        "sample_count",
                        "positive_count",
                        "negative_count",
                        "supported_count",
                        "imputed_count",
                        "unimputable_count",
                        "true_positive",
                        "false_positive",
                        "true_negative",
                        "false_negative",
                    )
                },
                "result_manifest_file_sha256": completion[
                    "result_manifest_file_sha256"
                ],
                "outer_group_metrics_file_sha256": result[
                    "outer_group_metrics_file_sha256"
                ],
            }
        )
    table_sha = _atomic_csv(destination / "outer_family_summary.csv", list(rows[0]), rows)
    macro = {field: float(np.mean([row[field] for row in rows])) for field in metric_fields}
    success = {
        "family_macro_f1_at_least_0_70": macro["f1"] >= 0.70,
        "at_least_four_families_f1_at_least_0_65": sum(
            row["f1"] >= 0.65 for row in rows
        )
        >= 4,
        "no_family_f1_below_0_50": min(row["f1"] for row in rows) >= 0.50,
        "family_macro_average_precision_at_least_0_60": macro[
            "average_precision"
        ]
        >= 0.60,
        "family_macro_balanced_accuracy_at_least_0_70": macro[
            "balanced_accuracy"
        ]
        >= 0.70,
        "family_macro_precision_at_least_0_60": macro["precision"] >= 0.60,
        "family_macro_recall_at_least_0_60": macro["recall"] >= 0.60,
    }
    aggregate_summary: dict[str, Any] = {
        "schema": "pathline_template_matching.scale_conditioned_nested_summary.v1",
        "experiment": EXPERIMENT,
        "status": "completed",
        "created_utc": _utc_now(),
        "aggregation_git_commit": git_commit,
        "aggregator_git_commit": git_commit,
        "aggregator_worktree_clean": True,
        "fold_numerical_git_commit": fold_numerical_git_commit,
        "fold_git_commits": sorted({row["git_commit"] for row in rows}),
        "config_path": str(plan.path),
        "config_sha256": plan.sha256,
        "outer_family_count": 5,
        "outer_families": list(plan.family_order),
        "aggregation": "equal_outer_physical_family_macro",
        "fold_summary_source": (
            "authenticated_outer_group_metrics_csv_recomputed_and_verified_"
            "against_result_manifest"
        ),
        "family_macro": macro,
        "success_stop_rule": success,
        "all_success_conditions_pass": all(success.values()),
        "outer_family_summary_file_sha256": table_sha,
        "folds": rows,
        "formal_confirmation": False,
        "evidence_scope": "exposed_train_only_nested_family_validation",
    }
    aggregate_summary["summary_content_sha256"] = canonical_json_sha256(
        aggregate_summary
    )
    summary_sha = _atomic_json(destination / "aggregate_summary.json", aggregate_summary)
    manifest = {
        "schema": "pathline_template_matching.scale_conditioned_nested_aggregate_manifest.v1",
        "experiment": EXPERIMENT,
        "status": "completed",
        "config_sha256": plan.sha256,
        "aggregation_git_commit": git_commit,
        "aggregator_git_commit": git_commit,
        "aggregator_worktree_clean": True,
        "fold_numerical_git_commit": fold_numerical_git_commit,
        "outer_family_summary_file_sha256": table_sha,
        "aggregate_summary_file_sha256": summary_sha,
        "source_result_manifests": [
            {
                "outer_family": row["outer_family"],
                "run_directory": row["run_directory"],
                "result_manifest_file_sha256": row[
                    "result_manifest_file_sha256"
                ],
                "outer_group_metrics_file_sha256": row[
                    "outer_group_metrics_file_sha256"
                ],
            }
            for row in rows
        ],
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    manifest_sha = _atomic_json(destination / "aggregate_manifest.json", manifest)
    completion = {
        "schema": "pathline_template_matching.aggregate_complete.v1",
        "experiment": EXPERIMENT,
        "config_sha256": plan.sha256,
        "aggregator_git_commit": git_commit,
        "aggregator_worktree_clean": True,
        "fold_numerical_git_commit": fold_numerical_git_commit,
        "aggregate_manifest_file": "aggregate_manifest.json",
        "aggregate_manifest_file_sha256": manifest_sha,
        "completed_utc": _utc_now(),
    }
    completion["completion_content_sha256"] = canonical_json_sha256(completion)
    _atomic_json(destination / "AGGREGATE_COMPLETE.json", completion)
    return aggregate_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-config-sha256")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = aggregate(
        arguments.config,
        arguments.run_dir,
        arguments.output_dir,
        expected_config_sha256=arguments.expected_config_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

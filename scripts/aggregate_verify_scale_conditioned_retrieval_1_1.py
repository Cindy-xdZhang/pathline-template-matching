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
    EXPERIMENT,
    _atomic_csv,
    _atomic_json,
    _git_identity,
    _lower_hex,
    _utc_now,
    load_plan,
)


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
    folds = [_load_fold(path, plan.sha256) for path in paths]
    fold_commits = {
        str(result["git_commit"]) for _, result in folds
    }
    _require(len(fold_commits) == 1, "all five folds must use one numerical Git commit")
    _require(
        next(iter(fold_commits)) == git_commit,
        "aggregate checkout must equal the five-fold numerical Git commit",
    )
    by_family = {result["outer_family"]: (path, completion, result) for path, (completion, result) in zip(paths, folds)}
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
        path, completion, result = by_family[family]
        summary = result["outer_summary"]
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
        "fold_git_commits": sorted({row["git_commit"] for row in rows}),
        "config_path": str(plan.path),
        "config_sha256": plan.sha256,
        "outer_family_count": 5,
        "outer_families": list(plan.family_order),
        "aggregation": "equal_outer_physical_family_macro",
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
        "outer_family_summary_file_sha256": table_sha,
        "aggregate_summary_file_sha256": summary_sha,
        "source_result_manifests": [
            {
                "outer_family": row["outer_family"],
                "run_directory": row["run_directory"],
                "result_manifest_file_sha256": row[
                    "result_manifest_file_sha256"
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

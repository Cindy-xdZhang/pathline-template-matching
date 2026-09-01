#!/usr/bin/env python3
"""Freshly replay and aggregate RankLikelihoodTemplate 1.1 folds."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.portable_flow import sha256_file  # noqa: E402
from scripts import (  # noqa: E402
    run_verify_source_centered_rank_likelihood_template_1_1 as runner,
)


AGGREGATE_SUMMARY_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_aggregate_summary.v1"
)
SINGLE_FOLD_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_single_fold.v1"
)
PARTIAL_FOLD_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_partial_fold.v1"
)
AGGREGATE_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_aggregate_manifest.v1"
)
AGGREGATE_COMPLETE_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_aggregate_complete.v1"
)


@dataclass(frozen=True)
class AuthenticatedFold:
    path: Path
    outer_family: str
    git_commit: str
    primary: runner.CandidateSpec
    control: runner.ControlSpec
    fresh_rows: tuple[Mapping[str, Any], ...]
    fresh_summary: Mapping[str, Any]
    completion_file_sha256: str
    result_manifest_file_sha256: str


def _candidate_from_payload(plan: runner.Plan, payload: object) -> runner.CandidateSpec:
    runner._require(isinstance(payload, Mapping), "primary candidate payload is invalid")
    candidate = runner.CandidateSpec(
        float(payload["weight"]),
        int(payload["bin_count"]),
        float(payload["beta"]),
        float(payload["sigma"]),
        float(payload["decision_value"]),
    )
    runner._require(
        dict(payload) == runner._json_safe(runner._candidate_payload(candidate)),
        "primary candidate payload is not canonical",
    )
    frozen = {item.candidate_id: item for item in runner.candidate_specs(plan)}
    runner._require(
        candidate.candidate_id in frozen and frozen[candidate.candidate_id] == candidate,
        "primary candidate is outside the frozen 540-member grid",
    )
    return candidate


def _control_from_payload(plan: runner.Plan, payload: object) -> runner.ControlSpec:
    runner._require(isinstance(payload, Mapping), "control candidate payload is invalid")
    candidate = runner.ControlSpec(
        float(payload["weight"]),
        float(payload["sigma"]),
        float(payload["decision_value"]),
    )
    runner._require(
        dict(payload) == runner._json_safe(runner._control_payload(candidate)),
        "control candidate payload is not canonical",
    )
    frozen = {item.candidate_id: item for item in runner.control_specs(plan)}
    runner._require(
        candidate.candidate_id in frozen and frozen[candidate.candidate_id] == candidate,
        "control candidate is outside the frozen 90-member grid",
    )
    return candidate


def _parse_outer_csv(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: Sequence[Mapping[str, Any]],
) -> None:
    payload = runner.source_runner.early._read_authenticated_bytes(
        path, expected_sha256=expected_sha256
    )
    with io.StringIO(payload.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        runner._require(tuple(reader.fieldnames or ()) == runner.OUTER_METRIC_FIELDS, "outer CSV fields drifted")
        rows = list(reader)
    runner._require(len(rows) == len(expected_rows), "outer CSV row count drifted")
    for index, (raw, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        for field in runner.OUTER_METRIC_FIELDS:
            text = str(runner.source_runner.early._csv_value(expected[field]))
            runner._require(raw[field] == text, f"outer CSV value drifted: {index}/{field}")


def _inner_paths(
    path: Path,
    selected: Mapping[str, Any],
) -> Mapping[str, tuple[Path, str]]:
    evidence = selected.get("inner_evidence")
    runner._require(isinstance(evidence, Mapping), "selected inner evidence is missing")
    result: dict[str, tuple[Path, str]] = {}
    for key, filename in (
        ("inner_group_metrics", "inner_group_metrics.csv"),
        ("inner_candidate_summary", "inner_candidate_summary.csv"),
        ("inner_fit_audits", "inner_fit_audits.json"),
    ):
        record = evidence.get(key)
        runner._require(
            isinstance(record, Mapping) and record.get("path") == filename,
            f"selected inner evidence path drifted: {key}",
        )
        child = path / filename
        runner._stable_file_identity(child, int(record["size_bytes"]), str(record["sha256"]))
        result[key] = (child, str(record["sha256"]))
    return result


def _authenticate_fold(
    plan: runner.Plan,
    fold_path: Path,
    *,
    expected_fold_commit: str,
) -> AuthenticatedFold:
    path = fold_path.resolve()
    runner._require(path.is_dir(), f"fold directory does not exist: {path}")
    runner._require(
        {child.name for child in path.iterdir()} == set(runner.REQUIRED_FOLD_FILES),
        f"{path}: fold file set drifted",
    )
    completion_path = path / "RUN_COMPLETE.json"
    completion_sha = sha256_file(completion_path)
    completion = runner.source_runner._read_self_hashed_json(
        completion_path, expected_file_sha256=completion_sha
    )
    outer_family = str(completion.get("outer_family"))
    runner._require(
        completion.get("schema") == runner.COMPLETE_SCHEMA
        and completion.get("experiment") == runner.EXPERIMENT
        and completion.get("git_commit") == expected_fold_commit
        and completion.get("config_sha256") == plan.sha256
        and outer_family in plan.family_order,
        "fold completion provenance drifted",
    )
    selected_path = path / "selected_candidate.json"
    selected_sha = sha256_file(selected_path)
    selected = runner.source_runner._read_self_hashed_json(
        selected_path, expected_file_sha256=selected_sha
    )
    runner._require(
        selected.get("schema") == runner.SELECTED_SCHEMA
        and selected.get("experiment") == runner.EXPERIMENT
        and selected.get("git_commit") == expected_fold_commit
        and selected.get("config_sha256") == plan.sha256
        and selected.get("outer_family") == outer_family,
        "selected-candidate provenance drifted",
    )
    primary = _candidate_from_payload(plan, selected.get("primary_candidate"))
    control = _control_from_payload(plan, selected.get("selected_control"))
    primary_summary = selected.get("primary_inner_selection_summary")
    control_summary = selected.get("control_inner_selection_summary")
    runner._require(
        isinstance(primary_summary, Mapping) and isinstance(control_summary, Mapping),
        "selected inner summaries are missing",
    )
    inner_paths = _inner_paths(path, selected)
    primary, primary_summary, control, control_summary = runner._fresh_inner_selection(
        plan,
        outer_family,
        inner_paths,
        claimed_primary=primary,
        claimed_primary_summary=primary_summary,
        claimed_control=control,
        claimed_control_summary=control_summary,
    )
    final_records = selected.get("final_artifacts")
    runner._require(isinstance(final_records, Mapping), "final artifact records are missing")
    final_hashes = {
        key: str(final_records[key]["manifest_file_sha256"])
        for key in ("model", "calibration", "control")
    }
    verified = runner._fresh_replay_before_reference(
        plan,
        path,
        outer_family=outer_family,
        primary=primary,
        primary_summary=primary_summary,
        control=control,
        control_summary=control_summary,
        inner_paths=inner_paths,
        git_commit=expected_fold_commit,
        final_manifest_sha256=final_hashes,
        selection_file_sha256=selected_sha,
        rank_binding_file_sha256=sha256_file(path / "outer_rank_binding.json"),
        prediction_manifest_file_sha256=sha256_file(
            path / "outer_prediction_manifest.json"
        ),
    )
    parent_control = runner.authenticate_parent_control(plan, outer_family)
    fresh_rows, fresh_access = runner.evaluate_outer_prediction(
        plan,
        primary,
        control,
        verified,
        parent_control,
        outer_family=outer_family,
    )
    fresh_summary = runner.outer_summary(fresh_rows, outer_family)

    result = runner.source_runner._read_self_hashed_json(
        path / "result_manifest.json",
        expected_file_sha256=str(completion["result_manifest_file_sha256"]),
    )
    runner._require(
        result.get("schema") == runner.RESULT_SCHEMA
        and result.get("experiment") == runner.EXPERIMENT
        and result.get("status") == "completed"
        and result.get("git_commit") == expected_fold_commit
        and result.get("config_sha256") == plan.sha256
        and result.get("outer_family") == outer_family
        and result.get("primary_candidate")
        == runner._json_safe(runner._candidate_payload(primary))
        and result.get("selected_control")
        == runner._json_safe(runner._control_payload(control))
        and result.get("content_sha256")
        == completion.get("result_manifest_content_sha256"),
        "fold result provenance drifted",
    )
    artifacts = result.get("artifacts")
    runner._require(isinstance(artifacts, Mapping), "fold artifact map is missing")
    for name in runner.REQUIRED_FOLD_FILES:
        if name in {"result_manifest.json", "RUN_COMPLETE.json"}:
            continue
        record = artifacts.get(name)
        runner._require(isinstance(record, Mapping), f"fold artifact record missing: {name}")
        runner._stable_file_identity(path / name, int(record["size_bytes"]), str(record["sha256"]))
    _parse_outer_csv(
        path / "outer_group_metrics.csv",
        expected_sha256=str(result["outer_group_metrics_file_sha256"]),
        expected_rows=fresh_rows,
    )
    persisted_summary = runner.source_runner._read_self_hashed_json(
        path / "outer_summary.json",
        expected_file_sha256=str(result["outer_summary_file_sha256"]),
    )
    runner._require(
        persisted_summary == runner._manifest(fresh_summary),
        "outer summary differs from fresh replay",
    )
    persisted_access = runner.source_runner._read_self_hashed_json(
        path / "outer_reference_access_audit.json",
        expected_file_sha256=str(result["outer_reference_access_audit_file_sha256"]),
    )
    expected_access = runner._manifest(
        {
            "schema": runner.REFERENCE_AUDIT_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "outer_family": outer_family,
            "first_open_phase": "after_new_prediction_file_manifest_and_complete_fresh_replay",
            "prediction_manifest_file_sha256": verified.manifest_file_sha256,
            "prediction_file_sha256": verified.artifact_file_sha256,
            "parent_control": dict(parent_control.evidence),
            "row_count": len(fresh_access),
            "rows": fresh_access,
            "reference_labels_all_opened": False,
            "fmt_features_opened": False,
            "raw_features_opened": False,
        }
    )
    runner._require(persisted_access == expected_access, "reference-access audit drifted")
    return AuthenticatedFold(
        path,
        outer_family,
        expected_fold_commit,
        primary,
        control,
        tuple(runner.source_runner.early._deep_freeze(row) for row in fresh_rows),
        runner.source_runner.early._deep_freeze(fresh_summary),
        completion_sha,
        str(completion["result_manifest_file_sha256"]),
    )


def _paired_bootstrap(
    plan: runner.Plan, folds: Sequence[AuthenticatedFold]
) -> dict[str, Any]:
    pairs: dict[str, list[tuple[float, float]]] = {
        family: [] for family in plan.family_order
    }
    for fold in folds:
        primary = {
            (str(row["dataset"]), int(row["source_ordinal"])): float(row["f1"])
            for row in fold.fresh_rows
            if row["arm"] == "dual_histogram_llr"
            and row["population"] == "all_parent_valid_rows"
        }
        parent = {
            (str(row["dataset"]), int(row["source_ordinal"])): float(row["f1"])
            for row in fold.fresh_rows
            if row["arm"] == "parent_source_centered_paired_scale"
            and row["population"] == "all_parent_valid_rows"
        }
        runner._require(primary.keys() == parent.keys(), "bootstrap pair identities drifted")
        pairs[fold.outer_family] = [
            (primary[key], parent[key]) for key in sorted(primary)
        ]
    runner._require(all(pairs[family] for family in plan.family_order), "bootstrap family missing")
    point = float(
        np.mean(
            [
                np.mean([primary - parent for primary, parent in pairs[family]])
                for family in plan.family_order
            ]
        )
    )
    rng = np.random.default_rng(17068)
    replicates = np.empty(5000, dtype=np.float64)
    for index in range(len(replicates)):
        family_values = []
        for family in plan.family_order:
            values = np.asarray(
                [primary - parent for primary, parent in pairs[family]],
                dtype=np.float64,
            )
            selected = rng.integers(0, len(values), size=len(values))
            family_values.append(float(np.mean(values[selected])))
        replicates[index] = float(np.mean(family_values))
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return {
        "method": "paired_dataset_source_bootstrap_equal_family_macro",
        "replicates": 5000,
        "seed": 17068,
        "target": "primary_dual_histogram_llr_f1_minus_authenticated_parent_on_exact_valid_rows",
        "point_estimate": point,
        "lower_95_percent": float(lower),
        "upper_95_percent": float(upper),
    }


def _success_rule(
    plan: runner.Plan,
    folds: Sequence[AuthenticatedFold],
    bootstrap: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    summaries = [fold.fresh_summary["primary"] for fold in folds]
    macro = {
        field: float(np.mean([float(summary[field]) for summary in summaries]))
        for field in runner.CLASSIFICATION_FIELDS
    }
    coverage = float(
        np.mean(
            [float(summary["unique_center_combined_coverage"]) for summary in summaries]
        )
    )
    f1_values = [float(summary["f1"]) for summary in summaries]
    outcomes = {
        "five_family_macro_f1_minimum_pass": macro["f1"] >= 0.70,
        "families_at_or_above_f1_0_65_minimum_pass": sum(value >= 0.65 for value in f1_values) >= 4,
        "minimum_single_family_f1_pass": min(f1_values) >= 0.50,
        "five_family_macro_average_precision_minimum_pass": macro["average_precision"] >= 0.60,
        "five_family_macro_balanced_accuracy_minimum_pass": macro["balanced_accuracy"] >= 0.70,
        "five_family_macro_precision_minimum_pass": macro["precision"] >= 0.60,
        "five_family_macro_recall_minimum_pass": macro["recall"] >= 0.60,
        "unique_center_combined_coverage_minimum_pass": coverage >= 0.90,
        "paired_bootstrap_lower_bound_strictly_above_zero_pass": float(bootstrap["lower_95_percent"]) > 0.0,
    }
    return {
        "classification_metric_population": "all_parent_valid_rows_after_exact_center_projection",
        "family_macro": macro,
        "family_f1": dict(zip(plan.family_order, f1_values, strict=True)),
        "unique_center_combined_coverage": coverage,
        "paired_bootstrap": dict(bootstrap),
        "outcomes": outcomes,
    }, all(outcomes.values())


def _partial_stop_certificate(
    folds: Sequence[AuthenticatedFold],
) -> dict[str, Any]:
    runner._require(1 <= len(folds) < 5, "partial certificate requires one to four folds")
    primaries = [fold.fresh_summary["primary"] for fold in folds]
    completed_count = len(folds)
    remaining_count = 5 - completed_count
    thresholds = {
        "f1": 0.70,
        "average_precision": 0.60,
        "balanced_accuracy": 0.70,
        "precision": 0.60,
        "recall": 0.60,
        "unique_center_combined_coverage": 0.90,
    }
    bounds = {}
    possible = {}
    for field, threshold in thresholds.items():
        values = [float(primary[field]) for primary in primaries]
        upper = (sum(values) + float(remaining_count)) / 5.0
        bounds[field] = {
            "completed_family_values": values,
            "remaining_family_count": remaining_count,
            "complete_five_macro_upper_bound": upper,
            "required_minimum": threshold,
            "possible": upper >= threshold,
        }
        possible[f"five_family_macro_{field}"] = upper >= threshold
    f1_values = [float(primary["f1"]) for primary in primaries]
    possible["families_at_or_above_f1_0_65"] = (
        sum(value >= 0.65 for value in f1_values) + remaining_count >= 4
    )
    possible["minimum_single_family_f1"] = min(f1_values) >= 0.50
    possible["paired_bootstrap_lower_bound"] = True
    impossible = sorted(name for name, value in possible.items() if not value)
    return {
        "mode": "partial_fold_optimistic_complete_five_certificate",
        "outer_families": [fold.outer_family for fold in folds],
        "completed_family_count": completed_count,
        "remaining_family_count": remaining_count,
        "optimistic_macro_bounds": bounds,
        "threshold_possible": possible,
        "impossible_complete_five_thresholds": impossible,
        "stop_version": bool(impossible),
    }


def _single_fold_stop_certificate(fold: AuthenticatedFold) -> dict[str, Any]:
    """Compatibility wrapper used by the first-fold release tests."""

    return _partial_stop_certificate((fold,))


def aggregate(
    config_path: str | Path,
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    expected_fold_commit: str,
    parent_binding_path: str | Path,
    parent_binding_file_sha256: str,
    binding_completion_path: str | Path,
    binding_completion_file_sha256: str,
    expected_config_sha256: str | None = runner.EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    plan = runner.load_plan(config_path)
    if expected_config_sha256 is not None:
        runner._require(plan.sha256 == expected_config_sha256, "expected config SHA drifted")
    commit, dirty = runner._git_identity()
    runner._require(not dirty and commit == expected_fold_commit, "aggregate checkout identity drifted")
    plan = runner.bind_parent_sidecar_release(
        plan,
        parent_binding_path=parent_binding_path,
        parent_binding_file_sha256=parent_binding_file_sha256,
        binding_completion_path=binding_completion_path,
        binding_completion_file_sha256=binding_completion_file_sha256,
    )
    folds = tuple(
        _authenticate_fold(
            plan, Path(path), expected_fold_commit=expected_fold_commit
        )
        for path in run_dirs
    )
    runner._require(
        1 <= len(folds) <= 5
        and len({fold.outer_family for fold in folds}) == len(folds),
        "aggregate requires one to five unique families",
    )
    if len(folds) < 5:
        mode = (
            "single_fold_authentication"
            if len(folds) == 1
            else f"partial_{len(folds)}_fold_authentication"
        )
        success_rule = None
        success = None
        bootstrap = None
        stop_certificate = _partial_stop_certificate(folds)
        stop_version = bool(stop_certificate["stop_version"])
    else:
        by_family = {fold.outer_family: fold for fold in folds}
        runner._require(set(by_family) == set(plan.family_order), "five-fold family set drifted")
        folds = tuple(by_family[family] for family in plan.family_order)
        mode = "complete_five_fold_aggregate"
        bootstrap = _paired_bootstrap(plan, folds)
        success_rule, success = _success_rule(plan, folds, bootstrap)
        f1_values = [float(fold.fresh_summary["primary"]["f1"]) for fold in folds]
        stop_version = min(f1_values) < 0.50 or sum(value < 0.65 for value in f1_values) >= 2 or not bool(success)
        stop_certificate = {
            "mode": "complete_five_fold_observed_certificate",
            "stop_version": stop_version,
            "all_primary_success_conditions_pass": bool(success),
        }

    destination = Path(output_dir).resolve()
    runner._require(not destination.exists(), f"immutable aggregate directory exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    family_rows = []
    for fold in folds:
        primary = fold.fresh_summary["primary"]
        row = {
            "outer_family": fold.outer_family,
            "run_directory": str(fold.path),
            "primary_candidate_id": fold.primary.candidate_id,
            "control_candidate_id": fold.control.candidate_id,
            **primary,
            "completion_file_sha256": fold.completion_file_sha256,
            "result_manifest_file_sha256": fold.result_manifest_file_sha256,
        }
        for source_key, prefix in (
            ("parent_control", "parent"),
            ("negative_ecdf_control_not_template_success", "negative_ecdf"),
            ("direct_rank_mean_top5_not_template_success", "direct"),
        ):
            for field in (*runner.CLASSIFICATION_FIELDS, *runner.CLASSIFICATION_COUNT_FIELDS):
                row[f"{prefix}_{field}"] = fold.fresh_summary[source_key][field]
        family_rows.append(row)
    table_path = destination / "outer_family_summary.csv"
    table_sha = runner.source_runner.early._atomic_csv(
        table_path, tuple(family_rows[0]), family_rows
    )
    evidence = {
        "config_sha256": plan.sha256,
        "parent_binding_file_sha256": plan.parent_binding_file_sha256,
        "binding_completion_file_sha256": plan.binding_completion_file_sha256,
        "historical_source_centered_evidence": runner._json_safe(plan.source_evidence),
    }
    report = runner._manifest(
        {
            "schema": (
                AGGREGATE_SUMMARY_SCHEMA
                if len(folds) == 5
                else SINGLE_FOLD_SCHEMA
                if len(folds) == 1
                else PARTIAL_FOLD_SCHEMA
            ),
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": mode,
            "config_sha256": plan.sha256,
            "aggregator_git_commit": commit,
            "fold_git_commit": expected_fold_commit,
            "source_centered_evidence": evidence,
            "outer_families": [fold.outer_family for fold in folds],
            "folds": family_rows,
            "success_rule": success_rule,
            "all_primary_success_conditions_pass": success,
            "stop_version": stop_version,
            "stop_certificate": stop_certificate,
            "paired_bootstrap": bootstrap,
            "controls_can_satisfy_primary_success": False,
            "outer_family_summary_file_sha256": table_sha,
            "formal_confirmation": False,
        }
    )
    report_path = destination / (
        "aggregate_summary.json"
        if len(folds) == 5
        else "single_fold_authentication_report.json"
        if len(folds) == 1
        else "partial_fold_authentication_report.json"
    )
    report_sha = runner.source_runner.early._atomic_json(report_path, report)
    manifest = runner._manifest(
        {
            "schema": AGGREGATE_MANIFEST_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": mode,
            "config_sha256": plan.sha256,
            "aggregator_git_commit": commit,
            "fold_git_commit": expected_fold_commit,
            "source_centered_evidence": evidence,
            "outer_family_summary_file": table_path.name,
            "outer_family_summary_file_sha256": table_sha,
            "report_file": report_path.name,
            "report_file_sha256": report_sha,
            "source_folds": [
                {
                    "outer_family": fold.outer_family,
                    "run_directory": str(fold.path),
                    "completion_file_sha256": fold.completion_file_sha256,
                    "result_manifest_file_sha256": fold.result_manifest_file_sha256,
                }
                for fold in folds
            ],
        }
    )
    manifest_path = destination / "aggregate_manifest.json"
    manifest_sha = runner.source_runner.early._atomic_json(manifest_path, manifest)
    completion = runner._manifest(
        {
            "schema": AGGREGATE_COMPLETE_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": mode,
            "config_sha256": plan.sha256,
            "aggregator_git_commit": commit,
            "fold_git_commit": expected_fold_commit,
            "source_centered_evidence": evidence,
            "aggregate_manifest_file": manifest_path.name,
            "aggregate_manifest_file_sha256": manifest_sha,
            "report_file": report_path.name,
            "report_file_sha256": report_sha,
            "completed_utc": runner._utc_now(),
        }
    )
    runner.source_runner.early._atomic_json(
        destination / "AGGREGATE_COMPLETE.json", completion
    )
    runner._require(
        {path.name for path in destination.iterdir()}
        == {
            "outer_family_summary.csv",
            report_path.name,
            "aggregate_manifest.json",
            "AGGREGATE_COMPLETE.json",
        },
        "aggregate output file set drifted",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "config"
            / "Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml"
        ),
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-fold-commit", required=True)
    parser.add_argument("--parent-binding", required=True)
    parser.add_argument("--parent-binding-sha256", required=True)
    parser.add_argument("--binding-completion", required=True)
    parser.add_argument("--binding-completion-sha256", required=True)
    parser.add_argument("--expected-config-sha256", default=runner.EXPECTED_CONFIG_SHA256)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = aggregate(
        arguments.config,
        arguments.run_dir,
        arguments.output_dir,
        expected_fold_commit=arguments.expected_fold_commit,
        parent_binding_path=arguments.parent_binding,
        parent_binding_file_sha256=arguments.parent_binding_sha256,
        binding_completion_path=arguments.binding_completion,
        binding_completion_file_sha256=arguments.binding_completion_sha256,
        expected_config_sha256=arguments.expected_config_sha256,
    )
    print(json.dumps(runner._json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

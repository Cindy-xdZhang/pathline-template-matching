#!/usr/bin/env python3
"""Fresh-replay and aggregate SourceCenteredPairedScaleTemplate 1.1 folds."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.portable_flow import sha256_file  # noqa: E402
from scripts import run_verify_source_centered_paired_scale_template_1_1 as runner  # noqa: E402


AGGREGATE_SUMMARY_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_aggregate_summary.v1"
)
SINGLE_FOLD_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_single_fold_report.v1"
)
AGGREGATE_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_aggregate_manifest.v1"
)
AGGREGATE_COMPLETE_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_aggregate_complete.v1"
)
AGGREGATE_DIAGNOSTIC_ARMS = {
    "source_centered_separate_legacy_block|all_unique_centers": "legacy_block",
    "source_centered_separate_expanded_block|all_unique_centers": "expanded_block",
    "direct_source_centered_min_dx_top5|all_unique_centers": "direct_min_dx_top5",
    "direct_source_centered_dx_rank_mean_top5|all_unique_centers": (
        "direct_dx_rank_mean_top5"
    ),
}


@dataclass(frozen=True)
class AuthenticatedFold:
    path: Path
    outer_family: str
    git_commit: str
    selected_candidate: Mapping[str, Any]
    fresh_rows: tuple[Mapping[str, Any], ...]
    fresh_summary: Mapping[str, Any]
    completion_file_sha256: str
    result_manifest_file_sha256: str


_AGGREGATE_ROW_EVIDENCE_FIELDS = (
    "dataset",
    "dataset_index",
    "physical_family",
    "source_ordinal",
    "source_index",
    "completion_file_sha256",
    "sidecar_file_sha256",
    "sidecar_combined_array_sha256",
    "valid_projection_sha256",
    "assigned_row_count",
    "valid_projection_row_count",
)


def _aggregate_evidence_binding(
    plan: runner.Plan,
    *,
    git_commit: str,
) -> dict[str, Any]:
    runner._require_evidence_bound(plan)
    runner._require(
        plan.sidecar_input_manifest_path is not None
        and plan.sidecar_population_path is not None
        and plan.sidecar_root is not None
        and plan.sidecar_population is not None,
        "aggregate source-centered evidence is unbound",
    )
    population = plan.sidecar_population
    rows = population.get("rows")
    runner._require(
        isinstance(rows, Sequence) and len(rows) == 32,
        "aggregate sidecar row evidence is incomplete",
    )
    row_identities = []
    for row in rows:
        runner._require(
            isinstance(row, Mapping)
            and all(name in row for name in _AGGREGATE_ROW_EVIDENCE_FIELDS),
            "aggregate sidecar row identity is incomplete",
        )
        row_identities.append(
            {name: row[name] for name in _AGGREGATE_ROW_EVIDENCE_FIELDS}
        )
    return {
        "experiment": runner.EXPERIMENT,
        "config_sha256": plan.sha256,
        "git_commit": git_commit,
        "source_centered_input_manifest": {
            "path": str(plan.sidecar_input_manifest_path),
            "size_bytes": plan.sidecar_input_manifest_path.stat().st_size,
            "file_sha256": plan.sidecar_input_manifest_file_sha256,
            "content_sha256": plan.sidecar_input_manifest_content_sha256,
        },
        "source_centered_sidecars": {
            "root": str(plan.sidecar_root),
            "population_manifest_path": str(plan.sidecar_population_path),
            "population_manifest_size_bytes": (
                plan.sidecar_population_path.stat().st_size
            ),
            "population_manifest_file_sha256": (
                plan.sidecar_population_file_sha256
            ),
            "population_manifest_content_sha256": (
                plan.sidecar_population_content_sha256
            ),
            "sidecar_count": int(population["sidecar_count"]),
            "rows_content_sha256": population["rows_content_sha256"],
            "assigned_row_count_total": int(
                population["assigned_row_count_total"]
            ),
            "valid_projection_row_count_total": int(
                population["valid_projection_row_count_total"]
            ),
            "row_identities": row_identities,
        },
    }


def _require_fold_source_evidence(
    plan: runner.Plan,
    result: Mapping[str, Any],
    candidate: runner.CandidateSpec,
    outer_family: str,
) -> None:
    expected = runner._evidence_binding(
        plan,
        representation=candidate.representation,
        fit_families=[
            family for family in plan.family_order if family != outer_family
        ],
    )
    runner._require(
        result.get("source_centered_evidence") == runner._json_safe(expected),
        "fold result source-centered evidence differs from aggregate inputs",
    )


def _candidate_from_payload(plan: runner.Plan, payload: object) -> runner.CandidateSpec:
    runner._require(isinstance(payload, Mapping), "selected candidate payload is invalid")
    candidate = runner.CandidateSpec(
        str(payload["representation"]),
        int(payload["k"]),
        float(payload["sigma"]),
        float(payload["weight"]),
        float(payload["decision_value"]),
    )
    runner._require(
        dict(payload) == runner._json_safe(runner._candidate_payload(candidate)),
        "selected candidate payload is not canonical",
    )
    frozen = {value.candidate_id: value for value in runner.candidate_specs(plan)}
    runner._require(
        candidate.candidate_id in frozen and frozen[candidate.candidate_id] == candidate,
        "selected candidate is outside the frozen 1,800-member set",
    )
    return candidate


def _parse_outer_csv(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: Sequence[Mapping[str, Any]],
) -> None:
    content = runner.early._read_authenticated_bytes(
        path, expected_sha256=expected_sha256
    )
    with io.StringIO(content.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        runner._require(
            tuple(reader.fieldnames or ()) == runner.OUTER_METRIC_FIELDS,
            "outer metric CSV fields drifted",
        )
        rows = list(reader)
    runner._require(len(rows) == len(expected_rows), "outer metric row count drifted")
    for row_index, (raw, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        for field in runner.OUTER_METRIC_FIELDS:
            expected_text = str(runner.early._csv_value(expected[field]))
            runner._require(
                raw[field] == expected_text,
                f"outer metric differs from fresh reference: row={row_index}/{field}",
            )


def _fresh_inner_selection(
    plan: runner.Plan,
    outer_family: str,
    inner_paths: Mapping[str, tuple[Path, str]],
    *,
    claimed_candidate: runner.CandidateSpec,
    claimed_summary: Mapping[str, Any],
) -> tuple[runner.CandidateSpec, dict[str, Any]]:
    replayed_inner_rows = runner._parse_inner_metric_csv(
        inner_paths["inner_group_metrics"][0],
        expected_sha256=inner_paths["inner_group_metrics"][1],
        plan=plan,
        outer_family=outer_family,
    )
    replayed_summaries, replayed_candidate, replayed_selected_summary = (
        runner.aggregate_and_select_inner(plan, replayed_inner_rows)
    )
    runner._authenticate_summary_csv(
        inner_paths["inner_candidate_summary"][0],
        expected_sha256=inner_paths["inner_candidate_summary"][1],
        expected=replayed_summaries,
    )
    runner._authenticate_inner_fit_audits(
        inner_paths["inner_fit_audits"][0],
        expected_sha256=inner_paths["inner_fit_audits"][1],
        plan=plan,
        outer_family=outer_family,
    )
    runner._require(
        claimed_candidate == replayed_candidate
        and claimed_summary == runner._json_safe(replayed_selected_summary),
        "selected candidate differs from fresh inner selection",
    )
    return replayed_candidate, replayed_selected_summary


def _authenticate_fold(
    plan: runner.Plan,
    fold_path: Path,
    *,
    expected_fold_commit: str,
    device: str,
) -> AuthenticatedFold:
    path = fold_path.resolve()
    runner._require(path.is_dir(), f"fold directory does not exist: {path}")
    runner._require(
        {child.name for child in path.iterdir()} == set(runner.REQUIRED_FOLD_FILES),
        f"{path}: fold file set drifted",
    )
    completion_path = path / "RUN_COMPLETE.json"
    completion_sha = sha256_file(completion_path)
    completion = runner._read_self_hashed_json(
        completion_path, expected_file_sha256=completion_sha
    )
    family = str(completion.get("outer_family"))
    runner._require(
        completion.get("schema") == runner.COMPLETE_SCHEMA
        and completion.get("experiment") == runner.EXPERIMENT
        and completion.get("git_commit") == expected_fold_commit
        and completion.get("config_sha256") == plan.sha256
        and family in plan.family_order,
        f"{path}: completion provenance drifted",
    )

    # Read only label-free artifacts before fresh replay.  selected_candidate
    # binds all inner evidence and both fitted model artifacts.
    selected_path = path / "selected_candidate.json"
    selected_sha = sha256_file(selected_path)
    selected_value, _ = runner._read_manifest(selected_path, selected_sha)
    runner._require(
        selected_value.get("schema") == runner.SELECTED_SCHEMA
        and selected_value.get("experiment") == runner.EXPERIMENT
        and selected_value.get("config_sha256") == plan.sha256
        and selected_value.get("git_commit") == expected_fold_commit
        and selected_value.get("outer_family") == family,
        f"{path}: selected-candidate provenance drifted",
    )
    candidate = _candidate_from_payload(plan, selected_value.get("candidate"))
    selected_summary = selected_value.get("inner_selection_summary")
    runner._require(isinstance(selected_summary, Mapping), "selected inner summary is missing")
    inner_evidence = selected_value.get("inner_evidence")
    runner._require(isinstance(inner_evidence, Mapping), "selected inner evidence is missing")
    inner_paths: dict[str, tuple[Path, str]] = {}
    for name, filename in (
        ("inner_group_metrics", "inner_group_metrics.csv"),
        ("inner_candidate_summary", "inner_candidate_summary.csv"),
        ("inner_fit_audits", "inner_fit_audits.json"),
    ):
        record = inner_evidence.get(name)
        runner._require(
            isinstance(record, Mapping) and record.get("path") == filename,
            f"selected inner evidence path drifted: {name}",
        )
        evidence_path = path / filename
        runner._stable_file_identity(
            evidence_path, int(record["size_bytes"]), str(record["sha256"])
        )
        inner_paths[name] = (evidence_path, str(record["sha256"]))

    # The selected artifact is only a claim until the complete authenticated
    # 1,800-candidate inner table has been parsed, aggregated, and selected
    # again.  This replay must precede every outer reference read.
    candidate, selected_summary = _fresh_inner_selection(
        plan,
        family,
        inner_paths,
        claimed_candidate=candidate,
        claimed_summary=selected_summary,
    )

    prediction_manifest_sha = sha256_file(path / "outer_prediction_manifest.json")
    source_binding_sha = sha256_file(path / "outer_source_centered_binding.json")
    scaler_manifest_sha = str(selected_value["scaler_manifest"]["file_sha256"])
    calibration_manifest_sha = str(
        selected_value["calibration_manifest"]["file_sha256"]
    )
    verified = runner._fresh_replay_before_reference(
        plan,
        path,
        selected=candidate,
        selected_summary=selected_summary,
        inner_paths=inner_paths,
        outer_family=family,
        git_commit=expected_fold_commit,
        device=device,
        scaler_manifest_sha256=scaler_manifest_sha,
        calibration_manifest_sha256=calibration_manifest_sha,
        selection_file_sha256=selected_sha,
        source_binding_file_sha256=source_binding_sha,
        prediction_manifest_sha256=prediction_manifest_sha,
    )
    fresh_rows, fresh_access = runner.evaluate_outer_prediction(
        plan, candidate, verified, outer_family=family
    )
    fresh_summary = runner.outer_summary(fresh_rows, family)

    # Performance-bearing result artifacts are opened only after the complete
    # label-free prediction replay above has authenticated.
    result_path = path / "result_manifest.json"
    result = runner._read_self_hashed_json(
        result_path,
        expected_file_sha256=str(completion["result_manifest_file_sha256"]),
    )
    runner._require(
        result.get("schema") == runner.RESULT_SCHEMA
        and result.get("experiment") == runner.EXPERIMENT
        and result.get("status") == "completed"
        and result.get("git_commit") == expected_fold_commit
        and result.get("config_sha256") == plan.sha256
        and result.get("outer_family") == family
        and result.get("selected_candidate")
        == runner._json_safe(runner._candidate_payload(candidate))
        and result.get("outer_source_centered_binding_file_sha256")
        == source_binding_sha
        and result.get("content_sha256")
        == completion.get("result_manifest_content_sha256"),
        f"{path}: result provenance drifted",
    )
    _require_fold_source_evidence(plan, result, candidate, family)
    artifacts = result.get("artifacts")
    runner._require(isinstance(artifacts, Mapping), "result artifact map is missing")
    for name in runner.REQUIRED_FOLD_FILES:
        if name in {"result_manifest.json", "RUN_COMPLETE.json"}:
            continue
        record = artifacts.get(name)
        runner._require(isinstance(record, Mapping), f"artifact record missing: {name}")
        runner._stable_file_identity(
            path / name, int(record["size_bytes"]), str(record["sha256"])
        )
    _parse_outer_csv(
        path / "outer_group_metrics.csv",
        expected_sha256=str(result["outer_group_metrics_file_sha256"]),
        expected_rows=fresh_rows,
    )
    persisted_summary = runner._read_self_hashed_json(
        path / "outer_summary.json",
        expected_file_sha256=str(result["outer_summary_file_sha256"]),
    )
    runner._require(
        persisted_summary == runner._manifest(fresh_summary),
        "outer summary differs from fresh evaluation",
    )
    persisted_access = runner._read_self_hashed_json(
        path / "outer_reference_access_audit.json",
        expected_file_sha256=str(result["outer_reference_access_audit_file_sha256"]),
    )
    expected_access = runner._manifest(
        {
            "schema": runner.REFERENCE_AUDIT_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "outer_family": family,
            "first_open_phase": "after_outer_prediction_file_manifest_and_fresh_replay_authentication",
            "prediction_manifest_file_sha256": verified.manifest_file_sha256,
            "prediction_file_sha256": verified.artifact_file_sha256,
            "row_count": len(fresh_access),
            "rows": fresh_access,
        }
    )
    runner._require(persisted_access == expected_access, "reference-access audit drifted")
    return AuthenticatedFold(
        path,
        family,
        expected_fold_commit,
        runner.early._deep_freeze(runner._candidate_payload(candidate)),
        tuple(runner.early._deep_freeze(row) for row in fresh_rows),
        runner.early._deep_freeze(fresh_summary),
        completion_sha,
        str(completion["result_manifest_file_sha256"]),
    )


def _paired_bootstrap(
    plan: runner.Plan,
    folds: Sequence[AuthenticatedFold],
) -> dict[str, Any]:
    pairs: dict[str, list[tuple[float, float]]] = {
        family: [] for family in plan.family_order
    }
    for fold in folds:
        primary = {
            (str(row["dataset"]), int(row["source_ordinal"])): float(row["f1"])
            for row in fold.fresh_rows
            if row["arm"] == "source_centered_paired_centers"
            and row["population"] == "all_parent_valid_rows"
        }
        parent = {
            (str(row["dataset"]), int(row["source_ordinal"])): float(row["f1"])
            for row in fold.fresh_rows
            if row["arm"] == "parent_current_replay"
            and row["population"] == "all_parent_valid_rows"
        }
        runner._require(primary.keys() == parent.keys(), "bootstrap paired source identities drifted")
        pairs[fold.outer_family] = [(primary[key], parent[key]) for key in sorted(primary)]
    runner._require(all(pairs[family] for family in plan.family_order), "bootstrap family population is incomplete")
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
        family_differences = []
        for family in plan.family_order:
            values = np.asarray(
                [primary - parent for primary, parent in pairs[family]],
                dtype=np.float64,
            )
            selected = rng.integers(0, len(values), size=len(values))
            family_differences.append(float(np.mean(values[selected])))
        replicates[index] = float(np.mean(family_differences))
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return {
        "method": "paired_dataset_source_bootstrap_equal_family_macro",
        "replicates": 5000,
        "seed": 17068,
        "target": "primary_valid_row_projection_f1_minus_parent_current_f1_on_exact_same_valid_rows",
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
        "paired_bootstrap": bootstrap,
        "outcomes": outcomes,
    }, all(outcomes.values())


def _single_fold_stop_certificate(fold: AuthenticatedFold) -> dict[str, Any]:
    """Prove whether any frozen complete-five threshold is already impossible."""

    primary = fold.fresh_summary["primary"]
    remaining_family_count = 4
    complete_family_count = 5
    macro_thresholds = {
        "f1": 0.70,
        "average_precision": 0.60,
        "balanced_accuracy": 0.70,
        "precision": 0.60,
        "recall": 0.60,
        "unique_center_combined_coverage": 0.90,
    }
    optimistic_bounds: dict[str, Any] = {}
    threshold_possible: dict[str, bool] = {}
    for field, threshold in macro_thresholds.items():
        observed = float(primary[field])
        upper = (
            (observed + remaining_family_count) / complete_family_count
            if np.isfinite(observed)
            else float("nan")
        )
        possible = bool(np.isfinite(upper) and upper >= threshold)
        optimistic_bounds[field] = {
            "completed_family_value": observed,
            "remaining_family_assumption": 1.0,
            "complete_five_macro_upper_bound": upper,
            "required_minimum": threshold,
            "possible": possible,
        }
        threshold_possible[f"five_family_macro_{field}"] = possible

    observed_f1 = float(primary["f1"])
    maximum_families_at_or_above_0_65 = remaining_family_count + int(
        np.isfinite(observed_f1) and observed_f1 >= 0.65
    )
    threshold_possible["families_at_or_above_f1_0_65"] = (
        maximum_families_at_or_above_0_65 >= 4
    )
    threshold_possible["minimum_single_family_f1"] = bool(
        np.isfinite(observed_f1) and observed_f1 >= 0.50
    )
    # With four unobserved family differences each bounded above by +1, one
    # completed family can never by itself prove the paired-bootstrap target
    # impossible.  The final lower confidence bound is still recomputed from
    # all five authenticated families.
    threshold_possible["paired_bootstrap_lower_bound"] = True
    impossible = sorted(
        name for name, possible in threshold_possible.items() if not possible
    )
    return {
        "mode": "single_fold_optimistic_complete_five_certificate",
        "outer_family": fold.outer_family,
        "completed_family_count": 1,
        "remaining_family_count": remaining_family_count,
        "optimistic_macro_bounds": optimistic_bounds,
        "maximum_families_at_or_above_f1_0_65": (
            maximum_families_at_or_above_0_65
        ),
        "threshold_possible": threshold_possible,
        "impossible_complete_five_thresholds": impossible,
        "stop_version": bool(impossible),
    }


def aggregate(
    config_path: str | Path,
    run_directories: Sequence[str | Path],
    output_dir: str | Path,
    *,
    expected_fold_commit: str,
    sidecar_input_manifest_path: str | Path,
    sidecar_input_manifest_file_sha256: str,
    sidecar_root: str | Path,
    sidecar_population_manifest_path: str | Path,
    sidecar_population_manifest_file_sha256: str,
    device: str = "cpu",
    expected_config_sha256: str | None = runner.EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    plan = runner.load_plan(config_path)
    if expected_config_sha256 is not None:
        runner._require(plan.sha256 == expected_config_sha256, "expected config SHA-256 drifted")
    commit, dirty = runner._git_identity()
    runner._require(not dirty and commit == expected_fold_commit, "aggregate/fold Git identity drifted")
    plan = runner.bind_source_centered_evidence(
        plan,
        input_manifest_path=sidecar_input_manifest_path,
        input_manifest_file_sha256=sidecar_input_manifest_file_sha256,
        sidecar_root=sidecar_root,
        population_manifest_path=sidecar_population_manifest_path,
        population_manifest_file_sha256=sidecar_population_manifest_file_sha256,
    )
    aggregate_evidence = _aggregate_evidence_binding(
        plan, git_commit=expected_fold_commit
    )
    runner._configure_execution(device)
    paths = tuple(Path(value).resolve() for value in run_directories)
    runner._require(len(paths) in {1, 5} and len(paths) == len(set(paths)), "aggregation requires one or five unique folds")
    folds = tuple(
        _authenticate_fold(
            plan,
            path,
            expected_fold_commit=expected_fold_commit,
            device=device,
        )
        for path in paths
    )
    if len(folds) == 1:
        runner._require(folds[0].outer_family == plan.family_order[0], "single-fold authentication is restricted to half_cylinder")
        mode = "single_fold_authentication"
        bootstrap = None
        success = None
        success_rule = None
        stop_certificate = _single_fold_stop_certificate(folds[0])
        stop_version = bool(stop_certificate["stop_version"])
    else:
        runner._require(
            {fold.outer_family for fold in folds} == set(plan.family_order),
            "complete aggregation requires every outer family exactly once",
        )
        by_family = {fold.outer_family: fold for fold in folds}
        folds = tuple(by_family[family] for family in plan.family_order)
        mode = "complete_five_fold_aggregate"
        bootstrap = _paired_bootstrap(plan, folds)
        success_rule, success = _success_rule(plan, folds, bootstrap)
        f1_values = [float(fold.fresh_summary["primary"]["f1"]) for fold in folds]
        stop_version = (
            min(f1_values) < 0.50
            or sum(value < 0.65 for value in f1_values) >= 2
            or not bool(success)
        )
        stop_certificate = {
            "mode": "complete_five_fold_observed_certificate",
            "stop_version": stop_version,
            "all_template_success_conditions_pass": bool(success),
        }

    destination = Path(output_dir).resolve()
    runner._require(not destination.exists(), f"immutable aggregate directory exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    family_rows = []
    for fold in folds:
        primary_summary = fold.fresh_summary["primary"]
        parent_summary = fold.fresh_summary["parent_control"]
        diagnostic_summary = fold.fresh_summary[
            "diagnostics_not_template_success"
        ]
        row = {
            "outer_family": fold.outer_family,
            "run_directory": str(fold.path),
            "selected_candidate_id": fold.selected_candidate["candidate_id"],
            **primary_summary,
            "completion_file_sha256": fold.completion_file_sha256,
            "result_manifest_file_sha256": fold.result_manifest_file_sha256,
        }
        for field in (*runner.CLASSIFICATION_FIELDS, *runner.CLASSIFICATION_COUNT_FIELDS):
            row[f"parent_{field}"] = parent_summary[field]
        for key, prefix in AGGREGATE_DIAGNOSTIC_ARMS.items():
            runner._require(key in diagnostic_summary, f"missing diagnostic summary: {key}")
            for field in (
                *runner.CLASSIFICATION_FIELDS,
                *runner.CLASSIFICATION_COUNT_FIELDS,
            ):
                row[f"{prefix}_{field}"] = diagnostic_summary[key][field]
        family_rows.append(row)
    diagnostic_macro_label = (
        "five_family_macro" if len(folds) == 5 else "single_authenticated_family"
    )
    diagnostic_aggregate = {
        key: {
            "template_success_eligible": False,
            diagnostic_macro_label: {
                field: float(
                    np.mean(
                        [
                            float(
                                fold.fresh_summary[
                                    "diagnostics_not_template_success"
                                ][key][field]
                            )
                            for fold in folds
                        ]
                    )
                )
                for field in runner.CLASSIFICATION_FIELDS
            },
        }
        for key in AGGREGATE_DIAGNOSTIC_ARMS
    }
    family_fields = tuple(family_rows[0])
    table_path = destination / "outer_family_summary.csv"
    table_sha = runner.early._atomic_csv(table_path, family_fields, family_rows)
    report_payload = {
        "schema": AGGREGATE_SUMMARY_SCHEMA if len(folds) == 5 else SINGLE_FOLD_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": "completed",
        "mode": mode,
        "config_sha256": plan.sha256,
        "aggregator_git_commit": commit,
        "fold_git_commit": expected_fold_commit,
        "source_centered_evidence": aggregate_evidence,
        "outer_families": [fold.outer_family for fold in folds],
        "folds": family_rows,
        "success_rule": success_rule,
        "all_template_success_conditions_pass": success,
        "stop_version": stop_version,
        "stop_certificate": stop_certificate,
        "paired_bootstrap": bootstrap,
        "diagnostics_not_template_success": diagnostic_aggregate,
        "direct_diagnostics_can_satisfy_template_success": False,
        "outer_family_summary_file_sha256": table_sha,
        "formal_confirmation": False,
        "evidence_scope": "exposed_train_only_nested_family_validation",
    }
    report = runner._manifest(report_payload)
    report_path = destination / (
        "aggregate_summary.json" if len(folds) == 5 else "single_fold_authentication_report.json"
    )
    report_sha = runner.early._atomic_json(report_path, report)
    manifest = runner._manifest(
        {
            "schema": AGGREGATE_MANIFEST_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": mode,
            "config_sha256": plan.sha256,
            "aggregator_git_commit": commit,
            "fold_git_commit": expected_fold_commit,
            "source_centered_evidence": aggregate_evidence,
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
    manifest_sha = runner.early._atomic_json(manifest_path, manifest)
    completion = runner._manifest(
        {
            "schema": AGGREGATE_COMPLETE_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": mode,
            "config_sha256": plan.sha256,
            "aggregator_git_commit": commit,
            "fold_git_commit": expected_fold_commit,
            "source_centered_evidence": aggregate_evidence,
            "aggregate_manifest_file": manifest_path.name,
            "aggregate_manifest_file_sha256": manifest_sha,
            "report_file": report_path.name,
            "report_file_sha256": report_sha,
            "completed_utc": runner._utc_now(),
        }
    )
    runner.early._atomic_json(destination / "AGGREGATE_COMPLETE.json", completion)
    runner._require(len(list(destination.iterdir())) == 4, "aggregate output file set drifted")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "Verify_SourceCenteredPairedScaleTemplate_1.1.yaml"),
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-fold-commit", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sidecar-input-manifest", required=True)
    parser.add_argument("--sidecar-input-manifest-sha256", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--sidecar-population-manifest", required=True)
    parser.add_argument("--sidecar-population-manifest-sha256", required=True)
    parser.add_argument("--expected-config-sha256", default=runner.EXPECTED_CONFIG_SHA256)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = aggregate(
        arguments.config,
        arguments.run_dir,
        arguments.output_dir,
        expected_fold_commit=arguments.expected_fold_commit,
        sidecar_input_manifest_path=arguments.sidecar_input_manifest,
        sidecar_input_manifest_file_sha256=arguments.sidecar_input_manifest_sha256,
        sidecar_root=arguments.sidecar_root,
        sidecar_population_manifest_path=arguments.sidecar_population_manifest,
        sidecar_population_manifest_file_sha256=arguments.sidecar_population_manifest_sha256,
        device=arguments.device,
        expected_config_sha256=arguments.expected_config_sha256,
    )
    print(json.dumps(runner._json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

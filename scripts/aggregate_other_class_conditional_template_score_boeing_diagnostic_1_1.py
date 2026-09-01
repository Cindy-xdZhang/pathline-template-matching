#!/usr/bin/env python3
"""Authenticate and publish the Boeing-only post-stop diagnostic release.

The stopped Verify release is not resumed here.  This module authenticates one
fresh Boeing outer fold with the unchanged Verify fold authenticator, then
publishes a four-file diagnostic transaction that deliberately has no success,
stop, or multi-family macro semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import aggregate_verify_class_conditional_template_score_1_1 as base  # noqa: E402
from scripts import run_other_class_conditional_template_score_boeing_diagnostic_1_1 as runner  # noqa: E402


REPORT_SCHEMA = (
    "pathline_template_matching."
    "other_class_conditional_template_score_boeing_diagnostic_report.v1"
)
MANIFEST_SCHEMA = (
    "pathline_template_matching."
    "other_class_conditional_template_score_boeing_diagnostic_manifest.v1"
)
COMPLETE_SCHEMA = (
    "pathline_template_matching."
    "other_class_conditional_template_score_boeing_diagnostic_complete.v1"
)
STATUS = "completed"
EVIDENCE_SCOPE = runner.EVIDENCE_SCOPE
FOLD_SUMMARY_SOURCE = base.FOLD_SUMMARY_SOURCE
SUMMARY_FILE = "boeing_outer_summary.csv"
REPORT_FILE = "boeing_diagnostic_report.json"
MANIFEST_FILE = "diagnostic_manifest.json"
COMPLETE_FILE = "DIAGNOSTIC_COMPLETE.json"
RELEASE_FILES = (SUMMARY_FILE, REPORT_FILE, MANIFEST_FILE, COMPLETE_FILE)
SOURCE_FOLD_FIELDS = base.SOURCE_FOLD_FIELDS
EXPECTED_RESULT_ARTIFACTS = base.EXPECTED_RESULT_ARTIFACTS
FAMILY_SUMMARY_FIELDS = base.FAMILY_SUMMARY_FIELDS
FAMILY_METRIC_FIELDS = base.FAMILY_METRIC_FIELDS
FAMILY_COUNT_FIELDS = base.FAMILY_COUNT_FIELDS
METHOD_IDENTITY_FIELDS = base.METHOD_IDENTITY_FIELDS

REPORT_FIELDS = {
    "schema",
    "experiment",
    "status",
    "config_sha256",
    *METHOD_IDENTITY_FIELDS,
    runner.METHOD_BINDING_KEY,
    "aggregator_git_commit",
    "aggregator_worktree_clean",
    "fold_numerical_git_commit",
    "input_manifest_sha256",
    "input_manifest_rows_sha256",
    "outer_family",
    "class_conditional_support",
    "fold_summary_source",
    "fold",
    "boeing_outer_summary_file",
    "boeing_outer_summary_file_sha256",
    "formal_confirmation",
    "evidence_scope",
    "allowed_use",
    "content_sha256",
}
MANIFEST_FIELDS = {
    "schema",
    "experiment",
    "status",
    "config_sha256",
    *METHOD_IDENTITY_FIELDS,
    runner.METHOD_BINDING_KEY,
    "aggregator_git_commit",
    "aggregator_worktree_clean",
    "fold_numerical_git_commit",
    "boeing_outer_summary_file",
    "boeing_outer_summary_file_sha256",
    "report_file",
    "report_file_sha256",
    "source_folds",
    "evidence_scope",
    "content_sha256",
}
COMPLETE_FIELDS = {
    "schema",
    "experiment",
    "status",
    "config_sha256",
    *METHOD_IDENTITY_FIELDS,
    runner.METHOD_BINDING_KEY,
    "aggregator_git_commit",
    "aggregator_worktree_clean",
    "fold_numerical_git_commit",
    "diagnostic_manifest_file",
    "diagnostic_manifest_file_sha256",
    "report_file",
    "report_file_sha256",
    "boeing_outer_summary_file",
    "boeing_outer_summary_file_sha256",
    "evidence_scope",
    "completed_utc",
    "content_sha256",
}
FORBIDDEN_RELEASE_FIELDS = {
    "success_rule",
    "success_stop_rule",
    "stop_rule",
    "stop_version",
    "five_fold_success",
    "five_fold_success_evaluated",
    "family_macro",
    "five_family_macro",
    "all_success_conditions_pass",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_lower_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _strict_json_equal(actual: object, expected: object) -> bool:
    return base._strict_json_equal(actual, expected)


def _self_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    return base._self_hashed(payload)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return base._canonical_json_bytes(payload)


def _method_identity_fields() -> dict[str, str]:
    return base._method_identity_fields()


def _method_binding(plan: runner.Plan, git_commit: str) -> dict[str, Any]:
    return runner.inherited._json_safe(runner._method_binding(plan, git_commit))


def _validate_plan(plan: runner.Plan) -> None:
    _require(plan.sha256 == runner.EXPECTED_CONFIG_SHA256, "diagnostic config drifted")
    _require(
        plan.required_fold_files == runner.REQUIRED_FOLD_FILES
        and len(plan.required_fold_files) == 15
        and len(EXPECTED_RESULT_ARTIFACTS) == 13,
        "diagnostic fold transaction drifted",
    )
    candidates = runner.candidate_specs(plan)
    _require(
        len(candidates) == runner.FROZEN_CANDIDATE_COUNT == 3060,
        "diagnostic candidate count drifted",
    )
    with runner.diagnostic_verify_runtime(plan):
        base._validate_plan_output_contract(plan)


def _configured_evidence_arguments(plan: runner.Plan) -> dict[str, Any]:
    identity = plan.raw.get("input_identity")
    _require(isinstance(identity, Mapping), "configured input identity is invalid")
    kinematic = identity.get("kinematic_input_manifest")
    synthetic = identity.get("parent_synthetic_pass")
    population = identity.get("sidecar_population")
    sidecar_root = identity.get("sidecar_root")
    _require(
        isinstance(kinematic, Mapping)
        and isinstance(synthetic, Mapping)
        and isinstance(population, Mapping)
        and isinstance(sidecar_root, str)
        and isinstance(kinematic.get("path"), str)
        and _is_lower_hex(kinematic.get("sha256"), 64)
        and isinstance(synthetic.get("path"), str)
        and _is_lower_hex(synthetic.get("sha256"), 64)
        and isinstance(population.get("path"), str)
        and _is_lower_hex(population.get("sha256"), 64),
        "configured Early evidence identity is invalid",
    )
    return {
        "kinematic_input_manifest_path": kinematic["path"],
        "kinematic_input_manifest_file_sha256": kinematic["sha256"],
        "synthetic_pass_path": synthetic["path"],
        "synthetic_pass_file_sha256": synthetic["sha256"],
        "sidecar_root": sidecar_root,
        "sidecar_population_manifest_path": population["path"],
        "sidecar_population_manifest_file_sha256": population["sha256"],
    }


def _bind_configured_evidence(plan: runner.Plan) -> runner.Plan:
    return runner.bind_early_evidence(plan, **_configured_evidence_arguments(plan))


def _preflight_boeing_fold_envelope(
    plan: runner.Plan,
    fold_path: Path,
    *,
    expected_fold_commit: str,
) -> dict[str, Any]:
    """Authenticate the sealed Boeing metadata before any Verify replay."""

    fold_path = fold_path.resolve()
    _require(fold_path.is_dir(), f"fold directory does not exist: {fold_path}")
    _require(
        {path.name for path in fold_path.iterdir()} == set(runner.REQUIRED_FOLD_FILES)
        and all((fold_path / name).is_file() for name in runner.REQUIRED_FOLD_FILES),
        f"{fold_path}: completed fold must contain exactly the frozen 15 files",
    )
    expected_method = _method_binding(plan, expected_fold_commit)

    completion_path = fold_path / "RUN_COMPLETE.json"
    completion_snapshot = base._read_file_snapshot(completion_path)
    completion = base._json_from_snapshot(
        completion_snapshot,
        path=completion_path,
        self_hashed=True,
    )
    _require(
        set(completion) == base.COMPLETION_FIELDS,
        f"{fold_path}: completion fields drifted",
    )
    _require(
        completion.get("schema") == runner.COMPLETE_SCHEMA
        and completion.get("experiment") == runner.EXPERIMENT
        and isinstance(completion.get("completed_utc"), str)
        and completion.get("completed_utc") != ""
        and _is_lower_hex(completion.get("content_sha256"), 64),
        f"{fold_path}: completion schema or experiment drifted",
    )
    _require(
        completion.get("git_commit") == expected_fold_commit
        and completion.get("config_sha256") == plan.sha256,
        f"{fold_path}: completion commit/config drifted",
    )
    _require(
        _strict_json_equal(
            completion.get(runner.METHOD_BINDING_KEY), expected_method
        ),
        f"{fold_path}: completion method binding drifted",
    )
    _require(
        completion.get("result_manifest_file") == "result_manifest.json",
        f"{fold_path}: result path drifted",
    )
    _require(
        completion.get("outer_family") == runner.ONLY_OUTER_FAMILY,
        "diagnostic release accepts only the Boeing outer fold",
    )

    result_path = fold_path / "result_manifest.json"
    result_snapshot = base._read_file_snapshot(result_path)
    _require(
        result_snapshot.sha256 == completion.get("result_manifest_file_sha256"),
        f"{fold_path}: result manifest file SHA-256 mismatch",
    )
    result = base._json_from_snapshot(
        result_snapshot,
        path=result_path,
        self_hashed=True,
    )
    _require(
        set(result) == base.RESULT_FIELDS,
        f"{fold_path}: result fields drifted",
    )
    _require(
        result.get("schema") == runner.RESULT_SCHEMA
        and result.get("experiment") == runner.EXPERIMENT
        and result.get("status") == STATUS
        and isinstance(result.get("completed_utc"), str)
        and result.get("completed_utc") != ""
        and _is_lower_hex(result.get("content_sha256"), 64),
        f"{fold_path}: result schema/status drifted",
    )
    _require(
        result.get("git_commit") == expected_fold_commit
        and result.get("config_sha256") == plan.sha256
        and result.get("config_path") == str(plan.path),
        f"{fold_path}: result provenance drifted",
    )
    _require(
        result.get("outer_family") == runner.ONLY_OUTER_FAMILY,
        "diagnostic release accepts only the Boeing outer fold",
    )
    _require(
        _strict_json_equal(result.get(runner.METHOD_BINDING_KEY), expected_method),
        f"{fold_path}: result method binding drifted",
    )
    _require(
        completion.get("result_manifest_content_sha256")
        == result.get("content_sha256"),
        f"{fold_path}: result content binding drifted",
    )
    return {
        "RUN_COMPLETE.json": completion_snapshot,
        "result_manifest.json": result_snapshot,
    }


def _authenticate_fold(
    plan: runner.Plan,
    fold_path: Path,
    *,
    device: str,
    expected_fold_commit: str,
) -> base.AuthenticatedFold:
    fold_path = fold_path.resolve()
    envelope_snapshots = _preflight_boeing_fold_envelope(
        plan,
        fold_path,
        expected_fold_commit=expected_fold_commit,
    )
    with runner.diagnostic_verify_runtime(plan):
        fold = base._authenticate_fold(
            plan,
            fold_path,
            device=device,
            expected_fold_commit=expected_fold_commit,
        )
    for name, snapshot in envelope_snapshots.items():
        base._require_same_snapshot(fold_path / name, snapshot)
    _require(
        fold.outer_family == runner.ONLY_OUTER_FAMILY,
        "diagnostic release accepts only the Boeing outer fold",
    )
    return fold


def _source_fold(fold: base.AuthenticatedFold) -> dict[str, Any]:
    source = {
        "outer_family": fold.outer_family,
        "run_directory": str(fold.path),
        "completion_file_sha256": fold.completion_file_sha256,
        "result_manifest_file_sha256": fold.result_manifest_file_sha256,
        "artifact_count": len(fold.artifact_identities),
        "artifacts": fold.artifact_identities,
    }
    _require(set(source) == SOURCE_FOLD_FIELDS, "source-fold fields drifted")
    _require(
        source["artifact_count"] == len(EXPECTED_RESULT_ARTIFACTS)
        and set(source["artifacts"]) == set(EXPECTED_RESULT_ARTIFACTS),
        "source-fold artifact contract drifted",
    )
    return runner.inherited._json_safe(source)


def _support(plan: runner.Plan, fold: base.AuthenticatedFold) -> dict[str, Any]:
    return base._authenticate_support_audit(
        plan,
        runner.ONLY_OUTER_FAMILY,
        fold.summary["class_conditional_support"],
    )


def _report_payload(
    plan: runner.Plan,
    fold: base.AuthenticatedFold,
    *,
    git_commit: str,
    method: Mapping[str, Any],
    row: Mapping[str, Any],
    support: Mapping[str, Any],
    summary_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": REPORT_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": STATUS,
        "config_sha256": plan.sha256,
        **_method_identity_fields(),
        runner.METHOD_BINDING_KEY: method,
        "aggregator_git_commit": git_commit,
        "aggregator_worktree_clean": True,
        "fold_numerical_git_commit": fold.numerical_git_commit,
        "input_manifest_sha256": fold.input_manifest_sha256,
        "input_manifest_rows_sha256": fold.input_manifest_rows_sha256,
        "outer_family": fold.outer_family,
        "class_conditional_support": support,
        "fold_summary_source": FOLD_SUMMARY_SOURCE,
        "fold": row,
        "boeing_outer_summary_file": SUMMARY_FILE,
        "boeing_outer_summary_file_sha256": summary_sha256,
        "formal_confirmation": False,
        "evidence_scope": EVIDENCE_SCOPE,
        "allowed_use": "fixed_source_visualization_and_descriptive_error_analysis_only",
    }
    _require(not (set(payload) & FORBIDDEN_RELEASE_FIELDS), "forbidden report semantics")
    return _self_hashed(payload)


def _manifest_payload(
    plan: runner.Plan,
    fold: base.AuthenticatedFold,
    *,
    git_commit: str,
    method: Mapping[str, Any],
    summary_sha256: str,
    report_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": STATUS,
        "config_sha256": plan.sha256,
        **_method_identity_fields(),
        runner.METHOD_BINDING_KEY: method,
        "aggregator_git_commit": git_commit,
        "aggregator_worktree_clean": True,
        "fold_numerical_git_commit": fold.numerical_git_commit,
        "boeing_outer_summary_file": SUMMARY_FILE,
        "boeing_outer_summary_file_sha256": summary_sha256,
        "report_file": REPORT_FILE,
        "report_file_sha256": report_sha256,
        "source_folds": [source],
        "evidence_scope": EVIDENCE_SCOPE,
    }
    _require(not (set(payload) & FORBIDDEN_RELEASE_FIELDS), "forbidden manifest semantics")
    return _self_hashed(payload)


def _completion_payload(
    plan: runner.Plan,
    fold: base.AuthenticatedFold,
    *,
    git_commit: str,
    method: Mapping[str, Any],
    summary_sha256: str,
    report_sha256: str,
    manifest_sha256: str,
    completed_utc: str,
) -> dict[str, Any]:
    payload = {
        "schema": COMPLETE_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": STATUS,
        "config_sha256": plan.sha256,
        **_method_identity_fields(),
        runner.METHOD_BINDING_KEY: method,
        "aggregator_git_commit": git_commit,
        "aggregator_worktree_clean": True,
        "fold_numerical_git_commit": fold.numerical_git_commit,
        "diagnostic_manifest_file": MANIFEST_FILE,
        "diagnostic_manifest_file_sha256": manifest_sha256,
        "report_file": REPORT_FILE,
        "report_file_sha256": report_sha256,
        "boeing_outer_summary_file": SUMMARY_FILE,
        "boeing_outer_summary_file_sha256": summary_sha256,
        "evidence_scope": EVIDENCE_SCOPE,
        "completed_utc": completed_utc,
    }
    _require(not (set(payload) & FORBIDDEN_RELEASE_FIELDS), "forbidden completion semantics")
    return _self_hashed(payload)


def _release_identity(path: Path, sha256: str) -> dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256}


def aggregate(
    config_path: str | Path,
    run_directory: str | Path,
    output_dir: str | Path,
    *,
    expected_fold_commit: str,
    kinematic_input_manifest_path: str | Path,
    kinematic_input_manifest_file_sha256: str,
    synthetic_pass_path: str | Path,
    synthetic_pass_file_sha256: str,
    sidecar_root: str | Path,
    sidecar_population_manifest_path: str | Path,
    sidecar_population_manifest_file_sha256: str,
    device: str = "cpu",
    expected_config_sha256: str | None = runner.EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    """Freshly authenticate one Boeing fold and publish four immutable files."""

    _require(_is_lower_hex(expected_fold_commit, 40), "fold Git commit is invalid")
    _require(device == "cpu", "frozen Boeing diagnostic authenticator is CPU-only")
    plan = runner.load_plan(config_path)
    _validate_plan(plan)
    if expected_config_sha256 is not None:
        _require(plan.sha256 == expected_config_sha256, "diagnostic config SHA-256 mismatch")
    plan = runner.bind_early_evidence(
        plan,
        kinematic_input_manifest_path=kinematic_input_manifest_path,
        kinematic_input_manifest_file_sha256=kinematic_input_manifest_file_sha256,
        synthetic_pass_path=synthetic_pass_path,
        synthetic_pass_file_sha256=synthetic_pass_file_sha256,
        sidecar_root=sidecar_root,
        sidecar_population_manifest_path=sidecar_population_manifest_path,
        sidecar_population_manifest_file_sha256=(
            sidecar_population_manifest_file_sha256
        ),
    )
    fold_path = Path(run_directory).resolve()
    destination = Path(output_dir).resolve()
    _require(not destination.exists(), f"immutable output directory exists: {destination}")
    _require(fold_path not in destination.parents, "release directory cannot be inside fold")
    git_commit, dirty = runner._git_identity()
    _require(not dirty, "diagnostic aggregation requires a clean committed worktree")
    _require(
        git_commit == expected_fold_commit
        and plan.source_identity is not None
        and plan.source_identity.git_commit == git_commit,
        "diagnostic aggregate, fold, and Early evidence must use one exact commit",
    )
    base.parent._configure_execution(device)
    fold = _authenticate_fold(
        plan,
        fold_path,
        device=device,
        expected_fold_commit=expected_fold_commit,
    )
    _require(fold.path == fold_path, "authenticated fold path drifted")
    method = _method_binding(plan, expected_fold_commit)
    row = base._family_summary_row(fold)
    support = _support(plan, fold)
    source = _source_fold(fold)

    destination.mkdir(parents=True, exist_ok=False)
    summary_path = destination / SUMMARY_FILE
    summary_sha = base.parent._atomic_csv(summary_path, FAMILY_SUMMARY_FIELDS, [row])
    report = _report_payload(
        plan,
        fold,
        git_commit=git_commit,
        method=method,
        row=row,
        support=support,
        summary_sha256=summary_sha,
    )
    report_path = destination / REPORT_FILE
    report_sha = base.parent._atomic_json(report_path, report)
    manifest = _manifest_payload(
        plan,
        fold,
        git_commit=git_commit,
        method=method,
        summary_sha256=summary_sha,
        report_sha256=report_sha,
        source=source,
    )
    manifest_path = destination / MANIFEST_FILE
    manifest_sha = base.parent._atomic_json(manifest_path, manifest)
    published = {
        SUMMARY_FILE: (*_release_identity(summary_path, summary_sha).values(), False),
        REPORT_FILE: (*_release_identity(report_path, report_sha).values(), True),
        MANIFEST_FILE: (*_release_identity(manifest_path, manifest_sha).values(), True),
    }
    _require(
        {path.name for path in destination.iterdir()} == set(published),
        "pre-completion diagnostic release file set drifted",
    )
    for name, (size_bytes, sha256, self_hashed_json) in published.items():
        base._authenticate_published_output(
            destination / name,
            size_bytes=size_bytes,
            sha256=sha256,
            self_hashed_json=self_hashed_json,
        )
    completion = _completion_payload(
        plan,
        fold,
        git_commit=git_commit,
        method=method,
        summary_sha256=summary_sha,
        report_sha256=report_sha,
        manifest_sha256=manifest_sha,
        completed_utc=base.parent._utc_now(),
    )
    completion_path = destination / COMPLETE_FILE
    completion_sha = base.parent._atomic_json(completion_path, completion)
    persisted, _ = base._load_self_hashed_json(
        completion_path,
        expected_file_sha256=completion_sha,
    )
    _require(_strict_json_equal(persisted, completion), "completion content drifted")
    _require(
        {path.name for path in destination.iterdir()} == set(RELEASE_FILES),
        "diagnostic release must contain exactly four files",
    )
    for name, (size_bytes, sha256, self_hashed_json) in published.items():
        base._authenticate_published_output(
            destination / name,
            size_bytes=size_bytes,
            sha256=sha256,
            self_hashed_json=self_hashed_json,
        )
    return report


def _require_release_envelope(
    payload: Mapping[str, Any],
    *,
    expected_fields: set[str],
    expected_schema: str,
    plan: runner.Plan,
    expected_commit: str,
    expected_method: Mapping[str, Any],
    label: str,
) -> None:
    _require(set(payload) == expected_fields, f"{label} fields drifted")
    _require(
        payload.get("schema") == expected_schema
        and payload.get("experiment") == runner.EXPERIMENT
        and payload.get("status") == STATUS
        and payload.get("config_sha256") == plan.sha256
        and payload.get("aggregator_git_commit") == expected_commit
        and payload.get("aggregator_worktree_clean") is True
        and payload.get("fold_numerical_git_commit") == expected_commit
        and payload.get("evidence_scope") == EVIDENCE_SCOPE,
        f"{label} provenance drifted",
    )
    _require(
        all(payload.get(name) == value for name, value in _method_identity_fields().items()),
        f"{label} direct-parent/core identity drifted",
    )
    _require(
        _strict_json_equal(payload.get(runner.METHOD_BINDING_KEY), expected_method),
        f"{label} method binding drifted",
    )
    _require(not (set(payload) & FORBIDDEN_RELEASE_FIELDS), f"{label} has forbidden semantics")


def authenticate_diagnostic_release(
    output_directory: str | Path,
    *,
    expected_completion_sha256: str,
    expected_fold_commit: str,
    expected_config_sha256: str,
    expected_fold_directory: str | Path,
) -> dict[str, Any]:
    """Freshly reconstruct the four-file Boeing diagnostic and source fold."""

    root = Path(output_directory).resolve()
    fold_path = Path(expected_fold_directory).resolve()
    _require(root.is_dir(), f"diagnostic release directory is missing: {root}")
    _require(
        {path.name for path in root.iterdir()} == set(RELEASE_FILES),
        "diagnostic release file set drifted",
    )
    _require(_is_lower_hex(expected_completion_sha256, 64), "completion SHA-256 is invalid")
    _require(_is_lower_hex(expected_fold_commit, 40), "fold Git commit is invalid")
    plan = runner.load_plan(runner.CONFIG_PATH)
    _validate_plan(plan)
    _require(plan.sha256 == expected_config_sha256, "diagnostic release config drifted")
    plan = _bind_configured_evidence(plan)
    expected_method = _method_binding(plan, expected_fold_commit)

    # Rebuild prediction/support and recompute labels/metrics before trusting
    # any performance-bearing byte in the release directory.
    base.parent._configure_execution("cpu")
    fold = _authenticate_fold(
        plan,
        fold_path,
        device="cpu",
        expected_fold_commit=expected_fold_commit,
    )
    _require(fold.path == fold_path, "freshly authenticated fold path drifted")
    fold_marker_snapshots = {
        "RUN_COMPLETE.json": base._read_file_snapshot(fold_path / "RUN_COMPLETE.json"),
        "result_manifest.json": base._read_file_snapshot(
            fold_path / "result_manifest.json"
        ),
    }
    _require(
        fold_marker_snapshots["RUN_COMPLETE.json"].sha256
        == fold.completion_file_sha256
        and fold_marker_snapshots["result_manifest.json"].sha256
        == fold.result_manifest_file_sha256,
        "diagnostic source marker/result identity changed after fold authentication",
    )
    row = base._family_summary_row(fold)
    support = _support(plan, fold)
    source = _source_fold(fold)

    snapshots = {name: base._read_file_snapshot(root / name) for name in RELEASE_FILES}
    _require(
        snapshots[COMPLETE_FILE].sha256 == expected_completion_sha256,
        "diagnostic completion SHA-256 mismatch",
    )
    report = base._json_from_snapshot(
        snapshots[REPORT_FILE], path=root / REPORT_FILE, self_hashed=True
    )
    manifest = base._json_from_snapshot(
        snapshots[MANIFEST_FILE], path=root / MANIFEST_FILE, self_hashed=True
    )
    completion = base._json_from_snapshot(
        snapshots[COMPLETE_FILE], path=root / COMPLETE_FILE, self_hashed=True
    )
    _require_release_envelope(
        report,
        expected_fields=REPORT_FIELDS,
        expected_schema=REPORT_SCHEMA,
        plan=plan,
        expected_commit=expected_fold_commit,
        expected_method=expected_method,
        label="diagnostic report",
    )
    _require_release_envelope(
        manifest,
        expected_fields=MANIFEST_FIELDS,
        expected_schema=MANIFEST_SCHEMA,
        plan=plan,
        expected_commit=expected_fold_commit,
        expected_method=expected_method,
        label="diagnostic manifest",
    )
    _require_release_envelope(
        completion,
        expected_fields=COMPLETE_FIELDS,
        expected_schema=COMPLETE_SCHEMA,
        plan=plan,
        expected_commit=expected_fold_commit,
        expected_method=expected_method,
        label="diagnostic completion",
    )
    _require(
        isinstance(completion.get("completed_utc"), str)
        and completion["completed_utc"] != "",
        "diagnostic completion time is invalid",
    )

    summary_sha = snapshots[SUMMARY_FILE].sha256
    report_sha = snapshots[REPORT_FILE].sha256
    manifest_sha = snapshots[MANIFEST_FILE].sha256
    base._authenticate_single_fold_summary_csv(
        snapshots[SUMMARY_FILE], expected_fold_row=row
    )
    expected_report = _report_payload(
        plan,
        fold,
        git_commit=expected_fold_commit,
        method=expected_method,
        row=row,
        support=support,
        summary_sha256=summary_sha,
    )
    expected_manifest = _manifest_payload(
        plan,
        fold,
        git_commit=expected_fold_commit,
        method=expected_method,
        summary_sha256=summary_sha,
        report_sha256=report_sha,
        source=source,
    )
    expected_completion = _completion_payload(
        plan,
        fold,
        git_commit=expected_fold_commit,
        method=expected_method,
        summary_sha256=summary_sha,
        report_sha256=report_sha,
        manifest_sha256=manifest_sha,
        completed_utc=completion["completed_utc"],
    )
    for observed, snapshot, expected, label in (
        (report, snapshots[REPORT_FILE], expected_report, "diagnostic report"),
        (manifest, snapshots[MANIFEST_FILE], expected_manifest, "diagnostic manifest"),
        (completion, snapshots[COMPLETE_FILE], expected_completion, "diagnostic completion"),
    ):
        base._require_canonical_json_snapshot(observed, snapshot, expected, label=label)
    _require(
        hashlib.sha256(_canonical_json_bytes(expected_completion)).hexdigest()
        == expected_completion_sha256,
        "canonical diagnostic completion SHA-256 mismatch",
    )
    _require(
        report.get("boeing_outer_summary_file") == SUMMARY_FILE
        and manifest.get("boeing_outer_summary_file") == SUMMARY_FILE
        and completion.get("boeing_outer_summary_file") == SUMMARY_FILE
        and report.get("boeing_outer_summary_file_sha256") == summary_sha
        and manifest.get("boeing_outer_summary_file_sha256") == summary_sha
        and completion.get("boeing_outer_summary_file_sha256") == summary_sha
        and manifest.get("report_file") == REPORT_FILE
        and completion.get("report_file") == REPORT_FILE
        and manifest.get("report_file_sha256") == report_sha
        and completion.get("report_file_sha256") == report_sha
        and completion.get("diagnostic_manifest_file") == MANIFEST_FILE
        and completion.get("diagnostic_manifest_file_sha256") == manifest_sha,
        "diagnostic release cross-file binding drifted",
    )
    _require(
        report.get("outer_family") == runner.ONLY_OUTER_FAMILY
        and _strict_json_equal(report.get("fold"), row)
        and _strict_json_equal(report.get("class_conditional_support"), support)
        and report.get("fold_summary_source") == FOLD_SUMMARY_SOURCE
        and report.get("formal_confirmation") is False
        and report.get("allowed_use")
        == "fixed_source_visualization_and_descriptive_error_analysis_only",
        "diagnostic report scientific scope drifted",
    )
    source_folds = manifest.get("source_folds")
    _require(
        isinstance(source_folds, list)
        and len(source_folds) == 1
        and _strict_json_equal(source_folds[0], source),
        "diagnostic source-fold binding drifted",
    )
    _require(
        {path.name for path in fold_path.iterdir()} == set(runner.REQUIRED_FOLD_FILES),
        "diagnostic source fold file set drifted",
    )
    for name, identity in source["artifacts"].items():
        _require(
            isinstance(identity, Mapping)
            and set(identity) == {"size_bytes", "sha256"}
            and isinstance(identity.get("size_bytes"), int)
            and not isinstance(identity.get("size_bytes"), bool)
            and identity["size_bytes"] >= 0
            and _is_lower_hex(identity.get("sha256"), 64),
            f"diagnostic source artifact identity drifted: {name}",
        )
        base.parent._stable_file_identity(
            fold_path / name, identity["size_bytes"], identity["sha256"]
        )
    for name, snapshot in snapshots.items():
        base._require_same_snapshot(root / name, snapshot)
    for name, snapshot in fold_marker_snapshots.items():
        base._require_same_snapshot(fold_path / name, snapshot)
    return {
        "outer_family": runner.ONLY_OUTER_FAMILY,
        "fold_directory": str(fold_path),
        "fold_summary": runner.inherited._json_safe(row),
        "selected_candidate": runner.inherited._json_safe(fold.selected_candidate),
        "source_fold": source,
        "method_binding": expected_method,
        "release_files": {
            name: {
                "size_bytes": snapshots[name].identity.size,
                "sha256": snapshots[name].sha256,
            }
            for name in RELEASE_FILES
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(runner.CONFIG_PATH))
    parser.add_argument("--expected-config-sha256", default=runner.EXPECTED_CONFIG_SHA256)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-fold-commit", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--kinematic-input-manifest", required=True)
    parser.add_argument("--kinematic-input-manifest-sha256", required=True)
    parser.add_argument("--synthetic-pass", required=True)
    parser.add_argument("--synthetic-pass-sha256", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--sidecar-population-manifest", required=True)
    parser.add_argument("--sidecar-population-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    report = aggregate(
        arguments.config,
        arguments.run_dir,
        arguments.output_dir,
        expected_fold_commit=arguments.expected_fold_commit,
        kinematic_input_manifest_path=arguments.kinematic_input_manifest,
        kinematic_input_manifest_file_sha256=(
            arguments.kinematic_input_manifest_sha256
        ),
        synthetic_pass_path=arguments.synthetic_pass,
        synthetic_pass_file_sha256=arguments.synthetic_pass_sha256,
        sidecar_root=arguments.sidecar_root,
        sidecar_population_manifest_path=arguments.sidecar_population_manifest,
        sidecar_population_manifest_file_sha256=(
            arguments.sidecar_population_manifest_sha256
        ),
        device=arguments.device,
        expected_config_sha256=arguments.expected_config_sha256,
    )
    print(f"experiment={report['experiment']}")
    print(f"outer_family={report['outer_family']}")
    print(f"report_content_sha256={report['content_sha256']}")


if __name__ == "__main__":
    main()

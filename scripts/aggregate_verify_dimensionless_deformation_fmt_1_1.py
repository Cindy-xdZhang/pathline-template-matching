#!/usr/bin/env python3
"""Authenticate one or five Dimensionless-Deformation FMT outer folds.

Only the completed marker and the label-free artifact subset are opened before
fresh replay.  The replay reloads Raw672, rebuilds the frozen row-local
dimensionless representation and independent FMT, reconstructs the inherited
scaler/calibrator, and authenticates every prediction.  Result, labels and
outer metrics remain unopened until that replay has succeeded.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.portable_flow import sha256_file  # noqa: E402
from scripts import run_verify_dimensionless_deformation_fmt_1_1 as runner  # noqa: E402


parent = runner.inherited
AGGREGATE_SUMMARY_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_aggregate_summary.v1"
)
SINGLE_FOLD_REPORT_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_"
    "single_fold_authentication_report.v1"
)
EARLY_STOP_CERTIFICATE_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_early_stop_certificate.v1"
)
AGGREGATE_MANIFEST_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_aggregate_manifest.v1"
)
AGGREGATE_COMPLETE_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_aggregate_complete.v1"
)

EXPECTED_FOLD_FILES = runner.REQUIRED_FOLD_FILES
EXPECTED_RESULT_ARTIFACTS = tuple(
    name
    for name in EXPECTED_FOLD_FILES
    if name not in {"result_manifest.json", "RUN_COMPLETE.json"}
)
LABEL_FREE_PRE_RESULT_FILES = (
    "inner_group_metrics.csv",
    "inner_candidate_summary.csv",
    "inner_fit_audits.json",
    "final_per_scale_scaler.npz",
    "final_per_scale_scaler_manifest.json",
    "final_tail_calibration.npz",
    "final_tail_calibration_manifest.json",
    "selected_candidate.json",
    "outer_predictions.npz",
    "outer_prediction_manifest.json",
)
COMPLETION_FIELDS = {
    "schema",
    "experiment",
    "outer_family",
    "git_commit",
    "config_sha256",
    "result_manifest_file",
    "result_manifest_file_sha256",
    "result_manifest_content_sha256",
    "completed_utc",
    runner.METHOD_BINDING_KEY,
    "content_sha256",
}
RESULT_FIELDS = {
    "schema",
    "experiment",
    "status",
    "completed_utc",
    "git_commit",
    "config_path",
    "config_sha256",
    "input_manifest",
    "outer_family",
    "selected_candidate",
    "selected_candidate_file",
    "selected_candidate_file_sha256",
    "selected_candidate_content_sha256",
    "final_scaler_manifest_file_sha256",
    "final_scaler_file_sha256",
    "final_calibration_manifest_file_sha256",
    "final_calibration_file_sha256",
    "prediction_manifest_file_sha256",
    "prediction_file_sha256",
    "inner_group_metrics_file_sha256",
    "inner_candidate_summary_file_sha256",
    "inner_fit_audits_file_sha256",
    "outer_group_metrics_file_sha256",
    "outer_summary_file_sha256",
    "outer_reference_access_audit_file_sha256",
    "outer_summary",
    "environment",
    "artifacts",
    runner.METHOD_BINDING_KEY,
    "content_sha256",
}
DIRECT_ARTIFACT_HASH_FIELDS = {
    "inner_group_metrics.csv": "inner_group_metrics_file_sha256",
    "inner_candidate_summary.csv": "inner_candidate_summary_file_sha256",
    "inner_fit_audits.json": "inner_fit_audits_file_sha256",
    "final_per_scale_scaler.npz": "final_scaler_file_sha256",
    "final_per_scale_scaler_manifest.json": "final_scaler_manifest_file_sha256",
    "final_tail_calibration.npz": "final_calibration_file_sha256",
    "final_tail_calibration_manifest.json": "final_calibration_manifest_file_sha256",
    "selected_candidate.json": "selected_candidate_file_sha256",
    "outer_predictions.npz": "prediction_file_sha256",
    "outer_prediction_manifest.json": "prediction_manifest_file_sha256",
    "outer_group_metrics.csv": "outer_group_metrics_file_sha256",
    "outer_summary.json": "outer_summary_file_sha256",
    "outer_reference_access_audit.json": "outer_reference_access_audit_file_sha256",
}

FAMILY_METRIC_FIELDS = (
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
    "retrieval_support_fraction",
    "calibration_support_fraction",
    "spatial_imputed_fraction",
    "spatial_unimputable_fraction",
)
FAMILY_COUNT_FIELDS = (
    "sample_count",
    "positive_count",
    "negative_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "retrieval_supported_count",
    "calibration_supported_count",
    "imputed_count",
    "unimputable_count",
    "calibration_mode_0_count",
    "calibration_mode_1_count",
    "calibration_mode_2_count",
    "calibration_mode_3_count",
    "calibration_mode_4_count",
    "calibration_mode_5_count",
    "scaler_mode_0_count",
    "scaler_mode_1_count",
    "scaler_mode_2_count",
    "scaler_mode_3_count",
)
FAMILY_SUMMARY_FIELDS = (
    "outer_family",
    "run_directory",
    "numerical_git_commit",
    "config_sha256",
    "parent_config_sha256",
    "core_sha256",
    "input_manifest_sha256",
    "input_manifest_rows_sha256",
    "requested_device",
    "selected_candidate_id",
    *FAMILY_METRIC_FIELDS,
    *FAMILY_COUNT_FIELDS,
    "completion_file_sha256",
    "completion_content_sha256",
    "result_manifest_file_sha256",
    "result_manifest_content_sha256",
    "outer_group_metrics_file_sha256",
)
CSV_STRING_FIELDS = {
    "outer_family",
    "inner_family",
    "dataset",
    "block",
    "candidate_id",
    "representation",
    "decision_rule",
}
CSV_INTEGER_FIELDS = {"source_ordinal", "k", *FAMILY_COUNT_FIELDS}
SUMMARY_RELATIVE_TOLERANCE = 1.0e-9
SUMMARY_ABSOLUTE_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class AuthenticatedFold:
    path: Path
    outer_family: str
    numerical_git_commit: str
    config_sha256: str
    parent_config_sha256: str
    core_sha256: str
    input_manifest_sha256: str
    input_manifest_rows_sha256: str
    requested_device: str
    selected_candidate: Mapping[str, Any]
    summary: Mapping[str, Any]
    artifact_identities: Mapping[str, Mapping[str, Any]]
    completion_file_sha256: str
    completion_content_sha256: str
    result_manifest_file_sha256: str
    result_manifest_content_sha256: str


@dataclass(frozen=True)
class _FileIdentity:
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_FileIdentity":
        return cls(
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
        )


@dataclass(frozen=True)
class _FileSnapshot:
    content: bytes
    identity: _FileIdentity
    sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_lower_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _read_file_snapshot(path: Path) -> _FileSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        before = _FileIdentity.from_stat(os.fstat(descriptor))
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = _FileIdentity.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    _require(before == after, f"file identity changed while reading: {path}")
    path_after = _FileIdentity.from_stat(path.stat(follow_symlinks=False))
    _require(path_after == before, f"file path was replaced while reading: {path}")
    content = b"".join(chunks)
    _require(len(content) == before.size, f"file size changed while reading: {path}")
    return _FileSnapshot(
        content=content,
        identity=before,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _require_same_snapshot(path: Path, expected: _FileSnapshot) -> None:
    observed = _read_file_snapshot(path)
    _require(
        observed.identity == expected.identity and observed.sha256 == expected.sha256,
        f"file identity or content changed during authentication: {path}",
    )


def _json_from_snapshot(
    snapshot: _FileSnapshot,
    *,
    path: Path,
    self_hashed: bool,
) -> dict[str, Any]:
    value = json.loads(snapshot.content.decode("utf-8"))
    _require(isinstance(value, Mapping), f"JSON root is invalid: {path}")
    if self_hashed:
        parent._authenticate_self_hash(value)
    return dict(value)


def _load_self_hashed_json(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    snapshot = _read_file_snapshot(path)
    if expected_file_sha256 is not None:
        _require(
            _is_lower_hex(expected_file_sha256, 64)
            and snapshot.sha256 == expected_file_sha256,
            f"file SHA-256 mismatch: {path}",
        )
    return (
        _json_from_snapshot(snapshot, path=path, self_hashed=True),
        snapshot.sha256,
    )


def _stage_snapshot(path: Path, snapshot: _FileSnapshot) -> None:
    observed_sha = runner._atomic_bytes(path, snapshot.content)
    _require(observed_sha == snapshot.sha256, f"staged SHA-256 drifted: {path.name}")
    observed = _read_file_snapshot(path)
    _require(
        observed.content == snapshot.content and observed.sha256 == snapshot.sha256,
        f"staged artifact bytes drifted: {path.name}",
    )


def _method_binding(plan: runner.Plan, git_commit: str) -> dict[str, Any]:
    return runner._json_safe(runner._method_binding(plan, git_commit))


def _require_method_binding(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(
        payload.get(runner.METHOD_BINDING_KEY) == expected,
        f"{label}: dimensionless method/config/parent/core/scope binding drifted",
    )


def _validate_plan_output_contract(plan: runner.Plan) -> None:
    _require(
        plan.required_fold_files == EXPECTED_FOLD_FILES == runner.REQUIRED_FOLD_FILES,
        "frozen required fold file order or names drifted",
    )
    _require(len(EXPECTED_FOLD_FILES) == 15, "fold file count drifted")
    _require(len(EXPECTED_RESULT_ARTIFACTS) == 13, "result artifact count drifted")
    _require(
        len(runner.PREDICTION_ARRAY_DTYPES) == 19,
        "outer prediction array contract drifted",
    )
    _require(
        plan.sha256 == runner.EXPECTED_CONFIG_SHA256
        and plan.parent_experiment_config_sha256
        == runner.EXPECTED_PARENT_CONFIG_SHA256
        and plan.core_sha256 == runner.EXPECTED_CORE_SHA256,
        "active config, parent config, or numerical core identity drifted",
    )


def _artifact_identities(
    fold_path: Path,
    result: Mapping[str, Any],
    snapshots: Mapping[str, _FileSnapshot],
) -> dict[str, dict[str, Any]]:
    artifacts = result.get("artifacts")
    _require(isinstance(artifacts, Mapping), f"{fold_path}: artifact map is invalid")
    _require(
        set(artifacts) == set(EXPECTED_RESULT_ARTIFACTS),
        f"{fold_path}: result must authenticate exactly 13 non-marker artifacts",
    )
    identities: dict[str, dict[str, Any]] = {}
    for name in EXPECTED_RESULT_ARTIFACTS:
        record = artifacts[name]
        _require(
            isinstance(record, Mapping) and set(record) == {"size_bytes", "sha256"},
            f"{fold_path}: artifact identity contract drifted: {name}",
        )
        snapshot = snapshots[name]
        _require(
            record.get("size_bytes") == snapshot.identity.size
            and record.get("sha256") == snapshot.sha256,
            f"{fold_path}: artifact snapshot identity drifted: {name}",
        )
        _require(
            result.get(DIRECT_ARTIFACT_HASH_FIELDS[name]) == snapshot.sha256,
            f"{fold_path}: direct/result artifact SHA drifted: {name}",
        )
        identities[name] = {
            "size_bytes": snapshot.identity.size,
            "sha256": snapshot.sha256,
        }
    return identities


def _reauthenticate_artifact_identities(
    fold_path: Path,
    identities: Mapping[str, Mapping[str, Any]],
) -> None:
    for name in EXPECTED_RESULT_ARTIFACTS:
        snapshot = _read_file_snapshot(fold_path / name)
        record = identities[name]
        _require(
            snapshot.identity.size == record["size_bytes"]
            and snapshot.sha256 == record["sha256"],
            f"{fold_path}: artifact identity changed after authentication: {name}",
        )


def _candidate_from_payload(
    plan: runner.Plan,
    payload: object,
) -> parent.TailCandidateSpec:
    _require(isinstance(payload, Mapping), "selected candidate is invalid")
    expected_fields = {
        "candidate_id",
        "representation",
        "k",
        "sigma",
        "decision_rule",
        "decision_value",
    }
    _require(set(payload) == expected_fields, "selected candidate fields drifted")
    candidate = parent.TailCandidateSpec(
        representation=str(payload["representation"]),
        k=int(payload["k"]),
        sigma=float(payload["sigma"]),
        decision_rule=str(payload["decision_rule"]),
        decision_value=float(payload["decision_value"]),
    )
    _require(
        runner._json_safe(parent._candidate_payload(candidate)) == dict(payload),
        "selected candidate is not the canonical typed rule",
    )
    candidates = {value.candidate_id: value for value in runner.candidate_specs(plan)}
    _require(
        candidate.candidate_id in candidates
        and candidates[candidate.candidate_id] == candidate,
        "selected candidate is outside the frozen 3060-candidate grid",
    )
    return candidate


def _csv_text(value: Any) -> str:
    if isinstance(value, (np.bool_, bool)):
        return str(int(bool(value)))
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return "" if not np.isfinite(numeric) else format(numeric, ".12g")
    if value is None:
        return ""
    return str(value)


def _parse_csv_row(row: Mapping[str, str], *, label: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for field in parent.METRIC_FIELDS:
        text = row[field]
        if field in CSV_STRING_FIELDS:
            parsed[field] = text
        elif field in CSV_INTEGER_FIELDS:
            _require(text != "", f"{label}: missing integer {field}")
            try:
                value = int(text, 10)
            except ValueError as error:
                raise ValueError(f"{label}: invalid integer {field}") from error
            _require(str(value) == text, f"{label}: noncanonical integer {field}")
            parsed[field] = value
        elif text == "":
            parsed[field] = float("nan")
        else:
            try:
                value = float(text)
            except ValueError as error:
                raise ValueError(f"{label}: invalid float {field}") from error
            _require(np.isfinite(value), f"{label}: nonfinite persisted float {field}")
            parsed[field] = value
    return parsed


def _authenticate_outer_metric_csv(
    *,
    snapshot: _FileSnapshot,
    expected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    with io.StringIO(snapshot.content.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        _require(
            tuple(reader.fieldnames or ()) == parent.METRIC_FIELDS,
            "outer group metric CSV fields drifted",
        )
        persisted = list(reader)
    _require(
        len(persisted) == len(expected_rows),
        "outer group metric row count differs from fresh label evaluation",
    )
    parsed: list[dict[str, Any]] = []
    for index, (raw, expected) in enumerate(zip(persisted, expected_rows)):
        _require(
            tuple(expected) == parent.METRIC_FIELDS,
            f"fresh outer metric row field contract drifted: {index}",
        )
        for field in parent.METRIC_FIELDS:
            _require(
                raw[field] == _csv_text(expected[field]),
                f"outer group metric differs from fresh labels: row={index}/{field}",
            )
        parsed.append(_parse_csv_row(raw, label=f"outer group metric row {index}"))
    return parsed


def _family_summary_from_rows(
    plan: runner.Plan,
    family: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_groups = {
        (dataset, source_ordinal, block)
        for dataset in plan.families[family]
        for source_ordinal in range(4)
        for block in runner.BLOCK_NAMES
    }
    observed_groups: set[tuple[str, int, str]] = set()
    for row in rows:
        _require(row["outer_family"] == family, "outer metric family binding drifted")
        _require(
            row["inner_family"] == "outer_evaluation_only",
            "outer metric inner-family sentinel drifted",
        )
        key = (str(row["dataset"]), int(row["source_ordinal"]), str(row["block"]))
        _require(key not in observed_groups, "duplicate outer metric group")
        observed_groups.add(key)
    _require(observed_groups == expected_groups, "outer metric group set is incomplete")
    return runner._outer_summary(rows, family)


def _require_summary_equal(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(set(actual) == set(expected), f"{label}: field set drifted")
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if field in FAMILY_METRIC_FIELDS:
            expected_numeric = float(expected_value)
            if not np.isfinite(expected_numeric):
                _require(
                    actual_value is None or not np.isfinite(float(actual_value)),
                    f"{label}: {field} mismatch",
                )
            else:
                _require(
                    isinstance(actual_value, (int, float))
                    and not isinstance(actual_value, bool)
                    and math.isclose(
                        float(actual_value),
                        expected_numeric,
                        rel_tol=SUMMARY_RELATIVE_TOLERANCE,
                        abs_tol=SUMMARY_ABSOLUTE_TOLERANCE,
                    ),
                    f"{label}: {field} mismatch",
                )
        else:
            _require(actual_value == expected_value, f"{label}: {field} mismatch")


def _authenticate_outer_summary(
    fold_path: Path,
    *,
    fresh_rows: Sequence[Mapping[str, Any]],
    csv_rows: Sequence[Mapping[str, Any]],
    plan: runner.Plan,
    family: str,
    snapshot: _FileSnapshot,
    expected_method: Mapping[str, Any],
) -> dict[str, Any]:
    label_summary = _family_summary_from_rows(plan, family, fresh_rows)
    csv_summary = _family_summary_from_rows(plan, family, csv_rows)
    _require_summary_equal(
        csv_summary,
        label_summary,
        label="CSV summary versus fresh-label summary",
    )
    persisted = _json_from_snapshot(
        snapshot,
        path=fold_path / "outer_summary.json",
        self_hashed=True,
    )
    _require_method_binding(persisted, expected_method, label="outer summary")
    persisted_without_auth = dict(persisted)
    persisted_without_auth.pop("content_sha256")
    persisted_without_auth.pop(runner.METHOD_BINDING_KEY)
    _require_summary_equal(
        persisted_without_auth,
        label_summary,
        label="outer_summary.json versus fresh-label summary",
    )
    return label_summary


def _authenticate_reference_audit(
    fold_path: Path,
    *,
    result: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
    snapshot: _FileSnapshot,
    expected_method: Mapping[str, Any],
) -> None:
    persisted = _json_from_snapshot(
        snapshot,
        path=fold_path / "outer_reference_access_audit.json",
        self_hashed=True,
    )
    _require_method_binding(persisted, expected_method, label="outer reference audit")
    expected = parent._manifest_with_self_hash(
        {
            "schema": runner.REFERENCE_AUDIT_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "outer_family": result["outer_family"],
            "first_open_phase": "after_outer_prediction_file_and_manifest_authentication",
            "prediction_manifest_file_sha256": result[
                "prediction_manifest_file_sha256"
            ],
            "prediction_file_sha256": result["prediction_file_sha256"],
            "row_count": len(expected_rows),
            "rows": list(expected_rows),
            runner.METHOD_BINDING_KEY: expected_method,
        }
    )
    _require(
        persisted == runner._json_safe(expected),
        "outer reference access audit differs from the fresh label gate",
    )


def _require_result_input_manifest_binding(
    result: Mapping[str, Any],
    plan: runner.Plan,
) -> None:
    observed = result.get("input_manifest")
    _require(isinstance(observed, Mapping), "result input manifest is invalid")
    expected = {
        "schema": plan.manifest_schema,
        "path": str(plan.manifest_path),
        "size_bytes": plan.manifest_size,
        "sha256": plan.manifest_sha256,
        "rows_content_sha256": plan.manifest_rows_sha256,
    }
    _require(dict(observed) == expected, "result input manifest scope drifted")


def _authenticate_prediction_array_contract(snapshot: _FileSnapshot) -> None:
    with np.load(io.BytesIO(snapshot.content), allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == tuple(runner.PREDICTION_ARRAY_DTYPES),
            "outer prediction must contain the exact ordered 19 arrays",
        )
        row_counts: set[int] = set()
        for name, dtype in runner.PREDICTION_ARRAY_DTYPES.items():
            values = np.asarray(archive[name])
            _require(values.ndim == 1, f"prediction array is not one-dimensional: {name}")
            _require(values.dtype == dtype, f"prediction dtype drifted: {name}")
            row_counts.add(len(values))
        _require(len(row_counts) == 1, "prediction array row counts drifted")


def _authenticate_fold(
    plan: runner.Plan,
    fold_path: Path,
    *,
    device: str,
    expected_fold_commit: str,
) -> AuthenticatedFold:
    fold_path = fold_path.resolve()
    _require(fold_path.is_dir(), f"fold directory does not exist: {fold_path}")
    _require(
        {path.name for path in fold_path.iterdir()} == set(EXPECTED_FOLD_FILES)
        and all((fold_path / name).is_file() for name in EXPECTED_FOLD_FILES),
        f"{fold_path}: completed fold must contain exactly the frozen 15 files",
    )
    expected_method = _method_binding(plan, expected_fold_commit)

    # RUN_COMPLETE carries no performance values and binds the immutable fold.
    completion_snapshot = _read_file_snapshot(fold_path / "RUN_COMPLETE.json")
    completion = _json_from_snapshot(
        completion_snapshot,
        path=fold_path / "RUN_COMPLETE.json",
        self_hashed=True,
    )
    _require(set(completion) == COMPLETION_FIELDS, f"{fold_path}: completion fields drifted")
    _require(
        completion.get("schema") == runner.COMPLETE_SCHEMA
        and completion.get("experiment") == runner.EXPERIMENT,
        f"{fold_path}: completion schema or experiment drifted",
    )
    _require(
        completion.get("git_commit") == expected_fold_commit
        and completion.get("config_sha256") == plan.sha256,
        f"{fold_path}: completion commit/config drifted",
    )
    _require_method_binding(completion, expected_method, label="completion")
    _require(
        completion.get("result_manifest_file") == "result_manifest.json",
        f"{fold_path}: result path drifted",
    )
    family = str(completion.get("outer_family"))
    _require(family in plan.family_order, f"{fold_path}: unknown outer family")

    # This authenticates only the train-only manifest envelope; it opens no
    # cache feature or label member.
    _, input_identity = runner.load_cache_rows(plan)

    # Freeze only label-free bytes.  Result, outer metric, outer summary and
    # outer label-audit files are deliberately absent from this mapping.
    snapshots: dict[str, _FileSnapshot] = {
        name: _read_file_snapshot(fold_path / name)
        for name in LABEL_FREE_PRE_RESULT_FILES
    }
    with tempfile.TemporaryDirectory(prefix="ptm_dimensionless_fold_auth_") as temporary:
        staged = Path(temporary)
        for name in LABEL_FREE_PRE_RESULT_FILES:
            _stage_snapshot(staged / name, snapshots[name])

        selected_payload = _json_from_snapshot(
            snapshots["selected_candidate.json"],
            path=fold_path / "selected_candidate.json",
            self_hashed=True,
        )
        _require(
            selected_payload.get("schema") == runner.SELECTED_SCHEMA
            and selected_payload.get("experiment") == runner.EXPERIMENT
            and selected_payload.get("config_sha256") == plan.sha256
            and selected_payload.get("git_commit") == expected_fold_commit
            and selected_payload.get("outer_family") == family,
            f"{fold_path}: label-free selection binding drifted",
        )
        _require_method_binding(
            selected_payload,
            expected_method,
            label="selected candidate",
        )
        candidate = _candidate_from_payload(plan, selected_payload.get("candidate"))

        # The runner must rebuild Raw672 -> dimensionless deformation -> FMT,
        # scaler, calibrator and prediction.  It opens labels only after that
        # exact prediction replay has authenticated successfully.
        fresh_rows, reference_rows = runner.evaluate_outer_prediction(
            plan,
            candidate,
            staged,
            outer_family=family,
            git_commit=expected_fold_commit,
            device=device,
            expected_scaler_manifest_sha256=snapshots[
                "final_per_scale_scaler_manifest.json"
            ].sha256,
            expected_calibration_manifest_sha256=snapshots[
                "final_tail_calibration_manifest.json"
            ].sha256,
            expected_selected_candidate_sha256=snapshots[
                "selected_candidate.json"
            ].sha256,
            expected_prediction_manifest_sha256=snapshots[
                "outer_prediction_manifest.json"
            ].sha256,
            inner_group_metrics_path=staged / "inner_group_metrics.csv",
            inner_group_metrics_sha256=snapshots["inner_group_metrics.csv"].sha256,
            inner_candidate_summary_path=staged / "inner_candidate_summary.csv",
            inner_candidate_summary_sha256=snapshots[
                "inner_candidate_summary.csv"
            ].sha256,
            inner_fit_audits_path=staged / "inner_fit_audits.json",
            inner_fit_audits_sha256=snapshots["inner_fit_audits.json"].sha256,
        )

        # Fresh replay and the runner-owned label gate have closed.  Only now
        # may authentication open result or any performance-bearing artifact.
        result_snapshot = _read_file_snapshot(fold_path / "result_manifest.json")
        _require(
            result_snapshot.sha256
            == completion.get("result_manifest_file_sha256"),
            f"{fold_path}: result manifest file SHA-256 mismatch",
        )
        result = _json_from_snapshot(
            result_snapshot,
            path=fold_path / "result_manifest.json",
            self_hashed=True,
        )
        _require(set(result) == RESULT_FIELDS, f"{fold_path}: result fields drifted")
        _require(
            result.get("schema") == runner.RESULT_SCHEMA
            and result.get("experiment") == runner.EXPERIMENT
            and result.get("status") == "completed",
            f"{fold_path}: result schema/status drifted",
        )
        _require(
            result.get("git_commit") == expected_fold_commit
            and result.get("config_sha256") == plan.sha256
            and result.get("config_path") == str(plan.path)
            and result.get("outer_family") == family,
            f"{fold_path}: result provenance drifted",
        )
        _require_method_binding(result, expected_method, label="result")
        _require(
            completion.get("result_manifest_content_sha256")
            == result.get("content_sha256"),
            f"{fold_path}: result content binding drifted",
        )
        _require_result_input_manifest_binding(result, plan)
        _require(
            result.get("input_manifest") == runner._json_safe(input_identity),
            f"{fold_path}: result input identity differs from fresh manifest",
        )
        environment = result.get("environment")
        _require(isinstance(environment, Mapping), f"{fold_path}: environment is invalid")
        _require(
            environment.get("requested_device") == device,
            f"{fold_path}: aggregation device differs from numerical fold device",
        )
        canonical_candidate = runner._json_safe(parent._candidate_payload(candidate))
        _require(
            result.get("selected_candidate") == canonical_candidate
            and result.get("selected_candidate_file") == "selected_candidate.json",
            f"{fold_path}: result/selection candidate drifted",
        )

        for name in (
            "outer_group_metrics.csv",
            "outer_summary.json",
            "outer_reference_access_audit.json",
        ):
            snapshots[name] = _read_file_snapshot(fold_path / name)
        identities = _artifact_identities(fold_path, result, snapshots)
        _authenticate_prediction_array_contract(snapshots["outer_predictions.npz"])
        _require(
            selected_payload.get("content_sha256")
            == result["selected_candidate_content_sha256"],
            f"{fold_path}: selected-candidate content hash drifted",
        )
        parsed_rows = _authenticate_outer_metric_csv(
            snapshot=snapshots["outer_group_metrics.csv"],
            expected_rows=fresh_rows,
        )
        summary = _authenticate_outer_summary(
            fold_path,
            fresh_rows=fresh_rows,
            csv_rows=parsed_rows,
            plan=plan,
            family=family,
            snapshot=snapshots["outer_summary.json"],
            expected_method=expected_method,
        )
        _require_summary_equal(
            result["outer_summary"],
            summary,
            label="result embedded summary versus fresh-label summary",
        )
        _authenticate_reference_audit(
            fold_path,
            result=result,
            expected_rows=reference_rows,
            snapshot=snapshots["outer_reference_access_audit.json"],
            expected_method=expected_method,
        )

    for name, snapshot in snapshots.items():
        _require_same_snapshot(fold_path / name, snapshot)
    _require_same_snapshot(fold_path / "RUN_COMPLETE.json", completion_snapshot)
    _require_same_snapshot(fold_path / "result_manifest.json", result_snapshot)
    _reauthenticate_artifact_identities(fold_path, identities)
    _require(
        {path.name for path in fold_path.iterdir()} == set(EXPECTED_FOLD_FILES),
        f"{fold_path}: fold file set changed during authentication",
    )
    return AuthenticatedFold(
        path=fold_path,
        outer_family=family,
        numerical_git_commit=expected_fold_commit,
        config_sha256=plan.sha256,
        parent_config_sha256=plan.parent_experiment_config_sha256,
        core_sha256=plan.core_sha256,
        input_manifest_sha256=plan.manifest_sha256,
        input_manifest_rows_sha256=plan.manifest_rows_sha256,
        requested_device=device,
        selected_candidate=runner._deep_freeze(canonical_candidate),
        summary=runner._deep_freeze(summary),
        artifact_identities=runner._deep_freeze(identities),
        completion_file_sha256=completion_snapshot.sha256,
        completion_content_sha256=str(completion["content_sha256"]),
        result_manifest_file_sha256=result_snapshot.sha256,
        result_manifest_content_sha256=str(result["content_sha256"]),
    )


def _stop_rule_thresholds(plan: runner.Plan) -> dict[str, Any]:
    success = plan.raw.get("success_rule")
    early = plan.raw.get("early_stop_rule")
    _require(isinstance(success, Mapping), "success rule is missing")
    _require(isinstance(early, Mapping), "early-stop rule is missing")
    expected_success = {
        "five_family_macro_f1_min",
        "families_with_f1_at_least_0_65_min",
        "single_family_f1_min",
        "five_family_macro_average_precision_min",
        "five_family_macro_balanced_accuracy_min",
        "five_family_macro_precision_min",
        "five_family_macro_recall_min",
        "all_conditions_required",
    }
    _require(set(success) == expected_success, "success rule fields drifted")
    _require(success.get("all_conditions_required") is True, "success conjunction drifted")
    _require(
        early.get("permitted_only_for_mathematical_impossibility") is True
        and tuple(early.get("conditions", ()))
        == (
            "any_completed_family_f1_below_0_50",
            "two_completed_families_f1_below_0_65",
            "optimistic_remaining_family_bound_cannot_meet_any_macro_threshold",
        ),
        "mathematical early-stop rule drifted",
    )
    return {
        "five_family_macro_f1_min": float(success["five_family_macro_f1_min"]),
        "families_with_f1_at_least_0_65_min": int(
            success["families_with_f1_at_least_0_65_min"]
        ),
        "single_family_f1_min": float(success["single_family_f1_min"]),
        "five_family_macro_average_precision_min": float(
            success["five_family_macro_average_precision_min"]
        ),
        "five_family_macro_balanced_accuracy_min": float(
            success["five_family_macro_balanced_accuracy_min"]
        ),
        "five_family_macro_precision_min": float(
            success["five_family_macro_precision_min"]
        ),
        "five_family_macro_recall_min": float(
            success["five_family_macro_recall_min"]
        ),
    }


def _stop_rule(
    plan: runner.Plan,
    family_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    _require(len(family_rows) == 5, "five-fold stop rule requires five families")
    thresholds = _stop_rule_thresholds(plan)
    macro = {
        field: float(
            np.mean([float(row[field]) for row in family_rows], dtype=np.float64)
        )
        for field in FAMILY_METRIC_FIELDS
    }
    _require(
        np.isfinite(
            [
                macro["f1"],
                macro["average_precision"],
                macro["balanced_accuracy"],
                macro["precision"],
                macro["recall"],
            ]
        ).all(),
        "complete-fold stop-rule metrics must be finite",
    )
    outcomes = {
        "five_family_macro_f1_pass": (
            macro["f1"] >= thresholds["five_family_macro_f1_min"]
        ),
        "families_with_f1_at_least_0_65_pass": (
            sum(float(row["f1"]) >= 0.65 for row in family_rows)
            >= thresholds["families_with_f1_at_least_0_65_min"]
        ),
        "single_family_f1_min_pass": (
            min(float(row["f1"]) for row in family_rows)
            >= thresholds["single_family_f1_min"]
        ),
        "five_family_macro_average_precision_pass": (
            macro["average_precision"]
            >= thresholds["five_family_macro_average_precision_min"]
        ),
        "five_family_macro_balanced_accuracy_pass": (
            macro["balanced_accuracy"]
            >= thresholds["five_family_macro_balanced_accuracy_min"]
        ),
        "five_family_macro_precision_pass": (
            macro["precision"] >= thresholds["five_family_macro_precision_min"]
        ),
        "five_family_macro_recall_pass": (
            macro["recall"] >= thresholds["five_family_macro_recall_min"]
        ),
    }
    return {"thresholds": thresholds, "family_macro": macro}, outcomes


def _early_stop_certificate(
    plan: runner.Plan,
    family_rows: Sequence[Mapping[str, Any]],
    *,
    numerical_git_commit: str,
) -> dict[str, Any]:
    """Return a proof based only on optimistic mathematical upper bounds."""

    _require(len(family_rows) == 1, "single-fold certificate requires one fold")
    thresholds = _stop_rule_thresholds(plan)
    total_count = len(plan.family_order)
    observed_count = len(family_rows)
    remaining_count = total_count - observed_count
    observed_f1 = [float(row["f1"]) for row in family_rows]
    minimum_impossible = min(observed_f1) < thresholds["single_family_f1_min"]
    below_065 = sum(value < 0.65 for value in observed_f1)
    count_impossible = (
        below_065
        > total_count - thresholds["families_with_f1_at_least_0_65_min"]
    )
    macro_thresholds = {
        "f1": thresholds["five_family_macro_f1_min"],
        "average_precision": thresholds[
            "five_family_macro_average_precision_min"
        ],
        "balanced_accuracy": thresholds[
            "five_family_macro_balanced_accuracy_min"
        ],
        "precision": thresholds["five_family_macro_precision_min"],
        "recall": thresholds["five_family_macro_recall_min"],
    }
    upper_bounds: dict[str, Any] = {}
    for field, threshold in macro_thresholds.items():
        values = [float(row[field]) for row in family_rows]
        _require(np.isfinite(values).all(), f"single-fold {field} must be finite")
        optimistic = (sum(values) + remaining_count) / total_count
        upper_bounds[field] = {
            "observed_sum": float(sum(values)),
            "remaining_family_count": remaining_count,
            "unobserved_metric_upper_bound": 1.0,
            "best_possible_five_family_macro": float(optimistic),
            "required_minimum": float(threshold),
            "mathematically_impossible": bool(optimistic < threshold),
        }
    reasons: list[str] = []
    if minimum_impossible:
        reasons.append("observed_family_f1_below_frozen_minimum")
    if count_impossible:
        reasons.append("too_many_observed_families_below_0.65_for_four_of_five_rule")
    reasons.extend(
        f"best_possible_{field}_macro_below_frozen_minimum"
        for field, record in upper_bounds.items()
        if record["mathematically_impossible"]
    )
    impossible = bool(reasons)
    return {
        "schema": EARLY_STOP_CERTIFICATE_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": "evaluated",
        "config_sha256": plan.sha256,
        "parent_config_sha256": plan.parent_experiment_config_sha256,
        "core_sha256": plan.core_sha256,
        "fold_numerical_git_commit": numerical_git_commit,
        "observed_outer_families": [str(row["outer_family"]) for row in family_rows],
        "observed_outer_family_count": observed_count,
        "frozen_total_outer_family_count": total_count,
        "remaining_outer_family_count": remaining_count,
        "thresholds": thresholds,
        "any_observed_family_f1_below_minimum": minimum_impossible,
        "at_least_two_observed_families_f1_below_0_65": count_impossible,
        "macro_upper_bound_proofs": upper_bounds,
        "mathematically_impossible_to_pass": impossible,
        "impossibility_reasons": reasons,
        "stop_version": impossible,
        "five_fold_success_evaluated": False,
        "five_fold_success": None,
        runner.METHOD_BINDING_KEY: _method_binding(plan, numerical_git_commit),
    }


def _self_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    return parent._manifest_with_self_hash(runner._json_safe(dict(payload)))


def _authenticate_published_output(
    path: Path,
    *,
    size_bytes: int,
    sha256: str,
    self_hashed_json: bool,
) -> None:
    parent._stable_file_identity(path, size_bytes, sha256)
    if self_hashed_json:
        _load_self_hashed_json(path, expected_file_sha256=sha256)


def aggregate(
    config_path: str | Path,
    run_directories: Sequence[str | Path],
    output_dir: str | Path,
    *,
    expected_fold_commit: str,
    mode: str = "auto",
    device: str = "cpu",
    expected_config_sha256: str | None = runner.EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    plan = runner.load_plan(config_path)
    _validate_plan_output_contract(plan)
    if expected_config_sha256 is not None:
        _require(
            plan.sha256 == expected_config_sha256,
            "frozen config SHA-256 mismatch",
        )
    _require(
        _is_lower_hex(expected_fold_commit, 40),
        "expected fold Git commit must be a full lowercase SHA-1",
    )
    _require(
        mode in {"auto", "single-fold", "complete-five-fold"},
        "invalid aggregation mode",
    )
    paths = tuple(Path(value).resolve() for value in run_directories)
    _require(len(paths) == len(set(paths)), "fold directories must be unique")
    if mode == "single-fold":
        _require(len(paths) == 1, "single-fold mode requires exactly one fold")
        effective_mode = "single_fold_authentication"
    elif mode == "complete-five-fold":
        _require(len(paths) == 5, "complete-five-fold mode requires exactly five folds")
        effective_mode = "complete_five_fold_aggregate"
    elif len(paths) == 1:
        effective_mode = "single_fold_authentication"
    elif len(paths) == 5:
        effective_mode = "complete_five_fold_aggregate"
    else:
        raise ValueError("auto mode requires either one first-family fold or all five folds")

    destination = Path(output_dir).resolve()
    _require(not destination.exists(), f"immutable output directory exists: {destination}")
    _require(
        all(path not in destination.parents for path in paths),
        "aggregate output directory must be outside every fold directory",
    )
    aggregator_commit, dirty = runner._git_identity()
    _require(not dirty, "aggregate requires a clean committed worktree")
    _require(
        aggregator_commit == expected_fold_commit,
        "aggregate and folds must use one exact commit",
    )
    expected_method = _method_binding(plan, expected_fold_commit)
    _require(
        expected_method["experiment_config"]["sha256"] == plan.sha256
        and expected_method["parent_per_scale_config"]["sha256"]
        == plan.parent_experiment_config_sha256
        and expected_method["core"]["sha256"] == plan.core_sha256
        and expected_method["input_member"] == "raw_features"
        and expected_method["forbidden_cache_member"] == "fmt_features",
        "aggregate method scope is not the frozen Raw-only dimensionless method",
    )
    runner._configure_execution(device)
    folds = tuple(
        _authenticate_fold(
            plan,
            path,
            device=device,
            expected_fold_commit=expected_fold_commit,
        )
        for path in paths
    )
    if effective_mode == "single_fold_authentication":
        _require(
            folds[0].outer_family == plan.family_order[0] == "half_cylinder",
            "single-fold mode is restricted to the first frozen half_cylinder fold",
        )
    else:
        observed = [fold.outer_family for fold in folds]
        _require(
            len(set(observed)) == 5 and set(observed) == set(plan.family_order),
            "complete aggregation requires each frozen outer family exactly once",
        )
    _require(
        {fold.numerical_git_commit for fold in folds} == {expected_fold_commit},
        "folds mix numerical Git commits",
    )
    _require(
        {fold.config_sha256 for fold in folds} == {plan.sha256},
        "folds mix active configs",
    )
    _require(
        {fold.parent_config_sha256 for fold in folds}
        == {plan.parent_experiment_config_sha256},
        "folds mix parent PerScale configs",
    )
    _require(
        {fold.core_sha256 for fold in folds} == {plan.core_sha256},
        "folds mix dimensionless numerical cores",
    )
    _require(
        len(
            {
                (fold.input_manifest_sha256, fold.input_manifest_rows_sha256)
                for fold in folds
            }
        )
        == 1,
        "folds mix train-only input scope",
    )
    by_family = {fold.outer_family: fold for fold in folds}
    ordered_folds = tuple(
        by_family[family] for family in plan.family_order if family in by_family
    )

    family_rows: list[dict[str, Any]] = []
    for fold in ordered_folds:
        row: dict[str, Any] = {
            "outer_family": fold.outer_family,
            "run_directory": str(fold.path),
            "numerical_git_commit": fold.numerical_git_commit,
            "config_sha256": fold.config_sha256,
            "parent_config_sha256": fold.parent_config_sha256,
            "core_sha256": fold.core_sha256,
            "input_manifest_sha256": fold.input_manifest_sha256,
            "input_manifest_rows_sha256": fold.input_manifest_rows_sha256,
            "requested_device": fold.requested_device,
            "selected_candidate_id": fold.selected_candidate["candidate_id"],
        }
        row.update({field: fold.summary[field] for field in FAMILY_METRIC_FIELDS})
        row.update({field: int(fold.summary[field]) for field in FAMILY_COUNT_FIELDS})
        row.update(
            {
                "completion_file_sha256": fold.completion_file_sha256,
                "completion_content_sha256": fold.completion_content_sha256,
                "result_manifest_file_sha256": fold.result_manifest_file_sha256,
                "result_manifest_content_sha256": fold.result_manifest_content_sha256,
                "outer_group_metrics_file_sha256": fold.artifact_identities[
                    "outer_group_metrics.csv"
                ]["sha256"],
            }
        )
        family_rows.append(row)

    early_stop_payload: dict[str, Any] | None = None
    if effective_mode == "complete_five_fold_aggregate":
        stop_inputs, outcomes = _stop_rule(plan, family_rows)
        family_macro: Mapping[str, Any] | None = stop_inputs["family_macro"]
        stop_rule: Mapping[str, Any] = {
            "evaluated": True,
            **stop_inputs,
            "outcomes": outcomes,
            "all_success_conditions_pass": all(outcomes.values()),
        }
        all_success: bool | None = all(outcomes.values())
    else:
        family_macro = None
        early_stop_payload = _early_stop_certificate(
            plan,
            family_rows,
            numerical_git_commit=expected_fold_commit,
        )
        stop_rule = {
            "evaluated": False,
            "reason": "complete_five_outer_family_set_required",
        }
        all_success = None

    destination.mkdir(parents=True, exist_ok=False)
    table_path = destination / "outer_family_summary.csv"
    table_sha = runner._atomic_csv(table_path, FAMILY_SUMMARY_FIELDS, family_rows)
    early_stop_record: dict[str, Any] | None = None
    if effective_mode == "complete_five_fold_aggregate":
        report_name = "aggregate_summary.json"
        report_payload: dict[str, Any] = {
            "schema": AGGREGATE_SUMMARY_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": effective_mode,
            "config_sha256": plan.sha256,
            "parent_config_sha256": plan.parent_experiment_config_sha256,
            "core_sha256": plan.core_sha256,
            runner.METHOD_BINDING_KEY: expected_method,
            "aggregator_git_commit": aggregator_commit,
            "aggregator_worktree_clean": True,
            "fold_numerical_git_commit": expected_fold_commit,
            "input_manifest_sha256": folds[0].input_manifest_sha256,
            "input_manifest_rows_sha256": folds[0].input_manifest_rows_sha256,
            "outer_families": [fold.outer_family for fold in ordered_folds],
            "outer_family_count": len(ordered_folds),
            "aggregation": "equal_outer_physical_family_macro",
            "fold_summary_source": (
                "fresh_Raw672_to_dimensionless_to_FMT_to_prediction_replay_"
                "before_outer_label_gate_then_exact_metric_recomputation"
            ),
            "family_macro": family_macro,
            "success_stop_rule": stop_rule,
            "all_success_conditions_pass": all_success,
            "outer_family_summary_file_sha256": table_sha,
            "folds": family_rows,
            "formal_confirmation": False,
            "evidence_scope": "exposed_train_only_nested_family_validation",
        }
    else:
        assert early_stop_payload is not None
        certificate = _self_hashed(early_stop_payload)
        certificate_path = destination / "early_stop_certificate.json"
        certificate_sha = runner._atomic_json(certificate_path, certificate)
        early_stop_record = {
            "path": certificate_path.name,
            "size_bytes": certificate_path.stat().st_size,
            "sha256": certificate_sha,
            "content_sha256": certificate["content_sha256"],
        }
        report_name = "single_fold_authentication_report.json"
        report_payload = {
            "schema": SINGLE_FOLD_REPORT_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": effective_mode,
            "config_sha256": plan.sha256,
            "parent_config_sha256": plan.parent_experiment_config_sha256,
            "core_sha256": plan.core_sha256,
            runner.METHOD_BINDING_KEY: expected_method,
            "aggregator_git_commit": aggregator_commit,
            "aggregator_worktree_clean": True,
            "fold_numerical_git_commit": expected_fold_commit,
            "input_manifest_sha256": folds[0].input_manifest_sha256,
            "input_manifest_rows_sha256": folds[0].input_manifest_rows_sha256,
            "outer_family": folds[0].outer_family,
            "fold_summary_source": (
                "fresh_Raw672_to_dimensionless_to_FMT_to_prediction_replay_"
                "before_outer_label_gate_then_exact_metric_recomputation"
            ),
            "fold": family_rows[0],
            "early_stop_certificate": early_stop_record,
            "stop_version": bool(early_stop_payload["stop_version"]),
            "five_fold_success_evaluated": False,
            "five_fold_success": None,
            "outer_family_summary_file_sha256": table_sha,
            "formal_confirmation": False,
            "evidence_scope": "exposed_train_only_nested_family_validation",
        }
    report = _self_hashed(report_payload)
    report_path = destination / report_name
    report_sha = runner._atomic_json(report_path, report)
    manifest = _self_hashed(
        {
            "schema": AGGREGATE_MANIFEST_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": effective_mode,
            "config_sha256": plan.sha256,
            "parent_config_sha256": plan.parent_experiment_config_sha256,
            "core_sha256": plan.core_sha256,
            runner.METHOD_BINDING_KEY: expected_method,
            "aggregator_git_commit": aggregator_commit,
            "aggregator_worktree_clean": True,
            "fold_numerical_git_commit": expected_fold_commit,
            "outer_family_summary_file": table_path.name,
            "outer_family_summary_file_sha256": table_sha,
            "report_file": report_path.name,
            "report_file_sha256": report_sha,
            "early_stop_certificate": early_stop_record,
            "source_folds": [
                {
                    "outer_family": fold.outer_family,
                    "run_directory": str(fold.path),
                    "completion_file_sha256": fold.completion_file_sha256,
                    "result_manifest_file_sha256": fold.result_manifest_file_sha256,
                    "artifact_count": len(fold.artifact_identities),
                    "artifacts": fold.artifact_identities,
                }
                for fold in ordered_folds
            ],
        }
    )
    manifest_path = destination / "aggregate_manifest.json"
    manifest_sha = runner._atomic_json(manifest_path, manifest)
    published_outputs = {
        table_path.name: {
            "size_bytes": table_path.stat().st_size,
            "sha256": table_sha,
            "self_hashed_json": False,
        },
        report_path.name: {
            "size_bytes": report_path.stat().st_size,
            "sha256": report_sha,
            "self_hashed_json": True,
        },
        manifest_path.name: {
            "size_bytes": manifest_path.stat().st_size,
            "sha256": manifest_sha,
            "self_hashed_json": True,
        },
    }
    if early_stop_record is not None:
        published_outputs["early_stop_certificate.json"] = {
            "size_bytes": early_stop_record["size_bytes"],
            "sha256": early_stop_record["sha256"],
            "self_hashed_json": True,
        }
    _require(
        {path.name for path in destination.iterdir()} == set(published_outputs),
        "pre-completion aggregate output file set drifted",
    )
    for name, record in published_outputs.items():
        _authenticate_published_output(
            destination / name,
            size_bytes=int(record["size_bytes"]),
            sha256=str(record["sha256"]),
            self_hashed_json=bool(record["self_hashed_json"]),
        )

    completion = _self_hashed(
        {
            "schema": AGGREGATE_COMPLETE_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": effective_mode,
            "config_sha256": plan.sha256,
            "parent_config_sha256": plan.parent_experiment_config_sha256,
            "core_sha256": plan.core_sha256,
            runner.METHOD_BINDING_KEY: expected_method,
            "aggregator_git_commit": aggregator_commit,
            "aggregator_worktree_clean": True,
            "fold_numerical_git_commit": expected_fold_commit,
            "aggregate_manifest_file": manifest_path.name,
            "aggregate_manifest_file_sha256": manifest_sha,
            "report_file": report_path.name,
            "report_file_sha256": report_sha,
            "early_stop_certificate": early_stop_record,
            "completed_utc": parent._utc_now(),
        }
    )
    completion_path = destination / "AGGREGATE_COMPLETE.json"
    completion_sha = runner._atomic_json(completion_path, completion)
    persisted_completion, _ = _load_self_hashed_json(
        completion_path,
        expected_file_sha256=completion_sha,
    )
    _require(
        persisted_completion == runner._json_safe(completion),
        "aggregate completion content drifted",
    )
    for name, record in published_outputs.items():
        _authenticate_published_output(
            destination / name,
            size_bytes=int(record["size_bytes"]),
            sha256=str(record["sha256"]),
            self_hashed_json=bool(record["self_hashed_json"]),
        )
    _require(
        {path.name for path in destination.iterdir()}
        == {*published_outputs, completion_path.name},
        "aggregate output file set drifted",
    )
    return report


def authenticate_single_fold_release(
    output_directory: str | Path,
    *,
    expected_completion_sha256: str,
    expected_fold_commit: str,
    expected_config_sha256: str,
    expected_fold_directory: str | Path,
) -> dict[str, Any]:
    """Reconstruct the first-fold release without trusting a stop boolean."""

    root = Path(output_directory).resolve()
    fold_path = Path(expected_fold_directory).resolve()
    _require(root.is_dir(), f"first-fold authentication directory is missing: {root}")
    expected_names = {
        "outer_family_summary.csv",
        "early_stop_certificate.json",
        "single_fold_authentication_report.json",
        "aggregate_manifest.json",
        "AGGREGATE_COMPLETE.json",
    }
    _require(
        {path.name for path in root.iterdir()} == expected_names,
        "single-fold authentication file set drifted",
    )
    _require(_is_lower_hex(expected_fold_commit, 40), "release commit is invalid")
    plan = runner.load_plan(runner.CONFIG_PATH)
    _validate_plan_output_contract(plan)
    _require(
        plan.sha256 == expected_config_sha256,
        "single-fold release config drifted",
    )
    expected_method = _method_binding(plan, expected_fold_commit)

    snapshots = {name: _read_file_snapshot(root / name) for name in expected_names}
    _require(
        snapshots["AGGREGATE_COMPLETE.json"].sha256
        == expected_completion_sha256,
        "single-fold aggregate completion SHA-256 mismatch",
    )
    completion = _json_from_snapshot(
        snapshots["AGGREGATE_COMPLETE.json"],
        path=root / "AGGREGATE_COMPLETE.json",
        self_hashed=True,
    )
    _require(
        completion.get("schema") == AGGREGATE_COMPLETE_SCHEMA
        and completion.get("experiment") == runner.EXPERIMENT
        and completion.get("status") == "completed"
        and completion.get("mode") == "single_fold_authentication",
        "single-fold aggregate completion schema/status drifted",
    )
    _require(
        completion.get("aggregator_git_commit") == expected_fold_commit
        and completion.get("fold_numerical_git_commit") == expected_fold_commit
        and completion.get("config_sha256") == plan.sha256
        and completion.get("parent_config_sha256")
        == plan.parent_experiment_config_sha256
        and completion.get("core_sha256") == plan.core_sha256,
        "single-fold aggregate completion provenance drifted",
    )
    _require_method_binding(completion, expected_method, label="aggregate completion")
    _require(
        completion.get("report_file") == "single_fold_authentication_report.json"
        and completion.get("aggregate_manifest_file") == "aggregate_manifest.json",
        "single-fold completion output paths drifted",
    )

    report_name = str(completion["report_file"])
    manifest_name = str(completion["aggregate_manifest_file"])
    report = _json_from_snapshot(
        snapshots[report_name], path=root / report_name, self_hashed=True
    )
    manifest = _json_from_snapshot(
        snapshots[manifest_name], path=root / manifest_name, self_hashed=True
    )
    _require(
        snapshots[report_name].sha256 == completion["report_file_sha256"]
        and snapshots[manifest_name].sha256
        == completion["aggregate_manifest_file_sha256"],
        "single-fold report/manifest file binding drifted",
    )
    for payload, schema, label in (
        (report, SINGLE_FOLD_REPORT_SCHEMA, "single-fold report"),
        (manifest, AGGREGATE_MANIFEST_SCHEMA, "single-fold manifest"),
    ):
        _require(
            payload.get("schema") == schema
            and payload.get("experiment") == runner.EXPERIMENT
            and payload.get("mode") == "single_fold_authentication"
            and payload.get("aggregator_git_commit") == expected_fold_commit
            and payload.get("fold_numerical_git_commit") == expected_fold_commit
            and payload.get("config_sha256") == plan.sha256
            and payload.get("parent_config_sha256")
            == plan.parent_experiment_config_sha256
            and payload.get("core_sha256") == plan.core_sha256,
            f"{label} provenance drifted",
        )
        _require_method_binding(payload, expected_method, label=label)

    certificate_record = completion.get("early_stop_certificate")
    _require(
        isinstance(certificate_record, Mapping)
        and set(certificate_record)
        == {"path", "size_bytes", "sha256", "content_sha256"}
        and certificate_record.get("path") == "early_stop_certificate.json",
        "single-fold early-stop record drifted",
    )
    _require(
        report.get("early_stop_certificate") == certificate_record
        and manifest.get("early_stop_certificate") == certificate_record,
        "single-fold early-stop record is not consistently bound",
    )
    certificate_snapshot = snapshots["early_stop_certificate.json"]
    _require(
        certificate_snapshot.identity.size == certificate_record["size_bytes"]
        and certificate_snapshot.sha256 == certificate_record["sha256"],
        "single-fold certificate file identity drifted",
    )
    certificate = _json_from_snapshot(
        certificate_snapshot,
        path=root / "early_stop_certificate.json",
        self_hashed=True,
    )
    _require(
        certificate.get("content_sha256") == certificate_record["content_sha256"]
        and certificate.get("schema") == EARLY_STOP_CERTIFICATE_SCHEMA
        and certificate.get("fold_numerical_git_commit") == expected_fold_commit
        and certificate.get("config_sha256") == plan.sha256
        and certificate.get("parent_config_sha256")
        == plan.parent_experiment_config_sha256
        and certificate.get("core_sha256") == plan.core_sha256,
        "single-fold certificate provenance drifted",
    )
    _require_method_binding(certificate, expected_method, label="early-stop certificate")
    _require(
        report.get("outer_family") == "half_cylinder"
        and certificate.get("observed_outer_families") == ["half_cylinder"]
        and report.get("five_fold_success_evaluated") is False
        and report.get("five_fold_success") is None
        and certificate.get("five_fold_success_evaluated") is False
        and certificate.get("five_fold_success") is None,
        "single-fold release made an invalid five-fold claim",
    )
    fold_row = report.get("fold")
    _require(isinstance(fold_row, Mapping), "single-fold report row is invalid")
    expected_certificate = _self_hashed(
        _early_stop_certificate(
            plan,
            [fold_row],
            numerical_git_commit=expected_fold_commit,
        )
    )
    _require(
        certificate == runner._json_safe(expected_certificate),
        "single-fold mathematical certificate did not reconstruct",
    )
    _require(
        report.get("stop_version") == certificate.get("stop_version"),
        "single-fold report/certificate stop decision drifted",
    )

    source_folds = manifest.get("source_folds")
    _require(
        isinstance(source_folds, list) and len(source_folds) == 1,
        "single-fold source manifest must contain one fold",
    )
    source = source_folds[0]
    _require(
        isinstance(source, Mapping)
        and source.get("outer_family") == "half_cylinder"
        and Path(str(source.get("run_directory"))).resolve() == fold_path,
        "single-fold source directory binding drifted",
    )
    _require(
        {path.name for path in fold_path.iterdir()} == set(EXPECTED_FOLD_FILES),
        "released first-fold file set drifted",
    )
    _require(
        _read_file_snapshot(fold_path / "RUN_COMPLETE.json").sha256
        == source.get("completion_file_sha256")
        and _read_file_snapshot(fold_path / "result_manifest.json").sha256
        == source.get("result_manifest_file_sha256"),
        "released first-fold marker/result SHA-256 drifted",
    )
    artifacts = source.get("artifacts")
    _require(
        source.get("artifact_count") == len(EXPECTED_RESULT_ARTIFACTS)
        and isinstance(artifacts, Mapping)
        and set(artifacts) == set(EXPECTED_RESULT_ARTIFACTS),
        "released first-fold artifact map drifted",
    )
    for name, identity in artifacts.items():
        _require(
            isinstance(identity, Mapping)
            and set(identity) == {"size_bytes", "sha256"},
            f"released artifact identity drifted: {name}",
        )
        parent._stable_file_identity(
            fold_path / name,
            int(identity["size_bytes"]),
            str(identity["sha256"]),
        )
    _require(
        snapshots["outer_family_summary.csv"].sha256
        == report.get("outer_family_summary_file_sha256")
        == manifest.get("outer_family_summary_file_sha256"),
        "single-fold summary table binding drifted",
    )
    for name, snapshot in snapshots.items():
        _require_same_snapshot(root / name, snapshot)
    return {
        "schema": "pathline_template_matching.dimensionless_deformation_fmt_single_fold_release.v1",
        "outer_family": "half_cylinder",
        "fold_numerical_git_commit": expected_fold_commit,
        "config_sha256": plan.sha256,
        "parent_config_sha256": plan.parent_experiment_config_sha256,
        "core_sha256": plan.core_sha256,
        "stop_version": bool(certificate["stop_version"]),
        "mathematically_impossible_to_pass": bool(
            certificate["mathematically_impossible_to_pass"]
        ),
        "completion_sha256": expected_completion_sha256,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(runner.CONFIG_PATH))
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=("auto", "single-fold", "complete-five-fold"),
        default="auto",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-fold-commit", required=True)
    parser.add_argument(
        "--expected-config-sha256",
        default=runner.EXPECTED_CONFIG_SHA256,
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = aggregate(
        arguments.config,
        arguments.run_dir,
        arguments.output_dir,
        expected_fold_commit=arguments.expected_fold_commit,
        mode=arguments.mode,
        device=arguments.device,
        expected_config_sha256=arguments.expected_config_sha256,
    )
    print(json.dumps(runner._json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

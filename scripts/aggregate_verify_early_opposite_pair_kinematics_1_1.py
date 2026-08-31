#!/usr/bin/env python3
"""Authenticate and aggregate Verify_EarlyOppositePairKinematics_1.1 folds.

One-fold mode is restricted to the first frozen outer family and performs
authentication only.  The frozen success stop rule is evaluated only when all
five unique outer-family folds are present.
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
from scripts import run_verify_early_opposite_pair_kinematics_1_1 as runner  # noqa: E402


AGGREGATE_SUMMARY_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_aggregate_summary.v1"
)
SINGLE_FOLD_REPORT_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_single_fold_authentication_report.v1"
)
EARLY_STOP_CERTIFICATE_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_early_stop_certificate.v1"
)
AGGREGATE_MANIFEST_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_aggregate_manifest.v1"
)
AGGREGATE_COMPLETE_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_aggregate_complete.v1"
)

EXPECTED_FOLD_FILES = (
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
    "outer_group_metrics.csv",
    "outer_summary.json",
    "outer_reference_access_audit.json",
    "result_manifest.json",
    "RUN_COMPLETE.json",
)
EXPECTED_RESULT_ARTIFACTS = tuple(
    name
    for name in EXPECTED_FOLD_FILES
    if name not in {"result_manifest.json", "RUN_COMPLETE.json"}
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
    "early_evidence",
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
    "environment",
    "artifacts",
    "content_sha256",
}

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
DIRECT_ARTIFACT_HASH_FIELDS = {
    "inner_group_metrics.csv": "inner_group_metrics_file_sha256",
    "inner_candidate_summary.csv": "inner_candidate_summary_file_sha256",
    "inner_fit_audits.json": "inner_fit_audits_file_sha256",
    "final_per_scale_scaler.npz": "final_scaler_file_sha256",
    "final_per_scale_scaler_manifest.json": (
        "final_scaler_manifest_file_sha256"
    ),
    "final_tail_calibration.npz": "final_calibration_file_sha256",
    "final_tail_calibration_manifest.json": (
        "final_calibration_manifest_file_sha256"
    ),
    "selected_candidate.json": "selected_candidate_file_sha256",
    "outer_predictions.npz": "prediction_file_sha256",
    "outer_prediction_manifest.json": "prediction_manifest_file_sha256",
    "outer_group_metrics.csv": "outer_group_metrics_file_sha256",
    "outer_summary.json": "outer_summary_file_sha256",
    "outer_reference_access_audit.json": (
        "outer_reference_access_audit_file_sha256"
    ),
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
CSV_INTEGER_FIELDS = {
    "source_ordinal",
    "k",
    *FAMILY_COUNT_FIELDS,
}
SUMMARY_RELATIVE_TOLERANCE = 1.0e-9
SUMMARY_ABSOLUTE_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class AuthenticatedFold:
    path: Path
    outer_family: str
    numerical_git_commit: str
    config_sha256: str
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
    """Read one descriptor and bind bytes to pre/post FD and path identity."""

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
        runner._authenticate_self_hash(value)
    return dict(value)


def _load_self_hashed_json(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    snapshot = _read_file_snapshot(path)
    file_sha256 = snapshot.sha256
    if expected_file_sha256 is not None:
        _require(
            _is_lower_hex(expected_file_sha256, 64)
            and file_sha256 == expected_file_sha256,
            f"file SHA-256 mismatch: {path}",
        )
    value = _json_from_snapshot(snapshot, path=path, self_hashed=True)
    return dict(value), file_sha256


def _stage_snapshot(path: Path, snapshot: _FileSnapshot) -> None:
    observed_sha = runner._publish_no_replace(
        path,
        lambda stream: stream.write(snapshot.content),
        text_mode=False,
    )
    _require(observed_sha == snapshot.sha256, f"staged SHA-256 drifted: {path.name}")
    observed = _read_file_snapshot(path)
    _require(
        observed.content == snapshot.content and observed.sha256 == snapshot.sha256,
        f"staged artifact bytes drifted: {path.name}",
    )


def _validate_plan_output_contract(plan: runner.Plan) -> None:
    _require(
        plan.required_fold_files == EXPECTED_FOLD_FILES,
        "frozen required fold file order or names drifted",
    )
    _require(
        len(EXPECTED_RESULT_ARTIFACTS) == 13,
        "per-scale result artifact count drifted",
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
        size = record.get("size_bytes")
        digest = record.get("sha256")
        _require(
            isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0
            and _is_lower_hex(digest, 64),
            f"{fold_path}: artifact identity is invalid: {name}",
        )
        _require(name in snapshots, f"{fold_path}: artifact snapshot is missing: {name}")
        snapshot = snapshots[name]
        _require(
            snapshot.identity.size == int(size) and snapshot.sha256 == str(digest),
            f"{fold_path}: artifact snapshot identity drifted: {name}",
        )
        direct_field = DIRECT_ARTIFACT_HASH_FIELDS[name]
        _require(
            result.get(direct_field) == digest,
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
        record = identities[name]
        snapshot = _read_file_snapshot(fold_path / name)
        _require(
            snapshot.identity.size == int(record["size_bytes"])
            and snapshot.sha256 == str(record["sha256"]),
            f"{fold_path}: artifact identity changed after authentication: {name}",
        )


def _candidate_from_payload(
    plan: runner.Plan,
    payload: object,
) -> runner.TailCandidateSpec:
    _require(isinstance(payload, Mapping), "result selected candidate is invalid")
    expected_fields = {
        "candidate_id",
        "representation",
        "k",
        "sigma",
        "decision_rule",
        "decision_value",
    }
    _require(set(payload) == expected_fields, "selected candidate field set drifted")
    candidate = runner.TailCandidateSpec(
        representation=str(payload["representation"]),
        k=int(payload["k"]),
        sigma=float(payload["sigma"]),
        decision_rule=str(payload["decision_rule"]),
        decision_value=float(payload["decision_value"]),
    )
    _require(
        candidate.candidate_id == payload.get("candidate_id"),
        "selected candidate ID does not match its numerical rule",
    )
    _require(
        dict(payload) == runner._json_safe(runner._candidate_payload(candidate)),
        "selected candidate payload is not the canonical typed rule",
    )
    candidates = {value.candidate_id: value for value in runner.candidate_specs(plan)}
    _require(
        candidate.candidate_id in candidates
        and candidate == candidates[candidate.candidate_id],
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
    for field in runner.METRIC_FIELDS:
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
        else:
            if text == "":
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
    path: Path,
    *,
    snapshot: _FileSnapshot,
    expected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    with io.StringIO(snapshot.content.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        _require(
            tuple(reader.fieldnames or ()) == runner.METRIC_FIELDS,
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
            tuple(expected) == runner.METRIC_FIELDS,
            f"fresh outer metric row field contract drifted: {index}",
        )
        for field in runner.METRIC_FIELDS:
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

    # Reproduce the final fold runner exactly.  All datasets have the same
    # four-source/two-block group count; the runner averages all finite groups
    # directly and excludes only nonfinite values for that metric.
    return runner._outer_summary(rows, family)


def _require_summary_equal(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(set(actual) == set(expected), f"{label}: field set drifted")
    for field in expected:
        expected_value = expected[field]
        actual_value = actual[field]
        if field in FAMILY_METRIC_FIELDS:
            expected_numeric = float(expected_value)
            if not np.isfinite(expected_numeric):
                _require(actual_value is None or not np.isfinite(float(actual_value)), f"{label}: {field} mismatch")
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
    persisted_without_hash = dict(persisted)
    persisted_without_hash.pop("content_sha256")
    _require_summary_equal(
        persisted_without_hash,
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
) -> None:
    persisted = _json_from_snapshot(
        snapshot,
        path=fold_path / "outer_reference_access_audit.json",
        self_hashed=True,
    )
    expected = {
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
    }
    expected_hashed = runner._manifest_with_self_hash(expected)
    _require(
        persisted == expected_hashed,
        "outer reference access audit differs from the fresh label gate",
    )


def _authenticate_fold(
    plan: runner.Plan,
    fold_path: Path,
    *,
    device: str,
    expected_fold_commit: str,
) -> AuthenticatedFold:
    fold_path = fold_path.resolve()
    _require(fold_path.is_dir(), f"fold directory does not exist: {fold_path}")
    observed_names = {path.name for path in fold_path.iterdir()}
    _require(
        observed_names == set(EXPECTED_FOLD_FILES)
        and all((fold_path / name).is_file() for name in EXPECTED_FOLD_FILES),
        f"{fold_path}: completed fold must contain exactly the frozen 15 files",
    )

    completion_snapshot = _read_file_snapshot(fold_path / "RUN_COMPLETE.json")
    completion = _json_from_snapshot(
        completion_snapshot,
        path=fold_path / "RUN_COMPLETE.json",
        self_hashed=True,
    )
    completion_file_sha = completion_snapshot.sha256
    _require(set(completion) == COMPLETION_FIELDS, f"{fold_path}: completion fields drifted")
    _require(completion.get("schema") == runner.COMPLETE_SCHEMA, f"{fold_path}: completion schema drifted")
    _require(completion.get("experiment") == runner.EXPERIMENT, f"{fold_path}: completion experiment drifted")
    _require(completion.get("config_sha256") == plan.sha256, f"{fold_path}: completion config drifted")
    _require(completion.get("result_manifest_file") == "result_manifest.json", f"{fold_path}: result path drifted")
    _require(_is_lower_hex(completion.get("git_commit"), 40), f"{fold_path}: completion Git commit is invalid")
    _require(
        completion.get("git_commit") == expected_fold_commit,
        f"{fold_path}: fold Git commit differs from the explicit expected commit",
    )
    family = str(completion.get("outer_family"))
    _require(family in plan.family_order, f"{fold_path}: unknown outer family")
    _, input_identity = runner.load_cache_rows(plan)
    fit_families = [value for value in plan.family_order if value != family]

    # Freeze only label-free bytes before replay.  In particular, neither the
    # result manifest nor any outer metric/summary/audit artifact is opened.
    snapshots: dict[str, _FileSnapshot] = {
        name: _read_file_snapshot(fold_path / name)
        for name in LABEL_FREE_PRE_RESULT_FILES
    }
    with tempfile.TemporaryDirectory(prefix="ptm_early_fold_auth_") as temporary:
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
        candidate = _candidate_from_payload(plan, selected_payload.get("candidate"))

        # This call first rebuilds scaler, calibration and selected candidate,
        # then fresh-loads all outer sidecars without labels, and authenticates
        # a recomputed prediction.  Only after that replay closes may it open
        # outer labels to produce fresh_rows/reference_rows.
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

        # The fresh prediction replay is complete.  It is now legal to open
        # the performance-bearing result and outer artifacts.
        result_snapshot = _read_file_snapshot(fold_path / "result_manifest.json")
        _require(
            result_snapshot.sha256
            == str(completion.get("result_manifest_file_sha256")),
            f"{fold_path}: result manifest file SHA-256 mismatch",
        )
        result = _json_from_snapshot(
            result_snapshot,
            path=fold_path / "result_manifest.json",
            self_hashed=True,
        )
        result_file_sha = result_snapshot.sha256
        _require(set(result) == RESULT_FIELDS, f"{fold_path}: result fields drifted")
        _require(result.get("schema") == runner.RESULT_SCHEMA, f"{fold_path}: result schema drifted")
        _require(result.get("experiment") == runner.EXPERIMENT, f"{fold_path}: result experiment drifted")
        _require(result.get("status") == "completed", f"{fold_path}: result is incomplete")
        _require(result.get("config_sha256") == plan.sha256, f"{fold_path}: result config drifted")
        _require(result.get("config_path") == str(plan.path), f"{fold_path}: result config path drifted")
        _require(result.get("git_commit") == expected_fold_commit, f"{fold_path}: result Git mismatch")
        _require(result.get("outer_family") == family, f"{fold_path}: result family mismatch")
        _require(
            completion.get("result_manifest_content_sha256")
            == result.get("content_sha256"),
            f"{fold_path}: result content hash binding drifted",
        )
        _require(
            result.get("input_manifest") == runner._json_safe(input_identity),
            f"{fold_path}: result input manifest identity drifted",
        )
        environment = result.get("environment")
        _require(isinstance(environment, Mapping), f"{fold_path}: environment audit is invalid")
        _require(
            environment.get("requested_device") == device,
            f"{fold_path}: aggregation device differs from numerical fold device",
        )
        _require(
            result.get("selected_candidate")
            == runner._json_safe(runner._candidate_payload(candidate)),
            f"{fold_path}: result/selection candidate drifted",
        )
        _require(
            result.get("early_evidence")
            == runner._early_artifact_binding(
                plan,
                representation=candidate.representation,
                fit_families=fit_families,
            ),
            f"{fold_path}: result Early evidence binding drifted",
        )
        _require(
            result.get("selected_candidate_file") == "selected_candidate.json",
            f"{fold_path}: selected-candidate path drifted",
        )

        for name in (
            "outer_group_metrics.csv",
            "outer_summary.json",
            "outer_reference_access_audit.json",
        ):
            snapshots[name] = _read_file_snapshot(fold_path / name)
        identities = _artifact_identities(fold_path, result, snapshots)
        _require(
            snapshots["final_per_scale_scaler.npz"].sha256
            == result["final_scaler_file_sha256"],
            f"{fold_path}: scaler file SHA drifted",
        )
        _require(
            snapshots["final_tail_calibration.npz"].sha256
            == result["final_calibration_file_sha256"],
            f"{fold_path}: calibration file SHA drifted",
        )
        _require(
            snapshots["outer_predictions.npz"].sha256
            == result["prediction_file_sha256"],
            f"{fold_path}: prediction file SHA drifted",
        )
        _require(
            selected_payload.get("content_sha256")
            == result["selected_candidate_content_sha256"],
            f"{fold_path}: selected-candidate content hash drifted",
        )

        parsed_rows = _authenticate_outer_metric_csv(
            fold_path / "outer_group_metrics.csv",
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
        )
        _authenticate_reference_audit(
            fold_path,
            result=result,
            expected_rows=reference_rows,
            snapshot=snapshots["outer_reference_access_audit.json"],
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
        numerical_git_commit=str(result["git_commit"]),
        config_sha256=str(result["config_sha256"]),
        input_manifest_sha256=str(input_identity["sha256"]),
        input_manifest_rows_sha256=str(input_identity["rows_content_sha256"]),
        requested_device=device,
        selected_candidate=runner._deep_freeze(dict(result["selected_candidate"])),
        summary=runner._deep_freeze(summary),
        artifact_identities=runner._deep_freeze(identities),
        completion_file_sha256=completion_file_sha,
        completion_content_sha256=str(completion["content_sha256"]),
        result_manifest_file_sha256=result_file_sha,
        result_manifest_content_sha256=str(result["content_sha256"]),
    )


def _stop_rule_thresholds(plan: runner.Plan) -> dict[str, Any]:
    specification = plan.raw.get("success_stop_rule")
    _require(isinstance(specification, Mapping), "success stop rule is missing")
    expected_keys = {
        "outer_family_macro_f1_minimum",
        "outer_families_at_or_above_0_65_minimum",
        "minimum_single_outer_family_f1",
        "outer_family_macro_average_precision_minimum",
        "outer_family_macro_balanced_accuracy_minimum",
        "outer_family_macro_precision_minimum",
        "outer_family_macro_recall_minimum",
        "threshold_changes_after_outer_results",
        "success_requires_complete_five_unique_outer_families",
        "single_fold_success_claim",
        "authenticated_early_stop_conditions",
        "early_stop_certificate_schema",
    }
    _require(set(specification) == expected_keys, "success stop rule fields drifted")
    _require(
        specification.get("threshold_changes_after_outer_results") == "forbidden",
        "post-outer threshold change rule drifted",
    )
    _require(
        specification.get("success_requires_complete_five_unique_outer_families")
        is True
        and specification.get("single_fold_success_claim") == "forbidden"
        and specification.get("early_stop_certificate_schema")
        == EARLY_STOP_CERTIFICATE_SCHEMA,
        "success/early-stop identity drifted",
    )
    return {
        "outer_family_macro_f1_minimum": float(
            specification["outer_family_macro_f1_minimum"]
        ),
        "outer_families_at_or_above_0_65_minimum": int(
            specification["outer_families_at_or_above_0_65_minimum"]
        ),
        "minimum_single_outer_family_f1": float(
            specification["minimum_single_outer_family_f1"]
        ),
        "outer_family_macro_average_precision_minimum": float(
            specification["outer_family_macro_average_precision_minimum"]
        ),
        "outer_family_macro_balanced_accuracy_minimum": float(
            specification["outer_family_macro_balanced_accuracy_minimum"]
        ),
        "outer_family_macro_precision_minimum": float(
            specification["outer_family_macro_precision_minimum"]
        ),
        "outer_family_macro_recall_minimum": float(
            specification["outer_family_macro_recall_minimum"]
        ),
    }


def _stop_rule(
    plan: runner.Plan,
    family_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    thresholds = _stop_rule_thresholds(plan)
    macro = {
        field: float(np.mean([float(row[field]) for row in family_rows], dtype=np.float64))
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
        "outer_family_macro_f1_minimum_pass": (
            macro["f1"] >= thresholds["outer_family_macro_f1_minimum"]
        ),
        "outer_families_at_or_above_0_65_minimum_pass": (
            sum(float(row["f1"]) >= 0.65 for row in family_rows)
            >= thresholds["outer_families_at_or_above_0_65_minimum"]
        ),
        "minimum_single_outer_family_f1_pass": (
            min(float(row["f1"]) for row in family_rows)
            >= thresholds["minimum_single_outer_family_f1"]
        ),
        "outer_family_macro_average_precision_minimum_pass": (
            macro["average_precision"]
            >= thresholds["outer_family_macro_average_precision_minimum"]
        ),
        "outer_family_macro_balanced_accuracy_minimum_pass": (
            macro["balanced_accuracy"]
            >= thresholds["outer_family_macro_balanced_accuracy_minimum"]
        ),
        "outer_family_macro_precision_minimum_pass": (
            macro["precision"]
            >= thresholds["outer_family_macro_precision_minimum"]
        ),
        "outer_family_macro_recall_minimum_pass": (
            macro["recall"]
            >= thresholds["outer_family_macro_recall_minimum"]
        ),
    }
    return {"thresholds": thresholds, "family_macro": macro}, outcomes


def _early_stop_certificate(
    plan: runner.Plan,
    family_rows: Sequence[Mapping[str, Any]],
    *,
    numerical_git_commit: str,
) -> dict[str, Any]:
    """Prove only mathematical impossibility; never infer five-fold success."""

    _require(len(family_rows) == 1, "single-fold certificate requires one fold")
    thresholds = _stop_rule_thresholds(plan)
    total_family_count = len(plan.family_order)
    observed_count = len(family_rows)
    remaining_count = total_family_count - observed_count
    observed_f1 = [float(row["f1"]) for row in family_rows]
    minimum_f1_impossible = (
        min(observed_f1) < thresholds["minimum_single_outer_family_f1"]
    )
    below_point_65_count = sum(value < 0.65 for value in observed_f1)
    point_65_count_impossible = (
        below_point_65_count
        > total_family_count
        - thresholds["outer_families_at_or_above_0_65_minimum"]
    )
    macro_thresholds = {
        "f1": thresholds["outer_family_macro_f1_minimum"],
        "average_precision": thresholds[
            "outer_family_macro_average_precision_minimum"
        ],
        "balanced_accuracy": thresholds[
            "outer_family_macro_balanced_accuracy_minimum"
        ],
        "precision": thresholds["outer_family_macro_precision_minimum"],
        "recall": thresholds["outer_family_macro_recall_minimum"],
    }
    macro_upper_bounds: dict[str, Any] = {}
    for field, threshold in macro_thresholds.items():
        values = [float(row[field]) for row in family_rows]
        _require(np.isfinite(values).all(), f"single-fold {field} must be finite")
        upper_bound = (sum(values) + remaining_count * 1.0) / total_family_count
        macro_upper_bounds[field] = {
            "observed_sum": float(sum(values)),
            "remaining_family_count": remaining_count,
            "unobserved_metric_upper_bound": 1.0,
            "best_possible_five_family_macro": float(upper_bound),
            "required_minimum": float(threshold),
            "mathematically_impossible": bool(upper_bound < threshold),
        }
    impossible_reasons: list[str] = []
    if minimum_f1_impossible:
        impossible_reasons.append("observed_family_f1_below_frozen_minimum")
    if point_65_count_impossible:
        impossible_reasons.append(
            "too_many_observed_families_below_0.65_for_four_of_five_rule"
        )
    impossible_reasons.extend(
        f"best_possible_{field}_macro_below_frozen_minimum"
        for field, record in macro_upper_bounds.items()
        if record["mathematically_impossible"]
    )
    mathematically_impossible = bool(impossible_reasons)
    return {
        "schema": EARLY_STOP_CERTIFICATE_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": "evaluated",
        "config_sha256": plan.sha256,
        "fold_numerical_git_commit": numerical_git_commit,
        "observed_outer_families": [str(row["outer_family"]) for row in family_rows],
        "observed_outer_family_count": observed_count,
        "frozen_total_outer_family_count": total_family_count,
        "remaining_outer_family_count": remaining_count,
        "thresholds": thresholds,
        "any_observed_family_f1_below_minimum": minimum_f1_impossible,
        "at_least_two_observed_families_f1_below_0_65": point_65_count_impossible,
        "macro_upper_bound_proofs": macro_upper_bounds,
        "mathematically_impossible_to_pass": mathematically_impossible,
        "impossibility_reasons": impossible_reasons,
        "stop_version": mathematically_impossible,
        "five_fold_success_evaluated": False,
        "five_fold_success": None,
    }


def _authenticate_published_output(
    path: Path,
    *,
    size_bytes: int,
    sha256: str,
    self_hashed_json: bool,
) -> None:
    runner._stable_file_identity(path, size_bytes, sha256)
    if self_hashed_json:
        _load_self_hashed_json(path, expected_file_sha256=sha256)


def _aggregate_early_binding(plan: runner.Plan) -> dict[str, Any]:
    runner._require_early_evidence_bound(plan)
    assert plan.sidecar_population is not None
    producer_commit = str(plan.sidecar_population["git_commit"])
    return {
        "kinematic_input_manifest": {
            "path": str(plan.kinematic_input_manifest_path),
            "file_sha256": plan.kinematic_input_manifest_file_sha256,
            "content_sha256": plan.kinematic_input_manifest_content_sha256,
            "producer_git_commit": producer_commit,
        },
        "synthetic_pass": {
            "path": str(plan.synthetic_pass_path),
            "file_sha256": plan.synthetic_pass_file_sha256,
            "producer_git_commit": producer_commit,
        },
        "sidecar_population_manifest": {
            "path": str(plan.sidecar_population_manifest_path),
            "file_sha256": plan.sidecar_population_manifest_file_sha256,
            "content_sha256": plan.sidecar_population_manifest_content_sha256,
            "sidecar_count": 32,
            "producer_git_commit": producer_commit,
        },
        "composite_descriptor_ids": dict(plan.composite_descriptor_ids),
        "clean_git_commit": (
            plan.source_identity.git_commit if plan.source_identity else None
        ),
        "config_sha256": plan.sha256,
    }


def _require_preparation_release_binding(
    *,
    early: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    synthetic_marker: Mapping[str, Any],
    population_manifest: Mapping[str, Any],
    current_fold_commit: str,
) -> None:
    """Bind sealed preparation evidence to its producer and a new fold commit."""

    _require(
        _is_lower_hex(current_fold_commit, 40),
        "current fold commit is invalid",
    )
    _require(
        early.get("clean_git_commit") == current_fold_commit,
        "Early evidence is not bound to the current fold commit",
    )
    producer_commit = runner.PREPARATION_ARTIFACT_GIT_COMMIT
    for name in (
        "kinematic_input_manifest",
        "synthetic_pass",
        "sidecar_population_manifest",
    ):
        record = early.get(name)
        _require(isinstance(record, Mapping), f"Early {name} binding is invalid")
        _require(
            record.get("producer_git_commit") == producer_commit,
            f"Early {name} producer commit drifted",
        )

    descriptors = input_manifest.get("composite_descriptors")
    _require(
        isinstance(descriptors, Mapping)
        and set(descriptors) == set(runner.REPRESENTATIONS),
        "input composite descriptor population changed",
    )
    input_descriptor_ids: dict[str, str] = {}
    for name in runner.REPRESENTATIONS:
        descriptor = descriptors[name]
        _require(
            isinstance(descriptor, Mapping)
            and descriptor.get("composite_representation") == name,
            f"input composite descriptor contract is invalid: {name}",
        )
        descriptor_id = descriptor.get("descriptor_id")
        _require(
            isinstance(descriptor_id, str) and descriptor_id,
            f"input composite descriptor ID is invalid: {name}",
        )
        input_descriptor_ids[name] = descriptor_id

    early_descriptor_ids = early.get("composite_descriptor_ids")
    synthetic_descriptor_ids = synthetic_marker.get("composite_descriptor_ids")
    population_descriptor_ids = population_manifest.get("composite_descriptor_ids")
    for observed, source in (
        (early_descriptor_ids, "Early aggregate evidence"),
        (synthetic_descriptor_ids, "synthetic marker"),
        (population_descriptor_ids, "sidecar population"),
    ):
        _require(isinstance(observed, Mapping), f"{source} descriptor IDs are invalid")
        _require(
            dict(observed) == input_descriptor_ids,
            f"{source} descriptor IDs drifted",
        )

    for artifact, source in (
        (input_manifest, "input manifest"),
        (synthetic_marker, "synthetic marker"),
        (population_manifest, "sidecar population"),
    ):
        _require(
            artifact.get("git_commit") == producer_commit,
            f"{source} producer commit drifted",
        )


def _require_result_input_manifest_binding(
    result: Mapping[str, Any],
    plan: runner.Plan,
) -> None:
    """Authenticate the parent-cache input identity in a fold result."""

    observed = result.get("input_manifest")
    _require(isinstance(observed, Mapping), "result input manifest is invalid")
    _require(
        set(observed)
        == {"schema", "path", "size_bytes", "sha256", "rows_content_sha256"},
        "result input manifest schema drifted",
    )
    _require(observed.get("schema") == plan.manifest_schema, "result input schema drifted")
    _require(
        Path(str(observed.get("path"))).resolve() == plan.manifest_path.resolve(),
        "result input path drifted",
    )
    _require(observed.get("size_bytes") == plan.manifest_size, "result input size drifted")
    _require(observed.get("sha256") == plan.manifest_sha256, "result input SHA-256 drifted")
    _require(
        observed.get("rows_content_sha256") == plan.manifest_rows_sha256,
        "result input rows SHA-256 drifted",
    )


def aggregate(
    config_path: str | Path,
    run_directories: Sequence[str | Path],
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
    mode: str = "auto",
    device: str = "cpu",
    expected_config_sha256: str | None = runner.EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    plan = runner.load_plan(config_path)
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
    _require(mode in {"auto", "single-fold", "complete-five-fold"}, "invalid aggregation mode")
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
        aggregator_commit == expected_fold_commit
        and plan.source_identity is not None
        and plan.source_identity.git_commit == aggregator_commit,
        "aggregate and folds must use one exact commit; preparation evidence is separately pinned",
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
        observed_families = [fold.outer_family for fold in folds]
        _require(
            len(set(observed_families)) == 5
            and set(observed_families) == set(plan.family_order),
            "complete aggregation requires each frozen outer family exactly once",
        )

    _require(
        {fold.numerical_git_commit for fold in folds}
        == {expected_fold_commit},
        "fold numerical Git commit differs from the explicit expected commit",
    )
    _require(
        {fold.config_sha256 for fold in folds} == {plan.sha256},
        "folds mix frozen configs",
    )
    _require(
        len(
            {
                (fold.input_manifest_sha256, fold.input_manifest_rows_sha256)
                for fold in folds
            }
        )
        == 1,
        "folds mix input manifests",
    )
    numerical_commit = folds[0].numerical_git_commit
    by_family = {fold.outer_family: fold for fold in folds}
    ordered_folds = tuple(
        by_family[family] for family in plan.family_order if family in by_family
    )

    family_rows: list[dict[str, Any]] = []
    for fold in ordered_folds:
        summary = fold.summary
        row: dict[str, Any] = {
            "outer_family": fold.outer_family,
            "run_directory": str(fold.path),
            "numerical_git_commit": fold.numerical_git_commit,
            "config_sha256": fold.config_sha256,
            "input_manifest_sha256": fold.input_manifest_sha256,
            "input_manifest_rows_sha256": fold.input_manifest_rows_sha256,
            "requested_device": fold.requested_device,
            "selected_candidate_id": fold.selected_candidate["candidate_id"],
        }
        row.update({field: summary[field] for field in FAMILY_METRIC_FIELDS})
        row.update({field: int(summary[field]) for field in FAMILY_COUNT_FIELDS})
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
        stop_inputs, stop_outcomes = _stop_rule(plan, family_rows)
        family_macro: Mapping[str, Any] | None = stop_inputs["family_macro"]
        stop_rule: Mapping[str, Any] = {
            "evaluated": True,
            **stop_inputs,
            "outcomes": stop_outcomes,
            "all_success_conditions_pass": all(stop_outcomes.values()),
        }
        all_success: bool | None = all(stop_outcomes.values())
    else:
        family_macro = None
        early_stop_payload = _early_stop_certificate(
            plan,
            family_rows,
            numerical_git_commit=numerical_commit,
        )
        stop_rule = {
            "evaluated": False,
            "reason": "complete_five_outer_family_set_required",
        }
        all_success = None

    destination.mkdir(parents=True, exist_ok=False)
    table_path = destination / "outer_family_summary.csv"
    table_sha = runner._atomic_csv(
        table_path,
        FAMILY_SUMMARY_FIELDS,
        family_rows,
    )
    if effective_mode == "complete_five_fold_aggregate":
        report_name = "aggregate_summary.json"
        report_payload = {
            "schema": AGGREGATE_SUMMARY_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": effective_mode,
            "config_sha256": plan.sha256,
            "early_evidence": _aggregate_early_binding(plan),
            "aggregator_git_commit": aggregator_commit,
            "aggregator_worktree_clean": True,
            "fold_numerical_git_commit": numerical_commit,
            "input_manifest_sha256": folds[0].input_manifest_sha256,
            "input_manifest_rows_sha256": folds[0].input_manifest_rows_sha256,
            "outer_families": [fold.outer_family for fold in ordered_folds],
            "outer_family_count": len(ordered_folds),
            "aggregation": (
                "equal_outer_physical_family_macro"
                if effective_mode == "complete_five_fold_aggregate"
                else "single_fold_authentication_only"
            ),
            "fold_summary_source": (
                "fresh_scaler_and_calibrator_query_then_label_gate_exact_CSV_comparison_"
                "and_hierarchical_recomputation"
            ),
            "family_macro": family_macro,
            "success_stop_rule": stop_rule,
            "all_success_conditions_pass": all_success,
            "outer_family_summary_file_sha256": table_sha,
            "folds": family_rows,
            "formal_confirmation": False,
            "evidence_scope": "exposed_train_only_nested_family_validation",
        }
        early_stop_record = None
    else:
        assert early_stop_payload is not None
        early_stop_path = destination / "early_stop_certificate.json"
        early_stop_manifest = runner._manifest_with_self_hash(early_stop_payload)
        early_stop_sha = runner._atomic_json(early_stop_path, early_stop_manifest)
        early_stop_record = {
            "path": early_stop_path.name,
            "size_bytes": early_stop_path.stat().st_size,
            "sha256": early_stop_sha,
            "content_sha256": early_stop_manifest["content_sha256"],
        }
        report_name = "single_fold_authentication_report.json"
        report_payload = {
            "schema": SINGLE_FOLD_REPORT_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": effective_mode,
            "config_sha256": plan.sha256,
            "early_evidence": _aggregate_early_binding(plan),
            "aggregator_git_commit": aggregator_commit,
            "aggregator_worktree_clean": True,
            "fold_numerical_git_commit": numerical_commit,
            "input_manifest_sha256": folds[0].input_manifest_sha256,
            "input_manifest_rows_sha256": folds[0].input_manifest_rows_sha256,
            "outer_family": folds[0].outer_family,
            "fold_summary_source": (
                "fresh_scaler_and_calibrator_query_then_label_gate_exact_CSV_comparison_"
                "and_hierarchical_recomputation"
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
    report = runner._manifest_with_self_hash(report_payload)
    report_path = destination / report_name
    report_sha = runner._atomic_json(report_path, report)
    manifest = runner._manifest_with_self_hash(
        {
            "schema": AGGREGATE_MANIFEST_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": effective_mode,
            "config_sha256": plan.sha256,
            "early_evidence": _aggregate_early_binding(plan),
            "aggregator_git_commit": aggregator_commit,
            "aggregator_worktree_clean": True,
            "fold_numerical_git_commit": numerical_commit,
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
            "size_bytes": int(early_stop_record["size_bytes"]),
            "sha256": str(early_stop_record["sha256"]),
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

    completion = runner._manifest_with_self_hash(
        {
            "schema": AGGREGATE_COMPLETE_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": effective_mode,
            "config_sha256": plan.sha256,
            "early_evidence": _aggregate_early_binding(plan),
            "aggregator_git_commit": aggregator_commit,
            "aggregator_worktree_clean": True,
            "fold_numerical_git_commit": numerical_commit,
            "aggregate_manifest_file": manifest_path.name,
            "aggregate_manifest_file_sha256": manifest_sha,
            "report_file": report_path.name,
            "report_file_sha256": report_sha,
            "early_stop_certificate": early_stop_record,
            "completed_utc": runner._utc_now(),
        }
    )
    completion_path = destination / "AGGREGATE_COMPLETE.json"
    completion_sha = runner._atomic_json(completion_path, completion)
    try:
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
    except Exception:
        # Preserve the failed publication exactly as written.  Consumers must
        # authenticate the expected SHA/self-hash and complete file set, so a
        # tampered or partial marker fails closed without destructive cleanup.
        raise
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "Verify_EarlyOppositePairKinematics_1.1.yaml"),
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=("auto", "single-fold", "complete-five-fold"),
        default="auto",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-fold-commit", required=True)
    parser.add_argument("--kinematic-input-manifest", required=True)
    parser.add_argument("--kinematic-input-manifest-sha256", required=True)
    parser.add_argument("--synthetic-pass", required=True)
    parser.add_argument("--synthetic-pass-sha256", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--sidecar-population-manifest", required=True)
    parser.add_argument("--sidecar-population-manifest-sha256", required=True)
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
        mode=arguments.mode,
        device=arguments.device,
        expected_config_sha256=arguments.expected_config_sha256,
    )
    print(json.dumps(runner._json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

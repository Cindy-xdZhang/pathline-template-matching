#!/usr/bin/env python3
"""Frozen Raw672-PCA161 nested runner for Verify_RawPCANegativeMetric_1.1.

This runner is intentionally independent from the parent's orchestration.  It
reuses the parent's frozen numerical scoring helpers, but owns every Raw cache
member access, Principal Component Analysis fit, artifact binding, prediction
replay, and label gate used by this experiment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence
from typing import BinaryIO, Iterator

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.per_scale_negative_metric import (  # noqa: E402
    SCALER_ARRAY_NAMES,
    PerScaleNegativeScaler,
    PerScaleNegativeTailModel,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    sha256_file,
)
from pathline_template_matching.raw_pca_representation import (  # noqa: E402
    RAW_INPUT_WIDTH,
    RAW_OUTPUT_WIDTH,
    RAW_PCA_ARRAY_NAMES,
    RAW_PCA_ROW_CHUNK_SIZE,
    RAW_PCA_SCHEMA,
    RAW_PCA_SOLVER,
    RawPCAArrayAudit,
    RawPCARepresentation,
    deserialize_raw_pca,
    fit_raw_pca,
    serialize_raw_pca,
)
from scripts import run_verify_per_scale_negative_metric_1_1 as inherited  # noqa: E402
from scripts.run_verify_scale_conditioned_retrieval_1_1 import (  # noqa: E402
    CacheProjection,
    CacheRow,
    _configure_execution,
    _git_identity,
    _json_safe,
    _json_safe_content_sha256,
    _require,
    _utc_now,
)


EXPERIMENT = "Verify_RawPCANegativeMetric_1.1"
EXPECTED_CONFIG_SHA256 = (
    "6f4718ce6d6385bd0bd5b41a7a04e74cb8f2064fee64097f162999e9eefe6440"
)
FAMILY_ORDER = (
    "half_cylinder",
    "delta_wing",
    "f22_raptor",
    "channel",
    "boeing_747",
)
REPRESENTATIONS = ("raw_pca161",)
K_VALUES = (1, 5, 15, 31)
SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
TAIL_THRESHOLDS = tuple(round(0.50 + 0.01 * index, 2) for index in range(50))
GRID_SHAPE = (40, 40, 40)
BLOCK_NAMES = ("legacy_2_1", "expanded_3_1")
FROZEN_CANDIDATE_COUNT = 1020

PCA_MANIFEST_SCHEMA = "pathline_template_matching.raw_pca161_manifest.v1"
SCALER_ARTIFACT_SCHEMA = "pathline_template_matching.per_scale_negative_metric.v1"
SCALER_MANIFEST_SCHEMA = (
    "pathline_template_matching.raw_pca_negative_metric_scaler_manifest.v1"
)
CALIBRATION_ARTIFACT_SCHEMA = (
    "pathline_template_matching.per_scale_negative_tail_calibration_artifact.v1"
)
CALIBRATION_MANIFEST_SCHEMA = (
    "pathline_template_matching.raw_pca_negative_metric_calibration_manifest.v1"
)
SELECTED_SCHEMA = (
    "pathline_template_matching.raw_pca_negative_metric_selected_candidate.v1"
)
PREDICTION_SCHEMA = (
    "pathline_template_matching.raw_pca_negative_metric_outer_prediction.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "pathline_template_matching.raw_pca_negative_metric_outer_prediction_manifest.v1"
)
RESULT_SCHEMA = "pathline_template_matching.raw_pca_negative_metric_result.v1"
COMPLETE_SCHEMA = "pathline_template_matching.raw_pca_negative_metric_run_complete.v1"
INNER_AUDIT_SCHEMA = (
    "pathline_template_matching.raw_pca_negative_metric_inner_fit_audits.v1"
)
REFERENCE_AUDIT_SCHEMA = (
    "pathline_template_matching.raw_pca_negative_metric_outer_reference_access.v1"
)


def _fsync_parent_directory(path: Path) -> None:
    """Persist a same-directory publication on POSIX after a hard-link commit."""

    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_verified_temporary(temporary: Path, destination: Path) -> None:
    """Publish an authenticated temporary without ever replacing a winner."""

    try:
        os.link(temporary, destination)
        _fsync_parent_directory(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        _fsync_parent_directory(destination)


def _new_same_directory_temporary(destination: Path) -> tuple[int, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
    )
    return descriptor, Path(raw_path)


def _atomic_bytes(path: Path, payload: bytes) -> str:
    """Fsync, authenticate, and hard-link bytes with no-replace semantics."""

    destination = Path(path)
    descriptor, temporary = _new_same_directory_temporary(destination)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        expected_sha = hashlib.sha256(payload).hexdigest()
        before = temporary.stat()
        actual_sha = sha256_file(temporary)
        after = temporary.stat()
        _require(
            (before.st_size, before.st_mtime_ns)
            == (after.st_size, after.st_mtime_ns),
            "temporary artifact changed while authenticating",
        )
        _require(after.st_size == len(payload), "temporary artifact size drifted")
        _require(actual_sha == expected_sha, "temporary artifact SHA-256 drifted")
        _publish_verified_temporary(temporary, destination)
        return expected_sha
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> str:
    safe_value = _json_safe(value)
    payload = json.dumps(
        safe_value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    parsed = json.loads(payload.decode("utf-8"))
    _require(parsed == safe_value, "JSON serialization failed its exact replay gate")
    return _atomic_bytes(path, payload)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return int(bool(value))
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return "" if not np.isfinite(numeric) else format(numeric, ".12g")
    if value is None:
        return ""
    return value


def _atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> str:
    names = tuple(str(name) for name in fieldnames)
    normalized = [
        {name: str(_csv_value(row.get(name))) for name in names} for row in rows
    ]
    text_stream = io.StringIO(newline="")
    writer = csv.DictWriter(text_stream, fieldnames=list(names), extrasaction="raise")
    writer.writeheader()
    writer.writerows(normalized)
    payload = text_stream.getvalue().encode("utf-8")
    replay = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    _require(replay == normalized, "CSV serialization failed its exact replay gate")
    return _atomic_bytes(path, payload)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Stream an NPZ, replay every member, then publish by no-replace link."""

    destination = Path(path)
    expected = {str(name): np.asarray(values) for name, values in arrays.items()}
    descriptor, temporary = _new_same_directory_temporary(destination)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **expected)
            stream.flush()
            os.fsync(stream.fileno())
        before = temporary.stat()
        with np.load(temporary, allow_pickle=False) as archive:
            _require(set(archive.files) == set(expected), "temporary NPZ member set drifted")
            for name, source in expected.items():
                replay = np.asarray(archive[name])
                _require(replay.dtype == source.dtype, f"temporary NPZ dtype drifted: {name}")
                _require(replay.shape == source.shape, f"temporary NPZ shape drifted: {name}")
                _require(np.array_equal(replay, source), f"temporary NPZ content drifted: {name}")
        digest = sha256_file(temporary)
        after = temporary.stat()
        _require(
            (before.st_size, before.st_mtime_ns)
            == (after.st_size, after.st_mtime_ns),
            "temporary NPZ changed while authenticating",
        )
        _publish_verified_temporary(temporary, destination)
        return digest
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class _OpenFileIdentity:
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_OpenFileIdentity":
        return cls(
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
            device=int(value.st_dev),
            inode=int(value.st_ino),
        )


@dataclass(frozen=True)
class _AuthenticatedOpenFile:
    stream: BinaryIO
    size_bytes: int
    sha256: str


@contextmanager
def _authenticated_open_file(
    path: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> Iterator[_AuthenticatedOpenFile]:
    """Hash and consume one descriptor while binding it to the path inode."""

    source = Path(path)
    path_before = _OpenFileIdentity.from_stat(source.stat())
    stream = source.open("rb")
    try:
        descriptor_before = _OpenFileIdentity.from_stat(os.fstat(stream.fileno()))
        _require(
            descriptor_before == path_before,
            f"file path changed before descriptor open: {source}",
        )
        digest = hashlib.sha256()
        byte_count = 0
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            byte_count += len(block)
        descriptor_after_hash = _OpenFileIdentity.from_stat(os.fstat(stream.fileno()))
        _require(
            descriptor_after_hash == descriptor_before,
            f"file descriptor changed while hashing: {source}",
        )
        actual_sha = digest.hexdigest()
        _require(
            byte_count == descriptor_before.size_bytes,
            f"file byte count changed while hashing: {source}",
        )
        if expected_size is not None:
            _require(
                descriptor_before.size_bytes == int(expected_size),
                f"file size mismatch: {source}",
            )
        if expected_sha256 is not None:
            _require(
                _lower_hex(expected_sha256) and actual_sha == str(expected_sha256),
                f"file SHA-256 mismatch: {source}",
            )
        stream.seek(0)
        yield _AuthenticatedOpenFile(
            stream=stream,
            size_bytes=descriptor_before.size_bytes,
            sha256=actual_sha,
        )
        descriptor_after_read = _OpenFileIdentity.from_stat(os.fstat(stream.fileno()))
        path_after = _OpenFileIdentity.from_stat(source.stat())
        _require(
            descriptor_after_read == descriptor_before and path_after == path_before,
            f"file path or descriptor changed while loading: {source}",
        )
    finally:
        stream.close()


def _read_authenticated_bytes(
    path: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    with _authenticated_open_file(
        path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    ) as opened:
        payload = opened.stream.read()
        identity = {
            "path": str(path),
            "size_bytes": opened.size_bytes,
            "sha256": opened.sha256,
        }
    return payload, identity

REQUIRED_FOLD_FILES = (
    "inner_group_metrics.csv",
    "inner_candidate_summary.csv",
    "inner_fit_audits.json",
    "final_pca.npz",
    "final_pca_manifest.json",
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

PREDICTION_ARRAY_DTYPES: Mapping[str, np.dtype[Any]] = MappingProxyType(
    dict(inherited.PREDICTION_ARRAY_DTYPES)
)
METRIC_FIELDS = inherited.METRIC_FIELDS
SUMMARY_FIELDS = inherited.SUMMARY_FIELDS
TailCandidateSpec = inherited.TailCandidateSpec
SpatialTailScores = inherited.SpatialTailScores
spatial_calibrated_tail_scores = inherited.spatial_calibrated_tail_scores
candidate_predictions = inherited.candidate_predictions
_metric_row = inherited._metric_row
_threshold_metric_rows = inherited._threshold_metric_rows
_candidate_payload = inherited._candidate_payload
_array_manifest = inherited._array_manifest


def _manifest_with_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(_json_safe(payload))
    _require("content_sha256" not in output, "manifest already contains a self hash")
    output["content_sha256"] = _json_safe_content_sha256(output)
    return output


def _authenticate_self_hash(manifest: Mapping[str, Any]) -> None:
    content = dict(manifest)
    stored = content.pop("content_sha256", None)
    _require(_lower_hex(stored), "manifest self hash is missing")
    _require(
        stored == _json_safe_content_sha256(content),
        "manifest self hash mismatch",
    )


def _lower_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        result = np.frombuffer(
            np.ascontiguousarray(value).tobytes(order="C"), dtype=value.dtype
        ).reshape(value.shape)
        result.setflags(write=False)
        return result
    return value


@dataclass(frozen=True)
class Plan:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    family_order: tuple[str, ...]
    families: Mapping[str, tuple[str, ...]]
    dataset_to_family: Mapping[str, str]
    manifest_path: Path
    manifest_size: int
    manifest_sha256: str
    manifest_schema: str
    manifest_rows_sha256: str
    cache_commit: str
    parent_config_sha256: str
    cache_schema: str
    representations: tuple[str, ...]
    ks: tuple[int, ...]
    sigmas: tuple[float, ...]
    thresholds: tuple[float, ...]
    fixed_top_fraction: float
    grid_shape: tuple[int, int, int]
    gaussian_truncate: float
    query_chunk_size: int
    library_chunk_size: int
    shrinkage_lambda: float
    expected_sample_counts: Mapping[tuple[str, str], int]
    expected_family_rows: Mapping[str, int]
    expected_total_rows: int
    output_root: Path
    required_fold_files: tuple[str, ...]


@dataclass(frozen=True)
class RawCacheProjection:
    row: CacheRow
    raw_features: np.ndarray
    scale_ids: np.ndarray
    center_indices: np.ndarray
    block_indices: np.ndarray
    assigned_row_indices: np.ndarray
    labels: np.ndarray | None
    metadata: Mapping[str, Any]
    opened_members: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.raw_features)


@dataclass(frozen=True)
class RawPCAFitResult:
    model: RawPCARepresentation
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedPCAArtifact:
    manifest_path: Path
    manifest_file_sha256: str
    pca_file_sha256: str
    manifest: Mapping[str, Any]
    model: RawPCARepresentation
    _authentication_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedScalerArtifact:
    manifest_path: Path
    manifest_file_sha256: str
    scaler_file_sha256: str
    manifest: Mapping[str, Any]
    scaler: PerScaleNegativeScaler
    _authentication_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedCalibrationArtifact:
    manifest_path: Path
    manifest_file_sha256: str
    calibration_file_sha256: str
    manifest: Mapping[str, Any]
    model: PerScaleNegativeTailModel
    _authentication_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedSelectedCandidate:
    path: Path
    file_sha256: str
    manifest: Mapping[str, Any]
    _authentication_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedOuterPrediction:
    manifest_path: Path
    manifest_file_sha256: str
    prediction_file_sha256: str
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    replayed_outer: tuple[CacheProjection, ...]
    _authentication_seal: object = field(repr=False, compare=False)


_PCA_AUTHENTICATION_SEAL = object()
_SCALER_AUTHENTICATION_SEAL = object()
_CALIBRATION_AUTHENTICATION_SEAL = object()
_SELECTION_AUTHENTICATION_SEAL = object()
_PREDICTION_AUTHENTICATION_SEAL = object()


def load_plan(config_path: str | Path) -> Plan:
    """Load and authenticate the complete frozen Raw-PCA experiment contract."""

    path = Path(config_path).resolve()
    payload, _identity = _read_authenticated_bytes(
        path, expected_sha256=EXPECTED_CONFIG_SHA256
    )
    raw = yaml.safe_load(payload.decode("utf-8"))
    _require(isinstance(raw, Mapping), "config root must be a mapping")
    _require(raw.get("experiment") == EXPERIMENT, "experiment identity drifted")
    _require(
        raw.get("phase") == "exposed_train_only_nested_family_validation",
        "phase drifted",
    )
    _require(
        raw.get("status") == "frozen_pre_run_not_implemented",
        "frozen status drifted",
    )
    evidence = raw.get("evidence_scope")
    _require(isinstance(evidence, Mapping), "evidence scope is missing")
    _require(
        evidence.get("allowed_inputs")
        == "mainExp_TemplateMatching_3.1_train_caches_only",
        "input scope drifted",
    )
    _require(
        evidence.get("forbidden_datasets") == ["tangaroa", "smokeBuoyancy"],
        "forbidden datasets drifted",
    )

    families_raw = raw.get("families")
    split = raw.get("nested_split")
    _require(
        isinstance(families_raw, Mapping) and isinstance(split, Mapping),
        "family split is missing",
    )
    family_order = tuple(str(value) for value in split.get("outer_order", ()))
    _require(family_order == FAMILY_ORDER, "outer family order drifted")
    _require(tuple(split.get("inner_order", ())) == FAMILY_ORDER, "inner order drifted")
    _require(split.get("pca_refit_per_inner_fold") is True, "inner PCA refit drifted")
    _require(
        split.get("pca_refit_for_final_outer_model") is True,
        "final PCA refit drifted",
    )
    _require(
        split.get("outer_features_available_to_pca_or_selection") is False,
        "outer PCA gate drifted",
    )
    _require(
        split.get("outer_labels_available_to_selection") is False,
        "outer-label gate drifted",
    )
    families = {
        str(key): tuple(str(value) for value in values)
        for key, values in families_raw.items()
    }
    _require(tuple(families) == FAMILY_ORDER, "family mapping order drifted")
    datasets = [dataset for family in FAMILY_ORDER for dataset in families[family]]
    _require(
        len(datasets) == 8 and len(set(datasets)) == 8,
        "families must contain eight unique datasets",
    )
    dataset_to_family = {
        dataset: family for family in FAMILY_ORDER for dataset in families[family]
    }

    parent = raw.get("parent_identity")
    _require(isinstance(parent, Mapping), "parent identity is missing")
    _require(
        parent.get("cache_builder_git_commit")
        == "260a07ad380d64fc300cabe8926244e92d8ba04a",
        "cache commit drifted",
    )
    _require(
        parent.get("main_config_sha256")
        == "771980f14a6019a1f6e4bf03668d9f37dcf63495ae2dafa866312b12fc71855e",
        "parent config drifted",
    )
    manifest = parent.get("input_manifest")
    _require(isinstance(manifest, Mapping), "input manifest identity is missing")
    _require(int(manifest.get("row_count", -1)) == 32, "manifest row count drifted")
    _require(
        _lower_hex(manifest.get("sha256"))
        and _lower_hex(manifest.get("rows_content_sha256")),
        "input manifest hash is invalid",
    )

    raw_input = raw.get("raw_cache_input")
    pca = raw.get("principal_component_analysis")
    _require(
        isinstance(raw_input, Mapping) and isinstance(pca, Mapping),
        "Raw/PCA contract is missing",
    )
    _require(raw_input.get("archive_member") == "raw_features", "Raw member drifted")
    _require(raw_input.get("array_dtype") == "float32", "Raw dtype drifted")
    _require(raw_input.get("array_shape") == ["valid_row_count", 672], "Raw shape drifted")
    _require(int(raw_input.get("feature_width", -1)) == RAW_INPUT_WIDTH, "Raw width drifted")
    _require(int(raw_input.get("expected_train_cache_count", -1)) == 32, "cache count drifted")
    _require(pca.get("representation_id") == "raw_pca161", "PCA identity drifted")
    _require(int(pca.get("input_width", -1)) == RAW_INPUT_WIDTH, "PCA input drifted")
    _require(int(pca.get("output_width", -1)) == RAW_OUTPUT_WIDTH, "PCA output drifted")
    _require(pca.get("solver") == RAW_PCA_SOLVER, "PCA solver drifted")
    _require(pca.get("whitening") is False, "PCA whitening drifted")
    _require(
        pca.get("post_transform_global_standardization") is False,
        "post-PCA standardization drifted",
    )
    _require(
        int(pca.get("second_pass", {}).get("row_chunk_size", -1))
        == RAW_PCA_ROW_CHUNK_SIZE,
        "PCA chunk size drifted",
    )
    expected_counts: dict[tuple[str, str], int] = {}
    counts_raw = pca.get("expected_sample_counts")
    _require(isinstance(counts_raw, Mapping), "expected PCA sample counts are missing")
    for outer_key, values in counts_raw.items():
        _require(str(outer_key).startswith("outer_"), "invalid outer count key")
        outer = str(outer_key)[len("outer_") :]
        _require(outer in FAMILY_ORDER and isinstance(values, Mapping), "invalid outer counts")
        for inner_key, count in values.items():
            inner_text = str(inner_key)
            held_out = "final" if inner_text == "final" else inner_text[len("inner_") :]
            _require(
                held_out == "final" or held_out in FAMILY_ORDER,
                "invalid inner count key",
            )
            expected_counts[(outer, held_out)] = int(count)
    _require(len(expected_counts) == 25, "PCA sample-count table is incomplete")

    representations = tuple(raw.get("representations", {}).get("order", ()))
    _require(representations == REPRESENTATIONS, "representation grid drifted")
    retrieval = raw.get("retrieval")
    calibration = raw.get("negative_tail_calibration")
    transform = raw.get("group_transform")
    decisions = raw.get("decision_candidates")
    library = raw.get("library")
    _require(
        all(
            isinstance(value, Mapping)
            for value in (retrieval, calibration, transform, decisions, library)
        ),
        "numerical contract is incomplete",
    )
    assert isinstance(retrieval, Mapping)
    assert isinstance(calibration, Mapping)
    assert isinstance(transform, Mapping)
    assert isinstance(decisions, Mapping)
    assert isinstance(library, Mapping)
    ks = tuple(int(value) for value in retrieval.get("ks", ()))
    sigmas = tuple(float(value) for value in transform.get("gaussian_sigmas_grid_indices", ()))
    thresholds = tuple(
        float(value)
        for value in decisions.get("calibrated_tail_anomaly_threshold", {}).get(
            "values", ()
        )
    )
    _require(ks == K_VALUES, "k grid drifted")
    _require(sigmas == SIGMAS, "spatial sigma grid drifted")
    _require(thresholds == TAIL_THRESHOLDS, "decision threshold grid drifted")
    _require(
        int(decisions.get("frozen_candidate_count", -1)) == FROZEN_CANDIDATE_COUNT,
        "candidate count drifted",
    )
    fixed_fraction = float(decisions.get("fixed_top_fraction", {}).get("fraction", -1))
    _require(fixed_fraction == 0.05, "fixed-top fraction drifted")
    _require(
        retrieval.get("conditioning") == "exact_numeric_scale_id",
        "same-scale retrieval drifted",
    )
    _require(float(library.get("shrinkage_lambda", -1)) == 64.0, "scaler lambda drifted")
    _require(float(calibration.get("shrinkage_lambda", -1)) == 64.0, "tail lambda drifted")
    _require(transform.get("query_group_rank") == "forbidden", "rank transform is forbidden")

    serialization = raw.get("serialization")
    output = raw.get("output")
    _require(
        isinstance(serialization, Mapping) and isinstance(output, Mapping),
        "serialization/output contract is missing",
    )
    _require(serialization.get("pca_schema") == RAW_PCA_SCHEMA, "PCA schema drifted")
    _require(
        tuple(serialization.get("final_pca_required_arrays", {}))
        == RAW_PCA_ARRAY_NAMES,
        "PCA array contract drifted",
    )
    _require(
        int(serialization.get("inner_fit_audit_count_per_outer_fold", -1)) == 4,
        "inner PCA audit count drifted",
    )
    required = tuple(str(value) for value in output.get("required_fold_files", ()))
    _require(required == REQUIRED_FOLD_FILES, "required fold file contract drifted")
    _require(int(output.get("expected_fold_file_count", -1)) == 17, "fold file count drifted")
    _require(
        int(output.get("expected_result_artifact_count", -1)) == 15,
        "result artifact count drifted",
    )
    _require(output.get("overwrite") == "forbidden", "overwrite policy drifted")
    for key in (
        "final_pca_written_before_outer_feature_open",
        "final_scaler_written_before_outer_feature_open",
        "final_calibration_written_before_outer_feature_open",
        "predictions_written_before_outer_reference_open",
    ):
        _require(output.get(key) is True, f"output access gate drifted: {key}")

    expected_family_rows_raw = raw_input.get("expected_valid_rows_by_family")
    _require(isinstance(expected_family_rows_raw, Mapping), "family row counts missing")
    expected_family_rows = {
        str(key): int(value) for key, value in expected_family_rows_raw.items()
    }
    _require(tuple(expected_family_rows) == FAMILY_ORDER, "family row-count order drifted")
    expected_total_rows = int(raw_input["expected_valid_rows_total"])
    _require(
        sum(expected_family_rows.values()) == expected_total_rows,
        "family/total Raw row counts disagree",
    )
    for outer in FAMILY_ORDER:
        _require(
            expected_counts[(outer, "final")]
            == expected_total_rows - expected_family_rows[outer],
            f"final PCA count does not match family totals: {outer}",
        )
        for inner in FAMILY_ORDER:
            if inner == outer:
                continue
            _require(
                expected_counts[(outer, inner)]
                == expected_total_rows
                - expected_family_rows[outer]
                - expected_family_rows[inner],
                f"inner PCA count does not match family totals: {outer}/{inner}",
            )
    return Plan(
        path=path,
        sha256=str(_identity["sha256"]),
        raw=_deep_freeze(raw),
        family_order=family_order,
        families=MappingProxyType(families),
        dataset_to_family=MappingProxyType(dataset_to_family),
        manifest_path=Path(str(manifest["path"])),
        manifest_size=int(manifest["size_bytes"]),
        manifest_sha256=str(manifest["sha256"]),
        manifest_schema=str(manifest["schema"]),
        manifest_rows_sha256=str(manifest["rows_content_sha256"]),
        cache_commit=str(parent["cache_builder_git_commit"]),
        parent_config_sha256=str(parent["main_config_sha256"]),
        cache_schema=str(parent["cache_schema"]),
        representations=representations,
        ks=ks,
        sigmas=sigmas,
        thresholds=thresholds,
        fixed_top_fraction=fixed_fraction,
        grid_shape=GRID_SHAPE,
        gaussian_truncate=float(transform["gaussian_truncate"]),
        query_chunk_size=int(retrieval["query_chunk_size"]),
        library_chunk_size=int(retrieval["library_chunk_size"]),
        shrinkage_lambda=float(calibration["shrinkage_lambda"]),
        expected_sample_counts=MappingProxyType(expected_counts),
        expected_family_rows=MappingProxyType(expected_family_rows),
        expected_total_rows=expected_total_rows,
        output_root=Path(str(output["root"])),
        required_fold_files=required,
    )


def candidate_specs(plan: Plan) -> tuple[TailCandidateSpec, ...]:
    candidates: list[TailCandidateSpec] = []
    for representation in plan.representations:
        for k in plan.ks:
            for sigma in plan.sigmas:
                candidates.append(
                    TailCandidateSpec(
                        representation,
                        k,
                        sigma,
                        "fixed_top_fraction",
                        plan.fixed_top_fraction,
                    )
                )
                candidates.extend(
                    TailCandidateSpec(
                        representation,
                        k,
                        sigma,
                        "calibrated_tail_anomaly_threshold",
                        threshold,
                    )
                    for threshold in plan.thresholds
                )
    _require(len(candidates) == FROZEN_CANDIDATE_COUNT, "candidate grid drifted")
    _require(
        len({candidate.candidate_id for candidate in candidates})
        == FROZEN_CANDIDATE_COUNT,
        "candidate IDs are not unique",
    )
    return tuple(candidates)


def nested_pca_fit_schedule(
    plan: Plan, outer_family: str
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the four inner fits followed by the final four-family fit."""

    _require(outer_family in plan.family_order, "unknown outer family")
    nonouter = tuple(family for family in plan.family_order if family != outer_family)
    schedule = tuple(
        (
            inner_family,
            tuple(family for family in nonouter if family != inner_family),
        )
        for inner_family in nonouter
    ) + (("final", nonouter),)
    _require(len(schedule) == 5, "nested PCA schedule must contain 4 inner + final")
    _require(all(len(families) == 3 for _, families in schedule[:4]), "inner PCA fit width drifted")
    _require(len(schedule[-1][1]) == 4, "final PCA fit width drifted")
    return schedule


def load_cache_rows(plan: Plan) -> tuple[list[CacheRow], dict[str, Any]]:
    """Authenticate the frozen 32-row train-cache manifest without cache access."""

    payload, identity = _read_authenticated_bytes(
        plan.manifest_path,
        expected_size=plan.manifest_size,
        expected_sha256=plan.manifest_sha256,
    )
    manifest = json.loads(payload.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "input manifest root is invalid")
    _require(manifest.get("schema") == plan.manifest_schema, "manifest schema drifted")
    _require(int(manifest.get("row_count", -1)) == 32, "manifest row count drifted")
    _require(
        manifest.get("rows_content_sha256") == plan.manifest_rows_sha256,
        "manifest row-content hash drifted",
    )
    _require(manifest.get("git_commit") == plan.cache_commit, "cache commit drifted")
    _require(
        manifest.get("main_config_sha256") == plan.parent_config_sha256,
        "parent config drifted",
    )
    _require(manifest.get("test_dataset_access") is False, "manifest accessed test data")
    raw_rows = manifest.get("rows")
    _require(isinstance(raw_rows, list), "manifest rows are missing")
    rows: list[CacheRow] = []
    dataset_order = {dataset: index for index, dataset in enumerate(plan.dataset_to_family)}
    for raw_row in raw_rows:
        _require(isinstance(raw_row, Mapping), "manifest cache row is invalid")
        dataset = str(raw_row.get("dataset"))
        _require(dataset in plan.dataset_to_family, f"unauthorized dataset: {dataset}")
        path = Path(str(raw_row.get("cache_path")))
        path_text = path.as_posix()
        _require("/primitive_cache/train/" in path_text, f"non-train cache path: {path}")
        _require(
            not any(name in path_text for name in ("tangaroa", "smokeBuoyancy")),
            "test cache path is forbidden",
        )
        digest = str(raw_row.get("cache_file_sha256"))
        _require(_lower_hex(digest), f"invalid cache SHA-256: {path}")
        rows.append(
            CacheRow(
                dataset=dataset,
                family=plan.dataset_to_family[dataset],
                source_ordinal=int(raw_row["source_ordinal"]),
                source_index=int(raw_row["source_index"]),
                path=path,
                size_bytes=int(raw_row["cache_size_bytes"]),
                sha256=digest,
            )
        )
    _require(len(rows) == 32, "manifest did not yield exactly 32 cache rows")
    _require(
        len({(row.dataset, row.source_ordinal) for row in rows}) == 32,
        "duplicate cache identity",
    )
    for dataset in plan.dataset_to_family:
        ordinals = sorted(row.source_ordinal for row in rows if row.dataset == dataset)
        _require(ordinals == [0, 1, 2, 3], f"{dataset}: source ordinals drifted")
    rows.sort(key=lambda row: (dataset_order[row.dataset], row.source_ordinal))
    identity.update(
        {
            "schema": manifest["schema"],
            "rows_content_sha256": manifest["rows_content_sha256"],
            "row_count": 32,
        }
    )
    return rows, identity


def _cache_metadata(plan: Plan, row: CacheRow, scalar: np.ndarray) -> Mapping[str, Any]:
    _require(scalar.shape == (), f"{row.path}: metadata_json is not scalar")
    metadata = json.loads(str(scalar.item()))
    _require(isinstance(metadata, Mapping), f"{row.path}: metadata root is invalid")
    _require(metadata.get("schema") == plan.cache_schema, f"{row.path}: schema drifted")
    _require(
        metadata.get("experiment") == "mainExp_TemplateMatching_3.1",
        f"{row.path}: parent experiment drifted",
    )
    _require(metadata.get("split") == "train", f"{row.path}: split drifted")
    _require(metadata.get("dataset") == row.dataset, f"{row.path}: dataset drifted")
    _require(
        metadata.get("physical_family") == row.family,
        f"{row.path}: family drifted",
    )
    _require(
        int(metadata.get("source_ordinal", -1)) == row.source_ordinal,
        f"{row.path}: source ordinal drifted",
    )
    _require(
        int(metadata.get("source_index", -1)) == row.source_index,
        f"{row.path}: source index drifted",
    )
    _require(
        metadata.get("config_sha256") == plan.parent_config_sha256,
        f"{row.path}: parent config drifted",
    )
    _require(
        metadata.get("cache_builder_git_commit") == plan.cache_commit,
        f"{row.path}: cache commit drifted",
    )
    _require(int(metadata.get("valid_count", -1)) > 0, f"{row.path}: invalid count")
    _require(
        isinstance(metadata.get("array_sha256"), Mapping),
        f"{row.path}: array hash table is missing",
    )
    return metadata


def _validate_identity_arrays(
    row: CacheRow,
    raw_features: np.ndarray,
    scale_ids: np.ndarray,
    center_indices: np.ndarray,
    block_indices: np.ndarray,
    assigned_indices: np.ndarray,
) -> None:
    count = len(raw_features)
    _require(count > 0, f"{row.path}: empty Raw cache projection")
    _require(
        raw_features.dtype == np.dtype(np.float32)
        and raw_features.shape == (count, RAW_INPUT_WIDTH),
        f"{row.path}: raw_features dtype/shape drifted",
    )
    _require(np.isfinite(raw_features).all(), f"{row.path}: nonfinite Raw672 row")
    expected = (
        (scale_ids, np.dtype(np.int32)),
        (center_indices, np.dtype(np.int64)),
        (block_indices, np.dtype(np.int8)),
        (assigned_indices, np.dtype(np.int64)),
    )
    _require(
        all(values.dtype == dtype and values.shape == (count,) for values, dtype in expected),
        f"{row.path}: row identity dtype/shape drifted",
    )
    _require(np.all((scale_ids >= 0) & (scale_ids < 2000)), f"{row.path}: invalid scale")
    _require(
        np.all((center_indices >= 0) & (center_indices < 64000)),
        f"{row.path}: invalid center index",
    )
    _require(np.all((block_indices >= 0) & (block_indices < 2)), f"{row.path}: invalid block")
    _require(
        np.array_equal(block_indices, (scale_ids >= 1000).astype(np.int8)),
        f"{row.path}: scale/block identity drifted",
    )
    _require(
        np.array_equal(
            assigned_indices,
            block_indices.astype(np.int64) * 64000 + center_indices,
        ),
        f"{row.path}: assigned-row identity drifted",
    )
    for block in (0, 1):
        selected = center_indices[block_indices == block]
        _require(
            len(selected) == len(np.unique(selected)),
            f"{row.path}: duplicate center within block",
        )


def _authenticate_member_hashes(
    row: CacheRow,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    hashes = metadata.get("array_sha256")
    _require(isinstance(hashes, Mapping), f"{row.path}: array hashes missing")
    for name, values in arrays.items():
        if name == "metadata_json":
            continue
        _require(
            hashes.get(name) == canonical_array_sha256(np.asarray(values)),
            f"{row.path}: array hash mismatch: {name}",
        )


def _open_cache_members(
    row: CacheRow, names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Open only an explicit Raw experiment member list after file auth."""

    requested = tuple(str(name) for name in names)
    _require(len(requested) == len(set(requested)), "duplicate cache member request")
    _require("fmt_features" not in requested, "FMT cache member is forbidden")
    allowed = {
        "raw_features",
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
        "valid_labels",
        "metadata_json",
    }
    _require(set(requested).issubset(allowed), "non-Raw cache member requested")
    with _authenticated_open_file(
        row.path, expected_size=row.size_bytes, expected_sha256=row.sha256
    ) as opened:
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(
                all(name in archive.files for name in requested),
                f"{row.path}: required Raw projection member is missing",
            )
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in requested
            }
    return arrays


def load_pca_fit_block(
    plan: Plan, row: CacheRow
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Load Raw672 plus authentication metadata, never labels or FMT."""

    opened = ("raw_features", "metadata_json")
    arrays = _open_cache_members(row, opened)
    metadata = _cache_metadata(plan, row, arrays["metadata_json"])
    raw = np.asarray(arrays["raw_features"])
    count = int(metadata["valid_count"])
    _require(
        raw.dtype == np.dtype(np.float32) and raw.shape == (count, RAW_INPUT_WIDTH),
        f"{row.path}: PCA Raw672 contract drifted",
    )
    _require(np.isfinite(raw).all(), f"{row.path}: nonfinite PCA Raw672 row")
    _authenticate_member_hashes(row, metadata, {"raw_features": raw})
    audit = {
        "dataset": row.dataset,
        "family": row.family,
        "source_ordinal": row.source_ordinal,
        "source_index": row.source_index,
        "path": str(row.path),
        "size_bytes": row.size_bytes,
        "file_sha256": row.sha256,
        "row_count": count,
        "raw_features_sha256": canonical_array_sha256(raw),
        "opened_members": list(opened),
        "valid_labels_opened": False,
        "fmt_features_opened": False,
    }
    return np.ascontiguousarray(raw), _deep_freeze(audit)


def load_raw_projection(
    plan: Plan, row: CacheRow, *, include_labels: bool
) -> RawCacheProjection:
    """Load a strict Raw projection; label-free mode never opens metadata."""

    names = [
        "raw_features",
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
    ]
    if include_labels:
        names.extend(("valid_labels", "metadata_json"))
    arrays = _open_cache_members(row, names)
    raw = np.asarray(arrays["raw_features"])
    scales = np.asarray(arrays["valid_scale_id"])
    centers = np.asarray(arrays["valid_center_seed_index"])
    blocks = np.asarray(arrays["valid_scale_block_index"])
    assigned = np.asarray(arrays["valid_assigned_row_index"])
    _validate_identity_arrays(row, raw, scales, centers, blocks, assigned)
    metadata: Mapping[str, Any] = {}
    labels: np.ndarray | None = None
    if include_labels:
        metadata = _cache_metadata(plan, row, arrays["metadata_json"])
        _require(
            int(metadata["valid_count"]) == len(raw),
            f"{row.path}: valid count drifted",
        )
        labels = np.asarray(arrays["valid_labels"])
        _require(
            labels.dtype == np.dtype(np.bool_) and labels.shape == (len(raw),),
            f"{row.path}: valid_labels contract drifted",
        )
        _authenticate_member_hashes(
            row,
            metadata,
            {
                "raw_features": raw,
                "valid_scale_id": scales,
                "valid_center_seed_index": centers,
                "valid_scale_block_index": blocks,
                "valid_assigned_row_index": assigned,
                "valid_labels": labels,
            },
        )
    return RawCacheProjection(
        row=row,
        raw_features=np.ascontiguousarray(raw),
        scale_ids=np.ascontiguousarray(scales),
        center_indices=np.ascontiguousarray(centers),
        block_indices=np.ascontiguousarray(blocks),
        assigned_row_indices=np.ascontiguousarray(assigned),
        labels=None if labels is None else np.ascontiguousarray(labels),
        metadata=_deep_freeze(metadata),
        opened_members=tuple(names),
    )


def _raw_pca_array_records(model: RawPCARepresentation) -> list[dict[str, Any]]:
    serialized = serialize_raw_pca(model)
    return [
        {
            "name": audit.name,
            "dtype": audit.dtype,
            "shape": list(audit.shape),
            "sha256": audit.sha256,
        }
        for audit in serialized.array_audits
    ]


def _pca_solver_manifest() -> dict[str, Any]:
    return {
        "name": RAW_PCA_SOLVER,
        "row_chunk_size": RAW_PCA_ROW_CHUNK_SIZE,
        "first_pass_feature_sum_dtype": "float64",
        "first_pass_sample_count_dtype": "int64",
        "mean_formula": "feature_sum_divided_by_sample_count",
        "second_pass_raw_cast_dtype": "float64",
        "second_pass_scatter_accumulator_dtype": "float64",
        "scatter_formula": "sum_of_centered_transpose_times_centered",
        "symmetrization": "0.5_times_scatter_plus_scatter_transpose",
        "eigendecomposition": "numpy.linalg.eigh",
        "stable_descending_order": True,
        "materially_negative_tolerance": (
            "max_1e_minus_12_and_largest_nonnegative_eigenvalue_times_1e_minus_10"
        ),
        "materially_negative_policy": "fail_closed",
        "remaining_negative_policy": "clamp_to_zero",
        "component_sign_pivot": "first_argmax_absolute_loading",
        "component_sign_rule": "pivot_loading_nonnegative",
        "singular_value_formula": "sqrt_selected_clamped_eigenvalue",
        "explained_variance_ratio_formula": (
            "selected_clamped_eigenvalue_divided_by_all_clamped_eigenvalue_sum"
        ),
        "transform_formula": (
            "contiguous_float32_of_raw_float32_minus_mean_float32_times_components_transpose"
        ),
        "whitening": False,
        "post_transform_global_standardization": False,
    }


def fit_pca_from_cache_rows(
    plan: Plan,
    rows: Sequence[CacheRow],
    *,
    outer_family: str,
    held_out: str,
    expected_sample_count: int | None = None,
) -> RawPCAFitResult:
    """Fit two-pass PCA through the label-free Raw-only loader."""

    fit_families = tuple(
        family for family in plan.family_order if any(row.family == family for row in rows)
    )
    nonouter = tuple(family for family in plan.family_order if family != outer_family)
    expected_families = (
        nonouter
        if held_out == "final"
        else tuple(family for family in nonouter if family != held_out)
    )
    _require(
        held_out == "final" or held_out in nonouter,
        "invalid PCA held-out family",
    )
    _require(fit_families == expected_families, "PCA fit-family set drifted")
    expected_keys = tuple(
        (dataset, family, source_ordinal)
        for family in expected_families
        for dataset in plan.families[family]
        for source_ordinal in range(4)
    )
    actual_keys = tuple(
        (row.dataset, row.family, row.source_ordinal) for row in rows
    )
    _require(actual_keys == expected_keys, "PCA fit cache scope/order drifted")
    _require(rows and len(fit_families) in (3, 4), "invalid PCA fit population")
    pass_audits: list[tuple[Mapping[str, Any], ...]] = []

    def block_factory() -> Iterable[np.ndarray]:
        current: list[Mapping[str, Any]] = []
        for row in rows:
            raw, audit = load_pca_fit_block(plan, row)
            current.append(audit)
            yield raw
            del raw
        pass_audits.append(tuple(current))

    model = fit_raw_pca(block_factory)
    _require(len(pass_audits) == 2, "Raw-PCA did not execute exactly two passes")
    _require(
        _json_safe(pass_audits[0]) == _json_safe(pass_audits[1]),
        "Raw-PCA cache population changed between passes",
    )
    frozen_count = (
        plan.expected_sample_counts[(outer_family, held_out)]
        if expected_sample_count is None
        else int(expected_sample_count)
    )
    _require(model.sample_count == frozen_count, "Raw-PCA sample count drifted")
    audit = {
        "outer_family": outer_family,
        "held_out": held_out,
        "ordered_fit_family_set": list(fit_families),
        "ordered_fit_caches": list(pass_audits[0]),
        "cache_count": len(rows),
        "sample_count": model.sample_count,
        "fit_population": "every_valid_raw_row_irrespective_of_label",
        "labels_opened_for_pca": False,
        "fmt_features_opened": False,
        "outer_raw_features_opened": False,
        "solver": RAW_PCA_SOLVER,
        "row_chunk_size": RAW_PCA_ROW_CHUNK_SIZE,
        "first_and_second_pass_dtype": "float64",
        "serialized_mean_and_components_dtype": "float32",
        "whitening": False,
        "post_transform_global_standardization": False,
        "pca_arrays": _raw_pca_array_records(model),
    }
    return RawPCAFitResult(model=model, audit=_deep_freeze(audit))


def project_cache_rows(
    plan: Plan,
    rows: Sequence[CacheRow],
    pca: RawPCARepresentation,
    *,
    include_labels: bool,
) -> list[CacheProjection]:
    """Stream Raw caches and retain only PCA161 projections and row identity."""

    output: list[CacheProjection] = []
    for row in rows:
        raw = load_raw_projection(plan, row, include_labels=include_labels)
        features = pca.transform(raw.raw_features)
        output.append(
            CacheProjection(
                row=row,
                fmt_features=_deep_freeze(features),
                scale_ids=_deep_freeze(raw.scale_ids),
                center_indices=_deep_freeze(raw.center_indices),
                block_indices=_deep_freeze(raw.block_indices),
                assigned_row_indices=_deep_freeze(raw.assigned_row_indices),
                labels=(None if raw.labels is None else _deep_freeze(raw.labels)),
                metadata=raw.metadata,
            )
        )
        del raw
        gc.collect()
    return output


def load_reference_for_projection(
    plan: Plan, projection: CacheProjection
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Open labels/metadata only after a label-free score or replay is closed."""

    row = projection.row
    names = (
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
        "valid_labels",
        "metadata_json",
    )
    arrays = _open_cache_members(row, names)
    metadata = _cache_metadata(plan, row, arrays["metadata_json"])
    count = projection.count
    _require(
        int(metadata["valid_count"]) == count,
        f"{row.path}: reference valid count drifted",
    )
    labels = np.asarray(arrays["valid_labels"])
    _require(
        labels.dtype == np.dtype(np.bool_) and labels.shape == (count,),
        f"{row.path}: reference label contract drifted",
    )
    expected = {
        "valid_scale_id": projection.scale_ids,
        "valid_center_seed_index": projection.center_indices,
        "valid_scale_block_index": projection.block_indices,
        "valid_assigned_row_index": projection.assigned_row_indices,
    }
    for name, values in expected.items():
        _require(
            np.array_equal(np.asarray(arrays[name]), values),
            f"{row.path}: prediction/reference row identity drifted: {name}",
        )
    _authenticate_member_hashes(
        row,
        metadata,
        {
            **{name: np.asarray(arrays[name]) for name in expected},
            "valid_labels": labels,
        },
    )
    return np.ascontiguousarray(labels), _deep_freeze(metadata)


def _with_reference(
    projection: CacheProjection, labels: np.ndarray, metadata: Mapping[str, Any]
) -> CacheProjection:
    return CacheProjection(
        row=projection.row,
        fmt_features=projection.fmt_features,
        scale_ids=projection.scale_ids,
        center_indices=projection.center_indices,
        block_indices=projection.block_indices,
        assigned_row_indices=projection.assigned_row_indices,
        labels=_deep_freeze(np.ascontiguousarray(labels)),
        metadata=metadata,
    )


def _fit_tail_model(
    caches: Sequence[CacheProjection],
    plan: Plan,
    *,
    device: str,
    ks: Sequence[int] | None = None,
) -> PerScaleNegativeTailModel:
    feature_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    for cache in caches:
        _require(cache.labels is not None, "fit cache labels are unavailable")
        _require(
            cache.fmt_features.dtype == np.dtype(np.float32)
            and cache.fmt_features.shape == (cache.count, RAW_OUTPUT_WIDTH),
            "fit PCA161 projection drifted",
        )
        negative = ~np.asarray(cache.labels, dtype=bool)
        if negative.any():
            feature_parts.append(
                np.ascontiguousarray(cache.fmt_features[negative], dtype=np.float32)
            )
            scale_parts.append(
                np.ascontiguousarray(cache.scale_ids[negative], dtype=np.int64)
            )
    _require(feature_parts, "fit families contain no natural negatives")
    features = np.ascontiguousarray(np.concatenate(feature_parts), dtype=np.float32)
    scales = np.ascontiguousarray(np.concatenate(scale_parts), dtype=np.int64)
    model = PerScaleNegativeTailModel(
        features,
        scales,
        ks=plan.ks if ks is None else tuple(int(value) for value in ks),
        shrinkage_lambda=plan.shrinkage_lambda,
        device=device,
        query_chunk_size=plan.query_chunk_size,
        library_chunk_size=plan.library_chunk_size,
    )
    del features, scales, feature_parts, scale_parts
    return model


def _query_cache_batch(
    model: PerScaleNegativeTailModel,
    caches: Sequence[CacheProjection],
    plan: Plan,
    *,
    device: str,
    ks: Sequence[int] | None = None,
) -> dict[int, list[dict[str, np.ndarray]]]:
    requested = model.ks if ks is None else tuple(int(value) for value in ks)
    _require(requested and set(requested).issubset(plan.ks), "query k set drifted")
    offsets = np.cumsum([0, *(cache.count for cache in caches)], dtype=np.int64)
    if caches:
        features = np.ascontiguousarray(
            np.concatenate([cache.fmt_features for cache in caches]), dtype=np.float32
        )
        scales = np.ascontiguousarray(
            np.concatenate([cache.scale_ids for cache in caches]), dtype=np.int64
        )
    else:
        features = np.empty((0, RAW_OUTPUT_WIDTH), dtype=np.float32)
        scales = np.empty(0, dtype=np.int64)
    result = model.query(
        features,
        scales,
        ks=requested,
        device=device,
        query_chunk_size=plan.query_chunk_size,
        library_chunk_size=plan.library_chunk_size,
    )
    scaler_modes = model.scaler.mode_for_scales(scales)
    output: dict[int, list[dict[str, np.ndarray]]] = {}
    for k in requested:
        parts: list[dict[str, np.ndarray]] = []
        for index in range(len(caches)):
            selected = slice(int(offsets[index]), int(offsets[index + 1]))
            parts.append(
                {
                    "raw_distance": result.raw_distances[k][selected],
                    "tail_probability": result.tail_probabilities[k][selected],
                    "tail_anomaly": result.anomaly_scores[k][selected],
                    "retrieval_supported": result.retrieval_supported[k][selected],
                    "calibration_supported": result.calibration_supported[k][selected],
                    "calibration_mode": result.calibration_modes[k][selected],
                    "scaler_mode": scaler_modes[selected],
                }
            )
        output[k] = parts
    del features, scales
    return output


def _inner_metric_rows(
    plan: Plan,
    cache_rows: Sequence[CacheRow],
    outer_family: str,
    *,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, TailCandidateSpec], list[dict[str, Any]]]:
    """Execute four independent three-family PCA fits and inner evaluations."""

    rows: list[dict[str, Any]] = []
    candidates: dict[str, TailCandidateSpec] = {}
    audits: list[dict[str, Any]] = []
    schedule = nested_pca_fit_schedule(plan, outer_family)
    for inner_family, fit_families in schedule[:4]:
        fit_rows = [row for row in cache_rows if row.family in fit_families]
        query_rows = [row for row in cache_rows if row.family == inner_family]
        _require(fit_rows and query_rows, f"{inner_family}: empty nested fold")
        print(
            f"[{_utc_now()}] outer={outer_family} inner={inner_family} "
            "representation=raw_pca161 pca_fit_start",
            flush=True,
        )
        # Security boundary: this fit can only request Raw672 and metadata.  No
        # valid_labels or FMT member has been opened by this fold yet.
        pca_fit = fit_pca_from_cache_rows(
            plan,
            fit_rows,
            outer_family=outer_family,
            held_out=inner_family,
        )
        # Labels become available only after the fitted PCA state and complete
        # fit population identity have been closed in pca_fit.audit.
        fit_caches = project_cache_rows(
            plan, fit_rows, pca_fit.model, include_labels=True
        )
        model = _fit_tail_model(fit_caches, plan, device=device)
        query_caches = project_cache_rows(
            plan, query_rows, pca_fit.model, include_labels=False
        )
        query = _query_cache_batch(model, query_caches, plan, device=device)
        audit = dict(_json_safe(pca_fit.audit))
        audit.update(
            {
                "inner_family": inner_family,
                "representation": "raw_pca161",
                "per_scale_scaler_fit_audit": model.scaler.fit_audit,
                "negative_tail_calibration_fit_audit": model.tail_calibrator.fit_audit,
                "device": device,
                "query_chunk_size": plan.query_chunk_size,
                "library_chunk_size": plan.library_chunk_size,
            }
        )
        audits.append(audit)
        for cache_index, label_free_cache in enumerate(query_caches):
            # Tail distances are already closed above.  Labels now enter only
            # the candidate-metric computation below.
            labels, metadata = load_reference_for_projection(plan, label_free_cache)
            cache = _with_reference(label_free_cache, labels, metadata)
            for block_index, block_name in enumerate(BLOCK_NAMES):
                block = np.asarray(cache.block_indices == block_index)
                _require(
                    block.any(),
                    f"{cache.row.dataset}/{cache.row.source_ordinal}/{block_name}: empty group",
                )
                group_labels = np.asarray(labels[block], dtype=bool)
                centers = np.asarray(cache.center_indices[block], dtype=np.int64)
                _require(
                    group_labels.any() and (~group_labels).any(),
                    f"{cache.row.dataset}/{cache.row.source_ordinal}/{block_name}: single-class group",
                )
                for k in plan.ks:
                    values = query[k][cache_index]
                    for sigma in plan.sigmas:
                        spatial = spatial_calibrated_tail_scores(
                            values["tail_anomaly"][block],
                            values["calibration_supported"][block],
                            centers,
                            sigma=sigma,
                            grid_shape=plan.grid_shape,
                            truncate=plan.gaussian_truncate,
                        )
                        eligible = spatial.calibration_supported | spatial.imputed
                        base = (
                            TailCandidateSpec(
                                "raw_pca161",
                                k,
                                sigma,
                                "fixed_top_fraction",
                                plan.fixed_top_fraction,
                            ),
                            *(
                                TailCandidateSpec(
                                    "raw_pca161",
                                    k,
                                    sigma,
                                    "calibrated_tail_anomaly_threshold",
                                    threshold,
                                )
                                for threshold in plan.thresholds
                            ),
                        )
                        for candidate in base:
                            previous = candidates.setdefault(
                                candidate.candidate_id, candidate
                            )
                            _require(previous == candidate, "candidate ID collision")
                        ranking = inherited._ranking_metrics_one_sort(
                            group_labels, spatial.scores
                        )
                        fixed_prediction = candidate_predictions(
                            base[0], spatial.scores, centers, eligible
                        )
                        rows.append(
                            _metric_row(
                                outer_family=outer_family,
                                inner_family=inner_family,
                                cache=cache,
                                block_name=block_name,
                                candidate=base[0],
                                labels=group_labels,
                                scores=spatial.scores,
                                predictions=fixed_prediction,
                                retrieval_supported=values["retrieval_supported"][block],
                                calibration_supported=spatial.calibration_supported,
                                imputed=spatial.imputed,
                                unimputable=spatial.unimputable,
                                calibration_modes=values["calibration_mode"][block],
                                scaler_modes=values["scaler_mode"][block],
                                ranking_metrics=ranking,
                            )
                        )
                        rows.extend(
                            _threshold_metric_rows(
                                outer_family=outer_family,
                                inner_family=inner_family,
                                cache=cache,
                                block_name=block_name,
                                candidates=base[1:],
                                labels=group_labels,
                                scores=spatial.scores,
                                eligible=eligible,
                                retrieval_supported=values["retrieval_supported"][block],
                                calibration_supported=spatial.calibration_supported,
                                imputed=spatial.imputed,
                                unimputable=spatial.unimputable,
                                calibration_modes=values["calibration_mode"][block],
                                scaler_modes=values["scaler_mode"][block],
                                ranking_metrics=ranking,
                            )
                        )
        del pca_fit, fit_caches, query_caches, model, query
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _require(len(audits) == 4, "inner PCA fit audit count drifted")
    _require(len(candidates) == FROZEN_CANDIDATE_COUNT, "candidate count drifted")
    return rows, candidates, audits


def _hierarchical_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    families = sorted({str(row["inner_family"]) for row in rows})
    family_values: list[float] = []
    for family in families:
        values = np.asarray(
            [float(row[field]) for row in rows if row["inner_family"] == family],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        family_values.append(float(np.mean(finite)) if len(finite) else float("nan"))
    finite = np.asarray(family_values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _aggregate_and_select(
    rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, TailCandidateSpec],
) -> tuple[list[dict[str, Any]], TailCandidateSpec, dict[str, Any]]:
    _require(
        rows and len(candidates) == FROZEN_CANDIDATE_COUNT,
        "inner selection population is incomplete",
    )
    grouped = {candidate_id: [] for candidate_id in candidates}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        _require(candidate_id in grouped, "unknown candidate metric")
        grouped[candidate_id].append(row)
    expected_keys: set[tuple[str, str, int, str]] | None = None
    summaries: list[dict[str, Any]] = []
    fields = (
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
    for candidate_id in sorted(grouped):
        candidate_rows = grouped[candidate_id]
        keys = {
            (
                str(row["inner_family"]),
                str(row["dataset"]),
                int(row["source_ordinal"]),
                str(row["block"]),
            )
            for row in candidate_rows
        }
        _require(len(keys) == len(candidate_rows), "duplicate inner candidate group")
        if expected_keys is None:
            expected_keys = keys
        _require(keys == expected_keys, "inner candidate group coverage drifted")
        _require(len({key[0] for key in keys}) == 4, "expected four inner families")
        summary = _candidate_payload(candidates[candidate_id])
        for name in fields:
            summary[name] = _hierarchical_mean(candidate_rows, name)
        summary["inner_family_count"] = 4
        summary["group_count"] = len(candidate_rows)
        summaries.append(summary)

    def selection_key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -float(summary["f1"]),
            -float(summary["average_precision"]),
            -float(summary["balanced_accuracy"]),
            -float(summary["precision"]),
            -float(summary["recall"]),
            str(summary["candidate_id"]),
        )

    selected_summary = min(summaries, key=selection_key)
    selected = candidates[str(selected_summary["candidate_id"])]
    return summaries, selected, dict(selected_summary)


def _pca_binding(pca: VerifiedPCAArtifact) -> dict[str, Any]:
    _require(
        pca._authentication_seal is _PCA_AUTHENTICATION_SEAL,
        "an authenticated final PCA is required",
    )
    return {
        "manifest": {
            "path": pca.manifest_path.name,
            "file_sha256": pca.manifest_file_sha256,
            "content_sha256": pca.manifest["content_sha256"],
        },
        "file": {
            "path": "final_pca.npz",
            "sha256": pca.pca_file_sha256,
        },
    }


def write_final_pca_artifact(
    output_directory: Path,
    fitted: RawPCAFitResult,
    *,
    plan: Plan,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
) -> tuple[Path, Path, str, str]:
    """Publish the final PCA before any outer Raw member is opened."""

    _require(
        fitted.audit.get("held_out") == "final",
        "final PCA artifact requires the final fit population",
    )
    _require(
        fitted.audit.get("outer_raw_features_opened") is False,
        "outer Raw feature was opened before final PCA publication",
    )
    _require(
        tuple(fitted.audit.get("ordered_fit_family_set", ()))
        == tuple(fit_families),
        "final PCA audit fit-family binding drifted",
    )
    serialized = serialize_raw_pca(fitted.model)
    pca_path = output_directory / "final_pca.npz"
    pca_sha = _atomic_bytes(pca_path, serialized.payload)
    _require(pca_sha == serialized.npz_sha256, "serialized PCA hash drifted")
    array_records = [
        {
            "name": audit.name,
            "dtype": audit.dtype,
            "shape": list(audit.shape),
            "sha256": audit.sha256,
        }
        for audit in serialized.array_audits
    ]
    manifest = _manifest_with_self_hash(
        {
            "schema": PCA_MANIFEST_SCHEMA,
            "artifact_schema": RAW_PCA_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_path": str(plan.path),
            "config_sha256": plan.sha256,
            "input_manifest_sha256": plan.manifest_sha256,
            "parent_cache_commit": plan.cache_commit,
            "parent_config_sha256": plan.parent_config_sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "ordered_fit_family_set": list(fit_families),
            "ordered_fit_caches": fitted.audit["ordered_fit_caches"],
            "total_sample_count": fitted.model.sample_count,
            "fit_population": fitted.audit["fit_population"],
            "labels_opened_for_pca": False,
            "fmt_features_opened": False,
            "outer_raw_features_opened": False,
            "solver": _pca_solver_manifest(),
            "artifact_file": {
                "path": pca_path.name,
                "size_bytes": pca_path.stat().st_size,
                "sha256": pca_sha,
            },
            "array_count": len(array_records),
            "arrays": array_records,
        }
    )
    manifest_path = output_directory / "final_pca_manifest.json"
    manifest_file_sha = _atomic_json(manifest_path, manifest)
    return pca_path, manifest_path, pca_sha, manifest_file_sha


def authenticate_and_rebuild_final_pca(
    pca_path: Path,
    manifest_path: Path,
    *,
    plan: Plan,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
    expected_manifest_file_sha256: str,
) -> VerifiedPCAArtifact:
    """Authenticate NPZ bytes, every PCA member, then rebuild the model."""

    manifest_bytes, manifest_identity = _read_authenticated_bytes(
        manifest_path, expected_sha256=expected_manifest_file_sha256
    )
    manifest_file_sha = str(manifest_identity["sha256"])
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "PCA manifest root is invalid")
    _authenticate_self_hash(manifest)
    _require(manifest.get("schema") == PCA_MANIFEST_SCHEMA, "PCA manifest schema drifted")
    _require(manifest.get("artifact_schema") == RAW_PCA_SCHEMA, "PCA artifact schema drifted")
    _require(manifest.get("experiment") == EXPERIMENT, "PCA experiment drifted")
    _require(manifest.get("config_sha256") == plan.sha256, "PCA config drifted")
    _require(
        manifest.get("input_manifest_sha256") == plan.manifest_sha256,
        "PCA input manifest drifted",
    )
    _require(manifest.get("parent_cache_commit") == plan.cache_commit, "PCA cache commit drifted")
    _require(
        manifest.get("parent_config_sha256") == plan.parent_config_sha256,
        "PCA parent config drifted",
    )
    _require(manifest.get("git_commit") == git_commit, "PCA Git binding drifted")
    _require(manifest.get("outer_family") == outer_family, "PCA outer family drifted")
    _require(
        manifest.get("ordered_fit_family_set") == list(fit_families),
        "PCA fit-family binding drifted",
    )
    cache_records = manifest.get("ordered_fit_caches")
    expected_cache_keys = [
        (family, dataset, source_ordinal)
        for family in fit_families
        for dataset in plan.families[family]
        for source_ordinal in range(4)
    ]
    _require(
        isinstance(cache_records, list)
        and [
            (
                str(record.get("family")),
                str(record.get("dataset")),
                int(record.get("source_ordinal", -1)),
            )
            for record in cache_records
            if isinstance(record, Mapping)
        ]
        == expected_cache_keys,
        "PCA fit-cache population/order drifted",
    )
    for record in cache_records:
        _require(isinstance(record, Mapping), "PCA fit-cache record is invalid")
        _require(
            int(record.get("size_bytes", -1)) > 0
            and int(record.get("row_count", -1)) > 0
            and _lower_hex(record.get("file_sha256"))
            and _lower_hex(record.get("raw_features_sha256")),
            "PCA fit-cache identity is invalid",
        )
        _require(
            record.get("opened_members") == ["raw_features", "metadata_json"]
            and record.get("valid_labels_opened") is False
            and record.get("fmt_features_opened") is False,
            "PCA fit-cache access audit drifted",
        )
    _require(
        int(manifest.get("total_sample_count", -1))
        == plan.expected_sample_counts[(outer_family, "final")],
        "PCA final sample count drifted",
    )
    _require(
        sum(int(record["row_count"]) for record in cache_records)
        == int(manifest["total_sample_count"]),
        "PCA fit-cache row counts do not sum to total",
    )
    _require(
        manifest.get("fit_population")
        == "every_valid_raw_row_irrespective_of_label",
        "PCA fit population drifted",
    )
    _require(
        manifest.get("labels_opened_for_pca") is False
        and manifest.get("fmt_features_opened") is False
        and manifest.get("outer_raw_features_opened") is False,
        "PCA access gate drifted",
    )
    _require(
        manifest.get("solver") == _pca_solver_manifest(),
        "PCA numerical solver manifest drifted",
    )
    file_record = manifest.get("artifact_file")
    _require(isinstance(file_record, Mapping), "PCA file identity is missing")
    _require(file_record.get("path") == pca_path.name, "PCA file path drifted")
    records = manifest.get("arrays")
    _require(
        isinstance(records, list)
        and len(records) == len(RAW_PCA_ARRAY_NAMES)
        and int(manifest.get("array_count", -1)) == len(RAW_PCA_ARRAY_NAMES),
        "PCA array manifest is incomplete",
    )
    audits: list[RawPCAArrayAudit] = []
    for expected_name, record in zip(RAW_PCA_ARRAY_NAMES, records, strict=True):
        _require(isinstance(record, Mapping), "PCA array record is invalid")
        _require(record.get("name") == expected_name, "PCA array order drifted")
        audits.append(
            RawPCAArrayAudit(
                name=expected_name,
                dtype=str(record["dtype"]),
                shape=tuple(int(value) for value in record["shape"]),
                sha256=str(record["sha256"]),
            )
        )
    payload, identity = _read_authenticated_bytes(
        pca_path,
        expected_size=int(file_record["size_bytes"]),
        expected_sha256=str(file_record["sha256"]),
    )
    rebuilt = deserialize_raw_pca(
        payload,
        expected_npz_sha256=str(identity["sha256"]),
        expected_array_audits=tuple(audits),
    )
    _require(
        rebuilt.sample_count == int(manifest["total_sample_count"]),
        "rebuilt PCA sample count drifted",
    )
    return VerifiedPCAArtifact(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_sha,
        pca_file_sha256=str(identity["sha256"]),
        manifest=_deep_freeze(manifest),
        model=rebuilt,
        _authentication_seal=_PCA_AUTHENTICATION_SEAL,
    )


def _authenticate_npz_arrays(
    path: Path,
    file_record: Mapping[str, Any],
    records: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], str]:
    _require(file_record.get("path") == path.name, f"artifact path drifted: {path.name}")
    with _authenticated_open_file(
        path,
        expected_size=int(file_record["size_bytes"]),
        expected_sha256=str(file_record["sha256"]),
    ) as opened:
        identity = {
            "size_bytes": opened.size_bytes,
            "sha256": opened.sha256,
        }
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(set(archive.files) == set(records), f"NPZ member set drifted: {path}")
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    for name, values in arrays.items():
        record = records[name]
        _require(isinstance(record, Mapping), f"invalid array record: {name}")
        _require(values.dtype.str == record.get("dtype"), f"dtype drifted: {name}")
        _require(list(values.shape) == record.get("shape"), f"shape drifted: {name}")
        _require(
            canonical_array_sha256(values) == record.get("sha256"),
            f"array SHA-256 drifted: {name}",
        )
    return arrays, str(identity["sha256"])


def write_final_scaler_artifact(
    output_directory: Path,
    model: PerScaleNegativeTailModel,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    pca: VerifiedPCAArtifact,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
) -> tuple[Path, Path, str, str]:
    arrays = model.scaler.export_arrays()
    _require(tuple(arrays) == SCALER_ARRAY_NAMES, "scaler member order drifted")
    scaler_path = output_directory / "final_per_scale_scaler.npz"
    scaler_sha = _atomic_npz(scaler_path, arrays)
    manifest = _manifest_with_self_hash(
        {
            "schema": SCALER_MANIFEST_SCHEMA,
            "artifact_schema": SCALER_ARTIFACT_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_sha256": plan.sha256,
            "input_manifest_sha256": plan.manifest_sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "fit_families": list(fit_families),
            "representation": "raw_pca161",
            "selected_candidate": _candidate_payload(selected),
            "final_pca": _pca_binding(pca),
            "outer_raw_features_opened": False,
            "artifact_file": {
                "path": scaler_path.name,
                "size_bytes": scaler_path.stat().st_size,
                "sha256": scaler_sha,
            },
            "array_count": len(arrays),
            "arrays": _array_manifest(arrays),
            "fit_audit": model.scaler.fit_audit,
        }
    )
    manifest_path = output_directory / "final_per_scale_scaler_manifest.json"
    manifest_sha = _atomic_json(manifest_path, manifest)
    return scaler_path, manifest_path, scaler_sha, manifest_sha


def authenticate_and_rebuild_final_scaler(
    scaler_path: Path,
    manifest_path: Path,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    pca: VerifiedPCAArtifact,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
    expected_manifest_file_sha256: str,
) -> VerifiedScalerArtifact:
    _require(pca._authentication_seal is _PCA_AUTHENTICATION_SEAL, "PCA auth required")
    payload, manifest_identity = _read_authenticated_bytes(
        manifest_path, expected_sha256=expected_manifest_file_sha256
    )
    manifest_sha = str(manifest_identity["sha256"])
    manifest = json.loads(payload.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "scaler manifest root is invalid")
    _authenticate_self_hash(manifest)
    _require(manifest.get("schema") == SCALER_MANIFEST_SCHEMA, "scaler schema drifted")
    _require(manifest.get("artifact_schema") == SCALER_ARTIFACT_SCHEMA, "scaler artifact drifted")
    _require(manifest.get("experiment") == EXPERIMENT, "scaler experiment drifted")
    _require(manifest.get("config_sha256") == plan.sha256, "scaler config drifted")
    _require(manifest.get("git_commit") == git_commit, "scaler Git binding drifted")
    _require(manifest.get("outer_family") == outer_family, "scaler outer family drifted")
    _require(manifest.get("fit_families") == list(fit_families), "scaler fit families drifted")
    _require(
        manifest.get("selected_candidate") == _json_safe(_candidate_payload(selected)),
        "scaler candidate drifted",
    )
    _require(manifest.get("final_pca") == _json_safe(_pca_binding(pca)), "scaler PCA binding drifted")
    _require(manifest.get("outer_raw_features_opened") is False, "scaler opened outer Raw early")
    file_record = manifest.get("artifact_file")
    records = manifest.get("arrays")
    _require(isinstance(file_record, Mapping) and isinstance(records, Mapping), "scaler arrays missing")
    _require(set(records) == set(SCALER_ARRAY_NAMES), "scaler member set drifted")
    _require(
        int(manifest.get("array_count", -1)) == len(SCALER_ARRAY_NAMES),
        "scaler array count drifted",
    )
    arrays, file_sha = _authenticate_npz_arrays(scaler_path, file_record, records)
    rebuilt = PerScaleNegativeScaler.from_arrays(arrays)
    _require(_json_safe(rebuilt.fit_audit) == manifest.get("fit_audit"), "scaler audit drifted")
    return VerifiedScalerArtifact(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_sha,
        scaler_file_sha256=file_sha,
        manifest=_deep_freeze(manifest),
        scaler=rebuilt,
        _authentication_seal=_SCALER_AUTHENTICATION_SEAL,
    )


def _scaler_binding(scaler: VerifiedScalerArtifact) -> dict[str, Any]:
    _require(
        scaler._authentication_seal is _SCALER_AUTHENTICATION_SEAL,
        "authenticated scaler required",
    )
    return {
        "manifest": {
            "path": scaler.manifest_path.name,
            "file_sha256": scaler.manifest_file_sha256,
            "content_sha256": scaler.manifest["content_sha256"],
        },
        "file": {
            "path": "final_per_scale_scaler.npz",
            "sha256": scaler.scaler_file_sha256,
        },
    }


def write_final_calibration_artifact(
    output_directory: Path,
    model: PerScaleNegativeTailModel,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    pca: VerifiedPCAArtifact,
    scaler: VerifiedScalerArtifact,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
) -> tuple[Path, Path, str, str]:
    _require(
        all(
            canonical_array_sha256(model.scaler.export_arrays()[name])
            == canonical_array_sha256(scaler.scaler.export_arrays()[name])
            for name in SCALER_ARRAY_NAMES
        ),
        "calibrator does not use authenticated scaler",
    )
    arrays = model.tail_calibrator.export_arrays()
    path = output_directory / "final_tail_calibration.npz"
    file_sha = _atomic_npz(path, arrays)
    manifest = _manifest_with_self_hash(
        {
            "schema": CALIBRATION_MANIFEST_SCHEMA,
            "artifact_schema": CALIBRATION_ARTIFACT_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_sha256": plan.sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "fit_families": list(fit_families),
            "selected_candidate": _candidate_payload(selected),
            "final_pca": _pca_binding(pca),
            "final_scaler": _scaler_binding(scaler),
            "outer_raw_features_opened": False,
            "artifact_file": {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha,
            },
            "array_count": len(arrays),
            "arrays": _array_manifest(arrays),
            "fit_audit": model.tail_calibrator.fit_audit,
        }
    )
    manifest_path = output_directory / "final_tail_calibration_manifest.json"
    manifest_sha = _atomic_json(manifest_path, manifest)
    return path, manifest_path, file_sha, manifest_sha


def authenticate_and_rebuild_final_calibration(
    calibration_path: Path,
    manifest_path: Path,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    pca: VerifiedPCAArtifact,
    scaler: VerifiedScalerArtifact,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
    expected_manifest_file_sha256: str,
) -> VerifiedCalibrationArtifact:
    _require(pca._authentication_seal is _PCA_AUTHENTICATION_SEAL, "PCA auth required")
    _require(scaler._authentication_seal is _SCALER_AUTHENTICATION_SEAL, "scaler auth required")
    payload, manifest_identity = _read_authenticated_bytes(
        manifest_path, expected_sha256=expected_manifest_file_sha256
    )
    manifest_sha = str(manifest_identity["sha256"])
    manifest = json.loads(payload.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "calibration manifest root is invalid")
    _authenticate_self_hash(manifest)
    _require(manifest.get("schema") == CALIBRATION_MANIFEST_SCHEMA, "calibration schema drifted")
    _require(
        manifest.get("artifact_schema") == CALIBRATION_ARTIFACT_SCHEMA,
        "calibration artifact schema drifted",
    )
    _require(manifest.get("experiment") == EXPERIMENT, "calibration experiment drifted")
    _require(manifest.get("config_sha256") == plan.sha256, "calibration config drifted")
    _require(manifest.get("git_commit") == git_commit, "calibration Git binding drifted")
    _require(manifest.get("outer_family") == outer_family, "calibration outer family drifted")
    _require(manifest.get("fit_families") == list(fit_families), "calibration fit families drifted")
    _require(
        manifest.get("selected_candidate") == _json_safe(_candidate_payload(selected)),
        "calibration candidate drifted",
    )
    _require(manifest.get("final_pca") == _json_safe(_pca_binding(pca)), "calibration PCA binding drifted")
    _require(
        manifest.get("final_scaler") == _json_safe(_scaler_binding(scaler)),
        "calibration scaler binding drifted",
    )
    _require(
        manifest.get("outer_raw_features_opened") is False,
        "calibration opened outer Raw early",
    )
    file_record = manifest.get("artifact_file")
    records = manifest.get("arrays")
    _require(
        isinstance(file_record, Mapping) and isinstance(records, Mapping),
        "calibration arrays missing",
    )
    _require(
        int(manifest.get("array_count", -1)) == len(records),
        "calibration array count drifted",
    )
    arrays, file_sha = _authenticate_npz_arrays(
        calibration_path, file_record, records
    )
    rebuilt = PerScaleNegativeTailModel.from_artifacts(
        scaler.scaler.export_arrays(), arrays
    )
    _require(rebuilt.ks == (selected.k,), "rebuilt calibration k drifted")
    _require(
        _json_safe(rebuilt.tail_calibrator.fit_audit) == manifest.get("fit_audit"),
        "calibration fit audit drifted",
    )
    return VerifiedCalibrationArtifact(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_sha,
        calibration_file_sha256=file_sha,
        manifest=_deep_freeze(manifest),
        model=rebuilt,
        _authentication_seal=_CALIBRATION_AUTHENTICATION_SEAL,
    )


def _calibration_binding(calibration: VerifiedCalibrationArtifact) -> dict[str, Any]:
    _require(
        calibration._authentication_seal is _CALIBRATION_AUTHENTICATION_SEAL,
        "authenticated calibrator required",
    )
    return {
        "manifest": {
            "path": calibration.manifest_path.name,
            "file_sha256": calibration.manifest_file_sha256,
            "content_sha256": calibration.manifest["content_sha256"],
        },
        "file": {
            "path": "final_tail_calibration.npz",
            "sha256": calibration.calibration_file_sha256,
        },
    }


def _inner_evidence_records(
    evidence_paths: Sequence[tuple[str, Path, str]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, path, expected_sha in evidence_paths:
        _payload, identity = _read_authenticated_bytes(
            path, expected_sha256=expected_sha
        )
        records[name] = {
            "path": path.name,
            "size_bytes": identity["size_bytes"],
            "sha256": identity["sha256"],
        }
    return records


def _validate_candidate_csv_identity(
    row: Mapping[str, str], candidate: TailCandidateSpec
) -> None:
    _require(row.get("candidate_id") == candidate.candidate_id, "candidate ID drifted")
    _require(row.get("representation") == "raw_pca161", "candidate representation drifted")
    _require(int(row["k"]) == candidate.k, "candidate k drifted")
    _require(float(row["sigma"]) == candidate.sigma, "candidate sigma drifted")
    _require(row.get("decision_rule") == candidate.decision_rule, "decision rule drifted")
    _require(float(row["decision_value"]) == candidate.decision_value, "decision value drifted")


def _authenticate_inner_selection_evidence(
    *,
    plan: Plan,
    outer_family: str,
    inner_group_metrics_path: Path,
    inner_group_metrics_sha256: str,
    inner_candidate_summary_path: Path,
    inner_candidate_summary_sha256: str,
    inner_fit_audits_path: Path,
    inner_fit_audits_sha256: str,
) -> tuple[TailCandidateSpec, dict[str, Any]]:
    """Rebuild the frozen 1,020-candidate selection from closed CSV evidence."""

    candidates = {candidate.candidate_id: candidate for candidate in candidate_specs(plan)}
    summary_rows: dict[str, dict[str, Any]] = {}
    summary_payload, _summary_identity = _read_authenticated_bytes(
        inner_candidate_summary_path,
        expected_sha256=inner_candidate_summary_sha256,
    )
    aggregate_fields = (
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
    with io.StringIO(summary_payload.decode("utf-8"), newline="") as source:
        reader = csv.DictReader(source)
        _require(tuple(reader.fieldnames or ()) == SUMMARY_FIELDS, "summary fields drifted")
        for raw in reader:
            candidate_id = str(raw["candidate_id"])
            _require(
                candidate_id in candidates and candidate_id not in summary_rows,
                "summary candidate set drifted",
            )
            _validate_candidate_csv_identity(raw, candidates[candidate_id])
            values = {name: float(raw[name]) for name in aggregate_fields}
            _require(np.isfinite(list(values.values())).all(), "nonfinite summary evidence")
            _require(int(raw["inner_family_count"]) == 4, "summary family count drifted")
            values.update(
                {
                    **_candidate_payload(candidates[candidate_id]),
                    "inner_family_count": 4,
                    "group_count": int(raw["group_count"]),
                }
            )
            summary_rows[candidate_id] = values
    _require(set(summary_rows) == set(candidates), "summary lacks 1,020 candidates")

    family_sums: dict[str, dict[str, dict[str, float]]] = {
        candidate_id: {} for candidate_id in candidates
    }
    family_counts: dict[str, dict[str, int]] = {
        candidate_id: {} for candidate_id in candidates
    }
    group_keys: dict[str, set[tuple[str, str, int, str]]] = {
        candidate_id: set() for candidate_id in candidates
    }
    metric_payload, _metric_identity = _read_authenticated_bytes(
        inner_group_metrics_path,
        expected_sha256=inner_group_metrics_sha256,
    )
    with io.StringIO(metric_payload.decode("utf-8"), newline="") as source:
        reader = csv.DictReader(source)
        _require(tuple(reader.fieldnames or ()) == METRIC_FIELDS, "metric fields drifted")
        for raw in reader:
            candidate_id = str(raw["candidate_id"])
            _require(candidate_id in candidates, "unknown candidate metric")
            _validate_candidate_csv_identity(raw, candidates[candidate_id])
            _require(raw.get("outer_family") == outer_family, "metric outer family drifted")
            inner_family = str(raw["inner_family"])
            dataset = str(raw["dataset"])
            block = str(raw["block"])
            _require(
                inner_family in plan.family_order and inner_family != outer_family,
                "metric inner family drifted",
            )
            _require(dataset in plan.families[inner_family], "metric dataset/family drifted")
            _require(block in BLOCK_NAMES, "metric block drifted")
            key = (inner_family, dataset, int(raw["source_ordinal"]), block)
            _require(key not in group_keys[candidate_id], "duplicate metric group")
            group_keys[candidate_id].add(key)
            values = np.asarray(
                [float(raw[name]) for name in aggregate_fields], dtype=np.float64
            )
            _require(np.isfinite(values).all(), "nonfinite metric selection evidence")
            sums = family_sums[candidate_id].setdefault(
                inner_family, {name: 0.0 for name in aggregate_fields}
            )
            for name, value in zip(aggregate_fields, values, strict=True):
                sums[name] += float(value)
            family_counts[candidate_id][inner_family] = (
                family_counts[candidate_id].get(inner_family, 0) + 1
            )
    expected_groups = {
        (family, dataset, source_ordinal, block)
        for family in plan.family_order
        if family != outer_family
        for dataset in plan.families[family]
        for source_ordinal in range(4)
        for block in BLOCK_NAMES
    }
    for candidate_id in candidates:
        _require(group_keys[candidate_id] == expected_groups, "metric group coverage drifted")
        _require(
            int(summary_rows[candidate_id]["group_count"]) == len(expected_groups),
            "summary group count drifted",
        )
        for name in aggregate_fields:
            recomputed = float(
                np.mean(
                    [
                        family_sums[candidate_id][family][name]
                        / family_counts[candidate_id][family]
                        for family in plan.family_order
                        if family != outer_family
                    ]
                )
            )
            _require(
                math.isclose(
                    recomputed,
                    float(summary_rows[candidate_id][name]),
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                ),
                f"summary does not reproduce groups: {candidate_id}/{name}",
            )

    audit_payload, _audit_identity = _read_authenticated_bytes(
        inner_fit_audits_path,
        expected_sha256=inner_fit_audits_sha256,
    )
    audit_root = json.loads(audit_payload.decode("utf-8"))
    _require(isinstance(audit_root, Mapping), "inner fit audit root is invalid")
    _authenticate_self_hash(audit_root)
    _require(audit_root.get("schema") == INNER_AUDIT_SCHEMA, "inner audit schema drifted")
    _require(audit_root.get("experiment") == EXPERIMENT, "inner audit experiment drifted")
    _require(audit_root.get("outer_family") == outer_family, "inner audit outer drifted")
    fits = audit_root.get("fits")
    _require(
        isinstance(fits, list)
        and int(audit_root.get("fit_count", -1)) == 4
        and len(fits) == 4,
        "inner PCA fit audit count drifted",
    )
    expected_inner = {family for family in plan.family_order if family != outer_family}
    actual_inner = {
        str(item.get("inner_family"))
        for item in fits
        if isinstance(item, Mapping)
    }
    _require(actual_inner == expected_inner, "inner PCA fold coverage drifted")
    for item in fits:
        _require(isinstance(item, Mapping), "inner PCA fit audit is invalid")
        _require(item.get("representation") == "raw_pca161", "inner representation drifted")
        _require(item.get("labels_opened_for_pca") is False, "PCA opened labels")
        _require(item.get("fmt_features_opened") is False, "PCA opened FMT features")
        inner_family = str(item["inner_family"])
        expected_fit_families = tuple(
            family
            for family in plan.family_order
            if family not in {outer_family, inner_family}
        )
        _require(
            tuple(item.get("ordered_fit_family_set", ()))
            == expected_fit_families,
            "inner PCA fit-family audit drifted",
        )
        expected_cache_count = 4 * sum(
            len(plan.families[family]) for family in expected_fit_families
        )
        cache_records = item.get("ordered_fit_caches")
        _require(
            isinstance(cache_records, list)
            and len(cache_records) == expected_cache_count,
            "inner PCA cache audit coverage drifted",
        )
        for cache_record in cache_records:
            _require(isinstance(cache_record, Mapping), "inner cache audit is invalid")
            _require(
                _lower_hex(cache_record.get("file_sha256"))
                and _lower_hex(cache_record.get("raw_features_sha256")),
                "inner cache audit hash is invalid",
            )
            _require(
                cache_record.get("valid_labels_opened") is False
                and cache_record.get("fmt_features_opened") is False,
                "inner PCA cache audit exposed a forbidden member",
            )
        pca_arrays = item.get("pca_arrays")
        _require(
            isinstance(pca_arrays, list)
            and [record.get("name") for record in pca_arrays if isinstance(record, Mapping)]
            == list(RAW_PCA_ARRAY_NAMES),
            "inner PCA array audit drifted",
        )
        for record in pca_arrays:
            _require(
                isinstance(record, Mapping)
                and _lower_hex(record.get("sha256")),
                "inner PCA array hash is invalid",
            )
            RawPCAArrayAudit(
                name=str(record["name"]),
                dtype=str(record["dtype"]),
                shape=tuple(int(value) for value in record["shape"]),
                sha256=str(record["sha256"]),
            )
        _require(
            isinstance(item.get("per_scale_scaler_fit_audit"), Mapping)
            and isinstance(item.get("negative_tail_calibration_fit_audit"), Mapping),
            "inner metric/calibration fit audit is missing",
        )
        _require(
            int(item.get("sample_count", -1))
            == plan.expected_sample_counts[(outer_family, inner_family)],
            "inner PCA sample count drifted",
        )

    def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -float(row["f1"]),
            -float(row["average_precision"]),
            -float(row["balanced_accuracy"]),
            -float(row["precision"]),
            -float(row["recall"]),
            str(row["candidate_id"]),
        )

    selected_summary = min(summary_rows.values(), key=selection_key)
    selected = candidates[str(selected_summary["candidate_id"])]
    ordered_summary = {
        name: selected_summary[name] for name in SUMMARY_FIELDS
    }
    return selected, ordered_summary


def write_selected_candidate(
    output_directory: Path,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    selected_summary: Mapping[str, Any],
    pca: VerifiedPCAArtifact,
    scaler: VerifiedScalerArtifact,
    calibration: VerifiedCalibrationArtifact,
    inner_group_metrics_path: Path,
    inner_group_metrics_sha256: str,
    inner_candidate_summary_path: Path,
    inner_candidate_summary_sha256: str,
    inner_fit_audits_path: Path,
    inner_fit_audits_sha256: str,
    outer_family: str,
    git_commit: str,
) -> tuple[Path, str, Mapping[str, Any]]:
    evidence_paths = (
        ("inner_group_metrics", inner_group_metrics_path, inner_group_metrics_sha256),
        ("inner_candidate_summary", inner_candidate_summary_path, inner_candidate_summary_sha256),
        ("inner_fit_audits", inner_fit_audits_path, inner_fit_audits_sha256),
    )
    payload = _manifest_with_self_hash(
        {
            "schema": SELECTED_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_sha256": plan.sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "candidate": _candidate_payload(selected),
            "candidate_count": FROZEN_CANDIDATE_COUNT,
            "inner_selection_summary": selected_summary,
            "inner_evidence": _inner_evidence_records(evidence_paths),
            "final_pca": _pca_binding(pca),
            "final_scaler": _scaler_binding(scaler),
            "final_calibration": _calibration_binding(calibration),
            "outer_raw_features_opened": False,
            "outer_labels_opened": False,
        }
    )
    path = output_directory / "selected_candidate.json"
    return path, _atomic_json(path, payload), payload


def authenticate_selected_candidate(
    path: Path,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    pca: VerifiedPCAArtifact,
    scaler: VerifiedScalerArtifact,
    calibration: VerifiedCalibrationArtifact,
    inner_group_metrics_path: Path,
    inner_group_metrics_sha256: str,
    inner_candidate_summary_path: Path,
    inner_candidate_summary_sha256: str,
    inner_fit_audits_path: Path,
    inner_fit_audits_sha256: str,
    outer_family: str,
    git_commit: str,
    expected_file_sha256: str,
) -> VerifiedSelectedCandidate:
    payload, selected_identity = _read_authenticated_bytes(
        path, expected_sha256=expected_file_sha256
    )
    file_sha = str(selected_identity["sha256"])
    manifest = json.loads(payload.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "selected-candidate root is invalid")
    _authenticate_self_hash(manifest)
    _require(manifest.get("schema") == SELECTED_SCHEMA, "selected schema drifted")
    _require(manifest.get("experiment") == EXPERIMENT, "selected experiment drifted")
    _require(manifest.get("config_sha256") == plan.sha256, "selected config drifted")
    _require(manifest.get("git_commit") == git_commit, "selected Git binding drifted")
    _require(manifest.get("outer_family") == outer_family, "selected outer drifted")
    _require(
        manifest.get("candidate") == _json_safe(_candidate_payload(selected)),
        "selected numerical rule drifted",
    )
    _require(
        int(manifest.get("candidate_count", -1)) == FROZEN_CANDIDATE_COUNT,
        "selected candidate count drifted",
    )
    _require(manifest.get("final_pca") == _json_safe(_pca_binding(pca)), "selected PCA binding drifted")
    _require(manifest.get("final_scaler") == _json_safe(_scaler_binding(scaler)), "selected scaler binding drifted")
    _require(
        manifest.get("final_calibration") == _json_safe(_calibration_binding(calibration)),
        "selected calibration binding drifted",
    )
    _require(
        manifest.get("outer_raw_features_opened") is False
        and manifest.get("outer_labels_opened") is False,
        "selected candidate was not closed before outer access",
    )
    evidence_paths = (
        ("inner_group_metrics", inner_group_metrics_path, inner_group_metrics_sha256),
        ("inner_candidate_summary", inner_candidate_summary_path, inner_candidate_summary_sha256),
        ("inner_fit_audits", inner_fit_audits_path, inner_fit_audits_sha256),
    )
    expected_records = _inner_evidence_records(evidence_paths)
    _require(manifest.get("inner_evidence") == expected_records, "selected evidence binding drifted")
    authenticated_selected, authenticated_summary = _authenticate_inner_selection_evidence(
        plan=plan,
        outer_family=outer_family,
        inner_group_metrics_path=inner_group_metrics_path,
        inner_group_metrics_sha256=inner_group_metrics_sha256,
        inner_candidate_summary_path=inner_candidate_summary_path,
        inner_candidate_summary_sha256=inner_candidate_summary_sha256,
        inner_fit_audits_path=inner_fit_audits_path,
        inner_fit_audits_sha256=inner_fit_audits_sha256,
    )
    _require(authenticated_selected == selected, "selected tie-break drifted")
    stored_summary = manifest.get("inner_selection_summary")
    _require(isinstance(stored_summary, Mapping), "selected summary missing")
    for name in SUMMARY_FIELDS:
        expected = authenticated_summary[name]
        actual = stored_summary.get(name)
        if isinstance(expected, float):
            _require(
                math.isclose(float(actual), expected, rel_tol=1.0e-9, abs_tol=1.0e-9),
                f"selected summary drifted: {name}",
            )
        else:
            _require(actual == expected, f"selected summary drifted: {name}")
    return VerifiedSelectedCandidate(
        path=path,
        file_sha256=file_sha,
        manifest=_deep_freeze(manifest),
        _authentication_seal=_SELECTION_AUTHENTICATION_SEAL,
    )


def _selection_binding(selected: VerifiedSelectedCandidate) -> dict[str, Any]:
    _require(
        selected._authentication_seal is _SELECTION_AUTHENTICATION_SEAL,
        "authenticated selection required",
    )
    return {
        "path": selected.path.name,
        "file_sha256": selected.file_sha256,
        "content_sha256": selected.manifest["content_sha256"],
    }


def build_outer_prediction_arrays(
    caches: Sequence[CacheProjection],
    model: PerScaleNegativeTailModel,
    selected: TailCandidateSpec,
    plan: Plan,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    _require(
        caches
        and all(cache.labels is None and not cache.metadata for cache in caches),
        "outer PCA projection must be label-free",
    )
    _require(selected.representation == "raw_pca161", "outer representation drifted")
    query = _query_cache_batch(
        model, caches, plan, device=device, ks=(selected.k,)
    )[selected.k]
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in PREDICTION_ARRAY_DTYPES
    }
    group_audits: list[dict[str, Any]] = []
    for cache_index, cache in enumerate(caches):
        values = query[cache_index]
        for block_index, block_name in enumerate(BLOCK_NAMES):
            block = np.asarray(cache.block_indices == block_index)
            _require(
                block.any(),
                f"outer {cache.row.dataset}/{cache.row.source_ordinal}/{block_name}: empty group",
            )
            centers = np.asarray(cache.center_indices[block], dtype=np.int64)
            spatial = spatial_calibrated_tail_scores(
                values["tail_anomaly"][block],
                values["calibration_supported"][block],
                centers,
                sigma=selected.sigma,
                grid_shape=plan.grid_shape,
                truncate=plan.gaussian_truncate,
            )
            eligible = spatial.calibration_supported | spatial.imputed
            prediction = candidate_predictions(
                selected, spatial.scores, centers, eligible
            )
            count = int(block.sum())
            parts["dataset"].append(
                np.full(count, cache.row.dataset, dtype=PREDICTION_ARRAY_DTYPES["dataset"])
            )
            parts["source_ordinal"].append(
                np.full(count, cache.row.source_ordinal, dtype=np.int16)
            )
            parts["source_index"].append(
                np.full(count, cache.row.source_index, dtype=np.int64)
            )
            parts["scale_id"].append(cache.scale_ids[block])
            parts["center_seed_index"].append(cache.center_indices[block])
            parts["scale_block_index"].append(cache.block_indices[block])
            parts["assigned_row_index"].append(cache.assigned_row_indices[block])
            parts["raw_negative_distance"].append(values["raw_distance"][block])
            parts["tail_probability"].append(values["tail_probability"][block])
            parts["tail_anomaly"].append(values["tail_anomaly"][block])
            parts["spatial_score"].append(spatial.scores)
            parts["spatial_denominator"].append(spatial.denominator)
            parts["retrieval_supported"].append(values["retrieval_supported"][block])
            parts["calibration_supported"].append(spatial.calibration_supported)
            parts["spatial_imputed"].append(spatial.imputed)
            parts["spatial_unimputable"].append(spatial.unimputable)
            parts["calibration_mode"].append(values["calibration_mode"][block])
            parts["scaler_mode"].append(values["scaler_mode"][block])
            parts["prediction"].append(prediction)
            group_audits.append(
                {
                    "dataset": cache.row.dataset,
                    "source_ordinal": cache.row.source_ordinal,
                    "source_index": cache.row.source_index,
                    "block": block_name,
                    "sample_count": count,
                    "retrieval_supported_count": int(
                        values["retrieval_supported"][block].sum()
                    ),
                    "calibration_supported_count": int(
                        spatial.calibration_supported.sum()
                    ),
                    "imputed_count": int(spatial.imputed.sum()),
                    "unimputable_count": int(spatial.unimputable.sum()),
                    "calibration_mode_counts": {
                        str(mode): int(
                            np.count_nonzero(values["calibration_mode"][block] == mode)
                        )
                        for mode in range(6)
                    },
                    "scaler_mode_counts": {
                        str(mode): int(
                            np.count_nonzero(values["scaler_mode"][block] == mode)
                        )
                        for mode in range(4)
                    },
                    "prediction_count": int(prediction.sum()),
                }
            )
    arrays = {
        name: np.ascontiguousarray(np.concatenate(parts[name]), dtype=dtype)
        for name, dtype in PREDICTION_ARRAY_DTYPES.items()
    }
    return arrays, group_audits


def write_outer_prediction(
    output_directory: Path,
    arrays: Mapping[str, np.ndarray],
    group_audits: Sequence[Mapping[str, Any]],
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    selected_artifact: VerifiedSelectedCandidate,
    pca: VerifiedPCAArtifact,
    scaler: VerifiedScalerArtifact,
    calibration: VerifiedCalibrationArtifact,
    outer_family: str,
    git_commit: str,
) -> tuple[Path, Path, str, str]:
    _require(set(arrays) == set(PREDICTION_ARRAY_DTYPES), "prediction members drifted")
    path = output_directory / "outer_predictions.npz"
    file_sha = _atomic_npz(path, arrays)
    manifest = _manifest_with_self_hash(
        {
            "schema": PREDICTION_MANIFEST_SCHEMA,
            "prediction_schema": PREDICTION_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_sha256": plan.sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "selected_candidate": _candidate_payload(selected),
            "selected_candidate_artifact": _selection_binding(selected_artifact),
            "final_pca": _pca_binding(pca),
            "final_scaler": _scaler_binding(scaler),
            "final_calibration": _calibration_binding(calibration),
            "raw_features_transformed_by_authenticated_pca": True,
            "valid_labels_opened": False,
            "metadata_json_opened": False,
            "prediction_file": {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha,
            },
            "array_count": len(arrays),
            "row_count": len(np.asarray(arrays["prediction"])),
            "arrays": _array_manifest(arrays),
            "group_audits": list(group_audits),
        }
    )
    manifest_path = output_directory / "outer_prediction_manifest.json"
    manifest_sha = _atomic_json(manifest_path, manifest)
    return path, manifest_path, file_sha, manifest_sha


def _require_prediction_replay(
    persisted: Mapping[str, np.ndarray], replayed: Mapping[str, np.ndarray]
) -> None:
    """Authenticate every prediction member before a reference callback runs."""

    _require(set(persisted) == set(replayed), "prediction replay member set drifted")
    for name in sorted(persisted):
        _require(
            canonical_array_sha256(np.asarray(persisted[name]))
            == canonical_array_sha256(np.asarray(replayed[name])),
            f"prediction fresh replay drifted: {name}",
        )


def replay_then_open_references(
    persisted: Mapping[str, np.ndarray],
    replay_factory: Callable[[], Mapping[str, np.ndarray]],
    reference_factory: Callable[[], Any],
) -> tuple[Mapping[str, np.ndarray], Any]:
    """Small explicit gate used by production and synthetic failure tests."""

    replayed = replay_factory()
    _require_prediction_replay(persisted, replayed)
    references = reference_factory()
    return replayed, references


def authenticate_outer_prediction(
    prediction_path: Path,
    manifest_path: Path,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    selected_artifact: VerifiedSelectedCandidate,
    pca: VerifiedPCAArtifact,
    scaler: VerifiedScalerArtifact,
    calibration: VerifiedCalibrationArtifact,
    outer_rows: Sequence[CacheRow],
    outer_family: str,
    git_commit: str,
    device: str,
    expected_manifest_file_sha256: str,
) -> VerifiedOuterPrediction:
    """Authenticate, fresh-reload Raw, reproject PCA, and replay predictions."""

    payload, prediction_manifest_identity = _read_authenticated_bytes(
        manifest_path, expected_sha256=expected_manifest_file_sha256
    )
    manifest_sha = str(prediction_manifest_identity["sha256"])
    manifest = json.loads(payload.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "prediction manifest root is invalid")
    _authenticate_self_hash(manifest)
    _require(manifest.get("schema") == PREDICTION_MANIFEST_SCHEMA, "prediction schema drifted")
    _require(manifest.get("prediction_schema") == PREDICTION_SCHEMA, "prediction artifact drifted")
    _require(manifest.get("experiment") == EXPERIMENT, "prediction experiment drifted")
    _require(manifest.get("config_sha256") == plan.sha256, "prediction config drifted")
    _require(manifest.get("git_commit") == git_commit, "prediction Git binding drifted")
    _require(manifest.get("outer_family") == outer_family, "prediction outer family drifted")
    _require(
        manifest.get("selected_candidate") == _json_safe(_candidate_payload(selected)),
        "prediction candidate drifted",
    )
    _require(
        manifest.get("selected_candidate_artifact")
        == _json_safe(_selection_binding(selected_artifact)),
        "prediction selection binding drifted",
    )
    _require(manifest.get("final_pca") == _json_safe(_pca_binding(pca)), "prediction PCA binding drifted")
    _require(manifest.get("final_scaler") == _json_safe(_scaler_binding(scaler)), "prediction scaler binding drifted")
    _require(
        manifest.get("final_calibration") == _json_safe(_calibration_binding(calibration)),
        "prediction calibration binding drifted",
    )
    _require(
        manifest.get("valid_labels_opened") is False
        and manifest.get("metadata_json_opened") is False,
        "prediction opened outer references early",
    )
    file_record = manifest.get("prediction_file")
    records = manifest.get("arrays")
    _require(
        isinstance(file_record, Mapping) and isinstance(records, Mapping),
        "prediction arrays are missing",
    )
    _require(set(records) == set(PREDICTION_ARRAY_DTYPES), "prediction member set drifted")
    _require(
        int(manifest.get("array_count", -1)) == len(PREDICTION_ARRAY_DTYPES),
        "prediction array count drifted",
    )
    arrays, file_sha = _authenticate_npz_arrays(
        prediction_path, file_record, records
    )
    row_count = int(manifest.get("row_count", -1))
    _require(row_count > 0, "prediction artifact is empty")
    for name, dtype in PREDICTION_ARRAY_DTYPES.items():
        _require(
            arrays[name].dtype == dtype and arrays[name].shape == (row_count,),
            f"prediction dtype/shape drifted: {name}",
        )
    scales = arrays["scale_id"]
    centers = arrays["center_seed_index"]
    blocks = arrays["scale_block_index"]
    assigned = arrays["assigned_row_index"]
    _require(np.all((scales >= 0) & (scales < 2000)), "prediction scale drifted")
    _require(np.all((centers >= 0) & (centers < 64000)), "prediction center drifted")
    _require(np.array_equal(blocks, (scales >= 1000).astype(np.int8)), "prediction block drifted")
    _require(
        np.array_equal(assigned, blocks.astype(np.int64) * 64000 + centers),
        "prediction assigned identity drifted",
    )
    retrieval = arrays["retrieval_supported"]
    calibration_supported = arrays["calibration_supported"]
    imputed = arrays["spatial_imputed"]
    unimputable = arrays["spatial_unimputable"]
    _require(not np.any(calibration_supported & ~retrieval), "calibration exceeds retrieval")
    _require(
        np.array_equal(
            calibration_supported | imputed | unimputable,
            np.ones(row_count, dtype=bool),
        ),
        "prediction support states do not cover rows",
    )
    _require(
        not np.any(
            (calibration_supported & imputed)
            | (calibration_supported & unimputable)
            | (imputed & unimputable)
        ),
        "prediction support states overlap",
    )
    _require(
        np.isfinite(arrays["tail_probability"]).all()
        and np.isfinite(arrays["tail_anomaly"]).all()
        and np.isfinite(arrays["spatial_score"]).all()
        and np.isfinite(arrays["spatial_denominator"]).all(),
        "prediction score is nonfinite",
    )
    expected_datasets = tuple(plan.families[outer_family])
    expected_keys = {
        (dataset, source_ordinal)
        for dataset in expected_datasets
        for source_ordinal in range(4)
    }
    _require(
        {(row.dataset, row.source_ordinal) for row in outer_rows} == expected_keys,
        "outer cache scope is incomplete",
    )
    # Fresh replay is label-free by construction: load_raw_projection(False)
    # cannot request valid_labels or metadata_json.
    replayed_outer = tuple(
        project_cache_rows(plan, outer_rows, pca.model, include_labels=False)
    )
    _require(
        sum(cache.count for cache in replayed_outer)
        == plan.expected_family_rows[outer_family],
        "outer family Raw row population drifted",
    )
    replayed_arrays, replayed_audits = build_outer_prediction_arrays(
        replayed_outer,
        calibration.model,
        selected,
        plan,
        device=device,
    )
    _require_prediction_replay(arrays, replayed_arrays)
    _require(
        _json_safe(replayed_audits) == manifest.get("group_audits"),
        "prediction group audit fresh replay drifted",
    )
    return VerifiedOuterPrediction(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_sha,
        prediction_file_sha256=file_sha,
        manifest=_deep_freeze(manifest),
        arrays=_deep_freeze(arrays),
        replayed_outer=replayed_outer,
        _authentication_seal=_PREDICTION_AUTHENTICATION_SEAL,
    )


def load_outer_references_after_prediction(
    plan: Plan, verified: VerifiedOuterPrediction
) -> tuple[dict[tuple[str, int, int], tuple[np.ndarray, Mapping[str, Any]]], list[dict[str, Any]]]:
    """The sole outer-label opening function, guarded by fresh replay seal."""

    _require(
        verified._authentication_seal is _PREDICTION_AUTHENTICATION_SEAL,
        "outer references require authenticated fresh prediction replay",
    )
    references: dict[
        tuple[str, int, int], tuple[np.ndarray, Mapping[str, Any]]
    ] = {}
    audits: list[dict[str, Any]] = []
    arrays = verified.arrays
    for projection in verified.replayed_outer:
        labels, metadata = load_reference_for_projection(plan, projection)
        row = projection.row
        row_selected = (
            (arrays["dataset"] == row.dataset)
            & (arrays["source_ordinal"] == row.source_ordinal)
            & (arrays["source_index"] == row.source_index)
        )
        expected_order = np.concatenate(
            [np.flatnonzero(projection.block_indices == block) for block in (0, 1)]
        )
        _require(int(row_selected.sum()) == projection.count, "reference row count drifted")
        _require(
            np.array_equal(arrays["scale_id"][row_selected], projection.scale_ids[expected_order])
            and np.array_equal(
                arrays["center_seed_index"][row_selected],
                projection.center_indices[expected_order],
            )
            and np.array_equal(
                arrays["assigned_row_index"][row_selected],
                projection.assigned_row_indices[expected_order],
            ),
            "reference identity drifted",
        )
        key = (row.dataset, row.source_ordinal, row.source_index)
        _require(key not in references, "duplicate outer reference")
        ordered_labels = np.ascontiguousarray(labels[expected_order])
        ordered_labels.setflags(write=False)
        references[key] = (ordered_labels, metadata)
        audits.append(
            {
                "dataset": row.dataset,
                "source_ordinal": row.source_ordinal,
                "source_index": row.source_index,
                "cache_path": str(row.path),
                "cache_file_sha256": row.sha256,
                "metadata_schema": metadata.get("schema"),
                "label_member_opened_after_fresh_prediction_replay": True,
                "prediction_manifest_file_sha256": verified.manifest_file_sha256,
                "prediction_file_sha256": verified.prediction_file_sha256,
            }
        )
    return references, audits


def evaluate_outer_prediction(
    verified: VerifiedOuterPrediction,
    references: Mapping[
        tuple[str, int, int], tuple[np.ndarray, Mapping[str, Any]]
    ],
    selected: TailCandidateSpec,
    *,
    outer_family: str,
) -> list[dict[str, Any]]:
    arrays = verified.arrays
    rows: list[dict[str, Any]] = []
    for projection in verified.replayed_outer:
        row = projection.row
        labels, metadata = references[
            (row.dataset, row.source_ordinal, row.source_index)
        ]
        row_selected = (
            (arrays["dataset"] == row.dataset)
            & (arrays["source_ordinal"] == row.source_ordinal)
            & (arrays["source_index"] == row.source_index)
        )
        indices_for_row = np.flatnonzero(row_selected)
        _require(len(indices_for_row) == len(labels), "outer label length drifted")
        for block_index, block_name in enumerate(BLOCK_NAMES):
            within = arrays["scale_block_index"][indices_for_row] == block_index
            indices = indices_for_row[within]
            block_labels = labels[within]
            proxy = CacheProjection(
                row=row,
                fmt_features=np.empty((len(indices), RAW_OUTPUT_WIDTH), dtype=np.float32),
                scale_ids=arrays["scale_id"][indices],
                center_indices=arrays["center_seed_index"][indices],
                block_indices=arrays["scale_block_index"][indices],
                assigned_row_indices=arrays["assigned_row_index"][indices],
                labels=block_labels,
                metadata=metadata,
            )
            rows.append(
                _metric_row(
                    outer_family=outer_family,
                    inner_family="outer_evaluation_only",
                    cache=proxy,
                    block_name=block_name,
                    candidate=selected,
                    labels=block_labels,
                    scores=arrays["spatial_score"][indices],
                    predictions=arrays["prediction"][indices],
                    retrieval_supported=arrays["retrieval_supported"][indices],
                    calibration_supported=arrays["calibration_supported"][indices],
                    imputed=arrays["spatial_imputed"][indices],
                    unimputable=arrays["spatial_unimputable"][indices],
                    calibration_modes=arrays["calibration_mode"][indices],
                    scaler_modes=arrays["scaler_mode"][indices],
                )
            )
    return rows


def _outer_summary(
    rows: Sequence[Mapping[str, Any]], outer_family: str
) -> dict[str, Any]:
    _require(rows, "outer evaluation produced no groups")
    summary: dict[str, Any] = {
        "schema": "pathline_template_matching.raw_pca_negative_metric_outer_summary.v1",
        "experiment": EXPERIMENT,
        "outer_family": outer_family,
        "group_count": len(rows),
    }
    for name in (
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
    ):
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[name] = float(np.mean(finite)) if len(finite) else float("nan")
    for name in (
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
    ):
        summary[name] = int(sum(int(row[name]) for row in rows))
    return summary


def result_artifact_names(plan: Plan) -> tuple[str, ...]:
    names = tuple(
        name
        for name in plan.required_fold_files
        if name not in {"result_manifest.json", "RUN_COMPLETE.json"}
    )
    _require(len(names) == 15 and len(set(names)) == 15, "result artifact count drifted")
    return names


def run(
    config_path: str | Path,
    outer_family: str,
    output_dir: str | Path,
    *,
    device: str,
    expected_config_sha256: str | None = EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    plan = load_plan(config_path)
    _require(outer_family in plan.family_order, f"unknown outer family: {outer_family}")
    if expected_config_sha256 is not None:
        _require(plan.sha256 == expected_config_sha256, "config SHA-256 mismatch")
    git_commit, dirty = _git_identity()
    _require(not dirty, "Ibex numerical run requires a clean committed Git worktree")
    _configure_execution(device)
    destination = Path(output_dir).resolve()
    _require(not destination.exists(), f"immutable output directory exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    print(
        f"[{_utc_now()}] {EXPERIMENT} outer={outer_family} commit={git_commit}",
        flush=True,
    )

    cache_rows, input_manifest_identity = load_cache_rows(plan)
    nonouter_rows = [row for row in cache_rows if row.family != outer_family]
    outer_rows = [row for row in cache_rows if row.family == outer_family]
    _require(nonouter_rows and outer_rows, "outer split produced an empty side")

    inner_rows, candidates, inner_fit_audits = _inner_metric_rows(
        plan, nonouter_rows, outer_family, device=device
    )
    summaries, in_memory_selected, in_memory_summary = _aggregate_and_select(
        inner_rows, candidates
    )
    _require(tuple(inner_rows[0]) == METRIC_FIELDS, "inner metric fields drifted")
    _require(tuple(summaries[0]) == SUMMARY_FIELDS, "inner summary fields drifted")
    inner_metrics_path = destination / "inner_group_metrics.csv"
    inner_summary_path = destination / "inner_candidate_summary.csv"
    inner_fit_path = destination / "inner_fit_audits.json"
    inner_metrics_sha = _atomic_csv(inner_metrics_path, METRIC_FIELDS, inner_rows)
    inner_summary_sha = _atomic_csv(inner_summary_path, SUMMARY_FIELDS, summaries)
    inner_fit_sha = _atomic_json(
        inner_fit_path,
        _manifest_with_self_hash(
            {
                "schema": INNER_AUDIT_SCHEMA,
                "experiment": EXPERIMENT,
                "outer_family": outer_family,
                "fit_count": len(inner_fit_audits),
                "fits": inner_fit_audits,
            }
        ),
    )
    selected, selected_summary = _authenticate_inner_selection_evidence(
        plan=plan,
        outer_family=outer_family,
        inner_group_metrics_path=inner_metrics_path,
        inner_group_metrics_sha256=inner_metrics_sha,
        inner_candidate_summary_path=inner_summary_path,
        inner_candidate_summary_sha256=inner_summary_sha,
        inner_fit_audits_path=inner_fit_path,
        inner_fit_audits_sha256=inner_fit_sha,
    )
    _require(selected == in_memory_selected, "persisted selection drifted")
    _require(
        all(
            (
                math.isclose(
                    float(selected_summary[name]),
                    float(in_memory_summary[name]),
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                )
                if isinstance(selected_summary[name], float)
                else selected_summary[name] == in_memory_summary[name]
            )
            for name in SUMMARY_FIELDS
        ),
        "persisted selected summary drifted",
    )
    del inner_rows, candidates, summaries, inner_fit_audits
    gc.collect()

    fit_families = tuple(family for family in plan.family_order if family != outer_family)
    # No outer cache has been opened before this final fit and its publication.
    final_fit = fit_pca_from_cache_rows(
        plan,
        nonouter_rows,
        outer_family=outer_family,
        held_out="final",
    )
    pca_path, pca_manifest_path, _, pca_manifest_sha = write_final_pca_artifact(
        destination,
        final_fit,
        plan=plan,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
    )
    pca = authenticate_and_rebuild_final_pca(
        pca_path,
        pca_manifest_path,
        plan=plan,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=pca_manifest_sha,
    )
    del final_fit

    final_fit_caches = project_cache_rows(
        plan, nonouter_rows, pca.model, include_labels=True
    )
    final_model = _fit_tail_model(
        final_fit_caches, plan, device=device, ks=(selected.k,)
    )
    scaler_path, scaler_manifest_path, _, scaler_manifest_sha = (
        write_final_scaler_artifact(
            destination,
            final_model,
            plan=plan,
            selected=selected,
            pca=pca,
            outer_family=outer_family,
            fit_families=fit_families,
            git_commit=git_commit,
        )
    )
    scaler = authenticate_and_rebuild_final_scaler(
        scaler_path,
        scaler_manifest_path,
        plan=plan,
        selected=selected,
        pca=pca,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=scaler_manifest_sha,
    )
    calibration_path, calibration_manifest_path, _, calibration_manifest_sha = (
        write_final_calibration_artifact(
            destination,
            final_model,
            plan=plan,
            selected=selected,
            pca=pca,
            scaler=scaler,
            outer_family=outer_family,
            fit_families=fit_families,
            git_commit=git_commit,
        )
    )
    calibration = authenticate_and_rebuild_final_calibration(
        calibration_path,
        calibration_manifest_path,
        plan=plan,
        selected=selected,
        pca=pca,
        scaler=scaler,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=calibration_manifest_sha,
    )
    del final_model, final_fit_caches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    selected_path, selected_sha, selected_payload = write_selected_candidate(
        destination,
        plan=plan,
        selected=selected,
        selected_summary=selected_summary,
        pca=pca,
        scaler=scaler,
        calibration=calibration,
        inner_group_metrics_path=inner_metrics_path,
        inner_group_metrics_sha256=inner_metrics_sha,
        inner_candidate_summary_path=inner_summary_path,
        inner_candidate_summary_sha256=inner_summary_sha,
        inner_fit_audits_path=inner_fit_path,
        inner_fit_audits_sha256=inner_fit_sha,
        outer_family=outer_family,
        git_commit=git_commit,
    )
    selected_artifact = authenticate_selected_candidate(
        selected_path,
        plan=plan,
        selected=selected,
        pca=pca,
        scaler=scaler,
        calibration=calibration,
        inner_group_metrics_path=inner_metrics_path,
        inner_group_metrics_sha256=inner_metrics_sha,
        inner_candidate_summary_path=inner_summary_path,
        inner_candidate_summary_sha256=inner_summary_sha,
        inner_fit_audits_path=inner_fit_path,
        inner_fit_audits_sha256=inner_fit_sha,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=selected_sha,
    )

    # This is the first outer-cache member access in the fold.  It is both
    # label-free and metadata-free.
    outer_caches = project_cache_rows(
        plan, outer_rows, pca.model, include_labels=False
    )
    prediction_arrays, group_audits = build_outer_prediction_arrays(
        outer_caches,
        calibration.model,
        selected,
        plan,
        device=device,
    )
    prediction_path, prediction_manifest_path, prediction_sha, prediction_manifest_sha = (
        write_outer_prediction(
            destination,
            prediction_arrays,
            group_audits,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            pca=pca,
            scaler=scaler,
            calibration=calibration,
            outer_family=outer_family,
            git_commit=git_commit,
        )
    )
    del prediction_arrays, outer_caches
    verified_prediction = authenticate_outer_prediction(
        prediction_path,
        prediction_manifest_path,
        plan=plan,
        selected=selected,
        selected_artifact=selected_artifact,
        pca=pca,
        scaler=scaler,
        calibration=calibration,
        outer_rows=outer_rows,
        outer_family=outer_family,
        git_commit=git_commit,
        device=device,
        expected_manifest_file_sha256=prediction_manifest_sha,
    )
    references, reference_rows = load_outer_references_after_prediction(
        plan, verified_prediction
    )
    outer_metric_rows = evaluate_outer_prediction(
        verified_prediction,
        references,
        selected,
        outer_family=outer_family,
    )
    _require(tuple(outer_metric_rows[0]) == METRIC_FIELDS, "outer metric fields drifted")
    outer_metrics_sha = _atomic_csv(
        destination / "outer_group_metrics.csv", METRIC_FIELDS, outer_metric_rows
    )
    outer_summary = _outer_summary(outer_metric_rows, outer_family)
    outer_summary_sha = _atomic_json(
        destination / "outer_summary.json", _manifest_with_self_hash(outer_summary)
    )
    reference_manifest = _manifest_with_self_hash(
        {
            "schema": REFERENCE_AUDIT_SCHEMA,
            "experiment": EXPERIMENT,
            "outer_family": outer_family,
            "first_open_phase": "after_authenticated_fresh_outer_prediction_replay",
            "prediction_manifest_file_sha256": prediction_manifest_sha,
            "prediction_file_sha256": prediction_sha,
            "row_count": len(reference_rows),
            "rows": reference_rows,
        }
    )
    reference_sha = _atomic_json(
        destination / "outer_reference_access_audit.json", reference_manifest
    )

    artifact_names = result_artifact_names(plan)
    result_manifest = _manifest_with_self_hash(
        {
            "schema": RESULT_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "completed",
            "completed_utc": _utc_now(),
            "git_commit": git_commit,
            "config_path": str(plan.path),
            "config_sha256": plan.sha256,
            "input_manifest": input_manifest_identity,
            "outer_family": outer_family,
            "selected_candidate": _candidate_payload(selected),
            "selected_candidate_file_sha256": selected_sha,
            "selected_candidate_content_sha256": selected_payload["content_sha256"],
            "final_pca_manifest_file_sha256": pca.manifest_file_sha256,
            "final_pca_file_sha256": pca.pca_file_sha256,
            "final_scaler_manifest_file_sha256": scaler.manifest_file_sha256,
            "final_scaler_file_sha256": scaler.scaler_file_sha256,
            "final_calibration_manifest_file_sha256": calibration.manifest_file_sha256,
            "final_calibration_file_sha256": calibration.calibration_file_sha256,
            "prediction_manifest_file_sha256": prediction_manifest_sha,
            "prediction_file_sha256": prediction_sha,
            "inner_group_metrics_file_sha256": inner_metrics_sha,
            "inner_candidate_summary_file_sha256": inner_summary_sha,
            "inner_fit_audits_file_sha256": inner_fit_sha,
            "outer_group_metrics_file_sha256": outer_metrics_sha,
            "outer_summary_file_sha256": outer_summary_sha,
            "outer_reference_access_audit_file_sha256": reference_sha,
            "environment": inherited._environment_audit(device),
            "artifact_count": len(artifact_names),
            "artifacts": {
                name: {
                    "size_bytes": (destination / name).stat().st_size,
                    "sha256": sha256_file(destination / name),
                }
                for name in artifact_names
            },
        }
    )
    result_path = destination / "result_manifest.json"
    result_sha = _atomic_json(result_path, result_manifest)
    completion = _manifest_with_self_hash(
        {
            "schema": COMPLETE_SCHEMA,
            "experiment": EXPERIMENT,
            "outer_family": outer_family,
            "git_commit": git_commit,
            "config_sha256": plan.sha256,
            "result_manifest_file": result_path.name,
            "result_manifest_file_sha256": result_sha,
            "result_manifest_content_sha256": result_manifest["content_sha256"],
            "completed_utc": _utc_now(),
        }
    )
    _atomic_json(destination / "RUN_COMPLETE.json", completion)
    _require(
        set(path.name for path in destination.iterdir())
        == set(plan.required_fold_files),
        "completed 17-file fold contract drifted",
    )
    print(
        f"[{_utc_now()}] completed outer={outer_family} "
        f"F1={outer_summary['f1']:.6f}",
        flush=True,
    )
    return result_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "Verify_RawPCANegativeMetric_1.1.yaml"),
    )
    parser.add_argument("--outer-family", required=True, choices=FAMILY_ORDER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--expected-config-sha256", default=EXPECTED_CONFIG_SHA256
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    run(
        arguments.config,
        arguments.outer_family,
        arguments.output_dir,
        device=arguments.device,
        expected_config_sha256=arguments.expected_config_sha256,
    )


if __name__ == "__main__":
    main()

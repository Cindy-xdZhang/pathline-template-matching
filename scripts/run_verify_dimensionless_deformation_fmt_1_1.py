#!/usr/bin/env python3
"""Production fold runner for Verify_DimensionlessDeformationFMT_1.1.

The experiment inherits the complete PerScale nested-family scorer and its
15-file fold transaction, but it owns the cache access boundary.  Cache
``fmt_features`` are forbidden: every feature is rebuilt from authenticated
Raw672 rows by the frozen row-local dimensionless transform and unchanged FMT.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import MappingProxyType
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.dimensionless_deformation_fmt import (  # noqa: E402
    FROZEN_PRIMITIVE_ORDER,
    PARENT_DESCRIPTOR_ID,
    PARENT_REPRESENTATION_INDEX_SETS,
    RAW_INPUT_WIDTH,
    REPRESENTATION_NAMES,
    encode_dimensionless_deformation_fmt,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    sha256_file,
)
from scripts import run_verify_per_scale_negative_metric_1_1 as inherited  # noqa: E402
from scripts.run_verify_scale_conditioned_retrieval_1_1 import (  # noqa: E402
    CacheRow,
    _configure_execution,
    _git_identity,
    _json_safe,
    _require,
)


EXPERIMENT = "Verify_DimensionlessDeformationFMT_1.1"
# Capture the parent identity before ``dimensionless_parent_runtime``
# intentionally rebinds ``inherited.EXPERIMENT``.  ``load_plan`` is itself
# called through the rebound parent runner, so consulting that mutable global
# here makes the valid frozen child config reject itself in production.
PARENT_EXPERIMENT = inherited.EXPERIMENT
EXPECTED_CONFIG_SHA256 = (
    "c689b1d265bbc39327b2ed4147e8ffb22450dcd26f87b7c19ceae346c9ecfe18"
)
EXPECTED_PARENT_CONFIG_SHA256 = (
    "b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79"
)
EXPECTED_CORE_SHA256 = (
    "5fc4acb47c52c6505737e661cac7f8f503c429c5d88910992655e83cdc53a649"
)
CONFIG_PATH = ROOT / "config" / "Verify_DimensionlessDeformationFMT_1.1.yaml"
PARENT_CONFIG_PATH = ROOT / "config" / "Verify_PerScaleNegativeMetric_1.1.yaml"
CORE_PATH = ROOT / "src" / "pathline_template_matching" / "dimensionless_deformation_fmt.py"
ENCODE_CHUNK_ROWS = 4096
FAMILY_ORDER = inherited.FAMILY_ORDER
REPRESENTATIONS = REPRESENTATION_NAMES
K_VALUES = inherited.K_VALUES
SIGMAS = inherited.SIGMAS
TAIL_THRESHOLDS = inherited.TAIL_THRESHOLDS
GRID_SHAPE = inherited.GRID_SHAPE
BLOCK_NAMES = inherited.BLOCK_NAMES
FROZEN_CANDIDATE_COUNT = 3060
DEFAULT_OUTPUT_ROOT = Path(
    "/ibex/user/zhanx0o/pathline-template-matching/"
    "Verify_DimensionlessDeformationFMT_1.1"
)

SCALER_ARTIFACT_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_per_scale_metric.v1"
)
SCALER_MANIFEST_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_per_scale_metric_manifest.v1"
)
CALIBRATION_ARTIFACT_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_tail_calibration_artifact.v1"
)
CALIBRATION_MANIFEST_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_tail_calibration_manifest.v1"
)
SELECTED_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_selected_candidate.v1"
)
PREDICTION_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_outer_prediction.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_outer_prediction_manifest.v1"
)
OUTER_SUMMARY_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_outer_summary.v1"
)
REFERENCE_AUDIT_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_outer_reference_access.v1"
)
RESULT_SCHEMA = "pathline_template_matching.dimensionless_deformation_fmt_result.v1"
COMPLETE_SCHEMA = (
    "pathline_template_matching.dimensionless_deformation_fmt_run_complete.v1"
)
METHOD_BINDING_KEY = "dimensionless_deformation_method"
REQUIRED_FOLD_FILES = (
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
PREDICTION_ARRAY_DTYPES = inherited.PREDICTION_ARRAY_DTYPES
_INHERITED_LOAD_PLAN = inherited.load_plan
_INHERITED_OUTER_SUMMARY = inherited._outer_summary
_PARENT_RUNTIME_LOCK = threading.Lock()


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
    descriptor_id: str
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
    output_root: Path
    required_fold_files: tuple[str, ...]
    parent_experiment_config_path: Path
    parent_experiment_config_sha256: str
    core_path: Path
    core_sha256: str
    encode_chunk_rows: int


@dataclass
class DimensionlessCacheProjection:
    """One Raw-only cache projection retaining only rebuilt FMT161."""

    row: CacheRow
    fmt_features: np.ndarray
    scale_ids: np.ndarray
    center_indices: np.ndarray
    block_indices: np.ndarray
    assigned_row_indices: np.ndarray
    labels: np.ndarray | None
    metadata: Mapping[str, Any]
    opened_members: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.scale_ids)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        output = np.frombuffer(np.asarray(value).tobytes(order="C"), dtype=value.dtype)
        output = output.reshape(value.shape)
        output.setflags(write=False)
        return output
    return value


def _lower_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _candidate_count(plan: Plan) -> int:
    return len(plan.representations) * len(plan.ks) * len(plan.sigmas) * (
        1 + len(plan.thresholds)
    )


def load_plan(config_path: str | Path = CONFIG_PATH) -> Plan:
    """Authenticate both frozen configs and derive only inherited operations."""

    path = Path(config_path).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    _require(digest == EXPECTED_CONFIG_SHA256, "frozen config SHA-256 drifted")
    raw = yaml.safe_load(payload.decode("utf-8"))
    _require(isinstance(raw, Mapping), "dimensionless config root is invalid")
    _require(raw.get("experiment") == EXPERIMENT, "experiment identity drifted")
    _require(
        raw.get("phase") == "exposed_train_only_nested_family_validation",
        "phase drifted",
    )
    _require(
        raw.get("status") == "frozen_pre_run_not_implemented",
        "immutable freeze status drifted",
    )
    freeze = raw.get("freeze_provenance")
    _require(isinstance(freeze, Mapping), "freeze provenance is missing")
    _require(
        freeze.get("parent_experiment") == PARENT_EXPERIMENT
        and freeze.get("parent_config_sha256") == EXPECTED_PARENT_CONFIG_SHA256
        and freeze.get("historical_status_is_immutable") is True,
        "parent freeze identity drifted",
    )

    parent_path = PARENT_CONFIG_PATH.resolve()
    _require(
        sha256_file(parent_path) == EXPECTED_PARENT_CONFIG_SHA256,
        "parent PerScale config SHA-256 drifted",
    )
    parent = _INHERITED_LOAD_PLAN(parent_path)
    _require(
        parent.sha256 == EXPECTED_PARENT_CONFIG_SHA256,
        "authenticated parent config identity drifted",
    )
    _require(
        parent.required_fold_files == REQUIRED_FOLD_FILES,
        "parent 15-file fold transaction drifted",
    )

    evidence = raw.get("evidence_scope")
    _require(isinstance(evidence, Mapping), "evidence scope is missing")
    manifest = evidence.get("input_manifest")
    _require(isinstance(manifest, Mapping), "input manifest identity is missing")
    _require(
        evidence.get("allowed_inputs")
        == "mainExp_TemplateMatching_3.1_train_caches_only"
        and evidence.get("forbidden_datasets") == ["tangaroa", "smokeBuoyancy"]
        and evidence.get("formal_confirmation") is False,
        "input evidence scope drifted",
    )
    _require(
        int(manifest.get("row_count", -1)) == 32
        and manifest.get("test_dataset_access") is False
        and _lower_hex(manifest.get("sha256"))
        and _lower_hex(manifest.get("rows_content_sha256")),
        "input manifest contract drifted",
    )
    _require(
        Path(str(manifest["path"])) == parent.manifest_path
        and int(manifest["size_bytes"]) == parent.manifest_size
        and manifest["sha256"] == parent.manifest_sha256
        and manifest["rows_content_sha256"] == parent.manifest_rows_sha256,
        "dimensionless and parent input manifests differ",
    )

    families_raw = raw.get("families")
    split = raw.get("nested_split")
    _require(
        isinstance(families_raw, Mapping) and isinstance(split, Mapping),
        "complete-family split is missing",
    )
    families = {
        str(key): tuple(str(item) for item in values)
        for key, values in families_raw.items()
    }
    _require(
        tuple(families) == FAMILY_ORDER
        and families == dict(parent.families)
        and tuple(split.get("outer_order", ())) == FAMILY_ORDER
        and tuple(split.get("inner_order", ())) == FAMILY_ORDER,
        "complete-family split drifted",
    )
    _require(
        split.get("outer_features_available_to_selection") is False
        and split.get("outer_labels_available_to_selection") is False,
        "outer selection gate drifted",
    )

    primitive = raw.get("primitive_contract")
    descriptor = raw.get("descriptor")
    inherited_method = raw.get("inherited_negative_metric_and_calibration")
    grid = raw.get("candidate_grid")
    gates = raw.get("access_gates")
    _require(
        all(isinstance(item, Mapping) for item in (primitive, descriptor, inherited_method, grid, gates)),
        "numerical contract is incomplete",
    )
    assert isinstance(primitive, Mapping) and isinstance(descriptor, Mapping)
    assert isinstance(inherited_method, Mapping) and isinstance(grid, Mapping)
    assert isinstance(gates, Mapping)
    _require(
        primitive.get("input_member") == "raw_features"
        and primitive.get("input_dtype") == "float32"
        and primitive.get("input_shape") == ["N", 672]
        and tuple(primitive.get("primitive_order", ())) == FROZEN_PRIMITIVE_ORDER,
        "Raw672 primitive contract drifted",
    )
    _require(
        descriptor.get("parent_descriptor_id") == PARENT_DESCRIPTOR_ID
        and int(descriptor.get("full_width", -1)) == 161
        and tuple(descriptor.get("representations", ())) == REPRESENTATIONS
        and descriptor.get("parent_coordinate_index_sets_are_unchanged") is True,
        "dimensionless descriptor contract drifted",
    )
    _require(
        inherited_method.get("source") == PARENT_EXPERIMENT
        and inherited_method.get("exact_same_scale_retrieval") is True
        and float(inherited_method.get("variance_shrinkage_lambda", -1)) == 64.0
        and inherited_method.get("query_rank") == "forbidden",
        "inherited PerScale method drifted",
    )
    ks = tuple(int(value) for value in grid.get("k", ()))
    sigmas = tuple(float(value) for value in grid.get("spatial_sigma_grid_cells", ()))
    threshold = grid.get("calibrated_tail_threshold")
    _require(isinstance(threshold, Mapping), "threshold grid is missing")
    assert isinstance(threshold, Mapping)
    thresholds = tuple(
        round(float(threshold["start"]) + float(threshold["step"]) * index, 2)
        for index in range(
            int(
                round(
                    (float(threshold["stop_inclusive"]) - float(threshold["start"]))
                    / float(threshold["step"])
                )
            )
            + 1
        )
    )
    _require(
        ks == K_VALUES
        and sigmas == SIGMAS
        and thresholds == TAIL_THRESHOLDS
        and grid.get("fixed_top_fraction") == [0.05]
        and int(grid.get("candidate_count", -1)) == FROZEN_CANDIDATE_COUNT,
        "3060-candidate grid drifted",
    )
    _require(
        gates.get("artifact_overwrite") == "forbidden"
        and gates.get("publish_semantics")
        == "same_directory_fsync_then_hard_link_no_replace_then_parent_fsync",
        "publication contract drifted",
    )

    core_sha = sha256_file(CORE_PATH)
    _require(core_sha == EXPECTED_CORE_SHA256, "dimensionless core SHA-256 drifted")
    merged_raw = dict(raw)
    merged_raw["parent_identity"] = dict(parent.raw["parent_identity"])
    plan = Plan(
        path=path,
        sha256=digest,
        raw=_deep_freeze(merged_raw),
        family_order=FAMILY_ORDER,
        families=_deep_freeze(families),
        dataset_to_family=parent.dataset_to_family,
        manifest_path=parent.manifest_path,
        manifest_size=parent.manifest_size,
        manifest_sha256=parent.manifest_sha256,
        manifest_schema=parent.manifest_schema,
        manifest_rows_sha256=parent.manifest_rows_sha256,
        cache_commit=parent.cache_commit,
        parent_config_sha256=parent.parent_config_sha256,
        descriptor_id=PARENT_DESCRIPTOR_ID,
        cache_schema=parent.cache_schema,
        representations=REPRESENTATIONS,
        ks=ks,
        sigmas=sigmas,
        thresholds=thresholds,
        fixed_top_fraction=0.05,
        grid_shape=GRID_SHAPE,
        gaussian_truncate=parent.gaussian_truncate,
        query_chunk_size=parent.query_chunk_size,
        library_chunk_size=parent.library_chunk_size,
        shrinkage_lambda=parent.shrinkage_lambda,
        output_root=DEFAULT_OUTPUT_ROOT,
        required_fold_files=REQUIRED_FOLD_FILES,
        parent_experiment_config_path=parent_path,
        parent_experiment_config_sha256=parent.sha256,
        core_path=CORE_PATH.resolve(),
        core_sha256=core_sha,
        encode_chunk_rows=ENCODE_CHUNK_ROWS,
    )
    _require(
        _candidate_count(plan) == FROZEN_CANDIDATE_COUNT,
        "candidate count did not reconstruct to 3060",
    )
    _require(
        len(plan.required_fold_files) == 15
        and len(set(plan.required_fold_files)) == 15,
        "inherited 15-file fold contract drifted",
    )
    return plan


def _fsync_parent_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_verified_temporary(temporary: Path, destination: Path) -> None:
    """Commit one authenticated artifact with hard-link no-replace semantics."""

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
        _require(
            after.st_size == len(payload) and actual_sha == expected_sha,
            "temporary artifact authentication failed",
        )
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
    _require(
        json.loads(payload.decode("utf-8")) == safe_value,
        "JSON serialization failed exact replay",
    )
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
    _require(replay == normalized, "CSV serialization failed exact replay")
    return _atomic_bytes(path, payload)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
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
            _require(
                set(archive.files) == set(expected),
                "temporary NPZ member set drifted",
            )
            for name, source in expected.items():
                replay = np.asarray(archive[name])
                _require(
                    replay.dtype == source.dtype
                    and replay.shape == source.shape
                    and np.array_equal(replay, source),
                    f"temporary NPZ replay drifted: {name}",
                )
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
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_OpenFileIdentity":
        return cls(
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
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
    expected_size: int,
    expected_sha256: str,
) -> Iterator[_AuthenticatedOpenFile]:
    """Hash and consume one no-follow descriptor tied to the final path inode."""

    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    path_before = _OpenFileIdentity.from_stat(source.stat(follow_symlinks=False))
    descriptor = os.open(os.fspath(source), flags)
    stream = os.fdopen(descriptor, "rb")
    try:
        descriptor_before = _OpenFileIdentity.from_stat(os.fstat(stream.fileno()))
        _require(
            descriptor_before == path_before,
            f"cache path changed before descriptor open: {source}",
        )
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
        descriptor_after = _OpenFileIdentity.from_stat(os.fstat(stream.fileno()))
        path_after = _OpenFileIdentity.from_stat(source.stat(follow_symlinks=False))
        _require(
            descriptor_after == descriptor_before == path_after,
            f"cache identity changed during authentication: {source}",
        )
        actual_sha = digest.hexdigest()
        _require(
            byte_count == expected_size == descriptor_after.size_bytes,
            f"cache size drifted: {source}",
        )
        _require(actual_sha == expected_sha256, f"cache SHA-256 drifted: {source}")
        stream.seek(0)
        yield _AuthenticatedOpenFile(stream, byte_count, actual_sha)
        descriptor_final = _OpenFileIdentity.from_stat(os.fstat(stream.fileno()))
        path_final = _OpenFileIdentity.from_stat(source.stat(follow_symlinks=False))
        _require(
            descriptor_final == descriptor_before == path_final,
            f"cache identity changed while consuming members: {source}",
        )
    finally:
        stream.close()


def _open_cache_members(row: CacheRow, names: Sequence[str]) -> dict[str, np.ndarray]:
    """Open an explicit Raw-only member set; cached FMT is unconditionally banned."""

    requested = tuple(str(name) for name in names)
    _require(len(requested) == len(set(requested)), "duplicate cache member request")
    _require("fmt_features" not in requested, "parent fmt_features member is forbidden")
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
        row.path,
        expected_size=row.size_bytes,
        expected_sha256=row.sha256,
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


def _cache_metadata(plan: Plan, row: CacheRow, scalar: np.ndarray) -> Mapping[str, Any]:
    _require(scalar.shape == (), f"{row.path}: metadata_json is not scalar")
    metadata = json.loads(str(scalar.item()))
    _require(isinstance(metadata, Mapping), f"{row.path}: metadata root is invalid")
    _require(metadata.get("schema") == plan.cache_schema, f"{row.path}: schema drifted")
    _require(
        metadata.get("experiment") == "mainExp_TemplateMatching_3.1"
        and metadata.get("split") == "train",
        f"{row.path}: parent experiment/split drifted",
    )
    _require(
        metadata.get("dataset") == row.dataset
        and metadata.get("physical_family") == row.family
        and int(metadata.get("source_ordinal", -1)) == row.source_ordinal
        and int(metadata.get("source_index", -1)) == row.source_index,
        f"{row.path}: cache identity drifted",
    )
    _require(
        metadata.get("config_sha256") == plan.parent_config_sha256
        and metadata.get("cache_builder_git_commit") == plan.cache_commit,
        f"{row.path}: parent cache provenance drifted",
    )
    _require(int(metadata.get("valid_count", -1)) > 0, f"{row.path}: invalid count")
    _require(
        isinstance(metadata.get("array_sha256"), Mapping),
        f"{row.path}: array hashes are missing",
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
    _require(count > 0, f"{row.path}: empty Raw projection")
    _require(
        raw_features.dtype == np.dtype(np.float32)
        and raw_features.shape == (count, RAW_INPUT_WIDTH)
        and np.isfinite(raw_features).all(),
        f"{row.path}: Raw672 dtype/shape/finite contract drifted",
    )
    expected = (
        (scale_ids, np.dtype(np.int32)),
        (center_indices, np.dtype(np.int64)),
        (block_indices, np.dtype(np.int8)),
        (assigned_indices, np.dtype(np.int64)),
    )
    _require(
        all(values.dtype == dtype and values.shape == (count,) for values, dtype in expected),
        f"{row.path}: identity dtype/shape drifted",
    )
    _require(np.all((scale_ids >= 0) & (scale_ids < 2000)), f"{row.path}: scale drifted")
    _require(
        np.all((center_indices >= 0) & (center_indices < 64000)),
        f"{row.path}: center index drifted",
    )
    _require(np.all((block_indices >= 0) & (block_indices < 2)), f"{row.path}: block drifted")
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
    _require(isinstance(hashes, Mapping), f"{row.path}: array hashes are missing")
    assert isinstance(hashes, Mapping)
    for name, values in arrays.items():
        _require(
            hashes.get(name) == canonical_array_sha256(np.asarray(values)),
            f"{row.path}: member hash mismatch: {name}",
        )


def encode_raw_features_in_fixed_chunks(raw_features: object) -> np.ndarray:
    """Rebuild immutable FMT161 with the sole production chunk schedule."""

    raw = np.asarray(raw_features)
    _require(
        raw.ndim == 2
        and raw.shape[0] > 0
        and raw.shape[1] == RAW_INPUT_WIDTH
        and raw.dtype == np.dtype(np.float32)
        and np.isfinite(raw).all(),
        "fixed-chunk encoder requires finite float32 Raw672 [N,672]",
    )
    parts: list[np.ndarray] = []
    for start in range(0, len(raw), ENCODE_CHUNK_ROWS):
        stop = min(start + ENCODE_CHUNK_ROWS, len(raw))
        encoded = encode_dimensionless_deformation_fmt(
            np.ascontiguousarray(raw[start:stop]),
            primitive_order=FROZEN_PRIMITIVE_ORDER,
        )
        # Subsets are deliberately not retained in a cache projection.  They
        # are reconstructed from this one 161D array using the parent indices.
        full = np.asarray(encoded[REPRESENTATIONS[0]])
        _require(
            full.dtype == np.dtype(np.float32)
            and full.shape == (stop - start, 161)
            and np.isfinite(full).all(),
            "dimensionless FMT161 chunk contract drifted",
        )
        parts.append(np.array(full, dtype=np.float32, order="C", copy=True))
        del encoded
    output = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
    _require(output.shape == (len(raw), 161), "dimensionless FMT161 row count drifted")
    return output


def encode_raw_features_with_chunk_rows_for_test(
    raw_features: object,
    chunk_rows: int,
) -> np.ndarray:
    """Test-only chunk-invariance oracle; production never calls this helper."""

    _require(isinstance(chunk_rows, int) and chunk_rows > 0, "chunk_rows must be positive")
    raw = np.asarray(raw_features)
    _require(
        raw.ndim == 2
        and raw.shape[0] > 0
        and raw.shape[1] == RAW_INPUT_WIDTH
        and raw.dtype == np.dtype(np.float32),
        "chunk oracle requires float32 Raw672 [N,672]",
    )
    parts = []
    for start in range(0, len(raw), chunk_rows):
        encoded = encode_dimensionless_deformation_fmt(
            np.ascontiguousarray(raw[start : start + chunk_rows]),
            primitive_order=FROZEN_PRIMITIVE_ORDER,
        )
        parts.append(np.array(encoded[REPRESENTATIONS[0]], copy=True, order="C"))
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def dimensionless_representation_features(
    fmt161: object,
    representation: str,
) -> np.ndarray:
    """Project the retained 161D vector with the unchanged parent indices."""

    values = np.asarray(fmt161)
    _require(
        values.ndim == 2
        and values.shape[1] == 161
        and values.dtype == np.dtype(np.float32)
        and np.isfinite(values).all(),
        "retained dimensionless FMT161 contract drifted",
    )
    _require(representation in REPRESENTATIONS, "unknown dimensionless representation")
    indices = PARENT_REPRESENTATION_INDEX_SETS[representation]
    projected = np.ascontiguousarray(values[:, indices], dtype=np.float32)
    expected_width = {REPRESENTATIONS[0]: 161, REPRESENTATIONS[1]: 36, REPRESENTATIONS[2]: 35}[
        representation
    ]
    _require(
        projected.shape == (len(values), expected_width),
        "parent coordinate projection width drifted",
    )
    return projected


def load_dimensionless_projection(
    plan: Plan,
    row: CacheRow,
    *,
    include_labels: bool,
) -> DimensionlessCacheProjection:
    """Open Raw/identity only, rebuild FMT161, and optionally open references."""

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
    fmt161 = encode_raw_features_in_fixed_chunks(raw)
    metadata: Mapping[str, Any] = {}
    labels: np.ndarray | None = None
    if include_labels:
        metadata = _cache_metadata(plan, row, arrays["metadata_json"])
        _require(
            int(metadata["valid_count"]) == len(raw),
            f"{row.path}: metadata valid count drifted",
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
    return DimensionlessCacheProjection(
        row=row,
        fmt_features=fmt161,
        scale_ids=np.ascontiguousarray(scales),
        center_indices=np.ascontiguousarray(centers),
        block_indices=np.ascontiguousarray(blocks),
        assigned_row_indices=np.ascontiguousarray(assigned),
        labels=None if labels is None else np.ascontiguousarray(labels),
        metadata=_deep_freeze(metadata),
        opened_members=tuple(names),
    )


_INHERITED_LOAD_CACHE_ROWS = inherited.load_cache_rows


def load_cache_rows(plan: Plan) -> tuple[list[CacheRow], dict[str, Any]]:
    """Authenticate the same frozen 32-row train manifest as the parent."""

    return _INHERITED_LOAD_CACHE_ROWS(plan)


def candidate_specs(plan: Plan) -> tuple[inherited.TailCandidateSpec, ...]:
    """Enumerate the active 3060-candidate grid with new representation names."""

    expected = _candidate_count(plan)
    old_count = inherited.FROZEN_CANDIDATE_COUNT
    try:
        inherited.FROZEN_CANDIDATE_COUNT = expected
        return inherited.candidate_specs(plan)
    finally:
        inherited.FROZEN_CANDIDATE_COUNT = old_count


def _representation_indices_sha256() -> str:
    payload = {
        name: list(PARENT_REPRESENTATION_INDEX_SETS[name]) for name in REPRESENTATIONS
    }
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _method_binding(plan: Plan, git_commit: str) -> dict[str, Any]:
    """Return the complete method identity embedded in every JSON artifact."""

    _require(
        isinstance(git_commit, str)
        and len(git_commit) == 40
        and all(character in "0123456789abcdef" for character in git_commit),
        "numerical Git commit must be a full lowercase SHA-1",
    )
    return {
        "schema": "pathline_template_matching.dimensionless_deformation_method_binding.v1",
        "experiment": EXPERIMENT,
        "experiment_config": {
            "path": str(plan.path),
            "sha256": plan.sha256,
        },
        "parent_per_scale_config": {
            "path": str(plan.parent_experiment_config_path),
            "sha256": plan.parent_experiment_config_sha256,
        },
        "source_cache_main_config_sha256": plan.parent_config_sha256,
        "source_cache_builder_git_commit": plan.cache_commit,
        "input_manifest_sha256": plan.manifest_sha256,
        "input_manifest_rows_content_sha256": plan.manifest_rows_sha256,
        "core": {
            "path": str(plan.core_path),
            "sha256": plan.core_sha256,
            "parent_descriptor_id": PARENT_DESCRIPTOR_ID,
        },
        "input_member": "raw_features",
        "forbidden_cache_member": "fmt_features",
        "encode_chunk_rows": plan.encode_chunk_rows,
        "retained_cache_projection": "fresh_dimensionless_fmt161_only",
        "representations": list(plan.representations),
        "parent_representation_indices_sha256": _representation_indices_sha256(),
        "numerical_git_commit": git_commit,
    }


def _outer_summary(
    rows: Sequence[Mapping[str, Any]], outer_family: str
) -> dict[str, Any]:
    summary = _INHERITED_OUTER_SUMMARY(rows, outer_family)
    summary["schema"] = OUTER_SUMMARY_SCHEMA
    summary["experiment"] = EXPERIMENT
    return summary


def _set_inherited_global(name: str, value: Any) -> None:
    """Set one parent-module binding through an injectable transaction seam."""

    setattr(inherited, name, value)


@contextmanager
def dimensionless_parent_runtime(
    plan: Plan,
    git_commit: str,
) -> Iterator[None]:
    """Bind inherited scoring for one exclusive, non-nested fold call.

    The inherited runner resolves its numerical dependencies from module
    globals.  This transaction therefore takes a process-local non-blocking
    lock, snapshots every old value before the first mutation, and restores
    every attempted mutation in reverse order.  Concurrent and nested use is
    rejected rather than exposing a partially rebound parent module.
    """

    acquired = _PARENT_RUNTIME_LOCK.acquire(blocking=False)
    _require(
        acquired,
        "dimensionless parent runtime is already active; nested or concurrent use is forbidden",
    )

    attempted: list[str] = []
    old: dict[str, Any] = {}
    restoration_failures: list[tuple[str, BaseException]] = []
    try:
        binding = _method_binding(plan, git_commit)
        original_manifest = inherited._manifest_with_self_hash
        original_authenticate_self_hash = inherited._authenticate_self_hash
        original_outer_summary = inherited._outer_summary

        def bound_load_plan(config_path: str | Path = CONFIG_PATH) -> Plan:
            """Return the already authenticated child plan without parent re-entry.

            The original parent ``load_plan`` reads its constants from the
            parent module globals.  Those globals must be rebound to the child
            while ``inherited.run`` executes, so recursively rebuilding the
            parent plan here would authenticate the parent file against child
            constants.  Re-authenticate all three immutable source files and
            return the exact plan built before the transaction instead.
            """

            active_path = Path(config_path).resolve()
            _require(
                active_path == plan.path,
                "inherited runner requested a different child config path",
            )
            _require(
                sha256_file(active_path) == plan.sha256,
                "child config changed after pre-transaction authentication",
            )
            _require(
                sha256_file(plan.parent_experiment_config_path)
                == plan.parent_experiment_config_sha256,
                "parent config changed after pre-transaction authentication",
            )
            _require(
                sha256_file(plan.core_path) == plan.core_sha256,
                "dimensionless core changed after pre-transaction authentication",
            )
            return plan

        replacements: dict[str, Any] = {
            "EXPERIMENT": EXPERIMENT,
            "EXPECTED_CONFIG_SHA256": EXPECTED_CONFIG_SHA256,
            "REPRESENTATIONS": plan.representations,
            "FROZEN_CANDIDATE_COUNT": _candidate_count(plan),
            "SCALER_ARTIFACT_SCHEMA": SCALER_ARTIFACT_SCHEMA,
            "SCALER_MANIFEST_SCHEMA": SCALER_MANIFEST_SCHEMA,
            "CALIBRATION_ARTIFACT_SCHEMA": CALIBRATION_ARTIFACT_SCHEMA,
            "CALIBRATION_MANIFEST_SCHEMA": CALIBRATION_MANIFEST_SCHEMA,
            "SELECTED_SCHEMA": SELECTED_SCHEMA,
            "PREDICTION_SCHEMA": PREDICTION_SCHEMA,
            "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
            "RESULT_SCHEMA": RESULT_SCHEMA,
            "COMPLETE_SCHEMA": COMPLETE_SCHEMA,
            "representation_features": dimensionless_representation_features,
            "load_cache_projection": load_dimensionless_projection,
            "load_cache_rows": load_cache_rows,
            "load_plan": bound_load_plan,
            "_atomic_csv": _atomic_csv,
            "_atomic_json": _atomic_json,
            "_atomic_npz": _atomic_npz,
        }

        def bound_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
            values = dict(payload)
            if values.get("schema") == (
                "pathline_template_matching.per_scale_negative_metric_outer_reference_access.v1"
            ):
                values["schema"] = REFERENCE_AUDIT_SCHEMA
            existing = values.get(METHOD_BINDING_KEY)
            _require(
                existing is None or existing == binding,
                "artifact method binding conflicts with the active method",
            )
            values[METHOD_BINDING_KEY] = binding
            return original_manifest(values)

        def bound_authenticate_self_hash(manifest: Mapping[str, Any]) -> None:
            original_authenticate_self_hash(manifest)
            _require(
                manifest.get(METHOD_BINDING_KEY) == binding,
                "artifact method binding drifted",
            )

        def clean_exact_git_identity() -> tuple[str, bool]:
            observed, dirty = _git_identity()
            _require(not dirty, "numerical run requires a clean committed worktree")
            _require(observed == git_commit, "numerical Git commit changed during the fold")
            return observed, dirty

        replacements.update(
            {
                "_manifest_with_self_hash": bound_manifest,
                "_authenticate_self_hash": bound_authenticate_self_hash,
                "_outer_summary": _outer_summary,
                "_git_identity": clean_exact_git_identity,
            }
        )
        # Snapshot the complete parent state before the first mutation.  This
        # also makes a missing parent binding fail without changing anything.
        old = {name: getattr(inherited, name) for name in replacements}
        for name, value in replacements.items():
            # Record the attempt first: even a setter that mutates and then
            # raises must have its old value restored.
            attempted.append(name)
            _set_inherited_global(name, value)
        yield
    finally:
        for name in reversed(attempted):
            try:
                _set_inherited_global(name, old[name])
            except BaseException as error:  # pragma: no cover - catastrophic runtime corruption
                restoration_failures.append((name, error))
        _PARENT_RUNTIME_LOCK.release()
        if restoration_failures:
            names = ", ".join(name for name, _ in restoration_failures)
            raise RuntimeError(
                f"failed to restore inherited parent globals: {names}"
            ) from restoration_failures[0][1]


def evaluate_outer_prediction(
    plan: Plan,
    selected: inherited.TailCandidateSpec,
    output_directory: Path,
    *,
    outer_family: str,
    git_commit: str,
    device: str,
    expected_scaler_manifest_sha256: str,
    expected_calibration_manifest_sha256: str,
    expected_selected_candidate_sha256: str,
    expected_prediction_manifest_sha256: str,
    inner_group_metrics_path: Path,
    inner_group_metrics_sha256: str,
    inner_candidate_summary_path: Path,
    inner_candidate_summary_sha256: str,
    inner_fit_audits_path: Path,
    inner_fit_audits_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fresh Raw replay and prediction authentication before opening labels."""

    with dimensionless_parent_runtime(plan, git_commit):
        return inherited.evaluate_outer_prediction(
            plan,
            selected,
            output_directory,
            outer_family=outer_family,
            git_commit=git_commit,
            device=device,
            expected_scaler_manifest_sha256=expected_scaler_manifest_sha256,
            expected_calibration_manifest_sha256=expected_calibration_manifest_sha256,
            expected_selected_candidate_sha256=expected_selected_candidate_sha256,
            expected_prediction_manifest_sha256=expected_prediction_manifest_sha256,
            inner_group_metrics_path=inner_group_metrics_path,
            inner_group_metrics_sha256=inner_group_metrics_sha256,
            inner_candidate_summary_path=inner_candidate_summary_path,
            inner_candidate_summary_sha256=inner_candidate_summary_sha256,
            inner_fit_audits_path=inner_fit_audits_path,
            inner_fit_audits_sha256=inner_fit_audits_sha256,
        )


def run(
    config_path: str | Path,
    outer_family: str,
    output_dir: str | Path,
    *,
    device: str,
    expected_config_sha256: str | None = EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    """Run one immutable nested-family fold from a clean exact revision."""

    plan = load_plan(config_path)
    _require(outer_family in plan.family_order, f"unknown outer family: {outer_family}")
    if expected_config_sha256 is not None:
        _require(
            plan.sha256 == expected_config_sha256,
            "frozen config SHA-256 mismatch",
        )
    git_commit, dirty = _git_identity()
    _require(not dirty, "Ibex numerical run requires a clean committed Git worktree")
    # The inherited transaction is used only while every global dependency is
    # rebound and authenticated to this method.  It opens no parent FMT member.
    with dimensionless_parent_runtime(plan, git_commit):
        result = inherited.run(
            config_path,
            outer_family,
            output_dir,
            device=device,
            expected_config_sha256=plan.sha256,
        )
    _require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("experiment") == EXPERIMENT
        and result.get(METHOD_BINDING_KEY) == _method_binding(plan, git_commit),
        "completed result lost its dimensionless method binding",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--expected-config-sha256", default=EXPECTED_CONFIG_SHA256)
    parser.add_argument("--outer-family", required=True, choices=FAMILY_ORDER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
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

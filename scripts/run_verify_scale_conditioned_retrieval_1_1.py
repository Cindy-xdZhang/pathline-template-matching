#!/usr/bin/env python3
"""Nested physical-family validation of scale-conditioned negative retrieval.

The outer-family cache projection deliberately omits ``valid_labels`` until
the selected candidate's immutable prediction artifact and manifest have been
closed.  Inner-family labels are development-only selection evidence.  The
classifier's spatial step consumes a complete valid source/block grid and is
therefore transductive even though the inherited FMT encoder is independent
per primitive.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.metrics import (  # noqa: E402
    BinaryMetrics,
)
from pathline_template_matching.nested_scale_validation import (  # noqa: E402
    CandidateSpec,
    InnerGroupKey,
    InnerGroupMetrics,
    aggregate_inner_group_metrics,
    candidate_predictions,
    fixed_top_fraction_predictions,
    representation_features,
    select_inner_candidate,
    spatial_support_scores,
    threshold_predictions,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.scale_one_class import (  # noqa: E402
    ScaleConditionedNegativeKNN,
)


EXPERIMENT = "Verify_ScaleConditionedRetrieval_1.1"
PREDICTION_FILE = "outer_predictions.npz"
PREDICTION_MANIFEST_FILE = "outer_prediction_manifest.json"
BLOCK_NAMES = ("legacy_2_1", "expanded_3_1")
PREDICTION_ARRAY_DTYPES = {
    "dataset_code": np.dtype(np.int8),
    "source_ordinal": np.dtype(np.int8),
    "block_index": np.dtype(np.int8),
    "center_index": np.dtype(np.int64),
    "assigned_row_index": np.dtype(np.int64),
    "scale_id": np.dtype(np.int32),
    "raw_negative_distance": np.dtype(np.float32),
    "spatial_score": np.dtype(np.float64),
    "spatial_denominator": np.dtype(np.float64),
    "retrieval_supported": np.dtype(np.bool_),
    "spatial_imputed": np.dtype(np.bool_),
    "spatial_unimputable": np.dtype(np.bool_),
    "prediction": np.dtype(np.bool_),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lower_hex(value: Any, length: int = 64) -> bool:
    text = str(value)
    return len(text) == length and all(character in "0123456789abcdef" for character in text)


def _stable_file_identity(path: Path, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    before = path.stat()
    _require(before.st_size == expected_size, f"file size mismatch: {path}")
    digest = sha256_file(path)
    after = path.stat()
    _require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {path}",
    )
    _require(digest == expected_sha256, f"file SHA-256 mismatch: {path}")
    return {"path": str(path), "size_bytes": int(after.st_size), "sha256": digest}


@dataclass(frozen=True)
class CacheRow:
    dataset: str
    family: str
    source_ordinal: int
    source_index: int
    path: Path
    size_bytes: int
    sha256: str


@dataclass
class CacheProjection:
    row: CacheRow
    fmt_features: np.ndarray
    scale_ids: np.ndarray
    center_indices: np.ndarray
    block_indices: np.ndarray
    assigned_row_indices: np.ndarray
    labels: np.ndarray | None
    metadata: Mapping[str, Any]

    @property
    def count(self) -> int:
        return len(self.scale_ids)


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


@dataclass(frozen=True)
class VerifiedOuterPrediction:
    """Prediction arrays authenticated before any outer reference member opens."""

    manifest_path: Path
    manifest_file_sha256: str
    prediction_file_sha256: str
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


def load_plan(config_path: str | Path) -> Plan:
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, Mapping), "config root must be a mapping")
    _require(raw.get("experiment") == EXPERIMENT, f"experiment must be {EXPERIMENT}")
    _require(
        raw.get("evidence_scope", {}).get("allowed_inputs")
        == "mainExp_TemplateMatching_3.1_train_caches_only",
        "input scope must remain train-cache-only",
    )
    _require(
        raw.get("evidence_scope", {}).get("forbidden_datasets")
        == ["tangaroa", "smokeBuoyancy"],
        "the two historical test datasets must remain forbidden",
    )
    families_raw = raw.get("families")
    split_raw = raw.get("nested_split")
    _require(isinstance(families_raw, Mapping), "families must be a mapping")
    _require(isinstance(split_raw, Mapping), "nested_split must be a mapping")
    family_order = tuple(str(value) for value in split_raw.get("outer_order", ()))
    _require(
        family_order
        == ("half_cylinder", "delta_wing", "f22_raptor", "channel", "boeing_747"),
        "outer family order drifted",
    )
    _require(tuple(split_raw.get("inner_order", ())) == family_order, "inner family order drifted")
    families = {str(key): tuple(str(value) for value in values) for key, values in families_raw.items()}
    _require(tuple(families) == family_order, "family mapping order drifted")
    datasets = [dataset for family in family_order for dataset in families[family]]
    _require(len(datasets) == 8 and len(set(datasets)) == 8, "families must contain eight unique datasets")
    dataset_to_family = {
        dataset: family for family in family_order for dataset in families[family]
    }

    parent = raw.get("parent_identity")
    _require(isinstance(parent, Mapping), "parent_identity must be a mapping")
    manifest = parent.get("input_manifest")
    _require(isinstance(manifest, Mapping), "input_manifest must be a mapping")
    _require(int(manifest.get("row_count", -1)) == 32, "input manifest must contain 32 rows")
    _require(_lower_hex(manifest.get("sha256")), "invalid input-manifest SHA-256")
    _require(_lower_hex(manifest.get("rows_content_sha256")), "invalid input rows SHA-256")

    representations = tuple(raw.get("representations", {}).get("order", ()))
    _require(
        representations == ("fmt161", "real_neighbor36", "chirality_all35"),
        "representation order drifted",
    )
    retrieval = raw.get("retrieval")
    transform = raw.get("group_transform")
    decisions = raw.get("decision_candidates")
    _require(isinstance(retrieval, Mapping), "retrieval must be a mapping")
    _require(isinstance(transform, Mapping), "group_transform must be a mapping")
    _require(isinstance(decisions, Mapping), "decision_candidates must be a mapping")
    ks = tuple(int(value) for value in retrieval.get("ks", ()))
    _require(ks == (1, 5, 15, 31), "k candidates drifted")
    sigmas = tuple(float(value) for value in transform.get("gaussian_sigmas_grid_indices", ()))
    _require(sigmas == (0.0, 0.5, 1.0, 1.5, 2.0), "Gaussian candidates drifted")
    thresholds = tuple(
        float(value) for value in decisions.get("calibrated_rank_threshold", {}).get("values", ())
    )
    expected_thresholds = tuple(round(0.50 + 0.01 * index, 2) for index in range(50))
    _require(thresholds == expected_thresholds, "threshold grid drifted")
    fraction = float(decisions.get("fixed_top_fraction", {}).get("fraction", -1.0))
    _require(fraction == 0.05, "fixed top fraction drifted")
    shape = tuple(int(value) for value in transform.get("grid_shape_zyx", ()))
    _require(shape == (40, 40, 40), "grid shape drifted")
    _require(
        transform.get("positive_sigma_policy") == "support_mask_normalized_spatial_imputation",
        "unsupported spatial policy drifted",
    )
    _require(
        transform.get("classifier_minimum_query_batch")
        == "complete_valid_source_times_scale_block_grid",
        "minimum query-batch contract drifted",
    )
    return Plan(
        path=path,
        sha256=sha256_file(path),
        raw=raw,
        family_order=family_order,
        families=families,
        dataset_to_family=dataset_to_family,
        manifest_path=Path(str(manifest["path"])),
        manifest_size=int(manifest["size_bytes"]),
        manifest_sha256=str(manifest["sha256"]),
        manifest_schema=str(manifest["schema"]),
        manifest_rows_sha256=str(manifest["rows_content_sha256"]),
        cache_commit=str(parent["cache_builder_git_commit"]),
        parent_config_sha256=str(parent["main_config_sha256"]),
        descriptor_id=str(parent["descriptor_id"]),
        cache_schema=str(parent["cache_schema"]),
        representations=representations,
        ks=ks,
        sigmas=sigmas,
        thresholds=thresholds,
        fixed_top_fraction=fraction,
        grid_shape=shape,
        gaussian_truncate=float(transform["gaussian_truncate"]),
        query_chunk_size=int(retrieval["query_chunk_size"]),
        library_chunk_size=int(retrieval["library_chunk_size"]),
    )


def load_cache_rows(plan: Plan) -> tuple[list[CacheRow], dict[str, Any]]:
    identity = _stable_file_identity(
        plan.manifest_path, plan.manifest_size, plan.manifest_sha256
    )
    manifest = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == plan.manifest_schema, "input manifest schema drifted")
    _require(manifest.get("row_count") == 32, "input manifest row count drifted")
    _require(
        manifest.get("rows_content_sha256") == plan.manifest_rows_sha256,
        "input manifest row-content hash drifted",
    )
    _require(manifest.get("git_commit") == plan.cache_commit, "cache commit drifted")
    _require(
        manifest.get("main_config_sha256") == plan.parent_config_sha256,
        "parent config SHA-256 drifted",
    )
    _require(manifest.get("test_dataset_access") is False, "manifest accessed test data")
    rows: list[CacheRow] = []
    for raw_row in manifest.get("rows", ()):  # type: ignore[union-attr]
        dataset = str(raw_row.get("dataset"))
        _require(dataset in plan.dataset_to_family, f"unauthorized dataset: {dataset}")
        path = Path(str(raw_row.get("cache_path")))
        _require("/primitive_cache/train/" in path.as_posix(), f"non-train cache path: {path}")
        _require(not any(name in path.as_posix() for name in ("tangaroa", "smokeBuoyancy")), "test cache path is forbidden")
        rows.append(
            CacheRow(
                dataset=dataset,
                family=plan.dataset_to_family[dataset],
                source_ordinal=int(raw_row["source_ordinal"]),
                source_index=int(raw_row["source_index"]),
                path=path,
                size_bytes=int(raw_row["cache_size_bytes"]),
                sha256=str(raw_row["cache_file_sha256"]),
            )
        )
    _require(len(rows) == 32, "manifest did not yield exactly 32 cache rows")
    _require(len({(row.dataset, row.source_ordinal) for row in rows}) == 32, "duplicate cache identity")
    for dataset in plan.dataset_to_family:
        ordinals = sorted(row.source_ordinal for row in rows if row.dataset == dataset)
        _require(ordinals == [0, 1, 2, 3], f"{dataset}: source ordinals drifted")
    rows.sort(key=lambda row: (list(plan.dataset_to_family).index(row.dataset), row.source_ordinal))
    identity["schema"] = manifest["schema"]
    identity["rows_content_sha256"] = manifest["rows_content_sha256"]
    return rows, identity


def _validate_cache_arrays(
    plan: Plan,
    row: CacheRow,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    _require(metadata.get("schema") == plan.cache_schema, f"{row.path}: cache schema drifted")
    _require(metadata.get("experiment") == "mainExp_TemplateMatching_3.1", f"{row.path}: parent experiment drifted")
    _require(metadata.get("split") == "train", f"{row.path}: cache split is not train")
    _require(metadata.get("dataset") == row.dataset, f"{row.path}: dataset mismatch")
    _require(metadata.get("physical_family") == row.family, f"{row.path}: family mismatch")
    _require(int(metadata.get("source_ordinal", -1)) == row.source_ordinal, f"{row.path}: ordinal mismatch")
    _require(int(metadata.get("source_index", -1)) == row.source_index, f"{row.path}: source index mismatch")
    _require(metadata.get("config_sha256") == plan.parent_config_sha256, f"{row.path}: parent config mismatch")
    _require(metadata.get("cache_builder_git_commit") == plan.cache_commit, f"{row.path}: cache commit mismatch")
    _require(metadata.get("descriptor_id") == plan.descriptor_id, f"{row.path}: descriptor mismatch")
    count = int(metadata.get("valid_count", -1))
    expected = {
        "fmt_features": (np.dtype(np.float32), (count, 161)),
        "valid_scale_id": (np.dtype(np.int32), (count,)),
        "valid_center_seed_index": (np.dtype(np.int64), (count,)),
        "valid_scale_block_index": (np.dtype(np.int8), (count,)),
        "valid_assigned_row_index": (np.dtype(np.int64), (count,)),
    }
    if "valid_labels" in arrays:
        expected["valid_labels"] = (np.dtype(np.bool_), (count,))
    stored_hashes = metadata.get("array_sha256")
    _require(isinstance(stored_hashes, Mapping), f"{row.path}: missing array hashes")
    for name, (dtype, shape) in expected.items():
        values = np.asarray(arrays[name])
        _require(values.dtype == dtype and values.shape == shape, f"{row.path}: {name} contract drifted")
        _require(stored_hashes.get(name) == canonical_array_sha256(values), f"{row.path}: {name} hash mismatch")
    features = arrays["fmt_features"]
    scales = arrays["valid_scale_id"]
    centers = arrays["valid_center_seed_index"]
    blocks = arrays["valid_scale_block_index"]
    assigned = arrays["valid_assigned_row_index"]
    _require(np.isfinite(features).all(), f"{row.path}: nonfinite FMT feature")
    _require(np.all((scales >= 0) & (scales < 2000)), f"{row.path}: scale outside 0..1999")
    _require(np.all((centers >= 0) & (centers < 64000)), f"{row.path}: center outside grid")
    _require(np.all((blocks >= 0) & (blocks < 2)), f"{row.path}: invalid block index")
    _require(np.array_equal(blocks, (scales >= 1000).astype(np.int8)), f"{row.path}: scale/block mismatch")
    _require(np.array_equal(assigned, blocks.astype(np.int64) * 64000 + centers), f"{row.path}: assigned identity mismatch")
    for block_index in (0, 1):
        selected = centers[blocks == block_index]
        _require(len(np.unique(selected)) == len(selected), f"{row.path}: duplicate center in block")


def _validate_outer_prediction_projection(
    row: CacheRow, arrays: Mapping[str, np.ndarray]
) -> None:
    """Validate unlabeled outer members without opening label-bearing metadata."""

    features = np.asarray(arrays["fmt_features"])
    _require(
        features.ndim == 2
        and features.shape[1] == 161
        and features.dtype == np.dtype(np.float32),
        f"{row.path}: outer FMT projection contract drifted",
    )
    count = len(features)
    expected = {
        "valid_scale_id": np.dtype(np.int32),
        "valid_center_seed_index": np.dtype(np.int64),
        "valid_scale_block_index": np.dtype(np.int8),
        "valid_assigned_row_index": np.dtype(np.int64),
    }
    for name, dtype in expected.items():
        values = np.asarray(arrays[name])
        _require(
            values.shape == (count,) and values.dtype == dtype,
            f"{row.path}: outer {name} contract drifted",
        )
    _require(np.isfinite(features).all(), f"{row.path}: nonfinite outer FMT feature")
    scales = arrays["valid_scale_id"]
    centers = arrays["valid_center_seed_index"]
    blocks = arrays["valid_scale_block_index"]
    assigned = arrays["valid_assigned_row_index"]
    _require(np.all((scales >= 0) & (scales < 2000)), f"{row.path}: outer scale outside range")
    _require(np.all((centers >= 0) & (centers < 64000)), f"{row.path}: outer center outside grid")
    _require(np.all((blocks >= 0) & (blocks < 2)), f"{row.path}: outer block outside range")
    _require(np.array_equal(blocks, (scales >= 1000).astype(np.int8)), f"{row.path}: outer scale/block mismatch")
    _require(np.array_equal(assigned, blocks.astype(np.int64) * 64000 + centers), f"{row.path}: outer assigned identity mismatch")
    for block_index in (0, 1):
        selected = centers[blocks == block_index]
        _require(len(np.unique(selected)) == len(selected), f"{row.path}: duplicate outer center")


def load_cache_projection(plan: Plan, row: CacheRow, *, include_labels: bool) -> CacheProjection:
    _stable_file_identity(row.path, row.size_bytes, row.sha256)
    names = [
        "fmt_features",
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
    ]
    if include_labels:
        names.append("valid_labels")
    with np.load(row.path, allow_pickle=False) as archive:
        _require(all(name in archive.files for name in names), f"{row.path}: prediction projection is incomplete")
        arrays = {name: np.asarray(archive[name]) for name in names}
        if include_labels:
            _require("metadata_json" in archive.files, f"{row.path}: metadata_json is missing")
            metadata_scalar = np.asarray(archive["metadata_json"])
            _require(metadata_scalar.shape == (), f"{row.path}: metadata_json is not scalar")
            metadata: Mapping[str, Any] = json.loads(str(metadata_scalar.item()))
        else:
            # metadata_json includes reference-positive counts.  The outer
            # prediction projection must not even open that member.
            metadata = {}
    if include_labels:
        _validate_cache_arrays(plan, row, metadata, arrays)
    else:
        _validate_outer_prediction_projection(row, arrays)
    return CacheProjection(
        row=row,
        fmt_features=np.ascontiguousarray(arrays["fmt_features"]),
        scale_ids=np.ascontiguousarray(arrays["valid_scale_id"]),
        center_indices=np.ascontiguousarray(arrays["valid_center_seed_index"]),
        block_indices=np.ascontiguousarray(arrays["valid_scale_block_index"]),
        assigned_row_indices=np.ascontiguousarray(arrays["valid_assigned_row_index"]),
        labels=(None if not include_labels else np.ascontiguousarray(arrays["valid_labels"])),
        metadata=metadata,
    )


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
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


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)
    return sha256_file(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as destination:
        np.savez_compressed(destination, **arrays)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)
    return sha256_file(path)


def _git_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )
    _require(_lower_hex(commit, 40), "Git HEAD is not a full SHA-1")
    return commit, dirty


def _fit_negative_model(
    caches: Sequence[CacheProjection], representation: str
) -> ScaleConditionedNegativeKNN:
    feature_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    family_parts: list[np.ndarray] = []
    family_codes = {family: index for index, family in enumerate(sorted({cache.row.family for cache in caches}))}
    for cache in caches:
        _require(cache.labels is not None, "fit cache labels were not loaded")
        negative = ~np.asarray(cache.labels, dtype=bool)
        represented = representation_features(cache.fmt_features, representation)
        feature_parts.append(np.ascontiguousarray(represented[negative]))
        scale_parts.append(np.asarray(cache.scale_ids[negative], dtype=np.int64))
        family_parts.append(
            np.full(int(negative.sum()), family_codes[cache.row.family], dtype=np.int16)
        )
    _require(feature_parts and sum(len(part) for part in feature_parts) > 0, "negative library is empty")
    features = np.ascontiguousarray(np.concatenate(feature_parts, axis=0), dtype=np.float32)
    scales = np.ascontiguousarray(np.concatenate(scale_parts), dtype=np.int64)
    family_ids = np.ascontiguousarray(np.concatenate(family_parts), dtype=np.int16)
    model = ScaleConditionedNegativeKNN(features, scales, family_ids)
    del features, scales, family_ids, feature_parts, scale_parts, family_parts
    gc.collect()
    return model


def _partial_supported_query(
    model: ScaleConditionedNegativeKNN,
    features: np.ndarray,
    scale_ids: np.ndarray,
    ks: Sequence[int],
    *,
    device: str,
    query_chunk_size: int,
    library_chunk_size: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Query each support tier once and retain explicit unsupported sentinels."""

    counts = {int(key): int(value) for key, value in model.fit_audit["scale_counts"].items()}
    requested = tuple(int(k) for k in ks)
    _require(
        requested and tuple(sorted(set(requested))) == requested,
        "requested ks must be unique and strictly increasing",
    )
    count_table = np.zeros(2000, dtype=np.int64)
    for scale_id, count in counts.items():
        _require(0 <= scale_id < 2000, "fitted scale id is outside 0..1999")
        count_table[scale_id] = count
    _require(
        np.asarray(scale_ids).ndim == 1
        and np.all((np.asarray(scale_ids) >= 0) & (np.asarray(scale_ids) < 2000)),
        "query scale id is outside 0..1999",
    )
    row_counts = count_table[np.asarray(scale_ids, dtype=np.int64)]
    scores = {k: np.full(len(scale_ids), np.nan, dtype=np.float32) for k in requested}
    supports = {
        k: np.asarray(row_counts >= k, dtype=bool)
        for k in requested
    }
    for maximum_supported_k in requested:
        upper_candidates = [value for value in requested if value > maximum_supported_k]
        next_k = min(upper_candidates) if upper_candidates else None
        tier_mask = row_counts >= maximum_supported_k
        if next_k is not None:
            tier_mask &= row_counts < next_k
        rows = np.flatnonzero(tier_mask)
        if len(rows) == 0:
            continue
        supported_ks = tuple(k for k in requested if k <= maximum_supported_k)
        result = model.query(
            features[rows],
            scale_ids[rows],
            ks=supported_ks,
            device=device,
            query_chunk_size=query_chunk_size,
            library_chunk_size=library_chunk_size,
        )
        for k in supported_ks:
            scores[k][rows] = result[k]
    for k in requested:
        _require(np.isfinite(scores[k][supports[k]]).all(), f"k={k}: supported score is nonfinite")
        _require(np.isnan(scores[k][~supports[k]]).all(), f"k={k}: unsupported score sentinel drifted")
    return scores, supports


def _query_cache_batch(
    model: ScaleConditionedNegativeKNN,
    caches: Sequence[CacheProjection],
    representation: str,
    plan: Plan,
    *,
    device: str,
    ks: Sequence[int] | None = None,
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    requested_ks = plan.ks if ks is None else tuple(int(value) for value in ks)
    _require(requested_ks and set(requested_ks).issubset(plan.ks), "query k set is not frozen")
    offsets = np.cumsum([0, *[cache.count for cache in caches]], dtype=np.int64)
    features = np.ascontiguousarray(
        np.concatenate(
            [
                representation_features(cache.fmt_features, representation)
                for cache in caches
            ],
            axis=0,
        ),
        dtype=np.float32,
    )
    scales = np.ascontiguousarray(
        np.concatenate([cache.scale_ids for cache in caches]), dtype=np.int64
    )
    all_scores, all_supports = _partial_supported_query(
        model,
        features,
        scales,
        requested_ks,
        device=device,
        query_chunk_size=plan.query_chunk_size,
        library_chunk_size=plan.library_chunk_size,
    )
    del features, scales
    scores = {
        k: [all_scores[k][offsets[index] : offsets[index + 1]] for index in range(len(caches))]
        for k in requested_ks
    }
    supports = {
        k: [all_supports[k][offsets[index] : offsets[index + 1]] for index in range(len(caches))]
        for k in requested_ks
    }
    return scores, supports


def _classification_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int | float]:
    targets = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predictions, dtype=bool)
    _require(targets.ndim == 1 and targets.shape == predicted.shape and len(targets) > 0, "invalid classification arrays")
    tp = int(np.sum(targets & predicted))
    fp = int(np.sum(~targets & predicted))
    tn = int(np.sum(~targets & ~predicted))
    fn = int(np.sum(targets & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": len(targets),
        "positive_count": int(targets.sum()),
        "negative_count": int((~targets).sum()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(targets),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def _ranking_metrics_one_sort(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Tie-aware Average Precision and AUROC from one stable score sort."""

    targets = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    _require(
        targets.ndim == 1
        and targets.shape == values.shape
        and len(targets) > 0
        and np.isfinite(values).all(),
        "ranking metrics require aligned finite arrays",
    )
    positive_count = int(targets.sum())
    negative_count = len(targets) - positive_count
    _require(positive_count > 0 and negative_count > 0, "ranking metrics require both classes")
    order = np.argsort(-values, kind="mergesort")
    ordered_scores = values[order]
    ordered_targets = targets[order].astype(np.int64)
    group_ends = np.r_[
        np.flatnonzero(np.diff(ordered_scores) != 0), len(values) - 1
    ]
    group_starts = np.r_[0, group_ends[:-1] + 1]
    cumulative_positive = np.cumsum(ordered_targets, dtype=np.int64)[group_ends]
    retrieved = group_ends + 1
    recall = cumulative_positive / positive_count
    precision = cumulative_positive / retrieved
    average_precision_value = float(
        np.sum(np.diff(np.r_[0.0, recall]) * precision)
    )
    group_positive = np.add.reduceat(ordered_targets, group_starts)
    group_size = group_ends - group_starts + 1
    group_negative = group_size - group_positive
    negative_below = negative_count - np.cumsum(group_negative, dtype=np.int64)
    favorable = np.sum(
        group_positive
        * (negative_below.astype(np.float64) + 0.5 * group_negative)
    )
    auroc_value = float(favorable / (positive_count * negative_count))
    return average_precision_value, auroc_value


def _subset_f1(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return float("nan")
    return float(_classification_counts(labels[selected], predictions[selected])["f1"])


def _metric_row(
    *,
    outer_family: str,
    inner_family: str,
    dataset: str,
    source_ordinal: int,
    block: str,
    candidate: CandidateSpec,
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    supported: np.ndarray,
    imputed: np.ndarray,
    unimputable: np.ndarray,
    ranking_metrics: tuple[float, float] | None = None,
) -> dict[str, Any]:
    _require(np.array_equal(supported | imputed | unimputable, np.ones(len(labels), dtype=bool)), "support states do not cover group")
    _require(not np.any((supported & imputed) | (supported & unimputable) | (imputed & unimputable)), "support states overlap")
    counts = _classification_counts(labels, predictions)
    if ranking_metrics is None:
        ranking_metrics = _ranking_metrics_one_sort(labels, scores)
    row: dict[str, Any] = {
        "outer_family": outer_family,
        "inner_family": inner_family,
        "dataset": dataset,
        "source_ordinal": source_ordinal,
        "block": block,
        "candidate_id": candidate.candidate_id,
        "representation": candidate.representation,
        "k": candidate.k,
        "sigma": candidate.sigma,
        "decision_rule": candidate.decision_rule,
        "decision_value": candidate.decision_value,
        **counts,
        "average_precision": ranking_metrics[0],
        "auroc": ranking_metrics[1],
        "supported_count": int(supported.sum()),
        "imputed_count": int(imputed.sum()),
        "unimputable_count": int(unimputable.sum()),
        "retrieval_support_fraction": float(supported.mean()),
        "spatial_imputed_fraction": float(imputed.mean()),
        "spatial_unimputable_fraction": float(unimputable.mean()),
        "supported_subset_f1": _subset_f1(labels, predictions, supported),
        "imputed_subset_f1": _subset_f1(labels, predictions, imputed),
        "unimputable_subset_f1": _subset_f1(labels, predictions, unimputable),
    }
    _require(np.isfinite([row[name] for name in ("average_precision", "auroc", "f1", "balanced_accuracy")]).all(), "group must contain both classes and finite metrics")
    return row


def _classification_from_confusion(
    *, tp: int, fp: int, tn: int, fn: int
) -> dict[str, int | float]:
    sample_count = tp + fp + tn + fn
    _require(sample_count > 0, "confusion counts describe an empty population")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": sample_count,
        "positive_count": tp + fn,
        "negative_count": tn + fp,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / sample_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def _threshold_confusion_series(
    labels: np.ndarray,
    scores: np.ndarray,
    eligible: np.ndarray,
    thresholds: Sequence[float],
) -> list[dict[str, int | float]]:
    targets = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    allowed = np.asarray(eligible, dtype=bool)
    _require(targets.shape == values.shape == allowed.shape and len(targets) > 0, "invalid threshold population")
    eligible_scores = values[allowed]
    eligible_labels = targets[allowed]
    order = np.argsort(eligible_scores, kind="mergesort")
    sorted_scores = eligible_scores[order]
    sorted_labels = eligible_labels[order].astype(np.int64)
    suffix_positive = np.concatenate(
        (np.cumsum(sorted_labels[::-1], dtype=np.int64)[::-1], np.zeros(1, dtype=np.int64))
    )
    total_positive = int(targets.sum())
    total_negative = len(targets) - total_positive
    output: list[dict[str, int | float]] = []
    for threshold in thresholds:
        first = int(np.searchsorted(sorted_scores, float(threshold), side="left"))
        predicted_count = len(sorted_scores) - first
        tp = int(suffix_positive[first])
        fp = predicted_count - tp
        fn = total_positive - tp
        tn = total_negative - fp
        output.append(_classification_from_confusion(tp=tp, fp=fp, tn=tn, fn=fn))
    return output


def _threshold_metric_rows(
    *,
    plan: Plan,
    outer_family: str,
    inner_family: str,
    dataset: str,
    source_ordinal: int,
    block: str,
    candidates: Sequence[CandidateSpec],
    labels: np.ndarray,
    scores: np.ndarray,
    eligible: np.ndarray,
    supported: np.ndarray,
    imputed: np.ndarray,
    unimputable: np.ndarray,
    ranking: tuple[float, float],
) -> list[dict[str, Any]]:
    thresholds = [candidate.decision_value for candidate in candidates]
    whole = _threshold_confusion_series(labels, scores, eligible, thresholds)
    subset_series: dict[str, list[dict[str, int | float]] | None] = {}
    for name, mask in (
        ("supported_subset_f1", supported),
        ("imputed_subset_f1", imputed),
        ("unimputable_subset_f1", unimputable),
    ):
        if np.asarray(mask, dtype=bool).any():
            subset_series[name] = _threshold_confusion_series(
                labels[mask], scores[mask], eligible[mask], thresholds
            )
        else:
            subset_series[name] = None
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        counts = whole[index]
        row: dict[str, Any] = {
            "outer_family": outer_family,
            "inner_family": inner_family,
            "dataset": dataset,
            "source_ordinal": source_ordinal,
            "block": block,
            "candidate_id": candidate.candidate_id,
            "representation": candidate.representation,
            "k": candidate.k,
            "sigma": candidate.sigma,
            "decision_rule": candidate.decision_rule,
            "decision_value": candidate.decision_value,
            **counts,
            "average_precision": ranking[0],
            "auroc": ranking[1],
            "supported_count": int(supported.sum()),
            "imputed_count": int(imputed.sum()),
            "unimputable_count": int(unimputable.sum()),
            "retrieval_support_fraction": float(supported.mean()),
            "spatial_imputed_fraction": float(imputed.mean()),
            "spatial_unimputable_fraction": float(unimputable.mean()),
        }
        for name, series in subset_series.items():
            row[name] = float("nan") if series is None else float(series[index]["f1"])
        rows.append(row)
    return rows


def _base_candidates(plan: Plan, representation: str, k: int, sigma: float) -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(
            representation=representation,
            k=k,
            sigma=sigma,
            decision_rule="fixed_top_fraction",
            decision_value=plan.fixed_top_fraction,
        ),
        *tuple(
            CandidateSpec(
                representation=representation,
                k=k,
                sigma=sigma,
                decision_rule="rank_threshold",
                decision_value=threshold,
            )
            for threshold in plan.thresholds
        ),
    )


def _inner_metric_rows(
    plan: Plan,
    caches: Sequence[CacheProjection],
    outer_family: str,
    *,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, CandidateSpec], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidates: dict[str, CandidateSpec] = {}
    fit_audits: list[dict[str, Any]] = []
    nonouter_families = [family for family in plan.family_order if family != outer_family]
    for inner_family in nonouter_families:
        fit_families = [
            family for family in plan.family_order if family not in {outer_family, inner_family}
        ]
        fit_caches = [cache for cache in caches if cache.row.family in fit_families]
        query_caches = [cache for cache in caches if cache.row.family == inner_family]
        _require(fit_caches and query_caches, f"{inner_family}: empty nested fold")
        for representation in plan.representations:
            print(
                f"[{_utc_now()}] outer={outer_family} inner={inner_family} "
                f"representation={representation} fit_start",
                flush=True,
            )
            model = _fit_negative_model(fit_caches, representation)
            audit = model.fit_audit
            scale_counts = {
                int(scale_id): int(count)
                for scale_id, count in audit["scale_counts"].items()
            }
            all_scale_counts = [scale_counts.get(scale_id, 0) for scale_id in range(2000)]
            fit_audits.append(
                {
                    "outer_family": outer_family,
                    "inner_family": inner_family,
                    "fit_families": fit_families,
                    "representation": representation,
                    "negative_count": audit["count"],
                    "feature_width": audit["dim"],
                    "supported_scale_counts_by_k": {
                        str(k): sum(count >= k for count in all_scale_counts)
                        for k in plan.ks
                    },
                    "zero_support_scale_count": sum(count == 0 for count in all_scale_counts),
                    "minimum_scale_count_including_missing": min(all_scale_counts),
                    "maximum_scale_count": max(all_scale_counts),
                    "family_counts": audit.get("family_counts"),
                }
            )
            score_parts, support_parts = _query_cache_batch(
                model, query_caches, representation, plan, device=device
            )
            for cache_index, cache in enumerate(query_caches):
                labels_all = cache.labels
                _require(labels_all is not None, "inner query labels are unavailable")
                for block_index, block_name in enumerate(BLOCK_NAMES):
                    selected = np.asarray(cache.block_indices == block_index)
                    _require(selected.any(), f"{cache.row.dataset}/{cache.row.source_ordinal}/{block_name}: empty valid group")
                    labels = np.asarray(labels_all[selected], dtype=bool)
                    centers = np.asarray(cache.center_indices[selected], dtype=np.int64)
                    _require(labels.any() and (~labels).any(), f"{cache.row.dataset}/{cache.row.source_ordinal}/{block_name}: single-class group")
                    for k in plan.ks:
                        raw = np.asarray(score_parts[k][cache_index][selected], dtype=np.float32)
                        supported = np.asarray(support_parts[k][cache_index][selected], dtype=bool)
                        for sigma in plan.sigmas:
                            spatial = spatial_support_scores(
                                raw,
                                centers,
                                supported,
                                grid_shape=plan.grid_shape,
                                sigma=sigma,
                                truncate=plan.gaussian_truncate,
                            )
                            eligible = np.asarray(
                                spatial.supported_mask | spatial.imputed_mask, dtype=bool
                            )
                            ranking = _ranking_metrics_one_sort(
                                labels, spatial.scores
                            )
                            base_candidates = _base_candidates(
                                plan, representation, k, sigma
                            )
                            for candidate in base_candidates:
                                previous = candidates.setdefault(
                                    candidate.candidate_id, candidate
                                )
                                _require(previous == candidate, "candidate ID collision")
                            fixed_candidate = base_candidates[0]
                            fixed_prediction = candidate_predictions(
                                fixed_candidate, spatial.scores, centers, eligible
                            )
                            rows.append(
                                _metric_row(
                                    outer_family=outer_family,
                                    inner_family=inner_family,
                                    dataset=cache.row.dataset,
                                    source_ordinal=cache.row.source_ordinal,
                                    block=block_name,
                                    candidate=fixed_candidate,
                                    labels=labels,
                                    scores=spatial.scores,
                                    predictions=fixed_prediction,
                                    supported=spatial.supported_mask,
                                    imputed=spatial.imputed_mask,
                                    unimputable=spatial.unimputable_mask,
                                    ranking_metrics=ranking,
                                )
                            )
                            rows.extend(
                                _threshold_metric_rows(
                                    plan=plan,
                                    outer_family=outer_family,
                                    inner_family=inner_family,
                                    dataset=cache.row.dataset,
                                    source_ordinal=cache.row.source_ordinal,
                                    block=block_name,
                                    candidates=base_candidates[1:],
                                    labels=labels,
                                    scores=spatial.scores,
                                    eligible=eligible,
                                    supported=spatial.supported_mask,
                                    imputed=spatial.imputed_mask,
                                    unimputable=spatial.unimputable_mask,
                                    ranking=ranking,
                                )
                            )
            del model, score_parts, support_parts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[{_utc_now()}] outer={outer_family} inner={inner_family} "
                f"representation={representation} complete metric_rows={len(rows)}",
                flush=True,
            )
    return rows, candidates, fit_audits


def _binary_metrics_from_row(row: Mapping[str, Any]) -> BinaryMetrics:
    return BinaryMetrics(
        sample_count=int(row["sample_count"]),
        positive_count=int(row["positive_count"]),
        negative_count=int(row["negative_count"]),
        true_positive=int(row["true_positive"]),
        false_positive=int(row["false_positive"]),
        true_negative=int(row["true_negative"]),
        false_negative=int(row["false_negative"]),
        average_precision=float(row["average_precision"]),
        auroc=float(row["auroc"]),
        precision=float(row["precision"]),
        recall=float(row["recall"]),
        f1=float(row["f1"]),
        balanced_accuracy=float(row["balanced_accuracy"]),
    )


def _hierarchical_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    families = sorted({str(row["inner_family"]) for row in rows})
    return float(
        np.mean(
            [
                np.mean(
                    [float(row[field]) for row in rows if row["inner_family"] == family]
                )
                for family in families
            ]
        )
    )


def _aggregate_and_select(
    plan: Plan,
    rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, CandidateSpec],
) -> tuple[list[dict[str, Any]], CandidateSpec, dict[str, Any]]:
    _require(rows and candidates, "inner validation produced no candidate metrics")
    grouped: dict[str, list[Mapping[str, Any]]] = {candidate_id: [] for candidate_id in candidates}
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    expected_keys: set[tuple[str, str, int, str]] | None = None
    macros = []
    summaries: list[dict[str, Any]] = []
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
        _require(len(keys) == len(candidate_rows), f"{candidate_id}: duplicate inner group")
        if expected_keys is None:
            expected_keys = keys
        _require(keys == expected_keys, f"{candidate_id}: incomplete inner group set")
        inner_objects = [
            InnerGroupMetrics(
                candidate=candidates[candidate_id],
                physical_family=str(row["inner_family"]),
                group=InnerGroupKey(
                    dataset=str(row["dataset"]),
                    source_ordinal=int(row["source_ordinal"]),
                    block_id=BLOCK_NAMES.index(str(row["block"])),
                ),
                metrics=_binary_metrics_from_row(row),
            )
            for row in candidate_rows
        ]
        macro = aggregate_inner_group_metrics(inner_objects)
        _require(macro.family_count == 4, f"{candidate_id}: expected four inner families")
        macros.append(macro)
        summary = macro.as_dict()
        summary.update(
            {
                **_candidate_payload(candidates[candidate_id]),
                "accuracy": _hierarchical_mean(candidate_rows, "accuracy"),
                "retrieval_support_fraction": _hierarchical_mean(
                    candidate_rows, "retrieval_support_fraction"
                ),
                "spatial_imputed_fraction": _hierarchical_mean(
                    candidate_rows, "spatial_imputed_fraction"
                ),
                "spatial_unimputable_fraction": _hierarchical_mean(
                    candidate_rows, "spatial_unimputable_fraction"
                ),
                "supported_subset_f1": _hierarchical_mean(
                    candidate_rows, "supported_subset_f1"
                ),
                "imputed_subset_f1": _hierarchical_mean(
                    candidate_rows, "imputed_subset_f1"
                ),
                "unimputable_subset_f1": _hierarchical_mean(
                    candidate_rows, "unimputable_subset_f1"
                ),
            }
        )
        summaries.append(summary)
    selected_macro = select_inner_candidate(macros)
    selected = selected_macro.candidate
    selected_summary = next(
        row for row in summaries if row["candidate_id"] == selected.candidate_id
    )
    return summaries, selected, selected_summary


def _candidate_payload(candidate: CandidateSpec) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "representation": candidate.representation,
        "k": candidate.k,
        "sigma": candidate.sigma,
        "decision_rule": candidate.decision_rule,
        "decision_value": candidate.decision_value,
    }


def _build_outer_prediction_arrays(
    plan: Plan,
    outer_family: str,
    caches: Sequence[CacheProjection],
    candidate: CandidateSpec,
    model: ScaleConditionedNegativeKNN,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    score_parts, support_parts = _query_cache_batch(
        model,
        caches,
        candidate.representation,
        plan,
        device=device,
        ks=(candidate.k,),
    )
    dataset_names = plan.families[outer_family]
    dataset_codes = {dataset: index for index, dataset in enumerate(dataset_names)}
    output: dict[str, list[np.ndarray]] = {
        "dataset_code": [],
        "source_ordinal": [],
        "block_index": [],
        "center_index": [],
        "assigned_row_index": [],
        "scale_id": [],
        "raw_negative_distance": [],
        "spatial_score": [],
        "spatial_denominator": [],
        "retrieval_supported": [],
        "spatial_imputed": [],
        "spatial_unimputable": [],
        "prediction": [],
    }
    groups: list[dict[str, Any]] = []
    row_offset = 0
    for cache_index, cache in enumerate(caches):
        for block_index, block_name in enumerate(BLOCK_NAMES):
            selected = np.asarray(cache.block_indices == block_index)
            _require(selected.any(), f"outer {cache.row.dataset}/{cache.row.source_ordinal}/{block_name} is empty")
            raw = np.asarray(score_parts[candidate.k][cache_index][selected], dtype=np.float32)
            supported = np.asarray(
                support_parts[candidate.k][cache_index][selected], dtype=bool
            )
            centers = np.asarray(cache.center_indices[selected], dtype=np.int64)
            spatial = spatial_support_scores(
                raw,
                centers,
                supported,
                grid_shape=plan.grid_shape,
                sigma=candidate.sigma,
                truncate=plan.gaussian_truncate,
            )
            eligible = np.asarray(
                spatial.supported_mask | spatial.imputed_mask, dtype=bool
            )
            prediction = candidate_predictions(
                candidate, spatial.scores, centers, eligible
            )
            count = len(centers)
            output["dataset_code"].append(
                np.full(count, dataset_codes[cache.row.dataset], dtype=np.int8)
            )
            output["source_ordinal"].append(
                np.full(count, cache.row.source_ordinal, dtype=np.int8)
            )
            output["block_index"].append(
                np.full(count, block_index, dtype=np.int8)
            )
            output["center_index"].append(centers)
            output["assigned_row_index"].append(
                np.asarray(cache.assigned_row_indices[selected], dtype=np.int64)
            )
            output["scale_id"].append(
                np.asarray(cache.scale_ids[selected], dtype=np.int32)
            )
            output["raw_negative_distance"].append(raw)
            output["spatial_score"].append(np.asarray(spatial.scores, dtype=np.float64))
            output["spatial_denominator"].append(
                np.asarray(spatial.denominator, dtype=np.float64)
            )
            output["retrieval_supported"].append(spatial.supported_mask)
            output["spatial_imputed"].append(spatial.imputed_mask)
            output["spatial_unimputable"].append(spatial.unimputable_mask)
            output["prediction"].append(np.asarray(prediction, dtype=np.bool_))
            groups.append(
                {
                    "dataset": cache.row.dataset,
                    "source_ordinal": cache.row.source_ordinal,
                    "block": block_name,
                    "row_start": row_offset,
                    "row_stop_exclusive": row_offset + count,
                    "row_count": count,
                    "supported_count": int(spatial.supported_mask.sum()),
                    "imputed_count": int(spatial.imputed_mask.sum()),
                    "unimputable_count": int(spatial.unimputable_mask.sum()),
                }
            )
            row_offset += count
    arrays = {
        name: np.ascontiguousarray(np.concatenate(parts))
        for name, parts in output.items()
    }
    _require(len({len(values) for values in arrays.values()}) == 1, "outer arrays are misaligned")
    return arrays, groups


def _write_outer_prediction(
    plan: Plan,
    output_dir: Path,
    outer_family: str,
    caches: Sequence[CacheProjection],
    candidate: CandidateSpec,
    selected_candidate_sha256: str,
    arrays: Mapping[str, np.ndarray],
    groups: Sequence[Mapping[str, Any]],
    *,
    git_commit: str,
    device: str,
) -> tuple[dict[str, Any], str]:
    prediction_path = output_dir / PREDICTION_FILE
    prediction_sha = _atomic_npz(prediction_path, arrays)
    manifest: dict[str, Any] = {
        "schema": "pathline_template_matching.scale_conditioned_outer_prediction.v1",
        "experiment": EXPERIMENT,
        "phase": "outer_prediction_closed_before_outer_reference_open",
        "created_utc": _utc_now(),
        "git_commit": git_commit,
        "config_path": str(plan.path),
        "config_sha256": plan.sha256,
        "outer_family": outer_family,
        "dataset_names_by_code": list(plan.families[outer_family]),
        "selected_candidate": _candidate_payload(candidate),
        "selected_candidate_file_sha256": selected_candidate_sha256,
        "device": device,
        "source_cache_files": [
            {
                "dataset": cache.row.dataset,
                "source_ordinal": cache.row.source_ordinal,
                "path": str(cache.row.path),
                "size_bytes": cache.row.size_bytes,
                "sha256": cache.row.sha256,
                "opened_members": [
                    "fmt_features",
                    "valid_scale_id",
                    "valid_center_seed_index",
                    "valid_scale_block_index",
                    "valid_assigned_row_index",
                ],
                "metadata_json_opened": False,
                "valid_labels_opened": False,
            }
            for cache in caches
        ],
        "classifier_scope": "transductive_complete_valid_source_times_scale_block_grid",
        "row_count": len(arrays["prediction"]),
        "group_count": len(groups),
        "groups": list(groups),
        "prediction_file": PREDICTION_FILE,
        "prediction_file_size_bytes": prediction_path.stat().st_size,
        "prediction_file_sha256": prediction_sha,
        "prediction_array_sha256": {
            name: canonical_array_sha256(values) for name, values in arrays.items()
        },
        "outer_reference_opened": False,
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    manifest_sha = _atomic_json(output_dir / PREDICTION_MANIFEST_FILE, manifest)
    return manifest, manifest_sha


def _verify_outer_prediction_artifacts(
    plan: Plan,
    output_dir: Path,
    outer_family: str,
    candidate: CandidateSpec,
    selected_candidate_sha256: str,
    expected_manifest_sha256: str,
) -> VerifiedOuterPrediction:
    """Authenticate the closed prediction artifacts before reference access."""

    manifest_path = output_dir / PREDICTION_MANIFEST_FILE
    _require(manifest_path.is_file(), "prediction manifest must exist before reference open")
    _require(_lower_hex(expected_manifest_sha256), "invalid expected prediction manifest SHA-256")
    manifest_identity = _stable_file_identity(
        manifest_path,
        manifest_path.stat().st_size,
        expected_manifest_sha256,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, Mapping), "prediction manifest root must be a mapping")
    content = dict(manifest)
    stored_content_sha = content.pop("manifest_content_sha256", None)
    _require(
        _lower_hex(stored_content_sha)
        and canonical_json_sha256(content) == stored_content_sha,
        "prediction manifest content SHA-256 mismatch",
    )
    _require(
        manifest.get("schema")
        == "pathline_template_matching.scale_conditioned_outer_prediction.v1",
        "prediction manifest schema drifted",
    )
    _require(manifest.get("experiment") == EXPERIMENT, "prediction experiment drifted")
    _require(
        manifest.get("phase") == "outer_prediction_closed_before_outer_reference_open"
        and manifest.get("outer_reference_opened") is False,
        "prediction/reference phase contract drifted",
    )
    _require(manifest.get("config_sha256") == plan.sha256, "prediction config drifted")
    _require(manifest.get("outer_family") == outer_family, "prediction outer family drifted")
    _require(
        manifest.get("dataset_names_by_code") == list(plan.families[outer_family]),
        "prediction dataset codebook drifted",
    )
    _require(
        manifest.get("selected_candidate") == _candidate_payload(candidate),
        "prediction candidate drifted",
    )
    _require(
        _lower_hex(selected_candidate_sha256)
        and manifest.get("selected_candidate_file_sha256")
        == selected_candidate_sha256,
        "prediction selected-candidate identity drifted",
    )
    _require(manifest.get("prediction_file") == PREDICTION_FILE, "prediction filename drifted")
    prediction_size = manifest.get("prediction_file_size_bytes")
    prediction_sha = manifest.get("prediction_file_sha256")
    _require(
        isinstance(prediction_size, int)
        and not isinstance(prediction_size, bool)
        and prediction_size > 0
        and _lower_hex(prediction_sha),
        "prediction file identity is invalid",
    )
    prediction_path = output_dir / PREDICTION_FILE
    _stable_file_identity(prediction_path, prediction_size, str(prediction_sha))

    row_count = manifest.get("row_count")
    group_count = manifest.get("group_count")
    groups = manifest.get("groups")
    _require(
        isinstance(row_count, int)
        and not isinstance(row_count, bool)
        and row_count > 0,
        "prediction row count is invalid",
    )
    _require(
        isinstance(group_count, int)
        and not isinstance(group_count, bool)
        and group_count > 0
        and isinstance(groups, list)
        and len(groups) == group_count,
        "prediction group contract drifted",
    )
    expected_start = 0
    for group in groups:
        _require(isinstance(group, Mapping), "prediction group must be a mapping")
        start = group.get("row_start")
        stop = group.get("row_stop_exclusive")
        count = group.get("row_count")
        _require(
            isinstance(start, int)
            and isinstance(stop, int)
            and isinstance(count, int)
            and not any(isinstance(value, bool) for value in (start, stop, count))
            and start == expected_start
            and stop == start + count
            and count > 0,
            "prediction group row partition drifted",
        )
        expected_start = stop
    _require(expected_start == row_count, "prediction groups do not cover every row")

    array_hashes = manifest.get("prediction_array_sha256")
    _require(
        isinstance(array_hashes, Mapping)
        and set(array_hashes) == set(PREDICTION_ARRAY_DTYPES),
        "prediction array hash member set drifted",
    )
    with np.load(prediction_path, allow_pickle=False) as archive:
        _require(
            set(archive.files) == set(PREDICTION_ARRAY_DTYPES),
            "persisted outer prediction member set drifted",
        )
        arrays = {
            name: np.ascontiguousarray(np.asarray(archive[name]))
            for name in PREDICTION_ARRAY_DTYPES
        }
    for name, expected_dtype in PREDICTION_ARRAY_DTYPES.items():
        values = arrays[name]
        _require(
            values.dtype == expected_dtype and values.shape == (row_count,),
            f"persisted outer prediction {name} contract drifted",
        )
        stored_hash = array_hashes.get(name)
        _require(
            _lower_hex(stored_hash)
            and canonical_array_sha256(values) == stored_hash,
            f"persisted outer prediction {name} SHA-256 mismatch",
        )
    support_state_count = (
        arrays["retrieval_supported"].astype(np.int8)
        + arrays["spatial_imputed"].astype(np.int8)
        + arrays["spatial_unimputable"].astype(np.int8)
    )
    _require(
        np.array_equal(support_state_count, np.ones(row_count, dtype=np.int8)),
        "persisted outer prediction support states are not exclusive and exhaustive",
    )
    _require(
        np.isfinite(arrays["raw_negative_distance"]).all()
        and np.isfinite(arrays["spatial_score"]).all()
        and np.isfinite(arrays["spatial_denominator"]).all(),
        "persisted outer prediction contains non-finite numeric values",
    )
    return VerifiedOuterPrediction(
        manifest_path=manifest_path,
        manifest_file_sha256=str(manifest_identity["sha256"]),
        prediction_file_sha256=str(prediction_sha),
        manifest=manifest,
        arrays=arrays,
    )


def _load_outer_reference_after_prediction(
    plan: Plan,
    cache: CacheProjection,
    verified_prediction: VerifiedOuterPrediction,
) -> np.ndarray:
    _require(
        isinstance(verified_prediction, VerifiedOuterPrediction),
        "outer reference access requires an authenticated prediction token",
    )
    _stable_file_identity(cache.row.path, cache.row.size_bytes, cache.row.sha256)
    with np.load(cache.row.path, allow_pickle=False) as archive:
        _require("valid_labels" in archive.files and "metadata_json" in archive.files, f"{cache.row.path}: reference members missing")
        labels = np.asarray(archive["valid_labels"])
        metadata_scalar = np.asarray(archive["metadata_json"])
    _require(metadata_scalar.shape == (), f"{cache.row.path}: metadata is not scalar")
    metadata = json.loads(str(metadata_scalar.item()))
    _require(metadata.get("schema") == plan.cache_schema, f"{cache.row.path}: reference cache schema drifted")
    _require(metadata.get("dataset") == cache.row.dataset, f"{cache.row.path}: reference dataset drifted")
    _require(metadata.get("physical_family") == cache.row.family, f"{cache.row.path}: reference family drifted")
    _require(metadata.get("split") == "train", f"{cache.row.path}: reference split drifted")
    _require(metadata.get("config_sha256") == plan.parent_config_sha256, f"{cache.row.path}: reference config drifted")
    _require(metadata.get("descriptor_id") == plan.descriptor_id, f"{cache.row.path}: reference descriptor drifted")
    _require(labels.dtype == np.bool_ and labels.shape == (cache.count,), f"{cache.row.path}: reference label contract drifted")
    _require(
        metadata.get("array_sha256", {}).get("valid_labels")
        == canonical_array_sha256(labels),
        f"{cache.row.path}: reference label hash mismatch",
    )
    return np.ascontiguousarray(labels)


def _evaluate_outer_prediction(
    plan: Plan,
    output_dir: Path,
    outer_family: str,
    caches: Sequence[CacheProjection],
    candidate: CandidateSpec,
    selected_candidate_sha256: str,
    expected_prediction_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    verified_prediction = _verify_outer_prediction_artifacts(
        plan,
        output_dir,
        outer_family,
        candidate,
        selected_candidate_sha256,
        expected_prediction_manifest_sha256,
    )
    arrays = verified_prediction.arrays
    labels_by_identity: dict[tuple[str, int], np.ndarray] = {}
    for cache in caches:
        labels_by_identity[(cache.row.dataset, cache.row.source_ordinal)] = (
            _load_outer_reference_after_prediction(plan, cache, verified_prediction)
        )
    rows: list[dict[str, Any]] = []
    dataset_codes = {dataset: index for index, dataset in enumerate(plan.families[outer_family])}
    labels_all_parts: list[np.ndarray] = []
    for cache in caches:
        labels_cache = labels_by_identity[(cache.row.dataset, cache.row.source_ordinal)]
        for block_index, block_name in enumerate(BLOCK_NAMES):
            cache_selected = np.asarray(cache.block_indices == block_index)
            prediction_selected = (
                (arrays["dataset_code"] == dataset_codes[cache.row.dataset])
                & (arrays["source_ordinal"] == cache.row.source_ordinal)
                & (arrays["block_index"] == block_index)
            )
            positions = np.flatnonzero(prediction_selected)
            labels = np.asarray(labels_cache[cache_selected], dtype=bool)
            _require(len(positions) == len(labels), "outer reference/prediction group length mismatch")
            _require(
                np.array_equal(arrays["center_index"][positions], cache.center_indices[cache_selected])
                and np.array_equal(arrays["assigned_row_index"][positions], cache.assigned_row_indices[cache_selected])
                and np.array_equal(arrays["scale_id"][positions], cache.scale_ids[cache_selected]),
                "outer reference/prediction identity mismatch",
            )
            labels_all_parts.append(labels)
            rows.append(
                _metric_row(
                    outer_family=outer_family,
                    inner_family=outer_family,
                    dataset=cache.row.dataset,
                    source_ordinal=cache.row.source_ordinal,
                    block=block_name,
                    candidate=candidate,
                    labels=labels,
                    scores=np.asarray(arrays["spatial_score"][positions], dtype=np.float64),
                    predictions=np.asarray(arrays["prediction"][positions], dtype=bool),
                    supported=np.asarray(arrays["retrieval_supported"][positions], dtype=bool),
                    imputed=np.asarray(arrays["spatial_imputed"][positions], dtype=bool),
                    unimputable=np.asarray(arrays["spatial_unimputable"][positions], dtype=bool),
                )
            )
    for row in rows:
        row["physical_family"] = row.pop("inner_family")
    group_macro_fields = (
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
    summary: dict[str, Any] = {
        "outer_family": outer_family,
        "candidate": _candidate_payload(candidate),
        "group_count": len(rows),
        "dataset_count": len(plan.families[outer_family]),
        **{
            field: float(np.nanmean([float(row[field]) for row in rows]))
            for field in group_macro_fields
        },
        "sample_count": sum(int(row["sample_count"]) for row in rows),
        "positive_count": sum(int(row["positive_count"]) for row in rows),
        "negative_count": sum(int(row["negative_count"]) for row in rows),
        "supported_count": sum(int(row["supported_count"]) for row in rows),
        "imputed_count": sum(int(row["imputed_count"]) for row in rows),
        "unimputable_count": sum(int(row["unimputable_count"]) for row in rows),
        "true_positive": sum(int(row["true_positive"]) for row in rows),
        "false_positive": sum(int(row["false_positive"]) for row in rows),
        "true_negative": sum(int(row["true_negative"]) for row in rows),
        "false_negative": sum(int(row["false_negative"]) for row in rows),
        "aggregation": "equal_dataset_source_block_groups_within_outer_family",
    }
    reference_audit = {
        "outer_reference_first_open_phase": "after_prediction_file_and_manifest_authenticated",
        "prediction_manifest_path": str(verified_prediction.manifest_path),
        "prediction_manifest_sha256": verified_prediction.manifest_file_sha256,
        "prediction_file_sha256": verified_prediction.prediction_file_sha256,
        "prediction_array_sha256_verified": True,
        "cache_members_opened": ["valid_labels", "metadata_json"],
        "reference_array_sha256": {
            f"{cache.row.dataset}/source_{cache.row.source_ordinal}": canonical_array_sha256(
                labels_by_identity[(cache.row.dataset, cache.row.source_ordinal)]
            )
            for cache in caches
        },
    }
    return rows, summary, reference_audit


def _environment_audit(device: str) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "requested_device": device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        audit.update(
            {
                "cuda_version": torch.version.cuda,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(torch.device(device)),
                "cuda_device_capability": list(
                    torch.cuda.get_device_capability(torch.device(device))
                ),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            }
        )
    return audit


def _configure_execution(device: str) -> None:
    selected = torch.device(device)
    _require(selected.type in {"cpu", "cuda"}, "device must be cpu or cuda")
    if selected.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        _require(
            selected.index is None or selected.index < torch.cuda.device_count(),
            "requested CUDA device is unavailable",
        )
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def run(
    config_path: str | Path,
    outer_family: str,
    output_dir: str | Path,
    *,
    device: str,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    plan = load_plan(config_path)
    _require(outer_family in plan.family_order, f"unknown outer family: {outer_family}")
    if expected_config_sha256 is not None:
        _require(plan.sha256 == expected_config_sha256, "frozen config SHA-256 mismatch")
    git_commit, dirty = _git_identity()
    _require(not dirty, "Ibex numerical run requires a clean committed Git worktree")
    _configure_execution(device)
    destination = Path(output_dir).resolve()
    _require(not destination.exists(), f"immutable output directory exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    print(f"[{_utc_now()}] {EXPERIMENT} outer={outer_family} commit={git_commit}", flush=True)

    cache_rows, input_manifest_identity = load_cache_rows(plan)
    nonouter_rows = [row for row in cache_rows if row.family != outer_family]
    outer_rows = [row for row in cache_rows if row.family == outer_family]
    _require(nonouter_rows and outer_rows, "outer split produced an empty side")
    # The outer cache is not opened here.  Inner selection sees only the other
    # four physical families and their development labels.
    nonouter_caches = [
        load_cache_projection(plan, row, include_labels=True) for row in nonouter_rows
    ]
    print(
        f"[{_utc_now()}] loaded {len(nonouter_caches)} nonouter caches; starting inner folds",
        flush=True,
    )
    inner_rows, candidates, inner_fit_audits = _inner_metric_rows(
        plan, nonouter_caches, outer_family, device=device
    )
    summaries, selected, selected_summary = _aggregate_and_select(
        plan, inner_rows, candidates
    )
    inner_fields = list(inner_rows[0])
    inner_metrics_sha = _atomic_csv(
        destination / "inner_group_metrics.csv", inner_fields, inner_rows
    )
    summary_fields = list(summaries[0])
    inner_summary_sha = _atomic_csv(
        destination / "inner_candidate_summary.csv", summary_fields, summaries
    )
    inner_fit_sha = _atomic_json(
        destination / "inner_fit_audits.json",
        {
            "experiment": EXPERIMENT,
            "outer_family": outer_family,
            "fit_count": len(inner_fit_audits),
            "fits": inner_fit_audits,
        },
    )
    print(
        f"[{_utc_now()}] selected {selected.candidate_id} from {len(candidates)} candidates",
        flush=True,
    )

    # Refit only on the four nonouter families.  The fitted scaler and support
    # audit are frozen in selected_candidate.json before any outer NPZ member
    # is opened.
    final_model = _fit_negative_model(nonouter_caches, selected.representation)
    final_fit_audit = final_model.fit_audit
    final_scale_counts = {
        int(key): int(value)
        for key, value in final_fit_audit["scale_counts"].items()
    }
    _require(
        set(final_scale_counts) == set(range(2000))
        and min(final_scale_counts.values()) >= max(plan.ks),
        "outer final fit must support every one of 2000 scales at k=31",
    )
    selected_payload: dict[str, Any] = {
        "schema": "pathline_template_matching.nested_inner_selection.v1",
        "experiment": EXPERIMENT,
        "phase": "selection_and_final_fit_closed_before_outer_feature_open",
        "created_utc": _utc_now(),
        "git_commit": git_commit,
        "config_path": str(plan.path),
        "config_sha256": plan.sha256,
        "input_manifest": input_manifest_identity,
        "outer_family": outer_family,
        "outer_cache_members_opened_before_this_artifact": [],
        "inner_families": [family for family in plan.family_order if family != outer_family],
        "candidate_count": len(candidates),
        "selected_candidate": _candidate_payload(selected),
        "selected_inner_summary": selected_summary,
        "selection_order": [
            "highest_family_macro_f1",
            "highest_family_macro_average_precision",
            "highest_family_macro_balanced_accuracy",
            "highest_family_macro_precision",
            "highest_family_macro_recall",
            "lexicographically_smallest_candidate_id",
        ],
        "inner_group_metrics_file_sha256": inner_metrics_sha,
        "inner_candidate_summary_file_sha256": inner_summary_sha,
        "inner_fit_audits_file_sha256": inner_fit_sha,
        "final_fit_families": [family for family in plan.family_order if family != outer_family],
        "final_fit_audit": final_fit_audit,
    }
    selected_payload["selection_content_sha256"] = canonical_json_sha256(
        selected_payload
    )
    selected_sha = _atomic_json(destination / "selected_candidate.json", selected_payload)

    # Only now may the outer prediction projection open feature/identity
    # members.  metadata_json and valid_labels both remain unopened.
    outer_caches = [
        load_cache_projection(plan, row, include_labels=False) for row in outer_rows
    ]
    arrays, outer_groups = _build_outer_prediction_arrays(
        plan,
        outer_family,
        outer_caches,
        selected,
        final_model,
        device=device,
    )
    prediction_manifest, prediction_manifest_sha = _write_outer_prediction(
        plan,
        destination,
        outer_family,
        outer_caches,
        selected,
        selected_sha,
        arrays,
        outer_groups,
        git_commit=git_commit,
        device=device,
    )
    del final_model, arrays
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[{_utc_now()}] outer prediction closed; opening reference members for evaluation",
        flush=True,
    )
    outer_metric_rows, outer_summary, reference_audit = _evaluate_outer_prediction(
        plan,
        destination,
        outer_family,
        outer_caches,
        selected,
        selected_sha,
        prediction_manifest_sha,
    )
    outer_metrics_sha = _atomic_csv(
        destination / "outer_group_metrics.csv",
        list(outer_metric_rows[0]),
        outer_metric_rows,
    )
    outer_summary_sha = _atomic_json(destination / "outer_summary.json", outer_summary)
    reference_audit_sha = _atomic_json(
        destination / "outer_reference_access_audit.json", reference_audit
    )
    artifact_names = (
        "inner_group_metrics.csv",
        "inner_candidate_summary.csv",
        "inner_fit_audits.json",
        "selected_candidate.json",
        PREDICTION_FILE,
        PREDICTION_MANIFEST_FILE,
        "outer_group_metrics.csv",
        "outer_summary.json",
        "outer_reference_access_audit.json",
    )
    result_manifest: dict[str, Any] = {
        "schema": "pathline_template_matching.scale_conditioned_outer_result.v1",
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
        "prediction_manifest_file_sha256": prediction_manifest_sha,
        "outer_group_metrics_file_sha256": outer_metrics_sha,
        "outer_summary_file_sha256": outer_summary_sha,
        "outer_reference_access_audit_file_sha256": reference_audit_sha,
        "outer_summary": outer_summary,
        "prediction_before_reference_contract": {
            "prediction_manifest_phase": prediction_manifest["phase"],
            "prediction_manifest_outer_reference_opened": prediction_manifest[
                "outer_reference_opened"
            ],
            "reference_first_open_phase": reference_audit[
                "outer_reference_first_open_phase"
            ],
        },
        "environment": _environment_audit(device),
        "artifacts": {
            name: {
                "size_bytes": (destination / name).stat().st_size,
                "sha256": sha256_file(destination / name),
            }
            for name in artifact_names
        },
    }
    result_manifest["manifest_content_sha256"] = canonical_json_sha256(
        result_manifest
    )
    result_sha = _atomic_json(destination / "result_manifest.json", result_manifest)
    completion = {
        "schema": "pathline_template_matching.run_complete.v1",
        "experiment": EXPERIMENT,
        "outer_family": outer_family,
        "git_commit": git_commit,
        "config_sha256": plan.sha256,
        "result_manifest_file": "result_manifest.json",
        "result_manifest_file_sha256": result_sha,
        "completed_utc": _utc_now(),
    }
    completion["completion_content_sha256"] = canonical_json_sha256(completion)
    _atomic_json(destination / "RUN_COMPLETE.json", completion)
    print(
        f"[{_utc_now()}] completed outer={outer_family} F1={outer_summary['f1']:.6f}",
        flush=True,
    )
    return result_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "Verify_ScaleConditionedRetrieval_1.1.yaml"),
    )
    parser.add_argument("--outer-family", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-config-sha256")
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

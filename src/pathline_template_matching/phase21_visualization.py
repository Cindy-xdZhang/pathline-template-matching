"""Audited fixed triptychs for ``mainExp_TemplateMatching_2.1``.

This module is deliberately downstream of numerical evaluation.  By default it
consumes one already validated source-ordinal-2 test cache and an ordered FMT
exact-1NN assignment.  A versioned downstream experiment may explicitly pass a
different allowed dataset set and source split; the default test-only contract
does not change.  Display-pathline selection never receives predictions or metrics.
Panels two and three retain every valid query seed; only the explanatory
center-pathline overlay in panel one is reduced to a frozen 120-positive plus
120-negative deterministic maximin sample.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np
from skimage import measure

from .portable_flow import (
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)
from .visualization import DEFAULT_DPI, render_template_matching_triptych


EXPERIMENT = "mainExp_TemplateMatching_2.1"
EXPERIMENT31 = "mainExp_TemplateMatching_3.1"
SUPPORTED_EXPERIMENTS = (EXPERIMENT, EXPERIMENT31)
FIXED_SOURCE_ORDINAL = 2
DISPLAY_PATHLINE_COUNT = 240
DISPLAY_PER_CLASS = 120
DISPLAY_SELECTION_SEED = 15068
SCALE_AXIS_SIZE = 10
SCALE_COUNT = SCALE_AXIS_SIZE**3
TEST_DATASETS = ("tangaroa", "smokeBuoyancy")

DATASET_TITLES = {
    "cylinder3d": "Half-cylinder Re160",
    "halfcylinderRe640": "Half-cylinder Re640",
    "halfcylinderRe6400": "Half-cylinder Re6400",
    "boeing747": "Boeing 747",
    "tangaroa": "Tangaroa",
    "smokeBuoyancy": "Smoke buoyancy",
}
DATASET_VIEWS = {
    "cylinder3d": (22.0, -62.0),
    "halfcylinderRe640": (22.0, -62.0),
    "halfcylinderRe6400": (22.0, -62.0),
    "boeing747": (21.0, -58.0),
    "tangaroa": (23.0, -62.0),
    "smokeBuoyancy": (22.0, -58.0),
}

SCENE_ARRAY_NAMES = (
    "bounds",
    "seeds",
    "reference",
    "prediction",
    "valid_seed_index",
    "valid_scale_id",
    "display_pathlines",
    "selected_query_row",
    "selected_seed_index",
    "selected_reference",
    "ivd_mesh_vertices",
    "ivd_mesh_faces",
    "ivd_mesh_normals",
    "ivd_mesh_values",
    "ivd_mesh_level",
    "metadata_json",
)
SCENE31_EXTRA_ARRAY_NAMES = (
    "valid_assigned_row_index",
    "valid_center_seed_index",
    "valid_scale_block_index",
    "selected_assigned_row_index",
    "selected_center_seed_index",
    "selected_scale_block_index",
)


@dataclass(frozen=True, slots=True)
class ValidatedPhase21VisualizationInput:
    """One fixed test query slice with an explicitly ordered prediction."""

    dataset: str
    source_ordinal: int
    source_index: int
    raw_features: np.ndarray
    valid_seed_index: np.ndarray
    valid_scale_id: np.ndarray
    valid_labels: np.ndarray
    valid_seeds: np.ndarray
    center_sample_time: np.ndarray
    ivd_volume: np.ndarray
    prediction: np.ndarray
    metadata: dict[str, Any]
    valid_assigned_row_index: np.ndarray
    valid_center_seed_index: np.ndarray
    valid_scale_block_index: np.ndarray
    scale_block_id: str


@dataclass(frozen=True, slots=True)
class SceneArtifact:
    """Verified scene arrays and both evidence-file digests."""

    scene: dict[str, Any]
    metadata: dict[str, Any]
    npz_path: Path
    manifest_path: Path
    npz_sha256: str
    manifest_file_sha256: str


@dataclass(frozen=True, slots=True)
class RenderedTriptychArtifacts:
    """Paths and metadata for one non-overwriting PNG/PDF render bundle."""

    png_path: Path
    pdf_path: Path
    svg_path: Path | None
    metadata_path: Path
    alignment_path: Path
    metadata: dict[str, Any]


def _strict_binary(values: Any, *, name: str, count: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "buif":
        raise ValueError(f"{name} must be a one-dimensional numeric 0/1 array")
    if count is not None and array.shape != (int(count),):
        raise ValueError(f"{name} must have shape {(int(count),)}, got {array.shape}")
    if not np.isfinite(array).all() or not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must contain only finite 0/1 values")
    return np.ascontiguousarray(array, dtype=np.bool_)


def _integer_vector(
    values: Any, *, name: str, count: int, dtype: np.dtype[Any]
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (int(count),) or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer vector with shape {(int(count),)}")
    return np.ascontiguousarray(array, dtype=dtype)


def ordered_fmt_prediction(
    labels: Any, valid_seed_index: Any, valid_scale_id: Any
) -> dict[str, np.ndarray]:
    """Construct the explicit row-identity contract required by scene building."""

    predicted = _strict_binary(labels, name="FMT prediction")
    count = len(predicted)
    return {
        "labels": predicted,
        "valid_seed_index": _integer_vector(
            valid_seed_index,
            name="prediction valid_seed_index",
            count=count,
            dtype=np.dtype(np.int64),
        ),
        "valid_scale_id": _integer_vector(
            valid_scale_id,
            name="prediction valid_scale_id",
            count=count,
            dtype=np.dtype(np.int32),
        ),
    }


def _prediction_arrays(
    prediction: Mapping[str, Any], *, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(prediction, Mapping):
        raise TypeError(
            "FMT prediction must be a mapping with labels and row-identity copies"
        )
    required = {"labels", "valid_seed_index", "valid_scale_id"}
    missing = required - set(prediction)
    if missing:
        raise KeyError(f"FMT prediction is missing ordered identity fields: {sorted(missing)}")
    labels = _strict_binary(prediction["labels"], name="FMT prediction", count=count)
    seed_index = _integer_vector(
        prediction["valid_seed_index"],
        name="prediction valid_seed_index",
        count=count,
        dtype=np.dtype(np.int64),
    )
    scale_id = _integer_vector(
        prediction["valid_scale_id"],
        name="prediction valid_scale_id",
        count=count,
        dtype=np.dtype(np.int32),
    )
    return labels, seed_index, scale_id


def validate_phase21_visualization_input(
    cache: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    allowed_datasets: tuple[str, ...] = TEST_DATASETS,
    required_split: str = "test",
    required_source_ordinal: int = FIXED_SOURCE_ORDINAL,
) -> ValidatedPhase21VisualizationInput:
    """Validate the fixed cache/prediction order and visualization fields."""

    if not isinstance(cache, Mapping):
        raise TypeError("cache must be a mapping returned by the verified cache loader")
    required = {
        "raw_features",
        "valid_seed_index",
        "valid_scale_id",
        "valid_labels",
        "seeds_xyz",
        "center_sample_time",
        "ivd_volume",
        "metadata",
    }
    missing = required - set(cache)
    if missing:
        raise KeyError(f"cache is missing visualization fields: {sorted(missing)}")
    metadata_value = cache["metadata"]
    if not isinstance(metadata_value, Mapping):
        raise ValueError("cache metadata must be a mapping")
    metadata = dict(metadata_value)
    dataset = str(metadata.get("dataset", "")).strip()
    allowed = tuple(str(value) for value in allowed_datasets)
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError("allowed visualization datasets must be unique and non-empty")
    if dataset not in allowed:
        raise ValueError(
            f"triptych dataset is outside the explicitly allowed set: {dataset!r}"
        )
    if dataset not in DATASET_TITLES or dataset not in DATASET_VIEWS:
        raise ValueError(f"triptych display metadata is not registered: {dataset!r}")
    experiment = str(metadata.get("experiment", ""))
    if experiment not in SUPPORTED_EXPERIMENTS:
        raise ValueError("cache experiment is not a supported template-matching run")
    if not required_split or metadata.get("split") != required_split:
        raise ValueError(
            f"triptych cache split must be {required_split!r}, "
            f"got {metadata.get('split')!r}"
        )
    source_ordinal = int(metadata.get("source_ordinal", -1))
    if source_ordinal != int(required_source_ordinal):
        raise ValueError(
            f"triptychs require source ordinal {int(required_source_ordinal)}, "
            f"got {source_ordinal}"
        )
    source_index = int(metadata.get("source_index", -1))
    if source_index < 0:
        raise ValueError("cache source_index must be non-negative")

    raw = np.asarray(cache["raw_features"])
    if raw.ndim != 2 or raw.shape[1:] != (672,) or raw.dtype != np.float32:
        raise ValueError("raw_features must be float32 [N,672]")
    if not np.isfinite(raw).all():
        raise ValueError("raw_features contains NaN or Inf")
    count = len(raw)
    valid_seed_index = _integer_vector(
        cache["valid_seed_index"],
        name="cache valid_seed_index",
        count=count,
        dtype=np.dtype(np.int64),
    )
    valid_scale_id = _integer_vector(
        cache["valid_scale_id"],
        name="cache valid_scale_id",
        count=count,
        dtype=np.dtype(np.int32),
    )
    valid_labels = _strict_binary(
        cache["valid_labels"], name="cache valid_labels", count=count
    )
    if count == 0 or len(np.unique(valid_seed_index)) != count:
        raise ValueError("valid_seed_index must be non-empty and unique")
    if np.any(np.diff(valid_seed_index) <= 0):
        raise ValueError("valid_seed_index must preserve increasing cache row order")
    scale_count = 2_000 if experiment == EXPERIMENT31 else SCALE_COUNT
    if valid_scale_id.min() < 0 or valid_scale_id.max() >= scale_count:
        raise ValueError("valid_scale_id is outside the frozen scale union")

    if experiment == EXPERIMENT31:
        identity_fields = {
            "valid_assigned_row_index",
            "valid_center_seed_index",
            "valid_scale_block_index",
        }
        missing_identity = identity_fields - set(cache)
        if missing_identity:
            raise KeyError(
                f"3.1 cache is missing explicit row identities: {sorted(missing_identity)}"
            )
        valid_assigned_row_index = _integer_vector(
            cache["valid_assigned_row_index"],
            name="valid_assigned_row_index",
            count=count,
            dtype=np.dtype(np.int64),
        )
        valid_center_seed_index = _integer_vector(
            cache["valid_center_seed_index"],
            name="valid_center_seed_index",
            count=count,
            dtype=np.dtype(np.int64),
        )
        valid_scale_block_index = _integer_vector(
            cache["valid_scale_block_index"],
            name="valid_scale_block_index",
            count=count,
            dtype=np.dtype(np.int8),
        )
        center_count = int(metadata.get("unique_center_seed_count", -1))
        block_index = int(metadata.get("visualization_scale_block_index", -1))
        scale_block_id = str(metadata.get("visualization_scale_block_id", ""))
        scale_start = int(metadata.get("visualization_scale_id_start", -1))
        scale_stop = int(metadata.get("visualization_scale_id_stop_exclusive", -1))
        if center_count < 1 or block_index not in (0, 1) or not scale_block_id:
            raise ValueError("3.1 visualization block identity is incomplete")
        if not np.array_equal(valid_assigned_row_index, valid_seed_index):
            raise ValueError("3.1 assigned-row identity disagrees with legacy alias")
        if not np.array_equal(
            valid_center_seed_index, valid_assigned_row_index % center_count
        ):
            raise ValueError("3.1 center-seed identity disagrees with assigned rows")
        if not np.array_equal(
            valid_scale_block_index,
            valid_assigned_row_index // center_count,
        ) or not np.all(valid_scale_block_index == block_index):
            raise ValueError("3.1 visualization mixes scale blocks")
        if np.any(valid_scale_id < scale_start) or np.any(valid_scale_id >= scale_stop):
            raise ValueError("3.1 visualization scale IDs escape the selected block")
    else:
        valid_assigned_row_index = valid_seed_index
        valid_center_seed_index = valid_seed_index
        valid_scale_block_index = np.zeros(count, dtype=np.int8)
        scale_block_id = "legacy_2_1"

    all_seeds = np.asarray(cache["seeds_xyz"])
    if all_seeds.ndim != 2 or all_seeds.shape[1:] != (3,) or all_seeds.dtype != np.float64:
        raise ValueError("seeds_xyz must be float64 [assigned,3]")
    if not np.isfinite(all_seeds).all():
        raise ValueError("seeds_xyz contains NaN or Inf")
    if valid_seed_index.min() < 0 or valid_seed_index.max() >= len(all_seeds):
        raise ValueError("valid_seed_index points outside seeds_xyz")
    valid_seeds = np.ascontiguousarray(all_seeds[valid_seed_index], dtype=np.float64)
    center_time = np.asarray(cache["center_sample_time"])
    if center_time.shape != (count, 32) or center_time.dtype != np.float32:
        raise ValueError("center_sample_time must be float32 [N,32]")
    if not np.isfinite(center_time).all() or np.any(np.diff(center_time, axis=1) < -1e-7):
        raise ValueError("center_sample_time must be finite and non-decreasing")
    source_time = float(metadata.get("source_time", np.nan))
    if not np.isfinite(source_time):
        raise ValueError("cache metadata source_time must be finite")
    # The numerical integrator resets every loaded window to relative t=0.
    # Absolute source time is restored only in the explanatory scene.
    if not np.allclose(center_time[:, 0], 0.0, rtol=0.0, atol=1e-7):
        raise ValueError("center_sample_time must start at the cache-relative time zero")
    centered = raw.reshape(count, 7, 32, 3)
    if not np.array_equal(centered[:, 0, 0], np.zeros((count, 3), dtype=np.float32)):
        raise ValueError("centered Raw primitives do not start at the center-seed origin")

    ivd = np.asarray(cache["ivd_volume"])
    if ivd.ndim != 3 or min(ivd.shape) < 2 or ivd.dtype != np.float32:
        raise ValueError("ivd_volume must be float32 [Z,Y,X] with axes >=2")
    if not np.isfinite(ivd).all():
        raise ValueError("ivd_volume contains NaN or Inf")
    loaded_shape = metadata.get("loaded_shape_TZYXC")
    if not isinstance(loaded_shape, list) or len(loaded_shape) != 5:
        raise ValueError("metadata loaded_shape_TZYXC is invalid")
    if tuple(int(value) for value in loaded_shape[1:4]) != ivd.shape:
        raise ValueError("ivd_volume shape disagrees with loaded_shape_TZYXC")

    predicted, prediction_seed_index, prediction_scale_id = _prediction_arrays(
        prediction, count=count
    )
    if not np.array_equal(prediction_seed_index, valid_seed_index):
        raise ValueError("FMT prediction row order disagrees with valid_seed_index")
    if not np.array_equal(prediction_scale_id, valid_scale_id):
        raise ValueError("FMT prediction row order disagrees with valid_scale_id")

    stored_hashes = metadata.get("array_sha256")
    if isinstance(stored_hashes, Mapping):
        checked = {
            "raw_features": raw,
            "valid_labels": valid_labels,
            "valid_seed_index": valid_seed_index,
            "valid_scale_id": valid_scale_id,
            "center_sample_time": center_time,
            "seeds_xyz": all_seeds,
            "ivd_volume": ivd,
        }
        if experiment == EXPERIMENT31:
            checked.update(
                {
                    "valid_assigned_row_index": valid_assigned_row_index,
                    "valid_center_seed_index": valid_center_seed_index,
                    "valid_scale_block_index": valid_scale_block_index,
                }
            )
        mismatches = {
            name: (stored_hashes.get(name), canonical_array_sha256(values))
            for name, values in checked.items()
            if stored_hashes.get(name) != canonical_array_sha256(values)
        }
        if mismatches:
            raise ValueError(f"cache array SHA-256 mismatch before visualization: {mismatches}")

    return ValidatedPhase21VisualizationInput(
        dataset=dataset,
        source_ordinal=source_ordinal,
        source_index=source_index,
        raw_features=np.ascontiguousarray(raw),
        valid_seed_index=valid_seed_index,
        valid_scale_id=valid_scale_id,
        valid_labels=valid_labels,
        valid_seeds=valid_seeds,
        center_sample_time=np.ascontiguousarray(center_time),
        ivd_volume=np.ascontiguousarray(ivd),
        prediction=predicted,
        metadata=metadata,
        valid_assigned_row_index=valid_assigned_row_index,
        valid_center_seed_index=valid_center_seed_index,
        valid_scale_block_index=valid_scale_block_index,
        scale_block_id=scale_block_id,
    )


def scale_axis_indices(scale_id: Any) -> np.ndarray:
    """Decode ``((dx*10)+ds)*10+arc`` into the three frozen indices."""

    raw = np.asarray(scale_id)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("scale_id must be a one-dimensional integer array")
    values = raw.astype(np.int64, copy=False)
    if values.size and (values.min() < 0 or values.max() >= SCALE_COUNT):
        raise ValueError("scale_id is outside [0,1000)")
    return np.ascontiguousarray(
        np.column_stack((values // 100, (values // 10) % 10, values % 10)),
        dtype=np.int16,
    )


def _maximin_indices(points: np.ndarray, count: int, *, derived_seed: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    count = int(count)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("maximin points must be a non-empty finite matrix")
    if count < 1 or len(values) < count:
        raise ValueError("maximin candidate count is smaller than the requested count")
    rng = np.random.Generator(np.random.PCG64(np.uint64(derived_seed)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.integers(0, len(values)))
    tie_order = np.empty(len(values), dtype=np.int64)
    tie_order[rng.permutation(len(values))] = np.arange(len(values), dtype=np.int64)
    minimum = np.sum((values - values[selected[0]]) ** 2, axis=1)
    minimum[selected[0]] = -1.0
    for position in range(1, count):
        maximum = float(np.max(minimum))
        tied = np.flatnonzero(minimum == maximum)
        selected[position] = int(tied[np.argmin(tie_order[tied])])
        distance = np.sum((values - values[selected[position]]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[selected[: position + 1]] = -1.0
    return selected


def select_display_query_rows(
    seeds_xyz: Any,
    reference: Any,
    scale_id: Any,
    *,
    domain_bounds: Any,
    dataset: str,
    source_ordinal: int,
    base_seed: int = DISPLAY_SELECTION_SEED,
    scale_id_start: int = 0,
    identity_scope: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select 120 rows per class in normalized XYZ plus three scale indices.

    This function intentionally has no prediction or metric argument.  Its
    complete dependency set is recorded in the returned audit.
    """

    seeds = np.asarray(seeds_xyz, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1:] != (3,) or not np.isfinite(seeds).all():
        raise ValueError("seeds_xyz must be finite [N,3]")
    labels = _strict_binary(reference, name="reference", count=len(seeds))
    scales = _integer_vector(
        scale_id,
        name="scale_id",
        count=len(seeds),
        dtype=np.dtype(np.int32),
    )
    scale_id_start = int(scale_id_start)
    decoded = scale_axis_indices(scales.astype(np.int64) - scale_id_start)
    bounds = np.asarray(domain_bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not np.all(
        bounds[1] > bounds[0]
    ):
        raise ValueError("domain_bounds must be finite increasing [2,3]")
    normalized_xyz = (seeds - bounds[0]) / (bounds[1] - bounds[0])
    tolerance = 1e-7
    if np.any(normalized_xyz < -tolerance) or np.any(normalized_xyz > 1.0 + tolerance):
        raise ValueError("valid query seed lies outside metadata domain bounds")
    selection_space = np.column_stack(
        (np.clip(normalized_xyz, 0.0, 1.0), decoded.astype(np.float64) / 9.0)
    )
    chosen_blocks = []
    class_audit: dict[str, Any] = {}
    for class_value, class_name in ((False, "negative"), (True, "positive")):
        candidates = np.flatnonzero(labels == class_value)
        if len(candidates) < DISPLAY_PER_CLASS:
            raise RuntimeError(
                f"{dataset} source ordinal {source_ordinal} has only "
                f"{len(candidates)} {class_name} valid queries; "
                f"{DISPLAY_PER_CLASS} are required"
            )
        identity_hash = hashlib.sha256()
        identity_hash.update(
            f"{int(base_seed)}|{dataset}|{int(source_ordinal)}|{class_name}".encode(
                "utf-8"
            )
        )
        if identity_scope is not None:
            identity_hash.update(f"|{identity_scope}".encode("utf-8"))
        identity_hash.update(
            canonical_array_sha256(
                np.ascontiguousarray(selection_space[candidates], dtype=np.float64)
            ).encode("ascii")
        )
        derived_seed = int.from_bytes(identity_hash.digest()[:8], "little")
        local = _maximin_indices(
            selection_space[candidates], DISPLAY_PER_CLASS, derived_seed=derived_seed
        )
        selected = candidates[local]
        chosen_blocks.append(selected)
        class_audit[class_name] = {
            "candidate_count": int(len(candidates)),
            "selected_count": int(len(selected)),
            "derived_pcg64_seed_uint64": int(derived_seed),
            "candidate_selection_space_sha256": canonical_array_sha256(
                np.ascontiguousarray(selection_space[candidates], dtype=np.float64)
            ),
            "selected_query_row_sha256": canonical_array_sha256(
                np.ascontiguousarray(selected, dtype=np.int64)
            ),
            "selected_unique_scale_count": int(len(np.unique(scales[selected]))),
            "selected_scale_axis_index_coverage": {
                axis: sorted(np.unique(decoded[selected, index]).astype(int).tolist())
                for index, axis in enumerate(("dx", "ds", "arc"))
            },
        }
    # A deterministic class-alternating order prevents one class from becoming
    # a drawing-order block while keeping the exact 120/120 contract.
    selected_rows = np.column_stack(chosen_blocks).reshape(-1).astype(np.int64)
    if len(selected_rows) != DISPLAY_PATHLINE_COUNT or len(np.unique(selected_rows)) != len(
        selected_rows
    ):
        raise RuntimeError("display selection is not an exact unique 240-row set")
    if int(labels[selected_rows].sum()) != DISPLAY_PER_CLASS:
        raise RuntimeError("display selection lost its exact class balance")
    audit = {
        "method": "class-balanced deterministic maximin",
        "selection_dependencies": [
            "dataset",
            "source_ordinal",
            "fixed_base_seed",
            "reference_class",
            "normalized_seed_xyz",
            "dx_index",
            "ds_index",
            "arc_index",
        ],
        "forbidden_dependencies": ["FMT_prediction", "metric", "performance"],
        "base_seed": int(base_seed),
        "random_generator": "numpy.random.PCG64",
        "selection_space": "[normalized_x,normalized_y,normalized_z,dx_index/9,ds_index/9,arc_index/9]",
        "count": DISPLAY_PATHLINE_COUNT,
        "positive_count": DISPLAY_PER_CLASS,
        "negative_count": DISPLAY_PER_CLASS,
        "selected_query_row_sha256": canonical_array_sha256(selected_rows),
        "classes": class_audit,
    }
    if identity_scope is not None or scale_id_start != 0:
        audit.update(
            {
                "scale_id_start": scale_id_start,
                "identity_scope": identity_scope,
                "selection_space": (
                    "[normalized_x,normalized_y,normalized_z,"
                    "within_block_dx_index/9,within_block_ds_index/9,"
                    "within_block_arc_index/9]"
                ),
            }
        )
    return np.ascontiguousarray(selected_rows), audit


def _complete_ivd_mesh(
    ivd_volume: np.ndarray, metadata: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray | float], np.ndarray, dict[str, Any]]:
    spacing = np.asarray(metadata.get("spacing_xyz"), dtype=np.float64)
    lower = np.asarray(metadata.get("domain_min_xyz"), dtype=np.float64)
    upper = np.asarray(metadata.get("domain_max_xyz"), dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("metadata spacing_xyz must contain three positive values")
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.all(upper > lower)
    ):
        raise ValueError("metadata domain bounds are invalid")
    z_count, y_count, x_count = ivd_volume.shape
    expected_upper = lower + spacing * np.asarray(
        [x_count - 1, y_count - 1, z_count - 1], dtype=np.float64
    )
    tolerance = max(1e-6, float(np.max(spacing)) * 2e-5)
    if not np.allclose(expected_upper, upper, rtol=1e-4, atol=tolerance):
        raise ValueError(
            "metadata spacing/domain do not span the complete loaded IVD volume"
        )
    percentile = float(metadata.get("ivd_percentile", np.nan))
    if percentile != 95.0:
        raise ValueError("2.1 IVD mesh requires the frozen percentile 95")
    cached_level = float(metadata.get("ivd_threshold", np.nan))
    reconstructed_level = float(np.percentile(ivd_volume, 95.0))
    if not np.isfinite(cached_level) or not np.isclose(
        reconstructed_level, cached_level, rtol=1e-12, atol=1e-12
    ):
        raise ValueError(
            "cached IVD p95 level disagrees with the complete stored IVD volume"
        )
    volume_min = float(np.min(ivd_volume))
    volume_max = float(np.max(ivd_volume))
    if not volume_min < cached_level < volume_max:
        raise ValueError("IVD p95 level does not define a non-empty interior isosurface")
    dx, dy, dz = (float(value) for value in spacing)
    vertices_zyx, faces, normals_zyx, values = measure.marching_cubes(
        ivd_volume,
        level=cached_level,
        spacing=(dz, dy, dx),
        step_size=1,
        allow_degenerate=True,
        method="lewiner",
    )
    vertices = np.column_stack(
        (
            lower[0] + vertices_zyx[:, 2],
            lower[1] + vertices_zyx[:, 1],
            lower[2] + vertices_zyx[:, 0],
        )
    ).astype(np.float64)
    normals = np.column_stack(
        (normals_zyx[:, 2], normals_zyx[:, 1], normals_zyx[:, 0])
    ).astype(np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    values = np.ascontiguousarray(values, dtype=np.float32)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) < 3
        or not np.isfinite(vertices).all()
        or normals.shape != vertices.shape
        or not np.isfinite(normals).all()
        or values.shape != (len(vertices),)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("Marching Cubes returned invalid vertex attributes")
    if (
        faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(faces) < 1
        or int(faces.min()) < 0
        or int(faces.max()) >= len(vertices)
    ):
        raise RuntimeError("Marching Cubes returned invalid triangle indices")
    coordinate_tolerance = max(1e-9, float(np.max(spacing)) * 1e-5)
    if np.any(vertices < lower - coordinate_tolerance) or np.any(
        vertices > upper + coordinate_tolerance
    ):
        raise RuntimeError("Marching Cubes vertices escape metadata domain bounds")
    referenced_vertex_count = int(len(np.unique(faces)))
    if referenced_vertex_count != len(vertices):
        raise RuntimeError("Marching Cubes mesh contains unreferenced vertices")
    mesh = {
        "vertices": np.ascontiguousarray(vertices),
        "faces": faces,
        "normals": normals,
        "values": values,
        "level": cached_level,
    }
    bounds = np.stack((lower, upper))
    audit = {
        "method": "skimage.measure.marching_cubes",
        "algorithm": "Lewiner",
        "step_size": 1,
        "allow_degenerate": True,
        "triangle_subsampling": False,
        "complete_loaded_volume": True,
        "axis_input_order": "ZYX",
        "axis_output_order": "XYZ",
        "ivd_volume_shape_zyx": list(ivd_volume.shape),
        "spacing_xyz": spacing.tolist(),
        "domain_bounds_xyz": bounds.tolist(),
        "volume_min": volume_min,
        "volume_max": volume_max,
        "cached_level": cached_level,
        "reconstructed_p95_level": reconstructed_level,
        "level_absolute_difference": abs(cached_level - reconstructed_level),
        "ivd_finite": True,
        "mesh_finite": True,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "referenced_vertex_count": referenced_vertex_count,
        "ivd_volume_sha256": canonical_array_sha256(ivd_volume),
        "vertices_sha256": canonical_array_sha256(vertices),
        "faces_sha256": canonical_array_sha256(faces),
        "normals_sha256": canonical_array_sha256(normals),
        "values_sha256": canonical_array_sha256(values),
    }
    return mesh, bounds, audit


def build_phase21_visualization_scene(
    cache: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    allowed_datasets: tuple[str, ...] = TEST_DATASETS,
    required_split: str = "test",
    required_source_ordinal: int = FIXED_SOURCE_ORDINAL,
    regime: str = "exposed-development test",
    analysis_experiment: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one fixed source-ordinal-2 scene without reading performance."""

    display_regime = str(regime).strip()
    if not display_regime:
        raise ValueError("visualization regime must be non-empty")
    validated = validate_phase21_visualization_input(
        cache,
        prediction,
        allowed_datasets=allowed_datasets,
        required_split=required_split,
        required_source_ordinal=required_source_ordinal,
    )
    mesh, bounds, ivd_audit = _complete_ivd_mesh(
        validated.ivd_volume, validated.metadata
    )
    selected_rows, selection_audit = select_display_query_rows(
        validated.valid_seeds,
        validated.valid_labels,
        validated.valid_scale_id,
        domain_bounds=bounds,
        dataset=validated.dataset,
        source_ordinal=validated.source_ordinal,
        scale_id_start=int(
            validated.metadata.get("visualization_scale_id_start", 0)
        ),
        identity_scope=(
            validated.scale_block_id
            if validated.metadata.get("experiment") == EXPERIMENT31
            else None
        ),
    )
    primitives = validated.raw_features.reshape(-1, 7, 32, 3)
    center_xyz = (
        primitives[selected_rows, 0].astype(np.float64)
        + validated.valid_seeds[selected_rows, None, :]
    )
    display_pathlines = np.concatenate(
        (
            center_xyz,
            (
                validated.center_sample_time[selected_rows].astype(np.float64)
                + float(validated.metadata["source_time"])
            )[:, :, None],
        ),
        axis=2,
    )
    if not np.isfinite(display_pathlines).all() or not np.allclose(
        display_pathlines[:, 0, :3],
        validated.valid_seeds[selected_rows],
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError("absolute center-pathline reconstruction failed")
    seed_hash = canonical_array_sha256(validated.valid_seeds)
    reference_hash = canonical_array_sha256(validated.valid_labels)
    prediction_hash = canonical_array_sha256(validated.prediction)
    selected_seed_index = validated.valid_seed_index[selected_rows]
    scene = {
        "dataset": validated.dataset,
        "title": DATASET_TITLES[validated.dataset],
        "regime": display_regime,
        "source_ordinal": validated.source_ordinal,
        "bounds": bounds,
        "seeds": validated.valid_seeds,
        "reference": validated.valid_labels,
        "prediction": validated.prediction,
        "reference_seeds": validated.valid_seeds,
        "prediction_seeds": validated.valid_seeds,
        "display_pathlines": [pathline for pathline in display_pathlines],
        "ivd_points": None,
        "ivd_mesh": {
            "vertices": mesh["vertices"],
            "faces": mesh["faces"],
            "level": mesh["level"],
        },
        "valid_seed_index": validated.valid_seed_index,
        "valid_scale_id": validated.valid_scale_id,
        "valid_assigned_row_index": validated.valid_assigned_row_index,
        "valid_center_seed_index": validated.valid_center_seed_index,
        "valid_scale_block_index": validated.valid_scale_block_index,
        "scale_block_id": validated.scale_block_id,
        "selected_query_row": selected_rows,
        "selected_seed_index": selected_seed_index,
        "selected_assigned_row_index": validated.valid_assigned_row_index[
            selected_rows
        ],
        "selected_center_seed_index": validated.valid_center_seed_index[
            selected_rows
        ],
        "selected_scale_block_index": validated.valid_scale_block_index[
            selected_rows
        ],
        "selected_reference": validated.valid_labels[selected_rows],
        "ivd_mesh_normals": mesh["normals"],
        "ivd_mesh_values": mesh["values"],
    }
    audit = {
        "schema": (
            "pathline_template_matching.phase31_visualization_scene.v1"
            if validated.metadata.get("experiment") == EXPERIMENT31
            else "pathline_template_matching.phase21_visualization_scene.v1"
        ),
        "experiment": validated.metadata.get("experiment"),
        "dataset": validated.dataset,
        "split": str(validated.metadata.get("split")),
        "source_ordinal": validated.source_ordinal,
        "source_index": validated.source_index,
        "source_selection": (
            "fixed before performance; never metric-selected"
            if analysis_experiment is None
            else "fixed before classification; never metric-selected"
        ),
        "prediction_semantics": "FMT independent 161D global exact-1NN binary assignment",
        "query_count": int(len(validated.valid_labels)),
        "reference_positive_count": int(validated.valid_labels.sum()),
        "reference_negative_count": int((~validated.valid_labels).sum()),
        "prediction_positive_count": int(validated.prediction.sum()),
        "prediction_negative_count": int((~validated.prediction).sum()),
        "all_panels_use_complete_identical_valid_query_seed_population": True,
        "panel_query_seed_sha256": {label: seed_hash for label in ("a", "b", "c")},
        "valid_seed_index_sha256": canonical_array_sha256(
            validated.valid_seed_index
        ),
        "valid_scale_id_sha256": canonical_array_sha256(validated.valid_scale_id),
        "reference_sha256": reference_hash,
        "prediction_sha256": prediction_hash,
        "source_cache_array_sha256": {
            name: canonical_array_sha256(values)
            for name, values in (
                ("raw_features", validated.raw_features),
                ("valid_seed_index", validated.valid_seed_index),
                ("valid_scale_id", validated.valid_scale_id),
                ("valid_labels", validated.valid_labels),
                ("valid_seeds", validated.valid_seeds),
                ("center_sample_time", validated.center_sample_time),
                ("ivd_volume", validated.ivd_volume),
            )
        },
        "selection": selection_audit,
        "ivd_mesh": ivd_audit,
        "center_pathline_reconstruction": {
            "raw_shape": [int(len(validated.raw_features)), 7, 32, 3],
            "center_line_index": 0,
            "absolute_xyz_formula": "raw_centered_xyz + seeds_xyz[valid_seed_index]",
            "absolute_time_formula": "metadata.source_time + cache-relative center_sample_time",
            "display_pathlines_sha256": canonical_array_sha256(display_pathlines),
            "selected_seed_index_sha256": canonical_array_sha256(
                selected_seed_index
            ),
        },
        "cache_metadata_sha256": canonical_json_sha256(validated.metadata),
    }
    if analysis_experiment is not None:
        audit.update(
            {
                "analysis_experiment": str(analysis_experiment),
                "display_title": DATASET_TITLES[validated.dataset],
                "regime": display_regime,
            }
        )
    if validated.metadata.get("experiment") == EXPERIMENT31:
        audit.update(
            {
                "valid_assigned_row_index_sha256": canonical_array_sha256(
                    validated.valid_assigned_row_index
                ),
                "valid_center_seed_index_sha256": canonical_array_sha256(
                    validated.valid_center_seed_index
                ),
                "valid_scale_block_index_sha256": canonical_array_sha256(
                    validated.valid_scale_block_index
                ),
                "scale_block_id": validated.scale_block_id,
            }
        )
    return scene, audit


def _scene_arrays(scene: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, np.ndarray]:
    mesh = scene["ivd_mesh"]
    arrays = {
        "bounds": np.ascontiguousarray(scene["bounds"], dtype=np.float64),
        "seeds": np.ascontiguousarray(scene["seeds"], dtype=np.float64),
        "reference": np.ascontiguousarray(scene["reference"], dtype=np.bool_),
        "prediction": np.ascontiguousarray(scene["prediction"], dtype=np.bool_),
        "valid_seed_index": np.ascontiguousarray(
            scene["valid_seed_index"], dtype=np.int64
        ),
        "valid_scale_id": np.ascontiguousarray(scene["valid_scale_id"], dtype=np.int32),
        "display_pathlines": np.ascontiguousarray(
            scene["display_pathlines"], dtype=np.float64
        ),
        "selected_query_row": np.ascontiguousarray(
            scene["selected_query_row"], dtype=np.int64
        ),
        "selected_seed_index": np.ascontiguousarray(
            scene["selected_seed_index"], dtype=np.int64
        ),
        "selected_reference": np.ascontiguousarray(
            scene["selected_reference"], dtype=np.bool_
        ),
        "ivd_mesh_vertices": np.ascontiguousarray(
            mesh["vertices"], dtype=np.float64
        ),
        "ivd_mesh_faces": np.ascontiguousarray(mesh["faces"], dtype=np.int64),
        "ivd_mesh_normals": np.ascontiguousarray(
            scene["ivd_mesh_normals"], dtype=np.float32
        ),
        "ivd_mesh_values": np.ascontiguousarray(
            scene["ivd_mesh_values"], dtype=np.float32
        ),
        "ivd_mesh_level": np.asarray(float(mesh["level"]), dtype=np.float64),
        "metadata_json": np.asarray(
            json.dumps(audit, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        ),
    }
    expected_names = SCENE_ARRAY_NAMES
    if audit.get("experiment") == EXPERIMENT31:
        metadata_json = arrays.pop("metadata_json")
        for name, dtype in (
            ("valid_assigned_row_index", np.int64),
            ("valid_center_seed_index", np.int64),
            ("valid_scale_block_index", np.int8),
            ("selected_assigned_row_index", np.int64),
            ("selected_center_seed_index", np.int64),
            ("selected_scale_block_index", np.int8),
        ):
            arrays[name] = np.ascontiguousarray(scene[name], dtype=dtype)
        arrays["metadata_json"] = metadata_json
        expected_names = SCENE_ARRAY_NAMES[:-1] + SCENE31_EXTRA_ARRAY_NAMES + (
            "metadata_json",
        )
    if tuple(arrays) != expected_names:
        raise AssertionError("scene array order/schema changed")
    return arrays


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite scene artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in arrays:
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(arrays[name]), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
        with temporary.open("r+b") as source:
            os.fsync(source.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_phase21_scene_artifact(
    scene: Mapping[str, Any],
    audit: Mapping[str, Any],
    npz_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Write a deterministic scene and per-array hash manifest once."""

    scene_path = Path(npz_path)
    manifest = Path(manifest_path)
    if scene_path.suffix.lower() != ".npz":
        raise ValueError("scene path must end in .npz")
    if manifest.suffix.lower() != ".json":
        raise ValueError("scene manifest path must end in .json")
    if scene_path.exists() or manifest.exists():
        raise FileExistsError("scene NPZ and manifest are immutable and non-overwriting")
    arrays = _scene_arrays(scene, audit)
    experiment = str(audit.get("experiment", EXPERIMENT))
    array_order = list(arrays)
    array_manifest = {
        name: {
            "dtype": np.asarray(values).dtype.str,
            "shape": list(np.asarray(values).shape),
            "canonical_sha256": canonical_array_sha256(values),
        }
        for name, values in arrays.items()
    }
    _write_deterministic_npz(scene_path, arrays)
    manifest_payload: dict[str, Any] = {
        "schema": (
            "pathline_template_matching.phase31_scene_manifest.v1"
            if experiment == EXPERIMENT31
            else "pathline_template_matching.phase21_scene_manifest.v1"
        ),
        "experiment": experiment,
        "dataset": str(scene["dataset"]),
        "source_ordinal": int(scene["source_ordinal"]),
        "scene_npz": str(scene_path),
        "scene_npz_size_bytes": int(scene_path.stat().st_size),
        "scene_npz_sha256": sha256_file(scene_path),
        "array_order": array_order,
        "arrays": array_manifest,
        "combined_array_manifest_sha256": canonical_json_sha256(array_manifest),
        "scientific_audit_sha256": canonical_json_sha256(dict(audit)),
    }
    manifest_payload["manifest_content_sha256"] = canonical_json_sha256(
        manifest_payload
    )
    _write_json_no_overwrite(manifest, manifest_payload)
    return {
        **manifest_payload,
        "scene_manifest": str(manifest),
        "scene_manifest_file_sha256": sha256_file(manifest),
    }


def load_phase21_scene_artifact(
    npz_path: str | Path, manifest_path: str | Path
) -> SceneArtifact:
    """Load a scene only after verifying files, arrays, and scientific audit."""

    scene_path = Path(npz_path)
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    content_hash = str(manifest.get("manifest_content_sha256", ""))
    without_hash = {
        key: value for key, value in manifest.items() if key != "manifest_content_sha256"
    }
    if content_hash != canonical_json_sha256(without_hash):
        raise ValueError("scene manifest content SHA-256 mismatch")
    experiment = str(manifest.get("experiment", ""))
    expected_schema = (
        "pathline_template_matching.phase31_scene_manifest.v1"
        if experiment == EXPERIMENT31
        else "pathline_template_matching.phase21_scene_manifest.v1"
    )
    if experiment not in SUPPORTED_EXPERIMENTS or manifest.get("schema") != expected_schema:
        raise ValueError("unsupported template-matching scene manifest schema")
    if int(manifest.get("source_ordinal", -1)) != FIXED_SOURCE_ORDINAL:
        raise ValueError("scene manifest does not use fixed source ordinal 2")
    expected_array_names = (
        SCENE_ARRAY_NAMES[:-1] + SCENE31_EXTRA_ARRAY_NAMES + ("metadata_json",)
        if experiment == EXPERIMENT31
        else SCENE_ARRAY_NAMES
    )
    if manifest.get("array_order") != list(expected_array_names):
        raise ValueError("scene manifest array schema changed")
    if sha256_file(scene_path) != manifest.get("scene_npz_sha256"):
        raise ValueError("scene NPZ file SHA-256 mismatch")
    if int(manifest.get("scene_npz_size_bytes", -1)) != scene_path.stat().st_size:
        raise ValueError("scene NPZ size changed")
    with np.load(scene_path, allow_pickle=False) as archive:
        if tuple(archive.files) != expected_array_names:
            raise ValueError("scene NPZ key order/schema changed")
        arrays = {name: np.asarray(archive[name]) for name in expected_array_names}
    actual_array_manifest = {
        name: {
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "canonical_sha256": canonical_array_sha256(values),
        }
        for name, values in arrays.items()
    }
    if actual_array_manifest != manifest.get("arrays"):
        raise ValueError("scene per-array contract or SHA-256 mismatch")
    if canonical_json_sha256(actual_array_manifest) != manifest.get(
        "combined_array_manifest_sha256"
    ):
        raise ValueError("scene combined array-manifest SHA-256 mismatch")
    metadata_scalar = arrays["metadata_json"]
    if metadata_scalar.ndim != 0:
        raise ValueError("scene metadata_json must be scalar")
    metadata = json.loads(str(metadata_scalar.item()))
    if canonical_json_sha256(metadata) != manifest.get("scientific_audit_sha256"):
        raise ValueError("scene scientific-audit SHA-256 mismatch")

    count = len(arrays["seeds"])
    if (
        arrays["seeds"].dtype != np.float64
        or arrays["seeds"].shape != (count, 3)
        or arrays["reference"].shape != (count,)
        or arrays["prediction"].shape != (count,)
        or arrays["valid_seed_index"].shape != (count,)
        or arrays["valid_scale_id"].shape != (count,)
    ):
        raise ValueError("scene complete-query arrays disagree")
    if experiment == EXPERIMENT31:
        if (
            arrays["valid_assigned_row_index"].shape != (count,)
            or arrays["valid_center_seed_index"].shape != (count,)
            or arrays["valid_scale_block_index"].shape != (count,)
            or not np.array_equal(
                arrays["valid_assigned_row_index"], arrays["valid_seed_index"]
            )
            or not np.array_equal(
                arrays["selected_assigned_row_index"],
                arrays["valid_assigned_row_index"][arrays["selected_query_row"]],
            )
            or not np.array_equal(
                arrays["selected_center_seed_index"],
                arrays["valid_center_seed_index"][arrays["selected_query_row"]],
            )
            or not np.array_equal(
                arrays["selected_scale_block_index"],
                arrays["valid_scale_block_index"][arrays["selected_query_row"]],
            )
        ):
            raise ValueError("3.1 scene explicit row identity changed")
    if arrays["display_pathlines"].shape != (DISPLAY_PATHLINE_COUNT, 32, 4):
        raise ValueError("scene must contain exactly 240 center pathlines")
    selected = arrays["selected_query_row"]
    if (
        selected.shape != (DISPLAY_PATHLINE_COUNT,)
        or len(np.unique(selected)) != DISPLAY_PATHLINE_COUNT
        or int(arrays["selected_reference"].sum()) != DISPLAY_PER_CLASS
        or not np.array_equal(arrays["selected_reference"], arrays["reference"][selected])
        or not np.array_equal(
            arrays["selected_seed_index"], arrays["valid_seed_index"][selected]
        )
    ):
        raise ValueError("scene selected pathline identity/class contract changed")
    vertices = arrays["ivd_mesh_vertices"]
    faces = arrays["ivd_mesh_faces"]
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("scene IVD mesh is invalid")
    scene = {
        "dataset": str(metadata["dataset"]),
        "title": str(
            metadata.get(
                "display_title", DATASET_TITLES[str(metadata["dataset"])]
            )
        ),
        "regime": str(metadata.get("regime", "exposed-development test")),
        "source_ordinal": int(metadata["source_ordinal"]),
        "bounds": arrays["bounds"],
        "seeds": arrays["seeds"],
        "reference": arrays["reference"],
        "prediction": arrays["prediction"],
        "reference_seeds": arrays["seeds"],
        "prediction_seeds": arrays["seeds"],
        "display_pathlines": [value for value in arrays["display_pathlines"]],
        "ivd_points": None,
        "ivd_mesh": {
            "vertices": vertices,
            "faces": faces,
            "level": float(arrays["ivd_mesh_level"]),
        },
    }
    return SceneArtifact(
        scene=scene,
        metadata=metadata,
        npz_path=scene_path,
        manifest_path=manifest_file,
        npz_sha256=str(manifest["scene_npz_sha256"]),
        manifest_file_sha256=sha256_file(manifest_file),
    )


def render_phase21_scene_artifact(
    npz_path: str | Path,
    manifest_path: str | Path,
    output_stem: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> RenderedTriptychArtifacts:
    """Verify and render one immutable PNG/PDF/metadata/alignment bundle."""

    artifact = load_phase21_scene_artifact(npz_path, manifest_path)
    stem = Path(output_stem)
    if stem.suffix:
        raise ValueError("output_stem must not have a file extension")
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg") if artifact.metadata.get("experiment") == EXPERIMENT31 else None
    metadata_path = stem.with_suffix(".render.json")
    alignment_path = stem.with_suffix(".alignment.json")
    outputs = tuple(
        value
        for value in (png_path, pdf_path, svg_path, metadata_path, alignment_path)
        if value is not None
    )
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite render artifacts: {existing}")
    _, render_metadata = render_template_matching_triptych(
        artifact.scene,
        png_path,
        pdf_output_path=pdf_path,
        svg_output_path=svg_path,
        alignment_output_path=alignment_path,
        view=DATASET_VIEWS[str(artifact.scene["dataset"])],
        dpi=int(dpi),
    )
    counts = dict(render_metadata["counts"])
    expected_query_count = int(artifact.metadata["query_count"])
    if (
        counts.get("sample_count") != expected_query_count
        or counts.get("display_pathline_count") != DISPLAY_PATHLINE_COUNT
        or counts.get("reference_positive")
        != int(artifact.metadata["reference_positive_count"])
        or sum(
            int(counts[name])
            for name in (
                "true_negative",
                "true_positive",
                "false_positive",
                "false_negative",
            )
        )
        != expected_query_count
    ):
        raise RuntimeError("rendered counts disagree with the verified scene")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if alignment.get("status") != "PASS":
        raise RuntimeError("triptych alignment JSON did not pass")
    metadata: dict[str, Any] = {
        "schema": (
            "pathline_template_matching.phase31_triptych_render.v1"
            if artifact.metadata.get("experiment") == EXPERIMENT31
            else "pathline_template_matching.phase21_triptych_render.v1"
        ),
        "experiment": artifact.metadata.get("experiment"),
        "dataset": str(artifact.scene["dataset"]),
        "source_ordinal": FIXED_SOURCE_ORDINAL,
        "scene_npz": str(artifact.npz_path),
        "scene_npz_sha256": artifact.npz_sha256,
        "scene_manifest": str(artifact.manifest_path),
        "scene_manifest_file_sha256": artifact.manifest_file_sha256,
        "png": str(png_path),
        "png_sha256": sha256_file(png_path),
        "pdf": str(pdf_path),
        "pdf_sha256": sha256_file(pdf_path),
        "svg": None if svg_path is None else str(svg_path),
        "svg_sha256": None if svg_path is None else sha256_file(svg_path),
        "alignment_json": str(alignment_path),
        "alignment_json_sha256": sha256_file(alignment_path),
        "counts": counts,
        "alignment": alignment,
        "renderer": render_metadata,
    }
    if "analysis_experiment" in artifact.metadata:
        for name in (
            "analysis_experiment",
            "display_title",
            "regime",
            "fold_id",
            "held_out_physical_family",
            "library_contains_query_family",
            "scale_block",
        ):
            if name in artifact.metadata:
                metadata[name] = artifact.metadata[name]
    metadata["metadata_content_sha256"] = canonical_json_sha256(metadata)
    _write_json_no_overwrite(metadata_path, metadata)
    return RenderedTriptychArtifacts(
        png_path=png_path,
        pdf_path=pdf_path,
        svg_path=svg_path,
        metadata_path=metadata_path,
        alignment_path=alignment_path,
        metadata={**metadata, "metadata_file_sha256": sha256_file(metadata_path)},
    )


__all__ = [
    "DATASET_TITLES",
    "DATASET_VIEWS",
    "DISPLAY_PATHLINE_COUNT",
    "DISPLAY_PER_CLASS",
    "DISPLAY_SELECTION_SEED",
    "EXPERIMENT",
    "FIXED_SOURCE_ORDINAL",
    "RenderedTriptychArtifacts",
    "SCALE_COUNT",
    "SceneArtifact",
    "TEST_DATASETS",
    "ValidatedPhase21VisualizationInput",
    "build_phase21_visualization_scene",
    "load_phase21_scene_artifact",
    "ordered_fmt_prediction",
    "render_phase21_scene_artifact",
    "scale_axis_indices",
    "select_display_query_rows",
    "validate_phase21_visualization_input",
    "write_phase21_scene_artifact",
]

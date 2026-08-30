"""Family-held-out FMT classification figures derived from immutable 3.1 caches.

This module implements ``Other_MainExp31FamilyHeldOutVisualization_1.1``.  It
does not alter or reuse the fitted 3.1 library.  Each fold rebuilds the parent
balanced template selection and FMT standardization after excluding the whole
query physical family, then classifies the fixed source-ordinal-2 cache with
global exact one-nearest-neighbor search.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import skimage
import torch
import yaml

from .matcher import ExhaustiveMatchResult, ExhaustiveOneNearestNeighbor
from .phase21_pipeline import (
    CacheBuildSummary,
    Phase21Plan,
    ScaleAssignmentBlock,
    _atomic_bytes,
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _load_cache,
    _metric_values,
    _select_library_and_fit_pca,
    _validate_cache_provenance,
    authorize_phase31_portable_population_marker_path,
    configure_deterministic_execution,
    load_cache_summary_sidecar,
    load_phase31_plan,
)
from .phase21_visualization import (
    DATASET_TITLES,
    FIXED_SOURCE_ORDINAL,
    build_phase21_visualization_scene,
    ordered_fmt_prediction,
    render_phase21_scene_artifact,
    write_phase21_scene_artifact,
)
from .portable_flow import (
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)


EXPERIMENT = "Other_MainExp31FamilyHeldOutVisualization_1.1"
CONFIG_SHA256 = "6fec35d2f64a3b593a74e8b35674137b1665ce169491e3546384142514b46670"
PARENT_EXPERIMENT = "mainExp_TemplateMatching_3.1"
PARENT_NUMERICAL_COMMIT = "260a07ad380d64fc300cabe8926244e92d8ba04a"
PARENT_CONFIG_SHA256 = (
    "771980f14a6019a1f6e4bf03668d9f37dcf63495ae2dafa866312b12fc71855e"
)
REQUESTED_DATASETS = (
    "cylinder3d",
    "halfcylinderRe640",
    "halfcylinderRe6400",
    "boeing747",
)
REGIME = "physical-family-held-out exposed-development"


@dataclass(frozen=True, slots=True)
class FoldSpec:
    """One complete-physical-family exclusion fold."""

    fold_id: str
    held_out_family: str
    query_datasets: tuple[str, ...]
    library_datasets: tuple[str, ...]
    expected_template_count: int
    expected_per_class: int


@dataclass(frozen=True, slots=True)
class OtherVisualizationPlan:
    """Strict parsed form of the frozen downstream visualization config."""

    config_path: Path
    config_sha256: str
    config: dict[str, Any]
    parent_plan: Phase21Plan
    folds: tuple[FoldSpec, ...]
    output_root: PurePosixPath
    query_chunk_size: int
    library_chunk_size: int
    png_dpi: int


@dataclass(frozen=True, slots=True)
class QueryResultRecord:
    """Arrays needed for the final per-query provenance table."""

    fold: FoldSpec
    cache_row: dict[str, Any]
    block: ScaleAssignmentBlock
    cache_mask: np.ndarray
    cache: Mapping[str, Any]
    result: ExhaustiveMatchResult
    library_metadata: Mapping[str, np.ndarray]


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _as_unique_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list")
    output = tuple(str(item) for item in value)
    if not output or len(output) != len(set(output)) or any(not item for item in output):
        raise ValueError(f"{name} must contain unique non-empty strings")
    return output


def load_other_visualization_plan(config_path: str | Path) -> OtherVisualizationPlan:
    """Load and fail closed on any drift from the frozen two-fold contract."""

    path = Path(config_path).resolve()
    actual_config_sha256 = sha256_file(path)
    if actual_config_sha256 != CONFIG_SHA256:
        raise ValueError(
            "downstream visualization config SHA-256 changed: "
            f"{actual_config_sha256} != {CONFIG_SHA256}"
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _as_mapping(payload, name="config")
    if config.get("experiment") != EXPERIMENT:
        raise ValueError("downstream visualization experiment identity changed")
    if config.get("status") != "frozen_pre_run_not_run":
        raise ValueError("frozen pre-run status changed")
    project_root = path.parents[1]
    parent = _as_mapping(config.get("parent"), name="parent")
    expected_parent = {
        "experiment": PARENT_EXPERIMENT,
        "numerical_git_commit": PARENT_NUMERICAL_COMMIT,
        "config_sha256": PARENT_CONFIG_SHA256,
        "fitted_scaler_library_prediction_or_metric_reuse": "forbidden",
    }
    drift = {
        key: (parent.get(key), expected)
        for key, expected in expected_parent.items()
        if parent.get(key) != expected
    }
    if drift:
        raise ValueError(f"parent identity or reuse guard changed: {drift}")
    parent_config = (project_root / str(parent.get("config", ""))).resolve()
    parent_plan = load_phase31_plan(parent_config)
    if (
        parent_plan.experiment != PARENT_EXPERIMENT
        or parent_plan.config_sha256 != PARENT_CONFIG_SHA256
    ):
        raise ValueError("committed parent plan differs from the frozen identity")

    expected_folds = (
        FoldSpec(
            fold_id="holdout_half_cylinder",
            held_out_family="half_cylinder",
            query_datasets=(
                "cylinder3d",
                "halfcylinderRe640",
                "halfcylinderRe6400",
            ),
            library_datasets=(
                "deltaWing_resampled",
                "deltaWing_LBM",
                "f22raptor",
                "channel",
                "boeing747",
            ),
            expected_template_count=50_770,
            expected_per_class=25_385,
        ),
        FoldSpec(
            fold_id="holdout_boeing_747",
            held_out_family="boeing_747",
            query_datasets=("boeing747",),
            library_datasets=(
                "cylinder3d",
                "halfcylinderRe640",
                "halfcylinderRe6400",
                "deltaWing_resampled",
                "deltaWing_LBM",
                "f22raptor",
                "channel",
            ),
            expected_template_count=86_728,
            expected_per_class=43_364,
        ),
    )
    raw_folds = config.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != len(expected_folds):
        raise ValueError("exactly two frozen family-held-out folds are required")
    parsed_folds: list[FoldSpec] = []
    for raw, expected in zip(raw_folds, expected_folds, strict=True):
        row = _as_mapping(raw, name="fold")
        parsed = FoldSpec(
            fold_id=str(row.get("id", "")),
            held_out_family=str(row.get("held_out_physical_family", "")),
            query_datasets=_as_unique_strings(
                row.get("query_datasets"), name="fold.query_datasets"
            ),
            library_datasets=_as_unique_strings(
                row.get("library_datasets"), name="fold.library_datasets"
            ),
            expected_template_count=int(row.get("expected_selected_template_count", -1)),
            expected_per_class=int(row.get("expected_selected_templates_per_class", -1)),
        )
        if parsed != expected:
            raise ValueError(f"family-held-out fold changed: {parsed} != {expected}")
        family_members = tuple(
            dataset
            for dataset in parent_plan.train_datasets
            if parent_plan.family_by_dataset[dataset] == parsed.held_out_family
        )
        if family_members != parsed.query_datasets:
            raise ValueError("query datasets do not equal the complete held-out family")
        expected_library = tuple(
            dataset
            for dataset in parent_plan.train_datasets
            if parent_plan.family_by_dataset[dataset] != parsed.held_out_family
        )
        if expected_library != parsed.library_datasets:
            raise ValueError("fold library is not the exact non-held-out parent population")
        parsed_folds.append(parsed)
    query = _as_mapping(config.get("query"), name="query")
    if (
        int(query.get("source_ordinal", -1)) != FIXED_SOURCE_ORDINAL
        or _as_unique_strings(query.get("requested_datasets"), name="requested_datasets")
        != REQUESTED_DATASETS
        or query.get("balancing") != "disabled"
        or query.get("downsampling") != "disabled"
    ):
        raise ValueError("fixed query contract changed")
    blocks = config.get("scale_blocks")
    if not isinstance(blocks, list) or len(blocks) != 2:
        raise ValueError("the two parent scale blocks are required")
    expected_blocks = [
        (block.block_id, block.scale_id_start, block.scale_id_stop)
        for block in parent_plan.effective_scale_blocks
    ]
    actual_blocks = [
        (
            str(row.get("id")),
            int(row.get("scale_id_start", -1)),
            int(row.get("scale_id_stop_exclusive", -1)),
        )
        for row in blocks
        if isinstance(row, Mapping)
    ]
    if actual_blocks != expected_blocks:
        raise ValueError("scale-block contract changed")
    visualization = _as_mapping(config.get("visualization"), name="visualization")
    if (
        int(visualization.get("expected_figure_count", -1)) != 8
        or int(visualization.get("source_ordinal", -1)) != FIXED_SOURCE_ORDINAL
        or visualization.get("metric_or_prediction_based_scene_selection") != "forbidden"
    ):
        raise ValueError("visualization selection contract changed")
    execution = _as_mapping(config.get("execution"), name="execution")
    output_root = PurePosixPath(str(execution.get("output_root", "")))
    if not output_root.is_absolute():
        raise ValueError("Ibex output_root must be absolute")
    return OtherVisualizationPlan(
        config_path=path,
        config_sha256=actual_config_sha256,
        config=config,
        parent_plan=parent_plan,
        folds=tuple(parsed_folds),
        output_root=output_root,
        query_chunk_size=int(execution.get("query_chunk_size", 1024)),
        library_chunk_size=int(execution.get("library_chunk_size", 8192)),
        png_dpi=int(visualization.get("png_dpi", 360)),
    )


def discover_parent_train_inputs(
    plan: OtherVisualizationPlan, cache_root: str | Path
) -> tuple[list[CacheBuildSummary], list[dict[str, Any]]]:
    """Authenticate exactly 32 parent train sidecars and cache files."""

    root = Path(cache_root).resolve()
    train_root = (root / "train").resolve()
    sidecars = sorted(train_root.rglob("*.summary.json"))
    expected_keys = [
        (dataset, ordinal)
        for dataset in plan.parent_plan.train_datasets
        for ordinal in range(plan.parent_plan.source_count)
    ]
    if len(sidecars) != len(expected_keys):
        raise ValueError(
            f"expected exactly {len(expected_keys)} parent train sidecars, "
            f"found {len(sidecars)}"
        )
    found: dict[tuple[str, int], tuple[CacheBuildSummary, dict[str, Any]]] = {}
    for sidecar in sidecars:
        resolved_sidecar = sidecar.resolve()
        if train_root not in resolved_sidecar.parents:
            raise ValueError("train sidecar resolves outside the authorized cache root")
        summary = load_cache_summary_sidecar(resolved_sidecar)
        row = dict(summary.cache_row)
        dataset = str(row.get("dataset", ""))
        ordinal = int(row.get("source_ordinal", -1))
        key = (dataset, ordinal)
        expected = {
            "experiment": PARENT_EXPERIMENT,
            "config_sha256": PARENT_CONFIG_SHA256,
            "cache_builder_git_commit": PARENT_NUMERICAL_COMMIT,
            "portable_builder_git_commit": PARENT_NUMERICAL_COMMIT,
            "physical_family": plan.parent_plan.family_by_dataset.get(dataset),
            "split": "train",
            "assigned_count": plan.parent_plan.assigned_primitive_count,
            "assignment_count_per_seed": 2,
            "maximum_source_frame_intervals": 48.0,
        }
        drift = {
            name: (row.get(name), value)
            for name, value in expected.items()
            if row.get(name) != value
        }
        if key not in expected_keys or drift:
            raise ValueError(f"parent train sidecar identity changed for {key}: {drift}")
        if key in found:
            raise ValueError(f"duplicate parent train sidecar {key}")
        authorized_parent = (train_root / dataset).resolve()
        cache_path = Path(str(row.get("path", ""))).resolve()
        if resolved_sidecar.parent != authorized_parent or cache_path.parent != authorized_parent:
            raise ValueError("sidecar or cache path escapes its dataset train directory")
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        actual_cache_size = int(cache_path.stat().st_size)
        actual_cache_hash = sha256_file(cache_path)
        if (
            int(row.get("file_size", -1)) != actual_cache_size
            or row.get("file_sha256") != actual_cache_hash
        ):
            raise ValueError(f"cache file changed for {key}")
        portable_fields = {
            name: row.get(name)
            for name in (
                "portable_population_pass_path",
                "portable_population_pass_file_size",
                "portable_population_pass_file_sha256",
                "portable_population_scope",
                "portable_population_rows_content_sha256",
            )
        }
        if (
            portable_fields["portable_population_scope"] != "train-only"
            or not all(portable_fields.values())
        ):
            raise ValueError(f"train-only portable marker identity is incomplete for {key}")
        portable_marker = authorize_phase31_portable_population_marker_path(
            plan.parent_plan,
            str(portable_fields["portable_population_pass_path"]),
            access_scope="train-only",
        )
        if (
            int(portable_fields["portable_population_pass_file_size"])
            != portable_marker.stat().st_size
            or portable_fields["portable_population_pass_file_sha256"]
            != sha256_file(portable_marker)
        ):
            raise ValueError(f"train-only portable marker changed for {key}")
        evidence = {
            "dataset": dataset,
            "physical_family": row["physical_family"],
            "source_ordinal": ordinal,
            "source_index": int(row["source_index"]),
            "assigned_count": int(row["assigned_count"]),
            "valid_count": int(row["valid_count"]),
            "invalid_count": int(row["invalid_count"]),
            "sidecar_path": str(resolved_sidecar),
            "sidecar_size_bytes": int(resolved_sidecar.stat().st_size),
            "sidecar_file_sha256": sha256_file(resolved_sidecar),
            "cache_path": str(cache_path),
            "cache_size_bytes": actual_cache_size,
            "cache_file_sha256": actual_cache_hash,
            "cache_builder_git_commit": row["cache_builder_git_commit"],
            **portable_fields,
            "derived_window_combined_sha256": summary.derived_window_row[
                "combined_sha256"
            ],
            "portable_file_sha256": summary.derived_window_row[
                "portable_file_sha256"
            ],
            "fmt_features_sha256": summary.primitive_row["fmt_features_sha256"],
            "ivd_volume_sha256": summary.label_row["ivd_volume_sha256"],
        }
        found[key] = (summary, evidence)
    if set(found) != set(expected_keys):
        raise ValueError("parent train cache population is incomplete")
    summaries = [found[key][0] for key in expected_keys]
    evidence_rows = [found[key][1] for key in expected_keys]
    return summaries, evidence_rows


def _block_scene_cache(
    cache: Mapping[str, Any], block_index: int, block: ScaleAssignmentBlock
) -> tuple[dict[str, Any], np.ndarray]:
    scale_id = np.asarray(cache["valid_scale_id"], dtype=np.int32)
    mask = (scale_id >= block.scale_id_start) & (scale_id < block.scale_id_stop)
    if not mask.any():
        raise ValueError(f"query block {block.block_id} has no valid primitive")
    selected_names = (
        "raw_features",
        "fmt_features",
        "valid_labels",
        "valid_seed_index",
        "valid_scale_id",
        "center_sample_time",
        "valid_assigned_row_index",
        "valid_center_seed_index",
        "valid_scale_block_index",
    )
    output = dict(cache)
    for name in selected_names:
        output[name] = np.ascontiguousarray(np.asarray(cache[name])[mask])
    if not np.all(output["valid_scale_block_index"] == block_index):
        raise ValueError("query block identity differs from its scale-ID interval")
    metadata = dict(cache["metadata"])
    metadata.update(
        {
            "visualization_scale_block_index": int(block_index),
            "visualization_scale_block_id": block.block_id,
            "visualization_scale_id_start": block.scale_id_start,
            "visualization_scale_id_stop_exclusive": block.scale_id_stop,
        }
    )
    hashes = dict(metadata.get("array_sha256", {}))
    for name in (
        "raw_features",
        "fmt_features",
        "valid_labels",
        "valid_seed_index",
        "valid_scale_id",
        "center_sample_time",
        "valid_assigned_row_index",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "seeds_xyz",
        "ivd_volume",
    ):
        hashes[name] = canonical_array_sha256(output[name])
    metadata["array_sha256"] = hashes
    metadata["combined_array_sha256"] = canonical_json_sha256(hashes)
    output["metadata"] = metadata
    return output, np.ascontiguousarray(mask)


def _relative(path: str | Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(root.resolve())).replace("\\", "/")


def _export_row(path: Path, root: Path, export_kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size < 1:
        raise RuntimeError(f"missing visualization export: {resolved}")
    return {
        "relative_path": _relative(resolved, root),
        "export_kind": export_kind,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _library_counts(library: Mapping[str, np.ndarray]) -> dict[str, Any]:
    labels = np.asarray(library["labels"], dtype=bool)
    output: dict[str, Any] = {
        "template_count": int(len(labels)),
        "negative_template_count": int((~labels).sum()),
        "positive_template_count": int(labels.sum()),
        "by_dataset": {},
        "by_physical_family": {},
        "by_scale_block": {},
    }
    for key, field in (
        ("by_dataset", "dataset"),
        ("by_physical_family", "physical_family"),
        ("by_scale_block", "scale_block_id"),
    ):
        values = np.asarray(library[field]).astype(str)
        output[key] = {
            value: {
                "total": int(np.sum(values == value)),
                "negative": int(np.sum((values == value) & ~labels)),
                "positive": int(np.sum((values == value) & labels)),
            }
            for value in sorted(set(values.tolist()))
        }
    return output


def _query_rows(
    records: Sequence[QueryResultRecord],
) -> Iterable[dict[str, Any]]:
    for record in records:
        cache = record.cache
        mask = record.cache_mask
        result = record.result
        library = record.library_metadata
        nearest = np.asarray(result.nearest_indices, dtype=np.int64)
        predicted = np.asarray(result.labels, dtype=bool)
        if not np.array_equal(np.asarray(library["labels"])[nearest], predicted):
            raise RuntimeError("nearest-template labels differ from predictions")
        query_arrays = {
            "assigned": np.asarray(cache["valid_assigned_row_index"])[mask],
            "center": np.asarray(cache["valid_center_seed_index"])[mask],
            "block_index": np.asarray(cache["valid_scale_block_index"])[mask],
            "scale": np.asarray(cache["valid_scale_id"])[mask],
            "reference": np.asarray(cache["valid_labels"], dtype=bool)[mask],
        }
        count = len(predicted)
        if any(len(values) != count for values in query_arrays.values()):
            raise RuntimeError("per-query arrays changed after matching")
        for index in range(count):
            match_index = int(nearest[index])
            yield {
                "experiment": EXPERIMENT,
                "fold_id": record.fold.fold_id,
                "held_out_physical_family": record.fold.held_out_family,
                "query_dataset": record.cache_row["dataset"],
                "query_physical_family": record.cache_row["physical_family"],
                "query_source_ordinal": record.cache_row["source_ordinal"],
                "query_source_index": record.cache_row["source_index"],
                "query_assigned_row_index": int(query_arrays["assigned"][index]),
                "query_center_seed_index": int(query_arrays["center"][index]),
                "query_scale_block_index": int(query_arrays["block_index"][index]),
                "query_scale_block_id": record.block.block_id,
                "query_scale_id": int(query_arrays["scale"][index]),
                "reference_label": int(query_arrays["reference"][index]),
                "predicted_label": int(predicted[index]),
                "score": float(result.scores[index]),
                "nearest_template_index": match_index,
                "nearest_template_distance": float(result.nearest_distances[index]),
                "nearest_positive_distance": float(
                    result.nearest_positive_distances[index]
                ),
                "nearest_negative_distance": float(
                    result.nearest_negative_distances[index]
                ),
                "match_dataset": str(library["dataset"][match_index]),
                "match_physical_family": str(
                    library["physical_family"][match_index]
                ),
                "match_source_ordinal": int(library["source_ordinal"][match_index]),
                "match_source_index": int(library["source_index"][match_index]),
                "match_assigned_row_index": int(
                    library["assigned_row_index"][match_index]
                ),
                "match_center_seed_index": int(
                    library["center_seed_index"][match_index]
                ),
                "match_scale_block_index": int(
                    library["scale_block_index"][match_index]
                ),
                "match_scale_block_id": str(
                    library["scale_block_id"][match_index]
                ),
                "match_scale_id": int(library["scale_id"][match_index]),
                "match_label": int(library["labels"][match_index]),
            }


def _main_table_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = (
        "Dataset",
        "Block",
        "Valid / 64,000",
        "Coverage",
        "Accuracy",
        "AP",
        "F1",
        "Balanced accuracy",
        "AUROC",
        "TP / FP / TN / FN",
    )
    lines = [
        "# Family-held-out source-ordinal-2 classification table",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        values = (
            DATASET_TITLES[str(row["dataset"])],
            str(row["scale_block_id"]),
            f"{int(row['valid_count']):,} / {int(row['assigned_count']):,}",
            f"{float(row['coverage']):.4%}",
            f"{float(row['accuracy']):.4f}",
            f"{float(row['average_precision']):.4f}",
            f"{float(row['f1']):.4f}",
            f"{float(row['balanced_accuracy']):.4f}",
            f"{float(row['auroc']):.4f}",
            (
                f"{int(row['true_positive']):,} / {int(row['false_positive']):,} / "
                f"{int(row['true_negative']):,} / {int(row['false_negative']):,}"
            ),
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        (
            "",
            "All rows are exposed-development and use a fold-specific library that "
            "excludes the complete query physical family. AP means Average Precision; "
            "AUROC means Area Under the Receiver Operating Characteristic Curve.",
            "",
        )
    )
    return "\n".join(lines)


def run_family_heldout_visualization(
    plan: OtherVisualizationPlan,
    *,
    cache_root: str | Path,
    run_dir: str | Path,
    git_commit: str,
    device: str,
) -> dict[str, Any]:
    """Run both folds and write all immutable evidence and eight triptychs."""

    if len(git_commit) != 40 or any(value not in "0123456789abcdef" for value in git_commit):
        raise ValueError("git_commit must be a lowercase 40-character SHA-1")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    execution_contract = configure_deterministic_execution()
    root = Path(run_dir).resolve()
    if root.exists():
        raise FileExistsError(f"immutable run directory exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    _atomic_bytes(root / "frozen_config.yaml", plan.config_path.read_bytes())

    summaries, input_rows = discover_parent_train_inputs(plan, cache_root)
    input_manifest = {
        "schema": "pathline_template_matching.family_heldout_visualization_input.v1",
        "experiment": EXPERIMENT,
        "git_commit": git_commit,
        "worktree_clean": True,
        "config_sha256": plan.config_sha256,
        "parent_experiment": PARENT_EXPERIMENT,
        "parent_numerical_git_commit": PARENT_NUMERICAL_COMMIT,
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "array_access_after_manifest_write_only": True,
        "input_scope": "exactly_32_parent_3_1_train_caches_and_sidecars",
        "row_count": len(input_rows),
        "rows": input_rows,
        "rows_content_sha256": canonical_json_sha256(input_rows),
    }
    input_manifest["manifest_content_sha256"] = canonical_json_sha256(input_manifest)
    _atomic_json(root / "input_manifest.json", input_manifest)

    summary_by_key = {
        (str(summary.cache_row["dataset"]), int(summary.cache_row["source_ordinal"])): summary
        for summary in summaries
    }
    library_manifest_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    library_audit_rows: list[dict[str, Any]] = []
    visualization_entries: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    query_records: list[QueryResultRecord] = []

    for fold in plan.folds:
        eligible = [
            summary.cache_row
            for summary in summaries
            if str(summary.cache_row["dataset"]) in fold.library_datasets
        ]
        if len(eligible) != len(fold.library_datasets) * plan.parent_plan.source_count:
            raise ValueError(f"fold {fold.fold_id} eligible cache count changed")
        if any(
            plan.parent_plan.family_by_dataset[str(row["dataset"])]
            == fold.held_out_family
            for row in eligible
        ):
            raise RuntimeError(f"held-out family leaked into {fold.fold_id} library")
        pca, library, prior_fraction, audit = _select_library_and_fit_pca(
            plan.parent_plan,
            root / "folds" / fold.fold_id,
            eligible,
            verify_cache_hashes=True,
        )
        del pca
        if (
            len(library["labels"]) != fold.expected_template_count
            or int(np.sum(library["labels"])) != fold.expected_per_class
            or int(np.sum(~library["labels"])) != fold.expected_per_class
        ):
            raise RuntimeError(f"fold {fold.fold_id} selected template count changed")
        if fold.held_out_family in set(library["physical_family"].astype(str).tolist()):
            raise RuntimeError(f"held-out family is present in {fold.fold_id} library")
        matcher = ExhaustiveOneNearestNeighbor(
            library["fmt_features"], library["labels"], device=device
        )
        library_path = root / "folds" / fold.fold_id / "fmt_library.npz"
        library_arrays = {
            name: values
            for name, values in library.items()
            if name not in {"raw_features", "pca_features"}
        }
        _atomic_npz(library_path, library_arrays)
        preprocessing_path = root / "folds" / fold.fold_id / "fmt_preprocessing.npz"
        zero_variance = np.asarray(
            np.asarray(library["fmt_features"], dtype=np.float64).std(axis=0) < 1e-12,
            dtype=np.bool_,
        )
        _atomic_npz(
            preprocessing_path,
            {
                "feature_mean": matcher.feature_mean,
                "feature_standard_deviation": matcher.feature_scale,
                "zero_variance_feature_mask": zero_variance,
            },
        )
        counts = _library_counts(library)
        library_manifest_rows.append(
            {
                "fold_id": fold.fold_id,
                "held_out_physical_family": fold.held_out_family,
                "query_datasets": list(fold.query_datasets),
                "library_datasets": list(fold.library_datasets),
                "eligible_candidate_count": int(
                    sum(int(row["valid_count"]) for row in eligible)
                ),
                "eligible_positive_fraction": float(prior_fraction),
                "library_file": _relative(library_path, root),
                "library_file_size_bytes": int(library_path.stat().st_size),
                "library_file_sha256": sha256_file(library_path),
                "fmt_features_sha256": canonical_array_sha256(
                    library["fmt_features"]
                ),
                "labels_sha256": canonical_array_sha256(library["labels"]),
                "selection_seed": plan.parent_plan.library_seed,
                "selection_traversal": "parent_train_dataset_order_then_source_ordinal_then_scale_id_then_negative_then_positive",
                **counts,
            }
        )
        preprocessing_rows.append(
            {
                "fold_id": fold.fold_id,
                "fit_population": "fold_selected_balanced_FMT_library_only",
                "feature_width": 161,
                "normalization": "mean_and_population_standard_deviation",
                "zero_variance_action": "replace_divisor_with_one",
                "file": _relative(preprocessing_path, root),
                "file_size_bytes": int(preprocessing_path.stat().st_size),
                "file_sha256": sha256_file(preprocessing_path),
                "feature_mean_sha256": canonical_array_sha256(matcher.feature_mean),
                "feature_standard_deviation_sha256": canonical_array_sha256(
                    matcher.feature_scale
                ),
                "zero_variance_feature_count": int(zero_variance.sum()),
            }
        )
        for row in audit:
            library_audit_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "held_out_physical_family": fold.held_out_family,
                    **row,
                }
            )

        library_metadata = {
            name: np.asarray(library[name])
            for name in (
                "labels",
                "dataset",
                "physical_family",
                "source_ordinal",
                "source_index",
                "assigned_row_index",
                "center_seed_index",
                "scale_block_index",
                "scale_block_id",
                "scale_id",
            )
        }
        for dataset in fold.query_datasets:
            summary = summary_by_key[(dataset, FIXED_SOURCE_ORDINAL)]
            cache_row = dict(summary.cache_row)
            cache = _load_cache(
                Path(str(cache_row["path"])),
                expected_sha256=str(cache_row["file_sha256"]),
            )
            _validate_cache_provenance(plan.parent_plan, cache, cache_row)
            if cache_row["physical_family"] != fold.held_out_family:
                raise RuntimeError("query cache family differs from the excluded family")
            for block_index, block in enumerate(plan.parent_plan.effective_scale_blocks):
                scene_cache, cache_mask = _block_scene_cache(cache, block_index, block)
                result = matcher.query(
                    np.asarray(scene_cache["fmt_features"], dtype=np.float32),
                    query_chunk_size=plan.query_chunk_size,
                    library_chunk_size=plan.library_chunk_size,
                )
                prediction = ordered_fmt_prediction(
                    result.labels,
                    scene_cache["valid_seed_index"],
                    scene_cache["valid_scale_id"],
                )
                scene, scientific_audit = build_phase21_visualization_scene(
                    scene_cache,
                    prediction,
                    allowed_datasets=REQUESTED_DATASETS,
                    required_split="train",
                    required_source_ordinal=FIXED_SOURCE_ORDINAL,
                    regime=REGIME,
                    analysis_experiment=EXPERIMENT,
                )
                scene["title"] = f"{scene['title']} | {block.block_id}"
                scientific_audit["display_title"] = scene["title"]
                scientific_audit.update(
                    {
                        "fold_id": fold.fold_id,
                        "held_out_physical_family": fold.held_out_family,
                        "library_contains_query_family": False,
                        "library_file_sha256": sha256_file(library_path),
                        "preprocessing_file_sha256": sha256_file(
                            preprocessing_path
                        ),
                        "scale_block": {
                            "scale_block_index": block_index,
                            "scale_block_id": block.block_id,
                            "scale_id_start": block.scale_id_start,
                            "scale_id_stop_exclusive": block.scale_id_stop,
                        },
                    }
                )
                scene_stem = (
                    root
                    / "scenes"
                    / f"{dataset}_source_ordinal_2_{block.block_id}"
                )
                scene_path = scene_stem.with_suffix(".scene.npz")
                scene_manifest_path = scene_stem.with_suffix(".scene.json")
                scene_manifest = write_phase21_scene_artifact(
                    scene, scientific_audit, scene_path, scene_manifest_path
                )
                figure_stem = (
                    root
                    / "figures"
                    / f"{dataset}_source_ordinal_2_{block.block_id}_family_heldout_triptych"
                )
                rendered = render_phase21_scene_artifact(
                    scene_path,
                    scene_manifest_path,
                    figure_stem,
                    dpi=plan.png_dpi,
                )
                if rendered.svg_path is None:
                    raise RuntimeError("3.1-derived triptych must export SVG")
                metrics = _metric_values(
                    np.asarray(scene_cache["valid_labels"], dtype=bool),
                    result.labels,
                    result.scores,
                )
                assigned_count = int(
                    plan.config["query"]["assigned_count_per_dataset_source_block"]
                )
                metric_row = {
                    "experiment": EXPERIMENT,
                    "fold_id": fold.fold_id,
                    "held_out_physical_family": fold.held_out_family,
                    "dataset": dataset,
                    "physical_family": cache_row["physical_family"],
                    "source_ordinal": FIXED_SOURCE_ORDINAL,
                    "source_index": int(cache_row["source_index"]),
                    "scale_block_index": block_index,
                    "scale_block_id": block.block_id,
                    "scale_id_start": block.scale_id_start,
                    "scale_id_stop_exclusive": block.scale_id_stop,
                    "assigned_count": assigned_count,
                    "valid_count": int(len(result.labels)),
                    "invalid_count": assigned_count - int(len(result.labels)),
                    "coverage": float(len(result.labels) / assigned_count),
                    **metrics,
                }
                render_counts = rendered.metadata["counts"]
                for name in (
                    "true_positive",
                    "false_positive",
                    "true_negative",
                    "false_negative",
                ):
                    if int(metric_row[name]) != int(render_counts[name]):
                        raise RuntimeError("rendered confusion count differs from metrics")
                metric_rows.append(metric_row)
                required_exports = [
                    _export_row(scene_path, root, "scene_npz"),
                    _export_row(
                        rendered.svg_path,
                        root,
                        "svg_with_editable_text_and_rasterized_3d_marks",
                    ),
                    _export_row(
                        rendered.pdf_path,
                        root,
                        "pdf_with_editable_text_and_rasterized_3d_marks",
                    ),
                    _export_row(rendered.png_path, root, "png_360dpi"),
                    _export_row(
                        rendered.alignment_path, root, "panel_alignment_json"
                    ),
                ]
                visualization_entries.append(
                    {
                        "fold_id": fold.fold_id,
                        "held_out_physical_family": fold.held_out_family,
                        "dataset": dataset,
                        "physical_family": cache_row["physical_family"],
                        "source_ordinal": FIXED_SOURCE_ORDINAL,
                        "source_index": int(cache_row["source_index"]),
                        "scale_block_index": block_index,
                        "scale_block_id": block.block_id,
                        "source_cache": str(cache_row["path"]),
                        "source_cache_sha256": str(cache_row["file_sha256"]),
                        "fmt_prediction_sha256": canonical_array_sha256(
                            result.labels
                        ),
                        "score_sha256": canonical_array_sha256(result.scores),
                        "query_count": int(len(result.labels)),
                        "metrics": metric_row,
                        "required_exports": required_exports,
                        "additional_audit_files": [
                            _export_row(
                                scene_manifest_path, root, "scene_manifest_json"
                            ),
                            _export_row(
                                rendered.metadata_path,
                                root,
                                "render_metadata_json",
                            ),
                        ],
                    }
                )
                query_records.append(
                    QueryResultRecord(
                        fold=fold,
                        cache_row=cache_row,
                        block=block,
                        cache_mask=cache_mask,
                        cache=cache,
                        result=result,
                        library_metadata=library_metadata,
                    )
                )
        del matcher
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(visualization_entries) != 8 or len(metric_rows) != 8:
        raise RuntimeError("exactly eight dataset-by-block results are required")
    unique_keys = {
        (entry["dataset"], entry["scale_block_id"])
        for entry in visualization_entries
    }
    if len(unique_keys) != 8:
        raise RuntimeError("visualization dataset/block keys are not unique")

    library_audit_fields = (
        "fold_id",
        "held_out_physical_family",
        "dataset",
        "physical_family",
        "source_ordinal",
        "source_index",
        "scale_id",
        "scale_block_index",
        "scale_block_id",
        "negative_candidate_count",
        "positive_candidate_count",
        "selected_per_class",
        "selected_negative_seed_index",
        "selected_positive_seed_index",
        "selected_negative_assigned_row_index",
        "selected_positive_assigned_row_index",
        "selected_negative_center_seed_index",
        "selected_positive_center_seed_index",
        "skip_reason",
    )
    _atomic_csv(
        root / "library_audit.csv", library_audit_rows, library_audit_fields
    )
    library_manifest = {
        "schema": "pathline_template_matching.family_heldout_FMT_library.v1",
        "experiment": EXPERIMENT,
        "config_sha256": plan.config_sha256,
        "git_commit": git_commit,
        "parent_numerical_git_commit": PARENT_NUMERICAL_COMMIT,
        "sampler": "parent_global_PCG64_15068_restarted_per_fold",
        "held_out_family_leakage": False,
        "fold_count": len(library_manifest_rows),
        "folds": library_manifest_rows,
        "folds_content_sha256": canonical_json_sha256(library_manifest_rows),
        "library_audit_file": "library_audit.csv",
        "library_audit_file_sha256": sha256_file(root / "library_audit.csv"),
    }
    library_manifest["manifest_content_sha256"] = canonical_json_sha256(
        library_manifest
    )
    _atomic_json(root / "library_manifest.json", library_manifest)
    preprocessing_manifest = {
        "schema": "pathline_template_matching.family_heldout_FMT_preprocessing.v1",
        "experiment": EXPERIMENT,
        "query_statistics_used": False,
        "parent_fitted_artifact_reused": False,
        "folds": preprocessing_rows,
        "folds_content_sha256": canonical_json_sha256(preprocessing_rows),
    }
    preprocessing_manifest["manifest_content_sha256"] = canonical_json_sha256(
        preprocessing_manifest
    )
    _atomic_json(root / "preprocessing_manifest.json", preprocessing_manifest)

    query_fields = (
        "experiment",
        "fold_id",
        "held_out_physical_family",
        "query_dataset",
        "query_physical_family",
        "query_source_ordinal",
        "query_source_index",
        "query_assigned_row_index",
        "query_center_seed_index",
        "query_scale_block_index",
        "query_scale_block_id",
        "query_scale_id",
        "reference_label",
        "predicted_label",
        "score",
        "nearest_template_index",
        "nearest_template_distance",
        "nearest_positive_distance",
        "nearest_negative_distance",
        "match_dataset",
        "match_physical_family",
        "match_source_ordinal",
        "match_source_index",
        "match_assigned_row_index",
        "match_center_seed_index",
        "match_scale_block_index",
        "match_scale_block_id",
        "match_scale_id",
        "match_label",
    )
    _atomic_csv(root / "per_query_matches.csv", _query_rows(query_records), query_fields)
    metric_fields = (
        "experiment",
        "fold_id",
        "held_out_physical_family",
        "dataset",
        "physical_family",
        "source_ordinal",
        "source_index",
        "scale_block_index",
        "scale_block_id",
        "scale_id_start",
        "scale_id_stop_exclusive",
        "assigned_count",
        "valid_count",
        "invalid_count",
        "coverage",
        "sample_count",
        "positive_count",
        "negative_count",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "accuracy",
        "average_precision",
        "f1",
        "balanced_accuracy",
        "auroc",
        "precision",
        "recall",
        "single_class_group",
    )
    _atomic_csv(root / "per_figure_metrics.csv", metric_rows, metric_fields)
    _atomic_bytes(
        root / "main_table.md", _main_table_markdown(metric_rows).encode("utf-8")
    )

    visualization_manifest = {
        "schema": "pathline_template_matching.family_heldout_visualization.v1",
        "experiment": EXPERIMENT,
        "evidence_scope": REGIME,
        "aggregate_performance_proof": False,
        "config_sha256": plan.config_sha256,
        "git_commit": git_commit,
        "source_selection": "four_requested_datasets_fixed_source_ordinal_2_before_classification",
        "scene_selection_uses_predictions_or_metrics": False,
        "cross_block_aggregation_or_majority_vote": False,
        "unique_key": ["dataset", "scale_block_id"],
        "entry_count": len(visualization_entries),
        "entries": visualization_entries,
    }
    visualization_manifest["manifest_content_sha256"] = canonical_json_sha256(
        visualization_manifest
    )
    _atomic_json(root / "visualization_manifest.json", visualization_manifest)

    environment = {
        "schema": "pathline_template_matching.family_heldout_environment.v1",
        "experiment": EXPERIMENT,
        "git_commit": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit_image": skimage.__version__,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if device == "cuda" else None
        ),
        "deterministic_execution": execution_contract,
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cuda_tensorfloat32": False,
    }
    _atomic_json(root / "environment_versions.json", environment)

    artifact_rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"result_manifest.json", "RUN_COMPLETE.json"}:
            continue
        artifact_rows.append(
            {
                "relative_path": _relative(path, root),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    result_manifest = {
        "schema": "pathline_template_matching.family_heldout_result.v1",
        "experiment": EXPERIMENT,
        "status": "family_held_out_exposed_development_completed",
        "formal_confirmation": False,
        "aggregate_performance_proof": False,
        "git_commit": git_commit,
        "config_sha256": plan.config_sha256,
        "parent_numerical_git_commit": PARENT_NUMERICAL_COMMIT,
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "input_manifest_file_sha256": sha256_file(root / "input_manifest.json"),
        "library_manifest_file_sha256": sha256_file(root / "library_manifest.json"),
        "preprocessing_manifest_file_sha256": sha256_file(
            root / "preprocessing_manifest.json"
        ),
        "visualization_manifest_file_sha256": sha256_file(
            root / "visualization_manifest.json"
        ),
        "figure_count": 8,
        "query_count": int(sum(int(row["valid_count"]) for row in metric_rows)),
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "artifacts_content_sha256": canonical_json_sha256(artifact_rows),
    }
    result_manifest["manifest_content_sha256"] = canonical_json_sha256(
        result_manifest
    )
    _atomic_json(root / "result_manifest.json", result_manifest)
    complete = {
        "schema": "pathline_template_matching.family_heldout_run_complete.v1",
        "experiment": EXPERIMENT,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "config_sha256": plan.config_sha256,
        "run_directory": str(root),
        "result_manifest_file_sha256": sha256_file(root / "result_manifest.json"),
        "result_manifest_content_sha256": result_manifest[
            "manifest_content_sha256"
        ],
        "figure_count": 8,
        "query_count": result_manifest["query_count"],
        "completion_marker_write_order": "last_after_all_outputs_and_result_manifest_are_fsynced",
    }
    _atomic_json(root / "RUN_COMPLETE.json", complete)
    return {
        "run_dir": str(root),
        "result_manifest_file_sha256": complete["result_manifest_file_sha256"],
        "query_count": complete["query_count"],
        "figure_count": complete["figure_count"],
    }


__all__ = [
    "CONFIG_SHA256",
    "EXPERIMENT",
    "FoldSpec",
    "OtherVisualizationPlan",
    "REQUESTED_DATASETS",
    "discover_parent_train_inputs",
    "load_other_visualization_plan",
    "run_family_heldout_visualization",
]

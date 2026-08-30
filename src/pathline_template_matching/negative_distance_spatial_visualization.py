"""Audited triptychs for a frozen negative-distance spatial candidate.

``Other_NegativeDistanceSpatialVisualization_1.1`` is a downstream-only
visualization.  It authenticates eight immutable parent scenes, joins the
frozen candidate by the explicit dataset/source/block/center identity, and
changes only the prediction array and downstream analysis metadata.  It does
not select a candidate, threshold, source, block, or display pathline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import skimage
import yaml

from .phase21_pipeline import _atomic_bytes, _atomic_csv, _atomic_json, _metric_values
from .phase21_visualization import (
    DATASET_TITLES,
    FIXED_SOURCE_ORDINAL,
    SCENE31_EXTRA_ARRAY_NAMES,
    SCENE_ARRAY_NAMES,
    load_phase21_scene_artifact,
    render_phase21_scene_artifact,
    write_phase21_scene_artifact,
)
from .portable_flow import canonical_array_sha256, canonical_json_sha256, sha256_file


EXPERIMENT = "Other_NegativeDistanceSpatialVisualization_1.1"
PARENT_SCENE_EXPERIMENT = "Other_MainExp31FamilyHeldOutVisualization_1.1"
PARENT_CACHE_EXPERIMENT = "mainExp_TemplateMatching_3.1"
PARENT_CANDIDATE_EXPERIMENT = "Other_NegativeDistanceSpatial_1.1"
PARENT_SCENE_GIT_COMMIT = "86be29698eb689c0e269fe987a5b6d5f125a67be"
PARENT_SCENE_CONFIG_SHA256 = (
    "6fec35d2f64a3b593a74e8b35674137b1665ce169491e3546384142514b46670"
)
PARENT_CANDIDATE_GIT_COMMIT = "7118af6c17b964b5561e6e297609f431f81aa020"
PARENT_CANDIDATE_CONFIG_SHA256 = (
    "e891af14037c464a6042143625646be0d2f71c37e5e9ff30e50cc30dd553c141"
)
PREDICTION_MANIFEST_SHA256 = (
    "e811ed6cc861e4a153e30c0e8c54d7e720eff3b651579ad84d065efee1210c4e"
)
PARENT_PER_QUERY_SHA256 = (
    "5f2c3b303c97e0c60ee152ea7e417c9d93278e26918f98e053508da758093be0"
)
PARENT_PER_QUERY_PATH = (
    "/ibex/user/zhanx0o/pathline-template-matching/"
    "Other_MainExp31FamilyHeldOutVisualization_1.1/runs/"
    "slurm_51029080_86be29698eb6/per_query_matches.csv"
)
INPUT_ID = "main31_train_family_holdouts_source2"
SCORE_COLUMN = "masked_gaussian_rank_sigma_1"
PREDICTION_COLUMN = (
    "masked_gaussian_rank_sigma_1__fixed_top_fraction_0.05"
)
SCORE_VARIANT = "masked_gaussian_rank_sigma_1"
PREDICTION_RULE = "fixed_top_fraction_0.05"
REGIME = "family-held-out exposed-development"
DATASETS = (
    "cylinder3d",
    "halfcylinderRe640",
    "halfcylinderRe6400",
    "boeing747",
)
BLOCKS = ("legacy_2_1", "expanded_3_1")
PANEL_TITLES = (
    "IVD p95 + center pathlines",
    "FMT negative-distance + spatial top-5%",
    "TP / FP / FN / TN against IVD p95",
)
SCENE31_ARRAY_NAMES = (
    SCENE_ARRAY_NAMES[:-1] + SCENE31_EXTRA_ARRAY_NAMES + ("metadata_json",)
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset: str
    display_title: str
    fold_id: str
    held_out_family: str


@dataclass(frozen=True, slots=True)
class BlockSpec:
    block_id: str
    block_index: int
    scale_start: int
    scale_stop: int


@dataclass(frozen=True, slots=True)
class SceneInput:
    dataset: str
    block: str
    scene_npz: Path
    scene_manifest: Path
    render_metadata: Path


@dataclass(frozen=True, slots=True)
class VisualizationPlan:
    config_path: Path
    config_sha256: str
    config: dict[str, Any]
    parent_scene_run_root: Path
    parent_scene_result_manifest: Path
    parent_scene_result_sha256: str
    parent_visualization_manifest: Path
    parent_visualization_sha256: str
    candidate_run_root: Path
    candidate_result_manifest: Path
    candidate_result_sha256: str
    predictions_csv: Path
    predictions_sha256: str
    per_group_metrics_csv: Path
    per_group_metrics_sha256: str
    datasets: tuple[DatasetSpec, ...]
    blocks: tuple[BlockSpec, ...]
    prediction_semantics: str
    assigned_count: int
    metric_tolerance: float
    png_dpi: int
    output_root: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return dict(value)


def _lower_hex(value: Any, length: int) -> str:
    text = str(value)
    _require(
        len(text) == length and all(char in "0123456789abcdef" for char in text),
        f"expected a {length}-character lowercase hexadecimal digest",
    )
    return text


def _path(value: Any) -> Path:
    text = str(value).strip()
    _require(bool(text), "configured path must be non-empty")
    return Path(text).resolve()


def _inside(path: Path, root: Path, *, name: str) -> None:
    resolved = path.resolve()
    base = root.resolve()
    _require(resolved == base or base in resolved.parents, f"{name} escapes its run root")


def load_visualization_plan(config_path: str | Path) -> VisualizationPlan:
    """Parse the frozen method and reject any candidate or figure drift."""

    path = Path(config_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _mapping(payload, "config")
    _require(config.get("experiment") == EXPERIMENT, "experiment identity changed")
    _require(config.get("status") == "frozen_pre_run_not_run", "config is not frozen pre-run")

    scope = _mapping(config.get("evidence_scope"), "evidence_scope")
    _require(scope.get("formal_confirmation") is False, "formal confirmation must be false")
    _require("clustering" in scope.get("forbidden_claims", []), "clustering claim must be forbidden")

    parents = _mapping(config.get("parents"), "parents")
    scene_parent = _mapping(parents.get("family_held_out_scenes"), "family_held_out_scenes")
    candidate_parent = _mapping(
        parents.get("fixed_candidate_predictions"), "fixed_candidate_predictions"
    )
    _require(
        scene_parent.get("experiment") == PARENT_SCENE_EXPERIMENT,
        "parent scene experiment changed",
    )
    _require(
        candidate_parent.get("experiment") == PARENT_CANDIDATE_EXPERIMENT,
        "candidate parent experiment changed",
    )
    _require(
        scene_parent.get("numerical_git_commit") == PARENT_SCENE_GIT_COMMIT,
        "parent scene numerical Git commit changed",
    )
    _require(
        candidate_parent.get("numerical_git_commit") == PARENT_CANDIDATE_GIT_COMMIT,
        "candidate numerical Git commit changed",
    )
    scene_root = _path(scene_parent.get("run_root"))
    candidate_root = _path(candidate_parent.get("run_root"))
    scene_result = _path(scene_parent.get("result_manifest"))
    scene_visualization = _path(scene_parent.get("visualization_manifest"))
    candidate_result = _path(candidate_parent.get("result_manifest"))
    predictions = _path(candidate_parent.get("predictions"))
    per_group = _path(candidate_parent.get("per_group_metrics"))
    for item, root, name in (
        (scene_result, scene_root, "parent result manifest"),
        (scene_visualization, scene_root, "parent visualization manifest"),
        (candidate_result, candidate_root, "candidate result manifest"),
        (predictions, candidate_root, "candidate predictions"),
        (per_group, candidate_root, "candidate per-group metrics"),
    ):
        _inside(item, root, name=name)

    candidate = _mapping(config.get("candidate"), "candidate")
    _require(candidate.get("input_id") == INPUT_ID, "candidate input_id changed")
    _require(candidate.get("score_column") == SCORE_COLUMN, "candidate score column changed")
    _require(
        candidate.get("prediction_column") == PREDICTION_COLUMN,
        "candidate prediction column changed",
    )
    _require(float(candidate.get("sigma_grid_indices")) == 1.0, "candidate sigma changed")
    _require(float(candidate.get("fixed_top_fraction")) == 0.05, "top fraction changed")
    _require(candidate.get("oracle_threshold") == "forbidden", "oracle threshold must be forbidden")
    prediction_semantics = str(candidate.get("prediction_semantics", "")).strip()
    _require(bool(prediction_semantics), "candidate prediction semantics are required")

    query = _mapping(config.get("query"), "query")
    _require(int(query.get("source_ordinal", -1)) == FIXED_SOURCE_ORDINAL, "source ordinal changed")
    _require(query.get("complete_valid_rows_required") is True, "all valid rows are required")
    _require(
        query.get("row_join_key") == ["dataset", "source_ordinal", "block", "center_index"],
        "row join key changed",
    )
    _require(
        int(query.get("assigned_count_per_dataset_block", -1)) == 64_000,
        "assigned count per dataset/block changed",
    )
    expected_datasets = (
        DatasetSpec("cylinder3d", "Half-cylinder Re160", "holdout_half_cylinder", "half_cylinder"),
        DatasetSpec("halfcylinderRe640", "Half-cylinder Re640", "holdout_half_cylinder", "half_cylinder"),
        DatasetSpec("halfcylinderRe6400", "Half-cylinder Re6400", "holdout_half_cylinder", "half_cylinder"),
        DatasetSpec("boeing747", "Boeing 747", "holdout_boeing_747", "boeing_747"),
    )
    raw_datasets = query.get("datasets")
    _require(isinstance(raw_datasets, list), "query.datasets must be a list")
    datasets = tuple(
        DatasetSpec(
            str(row.get("id", "")),
            str(row.get("display_title", "")),
            str(row.get("fold_id", "")),
            str(row.get("held_out_physical_family", "")),
        )
        for row in raw_datasets
        if isinstance(row, Mapping)
    )
    _require(datasets == expected_datasets, "four dataset/fold identities changed")
    expected_blocks = (
        BlockSpec("legacy_2_1", 0, 0, 1000),
        BlockSpec("expanded_3_1", 1, 1000, 2000),
    )
    raw_blocks = query.get("scale_blocks")
    _require(isinstance(raw_blocks, list), "query.scale_blocks must be a list")
    blocks = tuple(
        BlockSpec(
            str(row.get("id", "")),
            int(row.get("index", -1)),
            int(row.get("scale_id_start", -1)),
            int(row.get("scale_id_stop_exclusive", -1)),
        )
        for row in raw_blocks
        if isinstance(row, Mapping)
    )
    _require(blocks == expected_blocks, "scale-block identities changed")

    metrics = _mapping(config.get("metrics"), "metrics")
    required_metrics = [
        "coverage", "accuracy", "average_precision", "f1", "balanced_accuracy",
        "auroc", "precision", "recall", "true_positive", "false_positive",
        "true_negative", "false_negative",
    ]
    _require(metrics.get("required") == required_metrics, "required metric contract changed")
    comparison = _mapping(metrics.get("comparison_to_parent_per_group_metrics"), "metric comparison")
    tolerance = float(comparison.get("absolute_tolerance", math.nan))
    _require(np.isfinite(tolerance) and 0.0 <= tolerance <= 1e-12, "metric tolerance is too loose")

    figure = _mapping(config.get("figure_contract"), "figure_contract")
    _require(figure.get("backend") == "python_matplotlib", "figure backend changed")
    _require(int(figure.get("expected_figure_count", -1)) == 8, "exactly eight figures are required")
    _require(tuple(figure.get("panel_titles", [])) == PANEL_TITLES, "panel titles changed")
    _require(int(figure.get("png_dpi", -1)) == 360, "production PNG must use 360 dpi")
    display = _mapping(figure.get("display_pathlines"), "display_pathlines")
    _require(
        (int(display.get("count", -1)), int(display.get("negative_count", -1)), int(display.get("positive_count", -1)))
        == (240, 120, 120),
        "display pathline contract changed",
    )
    execution = _mapping(config.get("execution"), "execution")
    _require(execution.get("device") == "cpu", "this downstream renderer is CPU-only")
    return VisualizationPlan(
        config_path=path,
        config_sha256=sha256_file(path),
        config=config,
        parent_scene_run_root=scene_root,
        parent_scene_result_manifest=scene_result,
        parent_scene_result_sha256=_lower_hex(scene_parent.get("result_manifest_sha256"), 64),
        parent_visualization_manifest=scene_visualization,
        parent_visualization_sha256=_lower_hex(scene_parent.get("visualization_manifest_sha256"), 64),
        candidate_run_root=candidate_root,
        candidate_result_manifest=candidate_result,
        candidate_result_sha256=_lower_hex(candidate_parent.get("result_manifest_sha256"), 64),
        predictions_csv=predictions,
        predictions_sha256=_lower_hex(candidate_parent.get("predictions_sha256"), 64),
        per_group_metrics_csv=per_group,
        per_group_metrics_sha256=_lower_hex(candidate_parent.get("per_group_metrics_sha256"), 64),
        datasets=datasets,
        blocks=blocks,
        prediction_semantics=prediction_semantics,
        assigned_count=int(query.get("assigned_count_per_dataset_block", -1)),
        metric_tolerance=tolerance,
        png_dpi=int(figure.get("png_dpi")),
        output_root=_path(execution.get("output_root")),
    )


def _stable_file(path: Path, expected_sha256: str, *, role: str) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{role} changed while hashing")
    if digest != expected_sha256:
        raise ValueError(f"{role} SHA-256 mismatch: {digest} != {expected_sha256}")
    return {"role": role, "path": str(path), "size_bytes": int(after.st_size), "sha256": digest}


def _read_self_hashed_json(path: Path, hash_field: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{path} must contain a JSON object")
    payload = dict(value)
    stored = payload.pop(hash_field, None)
    _require(stored == canonical_json_sha256(payload), f"{path.name} self-hash mismatch")
    return dict(value)


def _export_by_kind(entry: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    candidates = [
        row
        for name in ("required_exports", "additional_audit_files")
        for row in entry.get(name, [])
        if isinstance(row, Mapping) and row.get("export_kind") == kind
    ]
    _require(len(candidates) == 1, f"parent entry must contain one {kind} export")
    return candidates[0]


def _artifact_path(root: Path, row: Mapping[str, Any], *, role: str) -> Path:
    relative = str(row.get("relative_path", ""))
    _require(bool(relative), f"{role} has no relative_path")
    path = (root / relative).resolve()
    _inside(path, root, name=role)
    return path


def _manifest_artifact(
    manifest: Mapping[str, Any], relative_path: str, *, role: str
) -> Mapping[str, Any]:
    rows = manifest.get("artifacts")
    _require(isinstance(rows, list), f"{role} has no artifact list")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and (row.get("relative_path") == relative_path or row.get("path") == relative_path)
    ]
    _require(len(matches) == 1, f"{role} must contain exactly one {relative_path} artifact")
    return matches[0]


def _validate_parent_provenance(
    parent_result: Mapping[str, Any],
    parent_visualization: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    plan: VisualizationPlan,
) -> None:
    """Close the parent-result -> prediction-source provenance chain."""

    _require(parent_result.get("experiment") == PARENT_SCENE_EXPERIMENT, "parent result identity changed")
    _require(
        parent_result.get("status") == "family_held_out_exposed_development_completed",
        "parent result completion status changed",
    )
    _require(parent_result.get("git_commit") == PARENT_SCENE_GIT_COMMIT, "parent result Git commit changed")
    _require(parent_result.get("config_sha256") == PARENT_SCENE_CONFIG_SHA256, "parent result config changed")
    _require(
        parent_result.get("visualization_manifest_file_sha256")
        == plan.parent_visualization_sha256,
        "parent result does not authenticate the visualization manifest",
    )
    parent_visualization_artifact = _manifest_artifact(
        parent_result, "visualization_manifest.json", role="parent result"
    )
    _require(
        parent_visualization_artifact.get("sha256") == plan.parent_visualization_sha256,
        "parent visualization artifact SHA-256 changed",
    )
    parent_query_artifact = _manifest_artifact(
        parent_result, "per_query_matches.csv", role="parent result"
    )
    _require(
        parent_query_artifact.get("sha256") == PARENT_PER_QUERY_SHA256,
        "parent per-query artifact SHA-256 changed",
    )

    _require(parent_visualization.get("experiment") == PARENT_SCENE_EXPERIMENT, "parent visualization identity changed")
    _require(parent_visualization.get("git_commit") == PARENT_SCENE_GIT_COMMIT, "parent visualization Git commit changed")
    _require(parent_visualization.get("config_sha256") == PARENT_SCENE_CONFIG_SHA256, "parent visualization config changed")

    _require(candidate_result.get("experiment") == PARENT_CANDIDATE_EXPERIMENT, "candidate result identity changed")
    _require(candidate_result.get("status") == "complete", "candidate result completion status changed")
    _require(candidate_result.get("git_commit") == PARENT_CANDIDATE_GIT_COMMIT, "candidate result Git commit changed")
    _require(candidate_result.get("config_sha256") == PARENT_CANDIDATE_CONFIG_SHA256, "candidate result config changed")
    _require(
        candidate_result.get("prediction_manifest_file_sha256")
        == PREDICTION_MANIFEST_SHA256,
        "candidate result does not authenticate its prediction manifest",
    )
    for relative_path, expected_sha256 in (
        ("prediction_manifest.json", PREDICTION_MANIFEST_SHA256),
        ("predictions.csv", plan.predictions_sha256),
        ("per_group_metrics.csv", plan.per_group_metrics_sha256),
    ):
        artifact = _manifest_artifact(candidate_result, relative_path, role="candidate result")
        _require(
            artifact.get("sha256") == expected_sha256,
            f"candidate {relative_path} artifact SHA-256 changed",
        )

    _require(prediction_manifest.get("experiment") == PARENT_CANDIDATE_EXPERIMENT, "prediction manifest identity changed")
    _require(prediction_manifest.get("config_sha256") == PARENT_CANDIDATE_CONFIG_SHA256, "prediction manifest config changed")
    _require(prediction_manifest.get("phase") == "prediction_complete_before_explicit_reference_projection", "prediction-manifest phase changed")
    _require(prediction_manifest.get("predictions_file") == "predictions.csv", "prediction-manifest output path changed")
    _require(prediction_manifest.get("predictions_file_sha256") == plan.predictions_sha256, "prediction-manifest predictions SHA-256 changed")
    _require(prediction_manifest.get("reference_column_projection_to_prediction_logic") == "excluded", "reference entered prediction logic")
    input_files = prediction_manifest.get("input_files")
    _require(isinstance(input_files, list), "prediction manifest has no input-file list")
    selected = [
        row
        for row in input_files
        if isinstance(row, Mapping) and row.get("input_id") == INPUT_ID
    ]
    _require(len(selected) == 1, "prediction manifest must contain exactly one selected input")
    selected_input = selected[0]
    _require(selected_input.get("path") == PARENT_PER_QUERY_PATH, "selected prediction source path changed")
    _require(selected_input.get("file_sha256") == PARENT_PER_QUERY_SHA256, "selected prediction source SHA-256 changed")
    _require(int(selected_input.get("row_count", -1)) == 406_177, "selected prediction source row count changed")
    _require(selected_input.get("score_column") == "nearest_negative_distance", "selected prediction source score changed")
    _require(
        tuple(selected_input.get("observed_datasets", ()))
        == ("boeing747", "cylinder3d", "halfcylinderRe640", "halfcylinderRe6400"),
        "selected prediction source datasets changed",
    )


def authenticate_inputs(plan: VisualizationPlan) -> tuple[list[SceneInput], list[dict[str, Any]]]:
    """Authenticate all files without opening candidate rows or scene arrays."""

    prediction_manifest_path = (plan.candidate_run_root / "prediction_manifest.json").resolve()
    _inside(prediction_manifest_path, plan.candidate_run_root, name="candidate prediction manifest")
    evidence = [
        _stable_file(plan.parent_scene_result_manifest, plan.parent_scene_result_sha256, role="parent_result_manifest"),
        _stable_file(plan.parent_visualization_manifest, plan.parent_visualization_sha256, role="parent_visualization_manifest"),
        _stable_file(plan.candidate_result_manifest, plan.candidate_result_sha256, role="candidate_result_manifest"),
        _stable_file(plan.predictions_csv, plan.predictions_sha256, role="candidate_predictions_csv"),
        _stable_file(plan.per_group_metrics_csv, plan.per_group_metrics_sha256, role="candidate_per_group_metrics_csv"),
        _stable_file(prediction_manifest_path, PREDICTION_MANIFEST_SHA256, role="candidate_prediction_manifest"),
    ]
    parent_result = _read_self_hashed_json(plan.parent_scene_result_manifest, "manifest_content_sha256")
    parent_visualization = _read_self_hashed_json(plan.parent_visualization_manifest, "manifest_content_sha256")
    candidate_result = _read_self_hashed_json(plan.candidate_result_manifest, "manifest_content_sha256")
    prediction_manifest = _read_self_hashed_json(
        prediction_manifest_path, "manifest_content_sha256"
    )
    _validate_parent_provenance(
        parent_result,
        parent_visualization,
        candidate_result,
        prediction_manifest,
        plan,
    )
    entries = parent_visualization.get("entries")
    _require(isinstance(entries, list) and len(entries) == 8, "parent visualization must contain eight entries")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        _require(isinstance(entry, Mapping), "parent visualization entry must be a mapping")
        key = (str(entry.get("dataset", "")), str(entry.get("scale_block_id", "")))
        _require(key not in indexed, f"duplicate parent visualization key: {key}")
        indexed[key] = entry
    expected_keys = {(dataset.dataset, block.block_id) for dataset in plan.datasets for block in plan.blocks}
    _require(set(indexed) == expected_keys, "parent visualization dataset/block population changed")

    scenes: list[SceneInput] = []
    for dataset in plan.datasets:
        for block in plan.blocks:
            entry = indexed[(dataset.dataset, block.block_id)]
            _require(int(entry.get("source_ordinal", -1)) == FIXED_SOURCE_ORDINAL, "parent source ordinal changed")
            _require(entry.get("fold_id") == dataset.fold_id, "parent fold identity changed")
            paths: dict[str, Path] = {}
            for kind, role in (
                ("scene_npz", "parent_scene_npz"),
                ("scene_manifest_json", "parent_scene_manifest_json"),
                ("render_metadata_json", "parent_render_metadata_json"),
            ):
                row = _export_by_kind(entry, kind)
                file_path = _artifact_path(plan.parent_scene_run_root, row, role=role)
                expected_size = int(row.get("size_bytes", -1))
                item = _stable_file(file_path, _lower_hex(row.get("sha256"), 64), role=f"{role}:{dataset.dataset}:{block.block_id}")
                _require(item["size_bytes"] == expected_size, f"{role} size changed")
                evidence.append(item)
                paths[kind] = file_path
            scenes.append(
                SceneInput(
                    dataset=dataset.dataset,
                    block=block.block_id,
                    scene_npz=paths["scene_npz"],
                    scene_manifest=paths["scene_manifest_json"],
                    render_metadata=paths["render_metadata_json"],
                )
            )
    return scenes, evidence


def _strict_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    _require(str(parsed) == str(value).strip(), f"{name} must use canonical integer text")
    return parsed


def _strict_binary(value: str, *, name: str) -> bool:
    _require(str(value).strip() in {"0", "1"}, f"{name} must be 0 or 1")
    return str(value).strip() == "1"


def _csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty CSV: {path}") from error
        _require(len(header) == len(set(header)), f"duplicate CSV header: {path}")
        rows = list(reader)
    _require(all(len(row) == len(header) for row in rows), f"CSV row width mismatch: {path}")
    return header, rows


def read_candidate_groups(
    plan: VisualizationPlan,
) -> dict[tuple[str, str], dict[int, tuple[bool, float]]]:
    required = (
        "input_id",
        "dataset",
        "source_ordinal",
        "block",
        "center_index",
        SCORE_COLUMN,
        PREDICTION_COLUMN,
    )
    groups: dict[tuple[str, str], dict[int, tuple[bool, float]]] = {
        (dataset.dataset, block.block_id): {}
        for dataset in plan.datasets
        for block in plan.blocks
    }
    with plan.predictions_csv.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        header = list(reader.fieldnames or ())
        _require(bool(header), f"empty CSV: {plan.predictions_csv}")
        _require(
            len(header) == len(set(header)),
            f"duplicate CSV header: {plan.predictions_csv}",
        )
        missing = [name for name in required if name not in header]
        _require(not missing, f"candidate predictions are missing columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            _require(None not in row, f"CSV row width mismatch at line {line_number}")
            if row["input_id"] != INPUT_ID:
                continue
            source = _strict_int(
                row["source_ordinal"],
                name=f"line {line_number} source_ordinal",
            )
            dataset = row["dataset"]
            block = row["block"]
            _require(
                source == FIXED_SOURCE_ORDINAL,
                f"line {line_number} matched input_id but has unexpected source ordinal",
            )
            _require(
                dataset in DATASETS,
                f"line {line_number} matched input_id but has unexpected dataset",
            )
            _require(
                block in BLOCKS,
                f"line {line_number} matched input_id but has unexpected block",
            )
            center = _strict_int(
                row["center_index"], name=f"line {line_number} center_index"
            )
            _require(
                0 <= center < plan.assigned_count,
                f"line {line_number} center_index outside [0,64000)",
            )
            try:
                score = float(row[SCORE_COLUMN])
            except ValueError as error:
                raise ValueError(f"line {line_number} score is not numeric") from error
            _require(np.isfinite(score), f"line {line_number} score must be finite")
            prediction = _strict_binary(
                row[PREDICTION_COLUMN], name=f"line {line_number} prediction"
            )
            group = groups[(dataset, block)]
            _require(
                center not in group,
                f"duplicate candidate row identity: {(dataset, source, block, center)}",
            )
            group[center] = (prediction, score)
    _require(all(groups.values()), "one or more candidate dataset/block groups are empty")
    return groups


def exact_join_candidate_group(
    parent_centers: np.ndarray,
    candidates: Mapping[int, tuple[bool, float]],
    *,
    identity: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Require identical key population and ordering, then project values."""

    centers = np.asarray(parent_centers, dtype=np.int64)
    _require(centers.ndim == 1, "parent center identities must be one-dimensional")
    _require(
        len(np.unique(centers)) == len(centers),
        "parent scene has duplicate center indices",
    )
    candidate_order = np.fromiter(candidates.keys(), dtype=np.int64)
    parent_set = set(int(value) for value in centers)
    candidate_set = set(int(value) for value in candidate_order)
    _require(
        parent_set == candidate_set,
        f"exact join failed for {identity}: "
        f"missing={len(parent_set-candidate_set)}, "
        f"extra={len(candidate_set-parent_set)}",
    )
    _require(
        np.array_equal(candidate_order, centers),
        f"candidate row order differs from parent scene for {identity}",
    )
    prediction = np.asarray(
        [candidates[int(center)][0] for center in centers], dtype=np.bool_
    )
    score = np.asarray(
        [candidates[int(center)][1] for center in centers], dtype=np.float64
    )
    return prediction, score


def read_diagnostic_metrics(plan: VisualizationPlan) -> dict[tuple[str, str], dict[str, str]]:
    header, rows = _csv_rows(plan.per_group_metrics_csv)
    required = {
        "input_id", "dataset", "source_ordinal", "block", "score_variant", "prediction_rule",
        "sample_count", "positive_count", "negative_count", "true_positive", "false_positive",
        "true_negative", "false_negative", "coverage", "accuracy", "average_precision", "f1",
        "balanced_accuracy", "auroc", "precision", "recall",
    }
    _require(required.issubset(header), f"diagnostic metrics are missing columns: {sorted(required-set(header))}")
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        value = dict(zip(header, row, strict=True))
        if not (
            value["input_id"] == INPUT_ID
            and value["source_ordinal"] == str(FIXED_SOURCE_ORDINAL)
            and value["dataset"] in DATASETS
            and value["block"] in BLOCKS
            and value["score_variant"] == SCORE_VARIANT
            and value["prediction_rule"] == PREDICTION_RULE
        ):
            continue
        key = (value["dataset"], value["block"])
        _require(key not in result, f"duplicate diagnostic metric row: {key}")
        result[key] = value
    expected = {(dataset, block) for dataset in DATASETS for block in BLOCKS}
    _require(set(result) == expected, "diagnostic metric population is not the eight fixed groups")
    return result


def _compare_metrics(metrics: Mapping[str, Any], diagnostic: Mapping[str, str], tolerance: float) -> None:
    integer_names = (
        "sample_count", "positive_count", "negative_count", "true_positive",
        "false_positive", "true_negative", "false_negative",
    )
    for name in integer_names:
        _require(int(metrics[name]) == int(diagnostic[name]), f"diagnostic {name} mismatch")
    for name in ("coverage", "accuracy", "average_precision", "f1", "balanced_accuracy", "auroc", "precision", "recall"):
        observed = float(metrics[name])
        expected = float(diagnostic[name])
        _require(
            np.isfinite(observed) and np.isfinite(expected) and abs(observed - expected) <= tolerance,
            f"diagnostic {name} mismatch: {observed} != {expected}",
        )


def _load_parent_arrays(scene: SceneInput) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    artifact = load_phase21_scene_artifact(scene.scene_npz, scene.scene_manifest)
    _require(artifact.metadata.get("experiment") == PARENT_CACHE_EXPERIMENT, "parent scene cache experiment changed")
    _require(artifact.metadata.get("analysis_experiment") == PARENT_SCENE_EXPERIMENT, "parent scene analysis experiment changed")
    _require(artifact.metadata.get("dataset") == scene.dataset, "parent scene dataset changed")
    _require(artifact.metadata.get("scale_block_id") == scene.block, "parent scene block changed")
    _require(
        artifact.metadata.get("display_title")
        == f"{DATASET_TITLES[scene.dataset]} | {scene.block}",
        "parent scene visible title does not identify its scale block",
    )
    with np.load(scene.scene_npz, allow_pickle=False) as archive:
        _require(tuple(archive.files) == SCENE31_ARRAY_NAMES, "parent scene array schema changed")
        arrays = {name: np.asarray(archive[name]) for name in SCENE31_ARRAY_NAMES}
    return artifact.metadata, arrays


def _child_scene(parent_metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], prediction: np.ndarray) -> dict[str, Any]:
    return {
        "dataset": str(parent_metadata["dataset"]),
        "title": str(parent_metadata["display_title"]),
        "regime": REGIME,
        "source_ordinal": int(parent_metadata["source_ordinal"]),
        "bounds": arrays["bounds"],
        "seeds": arrays["seeds"],
        "reference": arrays["reference"],
        "prediction": prediction,
        "reference_seeds": arrays["seeds"],
        "prediction_seeds": arrays["seeds"],
        "display_pathlines": [value for value in arrays["display_pathlines"]],
        "ivd_points": None,
        "ivd_mesh": {
            "vertices": arrays["ivd_mesh_vertices"],
            "faces": arrays["ivd_mesh_faces"],
            "level": float(arrays["ivd_mesh_level"]),
        },
        "valid_seed_index": arrays["valid_seed_index"],
        "valid_scale_id": arrays["valid_scale_id"],
        "valid_assigned_row_index": arrays["valid_assigned_row_index"],
        "valid_center_seed_index": arrays["valid_center_seed_index"],
        "valid_scale_block_index": arrays["valid_scale_block_index"],
        "scale_block_id": str(parent_metadata["scale_block_id"]),
        "selected_query_row": arrays["selected_query_row"],
        "selected_seed_index": arrays["selected_seed_index"],
        "selected_assigned_row_index": arrays["selected_assigned_row_index"],
        "selected_center_seed_index": arrays["selected_center_seed_index"],
        "selected_scale_block_index": arrays["selected_scale_block_index"],
        "selected_reference": arrays["selected_reference"],
        "ivd_mesh_normals": arrays["ivd_mesh_normals"],
        "ivd_mesh_values": arrays["ivd_mesh_values"],
    }


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _export(path: Path, root: Path, kind: str) -> dict[str, Any]:
    _require(path.is_file() and path.stat().st_size > 0, f"missing output export: {path}")
    return {
        "relative_path": _relative(path, root),
        "export_kind": kind,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _main_table(rows: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        "# Fixed negative-distance spatial candidate: source ordinal 2",
        "",
        "| Dataset | Block | Valid / 64,000 | Coverage | Accuracy | AP | F1 | BA | AUROC | Precision | Recall | TP / FP / TN / FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                (
                    DATASET_TITLES[str(row["dataset"])], str(row["scale_block_id"]),
                    f"{int(row['valid_count']):,} / 64,000", f"{float(row['coverage']):.4%}",
                    f"{float(row['accuracy']):.4f}", f"{float(row['average_precision']):.4f}",
                    f"{float(row['f1']):.4f}", f"{float(row['balanced_accuracy']):.4f}",
                    f"{float(row['auroc']):.4f}", f"{float(row['precision']):.4f}",
                    f"{float(row['recall']):.4f}",
                    f"{int(row['true_positive']):,} / {int(row['false_positive']):,} / {int(row['true_negative']):,} / {int(row['false_negative']):,}",
                )
            ) + " |"
        )
    lines.extend((
        "", "AP is Average Precision; BA is balanced accuracy; AUROC is Area Under the Receiver Operating Characteristic Curve.",
        "These are family-held-out exposed-development figures. The fixed candidate was selected from exposed metrics, so the figures do not establish formal generalization.",
        "The 240 displayed pathlines are reference-balanced explanatory context, not the natural query class proportion.", "",
    ))
    return "\n".join(lines).encode("utf-8")


def run_negative_distance_spatial_visualization(
    plan: VisualizationPlan, *, run_dir: str | Path, git_commit: str
) -> dict[str, Any]:
    """Create eight immutable scenes, figures, metrics, and manifests."""

    _lower_hex(git_commit, 40)
    root = Path(run_dir).resolve()
    if root.exists():
        raise FileExistsError(f"immutable run directory exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    _atomic_bytes(root / "frozen_config.yaml", plan.config_path.read_bytes())
    scenes, input_rows = authenticate_inputs(plan)
    input_manifest = {
        "schema": "pathline_template_matching.negative_distance_spatial_visualization_input.v1",
        "experiment": EXPERIMENT,
        "git_commit": git_commit,
        "config_sha256": plan.config_sha256,
        "array_and_candidate_row_access_after_manifest_write_only": True,
        "input_file_count": len(input_rows),
        "files": input_rows,
        "files_content_sha256": canonical_json_sha256(input_rows),
    }
    input_manifest["manifest_content_sha256"] = canonical_json_sha256(input_manifest)
    _atomic_json(root / "input_manifest.json", input_manifest)

    candidate_groups = read_candidate_groups(plan)
    diagnostics = read_diagnostic_metrics(plan)
    metric_rows: list[dict[str, Any]] = []
    visualization_entries: list[dict[str, Any]] = []
    dataset_specs = {value.dataset: value for value in plan.datasets}
    block_specs = {value.block_id: value for value in plan.blocks}
    for parent in scenes:
        parent_metadata, arrays = _load_parent_arrays(parent)
        centers = np.asarray(arrays["valid_center_seed_index"], dtype=np.int64)
        candidates = candidate_groups[(parent.dataset, parent.block)]
        prediction, score = exact_join_candidate_group(
            centers,
            candidates,
            identity=(parent.dataset, parent.block),
        )
        reference = np.asarray(arrays["reference"], dtype=np.bool_)
        values = _metric_values(reference, prediction, score)
        coverage = len(reference) / plan.assigned_count
        metric_values = {"coverage": float(coverage), **values}
        _compare_metrics(metric_values, diagnostics[(parent.dataset, parent.block)], plan.metric_tolerance)

        parent_render = _read_self_hashed_json(parent.render_metadata, "metadata_content_sha256")
        parent_audit = dict(parent_metadata)
        audit = dict(parent_audit)
        audit.update(
            {
                "analysis_experiment": EXPERIMENT,
                "parent_analysis_experiment": PARENT_SCENE_EXPERIMENT,
                "regime": REGIME,
                "prediction_semantics": plan.prediction_semantics,
                "renderer_panel_titles": list(PANEL_TITLES),
                "renderer_prediction_semantics": plan.prediction_semantics,
                "prediction_positive_count": int(prediction.sum()),
                "prediction_negative_count": int((~prediction).sum()),
                "prediction_sha256": canonical_array_sha256(prediction),
                "candidate": {
                    "input_id": INPUT_ID,
                    "score_column": SCORE_COLUMN,
                    "prediction_column": PREDICTION_COLUMN,
                    "score_sha256": canonical_array_sha256(score),
                    "candidate_selected_from_exposed_metrics": True,
                },
                "exact_join": {
                    "key": ["dataset", "source_ordinal", "block", "center_index"],
                    "parent_count": int(len(centers)),
                    "candidate_count": int(len(candidates)),
                    "missing_count": 0,
                    "extra_count": 0,
                    "duplicate_parent_count": 0,
                    "duplicate_candidate_count": 0,
                    "ordered_center_index_sha256": canonical_array_sha256(centers),
                },
                "parent_scene_reuse": {
                    "scene_npz_sha256": sha256_file(parent.scene_npz),
                    "scene_manifest_file_sha256": sha256_file(parent.scene_manifest),
                    "unchanged_arrays": [name for name in SCENE31_ARRAY_NAMES if name not in {"prediction", "metadata_json"}],
                    "camera_source": "authenticated_parent_render_metadata",
                    "selection_changed": False,
                },
                "figure_interpretation": {
                    "scope": REGIME,
                    "formal_generalization_claim": False,
                    "display_pathlines": "240 reference-balanced context lines, not natural prevalence",
                },
                "formal_confirmation": False,
            }
        )
        scene = _child_scene(parent_metadata, arrays, prediction)
        scene_stem = root / "scenes" / f"{parent.dataset}_source_ordinal_2_{parent.block}"
        scene_path = scene_stem.with_suffix(".scene.npz")
        scene_manifest_path = scene_stem.with_suffix(".scene.json")
        write_phase21_scene_artifact(scene, audit, scene_path, scene_manifest_path)
        with np.load(scene_path, allow_pickle=False) as child:
            for name in SCENE31_ARRAY_NAMES:
                if name not in {"prediction", "metadata_json"}:
                    _require(
                        np.array_equal(np.asarray(child[name]), arrays[name])
                        and canonical_array_sha256(np.asarray(child[name]))
                        == canonical_array_sha256(arrays[name]),
                        f"parent scene array changed during clone: {name}",
                    )

        figure_stem = root / "figures" / f"{parent.dataset}_source_ordinal_2_{parent.block}_negative_distance_spatial_triptych"
        rendered = render_phase21_scene_artifact(
            scene_path, scene_manifest_path, figure_stem, dpi=plan.png_dpi
        )
        _require(rendered.svg_path is not None, "3.1-derived triptych must export SVG")
        _require(
            rendered.metadata["renderer"]["camera"] == parent_render["renderer"]["camera"],
            "rendered camera differs from the authenticated parent camera",
        )
        _require(rendered.metadata["renderer"]["panel_order"] == list(PANEL_TITLES), "renderer panel-title override failed")
        _require(rendered.metadata["renderer"]["prediction_semantics"] == plan.prediction_semantics, "renderer prediction-semantics override failed")
        counts = rendered.metadata["counts"]
        for name in ("true_positive", "false_positive", "true_negative", "false_negative"):
            _require(int(counts[name]) == int(values[name]), "rendered confusion count differs from metrics")

        dataset_spec = dataset_specs[parent.dataset]
        block_spec = block_specs[parent.block]
        metric_row = {
            "experiment": EXPERIMENT,
            "dataset": parent.dataset,
            "display_title": dataset_spec.display_title,
            "fold_id": dataset_spec.fold_id,
            "held_out_physical_family": dataset_spec.held_out_family,
            "source_ordinal": FIXED_SOURCE_ORDINAL,
            "source_index": int(parent_metadata["source_index"]),
            "scale_block_id": parent.block,
            "scale_block_index": block_spec.block_index,
            "assigned_count": plan.assigned_count,
            "valid_count": int(len(reference)),
            "invalid_count": int(plan.assigned_count - len(reference)),
            "score_variant": SCORE_VARIANT,
            "prediction_rule": PREDICTION_RULE,
            **metric_values,
        }
        metric_rows.append(metric_row)
        exports = [
            _export(scene_path, root, "scene_npz"),
            _export(scene_manifest_path, root, "scene_manifest_json"),
            _export(rendered.svg_path, root, "svg_with_editable_text_and_rasterized_3d_marks"),
            _export(rendered.pdf_path, root, "pdf_with_editable_text_and_rasterized_3d_marks"),
            _export(rendered.png_path, root, "png_360dpi"),
            _export(rendered.alignment_path, root, "panel_alignment_json"),
            _export(rendered.metadata_path, root, "render_metadata_json"),
        ]
        visualization_entries.append(
            {
                "dataset": parent.dataset,
                "source_ordinal": FIXED_SOURCE_ORDINAL,
                "scale_block_id": parent.block,
                "fold_id": dataset_spec.fold_id,
                "parent_scene_npz": str(parent.scene_npz),
                "parent_scene_npz_sha256": sha256_file(parent.scene_npz),
                "parent_scene_manifest": str(parent.scene_manifest),
                "prediction_sha256": canonical_array_sha256(prediction),
                "score_sha256": canonical_array_sha256(score),
                "exact_join_missing_count": 0,
                "exact_join_extra_count": 0,
                "query_count": int(len(reference)),
                "metrics": metric_row,
                "required_exports": exports,
            }
        )

    _require(len(metric_rows) == 8 and len(visualization_entries) == 8, "exactly eight outputs are required")
    metric_fields = tuple(metric_rows[0])
    _atomic_csv(root / "per_figure_metrics.csv", metric_rows, metric_fields)
    _atomic_bytes(root / "main_table.md", _main_table(metric_rows))
    visualization_manifest = {
        "schema": "pathline_template_matching.negative_distance_spatial_visualization.v1",
        "experiment": EXPERIMENT,
        "evidence_scope": REGIME,
        "formal_confirmation": False,
        "formal_generalization_claim": False,
        "config_sha256": plan.config_sha256,
        "git_commit": git_commit,
        "source_selection": "fixed source ordinal 2; no metric or label selection",
        "candidate_selection": "fixed before this visualization; selected from exposed development metrics",
        "cross_block_aggregation": False,
        "display_pathline_interpretation": "240 reference-balanced explanatory context lines, not natural prevalence",
        "unique_key": ["dataset", "scale_block_id"],
        "entry_count": 8,
        "entries": visualization_entries,
    }
    visualization_manifest["manifest_content_sha256"] = canonical_json_sha256(visualization_manifest)
    _atomic_json(root / "visualization_manifest.json", visualization_manifest)
    environment = {
        "schema": "pathline_template_matching.negative_distance_spatial_visualization_environment.v1",
        "experiment": EXPERIMENT,
        "git_commit": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit_image": skimage.__version__,
        "device": "cpu",
    }
    _atomic_json(root / "environment_versions.json", environment)

    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"result_manifest.json", "RUN_COMPLETE.json"}:
            artifacts.append(
                {"relative_path": _relative(path, root), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
            )
    result_manifest = {
        "schema": "pathline_template_matching.negative_distance_spatial_visualization_result.v1",
        "experiment": EXPERIMENT,
        "status": "family_held_out_exposed_development_visualization_completed",
        "formal_confirmation": False,
        "formal_generalization_claim": False,
        "git_commit": git_commit,
        "config_sha256": plan.config_sha256,
        "input_manifest_file_sha256": sha256_file(root / "input_manifest.json"),
        "visualization_manifest_file_sha256": sha256_file(root / "visualization_manifest.json"),
        "per_figure_metrics_file_sha256": sha256_file(root / "per_figure_metrics.csv"),
        "figure_count": 8,
        "query_count": int(sum(row["valid_count"] for row in metric_rows)),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifacts_content_sha256": canonical_json_sha256(artifacts),
    }
    result_manifest["manifest_content_sha256"] = canonical_json_sha256(result_manifest)
    _atomic_json(root / "result_manifest.json", result_manifest)
    complete = {
        "schema": "pathline_template_matching.negative_distance_spatial_visualization_run_complete.v1",
        "experiment": EXPERIMENT,
        "status": "complete",
        "git_commit": git_commit,
        "config_sha256": plan.config_sha256,
        "figure_count": 8,
        "query_count": result_manifest["query_count"],
        "result_manifest_file_sha256": sha256_file(root / "result_manifest.json"),
        "result_manifest_content_sha256": result_manifest["manifest_content_sha256"],
    }
    complete["marker_content_sha256"] = canonical_json_sha256(complete)
    _atomic_json(root / "RUN_COMPLETE.json", complete)
    return {
        "run_dir": str(root),
        "figure_count": 8,
        "query_count": complete["query_count"],
        "result_manifest_file_sha256": complete["result_manifest_file_sha256"],
    }


__all__ = [
    "BLOCKS",
    "DATASETS",
    "EXPERIMENT",
    "PANEL_TITLES",
    "VisualizationPlan",
    "authenticate_inputs",
    "exact_join_candidate_group",
    "load_visualization_plan",
    "read_candidate_groups",
    "run_negative_distance_spatial_visualization",
]

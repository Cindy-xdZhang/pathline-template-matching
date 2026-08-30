"""Audited triptychs for the completed NegativeTail outer-fold classifiers.

``Other_NegativeTailVisualization_1.1`` is a downstream-only visualization.
It authenticates immutable source-ordinal-2 parent scenes and the completed
``Verify_NegativeTailCalibration_1.1`` prediction chain, writes an input
manifest before opening any NPZ array, and replaces only the parent scene's
prediction and analysis metadata.  It never selects a source, candidate,
threshold, representation, scale block, or display pathline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, Sequence
import zlib

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


EXPERIMENT = "Other_NegativeTailVisualization_1.1"
CONFIG_SHA256 = "5a82a9d1af406043066316262e5dcefb1a0d559f6d66e82da16440a2066df131"
PARENT_SCENE_EXPERIMENT = "Other_MainExp31FamilyHeldOutVisualization_1.1"
PARENT_CACHE_EXPERIMENT = "mainExp_TemplateMatching_3.1"
PARENT_TAIL_EXPERIMENT = "Verify_NegativeTailCalibration_1.1"
PARENT_SCENE_GIT_COMMIT = "86be29698eb689c0e269fe987a5b6d5f125a67be"
PARENT_SCENE_CONFIG_SHA256 = (
    "6fec35d2f64a3b593a74e8b35674137b1665ce169491e3546384142514b46670"
)
PARENT_TAIL_GIT_COMMIT = "e9d4d3f11428bd2e13fc0fabf657be7c7e57db7c"
PARENT_TAIL_CONFIG_SHA256 = (
    "4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b"
)
REGIME = "family-held-out exposed-development"
DATASETS = (
    "cylinder3d",
    "halfcylinderRe640",
    "halfcylinderRe6400",
    "boeing747",
)
BLOCKS = ("legacy_2_1", "expanded_3_1")
PHYSICAL_FAMILY_ORDER = (
    "half_cylinder",
    "delta_wing",
    "f22_raptor",
    "channel",
    "boeing_747",
)
PANEL_TITLES = (
    "IVD p95 + center pathlines",
    "FMT NegativeTail template classification",
    "TP / FP / FN / TN against IVD p95",
)
PREDICTION_ARRAY_NAMES = (
    "dataset",
    "source_ordinal",
    "source_index",
    "scale_id",
    "center_seed_index",
    "scale_block_index",
    "assigned_row_index",
    "raw_negative_distance",
    "tail_probability",
    "tail_anomaly",
    "spatial_score",
    "spatial_denominator",
    "retrieval_supported",
    "calibration_supported",
    "spatial_imputed",
    "spatial_unimputable",
    "calibration_mode",
    "prediction",
)
PREDICTION_DTYPES = {
    "dataset": "<U64",
    "source_ordinal": "<i2",
    "source_index": "<i8",
    "scale_id": "<i4",
    "center_seed_index": "<i8",
    "scale_block_index": "|i1",
    "assigned_row_index": "<i8",
    "raw_negative_distance": "<f4",
    "tail_probability": "<f8",
    "tail_anomaly": "<f8",
    "spatial_score": "<f8",
    "spatial_denominator": "<f8",
    "retrieval_supported": "|b1",
    "calibration_supported": "|b1",
    "spatial_imputed": "|b1",
    "spatial_unimputable": "|b1",
    "calibration_mode": "|i1",
    "prediction": "|b1",
}
TAIL_RESULT_ARTIFACT_NAMES = (
    "final_tail_calibration.npz",
    "final_tail_calibration_manifest.json",
    "inner_candidate_summary.csv",
    "inner_fit_audits.json",
    "inner_group_metrics.csv",
    "outer_group_metrics.csv",
    "outer_prediction_manifest.json",
    "outer_predictions.npz",
    "outer_reference_access_audit.json",
    "outer_summary.json",
    "selected_candidate.json",
)
JOIN_ARRAY_NAMES = (
    "center_seed_index",
    "assigned_row_index",
    "scale_id",
    "scale_block_index",
)
SCENE31_ARRAY_NAMES = (
    SCENE_ARRAY_NAMES[:-1] + SCENE31_EXTRA_ARRAY_NAMES + ("metadata_json",)
)
PDF_STREAM_START = re.compile(rb"stream\r?\n")
PDF_TF_OPERATOR = re.compile(
    rb"/([^\s/<>]+)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s+Tf\b"
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset: str
    display_title: str
    outer_family: str


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
class FoldSpec:
    outer_family: str
    task_id: int
    expected_inner_group_count: int
    run_root: Path
    result_manifest_sha256: str
    result_manifest_content_sha256: str
    prediction_manifest_sha256: str
    prediction_manifest_content_sha256: str
    predictions_sha256: str
    predictions_size_bytes: int
    predictions_row_count: int
    selected_candidate_sha256: str
    selected_candidate_content_sha256: str
    outer_group_metrics_sha256: str
    run_complete_sha256: str
    run_complete_content_sha256: str
    candidate: dict[str, Any]

    @property
    def result_manifest(self) -> Path:
        return self.run_root / "result_manifest.json"

    @property
    def prediction_manifest(self) -> Path:
        return self.run_root / "outer_prediction_manifest.json"

    @property
    def predictions(self) -> Path:
        return self.run_root / "outer_predictions.npz"

    @property
    def selected_candidate(self) -> Path:
        return self.run_root / "selected_candidate.json"

    @property
    def outer_group_metrics(self) -> Path:
        return self.run_root / "outer_group_metrics.csv"

    @property
    def run_complete(self) -> Path:
        return self.run_root / "RUN_COMPLETE.json"


@dataclass(frozen=True, slots=True)
class VisualizationPlan:
    config_path: Path
    config_sha256: str
    config: dict[str, Any]
    parent_scene_run_root: Path
    parent_scene_result_manifest: Path
    parent_scene_result_sha256: str
    parent_scene_result_content_sha256: str
    parent_visualization_manifest: Path
    parent_visualization_sha256: str
    parent_visualization_content_sha256: str
    parent_scene_run_complete: Path
    parent_scene_run_complete_sha256: str
    folds: tuple[FoldSpec, ...]
    datasets: tuple[DatasetSpec, ...]
    blocks: tuple[BlockSpec, ...]
    assigned_count: int
    metric_tolerance: float
    png_dpi: int
    output_root: Path


@dataclass(frozen=True, slots=True)
class AuthenticatedFold:
    spec: FoldSpec
    result_manifest: dict[str, Any]
    prediction_manifest: dict[str, Any]
    selected_candidate: dict[str, Any]
    run_complete: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PredictionGroup:
    dataset: str
    source_ordinal: int
    source_index: int
    block: str
    outer_family: str
    candidate: dict[str, Any]
    center_seed_index: np.ndarray
    assigned_row_index: np.ndarray
    scale_id: np.ndarray
    scale_block_index: np.ndarray
    spatial_score: np.ndarray
    prediction: np.ndarray


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return dict(value)


def _lower_hex(value: Any, length: int = 64) -> str:
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


def _candidate_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "representation": str(candidate.get("representation", "")),
        "k": int(candidate.get("k", -1)),
        "sigma": float(candidate.get("sigma", math.nan)),
        "decision_rule": str(candidate.get("decision_rule", "")),
        "decision_value": float(candidate.get("decision_value", math.nan)),
    }


def load_visualization_plan(config_path: str | Path) -> VisualizationPlan:
    """Parse the frozen visualization contract and reject method drift."""

    path = Path(config_path).resolve()
    config_digest = sha256_file(path)
    _require(config_digest == CONFIG_SHA256, f"frozen visualization config SHA-256 mismatch: {config_digest} != {CONFIG_SHA256}")
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
    _require(config.get("experiment") == EXPERIMENT, "experiment identity changed")
    _require(config.get("status") == "frozen_pre_run_not_run", "config is not frozen pre-run")
    scope = _mapping(config.get("evidence_scope"), "evidence_scope")
    _require(scope.get("formal_confirmation") is False, "formal confirmation must be false")
    forbidden = set(scope.get("forbidden_claims", ()))
    _require({"clustering", "independent_per_primitive_classifier"}.issubset(forbidden), "required forbidden claims changed")

    parents = _mapping(config.get("parents"), "parents")
    scene_parent = _mapping(parents.get("family_held_out_scenes"), "family_held_out_scenes")
    _require(scene_parent.get("experiment") == PARENT_SCENE_EXPERIMENT, "parent scene experiment changed")
    _require(scene_parent.get("numerical_git_commit") == PARENT_SCENE_GIT_COMMIT, "parent scene Git commit changed")
    _require(scene_parent.get("config_sha256") == PARENT_SCENE_CONFIG_SHA256, "parent scene config changed")
    scene_root = _path(scene_parent.get("run_root"))
    scene_result = _path(scene_parent.get("result_manifest"))
    scene_visualization = _path(scene_parent.get("visualization_manifest"))
    scene_complete = _path(scene_parent.get("run_complete"))
    for item, name in (
        (scene_result, "parent scene result"),
        (scene_visualization, "parent visualization manifest"),
        (scene_complete, "parent scene completion marker"),
    ):
        _inside(item, scene_root, name=name)

    tail_parent = _mapping(parents.get("negative_tail_folds"), "negative_tail_folds")
    _require(tail_parent.get("experiment") == PARENT_TAIL_EXPERIMENT, "tail experiment changed")
    _require(tail_parent.get("numerical_git_commit") == PARENT_TAIL_GIT_COMMIT, "tail Git commit changed")
    _require(tail_parent.get("config_sha256") == PARENT_TAIL_CONFIG_SHA256, "tail config changed")
    _require(int(tail_parent.get("array_job_id", -1)) == 51059479, "tail array job changed")
    raw_folds = tail_parent.get("folds")
    _require(isinstance(raw_folds, list) and len(raw_folds) == 2, "exactly two tail folds are required")
    folds: list[FoldSpec] = []
    for row in raw_folds:
        value = _mapping(row, "tail fold")
        root = _path(value.get("run_root"))
        candidate = _candidate_identity(_mapping(value.get("candidate"), "fold candidate"))
        folds.append(
            FoldSpec(
                outer_family=str(value.get("outer_family", "")),
                task_id=int(value.get("task_id", -1)),
                expected_inner_group_count=int(value.get("expected_inner_group_count", -1)),
                run_root=root,
                result_manifest_sha256=_lower_hex(value.get("result_manifest_sha256")),
                result_manifest_content_sha256=_lower_hex(value.get("result_manifest_content_sha256")),
                prediction_manifest_sha256=_lower_hex(value.get("prediction_manifest_sha256")),
                prediction_manifest_content_sha256=_lower_hex(value.get("prediction_manifest_content_sha256")),
                predictions_sha256=_lower_hex(value.get("predictions_sha256")),
                predictions_size_bytes=int(value.get("predictions_size_bytes", -1)),
                predictions_row_count=int(value.get("predictions_row_count", -1)),
                selected_candidate_sha256=_lower_hex(value.get("selected_candidate_sha256")),
                selected_candidate_content_sha256=_lower_hex(value.get("selected_candidate_content_sha256")),
                outer_group_metrics_sha256=_lower_hex(value.get("outer_group_metrics_sha256")),
                run_complete_sha256=_lower_hex(value.get("run_complete_sha256")),
                run_complete_content_sha256=_lower_hex(value.get("run_complete_content_sha256")),
                candidate=candidate,
            )
        )
    expected_candidates = {
        "half_cylinder": {
            "candidate_id": "representation=chirality_all35|k=15|sigma=1.0|fixed_top_fraction=0.05",
            "representation": "chirality_all35", "k": 15, "sigma": 1.0,
            "decision_rule": "fixed_top_fraction", "decision_value": 0.05,
        },
        "boeing_747": {
            "candidate_id": "representation=real_neighbor36|k=1|sigma=1.0|fixed_top_fraction=0.05",
            "representation": "real_neighbor36", "k": 1, "sigma": 1.0,
            "decision_rule": "fixed_top_fraction", "decision_value": 0.05,
        },
    }
    _require(tuple((fold.outer_family, fold.task_id) for fold in folds) == (("half_cylinder", 0), ("boeing_747", 4)), "tail fold identity/order changed")
    _require(
        {fold.outer_family: fold.expected_inner_group_count for fold in folds}
        == {"half_cylinder": 40, "boeing_747": 56},
        "tail inner group populations changed",
    )
    for fold in folds:
        _require(fold.candidate == expected_candidates[fold.outer_family], f"frozen candidate changed for {fold.outer_family}")
        for candidate_path, name in (
            (fold.result_manifest, "tail result"),
            (fold.prediction_manifest, "tail prediction manifest"),
            (fold.predictions, "tail predictions"),
            (fold.selected_candidate, "tail selected candidate"),
            (fold.outer_group_metrics, "tail group metrics"),
            (fold.run_complete, "tail completion marker"),
        ):
            _inside(candidate_path, fold.run_root, name=name)

    prediction_contract = _mapping(config.get("prediction_contract"), "prediction_contract")
    _require(prediction_contract.get("schema") == "pathline_template_matching.negative_tail_outer_prediction.v1", "prediction schema changed")
    _require(prediction_contract.get("manifest_schema") == "pathline_template_matching.negative_tail_outer_prediction_manifest.v1", "prediction manifest schema changed")
    _require(prediction_contract.get("score_array") == "spatial_score", "score array changed")
    _require(prediction_contract.get("prediction_array") == "prediction", "prediction array changed")
    _require(int(prediction_contract.get("complete_array_count", -1)) == len(PREDICTION_ARRAY_NAMES), "prediction array count changed")
    _require(tuple(prediction_contract.get("ordered_array_names", ())) == PREDICTION_ARRAY_NAMES, "prediction array order changed")
    _require(tuple(prediction_contract.get("ordered_join_identity", ())) == ("dataset", "source_ordinal", "scale_block", *JOIN_ARRAY_NAMES), "ordered join identity changed")

    query = _mapping(config.get("query"), "query")
    _require(int(query.get("source_ordinal", -1)) == FIXED_SOURCE_ORDINAL, "source ordinal changed")
    _require(query.get("complete_valid_rows_required") is True, "all valid rows are required")
    _require(int(query.get("assigned_count_per_dataset_block", -1)) == 64_000, "assigned count changed")
    expected_datasets = (
        DatasetSpec("cylinder3d", "Half-cylinder Re160", "half_cylinder"),
        DatasetSpec("halfcylinderRe640", "Half-cylinder Re640", "half_cylinder"),
        DatasetSpec("halfcylinderRe6400", "Half-cylinder Re6400", "half_cylinder"),
        DatasetSpec("boeing747", "Boeing 747", "boeing_747"),
    )
    raw_datasets = query.get("datasets")
    _require(isinstance(raw_datasets, list), "query.datasets must be a list")
    datasets = tuple(
        DatasetSpec(str(row.get("id", "")), str(row.get("display_title", "")), str(row.get("outer_family", "")))
        for row in raw_datasets if isinstance(row, Mapping)
    )
    _require(datasets == expected_datasets, "dataset/fold identities changed")
    expected_blocks = (
        BlockSpec("legacy_2_1", 0, 0, 1000),
        BlockSpec("expanded_3_1", 1, 1000, 2000),
    )
    raw_blocks = query.get("scale_blocks")
    _require(isinstance(raw_blocks, list), "query.scale_blocks must be a list")
    blocks = tuple(
        BlockSpec(str(row.get("id", "")), int(row.get("index", -1)), int(row.get("scale_id_start", -1)), int(row.get("scale_id_stop_exclusive", -1)))
        for row in raw_blocks if isinstance(row, Mapping)
    )
    _require(blocks == expected_blocks, "scale-block identities changed")

    metrics = _mapping(config.get("metrics"), "metrics")
    _require(metrics.get("score") == "spatial_score", "metric score changed")
    required_metrics = [
        "coverage", "accuracy", "average_precision", "f1", "balanced_accuracy",
        "auroc", "precision", "recall", "true_positive", "false_positive",
        "true_negative", "false_negative",
    ]
    _require(metrics.get("required") == required_metrics, "required metric contract changed")
    comparison = _mapping(metrics.get("comparison_to_parent_outer_group_metrics"), "metric comparison")
    tolerance = float(comparison.get("absolute_tolerance", math.nan))
    _require(np.isfinite(tolerance) and 0.0 <= tolerance <= 1e-12, "metric tolerance is too loose")

    figure = _mapping(config.get("figure_contract"), "figure_contract")
    _require(figure.get("backend") == "python_matplotlib", "figure backend changed")
    _require(int(figure.get("expected_figure_count", -1)) == 8, "exactly eight figures are required")
    _require(tuple(figure.get("panel_titles", ())) == PANEL_TITLES, "panel titles changed")
    _require(int(figure.get("png_dpi", -1)) == 360, "PNG DPI changed")
    display = _mapping(figure.get("display_pathlines"), "display_pathlines")
    _require((int(display.get("count", -1)), int(display.get("negative_count", -1)), int(display.get("positive_count", -1))) == (240, 120, 120), "display pathline contract changed")
    execution = _mapping(config.get("execution"), "execution")
    _require(execution.get("device") == "cpu", "renderer must be CPU-only")
    return VisualizationPlan(
        config_path=path,
        config_sha256=config_digest,
        config=config,
        parent_scene_run_root=scene_root,
        parent_scene_result_manifest=scene_result,
        parent_scene_result_sha256=_lower_hex(scene_parent.get("result_manifest_sha256")),
        parent_scene_result_content_sha256=_lower_hex(scene_parent.get("result_manifest_content_sha256")),
        parent_visualization_manifest=scene_visualization,
        parent_visualization_sha256=_lower_hex(scene_parent.get("visualization_manifest_sha256")),
        parent_visualization_content_sha256=_lower_hex(scene_parent.get("visualization_manifest_content_sha256")),
        parent_scene_run_complete=scene_complete,
        parent_scene_run_complete_sha256=_lower_hex(scene_parent.get("run_complete_sha256")),
        folds=tuple(folds),
        datasets=datasets,
        blocks=blocks,
        assigned_count=int(query.get("assigned_count_per_dataset_block")),
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
    _require(digest == expected_sha256, f"{role} SHA-256 mismatch: {digest} != {expected_sha256}")
    return {
        "role": role,
        "path": str(path),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _read_self_hashed_json(
    path: Path,
    hash_field: str,
    *,
    expected_content_sha256: str | None = None,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{path} must contain a JSON object")
    payload = dict(value)
    stored = payload.pop(hash_field, None)
    _require(stored == canonical_json_sha256(payload), f"{path.name} self-hash mismatch")
    if expected_content_sha256 is not None:
        _require(stored == expected_content_sha256, f"{path.name} pinned content hash changed")
    return dict(value)


def _artifact_record(
    manifest: Mapping[str, Any], relative_path: str, *, role: str
) -> Mapping[str, Any]:
    rows = manifest.get("artifacts")
    if isinstance(rows, Mapping):
        _require(relative_path in rows, f"{role} has no {relative_path} artifact")
        value = rows[relative_path]
        _require(isinstance(value, Mapping), f"{role} artifact is not a mapping")
        return value
    _require(isinstance(rows, list), f"{role} has no artifact collection")
    matches = [
        row for row in rows
        if isinstance(row, Mapping)
        and (row.get("relative_path") == relative_path or row.get("path") == relative_path)
    ]
    _require(len(matches) == 1, f"{role} must contain exactly one {relative_path} artifact")
    return matches[0]


def _export_by_kind(entry: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    candidates = [
        row
        for name in ("required_exports", "additional_audit_files")
        for row in entry.get(name, [])
        if isinstance(row, Mapping) and row.get("export_kind") == kind
    ]
    _require(len(candidates) == 1, f"parent entry must contain exactly one {kind} export")
    return candidates[0]


def _artifact_path(root: Path, row: Mapping[str, Any], *, role: str) -> Path:
    relative = str(row.get("relative_path", ""))
    _require(bool(relative), f"{role} has no relative_path")
    path = (root / relative).resolve()
    _inside(path, root, name=role)
    return path


def _validate_parent_scene_chain(
    plan: VisualizationPlan,
    result: Mapping[str, Any],
    visualization: Mapping[str, Any],
    complete: Mapping[str, Any],
) -> None:
    _require(result.get("schema") == "pathline_template_matching.family_heldout_result.v1", "parent scene result schema changed")
    _require(result.get("experiment") == PARENT_SCENE_EXPERIMENT, "parent scene result identity changed")
    _require(result.get("status") == "family_held_out_exposed_development_completed", "parent scene result is not complete")
    _require(result.get("git_commit") == PARENT_SCENE_GIT_COMMIT, "parent scene result Git commit changed")
    _require(result.get("config_sha256") == PARENT_SCENE_CONFIG_SHA256, "parent scene result config changed")
    _require(result.get("manifest_content_sha256") == plan.parent_scene_result_content_sha256, "parent scene result content changed")
    _require(result.get("visualization_manifest_file_sha256") == plan.parent_visualization_sha256, "parent scene result does not bind visualization manifest")
    visualization_artifact = _artifact_record(result, "visualization_manifest.json", role="parent scene result")
    _require(visualization_artifact.get("sha256") == plan.parent_visualization_sha256, "parent visualization artifact hash changed")

    _require(visualization.get("schema") == "pathline_template_matching.family_heldout_visualization.v1", "parent visualization schema changed")
    _require(visualization.get("experiment") == PARENT_SCENE_EXPERIMENT, "parent visualization identity changed")
    _require(visualization.get("git_commit") == PARENT_SCENE_GIT_COMMIT, "parent visualization Git commit changed")
    _require(visualization.get("config_sha256") == PARENT_SCENE_CONFIG_SHA256, "parent visualization config changed")
    _require(visualization.get("manifest_content_sha256") == plan.parent_visualization_content_sha256, "parent visualization content changed")

    _require(complete.get("schema") == "pathline_template_matching.family_heldout_run_complete.v1", "parent scene completion schema changed")
    _require(complete.get("experiment") == PARENT_SCENE_EXPERIMENT, "parent scene completion identity changed")
    _require(complete.get("status") == "complete", "parent scene completion status changed")
    _require(complete.get("git_commit") == PARENT_SCENE_GIT_COMMIT, "parent scene completion Git commit changed")
    _require(complete.get("config_sha256") == PARENT_SCENE_CONFIG_SHA256, "parent scene completion config changed")
    _require(complete.get("result_manifest_file_sha256") == plan.parent_scene_result_sha256, "parent completion does not bind result file")
    _require(complete.get("result_manifest_content_sha256") == plan.parent_scene_result_content_sha256, "parent completion does not bind result content")
    _require(int(complete.get("figure_count", -1)) == 8, "parent completion figure count changed")
    _require(int(complete.get("query_count", -1)) == 406_177, "parent completion query count changed")


def _validate_prediction_array_manifest(
    manifest: Mapping[str, Any], fold: FoldSpec
) -> None:
    _require(manifest.get("schema") == "pathline_template_matching.negative_tail_outer_prediction_manifest.v1", "prediction manifest schema changed")
    _require(manifest.get("prediction_schema") == "pathline_template_matching.negative_tail_outer_prediction.v1", "prediction archive schema changed")
    _require(int(manifest.get("array_count", -1)) == len(PREDICTION_ARRAY_NAMES), "prediction manifest array count changed")
    arrays = manifest.get("arrays")
    _require(isinstance(arrays, Mapping), "prediction manifest arrays must be a mapping")
    _require(set(arrays) == set(PREDICTION_ARRAY_NAMES), "prediction manifest 18-array schema changed")
    for name in PREDICTION_ARRAY_NAMES:
        row = arrays[name]
        _require(isinstance(row, Mapping), f"prediction manifest array {name} is invalid")
        _require(row.get("dtype") == PREDICTION_DTYPES[name], f"prediction manifest dtype changed for {name}")
        _require(row.get("shape") == [fold.predictions_row_count], f"prediction manifest shape changed for {name}")
        _lower_hex(row.get("sha256"))


def _validate_fold_chain(
    fold: FoldSpec,
    result: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
    complete: Mapping[str, Any],
) -> None:
    expected = fold.candidate
    _require(result.get("schema") == "pathline_template_matching.negative_tail_result.v1", "tail result schema changed")
    _require(result.get("experiment") == PARENT_TAIL_EXPERIMENT, "tail result identity changed")
    _require(result.get("status") == "completed", "tail result is not complete")
    _require(result.get("git_commit") == PARENT_TAIL_GIT_COMMIT, "tail result Git commit changed")
    _require(result.get("config_sha256") == PARENT_TAIL_CONFIG_SHA256, "tail result config changed")
    _require(result.get("outer_family") == fold.outer_family, "wrong outer fold in tail result")
    _require(result.get("content_sha256") == fold.result_manifest_content_sha256, "tail result content changed")
    _require(_candidate_identity(_mapping(result.get("selected_candidate"), "result selected candidate")) == expected, "wrong candidate in tail result")

    expected_artifacts = {
        "outer_prediction_manifest.json": fold.prediction_manifest_sha256,
        "outer_predictions.npz": fold.predictions_sha256,
        "selected_candidate.json": fold.selected_candidate_sha256,
        "outer_group_metrics.csv": fold.outer_group_metrics_sha256,
    }
    for relative, digest in expected_artifacts.items():
        artifact = _artifact_record(result, relative, role="tail result")
        _require(artifact.get("sha256") == digest, f"tail result artifact hash changed for {relative}")
    _require(result.get("prediction_manifest_file_sha256") == fold.prediction_manifest_sha256, "tail result does not bind prediction manifest")
    _require(result.get("prediction_file_sha256") == fold.predictions_sha256, "tail result does not bind prediction archive")
    _require(result.get("selected_candidate_file_sha256") == fold.selected_candidate_sha256, "tail result does not bind selected candidate file")
    _require(result.get("selected_candidate_content_sha256") == fold.selected_candidate_content_sha256, "tail result does not bind selected candidate content")
    _require(result.get("outer_group_metrics_file_sha256") == fold.outer_group_metrics_sha256, "tail result does not bind group metrics")

    _require(prediction_manifest.get("experiment") == PARENT_TAIL_EXPERIMENT, "prediction manifest identity changed")
    _require(prediction_manifest.get("git_commit") == PARENT_TAIL_GIT_COMMIT, "prediction manifest Git commit changed")
    _require(prediction_manifest.get("config_sha256") == PARENT_TAIL_CONFIG_SHA256, "prediction manifest config changed")
    _require(prediction_manifest.get("outer_family") == fold.outer_family, "wrong outer fold in prediction manifest")
    _require(prediction_manifest.get("content_sha256") == fold.prediction_manifest_content_sha256, "prediction manifest content changed")
    _require(int(prediction_manifest.get("row_count", -1)) == fold.predictions_row_count, "prediction row count changed")
    prediction_file = _mapping(prediction_manifest.get("prediction_file"), "prediction file")
    _require(prediction_file.get("path") == "outer_predictions.npz", "prediction file path changed")
    _require(prediction_file.get("sha256") == fold.predictions_sha256, "prediction manifest does not bind prediction archive")
    _require(int(prediction_file.get("size_bytes", -1)) == fold.predictions_size_bytes, "prediction file size changed")
    _require(_candidate_identity(_mapping(prediction_manifest.get("selected_candidate"), "manifest selected candidate")) == expected, "wrong candidate in prediction manifest")
    selected_artifact = _mapping(prediction_manifest.get("selected_candidate_artifact"), "selected candidate artifact")
    _require(selected_artifact.get("path") == "selected_candidate.json", "selected candidate artifact path changed")
    _require(selected_artifact.get("file_sha256") == fold.selected_candidate_sha256, "prediction manifest does not bind selected candidate file")
    _require(selected_artifact.get("content_sha256") == fold.selected_candidate_content_sha256, "prediction manifest does not bind selected candidate content")
    _require(prediction_manifest.get("valid_labels_opened") is False, "prediction phase opened labels")
    _require(prediction_manifest.get("metadata_json_opened") is False, "prediction phase opened metadata")
    _validate_prediction_array_manifest(prediction_manifest, fold)

    _require(selected.get("schema") == "pathline_template_matching.negative_tail_selected_candidate.v1", "selected candidate schema changed")
    _require(selected.get("experiment") == PARENT_TAIL_EXPERIMENT, "selected candidate experiment changed")
    _require(selected.get("git_commit") == PARENT_TAIL_GIT_COMMIT, "selected candidate Git commit changed")
    _require(selected.get("config_sha256") == PARENT_TAIL_CONFIG_SHA256, "selected candidate config changed")
    _require(selected.get("outer_family") == fold.outer_family, "wrong outer fold in selected candidate")
    _require(selected.get("content_sha256") == fold.selected_candidate_content_sha256, "selected candidate content changed")
    _require(_candidate_identity(_mapping(selected.get("candidate"), "selected candidate")) == expected, "wrong frozen outer-fold candidate")
    _require(selected.get("outer_feature_member_opened") is False, "selected candidate was written after outer feature access")
    _require(int(selected.get("candidate_count", -1)) == 3060, "candidate population changed")
    inner_evidence = _mapping(selected.get("inner_evidence"), "selected inner evidence")
    expected_inner_evidence = {
        "inner_candidate_summary": "inner_candidate_summary.csv",
        "inner_fit_audits": "inner_fit_audits.json",
        "inner_group_metrics": "inner_group_metrics.csv",
    }
    _require(set(inner_evidence) == set(expected_inner_evidence), "selected inner evidence population changed")
    for evidence_name, artifact_name in expected_inner_evidence.items():
        evidence_row = _mapping(inner_evidence.get(evidence_name), f"selected {evidence_name}")
        artifact_row = _artifact_record(result, artifact_name, role="tail result")
        _require(evidence_row.get("path") == artifact_name, f"selected inner evidence path changed for {evidence_name}")
        _require(evidence_row.get("sha256") == artifact_row.get("sha256"), f"selected inner evidence hash changed for {evidence_name}")
        _require(int(evidence_row.get("size_bytes", -1)) == int(artifact_row.get("size_bytes", -2)), f"selected inner evidence size changed for {evidence_name}")
    inner_selection = _mapping(selected.get("inner_selection_summary"), "selected inner selection summary")
    _require(_candidate_identity(inner_selection) == expected, "inner selection summary candidate changed")
    _require(int(inner_selection.get("inner_family_count", -1)) == 4, "inner selection family population changed")
    _require(
        int(inner_selection.get("group_count", -1))
        == fold.expected_inner_group_count,
        "inner selection group population changed",
    )
    final_file = _mapping(selected.get("final_calibration_file"), "selected final calibration file")
    final_manifest = _mapping(selected.get("final_calibration_manifest"), "selected final calibration manifest")
    prediction_final_manifest = _mapping(prediction_manifest.get("final_calibration_manifest"), "prediction final calibration manifest")
    _require(result.get("final_calibration_file_sha256") == final_file.get("sha256") == prediction_manifest.get("final_calibration_file_sha256"), "final calibrator file chain is broken")
    _require(result.get("final_calibration_manifest_file_sha256") == final_manifest.get("file_sha256") == prediction_final_manifest.get("file_sha256"), "final calibrator manifest file chain is broken")
    _require(final_manifest.get("content_sha256") == prediction_final_manifest.get("content_sha256"), "final calibrator manifest content chain is broken")

    _require(complete.get("schema") == "pathline_template_matching.negative_tail_run_complete.v1", "tail completion schema changed")
    _require(complete.get("experiment") == PARENT_TAIL_EXPERIMENT, "tail completion identity changed")
    _require(complete.get("git_commit") == PARENT_TAIL_GIT_COMMIT, "tail completion Git commit changed")
    _require(complete.get("config_sha256") == PARENT_TAIL_CONFIG_SHA256, "tail completion config changed")
    _require(complete.get("outer_family") == fold.outer_family, "wrong outer fold in completion marker")
    _require(complete.get("content_sha256") == fold.run_complete_content_sha256, "tail completion content changed")
    _require(complete.get("result_manifest_file") == "result_manifest.json", "tail completion result path changed")
    _require(complete.get("result_manifest_file_sha256") == fold.result_manifest_sha256, "tail completion does not bind result file")
    _require(complete.get("result_manifest_content_sha256") == fold.result_manifest_content_sha256, "tail completion does not bind result content")


def _validate_auxiliary_fold_json(
    fold: FoldSpec,
    result: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
    final_calibration_manifest: Mapping[str, Any],
    inner_fit_audits: Mapping[str, Any],
    outer_reference_access: Mapping[str, Any],
    outer_summary: Mapping[str, Any],
) -> None:
    _require(final_calibration_manifest.get("schema") == "pathline_template_matching.negative_tail_calibration_manifest.v1", "final calibration manifest schema changed")
    _require(final_calibration_manifest.get("experiment") == PARENT_TAIL_EXPERIMENT, "final calibration experiment changed")
    _require(final_calibration_manifest.get("git_commit") == PARENT_TAIL_GIT_COMMIT, "final calibration Git commit changed")
    _require(final_calibration_manifest.get("config_sha256") == PARENT_TAIL_CONFIG_SHA256, "final calibration config changed")
    _require(final_calibration_manifest.get("outer_family") == fold.outer_family, "wrong outer fold in final calibration manifest")
    _require(final_calibration_manifest.get("outer_feature_member_opened") is False, "final calibration was published after outer feature access")
    _require(_candidate_identity(_mapping(final_calibration_manifest.get("selected_candidate"), "calibration selected candidate")) == fold.candidate, "wrong candidate in final calibration manifest")
    expected_fit_families = tuple(
        family for family in PHYSICAL_FAMILY_ORDER if family != fold.outer_family
    )
    _require(
        tuple(final_calibration_manifest.get("fit_families", ()))
        == expected_fit_families,
        "final calibration fit-family population changed or includes the outer family",
    )
    selected_final = _mapping(selected.get("final_calibration_manifest"), "selected final calibration manifest")
    prediction_final = _mapping(prediction_manifest.get("final_calibration_manifest"), "prediction final calibration manifest")
    _require(final_calibration_manifest.get("content_sha256") == selected_final.get("content_sha256") == prediction_final.get("content_sha256"), "final calibration content chain is broken")
    calibration_artifact = _artifact_record(result, "final_tail_calibration.npz", role="tail result")
    calibration_file = _mapping(final_calibration_manifest.get("artifact_file"), "final calibration artifact file")
    _require(calibration_file.get("path") == "final_tail_calibration.npz", "final calibration artifact path changed")
    # The exact file identity is already closed by the result artifact and the
    # selected/prediction manifests; keep the comparison explicit here.
    _require(calibration_artifact.get("sha256") == calibration_file.get("sha256") == result.get("final_calibration_file_sha256"), "final calibration file is not bound by result")
    _require(int(calibration_artifact.get("size_bytes", -1)) == int(calibration_file.get("size_bytes", -2)), "final calibration artifact size chain is broken")

    _require(inner_fit_audits.get("schema") == "pathline_template_matching.negative_tail_inner_fit_audits.v1", "inner fit audit schema changed")
    _require(inner_fit_audits.get("experiment") == PARENT_TAIL_EXPERIMENT, "inner fit audit experiment changed")
    _require(inner_fit_audits.get("outer_family") == fold.outer_family, "wrong outer fold in inner fit audits")
    _require(int(inner_fit_audits.get("fit_count", -1)) == 12, "inner fit audit population changed")

    _require(outer_reference_access.get("schema") == "pathline_template_matching.negative_tail_outer_reference_access.v1", "outer reference audit schema changed")
    _require(outer_reference_access.get("experiment") == PARENT_TAIL_EXPERIMENT, "outer reference audit experiment changed")
    _require(outer_reference_access.get("outer_family") == fold.outer_family, "wrong outer fold in reference audit")
    _require(outer_reference_access.get("first_open_phase") == "after_outer_prediction_file_and_manifest_authentication", "outer reference gate changed")
    _require(outer_reference_access.get("prediction_file_sha256") == fold.predictions_sha256, "outer reference audit does not bind predictions")
    _require(outer_reference_access.get("prediction_manifest_file_sha256") == fold.prediction_manifest_sha256, "outer reference audit does not bind prediction manifest")

    _require(outer_summary.get("schema") == "pathline_template_matching.negative_tail_outer_summary.v1", "outer summary schema changed")
    _require(outer_summary.get("experiment") == PARENT_TAIL_EXPERIMENT, "outer summary experiment changed")
    _require(outer_summary.get("outer_family") == fold.outer_family, "wrong outer fold in outer summary")
    outer_summary_payload = dict(outer_summary)
    outer_summary_payload.pop("content_sha256", None)
    _require(
        dict(result.get("outer_summary", {})) == outer_summary_payload,
        "result embedded outer summary differs from authenticated file",
    )

    explicit_bindings = {
        "final_tail_calibration_manifest.json": result.get("final_calibration_manifest_file_sha256"),
        "inner_candidate_summary.csv": result.get("inner_candidate_summary_file_sha256"),
        "inner_fit_audits.json": result.get("inner_fit_audits_file_sha256"),
        "inner_group_metrics.csv": result.get("inner_group_metrics_file_sha256"),
        "outer_group_metrics.csv": result.get("outer_group_metrics_file_sha256"),
        "outer_reference_access_audit.json": result.get("outer_reference_access_audit_file_sha256"),
        "outer_summary.json": result.get("outer_summary_file_sha256"),
    }
    for name, digest in explicit_bindings.items():
        _require(_artifact_record(result, name, role="tail result").get("sha256") == digest, f"tail result explicit binding changed for {name}")


def _authenticate_exact_fold_file_set(
    fold: FoldSpec,
    result: Mapping[str, Any],
    result_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Hash the exact eleven result artifacts plus result and completion files."""

    result_artifacts = result.get("artifacts")
    _require(isinstance(result_artifacts, Mapping), "tail result artifacts must be a mapping")
    _require(
        set(result_artifacts) == set(TAIL_RESULT_ARTIFACT_NAMES),
        f"tail result must authenticate exact 11-artifact set for {fold.outer_family}",
    )
    evidence = [dict(result_evidence)]
    for name in TAIL_RESULT_ARTIFACT_NAMES:
        record = _artifact_record(result, name, role="tail result")
        path = (fold.run_root / name).resolve()
        _inside(path, fold.run_root, name=f"tail artifact {name}")
        digest = _lower_hex(record.get("sha256"))
        item = _stable_file(path, digest, role=f"tail_artifact:{fold.outer_family}:{name}")
        _require(item["size_bytes"] == int(record.get("size_bytes", -1)), f"tail artifact size changed for {name}")
        evidence.append(item)
    _require(_artifact_record(result, "outer_prediction_manifest.json", role="tail result").get("sha256") == fold.prediction_manifest_sha256, "pinned prediction manifest differs from exact result artifact")
    _require(_artifact_record(result, "outer_predictions.npz", role="tail result").get("sha256") == fold.predictions_sha256, "pinned predictions differ from exact result artifact")
    _require(int(_artifact_record(result, "outer_predictions.npz", role="tail result").get("size_bytes", -1)) == fold.predictions_size_bytes, "pinned prediction size differs from exact result artifact")
    _require(_artifact_record(result, "selected_candidate.json", role="tail result").get("sha256") == fold.selected_candidate_sha256, "pinned selected candidate differs from exact result artifact")
    _require(_artifact_record(result, "outer_group_metrics.csv", role="tail result").get("sha256") == fold.outer_group_metrics_sha256, "pinned group metrics differs from exact result artifact")
    evidence.append(
        _stable_file(
            fold.run_complete,
            fold.run_complete_sha256,
            role=f"tail_run_complete:{fold.outer_family}",
        )
    )
    _require(len(evidence) == 13, "authenticated fold file set must contain exactly 13 files")
    _require(len({row["path"] for row in evidence}) == 13, "authenticated fold file set contains duplicate paths")
    return evidence


def authenticate_inputs(
    plan: VisualizationPlan,
) -> tuple[list[SceneInput], tuple[AuthenticatedFold, ...], list[dict[str, Any]]]:
    """Authenticate every configured input without opening any NPZ member."""

    evidence = [
        _stable_file(plan.parent_scene_result_manifest, plan.parent_scene_result_sha256, role="parent_scene_result_manifest"),
        _stable_file(plan.parent_visualization_manifest, plan.parent_visualization_sha256, role="parent_visualization_manifest"),
        _stable_file(plan.parent_scene_run_complete, plan.parent_scene_run_complete_sha256, role="parent_scene_run_complete"),
    ]
    parent_result = _read_self_hashed_json(
        plan.parent_scene_result_manifest,
        "manifest_content_sha256",
        expected_content_sha256=plan.parent_scene_result_content_sha256,
    )
    parent_visualization = _read_self_hashed_json(
        plan.parent_visualization_manifest,
        "manifest_content_sha256",
        expected_content_sha256=plan.parent_visualization_content_sha256,
    )
    parent_complete = json.loads(plan.parent_scene_run_complete.read_text(encoding="utf-8"))
    _require(isinstance(parent_complete, Mapping), "parent completion marker must be a JSON object")
    _validate_parent_scene_chain(plan, parent_result, parent_visualization, parent_complete)

    entries = parent_visualization.get("entries")
    _require(isinstance(entries, list) and len(entries) == 8, "parent visualization must contain eight entries")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        _require(isinstance(entry, Mapping), "parent visualization entry must be a mapping")
        key = (str(entry.get("dataset", "")), str(entry.get("scale_block_id", "")))
        _require(key not in indexed, f"duplicate parent scene key: {key}")
        indexed[key] = entry
    expected_keys = {(dataset.dataset, block.block_id) for dataset in plan.datasets for block in plan.blocks}
    _require(set(indexed) == expected_keys, "parent scene dataset/block population changed")

    scenes: list[SceneInput] = []
    for dataset in plan.datasets:
        for block in plan.blocks:
            entry = indexed[(dataset.dataset, block.block_id)]
            _require(int(entry.get("source_ordinal", -1)) == FIXED_SOURCE_ORDINAL, "parent scene source ordinal changed")
            paths: dict[str, Path] = {}
            for kind, role in (
                ("scene_npz", "parent_scene_npz"),
                ("scene_manifest_json", "parent_scene_manifest_json"),
                ("render_metadata_json", "parent_render_metadata_json"),
            ):
                row = _export_by_kind(entry, kind)
                file_path = _artifact_path(plan.parent_scene_run_root, row, role=role)
                digest = _lower_hex(row.get("sha256"))
                artifact = _artifact_record(parent_result, str(row.get("relative_path", "")), role="parent scene result")
                _require(artifact.get("sha256") == digest, f"parent result does not bind {kind}")
                _require(int(artifact.get("size_bytes", -1)) == int(row.get("size_bytes", -2)), f"parent result size differs for {kind}")
                item = _stable_file(file_path, digest, role=f"{role}:{dataset.dataset}:{block.block_id}")
                _require(item["size_bytes"] == int(row.get("size_bytes", -1)), f"{kind} size changed")
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

    authenticated_folds: list[AuthenticatedFold] = []
    for fold in plan.folds:
        result_evidence = _stable_file(
            fold.result_manifest,
            fold.result_manifest_sha256,
            role=f"tail_result_manifest:{fold.outer_family}",
        )
        result = _read_self_hashed_json(
            fold.result_manifest,
            "content_sha256",
            expected_content_sha256=fold.result_manifest_content_sha256,
        )
        evidence.extend(_authenticate_exact_fold_file_set(fold, result, result_evidence))
        prediction_manifest = _read_self_hashed_json(
            fold.prediction_manifest,
            "content_sha256",
            expected_content_sha256=fold.prediction_manifest_content_sha256,
        )
        selected = _read_self_hashed_json(
            fold.selected_candidate,
            "content_sha256",
            expected_content_sha256=fold.selected_candidate_content_sha256,
        )
        complete = _read_self_hashed_json(
            fold.run_complete,
            "content_sha256",
            expected_content_sha256=fold.run_complete_content_sha256,
        )
        _validate_fold_chain(fold, result, prediction_manifest, selected, complete)
        final_calibration_manifest = _read_self_hashed_json(
            fold.run_root / "final_tail_calibration_manifest.json",
            "content_sha256",
        )
        inner_fit_audits = _read_self_hashed_json(
            fold.run_root / "inner_fit_audits.json",
            "content_sha256",
        )
        outer_reference_access = _read_self_hashed_json(
            fold.run_root / "outer_reference_access_audit.json",
            "content_sha256",
        )
        outer_summary = _read_self_hashed_json(
            fold.run_root / "outer_summary.json",
            "content_sha256",
        )
        _validate_auxiliary_fold_json(
            fold,
            result,
            prediction_manifest,
            selected,
            final_calibration_manifest,
            inner_fit_audits,
            outer_reference_access,
            outer_summary,
        )
        authenticated_folds.append(
            AuthenticatedFold(
                spec=fold,
                result_manifest=result,
                prediction_manifest=prediction_manifest,
                selected_candidate=selected,
                run_complete=complete,
            )
        )
    return scenes, tuple(authenticated_folds), evidence


def _manifest_array_spec(
    manifest: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    arrays = manifest.get("arrays")
    _require(isinstance(arrays, Mapping), "prediction manifest arrays must be a mapping")
    value = arrays.get(name)
    _require(isinstance(value, Mapping), f"prediction manifest lacks array {name}")
    return value


def load_prediction_groups(
    plan: VisualizationPlan,
    folds: Sequence[AuthenticatedFold],
) -> dict[tuple[str, str], PredictionGroup]:
    """Validate all 18 arrays, then project only fixed source-ordinal-2 groups."""

    dataset_to_family = {value.dataset: value.outer_family for value in plan.datasets}
    block_by_index = {value.block_index: value for value in plan.blocks}
    groups: dict[tuple[str, str], PredictionGroup] = {}
    for authenticated in folds:
        fold = authenticated.spec
        expected_datasets = tuple(
            value.dataset for value in plan.datasets if value.outer_family == fold.outer_family
        )
        before = fold.predictions.stat()
        with np.load(fold.predictions, allow_pickle=False) as archive:
            _require(tuple(archive.files) == PREDICTION_ARRAY_NAMES, f"prediction NPZ array order changed for {fold.outer_family}")
            for name in PREDICTION_ARRAY_NAMES:
                array = np.asarray(archive[name])
                spec = _manifest_array_spec(authenticated.prediction_manifest, name)
                _require(array.dtype.str == PREDICTION_DTYPES[name], f"prediction NPZ dtype changed for {fold.outer_family}/{name}")
                _require(list(array.shape) == spec.get("shape"), f"prediction NPZ shape differs from manifest for {fold.outer_family}/{name}")
                _require(canonical_array_sha256(array) == spec.get("sha256"), f"prediction NPZ array hash mismatch for {fold.outer_family}/{name}")

            dataset = np.asarray(archive["dataset"])
            source_ordinal = np.asarray(archive["source_ordinal"], dtype=np.int16)
            source_index = np.asarray(archive["source_index"], dtype=np.int64)
            scale_id = np.asarray(archive["scale_id"], dtype=np.int32)
            center = np.asarray(archive["center_seed_index"], dtype=np.int64)
            block_index = np.asarray(archive["scale_block_index"], dtype=np.int8)
            assigned = np.asarray(archive["assigned_row_index"], dtype=np.int64)
            score = np.asarray(archive["spatial_score"], dtype=np.float64)
            prediction = np.asarray(archive["prediction"], dtype=np.bool_)
            _require(set(np.unique(dataset).tolist()) == set(expected_datasets), f"cross-fold or missing dataset rows in {fold.outer_family} predictions")
            _require(set(np.unique(source_ordinal).tolist()) == {0, 1, 2, 3}, f"source ordinal population changed in {fold.outer_family} predictions")
            _require(set(np.unique(block_index).tolist()) == {0, 1}, f"scale block population changed in {fold.outer_family} predictions")
            _require(np.all((scale_id >= 0) & (scale_id < 2000)), f"scale ID outside frozen range in {fold.outer_family} predictions")
            _require(np.array_equal(block_index.astype(np.int32), scale_id // 1000), f"scale ID/block identity mismatch in {fold.outer_family} predictions")
            _require(np.all((center >= 0) & (center < plan.assigned_count)), f"center identity outside frozen range in {fold.outer_family} predictions")
            _require(np.array_equal(assigned, center + block_index.astype(np.int64) * plan.assigned_count), f"assigned-row identity mismatch in {fold.outer_family} predictions")
            _require(np.isfinite(score).all(), f"non-finite spatial score in {fold.outer_family} predictions")

            for dataset_id in expected_datasets:
                for source_value in range(4):
                    for block_value in block_by_index:
                        population_mask = (
                            (dataset == dataset_id)
                            & (source_ordinal == source_value)
                            & (block_index == block_value)
                        )
                        population_count = int(population_mask.sum())
                        population_key = (dataset_id, source_value, block_value)
                        _require(population_count > 0, f"missing prediction population: {population_key}")
                        _require(
                            len(np.unique(center[population_mask])) == population_count,
                            f"duplicate center_seed_index in prediction population: {population_key}",
                        )
                        _require(
                            len(np.unique(assigned[population_mask])) == population_count,
                            f"duplicate assigned_row_index in prediction population: {population_key}",
                        )

            for dataset_id in expected_datasets:
                _require(dataset_to_family[dataset_id] == fold.outer_family, "dataset-to-fold map changed")
                for block_value, block in block_by_index.items():
                    mask = (
                        (dataset == dataset_id)
                        & (source_ordinal == FIXED_SOURCE_ORDINAL)
                        & (block_index == block_value)
                    )
                    count = int(mask.sum())
                    _require(count > 0, f"missing fixed source group: {(dataset_id, block.block_id)}")
                    group_source_index = np.unique(source_index[mask])
                    _require(len(group_source_index) == 1, f"multiple physical source indices in {(dataset_id, block.block_id)}")
                    group = PredictionGroup(
                        dataset=dataset_id,
                        source_ordinal=FIXED_SOURCE_ORDINAL,
                        source_index=int(group_source_index[0]),
                        block=block.block_id,
                        outer_family=fold.outer_family,
                        candidate=dict(fold.candidate),
                        center_seed_index=np.array(center[mask], dtype=np.int64, copy=True),
                        assigned_row_index=np.array(assigned[mask], dtype=np.int64, copy=True),
                        scale_id=np.array(scale_id[mask], dtype=np.int32, copy=True),
                        scale_block_index=np.array(block_index[mask], dtype=np.int8, copy=True),
                        spatial_score=np.array(score[mask], dtype=np.float64, copy=True),
                        prediction=np.array(prediction[mask], dtype=np.bool_, copy=True),
                    )
                    identity = _identity_records(
                        group.center_seed_index,
                        group.assigned_row_index,
                        group.scale_id,
                        group.scale_block_index,
                    )
                    _require(len(np.unique(identity)) == count, f"duplicate prediction row identity in {(dataset_id, block.block_id)}")
                    key = (dataset_id, block.block_id)
                    _require(key not in groups, f"duplicate prediction group: {key}")
                    groups[key] = group
        after = fold.predictions.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"prediction archive changed while reading: {fold.outer_family}")
        _require(sha256_file(fold.predictions) == fold.predictions_sha256, f"prediction archive changed after array validation: {fold.outer_family}")
    expected_keys = {(dataset.dataset, block.block_id) for dataset in plan.datasets for block in plan.blocks}
    _require(set(groups) == expected_keys, "prediction group population is not the eight fixed figures")
    for dataset in plan.datasets:
        source_indices = {
            groups[(dataset.dataset, block.block_id)].source_index
            for block in plan.blocks
        }
        _require(
            len(source_indices) == 1,
            f"legacy/expanded prediction groups use different source_index for {dataset.dataset}",
        )
    return groups


def _identity_records(
    center_seed_index: np.ndarray,
    assigned_row_index: np.ndarray,
    scale_id: np.ndarray,
    scale_block_index: np.ndarray,
) -> np.ndarray:
    arrays = (
        np.asarray(center_seed_index, dtype=np.int64),
        np.asarray(assigned_row_index, dtype=np.int64),
        np.asarray(scale_id, dtype=np.int32),
        np.asarray(scale_block_index, dtype=np.int8),
    )
    _require(all(value.ndim == 1 for value in arrays), "join identity arrays must be one-dimensional")
    _require(len({len(value) for value in arrays}) == 1, "join identity arrays have different lengths")
    records = np.empty(
        len(arrays[0]),
        dtype=[("center", "<i8"), ("assigned", "<i8"), ("scale", "<i4"), ("block", "|i1")],
    )
    for name, value in zip(records.dtype.names or (), arrays, strict=True):
        records[name] = value
    return records


def exact_bind_prediction_group(
    parent_metadata: Mapping[str, Any],
    parent_arrays: Mapping[str, np.ndarray],
    group: PredictionGroup,
) -> tuple[np.ndarray, np.ndarray]:
    """Fail closed unless parent and prediction identities match in exact order."""

    _require(parent_metadata.get("dataset") == group.dataset, "prediction dataset differs from parent scene")
    _require(int(parent_metadata.get("source_ordinal", -1)) == group.source_ordinal, "prediction source ordinal differs from parent scene")
    _require(int(parent_metadata.get("source_index", -1)) == group.source_index, "prediction source index differs from parent scene")
    _require(parent_metadata.get("scale_block_id") == group.block, "prediction scale block differs from parent scene")
    parent = _identity_records(
        parent_arrays["valid_center_seed_index"],
        parent_arrays["valid_assigned_row_index"],
        parent_arrays["valid_scale_id"],
        parent_arrays["valid_scale_block_index"],
    )
    candidate = _identity_records(
        group.center_seed_index,
        group.assigned_row_index,
        group.scale_id,
        group.scale_block_index,
    )
    _require(
        len(np.unique(parent_arrays["valid_center_seed_index"])) == len(parent),
        "parent scene has duplicate center_seed_index",
    )
    _require(
        len(np.unique(parent_arrays["valid_assigned_row_index"])) == len(parent),
        "parent scene has duplicate assigned_row_index",
    )
    _require(
        len(np.unique(group.center_seed_index)) == len(candidate),
        "prediction group has duplicate center_seed_index",
    )
    _require(
        len(np.unique(group.assigned_row_index)) == len(candidate),
        "prediction group has duplicate assigned_row_index",
    )
    _require(len(np.unique(parent)) == len(parent), "parent scene has duplicate row identity")
    _require(len(np.unique(candidate)) == len(candidate), "prediction group has duplicate row identity")
    parent_sorted = np.sort(parent, order=parent.dtype.names)
    candidate_sorted = np.sort(candidate, order=candidate.dtype.names)
    if not np.array_equal(parent_sorted, candidate_sorted):
        parent_bytes = {value.tobytes() for value in parent}
        candidate_bytes = {value.tobytes() for value in candidate}
        raise ValueError(
            f"exact join failed for {(group.dataset, group.block)}: "
            f"missing={len(parent_bytes-candidate_bytes)}, "
            f"extra={len(candidate_bytes-parent_bytes)}"
        )
    _require(np.array_equal(parent, candidate), f"prediction row order differs from parent scene for {(group.dataset, group.block)}")
    _require(group.prediction.shape == (len(parent),), "prediction vector shape changed")
    _require(group.spatial_score.shape == (len(parent),), "score vector shape changed")
    _require(np.isfinite(group.spatial_score).all(), "prediction score contains NaN or Inf")
    return np.array(group.prediction, copy=True), np.array(group.spatial_score, copy=True)


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


def read_parent_group_metrics(
    plan: VisualizationPlan,
    folds: Sequence[AuthenticatedFold],
) -> dict[tuple[str, str], dict[str, str]]:
    required = {
        "outer_family", "inner_family", "dataset", "source_ordinal", "block",
        "candidate_id", "representation", "k", "sigma", "decision_rule",
        "decision_value", "sample_count", "positive_count", "negative_count",
        "true_positive", "false_positive", "true_negative", "false_negative",
        "accuracy", "average_precision", "f1", "balanced_accuracy", "auroc",
        "precision", "recall",
    }
    dataset_to_family = {value.dataset: value.outer_family for value in plan.datasets}
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for authenticated in folds:
        fold = authenticated.spec
        header, rows = _csv_rows(fold.outer_group_metrics)
        _require(required.issubset(header), f"outer group metrics missing columns: {sorted(required-set(header))}")
        expected_datasets = {name for name, family in dataset_to_family.items() if family == fold.outer_family}
        expected_population = {
            (dataset, source, block)
            for dataset in expected_datasets
            for source in range(4)
            for block in BLOCKS
        }
        observed_population: set[tuple[str, int, str]] = set()
        for raw in rows:
            value = dict(zip(header, raw, strict=True))
            _require(value["outer_family"] == fold.outer_family, "wrong outer fold in group metrics")
            _require(value["inner_family"] == "outer_evaluation_only", "group metrics include a non-outer row")
            _require(value["dataset"] in expected_datasets, "cross-fold dataset in group metrics")
            source = int(value["source_ordinal"])
            _require(source in {0, 1, 2, 3}, "unexpected source ordinal in group metrics")
            _require(value["block"] in BLOCKS, "unexpected scale block in group metrics")
            _require(value["candidate_id"] == fold.candidate["candidate_id"], "wrong candidate ID in group metrics")
            _require(value["representation"] == fold.candidate["representation"], "wrong representation in group metrics")
            _require(int(value["k"]) == fold.candidate["k"], "wrong k in group metrics")
            _require(abs(float(value["sigma"]) - fold.candidate["sigma"]) <= 0.0, "wrong sigma in group metrics")
            _require(value["decision_rule"] == fold.candidate["decision_rule"], "wrong decision rule in group metrics")
            _require(abs(float(value["decision_value"]) - fold.candidate["decision_value"]) <= 0.0, "wrong decision value in group metrics")
            population_key = (value["dataset"], source, value["block"])
            _require(population_key not in observed_population, f"duplicate outer metric row: {population_key}")
            observed_population.add(population_key)
            if source == FIXED_SOURCE_ORDINAL:
                key = (value["dataset"], value["block"])
                _require(key not in selected, f"duplicate fixed-source metric row: {key}")
                selected[key] = value
        _require(observed_population == expected_population, f"outer metric population changed for {fold.outer_family}")
        _require(sha256_file(fold.outer_group_metrics) == fold.outer_group_metrics_sha256, f"outer group metrics changed while reading: {fold.outer_family}")
    expected_keys = {(dataset.dataset, block.block_id) for dataset in plan.datasets for block in plan.blocks}
    _require(set(selected) == expected_keys, "fixed-source metric population is not the eight figures")
    return selected


def compare_metrics_to_parent(
    metrics: Mapping[str, Any],
    parent: Mapping[str, str],
    tolerance: float,
) -> None:
    for name in (
        "sample_count", "positive_count", "negative_count", "true_positive",
        "false_positive", "true_negative", "false_negative",
    ):
        _require(int(metrics[name]) == int(parent[name]), f"parent outer-group {name} mismatch")
    for name in (
        "accuracy", "average_precision", "f1", "balanced_accuracy", "auroc",
        "precision", "recall",
    ):
        observed = float(metrics[name])
        expected = float(parent[name])
        _require(
            np.isfinite(observed)
            and np.isfinite(expected)
            and abs(observed - expected) <= tolerance,
            f"parent outer-group {name} mismatch: {observed} != {expected}",
        )


def _load_parent_arrays(
    scene: SceneInput,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    artifact = load_phase21_scene_artifact(scene.scene_npz, scene.scene_manifest)
    _require(artifact.metadata.get("experiment") == PARENT_CACHE_EXPERIMENT, "parent scene cache experiment changed")
    _require(artifact.metadata.get("analysis_experiment") == PARENT_SCENE_EXPERIMENT, "parent scene analysis experiment changed")
    _require(artifact.metadata.get("dataset") == scene.dataset, "parent scene dataset changed")
    _require(artifact.metadata.get("scale_block_id") == scene.block, "parent scene block changed")
    _require(int(artifact.metadata.get("source_ordinal", -1)) == FIXED_SOURCE_ORDINAL, "parent scene source ordinal changed")
    _require(
        artifact.metadata.get("display_title") == f"{DATASET_TITLES[scene.dataset]} | {scene.block}",
        "parent scene visible title does not identify its scale block",
    )
    with np.load(scene.scene_npz, allow_pickle=False) as archive:
        _require(tuple(archive.files) == SCENE31_ARRAY_NAMES, "parent scene array schema changed")
        arrays = {name: np.asarray(archive[name]) for name in SCENE31_ARRAY_NAMES}
    return artifact.metadata, arrays


def _child_scene(
    parent_metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    prediction: np.ndarray,
) -> dict[str, Any]:
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


def _audit_pdf_text(
    pdf_path: Path,
    report_path: Path,
    *,
    minimum_pt: float = 5.0,
) -> dict[str, Any]:
    """Dependency-free audit of final-PDF ``Tf`` font-size operators."""

    _require(minimum_pt > 0 and math.isfinite(minimum_pt), "PDF text minimum must be positive")
    data = pdf_path.read_bytes()
    _require(data.startswith(b"%PDF-"), f"not a PDF file: {pdf_path}")
    streams: list[bytes] = []
    warnings: list[str] = []
    cursor = 0
    stream_number = 0
    while True:
        match = PDF_STREAM_START.search(data, cursor)
        if match is None:
            break
        stream_number += 1
        end = data.find(b"endstream", match.end())
        if end < 0:
            warnings.append(f"stream {stream_number} has no endstream marker")
            break
        payload = data[match.end():end]
        header = data[max(0, match.start() - 2048):match.start()]
        dictionary_start = header.rfind(b"<<")
        dictionary = header[dictionary_start:] if dictionary_start >= 0 else header
        if b"/FlateDecode" in dictionary:
            try:
                payload = zlib.decompress(payload)
            except zlib.error as error:
                warnings.append(f"stream {stream_number} FlateDecode failed: {error}")
                cursor = end + len(b"endstream")
                continue
        elif b"/Filter" in dictionary:
            warnings.append(f"stream {stream_number} uses an unsupported PDF filter")
            cursor = end + len(b"endstream")
            continue
        streams.append(payload)
        cursor = end + len(b"endstream")

    runs: list[dict[str, Any]] = []
    for decoded_index, stream in enumerate(streams, 1):
        for match in PDF_TF_OPERATOR.finditer(stream):
            try:
                size = float(match.group(2))
            except ValueError:
                continue
            if size > 0:
                runs.append(
                    {
                        "stream": decoded_index,
                        "font": match.group(1).decode("ascii", errors="replace"),
                        "size_pt": size,
                    }
                )
    below = [row for row in runs if float(row["size_pt"]) < minimum_pt]
    report = {
        "schema": "pathline_template_matching.negative_tail_pdf_text_audit.v1",
        "experiment": EXPERIMENT,
        "pdf": str(pdf_path),
        "pdf_sha256": sha256_file(pdf_path),
        "auditable": bool(runs),
        "minimum_required_pt": minimum_pt,
        "minimum_found_pt": min((float(row["size_pt"]) for row in runs), default=None),
        "text_run_count": len(runs),
        "below_minimum_count": len(below),
        "below_minimum": below,
        "warnings": warnings,
        "status": "PASS" if runs and not below else "FAIL",
        "scope_note": "Tf scanning does not replace final-size visual inspection or rendered collision review.",
    }
    _require(report_path != pdf_path and not report_path.exists(), f"refusing to overwrite PDF text audit: {report_path}")
    _atomic_json(report_path, report)
    _require(report["auditable"], f"PDF text is not auditable: {pdf_path}")
    _require(not below, f"PDF contains {len(below)} text runs below {minimum_pt:g} pt: {pdf_path}")
    return report


def _main_table(rows: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        "# NegativeTail FMT-template classification: source ordinal 2",
        "",
        "| Dataset | Block | Outer-fold candidate | Valid / 64,000 | Coverage | Accuracy | AP | F1 | BA | AUROC | Precision | Recall | TP / FP / TN / FN |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    DATASET_TITLES[str(row["dataset"])],
                    str(row["scale_block_id"]),
                    str(row["candidate_id"]),
                    f"{int(row['valid_count']):,} / 64,000",
                    f"{float(row['coverage']):.4%}",
                    f"{float(row['accuracy']):.4f}",
                    f"{float(row['average_precision']):.4f}",
                    f"{float(row['f1']):.4f}",
                    f"{float(row['balanced_accuracy']):.4f}",
                    f"{float(row['auroc']):.4f}",
                    f"{float(row['precision']):.4f}",
                    f"{float(row['recall']):.4f}",
                    f"{int(row['true_positive']):,} / {int(row['false_positive']):,} / {int(row['true_negative']):,} / {int(row['false_negative']):,}",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "AP is Average Precision; BA is balanced accuracy; AUROC is Area Under the Receiver Operating Characteristic Curve.",
            "The half-cylinder outer fold uses chirality_all35, k=15; the Boeing outer fold uses real_neighbor36, k=1. Both use sigma=1 and a fixed top-5% decision, as selected by the already completed nested experiment.",
            "These are fixed-source family-held-out exposed-development figures, not formal confirmation or multi-source aggregate evidence.",
            "The fixed top-5% decision is group-transductive. The 240 displayed pathlines are reference-balanced explanatory context, not the natural query prevalence.",
            "Visual key (not drawn as an in-panel legend to preserve the frozen FMT layout): panel b red=predicted vortex (alpha 0.92), blue=predicted non-vortex (alpha 0.24); panel c red circle=true positive, purple triangle=false positive, orange x=false negative, faint blue circle=true negative (alpha 0.035). More opaque/larger error marks may occlude faint background points by design.",
            "No confidence interval is reported because each figure contains one preregistered source timeslice.",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _verify_artifact_rows(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        path = (root / str(row["relative_path"])).resolve()
        _inside(path, root, name="result artifact")
        _require(path.is_file(), f"result artifact is missing: {path}")
        _require(path.stat().st_size == int(row["size_bytes"]), f"result artifact size changed: {path}")
        _require(sha256_file(path) == row["sha256"], f"result artifact hash changed: {path}")


def run_negative_tail_visualization(
    plan: VisualizationPlan,
    *,
    run_dir: str | Path,
    git_commit: str,
) -> dict[str, Any]:
    """Create eight immutable NegativeTail scenes, triptychs, and evidence."""

    _lower_hex(git_commit, 40)
    root = Path(run_dir).resolve()
    if root.exists():
        raise FileExistsError(f"immutable run directory exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    _atomic_bytes(root / "frozen_config.yaml", plan.config_path.read_bytes())

    scenes, authenticated_folds, input_rows = authenticate_inputs(plan)
    input_manifest = {
        "schema": "pathline_template_matching.negative_tail_visualization_input.v1",
        "experiment": EXPERIMENT,
        "git_commit": git_commit,
        "config_sha256": plan.config_sha256,
        "npz_array_access_before_manifest_write": False,
        "npz_array_access_after_manifest_write_only": True,
        "input_file_count": len(input_rows),
        "files": input_rows,
        "files_content_sha256": canonical_json_sha256(input_rows),
    }
    input_manifest["manifest_content_sha256"] = canonical_json_sha256(input_manifest)
    _atomic_json(root / "input_manifest.json", input_manifest)

    # This is deliberately the first point at which either parent scene or
    # NegativeTail prediction NPZ arrays may be opened.
    prediction_groups = load_prediction_groups(plan, authenticated_folds)
    parent_metrics = read_parent_group_metrics(plan, authenticated_folds)
    dataset_specs = {value.dataset: value for value in plan.datasets}
    block_specs = {value.block_id: value for value in plan.blocks}
    metric_rows: list[dict[str, Any]] = []
    visualization_entries: list[dict[str, Any]] = []

    for parent in scenes:
        parent_metadata, arrays = _load_parent_arrays(parent)
        group = prediction_groups[(parent.dataset, parent.block)]
        prediction, score = exact_bind_prediction_group(parent_metadata, arrays, group)
        reference = np.asarray(arrays["reference"], dtype=np.bool_)
        values = _metric_values(reference, prediction, score)
        compare_metrics_to_parent(values, parent_metrics[(parent.dataset, parent.block)], plan.metric_tolerance)
        coverage = len(reference) / plan.assigned_count

        parent_render = _read_self_hashed_json(parent.render_metadata, "metadata_content_sha256")
        candidate = dict(group.candidate)
        prediction_semantics = (
            "FMT NegativeTail outer-fold template classification using "
            f"{candidate['representation']}, exact same-scale k={candidate['k']} negative retrieval, "
            f"fit-negative tail calibration, support-mask-normalized Gaussian sigma={candidate['sigma']:g}, "
            "and a fixed top-5% positive decision within this dataset/source/block group."
        )
        audit = dict(parent_metadata)
        audit.update(
            {
                "analysis_experiment": EXPERIMENT,
                "parent_analysis_experiment": PARENT_SCENE_EXPERIMENT,
                "prediction_parent_experiment": PARENT_TAIL_EXPERIMENT,
                "regime": REGIME,
                "outer_family": group.outer_family,
                "candidate": candidate,
                "candidate_id": candidate["candidate_id"],
                "prediction_semantics": prediction_semantics,
                "renderer_panel_titles": list(PANEL_TITLES),
                "renderer_prediction_semantics": prediction_semantics,
                "prediction_positive_count": int(prediction.sum()),
                "prediction_negative_count": int((~prediction).sum()),
                "prediction_sha256": canonical_array_sha256(prediction),
                "spatial_score_sha256": canonical_array_sha256(score),
                "exact_ordered_join": {
                    "key": ["dataset", "source_ordinal", "scale_block", *JOIN_ARRAY_NAMES],
                    "parent_count": int(len(reference)),
                    "prediction_count": int(len(prediction)),
                    "missing_count": 0,
                    "extra_count": 0,
                    "duplicate_parent_count": 0,
                    "duplicate_prediction_count": 0,
                    "ordered_identity_sha256": {
                        "center_seed_index": canonical_array_sha256(group.center_seed_index),
                        "assigned_row_index": canonical_array_sha256(group.assigned_row_index),
                        "scale_id": canonical_array_sha256(group.scale_id),
                        "scale_block_index": canonical_array_sha256(group.scale_block_index),
                    },
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
                    "confidence_interval": "none; one preregistered source timeslice",
                    "decision_dependency": "fixed top-5% is group-transductive",
                    "display_pathlines": "240 reference-balanced context lines, not natural prevalence",
                    "coverage_warning": "legacy and expanded blocks contain different valid populations",
                    "visual_key": {
                        "panel_b": {
                            "predicted_vortex": "red circle, alpha 0.92",
                            "predicted_non_vortex": "blue circle, alpha 0.24",
                        },
                        "panel_c": {
                            "true_positive": "red circle, alpha 0.92",
                            "false_positive": "purple triangle, alpha 0.90",
                            "false_negative": "orange x, alpha 0.95",
                            "true_negative": "faint blue circle, alpha 0.035",
                        },
                        "occlusion_note": "Larger opaque error marks may occlude faint background points by design.",
                        "in_panel_legend": False,
                        "layout_reason": "unchanged inherited FMT triptych layout",
                    },
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
            _require(tuple(child.files) == SCENE31_ARRAY_NAMES, "child scene array schema changed")
            child_prediction = np.asarray(child["prediction"])
            _require(
                np.array_equal(child_prediction, prediction)
                and canonical_array_sha256(child_prediction)
                == canonical_array_sha256(prediction),
                "child scene prediction differs from exact-joined prediction",
            )
            for name in SCENE31_ARRAY_NAMES:
                if name not in {"prediction", "metadata_json"}:
                    child_array = np.asarray(child[name])
                    _require(
                        np.array_equal(child_array, arrays[name])
                        and canonical_array_sha256(child_array) == canonical_array_sha256(arrays[name]),
                        f"parent scene array changed during clone: {name}",
                    )

        figure_stem = root / "figures" / f"{parent.dataset}_source_ordinal_2_{parent.block}_negative_tail_triptych"
        rendered = render_phase21_scene_artifact(
            scene_path,
            scene_manifest_path,
            figure_stem,
            dpi=plan.png_dpi,
        )
        _require(rendered.svg_path is not None, "3D triptych must export SVG")
        _require(rendered.metadata["renderer"]["camera"] == parent_render["renderer"]["camera"], "rendered camera differs from authenticated parent")
        _require(rendered.metadata["renderer"]["panel_order"] == list(PANEL_TITLES), "renderer panel-title override failed")
        _require(rendered.metadata["renderer"]["prediction_semantics"] == prediction_semantics, "renderer prediction semantics changed")
        counts = rendered.metadata["counts"]
        for name in ("true_positive", "false_positive", "true_negative", "false_negative"):
            _require(int(counts[name]) == int(values[name]), "rendered confusion count differs from metrics")
        pdf_text_path = figure_stem.with_suffix(".pdf_text_audit.json")
        pdf_text_report = _audit_pdf_text(
            rendered.pdf_path,
            pdf_text_path,
            minimum_pt=5.0,
        )
        _require(pdf_text_report["pdf_sha256"] == sha256_file(rendered.pdf_path), "PDF text audit is not bound to the rendered PDF")

        dataset_spec = dataset_specs[parent.dataset]
        block_spec = block_specs[parent.block]
        metric_row = {
            "experiment": EXPERIMENT,
            "dataset": parent.dataset,
            "display_title": dataset_spec.display_title,
            "outer_family": dataset_spec.outer_family,
            "source_ordinal": FIXED_SOURCE_ORDINAL,
            "source_index": int(parent_metadata["source_index"]),
            "scale_block_id": parent.block,
            "scale_block_index": block_spec.block_index,
            "assigned_count": plan.assigned_count,
            "valid_count": int(len(reference)),
            "invalid_count": int(plan.assigned_count - len(reference)),
            "coverage": float(coverage),
            "candidate_id": candidate["candidate_id"],
            "representation": candidate["representation"],
            "k": candidate["k"],
            "sigma": candidate["sigma"],
            "decision_rule": candidate["decision_rule"],
            "decision_value": candidate["decision_value"],
            "score_array": "spatial_score",
            **values,
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
        additional_audits = [
            _export(pdf_text_path, root, "pdf_text_minimum_5pt_audit_json"),
        ]
        visualization_entries.append(
            {
                "dataset": parent.dataset,
                "source_ordinal": FIXED_SOURCE_ORDINAL,
                "scale_block_id": parent.block,
                "outer_family": group.outer_family,
                "candidate": candidate,
                "candidate_id": candidate["candidate_id"],
                "parent_scene_npz": str(parent.scene_npz),
                "parent_scene_npz_sha256": sha256_file(parent.scene_npz),
                "parent_scene_manifest": str(parent.scene_manifest),
                "prediction_parent_npz": str(next(value.spec.predictions for value in authenticated_folds if value.spec.outer_family == group.outer_family)),
                "prediction_sha256": canonical_array_sha256(prediction),
                "spatial_score_sha256": canonical_array_sha256(score),
                "exact_join_missing_count": 0,
                "exact_join_extra_count": 0,
                "query_count": int(len(reference)),
                "metrics": metric_row,
                "required_exports": exports,
                "additional_audit_files": additional_audits,
                "post_download_qa_required": {
                    "rendered_collision_audit": "Run the frozen nature-figure PyMuPDF audit locally after download; Ibex deepvortex has no PyMuPDF.",
                    "original_png_visual_review": True,
                },
            }
        )
        _require(sha256_file(parent.scene_npz) == next(row["sha256"] for row in input_rows if row["path"] == str(parent.scene_npz)), "parent scene changed after array use")

    _require(len(metric_rows) == 8 and len(visualization_entries) == 8, "exactly eight outputs are required")
    metric_fields = tuple(metric_rows[0])
    _atomic_csv(root / "per_figure_metrics.csv", metric_rows, metric_fields)
    _atomic_bytes(root / "main_table.md", _main_table(metric_rows))
    visualization_manifest = {
        "schema": "pathline_template_matching.negative_tail_visualization.v1",
        "experiment": EXPERIMENT,
        "evidence_scope": REGIME,
        "formal_confirmation": False,
        "formal_generalization_claim": False,
        "config_sha256": plan.config_sha256,
        "git_commit": git_commit,
        "source_selection": "fixed source ordinal 2; no metric or label selection",
        "candidate_selection": "two frozen candidates from the already completed nested outer-fold experiment; no visualization-time selection",
        "candidate_scope": "half-cylinder and Boeing use their own pre-existing outer-fold candidate",
        "cross_block_aggregation": False,
        "confidence_interval": "none; one preregistered source timeslice per figure",
        "display_pathline_interpretation": "240 reference-balanced explanatory context lines, not natural prevalence",
        "unique_key": ["dataset", "scale_block_id"],
        "entry_count": 8,
        "entries": visualization_entries,
    }
    visualization_manifest["manifest_content_sha256"] = canonical_json_sha256(visualization_manifest)
    _atomic_json(root / "visualization_manifest.json", visualization_manifest)
    environment = {
        "schema": "pathline_template_matching.negative_tail_visualization_environment.v1",
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

    artifacts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"result_manifest.json", "RUN_COMPLETE.json"}:
            artifacts.append(
                {
                    "relative_path": _relative(path, root),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    _verify_artifact_rows(root, artifacts)
    result_manifest = {
        "schema": "pathline_template_matching.negative_tail_visualization_result.v1",
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
    _read_self_hashed_json(root / "input_manifest.json", "manifest_content_sha256")
    _read_self_hashed_json(root / "visualization_manifest.json", "manifest_content_sha256")
    _read_self_hashed_json(root / "result_manifest.json", "manifest_content_sha256")
    _verify_artifact_rows(root, artifacts)
    for row in input_rows:
        _stable_file(Path(str(row["path"])), str(row["sha256"]), role=f"final_input_recheck:{row['role']}")

    complete = {
        "schema": "pathline_template_matching.negative_tail_visualization_run_complete.v1",
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
    "PREDICTION_ARRAY_NAMES",
    "PredictionGroup",
    "VisualizationPlan",
    "authenticate_inputs",
    "compare_metrics_to_parent",
    "exact_bind_prediction_group",
    "load_prediction_groups",
    "load_visualization_plan",
    "read_parent_group_metrics",
    "run_negative_tail_visualization",
]

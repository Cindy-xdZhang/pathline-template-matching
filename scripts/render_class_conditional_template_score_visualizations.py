#!/usr/bin/env python3
"""Render authenticated Class-Conditional Template Score triptychs.

This report is intentionally configuration-driven.  No production identities
are embedded here: a separately frozen report config must bind the completed
ClassConditional aggregate, its numerical commit, and the immutable Phase 3.1
parent scenes.  The aggregate completion proves that all five folds already
passed the authoritative fresh replay.  This downstream report authenticates
that release chain and writes its input manifest before opening any NPZ member.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for _search_path in (SOURCE_ROOT, REPOSITORY_ROOT):
    if str(_search_path) not in sys.path:
        sys.path.insert(0, str(_search_path))

from pathline_template_matching.negative_tail_visualization import (  # noqa: E402
    SCENE31_ARRAY_NAMES,
    PredictionGroup,
    _child_scene,
    exact_bind_prediction_group,
)
from pathline_template_matching.phase21_pipeline import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _metric_values,
)
from pathline_template_matching.phase21_visualization import (  # noqa: E402
    load_phase21_scene_artifact,
    render_phase21_scene_artifact,
    write_phase21_scene_artifact,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)


REPORT_EXPERIMENT = "Other_ClassConditionalTemplateScoreVisualization_1.1"
REPORT_CONFIG_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_visualization_config.v1"
)
PREDICTION_EXPERIMENT = "Verify_ClassConditionalTemplateScore_1.1"
PREDICTION_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_outer_prediction.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_outer_prediction_manifest.v1"
)
PREDICTION_RESULT_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_result.v1"
)
PREDICTION_COMPLETE_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_run_complete.v1"
)
PREDICTION_SELECTED_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_selected_candidate.v1"
)
AGGREGATE_MANIFEST_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_aggregate_manifest.v1"
)
AGGREGATE_COMPLETE_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_aggregate_complete.v1"
)
AGGREGATE_SUMMARY_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_aggregate_summary.v1"
)
AGGREGATE_FRESH_REPLAY_SOURCE = (
    "fresh_shared_scaler_family_class_template_LOO_bundle_and_"
    "prediction_support_replay_before_outer_label_gate_then_"
    "exact_metric_recomputation"
)
METHOD_BINDING_KEY = "class_conditional_template_score_method"

FAMILY_ORDER = (
    "half_cylinder",
    "delta_wing",
    "f22_raptor",
    "channel",
    "boeing_747",
)
TARGET_DATASETS = (
    "cylinder3d",
    "halfcylinderRe640",
    "halfcylinderRe6400",
    "boeing747",
)
DATASET_TO_FAMILY = {
    "cylinder3d": "half_cylinder",
    "halfcylinderRe640": "half_cylinder",
    "halfcylinderRe6400": "half_cylinder",
    "boeing747": "boeing_747",
}
DISPLAY_NAMES = {
    "cylinder3d": "Cylinder3D Re160",
    "halfcylinderRe640": "Cylinder3D Re640",
    "halfcylinderRe6400": "Cylinder3D Re6400",
    "boeing747": "Boeing 747",
}
BLOCKS = ("legacy_2_1", "expanded_3_1")
BLOCK_INDEX = {"legacy_2_1": 0, "expanded_3_1": 1}
BLOCK_SCALE_RANGE = {"legacy_2_1": (0, 999), "expanded_3_1": (1000, 1999)}
FIXED_SOURCE_ORDINAL = 2
ASSIGNED_PER_BLOCK = 64_000
PANEL_TITLES = (
    "IVD p95 + center pathlines",
    "FMT class-conditional template-score classification",
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
    "scaler_mode",
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
    "scaler_mode": "|i1",
    "prediction": "|b1",
}
FOLD_FILE_NAMES = (
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
FOLD_ARTIFACT_NAMES = tuple(
    name for name in FOLD_FILE_NAMES if name not in {"result_manifest.json", "RUN_COMPLETE.json"}
)
AGGREGATE_FILE_NAMES = (
    "outer_family_summary.csv",
    "aggregate_summary.json",
    "aggregate_manifest.json",
    "AGGREGATE_COMPLETE.json",
)
METRIC_FIELDS = (
    "outer_family", "inner_family", "dataset", "source_ordinal", "block",
    "candidate_id", "representation", "k", "sigma", "decision_rule", "decision_value",
    "sample_count", "positive_count", "negative_count", "true_positive", "false_positive",
    "true_negative", "false_negative", "accuracy", "precision", "recall", "f1",
    "balanced_accuracy", "average_precision", "auroc", "retrieval_supported_count",
    "calibration_supported_count", "imputed_count", "unimputable_count",
    "retrieval_support_fraction", "calibration_support_fraction", "spatial_imputed_fraction",
    "spatial_unimputable_fraction", "retrieval_supported_subset_f1",
    "calibration_supported_subset_f1", "imputed_subset_f1", "unimputable_subset_f1",
    "calibration_mode_0_count", "calibration_mode_1_count", "calibration_mode_2_count",
    "calibration_mode_3_count", "calibration_mode_4_count", "calibration_mode_5_count",
    "scaler_mode_0_count", "scaler_mode_1_count", "scaler_mode_2_count",
    "scaler_mode_3_count",
)
METRIC_INTEGER_FIELDS = {
    "source_ordinal", "k", "sample_count", "positive_count", "negative_count",
    "true_positive", "false_positive", "true_negative", "false_negative",
    "retrieval_supported_count", "calibration_supported_count", "imputed_count",
    "unimputable_count", *(f"calibration_mode_{index}_count" for index in range(6)),
    *(f"scaler_mode_{index}_count" for index in range(4)),
}
METRIC_STRING_FIELDS = {
    "outer_family", "inner_family", "dataset", "block", "candidate_id",
    "representation", "decision_rule",
}
AGGREGATE_COMPLETE_FIELDS = {
    "schema", "experiment", "status", "mode", "config_sha256",
    "direct_parent_config_sha256", "direct_parent_runner_sha256",
    "direct_parent_aggregator_sha256", "core_sha256", METHOD_BINDING_KEY,
    "aggregator_git_commit", "aggregator_worktree_clean", "fold_numerical_git_commit",
    "aggregate_manifest_file", "aggregate_manifest_file_sha256", "report_file",
    "report_file_sha256", "early_stop_certificate", "completed_utc", "content_sha256",
}
AGGREGATE_MANIFEST_FIELDS = {
    "schema", "experiment", "status", "mode", "config_sha256",
    "direct_parent_config_sha256", "direct_parent_runner_sha256",
    "direct_parent_aggregator_sha256", "core_sha256", METHOD_BINDING_KEY,
    "aggregator_git_commit", "aggregator_worktree_clean", "fold_numerical_git_commit",
    "outer_family_summary_file", "outer_family_summary_file_sha256", "report_file",
    "report_file_sha256", "early_stop_certificate", "source_folds", "content_sha256",
}
SOURCE_FOLD_FIELDS = {
    "outer_family", "run_directory", "completion_file_sha256",
    "result_manifest_file_sha256", "artifact_count", "artifacts",
}
PLACEHOLDER_TOKENS = ("TODO", "PLACEHOLDER", "REPLACE_ME", "PENDING_IDENTITY")


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset: str
    display_name: str
    outer_family: str


@dataclass(frozen=True, slots=True)
class BlockSpec:
    block: str
    block_index: int
    scale_start: int
    scale_stop: int


@dataclass(frozen=True, slots=True)
class ReportPlan:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    prediction_commit: str
    prediction_config_sha256: str
    aggregate_root: Path
    aggregate_complete_sha256: str
    parent_root: Path
    parent_experiment: str
    parent_commit: str
    parent_config_sha256: str
    parent_result_manifest_sha256: str
    parent_run_complete_sha256: str
    datasets: tuple[DatasetSpec, ...]
    blocks: tuple[BlockSpec, ...]
    source_ordinal: int
    assigned_per_block: int
    dpi: int


@dataclass(frozen=True, slots=True)
class FoldEvidence:
    outer_family: str
    root: Path
    result: Mapping[str, Any]
    completion: Mapping[str, Any]
    selected_candidate: Mapping[str, Any]
    prediction_manifest: Mapping[str, Any]
    artifacts: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class AggregateEvidence:
    manifest: Mapping[str, Any]
    completion: Mapping[str, Any]
    folds: Mapping[str, FoldEvidence]
    files: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class LoadedPredictionGroup:
    group: PredictionGroup
    retrieval_supported: np.ndarray
    calibration_supported: np.ndarray
    spatial_imputed: np.ndarray
    spatial_unimputable: np.ndarray
    calibration_mode: np.ndarray
    scaler_mode: np.ndarray
    group_audit: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_lower_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_json_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_strict_json_equal(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if expected is None:
        return actual is None
    return type(actual) is type(expected) and actual == expected


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON must contain an object: {path}")
    return value


def _read_self_hashed_json(path: Path, field: str = "content_sha256") -> dict[str, Any]:
    value = _read_json(path)
    claimed = value.get(field)
    _require(_is_lower_hex(claimed, 64), f"missing or invalid {field}: {path}")
    payload = {key: item for key, item in value.items() if key != field}
    _require(canonical_json_sha256(payload) == claimed, f"content hash mismatch: {path}")
    return value


def _file_row(path: Path, role: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing input file: {path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _require_file_identity(path: Path, record: Mapping[str, Any], *, role: str) -> None:
    _require(set(record) == {"size_bytes", "sha256"}, f"{role} identity fields drifted")
    _require(
        _is_strict_int(record.get("size_bytes"))
        and record["size_bytes"] >= 0
        and _is_lower_hex(record.get("sha256"), 64),
        f"{role} identity types drifted",
    )
    _require(path.is_file(), f"missing {role}: {path}")
    _require(path.stat().st_size == record["size_bytes"], f"{role} size mismatch")
    _require(sha256_file(path) == record["sha256"], f"{role} SHA-256 mismatch")


def _require_no_placeholders(value: object, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_no_placeholders(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_no_placeholders(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        upper = value.upper()
        _require(
            not any(token in upper for token in PLACEHOLDER_TOKENS),
            f"placeholder identity is forbidden: {path}",
        )


def _environment_record(device: str) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "requested_device": device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def load_report_plan(config_path: str | Path, expected_sha256: str) -> ReportPlan:
    """Load a separately frozen production report config.

    Tests may create an ephemeral synthetic config, but the repository must not
    contain a production-looking config with placeholder job IDs or hashes.
    """

    path = Path(config_path).resolve()
    _require(path.is_file(), f"missing report config: {path}")
    _require(_is_lower_hex(expected_sha256, 64), "report config SHA-256 is invalid")
    _require(sha256_file(path) == expected_sha256, "report config SHA-256 mismatch")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), "report config must contain a mapping")
    _require_no_placeholders(value)
    _require(
        set(value)
        == {"schema", "experiment", "status", "prediction_parent", "parent_scenes", "query", "figure_contract", "output_contract"},
        "report config top-level fields drifted",
    )
    _require(value.get("schema") == REPORT_CONFIG_SCHEMA, "report config schema changed")
    _require(value.get("experiment") == REPORT_EXPERIMENT, "report experiment changed")
    _require(value.get("status") == "frozen_pre_run_not_run", "report freeze status changed")

    prediction = _mapping(value.get("prediction_parent"), "prediction parent")
    _require(
        set(prediction)
        == {"experiment", "numerical_git_commit", "config_sha256", "aggregate_root", "aggregate_complete_sha256", "aggregate_mode", "aggregate_fresh_replay_required"},
        "prediction parent fields drifted",
    )
    _require(prediction.get("experiment") == PREDICTION_EXPERIMENT, "prediction experiment changed")
    _require(_is_lower_hex(prediction.get("numerical_git_commit"), 40), "prediction commit is invalid")
    _require(_is_lower_hex(prediction.get("config_sha256"), 64), "prediction config SHA is invalid")
    _require(_is_lower_hex(prediction.get("aggregate_complete_sha256"), 64), "aggregate completion SHA is invalid")
    _require(prediction.get("aggregate_mode") == "complete_five_fold_aggregate", "visualization requires the authenticated five-fold aggregate")
    _require(
        prediction.get("aggregate_fresh_replay_required") is True,
        "the five-fold aggregate must come from authoritative fresh replay",
    )
    aggregate_root_value = Path(str(prediction.get("aggregate_root")))
    _require(aggregate_root_value.is_absolute(), "aggregate root must be absolute")
    aggregate_root = aggregate_root_value.resolve()

    parent = _mapping(value.get("parent_scenes"), "parent scenes")
    _require(
        set(parent)
        == {"experiment", "numerical_git_commit", "config_sha256", "root", "result_manifest_sha256", "run_complete_sha256"},
        "parent scene fields drifted",
    )
    _require(isinstance(parent.get("experiment"), str) and parent["experiment"], "parent experiment is invalid")
    for name, length in (("numerical_git_commit", 40), ("config_sha256", 64), ("result_manifest_sha256", 64), ("run_complete_sha256", 64)):
        _require(_is_lower_hex(parent.get(name), length), f"parent scene {name} is invalid")
    parent_root_value = Path(str(parent.get("root")))
    _require(parent_root_value.is_absolute(), "parent scene root must be absolute")
    parent_root = parent_root_value.resolve()

    query = _mapping(value.get("query"), "query")
    _require(
        set(query) == {"source_ordinal", "assigned_per_block", "datasets", "scale_blocks"},
        "query fields drifted",
    )
    _require(_is_strict_int(query.get("source_ordinal")) and query["source_ordinal"] == FIXED_SOURCE_ORDINAL, "source ordinal changed")
    _require(_is_strict_int(query.get("assigned_per_block")) and query["assigned_per_block"] == ASSIGNED_PER_BLOCK, "assigned population changed")
    dataset_rows = query.get("datasets")
    _require(isinstance(dataset_rows, list) and len(dataset_rows) == 4, "exactly four datasets are required")
    datasets: list[DatasetSpec] = []
    for expected, row in zip(TARGET_DATASETS, dataset_rows, strict=True):
        row = _mapping(row, f"dataset {expected}")
        _require(set(row) == {"id", "display_name", "outer_family"}, f"dataset fields changed: {expected}")
        _require(row.get("id") == expected, "dataset order changed")
        _require(row.get("display_name") == DISPLAY_NAMES[expected], f"display name changed: {expected}")
        _require(row.get("outer_family") == DATASET_TO_FAMILY[expected], f"dataset family changed: {expected}")
        datasets.append(DatasetSpec(expected, DISPLAY_NAMES[expected], DATASET_TO_FAMILY[expected]))
    block_rows = query.get("scale_blocks")
    _require(isinstance(block_rows, list) and len(block_rows) == 2, "exactly two scale blocks are required")
    blocks: list[BlockSpec] = []
    for expected, row in zip(BLOCKS, block_rows, strict=True):
        row = _mapping(row, f"block {expected}")
        _require(set(row) == {"id", "index", "scale_start", "scale_stop"}, f"block fields changed: {expected}")
        start, stop = BLOCK_SCALE_RANGE[expected]
        _require(
            row.get("id") == expected
            and row.get("index") == BLOCK_INDEX[expected]
            and row.get("scale_start") == start
            and row.get("scale_stop") == stop,
            f"scale block identity changed: {expected}",
        )
        blocks.append(BlockSpec(expected, BLOCK_INDEX[expected], start, stop))

    figure = _mapping(value.get("figure_contract"), "figure contract")
    _require(
        set(figure) == {"expected_figure_count", "backend", "dpi", "panel_titles"},
        "figure contract fields drifted",
    )
    _require(figure.get("expected_figure_count") == 8, "figure count changed")
    _require(figure.get("backend") == "Python/matplotlib", "plotting backend changed")
    _require(_is_strict_int(figure.get("dpi")) and figure["dpi"] == 360, "figure DPI changed")
    _require(tuple(figure.get("panel_titles", ())) == PANEL_TITLES, "panel titles changed")
    output = _mapping(value.get("output_contract"), "output contract")
    _require(
        set(output) == {"overwrite", "required_global_files", "required_exports_per_figure"},
        "output contract fields drifted",
    )
    _require(output.get("overwrite") is False, "output overwrite must be forbidden")
    _require(
        tuple(output.get("required_global_files", ()))
        == ("frozen_config.yaml", "input_manifest.json", "figure_contract.json", "per_figure_metrics.csv", "visualization_manifest.json", "result_manifest.json", "RUN_COMPLETE.json"),
        "global output contract changed",
    )
    _require(
        tuple(output.get("required_exports_per_figure", ()))
        == ("scene_npz", "scene_manifest", "png", "pdf", "svg", "alignment", "render_metadata"),
        "per-figure output contract changed",
    )
    return ReportPlan(
        path=path,
        sha256=expected_sha256,
        raw=value,
        prediction_commit=str(prediction["numerical_git_commit"]),
        prediction_config_sha256=str(prediction["config_sha256"]),
        aggregate_root=aggregate_root,
        aggregate_complete_sha256=str(prediction["aggregate_complete_sha256"]),
        parent_root=parent_root,
        parent_experiment=str(parent["experiment"]),
        parent_commit=str(parent["numerical_git_commit"]),
        parent_config_sha256=str(parent["config_sha256"]),
        parent_result_manifest_sha256=str(parent["result_manifest_sha256"]),
        parent_run_complete_sha256=str(parent["run_complete_sha256"]),
        datasets=tuple(datasets),
        blocks=tuple(blocks),
        source_ordinal=FIXED_SOURCE_ORDINAL,
        assigned_per_block=ASSIGNED_PER_BLOCK,
        dpi=360,
    )


def _authenticate_method_binding(
    payload: Mapping[str, Any],
    plan: ReportPlan,
    *,
    label: str,
) -> Mapping[str, Any]:
    binding = _mapping(payload.get(METHOD_BINDING_KEY), f"{label} method binding")
    _require(binding.get("experiment") == PREDICTION_EXPERIMENT, f"{label} method experiment changed")
    config = _mapping(binding.get("config"), f"{label} method config")
    _require(config.get("sha256") == plan.prediction_config_sha256, f"{label} method config changed")
    score = _mapping(binding.get("score"), f"{label} score binding")
    threshold = _mapping(binding.get("threshold"), f"{label} threshold binding")
    _require(score.get("combine") == "equal_mean_over_jointly_supported_families", f"{label} score combine changed")
    _require(score.get("inner_support") == "2_of_3_joint_families", f"{label} inner support changed")
    _require(score.get("outer_support") == "3_of_4_joint_families", f"{label} outer support changed")
    _require(threshold.get("comparison") == "strict_greater_than", f"{label} threshold comparator changed")
    _require(threshold.get("equality_prediction") == "negative", f"{label} threshold tie policy changed")
    _require(binding.get("prediction_array_contract") == "unchanged_parent_19_arrays", f"{label} prediction contract changed")
    _require(binding.get("fold_transaction") == "unchanged_parent_15_files", f"{label} fold transaction changed")
    _require(binding.get("numerical_git_commit") == plan.prediction_commit, f"{label} numerical commit changed")
    return binding


def _candidate_identity(value: object, *, label: str) -> dict[str, Any]:
    candidate = _mapping(value, label)
    expected = {"candidate_id", "representation", "k", "sigma", "decision_rule", "decision_value"}
    _require(set(candidate) == expected, f"{label} fields drifted")
    _require(isinstance(candidate.get("candidate_id"), str) and candidate["candidate_id"], f"{label} ID is invalid")
    _require(isinstance(candidate.get("representation"), str) and candidate["representation"], f"{label} representation is invalid")
    _require(_is_strict_int(candidate.get("k")) and candidate["k"] in {1, 5, 15, 31}, f"{label} k is invalid")
    _require(isinstance(candidate.get("sigma"), (int, float)) and not isinstance(candidate.get("sigma"), bool), f"{label} sigma is invalid")
    _require(candidate["sigma"] in {0.0, 0.5, 1.0, 1.5, 2.0}, f"{label} sigma changed")
    decision_rule = candidate.get("decision_rule")
    decision_value = candidate.get("decision_value")
    _require(decision_rule in {"fixed_top_fraction", "calibrated_tail_anomaly_threshold"}, f"{label} decision rule is invalid")
    _require(isinstance(decision_value, (int, float)) and not isinstance(decision_value, bool), f"{label} decision value is invalid")
    if decision_rule == "fixed_top_fraction":
        _require(float(decision_value) == 0.05, f"{label} top fraction changed")
        _require("fixed_top_fraction=0.05" in candidate["candidate_id"], f"{label} ID omits top fraction")
    else:
        _require(0.50 <= float(decision_value) <= 0.99, f"{label} threshold is outside the frozen grid")
        _require("comparator=strict_greater_than" in candidate["candidate_id"], f"{label} ID omits strict comparator")
    return dict(candidate)


def _authenticate_fold_chain(
    plan: ReportPlan,
    source: Mapping[str, Any],
    aggregate_binding: Mapping[str, Any],
) -> tuple[FoldEvidence, list[dict[str, Any]]]:
    """Authenticate the aggregate-to-fold 15-file chain without opening NPZ members."""

    _require(set(source) == SOURCE_FOLD_FIELDS, "aggregate source-fold fields drifted")
    family = source.get("outer_family")
    _require(isinstance(family, str) and family in FAMILY_ORDER, "aggregate source fold is unknown")
    root_value = source.get("run_directory")
    _require(isinstance(root_value, str) and root_value, f"{family}: fold root is invalid")
    root_path = Path(root_value)
    _require(root_path.is_absolute(), f"{family}: fold root must be absolute")
    root = root_path.resolve()
    _require(root.is_dir(), f"{family}: fold root is missing")
    _require(
        {path.name for path in root.iterdir()} == set(FOLD_FILE_NAMES)
        and all((root / name).is_file() for name in FOLD_FILE_NAMES),
        f"{family}: completed fold must contain exactly 15 files",
    )
    _require(_is_lower_hex(source.get("completion_file_sha256"), 64), f"{family}: completion SHA is invalid")
    _require(_is_lower_hex(source.get("result_manifest_file_sha256"), 64), f"{family}: result SHA is invalid")
    _require(source.get("artifact_count") == 13, f"{family}: artifact count changed")
    aggregate_artifacts = _mapping(source.get("artifacts"), f"{family}: aggregate artifacts")
    _require(set(aggregate_artifacts) == set(FOLD_ARTIFACT_NAMES), f"{family}: aggregate artifact set changed")

    completion_path = root / "RUN_COMPLETE.json"
    result_path = root / "result_manifest.json"
    _require(sha256_file(completion_path) == source["completion_file_sha256"], f"{family}: completion file changed")
    _require(sha256_file(result_path) == source["result_manifest_file_sha256"], f"{family}: result file changed")
    completion = _read_self_hashed_json(completion_path)
    result = _read_self_hashed_json(result_path)
    for payload, schema, label in (
        (completion, PREDICTION_COMPLETE_SCHEMA, "completion"),
        (result, PREDICTION_RESULT_SCHEMA, "result"),
    ):
        _require(payload.get("schema") == schema, f"{family}: {label} schema changed")
        _require(payload.get("experiment") == PREDICTION_EXPERIMENT, f"{family}: {label} experiment changed")
        _require(payload.get("git_commit") == plan.prediction_commit, f"{family}: {label} commit changed")
        _require(payload.get("config_sha256") == plan.prediction_config_sha256, f"{family}: {label} config changed")
        _require(payload.get("outer_family") == family, f"{family}: {label} outer family changed")
        binding = _authenticate_method_binding(payload, plan, label=f"{family} {label}")
        _require(_strict_json_equal(binding, aggregate_binding), f"{family}: {label} method differs from aggregate")
    _require(result.get("status") == "completed", f"{family}: result is not completed")
    _require(completion.get("result_manifest_file") == "result_manifest.json", f"{family}: result path changed")
    _require(completion.get("result_manifest_file_sha256") == sha256_file(result_path), f"{family}: completion does not bind result file")
    _require(completion.get("result_manifest_content_sha256") == result.get("content_sha256"), f"{family}: completion does not bind result content")

    result_artifacts = _mapping(result.get("artifacts"), f"{family}: result artifacts")
    _require(set(result_artifacts) == set(FOLD_ARTIFACT_NAMES), f"{family}: result artifact set changed")
    evidence = [
        _file_row(result_path, f"prediction_fold:{family}:result_manifest"),
        _file_row(completion_path, f"prediction_fold:{family}:RUN_COMPLETE"),
    ]
    normalized_artifacts: dict[str, Mapping[str, Any]] = {}
    for name in FOLD_ARTIFACT_NAMES:
        aggregate_record = _mapping(aggregate_artifacts[name], f"{family}/{name} aggregate identity")
        result_record = _mapping(result_artifacts[name], f"{family}/{name} result identity")
        _require(_strict_json_equal(result_record, aggregate_record), f"{family}/{name}: aggregate/result identity differs")
        _require_file_identity(root / name, aggregate_record, role=f"{family}/{name}")
        normalized_artifacts[name] = dict(aggregate_record)
        evidence.append(_file_row(root / name, f"prediction_fold:{family}:{name}"))

    selected_path = root / "selected_candidate.json"
    prediction_manifest_path = root / "outer_prediction_manifest.json"
    selected = _read_self_hashed_json(selected_path)
    prediction_manifest = _read_self_hashed_json(prediction_manifest_path)
    _require(selected.get("schema") == PREDICTION_SELECTED_SCHEMA, f"{family}: selected schema changed")
    _require(prediction_manifest.get("schema") == PREDICTION_MANIFEST_SCHEMA, f"{family}: prediction manifest schema changed")
    _require(prediction_manifest.get("prediction_schema") == PREDICTION_SCHEMA, f"{family}: prediction schema changed")
    for payload, label in ((selected, "selected candidate"), (prediction_manifest, "prediction manifest")):
        _require(payload.get("experiment") == PREDICTION_EXPERIMENT, f"{family}: {label} experiment changed")
        _require(payload.get("git_commit") == plan.prediction_commit, f"{family}: {label} commit changed")
        _require(payload.get("config_sha256") == plan.prediction_config_sha256, f"{family}: {label} config changed")
        _require(payload.get("outer_family") == family, f"{family}: {label} fold changed")
        binding = _authenticate_method_binding(payload, plan, label=f"{family} {label}")
        _require(_strict_json_equal(binding, aggregate_binding), f"{family}: {label} method differs from aggregate")
    candidate = _candidate_identity(selected.get("candidate"), label=f"{family} selected candidate")
    _require(_strict_json_equal(prediction_manifest.get("selected_candidate"), candidate), f"{family}: prediction candidate differs")
    _require(_strict_json_equal(result.get("selected_candidate"), candidate), f"{family}: result candidate differs")
    _require(prediction_manifest.get("valid_labels_opened") is False, f"{family}: prediction opened labels")
    _require(prediction_manifest.get("metadata_json_opened") is False, f"{family}: prediction opened metadata")
    _require(prediction_manifest.get("array_count") == 19, f"{family}: prediction array count changed")
    arrays = _mapping(prediction_manifest.get("arrays"), f"{family}: prediction arrays")
    _require(set(arrays) == set(PREDICTION_ARRAY_NAMES), f"{family}: prediction array set changed")
    prediction_file = _mapping(prediction_manifest.get("prediction_file"), f"{family}: prediction file")
    _require(set(prediction_file) == {"path", "size_bytes", "sha256"}, f"{family}: prediction file fields drifted")
    _require(prediction_file.get("path") == "outer_predictions.npz", f"{family}: prediction path changed")
    _require_file_identity(
        root / "outer_predictions.npz",
        {"size_bytes": prediction_file.get("size_bytes"), "sha256": prediction_file.get("sha256")},
        role=f"{family}/outer_predictions.npz",
    )
    return (
        FoldEvidence(
            outer_family=family,
            root=root,
            result=result,
            completion=completion,
            selected_candidate=candidate,
            prediction_manifest=prediction_manifest,
            artifacts=normalized_artifacts,
        ),
        evidence,
    )


def authenticate_aggregate_chain(plan: ReportPlan) -> AggregateEvidence:
    """Authenticate aggregate completion -> manifest -> all five 15-file folds.

    This phase hashes NPZ files as opaque bytes but deliberately does not call
    ``numpy.load``.  Full numerical fold replay happens only after the report
    input manifest has been persisted.
    """

    root = plan.aggregate_root
    _require(root.is_dir(), f"aggregate root is missing: {root}")
    _require(
        {path.name for path in root.iterdir()} == set(AGGREGATE_FILE_NAMES)
        and all((root / name).is_file() for name in AGGREGATE_FILE_NAMES),
        "complete aggregate must contain exactly four files",
    )
    completion_path = root / "AGGREGATE_COMPLETE.json"
    _require(sha256_file(completion_path) == plan.aggregate_complete_sha256, "aggregate completion SHA-256 mismatch")
    completion = _read_self_hashed_json(completion_path)
    _require(set(completion) == AGGREGATE_COMPLETE_FIELDS, "aggregate completion fields drifted")
    _require(completion.get("schema") == AGGREGATE_COMPLETE_SCHEMA, "aggregate completion schema changed")
    _require(completion.get("experiment") == PREDICTION_EXPERIMENT, "aggregate experiment changed")
    _require(completion.get("status") == "completed", "aggregate is not complete")
    _require(completion.get("mode") == "complete_five_fold_aggregate", "aggregate is not complete-five-fold")
    _require(completion.get("config_sha256") == plan.prediction_config_sha256, "aggregate config changed")
    _require(completion.get("aggregator_git_commit") == plan.prediction_commit, "aggregate commit changed")
    _require(completion.get("fold_numerical_git_commit") == plan.prediction_commit, "fold commit changed")
    _require(completion.get("aggregator_worktree_clean") is True, "aggregate worktree was not clean")
    _require(completion.get("early_stop_certificate") is None, "complete aggregate must not carry an early-stop certificate")
    aggregate_binding = _authenticate_method_binding(completion, plan, label="aggregate completion")

    manifest_path = root / "aggregate_manifest.json"
    report_path = root / "aggregate_summary.json"
    table_path = root / "outer_family_summary.csv"
    _require(completion.get("aggregate_manifest_file") == manifest_path.name, "aggregate manifest path changed")
    _require(completion.get("aggregate_manifest_file_sha256") == sha256_file(manifest_path), "aggregate manifest file binding changed")
    _require(completion.get("report_file") == report_path.name, "aggregate report path changed")
    _require(completion.get("report_file_sha256") == sha256_file(report_path), "aggregate report file binding changed")
    manifest = _read_self_hashed_json(manifest_path)
    report = _read_self_hashed_json(report_path)
    _require(set(manifest) == AGGREGATE_MANIFEST_FIELDS, "aggregate manifest fields drifted")
    for payload, schema, label in (
        (manifest, AGGREGATE_MANIFEST_SCHEMA, "aggregate manifest"),
        (report, AGGREGATE_SUMMARY_SCHEMA, "aggregate report"),
    ):
        _require(payload.get("schema") == schema, f"{label} schema changed")
        _require(payload.get("experiment") == PREDICTION_EXPERIMENT, f"{label} experiment changed")
        _require(payload.get("status") == "completed", f"{label} is not complete")
        _require(payload.get("mode") == "complete_five_fold_aggregate", f"{label} mode changed")
        _require(payload.get("config_sha256") == plan.prediction_config_sha256, f"{label} config changed")
        _require(payload.get("aggregator_git_commit") == plan.prediction_commit, f"{label} commit changed")
        _require(payload.get("fold_numerical_git_commit") == plan.prediction_commit, f"{label} fold commit changed")
        binding = _authenticate_method_binding(payload, plan, label=label)
        _require(_strict_json_equal(binding, aggregate_binding), f"{label} method binding differs")
    _require(manifest.get("outer_family_summary_file") == table_path.name, "aggregate table path changed")
    _require(manifest.get("outer_family_summary_file_sha256") == sha256_file(table_path), "aggregate table binding changed")
    _require(manifest.get("report_file") == report_path.name, "aggregate report path differs in manifest")
    _require(manifest.get("report_file_sha256") == sha256_file(report_path), "aggregate report SHA differs in manifest")
    _require(manifest.get("early_stop_certificate") is None, "aggregate manifest carries an early-stop certificate")
    _require(report.get("outer_families") == list(FAMILY_ORDER), "aggregate family order changed")
    _require(report.get("outer_family_count") == 5, "aggregate family count changed")
    _require(report.get("formal_confirmation") is False, "aggregate made a formal-confirmation claim")
    _require(
        report.get("fold_summary_source") == AGGREGATE_FRESH_REPLAY_SOURCE,
        "aggregate did not bind authoritative fresh fold replay",
    )
    _require(
        report.get("evidence_scope")
        == "exposed_train_only_nested_family_validation",
        "aggregate evidence scope changed",
    )
    _require(report.get("outer_family_summary_file_sha256") == sha256_file(table_path), "aggregate report table binding changed")

    source_folds = manifest.get("source_folds")
    _require(isinstance(source_folds, list) and len(source_folds) == 5, "aggregate must bind five source folds")
    _require([row.get("outer_family") for row in source_folds if isinstance(row, Mapping)] == list(FAMILY_ORDER), "aggregate source-fold order changed")
    files: list[dict[str, Any]] = [
        _file_row(completion_path, "class_aggregate:AGGREGATE_COMPLETE"),
        _file_row(manifest_path, "class_aggregate:aggregate_manifest"),
        _file_row(report_path, "class_aggregate:aggregate_summary"),
        _file_row(table_path, "class_aggregate:outer_family_summary"),
    ]
    folds: dict[str, FoldEvidence] = {}
    for source in source_folds:
        source = _mapping(source, "aggregate source fold")
        fold, fold_files = _authenticate_fold_chain(plan, source, aggregate_binding)
        _require(fold.outer_family not in folds, f"duplicate aggregate fold: {fold.outer_family}")
        folds[fold.outer_family] = fold
        files.extend(fold_files)
    _require(tuple(folds) == FAMILY_ORDER, "aggregate fold population changed")
    _require(len({row["path"] for row in files}) == len(files), "aggregate evidence contains duplicate paths")
    return AggregateEvidence(
        manifest=manifest,
        completion=completion,
        folds=folds,
        files=tuple(files),
    )


def authenticate_parent_scene_chain(
    plan: ReportPlan,
) -> tuple[dict[tuple[str, str], dict[str, Path]], tuple[Mapping[str, Any], ...]]:
    """Authenticate the canonical source-2 parent scenes without opening NPZ members."""

    root = plan.parent_root
    result_path = root / "result_manifest.json"
    complete_path = root / "RUN_COMPLETE.json"
    _require(result_path.is_file() and complete_path.is_file(), "parent scene result chain is missing")
    _require(sha256_file(result_path) == plan.parent_result_manifest_sha256, "parent result manifest SHA changed")
    _require(sha256_file(complete_path) == plan.parent_run_complete_sha256, "parent completion SHA changed")
    result = _read_self_hashed_json(result_path, "manifest_content_sha256")
    complete = _read_json(complete_path)
    for payload, label in ((result, "parent result"), (complete, "parent completion")):
        _require(payload.get("experiment") == plan.parent_experiment, f"{label} experiment changed")
        _require(payload.get("git_commit") == plan.parent_commit, f"{label} commit changed")
    # The canonical Phase 3.1 RUN_COMPLETE schema binds the result bytes and
    # content digest but does not repeat config_sha256.  Authenticate that
    # identity on the self-hashed result manifest, then authenticate both
    # completion-to-result links below.
    _require(result.get("config_sha256") == plan.parent_config_sha256, "parent result config changed")
    _require(complete.get("result_manifest_file_sha256") == sha256_file(result_path), "parent completion does not bind result file")
    _require(complete.get("result_manifest_content_sha256") == result.get("manifest_content_sha256"), "parent completion does not bind result content")
    rows = result.get("artifacts")
    _require(isinstance(rows, list), "parent result artifact list is missing")
    by_relative = {
        str(row["relative_path"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("relative_path"), str)
    }
    scenes: dict[tuple[str, str], dict[str, Path]] = {}
    evidence: list[Mapping[str, Any]] = [
        _file_row(result_path, "parent_scene:result_manifest"),
        _file_row(complete_path, "parent_scene:RUN_COMPLETE"),
    ]
    for dataset in plan.datasets:
        for block in plan.blocks:
            stem = f"{dataset.dataset}_source_ordinal_{plan.source_ordinal}_{block.block}"
            paths = {
                "npz": root / "scenes" / f"{stem}.scene.npz",
                "manifest": root / "scenes" / f"{stem}.scene.json",
                "render": root / "figures" / f"{stem}_family_heldout_triptych.render.json",
            }
            for role, path in paths.items():
                relative = path.relative_to(root).as_posix()
                row = _mapping(by_relative.get(relative), f"parent artifact {relative}")
                _require(set(row) >= {"relative_path", "size_bytes", "sha256"}, f"parent artifact fields changed: {relative}")
                _require_file_identity(
                    path,
                    {"size_bytes": row.get("size_bytes"), "sha256": row.get("sha256")},
                    role=f"parent scene {relative}",
                )
                evidence.append(_file_row(path, f"parent_scene:{dataset.dataset}:{block.block}:{role}"))
            scenes[(dataset.dataset, block.block)] = paths
    expected = {(dataset.dataset, block.block) for dataset in plan.datasets for block in plan.blocks}
    _require(set(scenes) == expected, "parent scene population is not the fixed eight figures")
    return scenes, tuple(evidence)


def _authenticate_reporting_checkout(expected_commit: str) -> dict[str, str]:
    _require(_is_lower_hex(expected_commit, 40), "expected reporting commit is invalid")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(status == "", "reporting checkout must be clean")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(commit == expected_commit, "reporting checkout differs from expected commit")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "scripts/render_class_conditional_template_score_visualizations.py"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(bool(tracked), "reporting script is not tracked by the authenticated commit")
    return {
        "reporting_git_commit": commit,
        "reporting_script_sha256": sha256_file(Path(__file__).resolve()),
    }


def _manifest_array_spec(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    arrays = _mapping(manifest.get("arrays"), "prediction manifest arrays")
    return _mapping(arrays.get(name), f"prediction array {name}")


def _support_audit_for_group(
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    source_ordinal: int,
    source_index: int,
    block: str,
    count: int,
) -> Mapping[str, Any]:
    rows = manifest.get("group_audits")
    _require(isinstance(rows, list), "prediction group audits are missing")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("dataset") == dataset
        and row.get("source_ordinal") == source_ordinal
        and row.get("source_index") == source_index
        and row.get("block") == block
    ]
    _require(len(matches) == 1, f"prediction group audit is not unique: {(dataset, source_ordinal, block)}")
    audit = matches[0]
    _require(audit.get("sample_count") == count, f"prediction group audit count changed: {(dataset, source_ordinal, block)}")
    support = _mapping(audit.get("class_conditional_support"), "class-conditional support audit")
    _require(support.get("sample_count") == count, "support audit population changed")
    family_order = support.get("family_order")
    _require(isinstance(family_order, list) and len(family_order) == 4, "support audit family order changed")
    _require(support.get("required_joint_family_count") == 3, "support audit must retain the 3-of-4 gate")
    histogram = _mapping(support.get("joint_supported_family_count_histogram"), "joint family histogram")
    _require(set(histogram) == {"0", "1", "2", "3", "4"}, "joint-family histogram bins changed")
    _require(
        all(_is_strict_int(histogram[str(index)]) and histogram[str(index)] >= 0 for index in range(5))
        and sum(histogram[str(index)] for index in range(5)) == count,
        "joint-family histogram population changed",
    )
    families = _mapping(support.get("families"), "per-family support audit")
    _require(set(families) == set(family_order), "per-family support population changed")
    return audit


def load_prediction_groups(
    plan: ReportPlan,
    aggregate: AggregateEvidence,
) -> dict[tuple[str, str], LoadedPredictionGroup]:
    """Validate all 19 arrays, then project the eight fixed source-2 groups."""

    groups: dict[tuple[str, str], LoadedPredictionGroup] = {}
    for family in ("half_cylinder", "boeing_747"):
        fold = aggregate.folds[family]
        expected_datasets = tuple(
            dataset.dataset for dataset in plan.datasets if dataset.outer_family == family
        )
        prediction_path = fold.root / "outer_predictions.npz"
        before = prediction_path.stat()
        with np.load(prediction_path, allow_pickle=False) as archive:
            _require(tuple(archive.files) == PREDICTION_ARRAY_NAMES, f"{family}: prediction NPZ order changed")
            arrays: dict[str, np.ndarray] = {}
            row_count = fold.prediction_manifest.get("row_count")
            _require(_is_strict_int(row_count) and row_count > 0, f"{family}: prediction row count is invalid")
            for name in PREDICTION_ARRAY_NAMES:
                value = np.asarray(archive[name])
                spec = _manifest_array_spec(fold.prediction_manifest, name)
                _require(set(spec) == {"dtype", "shape", "sha256"}, f"{family}/{name}: array spec fields changed")
                _require(value.dtype.str == PREDICTION_DTYPES[name], f"{family}/{name}: dtype changed")
                _require(value.shape == (row_count,), f"{family}/{name}: shape changed")
                _require(spec.get("dtype") == value.dtype.str, f"{family}/{name}: manifest dtype changed")
                _require(spec.get("shape") == [row_count], f"{family}/{name}: manifest shape changed")
                _require(canonical_array_sha256(value) == spec.get("sha256"), f"{family}/{name}: array SHA changed")
                arrays[name] = value
            dataset_values = np.asarray(arrays["dataset"])
            source_ordinals = np.asarray(arrays["source_ordinal"], dtype=np.int16)
            source_indices = np.asarray(arrays["source_index"], dtype=np.int64)
            scale_ids = np.asarray(arrays["scale_id"], dtype=np.int32)
            centers = np.asarray(arrays["center_seed_index"], dtype=np.int64)
            block_indices = np.asarray(arrays["scale_block_index"], dtype=np.int8)
            assigned = np.asarray(arrays["assigned_row_index"], dtype=np.int64)
            raw_score = np.asarray(arrays["tail_anomaly"], dtype=np.float64)
            compatibility_tail = np.asarray(arrays["tail_probability"], dtype=np.float64)
            spatial_score = np.asarray(arrays["spatial_score"], dtype=np.float64)
            denominator = np.asarray(arrays["spatial_denominator"], dtype=np.float64)
            retrieval = np.asarray(arrays["retrieval_supported"], dtype=np.bool_)
            calibration = np.asarray(arrays["calibration_supported"], dtype=np.bool_)
            imputed = np.asarray(arrays["spatial_imputed"], dtype=np.bool_)
            unimputable = np.asarray(arrays["spatial_unimputable"], dtype=np.bool_)
            calibration_mode = np.asarray(arrays["calibration_mode"], dtype=np.int8)
            scaler_mode = np.asarray(arrays["scaler_mode"], dtype=np.int8)
            prediction = np.asarray(arrays["prediction"], dtype=np.bool_)
            _require(set(np.unique(dataset_values).tolist()) == set(expected_datasets), f"{family}: cross-fold or missing datasets")
            _require(set(np.unique(source_ordinals).tolist()) == {0, 1, 2, 3}, f"{family}: source population changed")
            _require(set(np.unique(block_indices).tolist()) == {0, 1}, f"{family}: block population changed")
            _require(np.all((scale_ids >= 0) & (scale_ids < 2000)), f"{family}: scale ID outside frozen range")
            _require(np.array_equal(block_indices.astype(np.int32), scale_ids // 1000), f"{family}: scale/block identity changed")
            _require(np.all((centers >= 0) & (centers < plan.assigned_per_block)), f"{family}: center identity outside range")
            _require(np.array_equal(assigned, centers + block_indices.astype(np.int64) * plan.assigned_per_block), f"{family}: assigned-row identity changed")
            _require(np.isfinite(raw_score).all() and np.isfinite(spatial_score).all(), f"{family}: score contains NaN or Inf")
            _require(np.isfinite(denominator).all() and np.all(denominator >= 0), f"{family}: spatial denominator is invalid")
            _require(np.allclose(compatibility_tail, 1.0 - raw_score, rtol=0.0, atol=1.0e-12), f"{family}: compatibility tail field no longer equals one minus class score")
            _require(np.array_equal(calibration | imputed | unimputable, np.ones(row_count, dtype=np.bool_)), f"{family}: spatial support states do not cover rows")
            _require(not np.any((calibration & imputed) | (calibration & unimputable) | (imputed & unimputable)), f"{family}: spatial support states overlap")
            _require(not np.any(calibration & ~retrieval), f"{family}: calibration support exceeds retrieval support")

            for dataset in expected_datasets:
                for source_ordinal in range(4):
                    for block, block_index in BLOCK_INDEX.items():
                        mask = (
                            (dataset_values == dataset)
                            & (source_ordinals == source_ordinal)
                            & (block_indices == block_index)
                        )
                        count = int(mask.sum())
                        key3 = (dataset, source_ordinal, block)
                        _require(count > 0, f"missing prediction population: {key3}")
                        _require(len(np.unique(centers[mask])) == count, f"duplicate center identity: {key3}")
                        _require(len(np.unique(assigned[mask])) == count, f"duplicate assigned identity: {key3}")
                        source_values = np.unique(source_indices[mask])
                        _require(len(source_values) == 1, f"source index is not unique: {key3}")
                        audit = _support_audit_for_group(
                            fold.prediction_manifest,
                            dataset=dataset,
                            source_ordinal=source_ordinal,
                            source_index=int(source_values[0]),
                            block=block,
                            count=count,
                        )
                        _require(audit.get("retrieval_supported_count") == int(retrieval[mask].sum()), f"retrieval audit count changed: {key3}")
                        _require(audit.get("calibration_supported_count") == int(calibration[mask].sum()), f"calibration audit count changed: {key3}")
                        _require(audit.get("imputed_count") == int(imputed[mask].sum()), f"imputation audit count changed: {key3}")
                        _require(audit.get("unimputable_count") == int(unimputable[mask].sum()), f"unimputable audit count changed: {key3}")
                        _require(audit.get("prediction_count") == int(prediction[mask].sum()), f"prediction audit count changed: {key3}")
                        if source_ordinal != plan.source_ordinal:
                            continue
                        group = PredictionGroup(
                            dataset=dataset,
                            source_ordinal=source_ordinal,
                            source_index=int(source_values[0]),
                            block=block,
                            outer_family=family,
                            candidate=dict(fold.selected_candidate),
                            center_seed_index=np.array(centers[mask], dtype=np.int64, copy=True),
                            assigned_row_index=np.array(assigned[mask], dtype=np.int64, copy=True),
                            scale_id=np.array(scale_ids[mask], dtype=np.int32, copy=True),
                            scale_block_index=np.array(block_indices[mask], dtype=np.int8, copy=True),
                            spatial_score=np.array(spatial_score[mask], dtype=np.float64, copy=True),
                            prediction=np.array(prediction[mask], dtype=np.bool_, copy=True),
                        )
                        key = (dataset, block)
                        _require(key not in groups, f"duplicate fixed-source prediction group: {key}")
                        groups[key] = LoadedPredictionGroup(
                            group=group,
                            retrieval_supported=np.array(retrieval[mask], copy=True),
                            calibration_supported=np.array(calibration[mask], copy=True),
                            spatial_imputed=np.array(imputed[mask], copy=True),
                            spatial_unimputable=np.array(unimputable[mask], copy=True),
                            calibration_mode=np.array(calibration_mode[mask], copy=True),
                            scaler_mode=np.array(scaler_mode[mask], copy=True),
                            group_audit=dict(audit),
                        )
        after = prediction_path.stat()
        _require(
            (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"{family}: prediction archive changed while reading",
        )
        _require(sha256_file(prediction_path) == fold.artifacts["outer_predictions.npz"]["sha256"], f"{family}: prediction archive changed after member validation")
    expected = {(dataset.dataset, block.block) for dataset in plan.datasets for block in plan.blocks}
    _require(set(groups) == expected, "prediction population is not the fixed eight figures")
    for dataset in plan.datasets:
        source_indices = {groups[(dataset.dataset, block.block)].group.source_index for block in plan.blocks}
        _require(len(source_indices) == 1, f"legacy/expanded source index differs: {dataset.dataset}")
    return groups


def _parse_metric_value(field: str, text: str) -> Any:
    if field in METRIC_STRING_FIELDS:
        return text
    if field in METRIC_INTEGER_FIELDS:
        _require(text != "", f"metric integer is missing: {field}")
        value = int(text, 10)
        _require(str(value) == text, f"metric integer is noncanonical: {field}")
        return value
    if text == "":
        return float("nan")
    value = float(text)
    _require(np.isfinite(value), f"metric float is nonfinite: {field}")
    return value


def read_outer_group_metrics(
    plan: ReportPlan,
    aggregate: AggregateEvidence,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    selected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for family in ("half_cylinder", "boeing_747"):
        fold = aggregate.folds[family]
        path = fold.root / "outer_group_metrics.csv"
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            _require(tuple(reader.fieldnames or ()) == METRIC_FIELDS, f"{family}: outer metric fields changed")
            rows = list(reader)
        expected_datasets = tuple(dataset.dataset for dataset in plan.datasets if dataset.outer_family == family)
        expected_population = {
            (dataset, source_ordinal, block.block)
            for dataset in expected_datasets
            for source_ordinal in range(4)
            for block in plan.blocks
        }
        observed: set[tuple[str, int, str]] = set()
        for raw in rows:
            row = {field: _parse_metric_value(field, raw[field]) for field in METRIC_FIELDS}
            _require(row["outer_family"] == family and row["inner_family"] == "outer", f"{family}: metric fold identity changed")
            _require(row["dataset"] in expected_datasets, f"{family}: cross-fold metric row")
            key3 = (row["dataset"], row["source_ordinal"], row["block"])
            _require(key3 not in observed, f"{family}: duplicate metric row {key3}")
            observed.add(key3)
            _require(
                _strict_json_equal(
                    {name: row[name] for name in ("candidate_id", "representation", "k", "sigma", "decision_rule", "decision_value")},
                    fold.selected_candidate,
                ),
                f"{family}: metric candidate changed",
            )
            if row["source_ordinal"] == plan.source_ordinal:
                selected[(row["dataset"], row["block"])] = row
        _require(observed == expected_population, f"{family}: outer metric population changed")
    expected = {(dataset.dataset, block.block) for dataset in plan.datasets for block in plan.blocks}
    _require(set(selected) == expected, "fixed-source outer metrics are incomplete")
    return selected


def _subset_f1(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=np.bool_)
    if not selected.any():
        return float("nan")
    values = _metric_values(
        np.asarray(reference, dtype=np.bool_)[selected],
        np.asarray(prediction, dtype=np.bool_)[selected],
        np.asarray(prediction, dtype=np.float64)[selected],
    )
    return float(values["f1"])


def recompute_complete_metric_row(
    *,
    reference: np.ndarray,
    loaded: LoadedPredictionGroup,
) -> dict[str, Any]:
    group = loaded.group
    prediction = np.asarray(group.prediction, dtype=np.bool_)
    score = np.asarray(group.spatial_score, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.bool_)
    _require(reference.shape == prediction.shape == score.shape, "metric arrays have different shapes")
    decision_metrics = dict(_metric_values(reference, prediction, score))
    # ``single_class_group`` is an internal diagnostic returned by the shared
    # metric helper.  The frozen outer_group_metrics.csv contract deliberately
    # excludes it, so recompute it but do not publish it as a CSV field.
    decision_metrics.pop("single_class_group", None)
    row: dict[str, Any] = {
        "outer_family": group.outer_family,
        "inner_family": "outer",
        "dataset": group.dataset,
        "source_ordinal": group.source_ordinal,
        "block": group.block,
        **dict(group.candidate),
        **decision_metrics,
        "retrieval_supported_count": int(loaded.retrieval_supported.sum()),
        "calibration_supported_count": int(loaded.calibration_supported.sum()),
        "imputed_count": int(loaded.spatial_imputed.sum()),
        "unimputable_count": int(loaded.spatial_unimputable.sum()),
        "retrieval_support_fraction": float(loaded.retrieval_supported.mean()),
        "calibration_support_fraction": float(loaded.calibration_supported.mean()),
        "spatial_imputed_fraction": float(loaded.spatial_imputed.mean()),
        "spatial_unimputable_fraction": float(loaded.spatial_unimputable.mean()),
        "retrieval_supported_subset_f1": _subset_f1(reference, prediction, loaded.retrieval_supported),
        "calibration_supported_subset_f1": _subset_f1(reference, prediction, loaded.calibration_supported),
        "imputed_subset_f1": _subset_f1(reference, prediction, loaded.spatial_imputed),
        "unimputable_subset_f1": _subset_f1(reference, prediction, loaded.spatial_unimputable),
    }
    for mode in range(6):
        row[f"calibration_mode_{mode}_count"] = int(np.count_nonzero(loaded.calibration_mode == mode))
    for mode in range(4):
        row[f"scaler_mode_{mode}_count"] = int(np.count_nonzero(loaded.scaler_mode == mode))
    _require(set(row) == set(METRIC_FIELDS), "recomputed outer metric fields drifted")
    return row


def compare_metric_rows(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    _require(set(observed) == set(expected) == set(METRIC_FIELDS), "metric field population changed")
    for field in METRIC_FIELDS:
        left = observed[field]
        right = expected[field]
        if field in METRIC_STRING_FIELDS or field in METRIC_INTEGER_FIELDS:
            _require(_strict_json_equal(left, right), f"metric mismatch: {field}")
        else:
            left_value = float(left)
            right_value = float(right)
            if math.isnan(left_value) or math.isnan(right_value):
                _require(math.isnan(left_value) and math.isnan(right_value), f"metric mismatch: {field}")
            else:
                _require(abs(left_value - right_value) <= 1.0e-12, f"metric mismatch: {field}")


def _load_parent_scene(
    plan: ReportPlan,
    paths: Mapping[str, Path],
    *,
    dataset: str,
    block: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    artifact = load_phase21_scene_artifact(paths["npz"], paths["manifest"])
    metadata = artifact.metadata
    _require(metadata.get("analysis_experiment") == plan.parent_experiment, "parent analysis experiment changed")
    _require(metadata.get("experiment") == "mainExp_TemplateMatching_3.1", "parent cache experiment changed")
    _require(metadata.get("dataset") == dataset, "parent scene dataset changed")
    _require(metadata.get("scale_block_id") == block, "parent scene block changed")
    _require(metadata.get("source_ordinal") == plan.source_ordinal, "parent scene source ordinal changed")
    with np.load(paths["npz"], allow_pickle=False) as archive:
        _require(tuple(archive.files) == SCENE31_ARRAY_NAMES, "parent scene array order changed")
        arrays = {name: np.asarray(archive[name]) for name in SCENE31_ARRAY_NAMES}
    return metadata, arrays


def _figure_contract() -> dict[str, Any]:
    return {
        "core_conclusion": (
            "At fixed source ordinal 2, authenticated class-conditional template-score "
            "predictions show where the current method agrees with or departs from the "
            "IVD-p95 vortex reference."
        ),
        "results_level_question": (
            "Where does the current class-conditional template classifier succeed or fail "
            "in Cylinder3D Re160/Re640/Re6400 and Boeing 747?"
        ),
        "archetype": "image plate + quantification",
        "backend": "Python/matplotlib",
        "panel_map": {
            "a": "spatial reference: IVD-p95 isosurface and the unchanged 240 center pathlines",
            "b": "primary evidence: all valid primitive-center binary classifications",
            "c": "error decomposition: mutually exclusive TP, FP, FN, and TN populations",
        },
        "selection": "dataset, source ordinal 2, scale blocks, camera, and pathlines were fixed independently of current predictions",
        "exclusions": "none; every valid query primitive in each selected dataset/source/block is rendered",
        "uncertainty": "none; each panel is one fixed source timeslice and carries no confidence interval",
        "reviewer_risks": [
            "tail_anomaly is a compatibility field carrying the raw class score, not a tail anomaly",
            "tail_probability equals one minus the class score and is not a posterior probability",
            "spatial_score is the continuous decision score",
            "fixed-top-fraction or positive-sigma candidates are group-transductive",
            "legacy and expanded blocks have different valid populations and are not a causal comparison",
            "the four flows are exposed-development data rather than sealed confirmation",
        ],
    }


def _verify_child_scene_invariance(
    parent_arrays: Mapping[str, np.ndarray],
    child_npz: Path,
    prediction: np.ndarray,
) -> dict[str, Any]:
    with np.load(child_npz, allow_pickle=False) as archive:
        _require(tuple(archive.files) == SCENE31_ARRAY_NAMES, "child scene array order changed")
        child = {name: np.asarray(archive[name]) for name in SCENE31_ARRAY_NAMES}
    _require(np.array_equal(child["prediction"], np.asarray(prediction, dtype=np.bool_)), "child prediction changed")
    unchanged = tuple(name for name in SCENE31_ARRAY_NAMES if name not in {"prediction", "metadata_json"})
    _require(len(unchanged) == 20, "Phase 3.1 unchanged-array contract changed")
    for name in unchanged:
        _require(
            child[name].dtype == parent_arrays[name].dtype
            and child[name].shape == parent_arrays[name].shape
            and canonical_array_sha256(child[name]) == canonical_array_sha256(parent_arrays[name]),
            f"child scene changed immutable parent array: {name}",
        )
    return {
        "unchanged_array_count": len(unchanged),
        "unchanged_array_names": list(unchanged),
        "only_prediction_and_metadata_changed": True,
    }


def _export_record(path: Path, root: Path, kind: str) -> dict[str, Any]:
    _require(path.is_file() and path.stat().st_size > 0, f"missing rendered export: {path}")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "export_kind": kind,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _reauthenticate_input_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        path = Path(str(row["path"]))
        _require(path.is_file(), f"input disappeared before completion: {path}")
        _require(path.stat().st_size == row["size_bytes"], f"input size changed before completion: {path}")
        _require(sha256_file(path) == row["sha256"], f"input SHA changed before completion: {path}")


def _published_metric_row(
    *,
    plan: ReportPlan,
    dataset: DatasetSpec,
    block: BlockSpec,
    source_index: int,
    complete_metric: Mapping[str, Any],
    loaded: LoadedPredictionGroup,
) -> dict[str, Any]:
    count = int(complete_metric["sample_count"])
    support = _mapping(
        loaded.group_audit.get("class_conditional_support"),
        "published class-conditional support",
    )
    decision_rule = str(complete_metric["decision_rule"])
    comparator = (
        "descending_score_then_center_index_fixed_top_fraction"
        if decision_rule == "fixed_top_fraction"
        else "strict_score_greater_than_threshold_tie_negative"
    )
    row: dict[str, Any] = {
        "experiment": REPORT_EXPERIMENT,
        "dataset": dataset.dataset,
        "display_name": dataset.display_name,
        "outer_family": dataset.outer_family,
        "source_ordinal": plan.source_ordinal,
        "source_index": source_index,
        "scale_block": block.block,
        "scale_start": block.scale_start,
        "scale_stop": block.scale_stop,
        "assigned_count": plan.assigned_per_block,
        "valid_count": count,
        "invalid_count": plan.assigned_per_block - count,
        "coverage": float(count / plan.assigned_per_block),
        "candidate_id": complete_metric["candidate_id"],
        "representation": complete_metric["representation"],
        "k": complete_metric["k"],
        "sigma": complete_metric["sigma"],
        "decision_rule": decision_rule,
        "decision_value": complete_metric["decision_value"],
        "decision_comparator": comparator,
        "raw_score_field": "tail_anomaly_is_class_conditional_template_score",
        "decision_score_field": "spatial_score",
        "compatibility_field": "tail_probability_is_one_minus_raw_score_not_probability",
        "uncertainty": "none_single_fixed_source",
    }
    for field in METRIC_FIELDS:
        if field not in {
            "outer_family", "inner_family", "dataset", "source_ordinal", "block",
            "candidate_id", "representation", "k", "sigma", "decision_rule", "decision_value",
        }:
            row[field] = complete_metric[field]
    row.update(
        {
            "predicted_positive_count": int(loaded.group.prediction.sum()),
            "joint_support_required_family_count": support["required_joint_family_count"],
            "joint_supported_family_count_histogram_json": json.dumps(
                support["joint_supported_family_count_histogram"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "per_family_support_json": json.dumps(
                support["families"], sort_keys=True, separators=(",", ":")
            ),
        }
    )
    return row


def render_bundle(
    plan: ReportPlan,
    *,
    output_root: Path,
    expected_reporting_commit: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build the eight-figure immutable report bundle."""

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"immutable output directory already exists: {output_root}")
    reporting_identity = _authenticate_reporting_checkout(expected_reporting_commit)
    _require(device == "cpu", "ClassConditional report authentication is frozen to CPU replay")
    environment = _environment_record(device)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "scenes").mkdir()
    (output_root / "figures").mkdir()
    frozen_config = output_root / "frozen_config.yaml"
    shutil.copyfile(plan.path, frozen_config)
    _require(sha256_file(frozen_config) == plan.sha256, "frozen report config copy changed")

    # Opaque file hashes and JSON chains are authenticated first.  No NPZ
    # member has been opened yet, so the input manifest can state that fact.
    aggregate = authenticate_aggregate_chain(plan)
    parent_scenes, parent_files = authenticate_parent_scene_chain(plan)
    input_rows = [*aggregate.files, *parent_files]
    input_rows.extend(
        [
            _file_row(plan.path, "report_config"),
            _file_row(Path(__file__).resolve(), "reporting_script"),
        ]
    )
    _require(len({row["path"] for row in input_rows}) == len(input_rows), "authenticated input paths are not unique")
    input_manifest = {
        "schema": "pathline_template_matching.class_conditional_template_score_visualization_input.v1",
        "experiment": REPORT_EXPERIMENT,
        **reporting_identity,
        "report_config_sha256": plan.sha256,
        "prediction_experiment": PREDICTION_EXPERIMENT,
        "prediction_git_commit": plan.prediction_commit,
        "prediction_config_sha256": plan.prediction_config_sha256,
        "aggregate_complete_sha256": plan.aggregate_complete_sha256,
        "parent_scene_experiment": plan.parent_experiment,
        "parent_scene_git_commit": plan.parent_commit,
        "environment": environment,
        "source_selection": "fixed source ordinal 2",
        "npz_file_bytes_hashed_before_manifest_write": True,
        "npz_array_member_access_before_manifest_write": False,
        "aggregate_completion_authenticates_prior_fresh_fold_replay": True,
        "additional_report_time_fresh_replay": False,
        "files": input_rows,
        "files_content_sha256": canonical_json_sha256(input_rows),
    }
    input_manifest["manifest_content_sha256"] = canonical_json_sha256(input_manifest)
    _atomic_json(output_root / "input_manifest.json", input_manifest)
    contract = _figure_contract()
    contract["contract_content_sha256"] = canonical_json_sha256(contract)
    _atomic_json(output_root / "figure_contract.json", contract)

    # From this point NPZ member access is allowed.  The authenticated
    # five-fold aggregate already reconstructed every source fold before it
    # published its immutable completion marker.
    groups = load_prediction_groups(plan, aggregate)
    parent_metrics = read_outer_group_metrics(plan, aggregate)
    expected_keys = {(dataset.dataset, block.block) for dataset in plan.datasets for block in plan.blocks}
    _require(set(groups) == set(parent_scenes) == set(parent_metrics) == expected_keys, "eight-figure input populations differ")

    metric_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    for dataset in plan.datasets:
        for block in plan.blocks:
            key = (dataset.dataset, block.block)
            paths = parent_scenes[key]
            parent_metadata, parent_arrays = _load_parent_scene(
                plan,
                paths,
                dataset=dataset.dataset,
                block=block.block,
            )
            loaded = groups[key]
            prediction, score = exact_bind_prediction_group(
                parent_metadata, parent_arrays, loaded.group
            )
            reference = np.asarray(parent_arrays["reference"], dtype=np.bool_)
            _require(0 < len(reference) <= plan.assigned_per_block, f"invalid valid-row population: {key}")
            complete_metric = recompute_complete_metric_row(
                reference=reference,
                loaded=loaded,
            )
            compare_metric_rows(complete_metric, parent_metrics[key])
            candidate = dict(loaded.group.candidate)
            if candidate["decision_rule"] == "fixed_top_fraction":
                decision_text = "fixed top-5% group decision"
            else:
                decision_text = (
                    f"strict spatial_score > {float(candidate['decision_value']):.2f}; "
                    "ties are negative"
                )
            semantics = (
                "Authenticated FMT class-conditional template-score outer-fold classification; "
                f"representation={candidate['representation']}, exact-scale k={candidate['k']}, "
                f"Gaussian sigma={float(candidate['sigma']):g}, {decision_text}. "
                "The compatibility field tail_anomaly stores the raw class score; "
                "spatial_score is the continuous decision score; tail_probability is one "
                "minus the raw score and is not a posterior probability."
            )
            audit = dict(parent_metadata)
            audit.update(
                {
                    "analysis_experiment": REPORT_EXPERIMENT,
                    "parent_analysis_experiment": plan.parent_experiment,
                    "prediction_parent_experiment": PREDICTION_EXPERIMENT,
                    "prediction_parent_git_commit": plan.prediction_commit,
                    "prediction_parent_config_sha256": plan.prediction_config_sha256,
                    **reporting_identity,
                    "regime": "family-held-out exposed-development fixed-source reporting",
                    "candidate": candidate,
                    "candidate_id": candidate["candidate_id"],
                    "prediction_semantics": semantics,
                    "renderer_panel_titles": list(PANEL_TITLES),
                    "renderer_prediction_semantics": semantics,
                    "prediction_positive_count": int(prediction.sum()),
                    "prediction_negative_count": int((~prediction).sum()),
                    "prediction_sha256": canonical_array_sha256(prediction),
                    "spatial_score_sha256": canonical_array_sha256(score),
                    "exact_join": {
                        "key": [
                            "dataset", "source_ordinal", "source_index", "scale_block",
                            "center_seed_index", "assigned_row_index", "scale_id",
                            "scale_block_index",
                        ],
                        "parent_count": int(len(reference)),
                        "prediction_count": int(len(prediction)),
                        "missing_count": 0,
                        "extra_count": 0,
                        "reordered_count": 0,
                    },
                    "parent_scene_reuse": {
                        "scene_npz_sha256": sha256_file(paths["npz"]),
                        "scene_manifest_file_sha256": sha256_file(paths["manifest"]),
                        "selection_changed": False,
                        "camera_changed": False,
                        "only_prediction_and_analysis_metadata_changed": True,
                    },
                    "figure_interpretation": contract,
                    "formal_confirmation": False,
                }
            )
            child_scene = _child_scene(parent_metadata, parent_arrays, prediction)
            scene_stem = (
                output_root
                / "scenes"
                / f"{dataset.dataset}_source_ordinal_2_{block.block}_class_conditional_template_score"
            )
            scene_npz = scene_stem.with_suffix(".scene.npz")
            scene_manifest = scene_stem.with_suffix(".scene.json")
            write_phase21_scene_artifact(
                child_scene,
                audit,
                scene_npz,
                scene_manifest,
            )
            invariance = _verify_child_scene_invariance(
                parent_arrays,
                scene_npz,
                prediction,
            )
            figure_stem = (
                output_root
                / "figures"
                / f"{dataset.dataset}_source_ordinal_2_{block.block}_class_conditional_template_score_triptych"
            )
            rendered = render_phase21_scene_artifact(
                scene_npz,
                scene_manifest,
                figure_stem,
                dpi=plan.dpi,
            )
            parent_render = _read_self_hashed_json(
                paths["render"], "metadata_content_sha256"
            )
            _require(
                rendered.metadata["renderer"]["camera"]
                == parent_render["renderer"]["camera"],
                f"camera changed: {key}",
            )
            _require(
                rendered.metadata["renderer"]["panel_order"] == list(PANEL_TITLES),
                f"panel order changed: {key}",
            )
            counts = rendered.metadata["counts"]
            for name in (
                "true_positive", "false_positive", "true_negative", "false_negative"
            ):
                _require(
                    int(counts[name]) == int(complete_metric[name]),
                    f"render confusion count changed: {key}/{name}",
                )
            metric_row = _published_metric_row(
                plan=plan,
                dataset=dataset,
                block=block,
                source_index=int(parent_metadata["source_index"]),
                complete_metric=complete_metric,
                loaded=loaded,
            )
            metric_rows.append(metric_row)
            exports = [
                _export_record(scene_npz, output_root, "scene_npz"),
                _export_record(scene_manifest, output_root, "scene_manifest"),
                _export_record(rendered.png_path, output_root, "png"),
                _export_record(rendered.pdf_path, output_root, "pdf"),
                _export_record(rendered.svg_path, output_root, "svg"),
                _export_record(rendered.alignment_path, output_root, "alignment"),
                _export_record(rendered.metadata_path, output_root, "render_metadata"),
            ]
            figure_rows.append(
                {
                    "dataset": dataset.dataset,
                    "display_name": dataset.display_name,
                    "scale_block": block.block,
                    "source_ordinal": plan.source_ordinal,
                    "source_index": int(parent_metadata["source_index"]),
                    "valid_count": len(reference),
                    "candidate": candidate,
                    "exact_join_passed": True,
                    "scene_invariance": invariance,
                    "exports": exports,
                    "exports_content_sha256": canonical_json_sha256(exports),
                    "metrics": metric_row,
                }
            )

    _require(len(metric_rows) == len(figure_rows) == 8, "exactly eight figures are required")
    _atomic_csv(
        output_root / "per_figure_metrics.csv",
        metric_rows,
        tuple(metric_rows[0]),
    )
    visualization_manifest = {
        "schema": "pathline_template_matching.class_conditional_template_score_visualization.v1",
        "experiment": REPORT_EXPERIMENT,
        **reporting_identity,
        "report_config_sha256": plan.sha256,
        "prediction_experiment": PREDICTION_EXPERIMENT,
        "prediction_git_commit": plan.prediction_commit,
        "prediction_config_sha256": plan.prediction_config_sha256,
        "aggregate_complete_sha256": plan.aggregate_complete_sha256,
        "evidence_scope": "family-held-out exposed-development fixed-source reporting",
        "formal_confirmation": False,
        "source_selection": "fixed source ordinal 2; no metric-based selection",
        "environment": environment,
        "figure_count": 8,
        "unique_key": ["dataset", "scale_block"],
        "entries": figure_rows,
    }
    visualization_manifest["manifest_content_sha256"] = canonical_json_sha256(
        visualization_manifest
    )
    _atomic_json(
        output_root / "visualization_manifest.json", visualization_manifest
    )

    _reauthenticate_input_rows(input_rows)
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"result_manifest.json", "RUN_COMPLETE.json"}:
            artifacts.append(
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    result = {
        "schema": "pathline_template_matching.class_conditional_template_score_visualization_result.v1",
        "experiment": REPORT_EXPERIMENT,
        **reporting_identity,
        "report_config_sha256": plan.sha256,
        "status": "completed_pending_local_pdf_collision_and_visual_QA",
        "formal_confirmation": False,
        "prediction_experiment": PREDICTION_EXPERIMENT,
        "prediction_git_commit": plan.prediction_commit,
        "prediction_config_sha256": plan.prediction_config_sha256,
        "aggregate_complete_sha256": plan.aggregate_complete_sha256,
        "environment": environment,
        "aggregate_fresh_replay_authenticated_families": list(FAMILY_ORDER),
        "report_projection_families": ["half_cylinder", "boeing_747"],
        "figure_count": 8,
        "query_count": int(sum(row["valid_count"] for row in metric_rows)),
        "input_manifest_file_sha256": sha256_file(output_root / "input_manifest.json"),
        "visualization_manifest_file_sha256": sha256_file(output_root / "visualization_manifest.json"),
        "per_figure_metrics_file_sha256": sha256_file(output_root / "per_figure_metrics.csv"),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifacts_content_sha256": canonical_json_sha256(artifacts),
    }
    result["manifest_content_sha256"] = canonical_json_sha256(result)
    _atomic_json(output_root / "result_manifest.json", result)
    complete = {
        "schema": "pathline_template_matching.class_conditional_template_score_visualization_run_complete.v1",
        "experiment": REPORT_EXPERIMENT,
        **reporting_identity,
        "report_config_sha256": plan.sha256,
        "status": "complete_pending_local_pdf_collision_and_visual_QA",
        "environment": environment,
        "figure_count": 8,
        "query_count": result["query_count"],
        "result_manifest_file_sha256": sha256_file(output_root / "result_manifest.json"),
        "result_manifest_content_sha256": result["manifest_content_sha256"],
    }
    complete["marker_content_sha256"] = canonical_json_sha256(complete)
    _atomic_json(output_root / "RUN_COMPLETE.json", complete)
    return result


def run_from_config(
    *,
    config_path: str | Path,
    config_sha256: str,
    output_root: str | Path,
    expected_reporting_commit: str,
    device: str = "cpu",
) -> dict[str, Any]:
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(
            f"immutable output directory already exists: {destination}"
        )
    plan = load_report_plan(config_path, config_sha256)
    return render_bundle(
        plan,
        output_root=destination,
        expected_reporting_commit=expected_reporting_commit,
        device=device,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-reporting-commit", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_from_config(
        config_path=args.config,
        config_sha256=args.config_sha256,
        output_root=args.output_root,
        expected_reporting_commit=args.expected_reporting_commit,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

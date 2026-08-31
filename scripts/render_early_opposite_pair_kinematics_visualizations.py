#!/usr/bin/env python3
"""Render authenticated source-ordinal-2 EarlyOppositePair triptychs.

This is a downstream reporting utility for the already completed
``Verify_EarlyOppositePairKinematics_1.1`` experiment.  It does not refit the
classifier, select a source, change a threshold, or select displayed
pathlines.  It reuses the immutable source-ordinal-2 Phase 3.1 scenes and
replaces only their prediction vector after an exact ordered identity join.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

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
    FIXED_SOURCE_ORDINAL,
    load_phase21_scene_artifact,
    render_phase21_scene_artifact,
    write_phase21_scene_artifact,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)


REPORT_EXPERIMENT = "Other_EarlyOppositePairKinematicsVisualization_1.1"
REPORT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "config"
    / "Other_EarlyOppositePairKinematicsVisualization_1.1.yaml"
)
REPORT_CONFIG_SHA256 = (
    "0b5053cdd2342fcd65950b82f08b520de4c8a2717c44ad15a5d13babd0caf1c8"
)
PREDICTION_EXPERIMENT = "Verify_EarlyOppositePairKinematics_1.1"
PREDICTION_COMMIT = "2c3774dca0d81db8edd5645e63576526b9e276f7"
PREDICTION_CONFIG_SHA256 = (
    "e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767"
)
PREDICTION_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_outer_prediction.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_outer_prediction_manifest.v1"
)
PREDICTION_RESULT_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_result.v1"
)
PREDICTION_COMPLETE_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_run_complete.v1"
)
PREDICTION_SELECTED_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_selected_candidate.v1"
)
PARENT_SCENE_EXPERIMENT = "Other_MainExp31FamilyHeldOutVisualization_1.1"
PARENT_SCENE_COMMIT = "86be29698eb689c0e269fe987a5b6d5f125a67be"
PARENT_SCENE_CONFIG_SHA256 = (
    "6fec35d2f64a3b593a74e8b35674137b1665ce169491e3546384142514b46670"
)
PARENT_SCENE_RESULT_MANIFEST_SHA256 = (
    "57f03ba16ad8cfa0e1e0a9efd93f2dde7ae5866f173fad20055efb6939d4188e"
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
PANEL_TITLES = (
    "IVD p95 + center pathlines",
    "FMT EarlyOppositePair template classification",
    "TP / FP / FN / TN against IVD p95",
)
EXPECTED_CANDIDATES = {
    "half_cylinder": {
        "candidate_id": (
            "representation=chirality_all35_plus_seed4|k=31|sigma=0.5|"
            "fixed_top_fraction=0.05"
        ),
        "representation": "chirality_all35_plus_seed4",
        "k": 31,
        "sigma": 0.5,
        "decision_rule": "fixed_top_fraction",
        "decision_value": 0.05,
    },
    "boeing_747": {
        "candidate_id": (
            "representation=real_neighbor36_plus_seed4|k=31|sigma=0.5|"
            "fixed_top_fraction=0.05"
        ),
        "representation": "real_neighbor36_plus_seed4",
        "k": 31,
        "sigma": 0.5,
        "decision_rule": "fixed_top_fraction",
        "decision_value": 0.05,
    },
}
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
FOLD_INPUT_NAMES = (
    "outer_predictions.npz",
    "outer_prediction_manifest.json",
    "outer_group_metrics.csv",
    "outer_reference_access_audit.json",
    "outer_summary.json",
    "selected_candidate.json",
    "result_manifest.json",
    "RUN_COMPLETE.json",
)
METRIC_FLOAT_NAMES = (
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
)
METRIC_INTEGER_NAMES = (
    "sample_count",
    "positive_count",
    "negative_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON must contain an object: {path}")
    return value


def _read_self_hashed_json(path: Path, field: str) -> dict[str, Any]:
    value = _read_json(path)
    claimed = value.get(field)
    _require(isinstance(claimed, str) and len(claimed) == 64, f"missing {field}: {path}")
    payload = {key: item for key, item in value.items() if key != field}
    _require(canonical_json_sha256(payload) == claimed, f"content hash mismatch: {path}")
    return value


def _authenticate_report_config(path: Path = REPORT_CONFIG_PATH) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"missing frozen report config: {path}")
    _require(
        sha256_file(path) == REPORT_CONFIG_SHA256,
        "report config SHA-256 mismatch",
    )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "report config must contain a mapping")
    _require(value.get("experiment") == REPORT_EXPERIMENT, "report experiment changed")
    _require(value.get("status") == "frozen_pre_run_not_run", "report freeze history changed")
    parent = value.get("parents")
    _require(isinstance(parent, Mapping), "report parent contract is missing")
    scenes = parent.get("family_held_out_scenes")
    _require(isinstance(scenes, Mapping), "parent scene contract is missing")
    _require(scenes.get("experiment") == PARENT_SCENE_EXPERIMENT, "scene experiment changed")
    _require(scenes.get("numerical_git_commit") == PARENT_SCENE_COMMIT, "scene commit changed")
    _require(scenes.get("config_sha256") == PARENT_SCENE_CONFIG_SHA256, "scene config changed")
    _require(
        scenes.get("result_manifest_sha256") == PARENT_SCENE_RESULT_MANIFEST_SHA256,
        "scene result identity changed",
    )
    early = parent.get("early_opposite_pair_folds")
    _require(isinstance(early, Mapping), "Early fold contract is missing")
    _require(early.get("experiment") == PREDICTION_EXPERIMENT, "prediction experiment changed")
    _require(early.get("numerical_git_commit") == PREDICTION_COMMIT, "prediction commit changed")
    _require(early.get("config_sha256") == PREDICTION_CONFIG_SHA256, "prediction config changed")
    folds = early.get("folds")
    _require(isinstance(folds, list) and len(folds) == 2, "exactly two report folds are required")
    observed_candidates = {
        str(row["outer_family"]): dict(row["candidate"])
        for row in folds
        if isinstance(row, Mapping)
        and isinstance(row.get("outer_family"), str)
        and isinstance(row.get("candidate"), Mapping)
    }
    _require(observed_candidates == EXPECTED_CANDIDATES, "frozen candidate contract changed")
    contract = value.get("prediction_contract")
    _require(isinstance(contract, Mapping), "prediction contract is missing")
    _require(contract.get("schema") == PREDICTION_SCHEMA, "prediction schema changed")
    _require(
        contract.get("manifest_schema") == PREDICTION_MANIFEST_SCHEMA,
        "prediction manifest schema changed",
    )
    _require(contract.get("result_schema") == PREDICTION_RESULT_SCHEMA, "result schema changed")
    _require(
        contract.get("completion_schema") == PREDICTION_COMPLETE_SCHEMA,
        "completion schema changed",
    )
    _require(
        tuple(contract.get("ordered_array_names", ())) == PREDICTION_ARRAY_NAMES,
        "prediction array order changed",
    )
    query = value.get("query")
    _require(isinstance(query, Mapping), "query contract is missing")
    datasets = query.get("datasets")
    blocks = query.get("scale_blocks")
    _require(
        isinstance(datasets, list)
        and tuple(row.get("id") for row in datasets if isinstance(row, Mapping))
        == tuple(DATASET_TO_FAMILY),
        "four-flow query order changed",
    )
    _require(
        isinstance(blocks, list)
        and tuple(row.get("id") for row in blocks if isinstance(row, Mapping)) == BLOCKS,
        "two-block query order changed",
    )
    _require(int(query.get("source_ordinal", -1)) == FIXED_SOURCE_ORDINAL, "source ordinal changed")
    figure = value.get("figure_contract")
    _require(isinstance(figure, Mapping), "figure contract is missing")
    _require(int(figure.get("expected_figure_count", -1)) == 8, "figure count changed")
    _require(tuple(figure.get("panel_titles", ())) == PANEL_TITLES, "panel title contract changed")
    return value


def _authenticate_reporting_checkout(expected_commit: str) -> dict[str, str]:
    _require(
        len(expected_commit) == 40
        and all(character in "0123456789abcdef" for character in expected_commit),
        "expected reporting commit must be a 40-character lowercase Git commit",
    )
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
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "scripts/render_early_opposite_pair_kinematics_visualizations.py",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(bool(tracked), "reporting script is not tracked by the authenticated commit")
    return {
        "reporting_git_commit": commit,
        "reporting_script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_report_config_sha256": REPORT_CONFIG_SHA256,
    }


def _file_row(path: Path, role: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing input file: {path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _verify_result_artifact(root: Path, result: Mapping[str, Any], name: str) -> dict[str, Any]:
    artifacts = result.get("artifacts")
    _require(isinstance(artifacts, Mapping), "fold result artifacts must be a mapping")
    spec = artifacts.get(name)
    _require(isinstance(spec, Mapping), f"fold result does not bind {name}")
    path = root / name
    _require(path.is_file(), f"missing downloaded fold artifact: {path}")
    _require(int(spec.get("size_bytes", -1)) == path.stat().st_size, f"size mismatch: {path}")
    _require(spec.get("sha256") == sha256_file(path), f"SHA-256 mismatch: {path}")
    return _file_row(path, f"prediction_fold:{root.name}:{name}")


def _authenticate_fold(root: Path, expected_family: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(expected_family in EXPECTED_CANDIDATES, "unexpected report fold")
    result_path = root / "result_manifest.json"
    complete_path = root / "RUN_COMPLETE.json"
    result = _read_self_hashed_json(result_path, "content_sha256")
    complete = _read_self_hashed_json(complete_path, "content_sha256")
    _require(result.get("schema") == PREDICTION_RESULT_SCHEMA, "result schema changed")
    _require(complete.get("schema") == PREDICTION_COMPLETE_SCHEMA, "completion schema changed")
    _require(result.get("status") == "completed", "prediction fold is not complete")
    for value, role in ((result, "result"), (complete, "completion")):
        _require(value.get("experiment") == PREDICTION_EXPERIMENT, f"wrong {role} experiment")
        _require(value.get("git_commit") == PREDICTION_COMMIT, f"wrong {role} commit")
        _require(value.get("config_sha256") == PREDICTION_CONFIG_SHA256, f"wrong {role} config")
        _require(value.get("outer_family") == expected_family, f"wrong {role} outer family")
    _require(
        complete.get("result_manifest_content_sha256") == result.get("content_sha256"),
        "completion marker does not bind result content",
    )
    _require(
        complete.get("result_manifest_file_sha256") == sha256_file(result_path),
        "completion marker does not bind result file",
    )
    evidence = [_file_row(result_path, f"prediction_fold:{expected_family}:result_manifest")]
    evidence.append(_file_row(complete_path, f"prediction_fold:{expected_family}:RUN_COMPLETE"))
    for name in FOLD_INPUT_NAMES:
        if name not in {"result_manifest.json", "RUN_COMPLETE.json"}:
            evidence.append(_verify_result_artifact(root, result, name))

    prediction_manifest = _read_self_hashed_json(root / "outer_prediction_manifest.json", "content_sha256")
    selected = _read_self_hashed_json(root / "selected_candidate.json", "content_sha256")
    _require(
        prediction_manifest.get("schema") == PREDICTION_MANIFEST_SCHEMA,
        "prediction manifest schema changed",
    )
    _require(
        prediction_manifest.get("prediction_schema") == PREDICTION_SCHEMA,
        "prediction archive schema changed",
    )
    _require(prediction_manifest.get("experiment") == PREDICTION_EXPERIMENT, "prediction experiment changed")
    _require(prediction_manifest.get("git_commit") == PREDICTION_COMMIT, "prediction commit changed")
    _require(prediction_manifest.get("config_sha256") == PREDICTION_CONFIG_SHA256, "prediction config changed")
    _require(prediction_manifest.get("outer_family") == expected_family, "prediction fold changed")
    _require(
        prediction_manifest.get("valid_labels_opened") is False
        and prediction_manifest.get("metadata_json_opened") is False,
        "prediction archive was not produced label-free",
    )
    _require(int(prediction_manifest.get("array_count", -1)) == len(PREDICTION_ARRAY_NAMES), "array count changed")
    _require(set(prediction_manifest.get("arrays", {})) == set(PREDICTION_ARRAY_NAMES), "array schema changed")
    prediction_file = prediction_manifest.get("prediction_file")
    _require(isinstance(prediction_file, Mapping), "prediction file binding is missing")
    _require(prediction_file.get("path") == "outer_predictions.npz", "prediction path changed")
    _require(prediction_file.get("sha256") == sha256_file(root / "outer_predictions.npz"), "prediction SHA changed")
    _require(int(prediction_file.get("size_bytes", -1)) == (root / "outer_predictions.npz").stat().st_size, "prediction size changed")
    candidate = selected.get("candidate")
    _require(isinstance(candidate, Mapping), "selected candidate is missing")
    _require(selected.get("schema") == PREDICTION_SELECTED_SCHEMA, "selected schema changed")
    _require(selected.get("experiment") == PREDICTION_EXPERIMENT, "selected experiment changed")
    _require(selected.get("git_commit") == PREDICTION_COMMIT, "selected commit changed")
    _require(selected.get("config_sha256") == PREDICTION_CONFIG_SHA256, "selected config changed")
    _require(dict(candidate) == prediction_manifest.get("selected_candidate"), "candidate differs across artifacts")
    _require(dict(candidate) == result.get("selected_candidate"), "candidate differs from result")
    _require(dict(candidate) == EXPECTED_CANDIDATES[expected_family], "selected candidate changed")
    _require(selected.get("outer_family") == expected_family, "selected candidate fold changed")
    return {
        "root": root,
        "family": expected_family,
        "result": result,
        "prediction_manifest": prediction_manifest,
        "candidate": dict(candidate),
    }, evidence


def _authenticate_parent(parent_root: Path) -> tuple[dict[tuple[str, str], dict[str, Path]], list[dict[str, Any]]]:
    result_path = parent_root / "result_manifest.json"
    complete_path = parent_root / "RUN_COMPLETE.json"
    result = _read_self_hashed_json(result_path, "manifest_content_sha256")
    complete = _read_json(complete_path)
    _require(
        sha256_file(result_path) == PARENT_SCENE_RESULT_MANIFEST_SHA256,
        "parent result manifest SHA-256 changed",
    )
    _require(result.get("experiment") == PARENT_SCENE_EXPERIMENT, "parent result experiment changed")
    _require(result.get("git_commit") == PARENT_SCENE_COMMIT, "parent result commit changed")
    _require(result.get("config_sha256") == PARENT_SCENE_CONFIG_SHA256, "parent result config changed")
    _require(complete.get("experiment") == PARENT_SCENE_EXPERIMENT, "parent completion experiment changed")
    _require(complete.get("git_commit") == PARENT_SCENE_COMMIT, "parent completion commit changed")
    _require(complete.get("result_manifest_file_sha256") == sha256_file(result_path), "parent completion does not bind result")
    _require(complete.get("result_manifest_content_sha256") == result.get("manifest_content_sha256"), "parent completion content binding changed")
    artifact_rows = result.get("artifacts")
    _require(isinstance(artifact_rows, list), "parent result artifact list changed")
    by_relative = {
        str(row["relative_path"]): row
        for row in artifact_rows
        if isinstance(row, Mapping) and "relative_path" in row
    }
    scenes: dict[tuple[str, str], dict[str, Path]] = {}
    evidence = [
        _file_row(result_path, "parent_scene:result_manifest"),
        _file_row(complete_path, "parent_scene:RUN_COMPLETE"),
    ]
    for dataset in DATASET_TO_FAMILY:
        for block in BLOCKS:
            stem = f"{dataset}_source_ordinal_2_{block}"
            paths = {
                "npz": parent_root / "scenes" / f"{stem}.scene.npz",
                "manifest": parent_root / "scenes" / f"{stem}.scene.json",
                "render": parent_root / "figures" / f"{stem}_family_heldout_triptych.render.json",
            }
            for role, path in paths.items():
                relative = path.relative_to(parent_root).as_posix()
                row = by_relative.get(relative)
                _require(isinstance(row, Mapping), f"parent result does not bind {relative}")
                _require(path.is_file(), f"missing parent input: {path}")
                _require(int(row.get("size_bytes", -1)) == path.stat().st_size, f"parent size mismatch: {path}")
                _require(row.get("sha256") == sha256_file(path), f"parent SHA mismatch: {path}")
                evidence.append(_file_row(path, f"parent_scene:{dataset}:{block}:{role}"))
            scenes[(dataset, block)] = paths
    return scenes, evidence


def _manifest_array_spec(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    arrays = manifest.get("arrays")
    _require(isinstance(arrays, Mapping), "prediction arrays mapping is missing")
    spec = arrays.get(name)
    _require(isinstance(spec, Mapping), f"prediction array spec missing: {name}")
    return spec


def _load_prediction_groups(authenticated: Mapping[str, Any]) -> dict[tuple[str, str], PredictionGroup]:
    root = Path(authenticated["root"])
    family = str(authenticated["family"])
    manifest = authenticated["prediction_manifest"]
    candidate = dict(authenticated["candidate"])
    expected_datasets = tuple(
        dataset for dataset, value in DATASET_TO_FAMILY.items() if value == family
    )
    groups: dict[tuple[str, str], PredictionGroup] = {}
    with np.load(root / "outer_predictions.npz", allow_pickle=False) as archive:
        _require(tuple(archive.files) == PREDICTION_ARRAY_NAMES, "prediction NPZ array order changed")
        arrays: dict[str, np.ndarray] = {}
        for name in PREDICTION_ARRAY_NAMES:
            value = np.asarray(archive[name])
            spec = _manifest_array_spec(manifest, name)
            _require(value.dtype.str == PREDICTION_DTYPES[name], f"prediction dtype changed: {name}")
            _require(list(value.shape) == spec.get("shape"), f"prediction shape changed: {name}")
            _require(canonical_array_sha256(value) == spec.get("sha256"), f"prediction array SHA mismatch: {name}")
            arrays[name] = value
        _require(set(np.unique(arrays["dataset"]).tolist()) == set(expected_datasets), "prediction dataset population changed")
        _require(set(np.unique(arrays["source_ordinal"]).tolist()) == {0, 1, 2, 3}, "prediction source population changed")
        _require(set(np.unique(arrays["scale_block_index"]).tolist()) == {0, 1}, "prediction block population changed")
        for dataset in expected_datasets:
            for block, block_index in BLOCK_INDEX.items():
                mask = (
                    (arrays["dataset"] == dataset)
                    & (arrays["source_ordinal"] == FIXED_SOURCE_ORDINAL)
                    & (arrays["scale_block_index"] == block_index)
                )
                count = int(mask.sum())
                _require(count > 0, f"missing prediction group: {(dataset, block)}")
                center = np.asarray(arrays["center_seed_index"][mask], dtype=np.int64)
                assigned = np.asarray(arrays["assigned_row_index"][mask], dtype=np.int64)
                scale_id = np.asarray(arrays["scale_id"][mask], dtype=np.int32)
                scale_block = np.asarray(arrays["scale_block_index"][mask], dtype=np.int8)
                _require(len(np.unique(center)) == count, f"duplicate center identity: {(dataset, block)}")
                _require(len(np.unique(assigned)) == count, f"duplicate assigned identity: {(dataset, block)}")
                _require(np.array_equal(assigned, center + block_index * 64000), f"assigned identity changed: {(dataset, block)}")
                _require(np.array_equal(scale_block.astype(np.int32), scale_id // 1000), f"scale/block identity changed: {(dataset, block)}")
                source_values = np.unique(arrays["source_index"][mask])
                _require(len(source_values) == 1, f"source index is not unique: {(dataset, block)}")
                groups[(dataset, block)] = PredictionGroup(
                    dataset=dataset,
                    source_ordinal=FIXED_SOURCE_ORDINAL,
                    source_index=int(source_values[0]),
                    block=block,
                    outer_family=family,
                    candidate=candidate,
                    center_seed_index=np.array(center, copy=True),
                    assigned_row_index=np.array(assigned, copy=True),
                    scale_id=np.array(scale_id, copy=True),
                    scale_block_index=np.array(scale_block, copy=True),
                    spatial_score=np.array(arrays["spatial_score"][mask], dtype=np.float64, copy=True),
                    prediction=np.array(arrays["prediction"][mask], dtype=np.bool_, copy=True),
                )
    expected = {(dataset, block) for dataset in expected_datasets for block in BLOCKS}
    _require(set(groups) == expected, f"fixed-source groups changed for {family}")
    return groups


def _load_parent_metrics(fold_root: Path, family: str) -> dict[tuple[str, str], dict[str, str]]:
    with (fold_root / "outer_group_metrics.csv").open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    expected_datasets = {dataset for dataset, value in DATASET_TO_FAMILY.items() if value == family}
    selected: dict[tuple[str, str], dict[str, str]] = {}
    observed: set[tuple[str, int, str]] = set()
    for row in rows:
        _require(row["outer_family"] == family, "outer metric fold changed")
        _require(row["dataset"] in expected_datasets, "cross-fold metric row")
        key3 = (row["dataset"], int(row["source_ordinal"]), row["block"])
        _require(key3 not in observed, f"duplicate outer metric row: {key3}")
        observed.add(key3)
        if int(row["source_ordinal"]) == FIXED_SOURCE_ORDINAL:
            selected[(row["dataset"], row["block"])] = row
    expected_population = {
        (dataset, source, block)
        for dataset in expected_datasets
        for source in range(4)
        for block in BLOCKS
    }
    _require(observed == expected_population, f"outer metric population changed for {family}")
    _require(set(selected) == {(dataset, block) for dataset in expected_datasets for block in BLOCKS}, "fixed metrics missing")
    return selected


def _compare_metrics(observed: Mapping[str, Any], expected: Mapping[str, str]) -> None:
    for name in METRIC_INTEGER_NAMES:
        _require(int(observed[name]) == int(expected[name]), f"metric mismatch: {name}")
    for name in METRIC_FLOAT_NAMES:
        _require(
            abs(float(observed[name]) - float(expected[name])) <= 1e-12,
            f"metric mismatch: {name}",
        )


def _load_parent_scene(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    artifact = load_phase21_scene_artifact(paths["npz"], paths["manifest"])
    _require(artifact.metadata.get("analysis_experiment") == PARENT_SCENE_EXPERIMENT, "parent analysis experiment changed")
    _require(artifact.metadata.get("experiment") == "mainExp_TemplateMatching_3.1", "parent cache experiment changed")
    with np.load(paths["npz"], allow_pickle=False) as archive:
        _require(tuple(archive.files) == SCENE31_ARRAY_NAMES, "parent scene array order changed")
        arrays = {name: np.asarray(archive[name]) for name in SCENE31_ARRAY_NAMES}
    return artifact.metadata, arrays


def _figure_contract() -> dict[str, Any]:
    return {
        "core_conclusion": (
            "At fixed source ordinal 2, authenticated EarlyOppositePair predictions "
            "show where FMT template classification agrees with or misses IVD-p95 "
            "vortex regions."
        ),
        "results_level_question": (
            "Where are EarlyOppositePair template predictions correct or incorrect in "
            "Cylinder3D Re160/Re640/Re6400 and Boeing 747?"
        ),
        "archetype": "image plate + quantification",
        "backend": "Python/matplotlib",
        "panel_map": {
            "a": "IVD-p95 spatial reference and fixed 240 center pathlines",
            "b": "complete valid-row EarlyOppositePair binary template classification",
            "c": "complete valid-row TP/FP/FN/TN spatial error decomposition",
        },
        "selection": "source ordinal 2 and displayed pathlines were fixed before these predictions",
        "exclusions": "none; all valid query primitives in the selected dataset/source/block are rendered",
        "uncertainty": "none; each figure is one preregistered source timeslice",
        "reviewer_risks": [
            "fixed top-5% decision is group-transductive",
            "240 pathlines are reference-balanced explanatory context, not natural prevalence",
            "legacy and expanded blocks have different valid populations and cannot be compared causally from these images alone",
            "all four flows are exposed-development data, not sealed confirmation",
        ],
    }


def render_bundle(
    *,
    parent_root: Path,
    half_fold_root: Path,
    boeing_fold_root: Path,
    output_root: Path,
    dpi: int,
    expected_reporting_commit: str | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"immutable output directory already exists: {output_root}")
    report_config = _authenticate_report_config()
    if expected_reporting_commit is None:
        expected_reporting_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    reporting_identity = _authenticate_reporting_checkout(expected_reporting_commit)
    _require(int(dpi) == 360, "frozen report DPI changed")
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "scenes").mkdir()
    (output_root / "figures").mkdir()
    frozen_config_path = output_root / "frozen_config.yaml"
    shutil.copyfile(REPORT_CONFIG_PATH, frozen_config_path)
    _require(sha256_file(frozen_config_path) == REPORT_CONFIG_SHA256, "frozen config copy changed")

    parent_scenes, input_rows = _authenticate_parent(parent_root)
    half_fold, half_rows = _authenticate_fold(half_fold_root, "half_cylinder")
    boeing_fold, boeing_rows = _authenticate_fold(boeing_fold_root, "boeing_747")
    input_rows.extend(half_rows)
    input_rows.extend(boeing_rows)
    input_rows.extend(
        [
            _file_row(REPORT_CONFIG_PATH, "report_config"),
            _file_row(Path(__file__).resolve(), "reporting_script"),
        ]
    )
    _require(len({row["path"] for row in input_rows}) == len(input_rows), "duplicate authenticated input path")
    input_manifest = {
        "schema": "pathline_template_matching.early_opposite_pair_visualization_input.v1",
        "experiment": REPORT_EXPERIMENT,
        **reporting_identity,
        "prediction_experiment": PREDICTION_EXPERIMENT,
        "prediction_git_commit": PREDICTION_COMMIT,
        "prediction_config_sha256": PREDICTION_CONFIG_SHA256,
        "parent_scene_experiment": PARENT_SCENE_EXPERIMENT,
        "parent_scene_git_commit": PARENT_SCENE_COMMIT,
        "report_config_status": report_config["status"],
        "source_selection": "fixed source ordinal 2",
        "npz_array_access_before_manifest_write": False,
        "files": input_rows,
        "files_content_sha256": canonical_json_sha256(input_rows),
    }
    input_manifest["manifest_content_sha256"] = canonical_json_sha256(input_manifest)
    _atomic_json(output_root / "input_manifest.json", input_manifest)
    contract = _figure_contract()
    contract["contract_content_sha256"] = canonical_json_sha256(contract)
    _atomic_json(output_root / "figure_contract.json", contract)

    groups = {}
    groups.update(_load_prediction_groups(half_fold))
    groups.update(_load_prediction_groups(boeing_fold))
    parent_metrics = {}
    parent_metrics.update(_load_parent_metrics(half_fold_root, "half_cylinder"))
    parent_metrics.update(_load_parent_metrics(boeing_fold_root, "boeing_747"))
    _require(set(groups) == set(parent_scenes) == set(parent_metrics), "eight-figure key population changed")

    metric_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    for dataset in DATASET_TO_FAMILY:
        for block in BLOCKS:
            key = (dataset, block)
            paths = parent_scenes[key]
            parent_metadata, arrays = _load_parent_scene(paths)
            prediction, score = exact_bind_prediction_group(parent_metadata, arrays, groups[key])
            reference = np.asarray(arrays["reference"], dtype=np.bool_)
            _require(0 < len(reference) <= 64_000, f"invalid coverage population: {key}")
            metrics = _metric_values(reference, prediction, score)
            _compare_metrics(metrics, parent_metrics[key])
            candidate = dict(groups[key].candidate)
            semantics = (
                "Authenticated FMT EarlyOppositePair outer-fold template classification; "
                "the parent FMT representation is augmented by the frozen 4D synchronous "
                "seed-time opposite-pair kinematics block; "
                f"representation={candidate['representation']}, exact same-scale negative retrieval "
                f"k={candidate['k']}, Gaussian sigma={candidate['sigma']:g}, "
                "per-scale fit-negative robust scaling, and fixed top-5% group decision."
            )
            audit = dict(parent_metadata)
            audit.update(
                {
                    "analysis_experiment": REPORT_EXPERIMENT,
                    "parent_analysis_experiment": PARENT_SCENE_EXPERIMENT,
                    "prediction_parent_experiment": PREDICTION_EXPERIMENT,
                    "prediction_parent_git_commit": PREDICTION_COMMIT,
                    "prediction_parent_config_sha256": PREDICTION_CONFIG_SHA256,
                    **reporting_identity,
                    "regime": "family-held-out exposed-development",
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
                            "dataset",
                            "source_ordinal",
                            "scale_block",
                            "center_seed_index",
                            "assigned_row_index",
                            "scale_id",
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
            child_scene = _child_scene(parent_metadata, arrays, prediction)
            scene_stem = output_root / "scenes" / f"{dataset}_source_ordinal_2_{block}_early_opposite_pair"
            scene_npz = scene_stem.with_suffix(".scene.npz")
            scene_manifest = scene_stem.with_suffix(".scene.json")
            write_phase21_scene_artifact(child_scene, audit, scene_npz, scene_manifest)
            figure_stem = output_root / "figures" / f"{dataset}_source_ordinal_2_{block}_early_opposite_pair_triptych"
            rendered = render_phase21_scene_artifact(
                scene_npz,
                scene_manifest,
                figure_stem,
                dpi=dpi,
            )
            parent_render = _read_self_hashed_json(paths["render"], "metadata_content_sha256")
            _require(
                rendered.metadata["renderer"]["camera"] == parent_render["renderer"]["camera"],
                f"camera changed: {key}",
            )
            _require(rendered.metadata["renderer"]["panel_order"] == list(PANEL_TITLES), "panel order changed")
            counts = rendered.metadata["counts"]
            for name in ("true_positive", "false_positive", "true_negative", "false_negative"):
                _require(int(counts[name]) == int(metrics[name]), f"render count changed: {key}/{name}")
            row = {
                "experiment": REPORT_EXPERIMENT,
                "dataset": dataset,
                "display_name": DISPLAY_NAMES[dataset],
                "outer_family": DATASET_TO_FAMILY[dataset],
                "source_ordinal": FIXED_SOURCE_ORDINAL,
                "source_index": int(parent_metadata["source_index"]),
                "scale_block": block,
                "valid_count": int(len(reference)),
                "assigned_count": 64_000,
                "invalid_count": 64_000 - int(len(reference)),
                "coverage": float(len(reference) / 64_000),
                "candidate_id": candidate["candidate_id"],
                "representation": candidate["representation"],
                "k": int(candidate["k"]),
                "sigma": float(candidate["sigma"]),
                "decision_rule": candidate["decision_rule"],
                "decision_value": float(candidate["decision_value"]),
                **metrics,
            }
            metric_rows.append(row)
            figure_rows.append(
                {
                    "dataset": dataset,
                    "scale_block": block,
                    "png": str(rendered.png_path.relative_to(output_root)).replace("\\", "/"),
                    "png_sha256": sha256_file(rendered.png_path),
                    "pdf": str(rendered.pdf_path.relative_to(output_root)).replace("\\", "/"),
                    "pdf_sha256": sha256_file(rendered.pdf_path),
                    "svg": str(rendered.svg_path.relative_to(output_root)).replace("\\", "/"),
                    "svg_sha256": sha256_file(rendered.svg_path),
                    "alignment": str(rendered.alignment_path.relative_to(output_root)).replace("\\", "/"),
                    "alignment_sha256": sha256_file(rendered.alignment_path),
                    "render_metadata": str(rendered.metadata_path.relative_to(output_root)).replace("\\", "/"),
                    "render_metadata_sha256": sha256_file(rendered.metadata_path),
                    "metrics": row,
                }
            )

    _require(len(metric_rows) == 8 and len(figure_rows) == 8, "exactly eight figures are required")
    _atomic_csv(output_root / "per_figure_metrics.csv", metric_rows, tuple(metric_rows[0]))
    visualization_manifest = {
        "schema": "pathline_template_matching.early_opposite_pair_visualization.v1",
        "experiment": REPORT_EXPERIMENT,
        **reporting_identity,
        "evidence_scope": "family-held-out exposed-development fixed-source reporting",
        "formal_confirmation": False,
        "source_selection": "fixed source ordinal 2; no metric-based selection",
        "figure_count": 8,
        "unique_key": ["dataset", "scale_block"],
        "entries": figure_rows,
    }
    visualization_manifest["manifest_content_sha256"] = canonical_json_sha256(visualization_manifest)
    _atomic_json(output_root / "visualization_manifest.json", visualization_manifest)

    artifacts = []
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
        "schema": "pathline_template_matching.early_opposite_pair_visualization_result.v1",
        "experiment": REPORT_EXPERIMENT,
        **reporting_identity,
        "status": "completed_pending_local_pdf_collision_qa",
        "formal_confirmation": False,
        "prediction_experiment": PREDICTION_EXPERIMENT,
        "prediction_git_commit": PREDICTION_COMMIT,
        "prediction_config_sha256": PREDICTION_CONFIG_SHA256,
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
        "schema": "pathline_template_matching.early_opposite_pair_visualization_run_complete.v1",
        "experiment": REPORT_EXPERIMENT,
        **reporting_identity,
        "status": "complete_pending_local_pdf_collision_qa",
        "figure_count": 8,
        "query_count": result["query_count"],
        "result_manifest_file_sha256": sha256_file(output_root / "result_manifest.json"),
        "result_manifest_content_sha256": result["manifest_content_sha256"],
    }
    complete["marker_content_sha256"] = canonical_json_sha256(complete)
    _atomic_json(output_root / "RUN_COMPLETE.json", complete)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--half-fold-root", type=Path, required=True)
    parser.add_argument("--boeing-fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=360)
    parser.add_argument("--expected-reporting-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = render_bundle(
        parent_root=args.parent_root.resolve(),
        half_fold_root=args.half_fold_root.resolve(),
        boeing_fold_root=args.boeing_fold_root.resolve(),
        output_root=args.output_root.resolve(),
        dpi=int(args.dpi),
        expected_reporting_commit=args.expected_reporting_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

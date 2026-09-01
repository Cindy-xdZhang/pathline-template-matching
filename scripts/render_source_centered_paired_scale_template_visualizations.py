#!/usr/bin/env python3
"""Render four authenticated source-centered paired-scale triptychs.

The report consumes only authenticated ``Verify_SourceCenteredPairedScaleTemplate_1.1``
folds and immutable Phase 3.1 parent scenes.  It writes the complete opaque
input manifest before opening any NPZ member, then joins the one paired-center
prediction to the union of the two parent block scenes.  No fitting, candidate
selection, threshold change, block vote, or result-based scene selection is
performed here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.phase21_pipeline import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _metric_values,
)
from pathline_template_matching.phase21_visualization import (  # noqa: E402
    DATASET_VIEWS,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.source_centered_visualization import (  # noqa: E402
    BLOCKS,
    CENTER_COUNT,
    PANEL_TITLES,
    SCENE_ARRAY_NAMES,
    SCENE_SCHEMA,
    SOURCE_ORDINAL,
    CombinedCenterScene,
    bind_valid_projection,
    combine_parent_block_scenes,
    render_source_centered_triptych,
    scene_arrays,
)
from scripts import (  # noqa: E402
    aggregate_verify_source_centered_paired_scale_template_1_1 as aggregate,
)
from scripts import run_verify_source_centered_paired_scale_template_1_1 as runner  # noqa: E402
from scripts.render_early_opposite_pair_kinematics_visualizations import (  # noqa: E402
    PARENT_SCENE_COMMIT,
    PARENT_SCENE_CONFIG_SHA256,
    PARENT_SCENE_EXPERIMENT,
    PARENT_SCENE_RESULT_MANIFEST_SHA256,
    _authenticate_parent,
    _load_parent_scene,
    _read_self_hashed_json as _read_parent_self_hashed_json,
)


EXPERIMENT = "Other_SourceCenteredPairedScaleTemplateVisualization_1.1"
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "config"
    / "Other_SourceCenteredPairedScaleTemplateVisualization_1.1.yaml"
)
CONFIG_SHA256 = "c9c9a14b02fc3f47a4ee934ccd1091a7c7accefdbd28f569100605bf8230ca4e"
TRUSTED_NUMERICAL_AGGREGATOR_GIT_COMMIT = (
    "a85c007ef961ce53bb40946ca3f38f033bf7a646"
)
DATASETS = (
    "cylinder3d",
    "halfcylinderRe640",
    "halfcylinderRe6400",
    "boeing747",
)
DISPLAY_NAMES = {
    "cylinder3d": "Cylinder3D Re160",
    "halfcylinderRe640": "Cylinder3D Re640",
    "halfcylinderRe6400": "Cylinder3D Re6400",
    "boeing747": "Boeing 747",
}
DATASET_TO_FAMILY = {
    "cylinder3d": "half_cylinder",
    "halfcylinderRe640": "half_cylinder",
    "halfcylinderRe6400": "half_cylinder",
    "boeing747": "boeing_747",
}
REQUIRED_FAMILIES = ("half_cylinder", "boeing_747")
METRIC_FLOAT_FIELDS = (
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
)
METRIC_INTEGER_FIELDS = (
    "sample_count",
    "positive_count",
    "negative_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
)
AVAILABILITY_FIELDS = (
    "unique_center_combined_coverage",
    "both_valid_count",
    "legacy_only_count",
    "expanded_only_count",
    "neither_valid_count",
)
REPORTING_DEPENDENCY_RELATIVE_PATHS = (
    "scripts/render_source_centered_paired_scale_template_visualizations.py",
    "scripts/audit_source_centered_paired_scale_template_visualizations.py",
    "ibex/other_source_centered_paired_scale_template_visualization_1.1.sh",
    "src/pathline_template_matching/source_centered_visualization.py",
    "src/pathline_template_matching/visualization.py",
    "src/pathline_template_matching/phase21_pipeline.py",
    "src/pathline_template_matching/portable_flow.py",
    "scripts/run_verify_source_centered_paired_scale_template_1_1.py",
    "scripts/aggregate_verify_source_centered_paired_scale_template_1_1.py",
    "scripts/render_early_opposite_pair_kinematics_visualizations.py",
)


@dataclass(frozen=True, slots=True)
class AuthenticatedRelease:
    root: Path
    mode: str
    aggregator_git_commit: str
    fold_git_commit: str
    manifest: Mapping[str, Any]
    report: Mapping[str, Any]
    source_centered_evidence: Mapping[str, Any]
    source_folds: Mapping[str, Mapping[str, Any]]
    evidence: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class AuthenticatedFold:
    root: Path
    outer_family: str
    git_commit: str
    result: Mapping[str, Any]
    prediction_manifest: Mapping[str, Any]
    candidate: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PredictionGroup:
    dataset: str
    outer_family: str
    candidate: Mapping[str, Any]
    unique: Mapping[str, np.ndarray]
    valid: Mapping[str, np.ndarray]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_row(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"missing authenticated input: {path}")
    return {
        "role": role,
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _read_self_hashed_json(path: Path) -> dict[str, Any]:
    return runner._read_self_hashed_json(  # type: ignore[attr-defined]
        path, expected_file_sha256=sha256_file(path)
    )


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_absolute_machine_path(value: object) -> bool:
    text = str(value)
    return Path(text).is_absolute() or PurePosixPath(text).is_absolute()


def _authenticate_release_source_evidence(
    value: object,
    *,
    aggregator_git_commit: str,
    fold_git_commit: str,
) -> Mapping[str, Any]:
    """Validate the aggregate's complete source-centered evidence envelope."""

    _require(isinstance(value, Mapping), "release source-centered evidence is missing")
    expected_fields = {
        "experiment",
        "config_sha256",
        "git_commit",
        "source_centered_input_manifest",
        "source_centered_sidecars",
    }
    _require(set(value) == expected_fields, "release source-centered evidence fields changed")
    _require(
        value.get("experiment") == runner.EXPERIMENT
        and value.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256
        and value.get("git_commit")
        == aggregator_git_commit
        == fold_git_commit
        == TRUSTED_NUMERICAL_AGGREGATOR_GIT_COMMIT,
        "release source-centered evidence Git/config identity changed",
    )
    input_manifest = value.get("source_centered_input_manifest")
    _require(
        isinstance(input_manifest, Mapping)
        and set(input_manifest) == {"path", "size_bytes", "file_sha256", "content_sha256"}
        and _is_absolute_machine_path(input_manifest.get("path", ""))
        and int(input_manifest.get("size_bytes", 0)) > 0
        and _is_lower_sha256(input_manifest.get("file_sha256"))
        and _is_lower_sha256(input_manifest.get("content_sha256")),
        "release source-centered input-manifest evidence changed",
    )
    sidecars = value.get("source_centered_sidecars")
    expected_sidecar_fields = {
        "root",
        "population_manifest_path",
        "population_manifest_size_bytes",
        "population_manifest_file_sha256",
        "population_manifest_content_sha256",
        "sidecar_count",
        "rows_content_sha256",
        "assigned_row_count_total",
        "valid_projection_row_count_total",
        "row_identities",
    }
    _require(
        isinstance(sidecars, Mapping)
        and set(sidecars) == expected_sidecar_fields
        and _is_absolute_machine_path(sidecars.get("root", ""))
        and _is_absolute_machine_path(sidecars.get("population_manifest_path", ""))
        and int(sidecars.get("population_manifest_size_bytes", 0)) > 0
        and _is_lower_sha256(sidecars.get("population_manifest_file_sha256"))
        and _is_lower_sha256(sidecars.get("population_manifest_content_sha256"))
        and _is_lower_sha256(sidecars.get("rows_content_sha256"))
        and int(sidecars.get("sidecar_count", -1)) == 32
        and int(sidecars.get("assigned_row_count_total", 0)) > 0
        and int(sidecars.get("valid_projection_row_count_total", 0)) > 0,
        "release source-centered sidecar evidence changed",
    )
    rows = sidecars.get("row_identities")
    _require(
        isinstance(rows, list)
        and len(rows) == 32
        and all(
            isinstance(row, Mapping)
            and set(row) == set(aggregate._AGGREGATE_ROW_EVIDENCE_FIELDS)  # type: ignore[attr-defined]
            and all(
                _is_lower_sha256(row[field])
                for field in (
                    "completion_file_sha256",
                    "sidecar_file_sha256",
                    "sidecar_combined_array_sha256",
                    "valid_projection_sha256",
                )
            )
            for row in rows
        ),
        "release source-centered row evidence changed",
    )
    return value


def _authenticate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file() and sha256_file(path) == CONFIG_SHA256, "frozen report config changed")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "report config must contain one mapping")
    _require(value.get("experiment") == EXPERIMENT, "report experiment changed")
    _require(
        value.get("status")
        == "frozen_before_reading_any_source_centered_prediction_metric_or_parent_scene",
        "report freeze history changed",
    )
    method = value.get("method_parent")
    _require(isinstance(method, Mapping), "method parent contract is missing")
    _require(method.get("experiment") == runner.EXPERIMENT, "method experiment changed")
    _require(method.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256, "method config changed")
    _require(tuple(method.get("required_outer_families", ())) == REQUIRED_FAMILIES, "outer-family report scope changed")
    _require(
        tuple(method.get("prediction_arrays", {}).get("unique", ()))
        == tuple(runner.UNIQUE_PREDICTION_DTYPES),
        "unique prediction array contract changed",
    )
    _require(
        tuple(method.get("prediction_arrays", {}).get("valid_projection", ()))
        == tuple(runner.VALID_PREDICTION_DTYPES),
        "valid prediction array contract changed",
    )
    parent = value.get("parent_scenes")
    _require(isinstance(parent, Mapping), "parent scene contract is missing")
    _require(
        parent.get("experiment") == PARENT_SCENE_EXPERIMENT
        and parent.get("numerical_git_commit") == PARENT_SCENE_COMMIT
        and parent.get("config_sha256") == PARENT_SCENE_CONFIG_SHA256
        and parent.get("result_manifest_sha256") == PARENT_SCENE_RESULT_MANIFEST_SHA256,
        "parent scene identity changed",
    )
    query = value.get("query")
    _require(isinstance(query, Mapping), "query contract is missing")
    _require(int(query.get("source_ordinal", -1)) == SOURCE_ORDINAL, "source ordinal changed")
    _require(
        tuple(row.get("id") for row in query.get("datasets", ()) if isinstance(row, Mapping))
        == DATASETS,
        "four-flow order changed",
    )
    _require(
        tuple(row.get("id") for row in query.get("scale_blocks", ()) if isinstance(row, Mapping))
        == BLOCKS,
        "scale-block order changed",
    )
    figure = value.get("figure_contract")
    _require(isinstance(figure, Mapping), "figure contract is missing")
    _require(int(figure.get("expected_figure_count", -1)) == 4, "figure count changed")
    _require(tuple(figure.get("panel_titles", ())) == PANEL_TITLES, "panel titles changed")
    _require(
        tuple(value.get("scene_schema", {}).get("ordered_arrays", ())) == SCENE_ARRAY_NAMES,
        "combined scene schema changed",
    )
    return value


def _authenticate_reporting_checkout(expected_commit: str) -> dict[str, Any]:
    _require(
        len(expected_commit) == 40
        and expected_commit == expected_commit.lower()
        and all(character in "0123456789abcdef" for character in expected_commit),
        "expected reporting commit must be one lowercase 40-hex commit",
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
        ["git", "ls-files", "--error-unmatch", *REPORTING_DEPENDENCY_RELATIVE_PATHS],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    _require(
        len(tracked) == len(REPORTING_DEPENDENCY_RELATIVE_PATHS)
        and {path.replace("\\", "/") for path in tracked}
        == set(REPORTING_DEPENDENCY_RELATIVE_PATHS),
        "reporting dependency file set is not tracked exactly",
    )
    return {
        "reporting_git_commit": commit,
        "reporting_script_sha256": sha256_file(Path(__file__).resolve()),
        "reporting_dependency_sha256": {
            relative: sha256_file(REPOSITORY_ROOT / relative)
            for relative in REPORTING_DEPENDENCY_RELATIVE_PATHS
        },
        "frozen_report_config_sha256": CONFIG_SHA256,
    }


def _authenticate_slurm_runtime(
    *,
    parent_root: Path,
    release_roots: Sequence[Path],
    half_fold_root: Path,
    boeing_fold_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Require the frozen CPU Rome Slurm envelope for production rendering."""

    expected_environment = {
        "SLURM_JOB_NAME": "PTMSCPairedViz",
        "SLURM_JOB_ACCOUNT": "pi-hadwigm",
        "SLURM_CPUS_PER_TASK": "32",
        "SLURM_MEM_PER_NODE": "131072",
    }
    for name, expected in expected_environment.items():
        _require(os.environ.get(name) == expected, f"production Slurm identity changed: {name}")
    _require(
        os.environ.get("SLURM_JOB_PARTITION") in {"cpu", "batch"},
        "production Slurm partition must be cpu or batch",
    )
    job_id = os.environ.get("SLURM_JOB_ID", "")
    _require(job_id.isdecimal(), "production Slurm job ID is missing or invalid")
    forbidden_gpu = {
        name: value
        for name, value in os.environ.items()
        if name in {"SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS"}
        and value.strip() not in {"", "0", "N/A"}
    }
    _require(not forbidden_gpu, "GPU allocation is forbidden for the CPU report")
    runtime_paths = {
        "parent_root": str(parent_root.resolve()),
        "release_roots": [str(path.resolve()) for path in release_roots],
        "half_fold_root": str(half_fold_root.resolve()),
        "boeing_fold_root": str(boeing_fold_root.resolve()),
        "output_root": str(output_root.resolve()),
    }
    _require(
        all(Path(value).is_absolute() for value in runtime_paths["release_roots"])
        and all(
            Path(str(runtime_paths[name])).is_absolute()
            for name in (
                "parent_root",
                "half_fold_root",
                "boeing_fold_root",
                "output_root",
            )
        ),
        "production runtime paths must be absolute",
    )
    return {
        "job_id": job_id,
        "job_name": os.environ["SLURM_JOB_NAME"],
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "account": os.environ["SLURM_JOB_ACCOUNT"],
        "node_list": os.environ.get("SLURM_JOB_NODELIST", ""),
        "cluster_name": os.environ.get("SLURM_CLUSTER_NAME", ""),
        "cpus_per_task": 32,
        "memory_mib_per_node": 131072,
        "gpu": "none",
        "runtime_paths": runtime_paths,
    }


def authenticate_release_root(root: Path) -> AuthenticatedRelease:
    root = root.resolve()
    _require(root.is_dir(), f"release root does not exist: {root}")
    complete_path = root / "AGGREGATE_COMPLETE.json"
    manifest_path = root / "aggregate_manifest.json"
    complete = _read_self_hashed_json(complete_path)
    manifest = _read_self_hashed_json(manifest_path)
    mode = str(complete.get("mode"))
    _require(mode in {"single_fold_authentication", "complete_five_fold_aggregate"}, "release mode is not allowed")
    report_name = (
        "single_fold_authentication_report.json"
        if mode == "single_fold_authentication"
        else "aggregate_summary.json"
    )
    expected_files = {
        "AGGREGATE_COMPLETE.json",
        "aggregate_manifest.json",
        "outer_family_summary.csv",
        report_name,
    }
    _require({path.name for path in root.iterdir()} == expected_files, "authenticated release file set changed")
    report_path = root / report_name
    report = _read_self_hashed_json(report_path)
    aggregator_commit = str(complete.get("aggregator_git_commit", ""))
    fold_commit = str(complete.get("fold_git_commit", ""))
    _require(
        aggregator_commit == TRUSTED_NUMERICAL_AGGREGATOR_GIT_COMMIT,
        "release aggregator commit is not the trusted numerical revision",
    )
    _require(len(fold_commit) == 40 and all(c in "0123456789abcdef" for c in fold_commit), "release fold commit is invalid")
    expected_report_schema = (
        aggregate.SINGLE_FOLD_SCHEMA
        if mode == "single_fold_authentication"
        else aggregate.AGGREGATE_SUMMARY_SCHEMA
    )
    for value, label in ((complete, "completion"), (manifest, "manifest"), (report, "report")):
        _require(value.get("experiment") == runner.EXPERIMENT, f"release {label} experiment changed")
        _require(value.get("status") == "completed", f"release {label} is not complete")
        _require(value.get("mode") == mode, f"release {label} mode changed")
        _require(value.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256, f"release {label} config changed")
        _require(
            value.get("aggregator_git_commit") == aggregator_commit,
            f"release {label} aggregator commit changed",
        )
        _require(value.get("fold_git_commit") == fold_commit, f"release {label} fold commit changed")
    source_centered_evidence = complete.get("source_centered_evidence")
    _require(
        manifest.get("source_centered_evidence") == source_centered_evidence
        and report.get("source_centered_evidence") == source_centered_evidence,
        "release source-centered evidence differs across completion/manifest/report",
    )
    _authenticate_release_source_evidence(
        source_centered_evidence,
        aggregator_git_commit=aggregator_commit,
        fold_git_commit=fold_commit,
    )
    _require(complete.get("schema") == aggregate.AGGREGATE_COMPLETE_SCHEMA, "aggregate completion schema changed")
    _require(manifest.get("schema") == aggregate.AGGREGATE_MANIFEST_SCHEMA, "aggregate manifest schema changed")
    _require(report.get("schema") == expected_report_schema, "aggregate report schema changed")
    _require(
        complete.get("aggregate_manifest_file") == manifest_path.name
        and complete.get("aggregate_manifest_file_sha256") == sha256_file(manifest_path)
        and complete.get("report_file") == report_path.name
        and complete.get("report_file_sha256") == sha256_file(report_path),
        "aggregate completion does not bind release files",
    )
    _require(
        manifest.get("report_file") == report_path.name
        and manifest.get("report_file_sha256") == sha256_file(report_path)
        and manifest.get("outer_family_summary_file") == "outer_family_summary.csv"
        and manifest.get("outer_family_summary_file_sha256")
        == sha256_file(root / "outer_family_summary.csv"),
        "aggregate manifest does not bind report/table",
    )
    source_rows = manifest.get("source_folds")
    _require(isinstance(source_rows, list) and source_rows, "release source-fold list is missing")
    source_folds: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        _require(isinstance(row, Mapping), "release source-fold row is invalid")
        family = str(row.get("outer_family", ""))
        _require(family in runner.FAMILY_ORDER and family not in source_folds, "release source-fold population changed")
        source_folds[family] = dict(row)
    _require(tuple(report.get("outer_families", ())) == tuple(source_folds), "release report/source-fold order changed")
    if mode == "single_fold_authentication":
        _require(tuple(source_folds) == ("half_cylinder",), "single-fold release is not the preregistered first family")
    else:
        _require(tuple(source_folds) == tuple(runner.FAMILY_ORDER), "complete release does not contain all five families")
    evidence = tuple(
        _file_row(root / name, f"authenticated_release:{root.name}:{name}")
        for name in sorted(expected_files)
    )
    return AuthenticatedRelease(
        root=root,
        mode=mode,
        aggregator_git_commit=aggregator_commit,
        fold_git_commit=fold_commit,
        manifest=manifest,
        report=report,
        source_centered_evidence=source_centered_evidence,
        source_folds=source_folds,
        evidence=evidence,
    )


def _release_record_for_family(
    releases: Sequence[AuthenticatedRelease], family: str
) -> Mapping[str, Any]:
    rows = [release.source_folds[family] for release in releases if family in release.source_folds]
    _require(len(rows) == 1, f"required family must be covered by exactly one release: {family}")
    return rows[0]


def authenticate_fold_root(
    root: Path,
    *,
    expected_family: str,
    release_record: Mapping[str, Any],
    expected_fold_commit: str,
) -> AuthenticatedFold:
    root = root.resolve()
    _require(root.is_dir(), f"fold root does not exist: {root}")
    _require({path.name for path in root.iterdir()} == set(runner.REQUIRED_FOLD_FILES), "fold file set changed")
    result_path = root / "result_manifest.json"
    completion_path = root / "RUN_COMPLETE.json"
    result = _read_self_hashed_json(result_path)
    completion = _read_self_hashed_json(completion_path)
    for value, label in ((result, "result"), (completion, "completion")):
        _require(value.get("experiment") == runner.EXPERIMENT, f"fold {label} experiment changed")
        _require(value.get("outer_family") == expected_family, f"fold {label} family changed")
        _require(value.get("git_commit") == expected_fold_commit, f"fold {label} commit changed")
        _require(value.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256, f"fold {label} config changed")
    _require(result.get("schema") == runner.RESULT_SCHEMA and result.get("status") == "completed", "fold result schema/status changed")
    _require(completion.get("schema") == runner.COMPLETE_SCHEMA, "fold completion schema changed")
    _require(
        completion.get("result_manifest_file_sha256") == sha256_file(result_path)
        and completion.get("result_manifest_content_sha256") == result.get("content_sha256"),
        "fold completion does not bind result",
    )
    _require(
        release_record.get("outer_family") == expected_family
        and Path(str(release_record.get("run_directory", ""))).name == root.name
        and release_record.get("completion_file_sha256") == sha256_file(completion_path)
        and release_record.get("result_manifest_file_sha256") == sha256_file(result_path),
        "authenticated release does not bind the supplied fold",
    )
    artifacts = result.get("artifacts")
    _require(isinstance(artifacts, Mapping), "fold artifact map is missing")
    evidence = [_file_row(result_path, f"fold:{expected_family}:result_manifest")]
    evidence.append(_file_row(completion_path, f"fold:{expected_family}:RUN_COMPLETE"))
    for name in runner.REQUIRED_FOLD_FILES:
        if name in {"result_manifest.json", "RUN_COMPLETE.json"}:
            continue
        record = artifacts.get(name)
        path = root / name
        _require(isinstance(record, Mapping), f"fold artifact is not result-bound: {name}")
        _require(
            int(record.get("size_bytes", -1)) == path.stat().st_size
            and record.get("sha256") == sha256_file(path),
            f"fold artifact identity changed: {name}",
        )
        evidence.append(_file_row(path, f"fold:{expected_family}:{name}"))
    selected = _read_self_hashed_json(root / "selected_candidate.json")
    prediction_manifest = _read_self_hashed_json(root / "outer_prediction_manifest.json")
    candidate = selected.get("candidate")
    _require(isinstance(candidate, Mapping), "fold selected candidate is missing")
    _require(
        selected.get("schema") == runner.SELECTED_SCHEMA
        and selected.get("experiment") == runner.EXPERIMENT
        and selected.get("outer_family") == expected_family
        and selected.get("git_commit") == expected_fold_commit
        and selected.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256
        and int(selected.get("candidate_count", -1)) == runner.FROZEN_CANDIDATE_COUNT,
        "fold selected-candidate provenance changed",
    )
    _require(
        prediction_manifest.get("schema") == runner.PREDICTION_MANIFEST_SCHEMA
        and prediction_manifest.get("prediction_schema") == runner.PREDICTION_SCHEMA
        and prediction_manifest.get("experiment") == runner.EXPERIMENT
        and prediction_manifest.get("outer_family") == expected_family
        and prediction_manifest.get("git_commit") == expected_fold_commit
        and prediction_manifest.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256
        and prediction_manifest.get("valid_labels_opened") is False
        and prediction_manifest.get("reference_labels_all_opened") is False,
        "fold prediction provenance changed",
    )
    _require(
        prediction_manifest.get("selected_candidate") == candidate
        and result.get("selected_candidate") == candidate,
        "candidate differs across selected/prediction/result artifacts",
    )
    file_record = prediction_manifest.get("prediction_file")
    records = prediction_manifest.get("arrays")
    _require(
        isinstance(file_record, Mapping)
        and file_record.get("path") == "outer_predictions.npz"
        and file_record.get("sha256") == sha256_file(root / "outer_predictions.npz")
        and int(file_record.get("size_bytes", -1)) == (root / "outer_predictions.npz").stat().st_size,
        "prediction file binding changed",
    )
    _require(
        isinstance(records, Mapping)
        and set(records) == set(runner.PREDICTION_DTYPES),
        "prediction array manifest member set changed",
    )
    return AuthenticatedFold(
        root=root,
        outer_family=expected_family,
        git_commit=expected_fold_commit,
        result=result,
        prediction_manifest=prediction_manifest,
        candidate=dict(candidate),
        evidence=tuple(evidence),
    )


def load_prediction_groups(
    folds: Mapping[str, AuthenticatedFold],
) -> dict[str, PredictionGroup]:
    groups: dict[str, PredictionGroup] = {}
    for family in REQUIRED_FAMILIES:
        fold = folds[family]
        records = fold.prediction_manifest["arrays"]
        file_record = fold.prediction_manifest["prediction_file"]
        arrays, artifact_sha = runner._verify_npz_arrays(  # type: ignore[attr-defined]
            fold.root / "outer_predictions.npz",
            file_record=file_record,
            records=records,
        )
        _require(artifact_sha == file_record["sha256"], "prediction archive SHA changed")
        unique_count = int(fold.prediction_manifest.get("unique_row_count", -1))
        valid_count = int(fold.prediction_manifest.get("valid_row_count", -1))
        for name, dtype in runner.UNIQUE_PREDICTION_DTYPES.items():
            _require(arrays[name].dtype == dtype and arrays[name].shape == (unique_count,), f"unique prediction dtype/shape changed: {name}")
        for name, dtype in runner.VALID_PREDICTION_DTYPES.items():
            _require(arrays[name].dtype == dtype and arrays[name].shape == (valid_count,), f"valid prediction dtype/shape changed: {name}")
        expected_datasets = tuple(
            dataset for dataset in DATASETS if DATASET_TO_FAMILY[dataset] == family
        )
        for dataset in expected_datasets:
            unique_mask = (
                (arrays["unique_dataset"] == dataset)
                & (arrays["unique_source_ordinal"] == SOURCE_ORDINAL)
            )
            valid_mask = (
                (arrays["valid_dataset"] == dataset)
                & (arrays["valid_source_ordinal"] == SOURCE_ORDINAL)
            )
            _require(int(unique_mask.sum()) == CENTER_COUNT and valid_mask.any(), f"fixed-source prediction group is incomplete: {dataset}")
            unique = {
                name: np.asarray(arrays[name][unique_mask])
                for name in runner.UNIQUE_PREDICTION_DTYPES
            }
            valid = {
                name: np.asarray(arrays[name][valid_mask])
                for name in runner.VALID_PREDICTION_DTYPES
            }
            _require(
                len(np.unique(unique["unique_source_index"])) == 1
                and len(np.unique(valid["valid_source_index"])) == 1
                and int(unique["unique_source_index"][0]) == int(valid["valid_source_index"][0]),
                f"fixed-source prediction source index changed: {dataset}",
            )
            _require(dataset not in groups, f"duplicate prediction group: {dataset}")
            groups[dataset] = PredictionGroup(
                dataset=dataset,
                outer_family=family,
                candidate=fold.candidate,
                unique=unique,
                valid=valid,
            )
    _require(tuple(groups) == DATASETS, "four fixed-source prediction groups changed")
    return groups


def read_producer_metric_rows(
    folds: Mapping[str, AuthenticatedFold],
    groups: Mapping[str, PredictionGroup],
) -> dict[tuple[str, str], Mapping[str, str]]:
    selected: dict[tuple[str, str], Mapping[str, str]] = {}
    for family, fold in folds.items():
        with (fold.root / "outer_group_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as source:
            reader = csv.DictReader(source)
            _require(
                tuple(reader.fieldnames or ()) == tuple(runner.OUTER_METRIC_FIELDS),
                f"producer metric field contract changed: {family}",
            )
            rows = list(reader)
        for row in rows:
            if int(row["source_ordinal"]) != SOURCE_ORDINAL:
                continue
            if row["dataset"] not in DATASETS or DATASET_TO_FAMILY[row["dataset"]] != family:
                continue
            if row["arm"] != "source_centered_paired_centers":
                continue
            population = row["population"]
            if population not in {"combined_valid_unique_centers", "all_parent_valid_rows"}:
                continue
            key = (row["dataset"], population)
            _require(key not in selected, f"duplicate producer metric row: {key}")
            group = groups[row["dataset"]]
            candidate = group.candidate
            expected_template_eligible = (
                "1" if population == "all_parent_valid_rows" else "0"
            )
            _require(
                row["outer_family"] == family
                and row["dataset"] == group.dataset
                and row["source_ordinal"] == str(SOURCE_ORDINAL)
                and row["source_index"]
                == str(int(group.unique["unique_source_index"][0]))
                and row["arm"] == "source_centered_paired_centers"
                and row["population"] == population
                and row["template_success_eligible"] == expected_template_eligible
                and row["candidate_id"] == str(candidate["candidate_id"])
                and row["representation"] == str(candidate["representation"])
                and int(row["k"]) == int(candidate["k"])
                and row["decision_rule"] == str(candidate["decision_rule"])
                and abs(float(row["sigma"]) - float(candidate["sigma"])) <= 1.0e-12
                and abs(float(row["weight"]) - float(candidate["weight"])) <= 1.0e-12
                and abs(
                    float(row["decision_value"])
                    - float(candidate["decision_value"])
                )
                <= 1.0e-12,
                f"producer metric identity changed: {key}",
            )
            selected[key] = row
    expected = {
        (dataset, population)
        for dataset in DATASETS
        for population in ("combined_valid_unique_centers", "all_parent_valid_rows")
    }
    _require(set(selected) == expected, "producer metric rows are incomplete")
    return selected


def _compare_metrics(
    observed: Mapping[str, Any], expected: Mapping[str, str], *, label: str
) -> None:
    for field in METRIC_INTEGER_FIELDS:
        _require(int(observed[field]) == int(expected[field]), f"{label} metric mismatch: {field}")
    for field in METRIC_FLOAT_FIELDS:
        left = float(observed[field])
        right = float(expected[field])
        if np.isnan(left) or np.isnan(right):
            _require(np.isnan(left) and np.isnan(right), f"{label} metric mismatch: {field}")
        else:
            _require(abs(left - right) <= 1.0e-12, f"{label} metric mismatch: {field}")


def _figure_contract() -> dict[str, Any]:
    return {
        "core_conclusion": (
            "At fixed source ordinal 2, one source-centered paired-scale template "
            "prediction has flow-specific spatial agreement and error structure "
            "relative to IVD p95."
        ),
        "results_level_question": (
            "Where does the paired-center prediction agree or disagree with IVD p95 "
            "in Cylinder3D Re160, Re640, Re6400, and Boeing 747?"
        ),
        "archetype": "image plate + quantification",
        "backend": "Python/matplotlib",
        "panel_map": {
            "a": "unchanged IVD-p95 background plus fixed first 120 legacy and first 120 expanded parent pathlines",
            "b": "one paired_prediction for all combined-valid unique centers",
            "c": "TP/FP/FN/TN for the exact same centers in the same order",
        },
        "primary_reported_not_plotted": (
            "valid_paired_prediction on all parent-valid rows, with both-valid centers "
            "represented once per valid block"
        ),
        "selection": "four flows, source ordinal 2, and both pathline prefixes were frozen before result access",
        "uncertainty": "none; each figure is one preregistered source timeslice",
        "reviewer_risks": [
            "source-centered mean uses target-flow unlabeled velocity and is transductive",
            "the four flows are exposed-development data",
            "combined-valid coverage differs by flow",
            "panel-a pathlines are block context and are not the panel-b population",
            "one source cannot replace complete five-family statistics",
        ],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _scene_manifest(
    *,
    scene_path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    dataset: str,
    reporting_identity: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema": "pathline_template_matching.source_centered_paired_scale_combined_scene_manifest.v1",
        "scene_schema": SCENE_SCHEMA,
        "experiment": EXPERIMENT,
        **reporting_identity,
        "dataset": dataset,
        "source_ordinal": SOURCE_ORDINAL,
        "source_index": int(metadata["source_index"]),
        "scene_file": {
            "path": scene_path.name,
            "size_bytes": int(scene_path.stat().st_size),
            "sha256": sha256_file(scene_path),
        },
        "ordered_array_names": list(SCENE_ARRAY_NAMES),
        "arrays": runner.early._array_manifest(arrays),
        "metadata": _json_safe(metadata),
    }
    return runner._manifest(manifest)


def _load_combined_scene(scene_path: Path, manifest_path: Path) -> CombinedCenterScene:
    manifest = _read_self_hashed_json(manifest_path)
    _require(
        manifest.get("scene_schema") == SCENE_SCHEMA
        and tuple(manifest.get("ordered_array_names", ())) == SCENE_ARRAY_NAMES,
        "combined scene manifest changed",
    )
    records = manifest.get("arrays")
    _require(
        isinstance(records, Mapping) and set(records) == set(SCENE_ARRAY_NAMES),
        "combined scene array records changed",
    )
    arrays, digest = runner._verify_npz_arrays(  # type: ignore[attr-defined]
        scene_path,
        file_record=manifest["scene_file"],
        records=records,
    )
    _require(digest == manifest["scene_file"]["sha256"], "combined scene archive changed")
    metadata = json.loads(str(np.asarray(arrays["metadata_json"]).reshape(())))
    _require(isinstance(metadata, dict), "combined scene metadata is invalid")
    return CombinedCenterScene(
        dataset=str(metadata["dataset"]),
        title=str(metadata["display_title"]),
        source_index=int(metadata["source_index"]),
        bounds=np.asarray(arrays["bounds"]),
        seeds=np.asarray(arrays["seeds"]),
        reference=np.asarray(arrays["reference"], dtype=np.bool_),
        prediction=np.asarray(arrays["prediction"], dtype=np.bool_),
        center_seed_index=np.asarray(arrays["center_seed_index"], dtype=np.int64),
        paired_score=np.asarray(arrays["paired_score"], dtype=np.float64),
        legacy_valid=np.asarray(arrays["legacy_valid"], dtype=np.bool_),
        expanded_valid=np.asarray(arrays["expanded_valid"], dtype=np.bool_),
        display_pathlines=np.asarray(arrays["display_pathlines"]),
        display_pathline_block_index=np.asarray(
            arrays["display_pathline_block_index"], dtype=np.int8
        ),
        ivd_mesh_vertices=np.asarray(arrays["ivd_mesh_vertices"]),
        ivd_mesh_faces=np.asarray(arrays["ivd_mesh_faces"]),
        ivd_mesh_normals=np.asarray(arrays["ivd_mesh_normals"]),
        ivd_mesh_values=np.asarray(arrays["ivd_mesh_values"]),
        ivd_mesh_level=np.asarray(arrays["ivd_mesh_level"]),
    )


def _camera_core(render_metadata: Mapping[str, Any]) -> dict[str, Any]:
    camera = render_metadata.get("camera")
    _require(isinstance(camera, Mapping), "render metadata camera is missing")
    return {
        key: camera[key]
        for key in (
            "projection",
            "elevation_degrees",
            "azimuth_degrees",
            "physical_bounds",
            "box_aspect",
        )
    }


def render_bundle(
    *,
    parent_root: Path,
    release_roots: Sequence[Path],
    half_fold_root: Path,
    boeing_fold_root: Path,
    output_root: Path,
    expected_reporting_commit: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"immutable output directory already exists: {output_root}")
    config = _authenticate_config()
    reporting_identity = _authenticate_reporting_checkout(expected_reporting_commit)
    reporting_identity = {
        **reporting_identity,
        "slurm_runtime": _authenticate_slurm_runtime(
            parent_root=parent_root,
            release_roots=release_roots,
            half_fold_root=half_fold_root,
            boeing_fold_root=boeing_fold_root,
            output_root=output_root,
        ),
    }
    _require(release_roots, "at least one authenticated release root is required")
    resolved_release_roots = tuple(path.resolve() for path in release_roots)
    _require(len(set(resolved_release_roots)) == len(resolved_release_roots), "duplicate release root")
    releases = tuple(authenticate_release_root(path) for path in resolved_release_roots)
    fold_commits = {release.fold_git_commit for release in releases}
    _require(len(fold_commits) == 1, "authenticated releases use different fold commits")
    fold_commit = next(iter(fold_commits))
    method_release_authentication = [
        {
            "root": str(release.root),
            "mode": release.mode,
            "aggregator_git_commit": release.aggregator_git_commit,
            "fold_git_commit": release.fold_git_commit,
            "source_centered_evidence_sha256": canonical_json_sha256(
                release.source_centered_evidence
            ),
        }
        for release in releases
    ]
    fold_roots = {
        "half_cylinder": half_fold_root,
        "boeing_747": boeing_fold_root,
    }
    folds = {
        family: authenticate_fold_root(
            fold_roots[family],
            expected_family=family,
            release_record=_release_record_for_family(releases, family),
            expected_fold_commit=fold_commit,
        )
        for family in REQUIRED_FAMILIES
    }
    parent_scenes, parent_evidence = _authenticate_parent(parent_root.resolve())
    expected_parent_keys = {(dataset, block) for dataset in DATASETS for block in BLOCKS}
    _require(set(parent_scenes) == expected_parent_keys, "parent scene population changed")

    input_rows: list[Mapping[str, Any]] = list(parent_evidence)
    for release in releases:
        input_rows.extend(release.evidence)
    for family in REQUIRED_FAMILIES:
        input_rows.extend(folds[family].evidence)
    input_rows.append(_file_row(CONFIG_PATH, "frozen_report_config"))
    input_rows.extend(
        _file_row(REPOSITORY_ROOT / relative, f"reporting_dependency:{relative}")
        for relative in REPORTING_DEPENDENCY_RELATIVE_PATHS
    )
    by_path: dict[str, dict[str, Any]] = {}
    for row in input_rows:
        key = str(row["path"])
        if key in by_path:
            _require(
                by_path[key]["size_bytes"] == row["size_bytes"]
                and by_path[key]["sha256"] == row["sha256"],
                "duplicate input path has inconsistent identity",
            )
            roles = set(str(by_path[key]["role"]).split(" | "))
            roles.add(str(row["role"]))
            by_path[key]["role"] = " | ".join(sorted(roles))
        else:
            by_path[key] = dict(row)
    opaque_rows = [by_path[key] for key in sorted(by_path)]

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "scenes").mkdir()
    (output_root / "figures").mkdir()
    frozen_copy = output_root / "frozen_config.yaml"
    shutil.copyfile(CONFIG_PATH, frozen_copy)
    _require(sha256_file(frozen_copy) == CONFIG_SHA256, "frozen config copy changed")
    input_manifest = runner._manifest(
        {
            "schema": "pathline_template_matching.source_centered_paired_scale_visualization_input.v1",
            "experiment": EXPERIMENT,
            **reporting_identity,
            "method_experiment": runner.EXPERIMENT,
            "method_config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "method_fold_git_commit": fold_commit,
            "method_release_authentication": method_release_authentication,
            "parent_scene_experiment": PARENT_SCENE_EXPERIMENT,
            "parent_scene_git_commit": PARENT_SCENE_COMMIT,
            "report_config_status": config["status"],
            "source_selection": "fixed source ordinal 2",
            "release_modes": [release.mode for release in releases],
            "npz_array_access_before_manifest_write": False,
            "files": opaque_rows,
            "files_content_sha256": canonical_json_sha256(opaque_rows),
        }
    )
    _atomic_json(output_root / "input_manifest.json", input_manifest)
    contract = runner._manifest(_figure_contract())
    _atomic_json(output_root / "figure_contract.json", contract)

    # The first NPZ member access in this reporting run occurs only after the
    # complete immutable input manifest and figure contract have been written.
    groups = load_prediction_groups(folds)
    producer_metrics = read_producer_metric_rows(folds, groups)
    metric_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        group = groups[dataset]
        parent_by_block: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
        parent_render_by_block: dict[str, Mapping[str, Any]] = {}
        for block in BLOCKS:
            paths = parent_scenes[(dataset, block)]
            parent_by_block[block] = _load_parent_scene(paths)
            parent_render_by_block[block] = _read_parent_self_hashed_json(
                paths["render"], "metadata_content_sha256"
            )["renderer"]
        legacy_metadata, legacy_arrays = parent_by_block["legacy_2_1"]
        expanded_metadata, expanded_arrays = parent_by_block["expanded_3_1"]
        combined = combine_parent_block_scenes(
            legacy_metadata=legacy_metadata,
            legacy_arrays=legacy_arrays,
            expanded_metadata=expanded_metadata,
            expanded_arrays=expanded_arrays,
            unique_prediction=group.unique,
            title=DISPLAY_NAMES[dataset],
        )
        projection = bind_valid_projection(
            legacy_metadata=legacy_metadata,
            legacy_arrays=legacy_arrays,
            expanded_metadata=expanded_metadata,
            expanded_arrays=expanded_arrays,
            unique_prediction=group.unique,
            valid_prediction=group.valid,
        )
        center_metrics = _metric_values(
            combined.reference, combined.prediction, combined.paired_score
        )
        projection_metrics = _metric_values(
            projection.reference, projection.prediction, projection.score
        )
        _compare_metrics(
            center_metrics,
            producer_metrics[(dataset, "combined_valid_unique_centers")],
            label=f"{dataset}/combined-valid-center",
        )
        _compare_metrics(
            projection_metrics,
            producer_metrics[(dataset, "all_parent_valid_rows")],
            label=f"{dataset}/valid-projection",
        )
        legacy_valid = np.asarray(group.unique["legacy_valid"], dtype=np.bool_)
        expanded_valid = np.asarray(group.unique["expanded_valid"], dtype=np.bool_)
        both = legacy_valid & expanded_valid
        legacy_only = legacy_valid & ~expanded_valid
        expanded_only = ~legacy_valid & expanded_valid
        neither = ~legacy_valid & ~expanded_valid
        availability = {
            "combined_valid_center_count": int((legacy_valid | expanded_valid).sum()),
            "unique_center_combined_coverage": float(
                (legacy_valid | expanded_valid).mean()
            ),
            "both_valid_count": int(both.sum()),
            "legacy_only_count": int(legacy_only.sum()),
            "expanded_only_count": int(expanded_only.sum()),
            "neither_valid_count": int(neither.sum()),
            "legacy_valid_row_count": int(len(legacy_arrays["reference"])),
            "expanded_valid_row_count": int(len(expanded_arrays["reference"])),
            "valid_projection_row_count": int(len(projection.reference)),
        }
        for field in AVAILABILITY_FIELDS:
            expected_value = producer_metrics[(dataset, "all_parent_valid_rows")][field]
            observed_value = availability[field]
            if field == "unique_center_combined_coverage":
                _require(abs(float(observed_value) - float(expected_value)) <= 1.0e-12, f"availability mismatch: {dataset}/{field}")
            else:
                _require(int(observed_value) == int(expected_value), f"availability mismatch: {dataset}/{field}")
        candidate = dict(group.candidate)
        scene_metadata = {
            "schema": SCENE_SCHEMA,
            "experiment": EXPERIMENT,
            **reporting_identity,
            "dataset": dataset,
            "display_title": DISPLAY_NAMES[dataset],
            "outer_family": DATASET_TO_FAMILY[dataset],
            "source_ordinal": SOURCE_ORDINAL,
            "source_index": combined.source_index,
            "method_experiment": runner.EXPERIMENT,
            "method_config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "method_fold_git_commit": fold_commit,
            "method_release_authentication": method_release_authentication,
            "candidate": candidate,
            "prediction_semantics": "one paired_prediction per combined-valid unique center",
            "exact_join": {
                "combined_center_count": int(len(combined.center_seed_index)),
                "valid_projection_row_count": int(len(projection.reference)),
                "missing_count": 0,
                "extra_count": 0,
                "duplicate_count": 0,
                "unique_center_reordered_count": 0,
                "overlap_seed_or_reference_mismatch_count": 0,
            },
            "parent_scene_files": {
                block: {
                    "scene_npz_sha256": sha256_file(parent_scenes[(dataset, block)]["npz"]),
                    "scene_manifest_sha256": sha256_file(parent_scenes[(dataset, block)]["manifest"]),
                    "render_metadata_sha256": sha256_file(parent_scenes[(dataset, block)]["render"]),
                    "pathline_prefix_count": 120,
                }
                for block in BLOCKS
            },
            "availability": availability,
            "center_metrics": _json_safe(center_metrics),
            "primary_valid_projection_metrics": _json_safe(projection_metrics),
            "figure_interpretation": contract,
            "formal_confirmation": False,
        }
        metadata_text = json.dumps(
            _json_safe(scene_metadata),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        arrays = scene_arrays(combined, metadata_text)
        scene_stem = output_root / "scenes" / f"{dataset}_source_ordinal_2_paired_centers"
        scene_path = scene_stem.with_suffix(".scene.npz")
        scene_manifest_path = scene_stem.with_suffix(".scene.json")
        _atomic_npz(scene_path, arrays)
        scene_manifest = _scene_manifest(
            scene_path=scene_path,
            arrays=arrays,
            metadata=scene_metadata,
            dataset=dataset,
            reporting_identity=reporting_identity,
        )
        _atomic_json(scene_manifest_path, scene_manifest)
        loaded_scene = _load_combined_scene(scene_path, scene_manifest_path)
        figure_stem = output_root / "figures" / f"{dataset}_source_ordinal_2_paired_center_triptych"
        png_path = figure_stem.with_suffix(".png")
        pdf_path = figure_stem.with_suffix(".pdf")
        svg_path = figure_stem.with_suffix(".svg")
        alignment_path = figure_stem.with_suffix(".alignment.json")
        render_metadata = render_source_centered_triptych(
            loaded_scene,
            png_path=png_path,
            pdf_path=pdf_path,
            svg_path=svg_path,
            alignment_path=alignment_path,
            view=DATASET_VIEWS[dataset],
            dpi=360,
        )
        parent_camera_legacy = _camera_core(parent_render_by_block["legacy_2_1"])
        parent_camera_expanded = _camera_core(parent_render_by_block["expanded_3_1"])
        rendered_camera = _camera_core(render_metadata)
        _require(parent_camera_legacy == parent_camera_expanded == rendered_camera, f"camera/bounds changed: {dataset}")
        _require(render_metadata["panel_order"] == list(PANEL_TITLES), "panel order changed")
        render_metadata_value = runner._manifest(
            {
                **render_metadata,
                "experiment": EXPERIMENT,
                **reporting_identity,
                "scene_npz": str(scene_path.relative_to(output_root)).replace("\\", "/"),
                "scene_npz_sha256": sha256_file(scene_path),
                "scene_manifest": str(scene_manifest_path.relative_to(output_root)).replace("\\", "/"),
                "scene_manifest_sha256": sha256_file(scene_manifest_path),
                "parent_camera_exact": True,
                "primary_valid_projection_metrics_reported_not_plotted": True,
            }
        )
        render_metadata_path = figure_stem.with_suffix(".render.json")
        _atomic_json(render_metadata_path, render_metadata_value)

        metric_row: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "dataset": dataset,
            "display_name": DISPLAY_NAMES[dataset],
            "outer_family": DATASET_TO_FAMILY[dataset],
            "source_ordinal": SOURCE_ORDINAL,
            "source_index": combined.source_index,
            "candidate_id": candidate["candidate_id"],
            "representation": candidate["representation"],
            "k": int(candidate["k"]),
            "sigma": float(candidate["sigma"]),
            "weight": float(candidate["weight"]),
            "decision_rule": candidate["decision_rule"],
            "decision_value": float(candidate["decision_value"]),
            **availability,
        }
        for prefix, values in (
            ("center", center_metrics),
            ("primary_valid_projection", projection_metrics),
        ):
            for field in (*METRIC_INTEGER_FIELDS, *METRIC_FLOAT_FIELDS):
                metric_row[f"{prefix}_{field}"] = values[field]
        metric_rows.append(metric_row)
        relative = lambda path: str(path.relative_to(output_root)).replace("\\", "/")
        figure_rows.append(
            {
                "dataset": dataset,
                "source_ordinal": SOURCE_ORDINAL,
                "population": "combined_valid_unique_centers",
                "scene_npz": relative(scene_path),
                "scene_npz_sha256": sha256_file(scene_path),
                "scene_manifest": relative(scene_manifest_path),
                "scene_manifest_sha256": sha256_file(scene_manifest_path),
                "png": relative(png_path),
                "png_sha256": sha256_file(png_path),
                "pdf": relative(pdf_path),
                "pdf_sha256": sha256_file(pdf_path),
                "svg": relative(svg_path),
                "svg_sha256": sha256_file(svg_path),
                "alignment": relative(alignment_path),
                "alignment_sha256": sha256_file(alignment_path),
                "render_metadata": relative(render_metadata_path),
                "render_metadata_sha256": sha256_file(render_metadata_path),
                "pending_local_qa": {
                    "svg_text_audit": relative(figure_stem.with_suffix(".svg-text-audit.json")),
                    "pdf_text_audit": relative(figure_stem.with_suffix(".pdf-text-audit.json")),
                    "collision_audit": relative(figure_stem.with_suffix(".collision-audit.json")),
                    "collision_overlay_pdf": relative(figure_stem.with_suffix(".collision-overlay.pdf")),
                },
                "metrics": _json_safe(metric_row),
            }
        )

    _require(len(metric_rows) == len(figure_rows) == 4, "exactly four figures are required")
    _atomic_csv(output_root / "per_figure_metrics.csv", metric_rows, tuple(metric_rows[0]))
    visualization_manifest = runner._manifest(
        {
            "schema": "pathline_template_matching.source_centered_paired_scale_visualization.v1",
            "experiment": EXPERIMENT,
            **reporting_identity,
            "method_experiment": runner.EXPERIMENT,
            "method_config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "method_fold_git_commit": fold_commit,
            "method_release_authentication": method_release_authentication,
            "evidence_scope": "family-held-out exposed-development fixed-source reporting",
            "formal_confirmation": False,
            "source_selection": "fixed source ordinal 2; no result-based selection",
            "figure_count": 4,
            "unique_key": ["dataset"],
            "classification_unit": "combined-valid unique center",
            "scale_blocks_are_context_not_separate_classifiers": True,
            "primary_valid_projection_metrics_reported_not_plotted": True,
            "entries": figure_rows,
        }
    )
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
    result = runner._manifest(
        {
            "schema": "pathline_template_matching.source_centered_paired_scale_visualization_result.v1",
            "experiment": EXPERIMENT,
            **reporting_identity,
            "status": "completed_pending_local_rendered_qa",
            "formal_confirmation": False,
            "method_experiment": runner.EXPERIMENT,
            "method_config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "method_fold_git_commit": fold_commit,
            "method_release_authentication": method_release_authentication,
            "figure_count": 4,
            "combined_valid_center_count": int(
                sum(row["combined_valid_center_count"] for row in metric_rows)
            ),
            "primary_valid_projection_row_count": int(
                sum(row["valid_projection_row_count"] for row in metric_rows)
            ),
            "input_manifest_file_sha256": sha256_file(output_root / "input_manifest.json"),
            "visualization_manifest_file_sha256": sha256_file(
                output_root / "visualization_manifest.json"
            ),
            "per_figure_metrics_file_sha256": sha256_file(
                output_root / "per_figure_metrics.csv"
            ),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifacts_content_sha256": canonical_json_sha256(artifacts),
            "local_qa_required_before_delivery": True,
        }
    )
    _atomic_json(output_root / "result_manifest.json", result)
    complete = runner._manifest(
        {
            "schema": "pathline_template_matching.source_centered_paired_scale_visualization_run_complete.v1",
            "experiment": EXPERIMENT,
            **reporting_identity,
            "status": "complete_pending_local_rendered_qa",
            "method_experiment": runner.EXPERIMENT,
            "method_config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "method_fold_git_commit": fold_commit,
            "method_release_authentication": method_release_authentication,
            "figure_count": 4,
            "result_manifest_file_sha256": sha256_file(output_root / "result_manifest.json"),
            "result_manifest_content_sha256": result["content_sha256"],
        }
    )
    _atomic_json(output_root / "RUN_COMPLETE.json", complete)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument(
        "--release-root",
        type=Path,
        action="append",
        required=True,
        help="authenticated aggregate release root; repeat only when two single releases are required",
    )
    parser.add_argument("--half-fold-root", type=Path, required=True)
    parser.add_argument("--boeing-fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-reporting-commit", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    render_bundle(
        parent_root=arguments.parent_root,
        release_roots=arguments.release_root,
        half_fold_root=arguments.half_fold_root,
        boeing_fold_root=arguments.boeing_fold_root,
        output_root=arguments.output_root,
        expected_reporting_commit=arguments.expected_reporting_commit,
    )


if __name__ == "__main__":
    main()

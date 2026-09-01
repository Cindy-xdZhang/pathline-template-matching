#!/usr/bin/env python3
"""Render four authenticated primary RankLikelihood triptychs.

All aggregate, fold, parent-scene, configuration, and reporting-source files
are authenticated as opaque files before ``input_manifest.json`` is closed.
Only then are the two required prediction archives and eight immutable parent
scene archives opened.  This reporter never opens a fold sidecar or label
member, never refits a model, and never switches from ``dual_histogram_llr``
to a control after seeing results.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.phase21_visualization import DATASET_VIEWS  # noqa: E402
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.source_centered_rank_likelihood_visualization import (  # noqa: E402
    BLOCKS,
    CENTER_COUNT,
    PANEL_TITLES,
    SCENE_ARRAY_NAMES,
    SCENE_SCHEMA,
    SOURCE_ORDINAL,
    CombinedCenterScene,
    bind_rank_likelihood_table_only_projection,
    bind_rank_likelihood_valid_projection,
    combine_rank_likelihood_parent_scenes,
    publish_bytes_without_overwrite,
    render_source_rank_likelihood_triptych,
    scene_arrays,
)
from scripts import (  # noqa: E402
    aggregate_verify_source_centered_rank_likelihood_template_1_1 as aggregate,
)
from scripts import (  # noqa: E402
    run_verify_source_centered_rank_likelihood_template_1_1 as runner,
)
from scripts.render_early_opposite_pair_kinematics_visualizations import (  # noqa: E402
    PARENT_SCENE_COMMIT,
    PARENT_SCENE_CONFIG_SHA256,
    PARENT_SCENE_EXPERIMENT,
    PARENT_SCENE_RESULT_MANIFEST_SHA256,
    _authenticate_parent,
    _load_parent_scene,
    _read_self_hashed_json as _read_parent_self_hashed_json,
)
from scripts.run_verify_early_opposite_pair_kinematics_1_1 import (  # noqa: E402
    _atomic_csv as _atomic_csv_no_replace,
    _atomic_json as _atomic_json_no_replace,
    _atomic_npz as _atomic_npz_no_replace,
)


EXPERIMENT = "Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1"
CONFIG_PATH = REPOSITORY_ROOT / "config" / f"{EXPERIMENT}.yaml"
CONFIG_SHA256 = "a464761eb8df3ebf43d55b6f05eee2e90302be770b43f3e5e75a5944f13ff9a3"
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
EXPECTED_SOURCE_INDEX = {
    "cylinder3d": 68,
    "halfcylinderRe640": 18,
    "halfcylinderRe6400": 68,
    "boeing747": 100,
}
REQUIRED_FAMILIES = ("half_cylinder", "boeing_747")
TABLE_ONLY_ARMS = ("negative_ecdf", "direct_rank_mean_top5")
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
    "config/Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1.yaml",
    "scripts/render_source_centered_rank_likelihood_template_visualizations.py",
    "scripts/audit_source_centered_rank_likelihood_template_visualizations.py",
    "ibex/other_source_centered_rank_likelihood_template_visualization_1.1.sh",
    "src/pathline_template_matching/source_centered_rank_likelihood_visualization.py",
    "src/pathline_template_matching/source_centered_visualization.py",
    "src/pathline_template_matching/visualization.py",
    "src/pathline_template_matching/phase21_pipeline.py",
    "src/pathline_template_matching/phase21_visualization.py",
    "src/pathline_template_matching/negative_tail_visualization.py",
    "src/pathline_template_matching/metrics.py",
    "src/pathline_template_matching/portable_flow.py",
    "scripts/run_verify_source_centered_rank_likelihood_template_1_1.py",
    "scripts/aggregate_verify_source_centered_rank_likelihood_template_1_1.py",
    "scripts/render_early_opposite_pair_kinematics_visualizations.py",
)
METHOD_INTERPRETATION_RELATIVE_PATHS = (
    "config/Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml",
    "config/Verify_SourceCenteredPairedScaleTemplate_1.1.yaml",
    "scripts/run_verify_source_centered_rank_likelihood_template_1_1.py",
    "scripts/aggregate_verify_source_centered_rank_likelihood_template_1_1.py",
    "scripts/run_verify_source_centered_paired_scale_template_1_1.py",
    "scripts/run_verify_early_opposite_pair_kinematics_1_1.py",
    "scripts/run_verify_scale_conditioned_retrieval_1_1.py",
    "src/pathline_template_matching/paired_scale_center_fusion.py",
    "src/pathline_template_matching/per_scale_negative_metric.py",
    "src/pathline_template_matching/nested_scale_validation.py",
    "src/pathline_template_matching/portable_flow.py",
    "src/pathline_template_matching/source_centered_sidecar.py",
    "src/pathline_template_matching/source_centered_rank_likelihood.py",
)
HISTORICAL_SOURCE_RELATIVE_PATHS = (
    "config/Verify_SourceCenteredPairedScaleTemplate_1.1.yaml",
    "scripts/prepare_verify_source_centered_paired_scale_template_1_1.py",
    "src/pathline_template_matching/arc_length_primitives.py",
    "src/pathline_template_matching/early_opposite_pair_kinematics.py",
    "src/pathline_template_matching/netcdf_io.py",
    "src/pathline_template_matching/portable_flow.py",
    "src/pathline_template_matching/seed_time_kinematic_sidecar.py",
    "src/pathline_template_matching/source_centered_seed_time_kinematics.py",
    "src/pathline_template_matching/source_centered_sidecar.py",
    "src/pathline_template_matching/vector_field.py",
)
SOURCE_INPUT_MANIFEST_CONTENT_SHA256 = (
    "6b8d5ad4eecdd8febe73d4518e9433907ae828b3e5c42a71077ebf32fa8a5532"
)
SOURCE_POPULATION_MANIFEST_CONTENT_SHA256 = (
    "cc0f1b36dbb067629423ea9c0194e032f4e7c474312a5d34f668cf8444d56880"
)
SOURCE_POPULATION_ROWS_CONTENT_SHA256 = (
    "d0c4ed0726fc66262104edaa2d0d26430e8530e3653ae7e56019fef340368231"
)
SOURCE_FILE_SHA256_CONTENT_SHA256 = (
    "fc030232c0596931bee09427d03c86f34cf01618416b146de5f6d5416b59ce33"
)
SOURCE_SIDECAR_COUNT = 32
SOURCE_VALID_PROJECTION_ROW_COUNT = 2_967_612

INPUT_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_visualization_input.v1"
)
SCENE_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_combined_scene_manifest.v1"
)
VISUALIZATION_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_visualization.v1"
)
RESULT_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_visualization_result.v1"
)
COMPLETE_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_visualization_run_complete.v1"
)


@dataclass(frozen=True, slots=True)
class AuthenticatedRelease:
    root: Path
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
    primary_candidate: Mapping[str, Any]
    selected_control: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PredictionGroup:
    dataset: str
    outer_family: str
    primary_candidate: Mapping[str, Any]
    selected_control: Mapping[str, Any]
    unique: Mapping[str, np.ndarray]
    valid: Mapping[str, np.ndarray]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _atomic_json(path: Path, value: Any) -> str:
    return _atomic_json_no_replace(path, value)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    return _atomic_npz_no_replace(path, arrays)


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> str:
    return _atomic_csv_no_replace(path, fieldnames, rows)


def _metric_values(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    """Replay the producer's exact single- and dual-class metric semantics."""

    return runner._classification_metric_values(labels, scores, predictions)


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_lower_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_absolute_machine_path(value: object) -> bool:
    text = str(value)
    return Path(text).is_absolute() or PurePosixPath(text).is_absolute()


def _require_disjoint_output_root(
    output_root: Path,
    input_roots: Sequence[Path],
) -> None:
    """Reject either direction of overlap with every immutable input root."""

    output = output_root.resolve()
    for input_root in input_roots:
        source = input_root.resolve()
        try:
            output.relative_to(source)
            output_within_source = True
        except ValueError:
            output_within_source = False
        try:
            source.relative_to(output)
            source_within_output = True
        except ValueError:
            source_within_output = False
        _require(
            not (output_within_source or source_within_output),
            f"output root overlaps immutable input root: {source}",
        )


def _file_row(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    _require(resolved.is_file(), f"missing authenticated input: {resolved}")
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _read_self_hashed_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing self-hashed JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be a mapping: {path}")
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    _require(
        _is_lower_sha256(claimed) and canonical_json_sha256(unsigned) == claimed,
        f"self hash changed: {path}",
    )
    return value


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


def _authenticate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file() and sha256_file(path) == CONFIG_SHA256, "frozen report config changed")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("experiment") == EXPERIMENT, "report config identity changed")
    _require(
        value.get("status")
        == "frozen_before_reading_any_rank_likelihood_prediction_metric_label_or_parent_scene_npz_member",
        "report freeze history changed",
    )
    method = value.get("method_parent")
    _require(isinstance(method, Mapping), "method parent contract is missing")
    _require(
        method.get("experiment") == runner.EXPERIMENT
        and method.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256
        and method.get("release_mode_required") == "complete_five_fold_aggregate"
        and tuple(method.get("required_outer_families", ())) == REQUIRED_FAMILIES
        and tuple(method.get("required_fold_files", ())) == runner.REQUIRED_FOLD_FILES,
        "method parent or 18-file fold contract changed",
    )
    _require(
        tuple(method.get("prediction_arrays", {}).get("unique", ()))
        == tuple(runner.UNIQUE_PREDICTION_DTYPES)
        and tuple(method.get("prediction_arrays", {}).get("valid_projection", ()))
        == tuple(runner.VALID_PREDICTION_DTYPES)
        and method.get("plotted_arm") == "dual_histogram_llr"
        and method.get("plotted_prediction_array") == "unique_primary_prediction"
        and method.get("plotted_score_array") == "unique_primary_spatial_score"
        and method.get("result_based_arm_switching") == "forbidden",
        "primary-only prediction contract changed",
    )
    parent = value.get("parent_scenes")
    _require(
        isinstance(parent, Mapping)
        and parent.get("experiment") == PARENT_SCENE_EXPERIMENT
        and parent.get("numerical_git_commit") == PARENT_SCENE_COMMIT
        and parent.get("config_sha256") == PARENT_SCENE_CONFIG_SHA256
        and parent.get("result_manifest_sha256") == PARENT_SCENE_RESULT_MANIFEST_SHA256,
        "parent scene identity changed",
    )
    query = value.get("query")
    _require(isinstance(query, Mapping) and int(query.get("source_ordinal", -1)) == SOURCE_ORDINAL, "source ordinal changed")
    datasets = tuple(query.get("datasets", ()))
    _require(
        tuple(row.get("id") for row in datasets if isinstance(row, Mapping)) == DATASETS
        and all(int(row["source_index"]) == EXPECTED_SOURCE_INDEX[str(row["id"])] for row in datasets),
        "four fixed flow/source identities changed",
    )
    _require(
        tuple(row.get("id") for row in query.get("scale_blocks", ()) if isinstance(row, Mapping)) == BLOCKS,
        "scale-block order changed",
    )
    figure = value.get("figure_contract")
    _require(
        isinstance(figure, Mapping)
        and int(figure.get("expected_figure_count", -1)) == 4
        and tuple(figure.get("panel_titles", ())) == PANEL_TITLES
        and figure.get("panel_b_must_not_contain_FMT_label") is True
        and all("FMT" not in title for title in PANEL_TITLES),
        "figure title or count contract changed",
    )
    _require(
        value.get("metrics", {}).get("producer_metric_comparison", {}).get("arm")
        == "dual_histogram_llr"
        and tuple(value.get("scene_schema", {}).get("ordered_arrays", ()))
        == SCENE_ARRAY_NAMES,
        "metric or scene contract changed",
    )
    return value


def _authenticate_reporting_checkout(
    expected_commit: str,
    *,
    expected_method_commit: str,
) -> dict[str, Any]:
    _require(_is_lower_commit(expected_commit), "expected reporting commit must be lowercase 40-hex")
    _require(
        _is_lower_commit(expected_method_commit),
        "expected method commit must be lowercase 40-hex",
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
    required_paths = tuple(
        sorted(
            set(REPORTING_DEPENDENCY_RELATIVE_PATHS)
            | set(METHOD_INTERPRETATION_RELATIVE_PATHS)
        )
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *required_paths],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    _require(
        {path.replace("\\", "/") for path in tracked}
        == set(required_paths),
        "reporting or method-interpretation dependencies are not tracked exactly",
    )
    method_blobs: dict[str, str] = {}
    for relative in METHOD_INTERPRETATION_RELATIVE_PATHS:
        method_blob = subprocess.run(
            ["git", "rev-parse", f"{expected_method_commit}:{relative}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        reporting_blob = subprocess.run(
            ["git", "rev-parse", f"{expected_commit}:{relative}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _require(
            bool(method_blob) and method_blob == reporting_blob,
            f"method interpretation source differs between commits: {relative}",
        )
        method_blobs[relative] = method_blob
    return {
        "reporting_git_commit": commit,
        "reporting_script_sha256": sha256_file(Path(__file__).resolve()),
        "reporting_dependency_sha256": {
            relative: sha256_file(REPOSITORY_ROOT / relative)
            for relative in REPORTING_DEPENDENCY_RELATIVE_PATHS
        },
        "method_interpretation_git_commit": expected_method_commit,
        "method_interpretation_git_blob_sha1": method_blobs,
        "method_interpretation_git_blob_sha1_content_sha256": canonical_json_sha256(
            method_blobs
        ),
        "frozen_report_config_sha256": CONFIG_SHA256,
    }


def _authenticate_slurm_runtime(
    *,
    parent_root: Path,
    release_root: Path,
    half_fold_root: Path,
    boeing_fold_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    expected = {
        "SLURM_JOB_NAME": "PTMSCRankViz",
        "SLURM_JOB_ACCOUNT": "pi-hadwigm",
        "SLURM_CPUS_PER_TASK": "32",
        "SLURM_MEM_PER_NODE": "131072",
    }
    for name, required in expected.items():
        _require(os.environ.get(name) == required, f"production Slurm identity changed: {name}")
    _require(os.environ.get("SLURM_JOB_PARTITION") in {"cpu", "batch"}, "production partition must be CPU")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    _require(job_id.isdecimal(), "production Slurm job ID is missing")
    _require(
        not any(
            value.strip() not in {"", "0", "N/A"}
            for name, value in os.environ.items()
            if name in {"SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS"}
        ),
        "GPU allocation is forbidden for this CPU report",
    )
    paths = {
        "parent_root": str(parent_root.resolve()),
        "release_root": str(release_root.resolve()),
        "half_fold_root": str(half_fold_root.resolve()),
        "boeing_fold_root": str(boeing_fold_root.resolve()),
        "output_root": str(output_root.resolve()),
    }
    _require(all(Path(value).is_absolute() for value in paths.values()), "runtime paths must be absolute")
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
        "runtime_paths": paths,
    }


def _historical_git_blob_sha256(relative_path: str) -> str:
    _require(
        relative_path in HISTORICAL_SOURCE_RELATIVE_PATHS,
        f"unexpected historical source path: {relative_path}",
    )
    result = subprocess.run(
        ["git", "show", f"{runner.SOURCE_NUMERICAL_COMMIT}:{relative_path}"],
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(
        result.returncode == 0,
        f"historical producer Git blob is unavailable: {relative_path}",
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _authenticate_historical_source_evidence(
    value: object,
    *,
    plan: runner.Plan,
) -> Mapping[str, Any]:
    """Authenticate the immutable evidence envelope without opening sidecars."""

    _require(isinstance(value, Mapping), "historical source evidence is missing")
    _require(
        set(value)
        == {
            "producer_experiment",
            "producer_git_commit",
            "producer_config_sha256",
            "input_manifest",
            "sidecar_population",
            "historical_source_file_sha256",
            "historical_source_file_sha256_content_sha256",
            "sidecar_npz_members_opened",
            "labels_or_references_opened",
            "authentication_mode",
        },
        "historical source evidence fields changed",
    )
    _require(
        value.get("producer_experiment") == runner.SOURCE_EXPERIMENT
        and value.get("producer_git_commit") == runner.SOURCE_NUMERICAL_COMMIT
        and value.get("producer_config_sha256") == runner.SOURCE_CONFIG_SHA256
        and value.get("sidecar_npz_members_opened") == []
        and value.get("labels_or_references_opened") == []
        and value.get("authentication_mode")
        == "historical_git_blob_and_complete_file_hash_replay",
        "historical source producer identity changed",
    )

    input_manifest = value.get("input_manifest")
    _require(
        isinstance(input_manifest, Mapping)
        and set(input_manifest) == {"path", "file_sha256", "content_sha256"},
        "historical source input-manifest fields changed",
    )
    _require(
        input_manifest.get("path") == str(plan.source_input_manifest_path)
        and input_manifest.get("file_sha256")
        == runner.SOURCE_INPUT_MANIFEST_SHA256
        and input_manifest.get("content_sha256")
        == SOURCE_INPUT_MANIFEST_CONTENT_SHA256,
        "historical source input-manifest identity changed",
    )

    population = value.get("sidecar_population")
    _require(
        isinstance(population, Mapping)
        and set(population)
        == {
            "root",
            "manifest_path",
            "manifest_file_sha256",
            "manifest_content_sha256",
            "row_count",
            "rows_content_sha256",
            "assigned_row_count_total",
            "valid_projection_row_count_total",
        },
        "historical source population fields changed",
    )
    _require(
        population.get("root") == str(plan.source_sidecar_root)
        and population.get("manifest_path")
        == str(plan.source_population_manifest_path)
        and population.get("manifest_file_sha256")
        == runner.SOURCE_POPULATION_MANIFEST_SHA256
        and population.get("manifest_content_sha256")
        == SOURCE_POPULATION_MANIFEST_CONTENT_SHA256
        and population.get("row_count") == SOURCE_SIDECAR_COUNT
        and population.get("rows_content_sha256")
        == SOURCE_POPULATION_ROWS_CONTENT_SHA256
        and population.get("assigned_row_count_total")
        == SOURCE_SIDECAR_COUNT * runner.ASSIGNED_ROW_COUNT
        and population.get("valid_projection_row_count_total")
        == SOURCE_VALID_PROJECTION_ROW_COUNT,
        "historical source population identity changed",
    )

    source_hashes = value.get("historical_source_file_sha256")
    _require(
        isinstance(source_hashes, Mapping)
        and set(source_hashes) == set(HISTORICAL_SOURCE_RELATIVE_PATHS),
        "historical source Git-blob map fields changed",
    )
    expected_source_hashes = {
        relative_path: _historical_git_blob_sha256(relative_path)
        for relative_path in HISTORICAL_SOURCE_RELATIVE_PATHS
    }
    source_hash_map = dict(source_hashes)
    _require(
        source_hash_map == expected_source_hashes
        and canonical_json_sha256(source_hash_map)
        == SOURCE_FILE_SHA256_CONTENT_SHA256
        and value.get("historical_source_file_sha256_content_sha256")
        == SOURCE_FILE_SHA256_CONTENT_SHA256,
        "historical source Git-blob map identity changed",
    )
    return value


def _authenticate_release_source_evidence(
    value: object,
    *,
    plan: runner.Plan,
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), "aggregate source evidence is missing")
    _require(
        set(value)
        == {
            "config_sha256",
            "parent_binding_file_sha256",
            "binding_completion_file_sha256",
            "historical_source_centered_evidence",
        },
        "aggregate source evidence fields changed",
    )
    _require(
        value.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256
        and _is_lower_sha256(value.get("parent_binding_file_sha256"))
        and _is_lower_sha256(value.get("binding_completion_file_sha256")),
        "aggregate source evidence identity changed",
    )
    _authenticate_historical_source_evidence(
        value.get("historical_source_centered_evidence"),
        plan=plan,
    )
    return value


def authenticate_release_root(
    root: Path,
    *,
    plan: runner.Plan,
    expected_method_commit: str,
) -> AuthenticatedRelease:
    root = root.resolve()
    _require(root.is_dir(), f"aggregate release root does not exist: {root}")
    expected_files = {
        "AGGREGATE_COMPLETE.json",
        "aggregate_manifest.json",
        "aggregate_summary.json",
        "outer_family_summary.csv",
    }
    _require({path.name for path in root.iterdir()} == expected_files, "aggregate release file set changed")
    complete_path = root / "AGGREGATE_COMPLETE.json"
    manifest_path = root / "aggregate_manifest.json"
    report_path = root / "aggregate_summary.json"
    complete = _read_self_hashed_json(complete_path)
    manifest = _read_self_hashed_json(manifest_path)
    report = _read_self_hashed_json(report_path)
    _require(_is_lower_commit(expected_method_commit), "expected method commit must be lowercase 40-hex")
    for value, label in ((complete, "completion"), (manifest, "manifest"), (report, "report")):
        _require(
            value.get("experiment") == runner.EXPERIMENT
            and value.get("status") == "completed"
            and value.get("mode") == "complete_five_fold_aggregate"
            and value.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256
            and value.get("aggregator_git_commit") == expected_method_commit
            and value.get("fold_git_commit") == expected_method_commit,
            f"aggregate {label} identity changed",
        )
    _require(
        complete.get("schema") == aggregate.AGGREGATE_COMPLETE_SCHEMA
        and manifest.get("schema") == aggregate.AGGREGATE_MANIFEST_SCHEMA
        and report.get("schema") == aggregate.AGGREGATE_SUMMARY_SCHEMA,
        "aggregate schema changed",
    )
    evidence = complete.get("source_centered_evidence")
    _require(
        manifest.get("source_centered_evidence") == evidence
        and report.get("source_centered_evidence") == evidence,
        "aggregate source evidence differs across release files",
    )
    _authenticate_release_source_evidence(evidence, plan=plan)
    _require(
        complete.get("aggregate_manifest_file") == manifest_path.name
        and complete.get("aggregate_manifest_file_sha256") == sha256_file(manifest_path)
        and complete.get("report_file") == report_path.name
        and complete.get("report_file_sha256") == sha256_file(report_path)
        and manifest.get("report_file") == report_path.name
        and manifest.get("report_file_sha256") == sha256_file(report_path)
        and manifest.get("outer_family_summary_file") == "outer_family_summary.csv"
        and manifest.get("outer_family_summary_file_sha256")
        == sha256_file(root / "outer_family_summary.csv"),
        "aggregate release file binding changed",
    )
    source_rows = manifest.get("source_folds")
    _require(isinstance(source_rows, list), "aggregate source-fold list is missing")
    source_folds: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        _require(isinstance(row, Mapping), "aggregate source-fold row is invalid")
        family = str(row.get("outer_family", ""))
        _require(family in runner.FAMILY_ORDER and family not in source_folds, "aggregate family row changed")
        source_folds[family] = dict(row)
    _require(
        tuple(source_folds) == tuple(runner.FAMILY_ORDER)
        and tuple(report.get("outer_families", ())) == tuple(runner.FAMILY_ORDER),
        "release is not the exact complete-five family population",
    )
    return AuthenticatedRelease(
        root=root,
        fold_git_commit=expected_method_commit,
        manifest=manifest,
        report=report,
        source_centered_evidence=evidence,
        source_folds=source_folds,
        evidence=tuple(
            _file_row(root / name, f"aggregate_release:{name}")
            for name in sorted(expected_files)
        ),
    )


def authenticate_fold_root(
    root: Path,
    *,
    expected_family: str,
    release_record: Mapping[str, Any],
    expected_method_commit: str,
    plan: runner.Plan,
    release_evidence: Mapping[str, Any],
) -> AuthenticatedFold:
    """Authenticate all 18 fold files without opening any NPZ member."""

    root = root.resolve()
    _require(root.is_dir(), f"fold root does not exist: {root}")
    _require({path.name for path in root.iterdir()} == set(runner.REQUIRED_FOLD_FILES), "fold file set is not exactly 18")
    result_path = root / "result_manifest.json"
    completion_path = root / "RUN_COMPLETE.json"
    result = _read_self_hashed_json(result_path)
    completion = _read_self_hashed_json(completion_path)
    for value, label in ((result, "result"), (completion, "completion")):
        _require(
            value.get("experiment") == runner.EXPERIMENT
            and value.get("outer_family") == expected_family
            and value.get("git_commit") == expected_method_commit
            and value.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256,
            f"fold {label} identity changed",
        )
    _require(
        result.get("schema") == runner.RESULT_SCHEMA
        and result.get("status") == "completed"
        and completion.get("schema") == runner.COMPLETE_SCHEMA,
        "fold result/completion schema changed",
    )
    _require(
        completion.get("result_manifest_file") == result_path.name
        and completion.get("result_manifest_file_sha256") == sha256_file(result_path)
        and completion.get("result_manifest_content_sha256") == result.get("content_sha256"),
        "fold completion does not bind result",
    )
    _require(
        release_record.get("outer_family") == expected_family
        and Path(str(release_record.get("run_directory", ""))).name == root.name
        and release_record.get("completion_file_sha256") == sha256_file(completion_path)
        and release_record.get("result_manifest_file_sha256") == sha256_file(result_path),
        "complete-five release does not bind the supplied fold",
    )
    fold_source = result.get("source_centered_evidence")
    _require(isinstance(fold_source, Mapping), "fold source evidence is missing")
    _require(
        fold_source.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256
        and fold_source.get("parent_binding", {}).get("file_sha256")
        == release_evidence.get("parent_binding_file_sha256")
        and fold_source.get("binding_completion", {}).get("file_sha256")
        == release_evidence.get("binding_completion_file_sha256")
        and fold_source.get("historical_source_centered_evidence")
        == release_evidence.get("historical_source_centered_evidence")
        and fold_source.get("fmt_features_opened") is False
        and fold_source.get("raw_features_opened") is False
        and fold_source.get("reference_labels_all_opened") is False,
        "fold source evidence differs from aggregate release",
    )
    artifacts = result.get("artifacts")
    _require(isinstance(artifacts, Mapping), "fold artifact map is missing")
    evidence = [
        _file_row(result_path, f"fold:{expected_family}:result_manifest"),
        _file_row(completion_path, f"fold:{expected_family}:RUN_COMPLETE"),
    ]
    for name in runner.REQUIRED_FOLD_FILES:
        if name in {"result_manifest.json", "RUN_COMPLETE.json"}:
            continue
        path = root / name
        record = artifacts.get(name)
        _require(
            isinstance(record, Mapping)
            and int(record.get("size_bytes", -1)) == path.stat().st_size
            and record.get("sha256") == sha256_file(path),
            f"fold artifact is not result-bound: {name}",
        )
        evidence.append(_file_row(path, f"fold:{expected_family}:{name}"))
    selected = _read_self_hashed_json(root / "selected_candidate.json")
    prediction_manifest = _read_self_hashed_json(root / "outer_prediction_manifest.json")
    _require(
        selected.get("schema") == runner.SELECTED_SCHEMA
        and selected.get("experiment") == runner.EXPERIMENT
        and selected.get("outer_family") == expected_family
        and selected.get("git_commit") == expected_method_commit
        and selected.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256
        and selected.get("outer_results_visible_to_selection") is False
        and selected.get("outer_sidecar_members_opened") is False
        and selected.get("outer_labels_opened") is False,
        "selected-candidate gate changed",
    )
    primary = aggregate._candidate_from_payload(plan, selected.get("primary_candidate"))
    control = aggregate._control_from_payload(plan, selected.get("selected_control"))
    primary_payload = runner._json_safe(runner._candidate_payload(primary))
    control_payload = runner._json_safe(runner._control_payload(control))
    _require(
        primary_payload.get("arm") == "dual_histogram_llr"
        and control_payload.get("arm") == "negative_ecdf",
        "primary/control arm identity changed",
    )
    _require(
        prediction_manifest.get("schema") == runner.PREDICTION_MANIFEST_SCHEMA
        and prediction_manifest.get("prediction_schema") == runner.PREDICTION_SCHEMA
        and prediction_manifest.get("experiment") == runner.EXPERIMENT
        and prediction_manifest.get("outer_family") == expected_family
        and prediction_manifest.get("git_commit") == expected_method_commit
        and prediction_manifest.get("config_sha256") == runner.EXPECTED_CONFIG_SHA256
        and prediction_manifest.get("primary_candidate") == primary_payload
        and prediction_manifest.get("selected_control") == control_payload
        and prediction_manifest.get("outer_labels_opened") is False
        and prediction_manifest.get("parent_control_prediction_opened") is False
        and prediction_manifest.get("fmt_features_opened") is False
        and prediction_manifest.get("raw_features_opened") is False
        and prediction_manifest.get("reference_labels_all_opened") is False,
        "label-free prediction manifest identity changed",
    )
    _require(
        result.get("primary_candidate") == primary_payload
        and result.get("selected_control") == control_payload
        and result.get("selected_candidate_file_sha256")
        == sha256_file(root / "selected_candidate.json"),
        "candidate differs across selected/prediction/result artifacts",
    )
    prediction_record = prediction_manifest.get("prediction_file")
    array_records = prediction_manifest.get("arrays")
    _require(
        isinstance(prediction_record, Mapping)
        and prediction_record.get("path") == "outer_predictions.npz"
        and int(prediction_record.get("size_bytes", -1))
        == (root / "outer_predictions.npz").stat().st_size
        and prediction_record.get("sha256") == sha256_file(root / "outer_predictions.npz")
        and isinstance(array_records, Mapping)
        and set(array_records) == set(runner.PREDICTION_DTYPES),
        "prediction archive/member binding changed",
    )
    return AuthenticatedFold(
        root=root,
        outer_family=expected_family,
        git_commit=expected_method_commit,
        result=result,
        prediction_manifest=prediction_manifest,
        primary_candidate=primary_payload,
        selected_control=control_payload,
        evidence=tuple(evidence),
    )


def load_prediction_groups(
    folds: Mapping[str, AuthenticatedFold],
) -> dict[str, PredictionGroup]:
    """First fold NPZ-member access; call only after the input manifest exists."""

    groups: dict[str, PredictionGroup] = {}
    for family in REQUIRED_FAMILIES:
        fold = folds[family]
        records = fold.prediction_manifest["arrays"]
        file_record = fold.prediction_manifest["prediction_file"]
        arrays, digest = runner.source_runner._verify_npz_arrays(
            fold.root / "outer_predictions.npz",
            file_record=file_record,
            records=records,
        )
        _require(digest == file_record["sha256"], "prediction archive SHA changed")
        unique_count = int(fold.prediction_manifest.get("unique_row_count", -1))
        valid_count = int(fold.prediction_manifest.get("valid_row_count", -1))
        for name, dtype in runner.UNIQUE_PREDICTION_DTYPES.items():
            _require(arrays[name].dtype == dtype and arrays[name].shape == (unique_count,), f"unique prediction dtype/shape changed: {name}")
        for name, dtype in runner.VALID_PREDICTION_DTYPES.items():
            _require(arrays[name].dtype == dtype and arrays[name].shape == (valid_count,), f"valid prediction dtype/shape changed: {name}")
        for dataset in (item for item in DATASETS if DATASET_TO_FAMILY[item] == family):
            unique_mask = (arrays["unique_dataset"] == dataset) & (arrays["unique_source_ordinal"] == SOURCE_ORDINAL)
            valid_mask = (arrays["valid_dataset"] == dataset) & (arrays["valid_source_ordinal"] == SOURCE_ORDINAL)
            _require(int(unique_mask.sum()) == CENTER_COUNT and valid_mask.any(), f"fixed-source group is incomplete: {dataset}")
            unique = {name: np.asarray(arrays[name][unique_mask]) for name in runner.UNIQUE_PREDICTION_DTYPES}
            valid = {name: np.asarray(arrays[name][valid_mask]) for name in runner.VALID_PREDICTION_DTYPES}
            _require(
                len(np.unique(unique["unique_source_index"])) == 1
                and int(unique["unique_source_index"][0]) == EXPECTED_SOURCE_INDEX[dataset]
                and len(np.unique(valid["valid_source_index"])) == 1
                and int(valid["valid_source_index"][0]) == EXPECTED_SOURCE_INDEX[dataset],
                f"fixed source index changed: {dataset}",
            )
            _require(dataset not in groups, f"duplicate prediction group: {dataset}")
            groups[dataset] = PredictionGroup(
                dataset,
                family,
                fold.primary_candidate,
                fold.selected_control,
                unique,
                valid,
            )
    _require(tuple(groups) == DATASETS, "four prediction groups changed")
    return groups


def read_producer_metric_rows(
    folds: Mapping[str, AuthenticatedFold],
    groups: Mapping[str, PredictionGroup],
) -> dict[tuple[str, str, str], Mapping[str, str]]:
    selected: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for family, fold in folds.items():
        with (fold.root / "outer_group_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as source:
            reader = csv.DictReader(source)
            _require(tuple(reader.fieldnames or ()) == runner.OUTER_METRIC_FIELDS, f"producer metric fields changed: {family}")
            rows = list(reader)
        for row in rows:
            if int(row["source_ordinal"]) != SOURCE_ORDINAL:
                continue
            if row["dataset"] not in DATASETS or DATASET_TO_FAMILY[row["dataset"]] != family:
                continue
            arm = row["arm"]
            if arm not in {"dual_histogram_llr", *TABLE_ONLY_ARMS}:
                continue
            population = row["population"]
            if population not in {
                "combined_valid_unique_centers",
                "all_parent_valid_rows",
            }:
                continue
            if arm in TABLE_ONLY_ARMS and population != "all_parent_valid_rows":
                continue
            key = (row["dataset"], arm, population)
            _require(key not in selected, f"duplicate producer metric row: {key}")
            group = groups[row["dataset"]]
            _require(
                row["outer_family"] == family
                and row["source_index"] == str(EXPECTED_SOURCE_INDEX[row["dataset"]])
                and row["primary_candidate_id"] == str(group.primary_candidate["candidate_id"])
                and row["control_candidate_id"] == str(group.selected_control["candidate_id"])
                and row["template_success_eligible"]
                == (
                    "1"
                    if arm == "dual_histogram_llr"
                    and population == "all_parent_valid_rows"
                    else "0"
                ),
                f"producer metric identity changed: {key}",
            )
            selected[key] = row
    expected = {
        (dataset, arm, population)
        for dataset in DATASETS
        for arm, population in (
            ("dual_histogram_llr", "combined_valid_unique_centers"),
            ("dual_histogram_llr", "all_parent_valid_rows"),
            ("negative_ecdf", "all_parent_valid_rows"),
            ("direct_rank_mean_top5", "all_parent_valid_rows"),
        )
    }
    _require(set(selected) == expected, "producer primary/control metric rows are incomplete")
    return selected


def _compare_metrics(observed: Mapping[str, Any], expected: Mapping[str, str], *, label: str) -> None:
    for field in METRIC_INTEGER_FIELDS:
        _require(int(observed[field]) == int(expected[field]), f"{label} metric mismatch: {field}")
    for field in METRIC_FLOAT_FIELDS:
        left = float(observed[field])
        raw = expected[field]
        if raw == "":
            right = float("nan")
        else:
            try:
                right = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{label} producer metric is not numeric: {field}"
                ) from exc
            _require(
                np.isfinite(right),
                f"{label} producer metric must be finite or blank: {field}",
            )
        if np.isnan(left) or np.isnan(right):
            _require(np.isnan(left) and np.isnan(right), f"{label} metric mismatch: {field}")
        else:
            _require(abs(left - right) <= 1.0e-12, f"{label} metric mismatch: {field}")


def _figure_contract() -> dict[str, Any]:
    return {
        "core_conclusion": (
            "At fixed source ordinal 2, the primary source-rank likelihood template "
            "prediction has flow-specific spatial agreement and error structure relative to IVD p95."
        ),
        "results_level_question": (
            "Where does dual_histogram_llr agree or disagree with IVD p95 in the four fixed flows?"
        ),
        "archetype": "image plate + quantification",
        "backend": "Python/matplotlib",
        "panel_map": {
            "a": "unchanged IVD-p95 background plus fixed first 120 legacy and first 120 expanded parent pathlines",
            "b": "primary dual_histogram_llr prediction for all combined-valid unique centers",
            "c": "TP/FP/FN/TN for the exact same centers in the same order",
        },
        "primary_reported_not_plotted": "valid_primary_prediction on all parent-valid rows",
        "controls_reported_not_plotted": ["negative_ecdf", "direct_rank_mean_top5"],
        "selection": "four flows and source ordinal 2 were fixed before result access; arm switching is forbidden",
        "uncertainty": "none; each figure is one preregistered source timeslice",
        "panel_b_is_not_FMT": True,
    }


def _scene_manifest(
    *,
    scene_path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    dataset: str,
    reporting_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return runner._manifest(
        {
            "schema": SCENE_MANIFEST_SCHEMA,
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
            "arrays": runner.source_runner.early._array_manifest(arrays),
            "metadata": _json_safe(metadata),
        }
    )


def _load_combined_scene(scene_path: Path, manifest_path: Path) -> CombinedCenterScene:
    manifest = _read_self_hashed_json(manifest_path)
    _require(
        manifest.get("scene_schema") == SCENE_SCHEMA
        and tuple(manifest.get("ordered_array_names", ())) == SCENE_ARRAY_NAMES,
        "combined scene manifest changed",
    )
    arrays, digest = runner.source_runner._verify_npz_arrays(
        scene_path,
        file_record=manifest["scene_file"],
        records=manifest["arrays"],
    )
    _require(digest == manifest["scene_file"]["sha256"], "combined scene archive changed")
    metadata = json.loads(str(np.asarray(arrays["metadata_json"]).reshape(())))
    return CombinedCenterScene(
        dataset=str(metadata["dataset"]),
        title=str(metadata["display_title"]),
        source_index=int(metadata["source_index"]),
        bounds=np.asarray(arrays["bounds"]),
        seeds=np.asarray(arrays["seeds"]),
        reference=np.asarray(arrays["reference"], dtype=np.bool_),
        prediction=np.asarray(arrays["prediction"], dtype=np.bool_),
        center_seed_index=np.asarray(arrays["center_seed_index"], dtype=np.int64),
        paired_score=np.asarray(arrays["primary_score"], dtype=np.float64),
        legacy_valid=np.asarray(arrays["legacy_valid"], dtype=np.bool_),
        expanded_valid=np.asarray(arrays["expanded_valid"], dtype=np.bool_),
        display_pathlines=np.asarray(arrays["display_pathlines"]),
        display_pathline_block_index=np.asarray(arrays["display_pathline_block_index"], dtype=np.int8),
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
    release_root: Path,
    half_fold_root: Path,
    boeing_fold_root: Path,
    output_root: Path,
    expected_reporting_commit: str,
    expected_method_commit: str,
) -> dict[str, Any]:
    immutable_input_roots = (
        parent_root,
        release_root,
        half_fold_root,
        boeing_fold_root,
    )
    _require(
        _is_absolute_machine_path(output_root)
        and all(_is_absolute_machine_path(path) for path in immutable_input_roots),
        "report input and output roots must be absolute",
    )
    _require_disjoint_output_root(output_root, immutable_input_roots)
    if output_root.exists():
        raise FileExistsError(f"immutable output directory already exists: {output_root}")
    reporting_identity = _authenticate_reporting_checkout(
        expected_reporting_commit,
        expected_method_commit=expected_method_commit,
    )
    config = _authenticate_config()
    plan = runner.load_plan(
        REPOSITORY_ROOT / "config" / "Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml"
    )
    reporting_identity = {
        **reporting_identity,
        "slurm_runtime": _authenticate_slurm_runtime(
            parent_root=parent_root,
            release_root=release_root,
            half_fold_root=half_fold_root,
            boeing_fold_root=boeing_fold_root,
            output_root=output_root,
        ),
    }
    release = authenticate_release_root(
        release_root, plan=plan, expected_method_commit=expected_method_commit
    )
    fold_roots = {"half_cylinder": half_fold_root, "boeing_747": boeing_fold_root}
    folds = {
        family: authenticate_fold_root(
            fold_roots[family],
            expected_family=family,
            release_record=release.source_folds[family],
            expected_method_commit=expected_method_commit,
            plan=plan,
            release_evidence=release.source_centered_evidence,
        )
        for family in REQUIRED_FAMILIES
    }
    parent_scenes, parent_evidence = _authenticate_parent(parent_root.resolve())
    _require(
        set(parent_scenes) == {(dataset, block) for dataset in DATASETS for block in BLOCKS},
        "parent scene population changed",
    )
    input_rows: list[Mapping[str, Any]] = [*parent_evidence, *release.evidence]
    for family in REQUIRED_FAMILIES:
        input_rows.extend(folds[family].evidence)
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
            roles = set(str(by_path[key]["role"]).split(" | ")) | {str(row["role"])}
            by_path[key]["role"] = " | ".join(sorted(roles))
        else:
            by_path[key] = dict(row)
    opaque_rows = [by_path[key] for key in sorted(by_path)]
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "scenes").mkdir()
    (output_root / "figures").mkdir()
    publish_bytes_without_overwrite(
        output_root / "frozen_config.yaml", CONFIG_PATH.read_bytes()
    )
    _require(sha256_file(output_root / "frozen_config.yaml") == CONFIG_SHA256, "frozen config copy changed")
    release_identity = {
        "root": str(release.root),
        "mode": "complete_five_fold_aggregate",
        "aggregator_git_commit": expected_method_commit,
        "fold_git_commit": expected_method_commit,
        "source_centered_evidence_sha256": canonical_json_sha256(release.source_centered_evidence),
    }
    input_manifest = runner._manifest(
        {
            "schema": INPUT_SCHEMA,
            "experiment": EXPERIMENT,
            **reporting_identity,
            "method_experiment": runner.EXPERIMENT,
            "method_config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "method_fold_git_commit": expected_method_commit,
            "method_release_authentication": [release_identity],
            "parent_scene_experiment": PARENT_SCENE_EXPERIMENT,
            "parent_scene_git_commit": PARENT_SCENE_COMMIT,
            "report_config_status": config["status"],
            "source_selection": "fixed source ordinal 2",
            "npz_array_access_before_manifest_write": False,
            "fold_sidecar_or_label_member_access": False,
            "all_18_files_authenticated_per_required_fold": True,
            "files": opaque_rows,
            "files_content_sha256": canonical_json_sha256(opaque_rows),
        }
    )
    _atomic_json(output_root / "input_manifest.json", input_manifest)
    contract = runner._manifest(_figure_contract())
    _atomic_json(output_root / "figure_contract.json", contract)

    # First input NPZ-member access in this report.  Config, aggregate, both
    # complete folds, all 18 files per fold, and all parent scene files are now
    # closed in input_manifest.json.
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
        combined = combine_rank_likelihood_parent_scenes(
            legacy_metadata=legacy_metadata,
            legacy_arrays=legacy_arrays,
            expanded_metadata=expanded_metadata,
            expanded_arrays=expanded_arrays,
            unique_prediction=group.unique,
            title=DISPLAY_NAMES[dataset],
        )
        projection = bind_rank_likelihood_valid_projection(
            legacy_metadata=legacy_metadata,
            legacy_arrays=legacy_arrays,
            expanded_metadata=expanded_metadata,
            expanded_arrays=expanded_arrays,
            unique_prediction=group.unique,
            valid_prediction=group.valid,
        )
        table_projections = {
            arm: bind_rank_likelihood_table_only_projection(
                arm=arm,
                legacy_metadata=legacy_metadata,
                legacy_arrays=legacy_arrays,
                expanded_metadata=expanded_metadata,
                expanded_arrays=expanded_arrays,
                unique_prediction=group.unique,
                valid_prediction=group.valid,
            )
            for arm in TABLE_ONLY_ARMS
        }
        center_metrics = _metric_values(combined.reference, combined.prediction, combined.paired_score)
        projection_metrics = _metric_values(projection.reference, projection.prediction, projection.score)
        table_metrics = {
            arm: _metric_values(item.reference, item.prediction, item.score)
            for arm, item in table_projections.items()
        }
        _compare_metrics(
            center_metrics,
            producer_metrics[
                (dataset, "dual_histogram_llr", "combined_valid_unique_centers")
            ],
            label=f"{dataset}/combined-valid-center",
        )
        _compare_metrics(
            projection_metrics,
            producer_metrics[(dataset, "dual_histogram_llr", "all_parent_valid_rows")],
            label=f"{dataset}/valid-projection",
        )
        for arm in TABLE_ONLY_ARMS:
            _compare_metrics(
                table_metrics[arm],
                producer_metrics[(dataset, arm, "all_parent_valid_rows")],
                label=f"{dataset}/{arm}/valid-projection",
            )
        legacy_valid = np.asarray(group.unique["unique_legacy_valid"], dtype=np.bool_)
        expanded_valid = np.asarray(group.unique["unique_expanded_valid"], dtype=np.bool_)
        availability = {
            "combined_valid_center_count": int((legacy_valid | expanded_valid).sum()),
            "unique_center_combined_coverage": float((legacy_valid | expanded_valid).mean()),
            "both_valid_count": int((legacy_valid & expanded_valid).sum()),
            "legacy_only_count": int((legacy_valid & ~expanded_valid).sum()),
            "expanded_only_count": int((~legacy_valid & expanded_valid).sum()),
            "neither_valid_count": int((~legacy_valid & ~expanded_valid).sum()),
            "legacy_valid_row_count": int(len(legacy_arrays["reference"])),
            "expanded_valid_row_count": int(len(expanded_arrays["reference"])),
            "valid_projection_row_count": int(len(projection.reference)),
        }
        for field in AVAILABILITY_FIELDS:
            expected = producer_metrics[
                (dataset, "dual_histogram_llr", "all_parent_valid_rows")
            ][field]
            observed = availability[field]
            if field == "unique_center_combined_coverage":
                _require(abs(float(observed) - float(expected)) <= 1.0e-12, f"availability mismatch: {dataset}/{field}")
            else:
                _require(int(observed) == int(expected), f"availability mismatch: {dataset}/{field}")
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
            "method_fold_git_commit": expected_method_commit,
            "method_release_authentication": [release_identity],
            "primary_candidate": dict(group.primary_candidate),
            "selected_control_not_plotted": dict(group.selected_control),
            "plotted_arm": "dual_histogram_llr",
            "prediction_semantics": "primary dual_histogram_llr prediction per combined-valid unique center",
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
            "table_only_control_metrics": {
                arm: _json_safe(table_metrics[arm]) for arm in TABLE_ONLY_ARMS
            },
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
        scene_stem = output_root / "scenes" / f"{dataset}_source_ordinal_2_rank_likelihood"
        scene_path = scene_stem.with_suffix(".scene.npz")
        scene_manifest_path = scene_stem.with_suffix(".scene.json")
        _atomic_npz(scene_path, arrays)
        _atomic_json(
            scene_manifest_path,
            _scene_manifest(
                scene_path=scene_path,
                arrays=arrays,
                metadata=scene_metadata,
                dataset=dataset,
                reporting_identity=reporting_identity,
            ),
        )
        loaded = _load_combined_scene(scene_path, scene_manifest_path)
        figure_stem = output_root / "figures" / f"{dataset}_source_ordinal_2_rank_likelihood_triptych"
        png_path = figure_stem.with_suffix(".png")
        pdf_path = figure_stem.with_suffix(".pdf")
        svg_path = figure_stem.with_suffix(".svg")
        alignment_path = figure_stem.with_suffix(".alignment.json")
        render_metadata = render_source_rank_likelihood_triptych(
            loaded,
            png_path=png_path,
            pdf_path=pdf_path,
            svg_path=svg_path,
            alignment_path=alignment_path,
            view=DATASET_VIEWS[dataset],
            dpi=360,
        )
        _require(
            _camera_core(parent_render_by_block["legacy_2_1"])
            == _camera_core(parent_render_by_block["expanded_3_1"])
            == _camera_core(render_metadata),
            f"camera/bounds changed: {dataset}",
        )
        _require(render_metadata["panel_order"] == list(PANEL_TITLES), "panel order changed")
        render_metadata_value = runner._manifest(
            {
                **render_metadata,
                "experiment": EXPERIMENT,
                **reporting_identity,
                "scene_npz": scene_path.relative_to(output_root).as_posix(),
                "scene_npz_sha256": sha256_file(scene_path),
                "scene_manifest": scene_manifest_path.relative_to(output_root).as_posix(),
                "scene_manifest_sha256": sha256_file(scene_manifest_path),
                "parent_camera_exact": True,
                "primary_valid_projection_metrics_reported_not_plotted": True,
                "controls_not_plotted": True,
            }
        )
        render_metadata_path = figure_stem.with_suffix(".render.json")
        _atomic_json(render_metadata_path, render_metadata_value)
        primary = group.primary_candidate
        metric_row: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "dataset": dataset,
            "display_name": DISPLAY_NAMES[dataset],
            "outer_family": DATASET_TO_FAMILY[dataset],
            "source_ordinal": SOURCE_ORDINAL,
            "source_index": combined.source_index,
            "arm": "dual_histogram_llr",
            "primary_candidate_id": primary["candidate_id"],
            "control_candidate_id": group.selected_control["candidate_id"],
            "weight": float(primary["weight"]),
            "bin_count": int(primary["bin_count"]),
            "beta": float(primary["beta"]),
            "sigma": float(primary["sigma"]),
            "decision_rule": primary["decision_rule"],
            "decision_value": float(primary["decision_value"]),
            **availability,
        }
        metric_groups = (
            ("center", center_metrics),
            ("primary_valid_projection", projection_metrics),
            ("negative_ecdf_valid_projection", table_metrics["negative_ecdf"]),
            (
                "direct_rank_mean_top5_valid_projection",
                table_metrics["direct_rank_mean_top5"],
            ),
        )
        for prefix, values in metric_groups:
            for field in (*METRIC_INTEGER_FIELDS, *METRIC_FLOAT_FIELDS):
                metric_row[f"{prefix}_{field}"] = values[field]
        metric_rows.append(metric_row)
        relative = lambda path: path.relative_to(output_root).as_posix()
        figure_rows.append(
            {
                "dataset": dataset,
                "source_ordinal": SOURCE_ORDINAL,
                "population": "combined_valid_unique_centers",
                "plotted_arm": "dual_histogram_llr",
                "table_only_control_arms": list(TABLE_ONLY_ARMS),
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
    visualization = runner._manifest(
        {
            "schema": VISUALIZATION_SCHEMA,
            "experiment": EXPERIMENT,
            **reporting_identity,
            "method_experiment": runner.EXPERIMENT,
            "method_config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "method_fold_git_commit": expected_method_commit,
            "method_release_authentication": [release_identity],
            "evidence_scope": "family-held-out exposed-development fixed-source reporting",
            "formal_confirmation": False,
            "source_selection": "fixed source ordinal 2; no result-based selection",
            "figure_count": 4,
            "unique_key": ["dataset"],
            "classification_unit": "combined-valid unique center",
            "plotted_arm": "dual_histogram_llr",
            "controls_not_plotted": True,
            "table_only_control_arms": list(TABLE_ONLY_ARMS),
            "table_only_control_metrics_reported": True,
            "panel_b_is_not_FMT": True,
            "scale_blocks_are_context_not_separate_classifiers": True,
            "primary_valid_projection_metrics_reported_not_plotted": True,
            "entries": figure_rows,
        }
    )
    _atomic_json(output_root / "visualization_manifest.json", visualization)
    artifacts = [
        {
            "relative_path": path.relative_to(output_root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name not in {"result_manifest.json", "RUN_COMPLETE.json"}
    ]
    result = runner._manifest(
        {
            "schema": RESULT_SCHEMA,
            "experiment": EXPERIMENT,
            **reporting_identity,
            "status": "completed_pending_local_rendered_qa",
            "formal_confirmation": False,
            "method_experiment": runner.EXPERIMENT,
            "method_config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "method_fold_git_commit": expected_method_commit,
            "method_release_authentication": [release_identity],
            "figure_count": 4,
            "plotted_arm": "dual_histogram_llr",
            "table_only_control_arms": list(TABLE_ONLY_ARMS),
            "table_only_control_metrics_reported": True,
            "combined_valid_center_count": int(sum(row["combined_valid_center_count"] for row in metric_rows)),
            "primary_valid_projection_row_count": int(sum(row["valid_projection_row_count"] for row in metric_rows)),
            "input_manifest_file_sha256": sha256_file(output_root / "input_manifest.json"),
            "visualization_manifest_file_sha256": sha256_file(output_root / "visualization_manifest.json"),
            "per_figure_metrics_file_sha256": sha256_file(output_root / "per_figure_metrics.csv"),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifacts_content_sha256": canonical_json_sha256(artifacts),
            "local_qa_required_before_delivery": True,
        }
    )
    _atomic_json(output_root / "result_manifest.json", result)
    complete = runner._manifest(
        {
            "schema": COMPLETE_SCHEMA,
            "experiment": EXPERIMENT,
            **reporting_identity,
            "status": "complete_pending_local_rendered_qa",
            "method_experiment": runner.EXPERIMENT,
            "method_config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "method_fold_git_commit": expected_method_commit,
            "method_release_authentication": [release_identity],
            "figure_count": 4,
            "plotted_arm": "dual_histogram_llr",
            "result_manifest_file_sha256": sha256_file(output_root / "result_manifest.json"),
            "result_manifest_content_sha256": result["content_sha256"],
        }
    )
    _atomic_json(output_root / "RUN_COMPLETE.json", complete)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--half-fold-root", type=Path, required=True)
    parser.add_argument("--boeing-fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-reporting-commit", required=True)
    parser.add_argument("--expected-method-commit", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    render_bundle(
        parent_root=arguments.parent_root,
        release_root=arguments.release_root,
        half_fold_root=arguments.half_fold_root,
        boeing_fold_root=arguments.boeing_fold_root,
        output_root=arguments.output_root,
        expected_reporting_commit=arguments.expected_reporting_commit,
        expected_method_commit=arguments.expected_method_commit,
    )


if __name__ == "__main__":
    main()

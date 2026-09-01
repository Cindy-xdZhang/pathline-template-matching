#!/usr/bin/env python3
"""Frozen five-family runner for SourceCenteredPairedScaleTemplate 1.1.

This runner reuses the authenticated Early negative-template metric, exact
same-scale retrieval, leave-one-out tail calibration, and spatial transform.
Its only template-method changes are the frozen source-centered four-coordinate
sidecar and label-free legacy/expanded fusion at each unique spatial center.

Outer source-centered members remain unopened until the nonouter scaler,
calibrator, and selected 1,800-member candidate are atomically published and
freshly authenticated.  Outer references remain unopened until the complete
unique-center and valid-row prediction artifact is freshly replayed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.paired_scale_center_fusion import (  # noqa: E402
    DEFAULT_CENTER_COUNT,
    DEFAULT_TOP_FRACTION,
    DirectSourceCenteredDiagnostics,
    PairedCenterFusion,
    direct_source_centered_diagnostics,
    fixed_top_fraction_over_centers,
    fuse_paired_scale_centers,
    separate_block_center_predictions,
)
from pathline_template_matching.per_scale_negative_metric import (  # noqa: E402
    SCALER_ARRAY_NAMES,
    PerScaleNegativeScaler,
    PerScaleNegativeTailModel,
)
from pathline_template_matching.nested_scale_validation import (  # noqa: E402
    representation_features,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.source_centered_sidecar import (  # noqa: E402
    ASSIGNED_ROW_COUNT,
    INPUT_MANIFEST_SCHEMA,
    POPULATION_MANIFEST_SCHEMA,
    ROW_COMPLETION_SCHEMA,
    SIDECAR_SCHEMA,
    capture_clean_source_identity,
    load_source_centered_sidecar,
)
from scripts import run_verify_early_opposite_pair_kinematics_1_1 as early  # noqa: E402
from scripts.run_verify_scale_conditioned_retrieval_1_1 import (  # noqa: E402
    CacheRow,
    _classification_counts,
    _configure_execution,
    _git_identity,
    _json_safe,
    _ranking_metrics_one_sort,
    _require,
    _stable_file_identity,
    _utc_now,
    _validate_cache_arrays,
    _validate_outer_prediction_projection,
    load_cache_rows,
)


EXPERIMENT = "Verify_SourceCenteredPairedScaleTemplate_1.1"
EXPECTED_CONFIG_SHA256 = (
    "15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc"
)
PARENT_EXPERIMENT = early.EXPERIMENT
PARENT_CONFIG_SHA256 = early.EXPECTED_CONFIG_SHA256
PARENT_NUMERICAL_COMMIT = "2c3774dca0d81db8edd5645e63576526b9e276f7"
FAMILY_ORDER = early.FAMILY_ORDER
BLOCK_NAMES = early.BLOCK_NAMES
GRID_SHAPE = early.GRID_SHAPE
REPRESENTATIONS = (
    "fmt161_plus_source_centered_seed4",
    "real_neighbor36_plus_source_centered_seed4",
    "chirality_all35_plus_source_centered_seed4",
)
PARENT_REPRESENTATION = MappingProxyType(
    {
        REPRESENTATIONS[0]: "fmt161",
        REPRESENTATIONS[1]: "real_neighbor36",
        REPRESENTATIONS[2]: "chirality_all35",
    }
)
COMPOSITE_WIDTH = MappingProxyType(
    {REPRESENTATIONS[0]: 165, REPRESENTATIONS[1]: 40, REPRESENTATIONS[2]: 39}
)
K_VALUES = (1, 5, 15, 31)
SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
PAIR_WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
TOP_FRACTIONS = (0.025, 0.04, 0.05, 0.06, 0.075, 0.10)
FROZEN_CANDIDATE_COUNT = 1800

SCALER_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_scaler_manifest.v1"
)
CALIBRATION_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_calibration_manifest.v1"
)
SELECTED_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_selected_candidate.v1"
)
PREDICTION_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_outer_prediction.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_outer_prediction_manifest.v1"
)
OUTER_SOURCE_BINDING_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_outer_source_binding.v1"
)
INNER_AUDIT_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_inner_fit_audits.v1"
)
OUTER_SUMMARY_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_outer_summary.v1"
)
REFERENCE_AUDIT_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_outer_reference_access.v1"
)
RESULT_SCHEMA = "pathline_template_matching.source_centered_paired_scale_result.v1"
COMPLETE_SCHEMA = "pathline_template_matching.source_centered_paired_scale_complete.v1"

REQUIRED_FOLD_FILES = (
    "inner_group_metrics.csv",
    "inner_candidate_summary.csv",
    "inner_fit_audits.json",
    "final_per_scale_scaler.npz",
    "final_per_scale_scaler_manifest.json",
    "final_tail_calibration.npz",
    "final_tail_calibration_manifest.json",
    "selected_candidate.json",
    "outer_source_centered_binding.json",
    "outer_predictions.npz",
    "outer_prediction_manifest.json",
    "outer_group_metrics.csv",
    "outer_summary.json",
    "outer_reference_access_audit.json",
    "result_manifest.json",
    "RUN_COMPLETE.json",
)


@dataclass(frozen=True)
class Plan:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    parent_plan: early.Plan
    family_order: tuple[str, ...]
    families: Mapping[str, tuple[str, ...]]
    dataset_to_family: Mapping[str, str]
    representations: tuple[str, ...]
    ks: tuple[int, ...]
    sigmas: tuple[float, ...]
    weights: tuple[float, ...]
    top_fractions: tuple[float, ...]
    grid_shape: tuple[int, int, int]
    gaussian_truncate: float
    query_chunk_size: int
    library_chunk_size: int
    shrinkage_lambda: float
    output_root: Path
    required_fold_files: tuple[str, ...]
    sidecar_input_manifest_path: Path | None = None
    sidecar_input_manifest_file_sha256: str | None = None
    sidecar_input_manifest_content_sha256: str | None = None
    sidecar_root: Path | None = None
    sidecar_population_path: Path | None = None
    sidecar_population_file_sha256: str | None = None
    sidecar_population_content_sha256: str | None = None
    sidecar_population: Mapping[str, Any] | None = None


@dataclass
class SourceCenteredCacheProjection:
    row: CacheRow
    fmt_features: np.ndarray
    source_centered_seed4: np.ndarray
    scale_ids: np.ndarray
    center_indices: np.ndarray
    block_indices: np.ndarray
    assigned_row_indices: np.ndarray
    labels: np.ndarray | None
    center_labels: np.ndarray | None
    metadata: Mapping[str, Any]
    assigned_center_indices: np.ndarray
    assigned_block_indices: np.ndarray
    assigned_scale_ids: np.ndarray
    assigned_physical_dx: np.ndarray
    assigned_source_centered_seed4: np.ndarray
    sidecar_file_sha256: str
    sidecar_combined_array_sha256: str
    sidecar_group_mean_curl_xyz: np.ndarray
    sidecar_group_mean_curl_xyz_sha256: str

    @property
    def count(self) -> int:
        return len(self.scale_ids)


@dataclass(frozen=True)
class CandidateSpec:
    representation: str
    k: int
    sigma: float
    weight: float
    top_fraction: float

    @property
    def candidate_id(self) -> str:
        return (
            f"representation={self.representation}|k={self.k}|"
            f"sigma={self.sigma:.1f}|legacy_weight={self.weight:.2f}|"
            f"fixed_top_fraction={self.top_fraction:.3f}"
        )


@dataclass(frozen=True)
class VerifiedScaler:
    manifest_path: Path
    manifest_file_sha256: str
    artifact_file_sha256: str
    manifest: Mapping[str, Any]
    scaler: PerScaleNegativeScaler
    seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedCalibration:
    manifest_path: Path
    manifest_file_sha256: str
    artifact_file_sha256: str
    manifest: Mapping[str, Any]
    model: PerScaleNegativeTailModel
    seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedSelection:
    path: Path
    file_sha256: str
    manifest: Mapping[str, Any]
    seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedPrediction:
    manifest_path: Path
    manifest_file_sha256: str
    artifact_file_sha256: str
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedOuterSourceBinding:
    path: Path
    file_sha256: str
    manifest: Mapping[str, Any]
    seal: object = field(repr=False, compare=False)


_AUTHENTICATION_SEAL = object()


def _candidate_payload(candidate: CandidateSpec) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "representation": candidate.representation,
        "k": candidate.k,
        "sigma": candidate.sigma,
        "weight": candidate.weight,
        "decision_rule": "fixed_top_fraction",
        "decision_value": candidate.top_fraction,
    }


def candidate_specs(plan: Plan) -> tuple[CandidateSpec, ...]:
    result = tuple(
        CandidateSpec(representation, k, sigma, weight, top_fraction)
        for representation in plan.representations
        for k in plan.ks
        for sigma in plan.sigmas
        for weight in plan.weights
        for top_fraction in plan.top_fractions
    )
    _require(len(result) == FROZEN_CANDIDATE_COUNT, "candidate count drifted")
    _require(len({value.candidate_id for value in result}) == len(result), "duplicate candidate ID")
    return result


def load_plan(config_path: str | Path) -> Plan:
    path = Path(config_path).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    _require(digest == EXPECTED_CONFIG_SHA256, "frozen config SHA-256 drifted")
    raw = yaml.safe_load(payload.decode("utf-8"))
    _require(isinstance(raw, Mapping), "config root must be a mapping")
    _require(raw.get("experiment") == EXPERIMENT, "experiment identity drifted")
    _require(
        raw.get("status") == "frozen_before_reading_any_source_centered_real_result",
        "freeze status drifted",
    )
    direct_parent = raw.get("direct_parent")
    _require(isinstance(direct_parent, Mapping), "direct parent is missing")
    _require(
        direct_parent.get("experiment") == PARENT_EXPERIMENT
        and direct_parent.get("config_sha256") == PARENT_CONFIG_SHA256
        and direct_parent.get("numerical_commit") == PARENT_NUMERICAL_COMMIT,
        "direct parent identity drifted",
    )
    parent_config_path = (ROOT / str(direct_parent["config_path"])).resolve()
    parent_plan = early.load_plan(parent_config_path)
    population = raw.get("input_population")
    _require(isinstance(population, Mapping), "input population is missing")
    parent_manifest = population.get("parent_train_cache_manifest")
    _require(isinstance(parent_manifest, Mapping), "parent train-cache manifest is missing")
    _require(
        Path(str(parent_manifest.get("path"))).resolve() == parent_plan.manifest_path.resolve()
        and parent_manifest.get("file_sha256") == parent_plan.manifest_sha256
        and int(parent_manifest.get("exact_rows", -1)) == 32,
        "parent cache population drifted",
    )
    families_raw = raw.get("families")
    split = raw.get("nested_split")
    _require(isinstance(families_raw, Mapping) and isinstance(split, Mapping), "family split is missing")
    family_order = tuple(str(value) for value in split.get("outer_order", ()))
    _require(family_order == FAMILY_ORDER and tuple(split.get("inner_order", ())) == FAMILY_ORDER, "family order drifted")
    _require(split.get("outer_labels_available_to_selection") is False, "outer label gate drifted")
    families = {str(name): tuple(str(value) for value in datasets) for name, datasets in families_raw.items()}
    _require(tuple(families) == FAMILY_ORDER, "family mapping order drifted")
    datasets = [dataset for family in FAMILY_ORDER for dataset in families[family]]
    _require(len(datasets) == 8 and len(set(datasets)) == 8, "family population drifted")
    _require(
        tuple(population.get("allowed_datasets", ())) == tuple(datasets)
        and population.get("forbidden_datasets") == ["tangaroa", "smokeBuoyancy"],
        "dataset scope drifted",
    )
    representations_raw = raw.get("representations")
    _require(isinstance(representations_raw, Mapping), "representations are missing")
    representations = tuple(representations_raw.get("order", ()))
    _require(representations == REPRESENTATIONS, "representation order drifted")
    for name, width in COMPOSITE_WIDTH.items():
        _require(int(representations_raw[name]["width"]) == width, f"{name} width drifted")
    score = raw.get("negative_template_score")
    fusion = raw.get("paired_center_fusion")
    candidates = raw.get("candidates")
    _require(all(isinstance(value, Mapping) for value in (score, fusion, candidates)), "numerical contract is incomplete")
    assert isinstance(score, Mapping) and isinstance(fusion, Mapping) and isinstance(candidates, Mapping)
    ks = tuple(int(value) for value in score.get("k_values", ()))
    spatial = score.get("spatial_transform")
    _require(isinstance(spatial, Mapping), "spatial transform is missing")
    sigmas = tuple(float(value) for value in spatial.get("gaussian_sigmas_grid_indices", ()))
    weights = tuple(float(value) for value in fusion.get("weights", ()))
    top_fractions = tuple(
        float(value) for value in fusion.get("top_fraction_values", ())
    )
    _require(
        ks == K_VALUES
        and sigmas == SIGMAS
        and weights == PAIR_WEIGHTS
        and top_fractions == TOP_FRACTIONS,
        "candidate axis drifted",
    )
    _require(
        fusion.get("top_fraction_selection") == "inner_physical_families_only"
        and fusion.get("prediction_population_per_dataset_source")
        == "all_64000_centers"
        and fusion.get("primary_metric_population")
        == "all_parent_valid_rows_after_exact_center_projection"
        and fusion.get("unique_center_metric_population")
        == "centers_with_at_least_one_valid_block"
        and fusion.get("neither_valid_centers_in_primary_classification_metrics")
        is False,
        "center decision contract drifted",
    )
    _require(int(candidates.get("exact_count", -1)) == FROZEN_CANDIDATE_COUNT, "frozen candidate count drifted")
    _require(
        int(candidates.get("representations", -1)) == len(REPRESENTATIONS)
        and int(candidates.get("k_values", -1)) == len(K_VALUES)
        and int(candidates.get("spatial_sigmas", -1)) == len(SIGMAS)
        and int(candidates.get("paired_weights", -1)) == len(PAIR_WEIGHTS)
        and int(candidates.get("decision_values", -1)) == len(TOP_FRACTIONS)
        and candidates.get("outer_result_selection") == "forbidden",
        "candidate grid or outer selection contract drifted",
    )
    output = raw.get("output")
    _require(isinstance(output, Mapping) and output.get("overwrite") == "forbidden", "output contract drifted")
    reported = raw.get("reported_arms")
    _require(isinstance(reported, Mapping), "reported arms are missing")
    _require(
        reported["direct_source_centered_min_dx_top5"]["status"]
        == "direct_kinematic_diagnostic_not_template_matching_success"
        and reported["direct_source_centered_dx_rank_mean_top5"]["status"]
        == "scale_robust_direct_kinematic_diagnostic_not_template_matching_success",
        "direct-diagnostic evidence boundary drifted",
    )
    return Plan(
        path=path,
        sha256=digest,
        raw=MappingProxyType(dict(raw)),
        parent_plan=parent_plan,
        family_order=family_order,
        families=MappingProxyType(families),
        dataset_to_family=MappingProxyType(
            {dataset: family for family in family_order for dataset in families[family]}
        ),
        representations=representations,
        ks=ks,
        sigmas=sigmas,
        weights=weights,
        top_fractions=top_fractions,
        grid_shape=tuple(int(value) for value in spatial["grid_shape_zyx"]),
        gaussian_truncate=float(spatial["gaussian_truncate"]),
        query_chunk_size=int(score["query_chunk_size"]),
        library_chunk_size=int(score["library_chunk_size"]),
        shrinkage_lambda=float(score["calibration_shrinkage_lambda"]),
        output_root=Path(str(output["root"])),
        required_fold_files=REQUIRED_FOLD_FILES,
    )


def composite_representation_features(
    cache: SourceCenteredCacheProjection,
    representation: str,
) -> np.ndarray:
    _require(representation in PARENT_REPRESENTATION, "unknown composite representation")
    parent = representation_features(
        cache.fmt_features, PARENT_REPRESENTATION[representation]
    )
    result = np.ascontiguousarray(
        np.concatenate((parent, cache.source_centered_seed4), axis=1),
        dtype=np.float32,
    )
    _require(
        result.shape == (cache.count, COMPOSITE_WIDTH[representation])
        and np.isfinite(result).all(),
        "source-centered composite feature contract drifted",
    )
    return result


_POPULATION_FIELDS = frozenset(
    {
        "schema",
        "experiment",
        "status",
        "git_commit",
        "worktree_clean",
        "verify_config_sha256",
        "source_file_sha256",
        "source_file_sha256_content_sha256",
        "input_manifest_path",
        "input_manifest_file_sha256",
        "input_manifest_content_sha256",
        "sidecar_count",
        "assigned_row_count_total",
        "valid_projection_row_count_total",
        "rows",
        "rows_content_sha256",
        "forbidden_parent_members_opened",
        "forbidden_dataset_access",
        "manifest_write_order",
        "content_sha256",
    }
)
_POPULATION_ROW_FIELDS = frozenset(
    {
        "dataset",
        "dataset_index",
        "physical_family",
        "source_ordinal",
        "source_index",
        "completion_relative_path",
        "completion_size_bytes",
        "completion_file_sha256",
        "sidecar_relative_path",
        "sidecar_size_bytes",
        "sidecar_file_sha256",
        "sidecar_combined_array_sha256",
        "valid_projection_sha256",
        "assigned_row_count",
        "valid_projection_row_count",
    }
)


def _read_self_hashed_json(
    path: Path,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    payload = early._read_authenticated_bytes(
        path, expected_sha256=expected_file_sha256
    )
    value = json.loads(payload.decode("utf-8"))
    _require(isinstance(value, dict), f"JSON root is invalid: {path}")
    content = dict(value)
    claimed = content.pop("content_sha256", None)
    _require(
        isinstance(claimed, str)
        and len(claimed) == 64
        and claimed == canonical_json_sha256(content),
        f"JSON content hash drifted: {path}",
    )
    return value


def bind_source_centered_evidence(
    plan: Plan,
    *,
    input_manifest_path: str | Path,
    input_manifest_file_sha256: str,
    sidecar_root: str | Path,
    population_manifest_path: str | Path,
    population_manifest_file_sha256: str,
) -> Plan:
    """Bind the complete sidecar envelope without opening any NPZ member."""

    identity = capture_clean_source_identity(ROOT)
    _require(identity.git_commit == _git_identity()[0], "source identity Git commit drifted")
    input_path = Path(input_manifest_path).resolve()
    root = Path(sidecar_root).resolve()
    population_path = Path(population_manifest_path).resolve()
    _require(
        root == plan.output_root / "source_centered_cache" / "train",
        "runtime sidecar root differs from frozen output contract",
    )
    _require(population_path == root / "SIDECAR_POPULATION.json", "population path drifted")
    input_manifest = _read_self_hashed_json(
        input_path, expected_file_sha256=input_manifest_file_sha256
    )
    _require(
        input_manifest.get("schema") == INPUT_MANIFEST_SCHEMA
        and input_manifest.get("experiment") == EXPERIMENT
        and input_manifest.get("status") == "frozen"
        and input_manifest.get("git_commit") == identity.git_commit
        and input_manifest.get("worktree_clean") is True
        and input_manifest.get("verify_config_sha256") == plan.sha256,
        "source-centered input manifest provenance drifted",
    )
    input_rows = input_manifest.get("rows")
    _require(
        isinstance(input_rows, list)
        and len(input_rows) == 32
        and input_manifest.get("rows_content_sha256")
        == canonical_json_sha256(input_rows),
        "source-centered input row population drifted",
    )
    expected_keys = [
        (dataset, source_ordinal)
        for family in plan.family_order
        for dataset in plan.families[family]
        for source_ordinal in range(4)
    ]
    _require(
        [(str(row["dataset"]), int(row["source_ordinal"])) for row in input_rows]
        == expected_keys,
        "source-centered input rows are missing, duplicated, extra, or reordered",
    )

    population = _read_self_hashed_json(
        population_path, expected_file_sha256=population_manifest_file_sha256
    )
    _require(set(population) == _POPULATION_FIELDS, "population field set drifted")
    expected_top = {
        "schema": POPULATION_MANIFEST_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "passed",
        "git_commit": identity.git_commit,
        "worktree_clean": True,
        "verify_config_sha256": plan.sha256,
        "source_file_sha256": dict(identity.source_file_sha256_items),
        "source_file_sha256_content_sha256": identity.source_content_sha256,
        "input_manifest_path": str(input_path),
        "input_manifest_file_sha256": input_manifest_file_sha256,
        "input_manifest_content_sha256": input_manifest["content_sha256"],
        "sidecar_count": 32,
        "assigned_row_count_total": 32 * ASSIGNED_ROW_COUNT,
        "forbidden_parent_members_opened": [],
        "forbidden_dataset_access": False,
        "manifest_write_order": "last_after_all_32_sidecars_and_completions_were_freshly_replayed",
    }
    drift = {
        name: (population.get(name), expected)
        for name, expected in expected_top.items()
        if population.get(name) != expected
    }
    _require(not drift, f"population provenance drifted: {drift}")
    rows = population.get("rows")
    _require(
        isinstance(rows, list)
        and len(rows) == 32
        and population.get("rows_content_sha256") == canonical_json_sha256(rows),
        "population row set/hash drifted",
    )
    expected_files = {population_path}
    valid_total = 0
    for input_row, row in zip(input_rows, rows, strict=True):
        _require(isinstance(row, Mapping) and set(row) == _POPULATION_ROW_FIELDS, "population row fields drifted")
        expected_identity = {
            "dataset": input_row["dataset"],
            "dataset_index": input_row["dataset_index"],
            "physical_family": input_row["physical_family"],
            "source_ordinal": input_row["source_ordinal"],
            "source_index": input_row["source_index"],
        }
        _require(
            all(row.get(name) == value for name, value in expected_identity.items()),
            "population/input identity drifted",
        )
        completion = (root / str(row["completion_relative_path"])).resolve()
        sidecar = (root / str(row["sidecar_relative_path"])).resolve()
        _require(completion.is_relative_to(root) and sidecar.is_relative_to(root), "population child escapes sidecar root")
        expected_files.update((completion, sidecar))
        for child, size_name, hash_name in (
            (completion, "completion_size_bytes", "completion_file_sha256"),
            (sidecar, "sidecar_size_bytes", "sidecar_file_sha256"),
        ):
            _stable_file_identity(child, int(row[size_name]), str(row[hash_name]))
        completion_value = _read_self_hashed_json(
            completion, expected_file_sha256=str(row["completion_file_sha256"])
        )
        _require(
            completion_value.get("schema") == ROW_COMPLETION_SCHEMA
            and completion_value.get("experiment") == EXPERIMENT
            and completion_value.get("dataset") == row["dataset"]
            and completion_value.get("source_ordinal") == row["source_ordinal"]
            and completion_value.get("sidecar_file_sha256")
            == row["sidecar_file_sha256"]
            and completion_value.get("sidecar_combined_array_sha256")
            == row["sidecar_combined_array_sha256"]
            and completion_value.get("valid_projection_sha256")
            == row["valid_projection_sha256"],
            "population/completion binding drifted",
        )
        _require(int(row["assigned_row_count"]) == ASSIGNED_ROW_COUNT, "assigned row count drifted")
        valid_total += int(row["valid_projection_row_count"])
    actual_files = {child.resolve() for child in root.rglob("*") if child.is_file()}
    _require(actual_files == expected_files, "sidecar population file set drifted")
    _require(int(population["valid_projection_row_count_total"]) == valid_total, "valid projection total drifted")
    return replace(
        plan,
        sidecar_input_manifest_path=input_path,
        sidecar_input_manifest_file_sha256=input_manifest_file_sha256,
        sidecar_input_manifest_content_sha256=str(input_manifest["content_sha256"]),
        sidecar_root=root,
        sidecar_population_path=population_path,
        sidecar_population_file_sha256=population_manifest_file_sha256,
        sidecar_population_content_sha256=str(population["content_sha256"]),
        sidecar_population=early._deep_freeze(population),
    )


def _require_evidence_bound(plan: Plan) -> None:
    _require(plan.sidecar_root is not None, "sidecar root is unbound")
    _require(plan.sidecar_population is not None, "sidecar population is unbound")
    _require(plan.sidecar_population_path is not None, "population path is unbound")


def _population_row(plan: Plan, row: CacheRow) -> Mapping[str, Any]:
    _require_evidence_bound(plan)
    assert plan.sidecar_population is not None
    matches = [
        value
        for value in plan.sidecar_population["rows"]
        if value["dataset"] == row.dataset
        and value["physical_family"] == row.family
        and int(value["source_ordinal"]) == row.source_ordinal
        and int(value["source_index"]) == row.source_index
    ]
    _require(len(matches) == 1, "sidecar population row does not resolve uniquely")
    return matches[0]


def _load_parent_projection(
    plan: Plan,
    row: CacheRow,
    *,
    include_references: bool,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    names = [
        "fmt_features",
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
    ]
    if include_references:
        names.extend(("valid_labels", "reference_labels_all"))
    with early._authenticated_open_file(
        row.path, expected_size=row.size_bytes, expected_sha256=row.sha256
    ) as opened:
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(all(name in archive.files for name in names), f"{row.path}: parent projection is incomplete")
            arrays = {name: np.array(archive[name], copy=True, order="C") for name in names}
            if include_references:
                metadata_scalar = np.array(archive["metadata_json"], copy=True)
                _require(metadata_scalar.shape == (), "parent metadata_json is not scalar")
                metadata: Mapping[str, Any] = json.loads(str(metadata_scalar.item()))
            else:
                metadata = {}
    base_arrays = {name: arrays[name] for name in names[:5]}
    if include_references:
        base_arrays["valid_labels"] = arrays["valid_labels"]
        _validate_cache_arrays(plan.parent_plan, row, metadata, base_arrays)
        reference = arrays["reference_labels_all"]
        _require(
            reference.dtype == np.dtype(np.bool_)
            and reference.shape == (2 * DEFAULT_CENTER_COUNT,)
            and metadata.get("array_sha256", {}).get("reference_labels_all")
            == canonical_array_sha256(reference),
            "parent reference_labels_all contract drifted",
        )
        _require(
            np.array_equal(reference[:DEFAULT_CENTER_COUNT], reference[DEFAULT_CENTER_COUNT:]),
            "legacy and expanded assigned rows disagree on the unique-center reference",
        )
    else:
        _validate_outer_prediction_projection(row, base_arrays)
    return arrays, metadata


def load_source_centered_cache_projection(
    plan: Plan,
    row: CacheRow,
    *,
    include_references: bool,
) -> SourceCenteredCacheProjection:
    """Open one sidecar only at the caller-authorized feature phase."""

    parent, metadata = _load_parent_projection(
        plan, row, include_references=include_references
    )
    population_row = _population_row(plan, row)
    assert plan.sidecar_root is not None
    sidecar_path = (plan.sidecar_root / str(population_row["sidecar_relative_path"])).resolve()
    _require(sidecar_path.is_relative_to(plan.sidecar_root), "sidecar path escapes root")
    loaded = load_source_centered_sidecar(
        sidecar_path,
        expected_file_sha256=str(population_row["sidecar_file_sha256"]),
    )
    payload = loaded.payload
    identities = (
        (payload.valid_assigned_row_index, parent["valid_assigned_row_index"], "assigned row"),
        (payload.valid_center_seed_index, parent["valid_center_seed_index"], "center"),
        (payload.valid_scale_block_index, parent["valid_scale_block_index"], "block"),
        (payload.valid_scale_id, parent["valid_scale_id"], "scale"),
    )
    for sidecar_values, parent_values, label in identities:
        _require(np.array_equal(sidecar_values, parent_values), f"sidecar/parent {label} join drifted")
    _require(
        loaded.metadata.get("schema") == SIDECAR_SCHEMA
        and loaded.metadata.get("combined_array_sha256")
        == population_row["sidecar_combined_array_sha256"]
        and loaded.metadata.get("valid_projection", {}).get("canonical_sha256")
        == population_row["valid_projection_sha256"],
        "sidecar population binding drifted",
    )
    valid_feature = np.ascontiguousarray(
        payload.source_centered_seed4[payload.valid_assigned_row_index],
        dtype=np.float32,
    )
    valid_labels = None if not include_references else np.ascontiguousarray(parent["valid_labels"])
    center_labels = (
        None
        if not include_references
        else np.ascontiguousarray(parent["reference_labels_all"][:DEFAULT_CENTER_COUNT])
    )
    return SourceCenteredCacheProjection(
        row=row,
        fmt_features=np.ascontiguousarray(parent["fmt_features"]),
        source_centered_seed4=valid_feature,
        scale_ids=np.ascontiguousarray(parent["valid_scale_id"]),
        center_indices=np.ascontiguousarray(parent["valid_center_seed_index"]),
        block_indices=np.ascontiguousarray(parent["valid_scale_block_index"]),
        assigned_row_indices=np.ascontiguousarray(parent["valid_assigned_row_index"]),
        labels=valid_labels,
        center_labels=center_labels,
        metadata=metadata,
        assigned_center_indices=np.ascontiguousarray(payload.center_seed_index),
        assigned_block_indices=np.ascontiguousarray(payload.scale_block_index),
        assigned_scale_ids=np.ascontiguousarray(payload.scale_id),
        assigned_physical_dx=np.ascontiguousarray(
            payload.physical_dx_by_scale[payload.scale_id], dtype=np.float64
        ),
        assigned_source_centered_seed4=np.ascontiguousarray(
            payload.source_centered_seed4
        ),
        sidecar_file_sha256=loaded.file_sha256,
        sidecar_combined_array_sha256=str(loaded.metadata["combined_array_sha256"]),
        sidecar_group_mean_curl_xyz=np.ascontiguousarray(
            payload.group_mean_curl_xyz, dtype=np.float64
        ),
        sidecar_group_mean_curl_xyz_sha256=canonical_array_sha256(
            payload.group_mean_curl_xyz
        ),
    )


def _evidence_binding(
    plan: Plan,
    *,
    representation: str,
    fit_families: Sequence[str],
) -> dict[str, Any]:
    _require_evidence_bound(plan)
    return {
        "source_centered_input_manifest": {
            "path": str(plan.sidecar_input_manifest_path),
            "file_sha256": plan.sidecar_input_manifest_file_sha256,
            "content_sha256": plan.sidecar_input_manifest_content_sha256,
        },
        "source_centered_population_manifest": {
            "path": str(plan.sidecar_population_path),
            "file_sha256": plan.sidecar_population_file_sha256,
            "content_sha256": plan.sidecar_population_content_sha256,
            "sidecar_count": 32,
        },
        "representation": representation,
        "fit_families": list(fit_families),
        "config_sha256": plan.sha256,
    }


def _fit_tail_model(
    caches: Sequence[SourceCenteredCacheProjection],
    representation: str,
    plan: Plan,
    *,
    device: str,
    ks: Sequence[int] | None = None,
) -> PerScaleNegativeTailModel:
    feature_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    for cache in caches:
        _require(cache.labels is not None, "fit labels are unavailable")
        negative = ~np.asarray(cache.labels, dtype=np.bool_)
        if negative.any():
            represented = composite_representation_features(cache, representation)
            feature_parts.append(np.ascontiguousarray(represented[negative], dtype=np.float32))
            scale_parts.append(np.ascontiguousarray(cache.scale_ids[negative], dtype=np.int64))
    _require(feature_parts, "fit families contain no natural negatives")
    features = np.ascontiguousarray(np.concatenate(feature_parts), dtype=np.float32)
    scales = np.ascontiguousarray(np.concatenate(scale_parts), dtype=np.int64)
    model = PerScaleNegativeTailModel(
        features,
        scales,
        ks=plan.ks if ks is None else tuple(int(value) for value in ks),
        shrinkage_lambda=plan.shrinkage_lambda,
        device=device,
        query_chunk_size=plan.query_chunk_size,
        library_chunk_size=plan.library_chunk_size,
    )
    del features, scales, feature_parts, scale_parts
    return model


def _query_cache_batch(
    model: PerScaleNegativeTailModel,
    caches: Sequence[SourceCenteredCacheProjection],
    representation: str,
    plan: Plan,
    *,
    device: str,
    ks: Sequence[int] | None = None,
) -> dict[int, list[dict[str, np.ndarray]]]:
    requested = model.ks if ks is None else tuple(int(value) for value in ks)
    offsets = np.cumsum([0, *(cache.count for cache in caches)], dtype=np.int64)
    features = np.ascontiguousarray(
        np.concatenate(
            [composite_representation_features(cache, representation) for cache in caches]
        ),
        dtype=np.float32,
    )
    scales = np.ascontiguousarray(
        np.concatenate([cache.scale_ids for cache in caches]), dtype=np.int64
    )
    result = model.query(
        features,
        scales,
        ks=requested,
        device=device,
        query_chunk_size=plan.query_chunk_size,
        library_chunk_size=plan.library_chunk_size,
    )
    scaler_modes = model.scaler.mode_for_scales(scales)
    del features, scales
    output: dict[int, list[dict[str, np.ndarray]]] = {}
    for k in requested:
        groups: list[dict[str, np.ndarray]] = []
        for index in range(len(caches)):
            selected = slice(int(offsets[index]), int(offsets[index + 1]))
            groups.append(
                {
                    "raw_distance": result.raw_distances[k][selected],
                    "tail_probability": result.tail_probabilities[k][selected],
                    "tail_anomaly": result.anomaly_scores[k][selected],
                    "retrieval_supported": result.retrieval_supported[k][selected],
                    "calibration_supported": result.calibration_supported[k][selected],
                    "calibration_mode": result.calibration_modes[k][selected],
                    "scaler_mode": scaler_modes[selected],
                }
            )
        output[k] = groups
    return output


def _row_spatial_scores(
    cache: SourceCenteredCacheProjection,
    values: Mapping[str, np.ndarray],
    *,
    sigma: float,
    plan: Plan,
) -> dict[str, np.ndarray]:
    count = cache.count
    spatial_score = np.zeros(count, dtype=np.float64)
    spatial_denominator = np.zeros(count, dtype=np.float64)
    imputed = np.zeros(count, dtype=np.bool_)
    unimputable = np.zeros(count, dtype=np.bool_)
    for block_index in (0, 1):
        selected = cache.block_indices == block_index
        _require(selected.any(), f"{cache.row.dataset}/{cache.row.source_ordinal}: empty block")
        spatial = early.spatial_calibrated_tail_scores(
            values["tail_anomaly"][selected],
            values["calibration_supported"][selected],
            cache.center_indices[selected],
            sigma=sigma,
            grid_shape=plan.grid_shape,
            truncate=plan.gaussian_truncate,
        )
        spatial_score[selected] = spatial.scores
        spatial_denominator[selected] = spatial.denominator
        imputed[selected] = spatial.imputed
        unimputable[selected] = spatial.unimputable
    supported = (
        np.asarray(values["calibration_supported"], dtype=np.bool_) | imputed
    ) & (spatial_score > 0.0)
    return {
        "spatial_score": spatial_score,
        "spatial_denominator": spatial_denominator,
        "spatial_imputed": imputed,
        "spatial_unimputable": unimputable,
        "score_supported": supported,
    }


def _classification_metric_values(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    *,
    ranking_metrics: tuple[float, float] | None = None,
) -> dict[str, Any]:
    reference = np.asarray(labels, dtype=np.bool_)
    values = np.asarray(scores, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.bool_)
    _require(reference.ndim == values.ndim == predicted.ndim == 1, "metric arrays must be vectors")
    _require(reference.shape == values.shape == predicted.shape and len(reference) > 0, "metric population is empty or misaligned")
    _require(np.isfinite(values).all(), "metric score is nonfinite")
    counts = _classification_counts(reference, predicted)
    if ranking_metrics is not None:
        average_precision, auroc = ranking_metrics
    elif reference.any() and (~reference).any():
        average_precision, auroc = _ranking_metrics_one_sort(reference, values)
    else:
        average_precision = auroc = float("nan")
    return {**counts, "average_precision": average_precision, "auroc": auroc}


INNER_METRIC_FIELDS = (
    "outer_family",
    "inner_family",
    "dataset",
    "source_ordinal",
    "candidate_id",
    "representation",
    "k",
    "sigma",
    "weight",
    "decision_rule",
    "decision_value",
    "sample_count",
    "positive_count",
    "negative_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "average_precision",
    "auroc",
    "combined_coverage",
    "both_valid_count",
    "legacy_only_count",
    "expanded_only_count",
    "neither_valid_count",
    "retrieval_support_fraction",
    "calibration_support_fraction",
    "spatial_imputed_fraction",
)
INNER_SUMMARY_FIELDS = (
    "candidate_id",
    "representation",
    "k",
    "sigma",
    "weight",
    "decision_rule",
    "decision_value",
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
    "combined_coverage",
    "retrieval_support_fraction",
    "calibration_support_fraction",
    "spatial_imputed_fraction",
    "inner_family_count",
    "group_count",
)


def _inner_metric_row(
    *,
    outer_family: str,
    inner_family: str,
    cache: SourceCenteredCacheProjection,
    candidate: CandidateSpec,
    values: Mapping[str, np.ndarray],
    spatial: Mapping[str, np.ndarray],
    fusion: PairedCenterFusion,
    ranking_metrics: tuple[float, float],
) -> dict[str, Any]:
    _require(cache.labels is not None, "inner labels are unavailable")
    row_scores = fusion.paired_score[cache.center_indices]
    metric = _classification_metric_values(
        np.asarray(cache.labels),
        row_scores,
        fusion.valid_row_prediction,
        ranking_metrics=ranking_metrics,
    )
    return {
        "outer_family": outer_family,
        "inner_family": inner_family,
        "dataset": cache.row.dataset,
        "source_ordinal": cache.row.source_ordinal,
        **_candidate_payload(candidate),
        **metric,
        "combined_coverage": fusion.combined_coverage,
        "both_valid_count": int(fusion.both_valid.sum()),
        "legacy_only_count": int(fusion.legacy_only.sum()),
        "expanded_only_count": int(fusion.expanded_only.sum()),
        "neither_valid_count": int(fusion.neither_valid.sum()),
        "retrieval_support_fraction": float(np.asarray(values["retrieval_supported"]).mean()),
        "calibration_support_fraction": float(np.asarray(values["calibration_supported"]).mean()),
        "spatial_imputed_fraction": float(np.asarray(spatial["spatial_imputed"]).mean()),
    }


def _inner_metric_rows(
    plan: Plan,
    caches: Sequence[SourceCenteredCacheProjection],
    outer_family: str,
    *,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    nonouter = [family for family in plan.family_order if family != outer_family]
    for inner_family in nonouter:
        fit_families = [family for family in nonouter if family != inner_family]
        fit_caches = [cache for cache in caches if cache.row.family in fit_families]
        query_caches = [cache for cache in caches if cache.row.family == inner_family]
        _require(fit_caches and query_caches, f"{inner_family}: empty nested split")
        for representation in plan.representations:
            print(
                f"[{_utc_now()}] outer={outer_family} inner={inner_family} "
                f"representation={representation} fit_start",
                flush=True,
            )
            model = _fit_tail_model(
                fit_caches, representation, plan, device=device
            )
            audit = dict(model.fit_audit)
            audit.update(
                {
                    "outer_family": outer_family,
                    "inner_family": inner_family,
                    "fit_families": fit_families,
                    "representation": representation,
                    "source_centered_population_manifest_file_sha256": plan.sidecar_population_file_sha256,
                    "device": device,
                }
            )
            audits.append(audit)
            query = _query_cache_batch(
                model, query_caches, representation, plan, device=device
            )
            for cache_index, cache in enumerate(query_caches):
                for k in plan.ks:
                    values = query[k][cache_index]
                    for sigma in plan.sigmas:
                        spatial = _row_spatial_scores(
                            cache, values, sigma=sigma, plan=plan
                        )
                        for weight in plan.weights:
                            base_fusion = fuse_paired_scale_centers(
                                cache.center_indices,
                                cache.block_indices,
                                spatial["spatial_score"],
                                spatial["score_supported"],
                                weight=weight,
                                center_count=DEFAULT_CENTER_COUNT,
                                top_fraction=plan.top_fractions[0],
                            )
                            _require(cache.labels is not None, "inner labels are unavailable")
                            inner_labels = np.asarray(cache.labels, dtype=np.bool_)
                            inner_scores = base_fusion.paired_score[
                                cache.center_indices
                            ]
                            if inner_labels.any() and (~inner_labels).any():
                                ranking_metrics = _ranking_metrics_one_sort(
                                    inner_labels, inner_scores
                                )
                            else:
                                ranking_metrics = (float("nan"), float("nan"))
                            for top_fraction in plan.top_fractions:
                                candidate = CandidateSpec(
                                    representation,
                                    k,
                                    sigma,
                                    weight,
                                    top_fraction,
                                )
                                if top_fraction == plan.top_fractions[0]:
                                    fusion = base_fusion
                                else:
                                    prediction = fixed_top_fraction_over_centers(
                                        base_fusion.paired_score,
                                        base_fusion.combined_eligible,
                                        fraction=top_fraction,
                                        require_strictly_positive_score=True,
                                    )
                                    fusion = replace(
                                        base_fusion,
                                        prediction=prediction,
                                        valid_row_prediction=prediction[
                                            cache.center_indices
                                        ],
                                        top_fraction=top_fraction,
                                    )
                                rows.append(
                                    _inner_metric_row(
                                        outer_family=outer_family,
                                        inner_family=inner_family,
                                        cache=cache,
                                        candidate=candidate,
                                        values=values,
                                        spatial=spatial,
                                        fusion=fusion,
                                        ranking_metrics=ranking_metrics,
                                    )
                                )
            del model, query
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    expected_candidates = {value.candidate_id for value in candidate_specs(plan)}
    _require({str(row["candidate_id"]) for row in rows} == expected_candidates, "inner candidate population drifted")
    return rows, audits


def _hierarchical_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values_by_family: list[float] = []
    for family in sorted({str(row["inner_family"]) for row in rows}):
        values = np.asarray(
            [float(row[field]) for row in rows if row["inner_family"] == family],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        values_by_family.append(float(np.mean(finite)) if len(finite) else float("nan"))
    family_values = np.asarray(values_by_family, dtype=np.float64)
    family_values = family_values[np.isfinite(family_values)]
    return float(np.mean(family_values)) if len(family_values) else float("nan")


def aggregate_and_select_inner(
    plan: Plan,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], CandidateSpec, dict[str, Any]]:
    candidates = {value.candidate_id: value for value in candidate_specs(plan)}
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in candidates}
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    expected_keys: set[tuple[str, str, int]] | None = None
    summaries: list[dict[str, Any]] = []
    aggregate_fields = (
        "accuracy",
        "average_precision",
        "f1",
        "balanced_accuracy",
        "auroc",
        "precision",
        "recall",
        "combined_coverage",
        "retrieval_support_fraction",
        "calibration_support_fraction",
        "spatial_imputed_fraction",
    )
    for candidate_id in sorted(candidates):
        candidate_rows = grouped[candidate_id]
        keys = {
            (str(row["inner_family"]), str(row["dataset"]), int(row["source_ordinal"]))
            for row in candidate_rows
        }
        _require(len(keys) == len(candidate_rows), f"{candidate_id}: duplicate inner group")
        if expected_keys is None:
            expected_keys = keys
        _require(keys == expected_keys and len({key[0] for key in keys}) == 4, f"{candidate_id}: incomplete inner group set")
        summary = _candidate_payload(candidates[candidate_id])
        for field in aggregate_fields:
            summary[field] = _hierarchical_mean(candidate_rows, field)
        summary["inner_family_count"] = 4
        summary["group_count"] = len(candidate_rows)
        summaries.append(summary)

    def key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -float(summary["f1"]),
            -float(summary["average_precision"]),
            -float(summary["balanced_accuracy"]),
            -float(summary["precision"]),
            -float(summary["recall"]),
            str(summary["candidate_id"]),
        )

    selected_summary = min(summaries, key=key)
    selected = candidates[str(selected_summary["candidate_id"])]
    return summaries, selected, dict(selected_summary)


def _manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return early._manifest_with_self_hash(payload)


def _read_manifest(path: Path, expected_file_sha256: str) -> tuple[dict[str, Any], str]:
    payload = early._read_authenticated_bytes(path, expected_sha256=expected_file_sha256)
    value = json.loads(payload.decode("utf-8"))
    _require(isinstance(value, dict), f"manifest root is invalid: {path}")
    early._authenticate_self_hash(value)
    return value, hashlib.sha256(payload).hexdigest()


def _verify_npz_arrays(
    path: Path,
    *,
    file_record: Mapping[str, Any],
    records: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], str]:
    with early._authenticated_open_file(
        path,
        expected_size=int(file_record["size_bytes"]),
        expected_sha256=str(file_record["sha256"]),
    ) as opened:
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(set(archive.files) == set(records), f"NPZ member set drifted: {path}")
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
        file_sha = opened.sha256
    for name, values in arrays.items():
        record = records[name]
        _require(
            values.dtype.str == record.get("dtype")
            and list(values.shape) == record.get("shape")
            and canonical_array_sha256(values) == record.get("sha256"),
            f"NPZ array contract drifted: {path}/{name}",
        )
    return arrays, file_sha


def write_final_scaler(
    destination: Path,
    model: PerScaleNegativeTailModel,
    *,
    plan: Plan,
    selected: CandidateSpec,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
) -> tuple[Path, Path, str]:
    arrays = model.scaler.export_arrays()
    _require(tuple(arrays) == SCALER_ARRAY_NAMES, "scaler export order drifted")
    artifact_path = destination / "final_per_scale_scaler.npz"
    artifact_sha = early._atomic_npz(artifact_path, arrays)
    value = _manifest(
        {
            "schema": SCALER_MANIFEST_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_path": str(plan.path),
            "config_sha256": plan.sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "fit_families": list(fit_families),
            "selected_candidate": _candidate_payload(selected),
            "source_centered_evidence": _evidence_binding(
                plan,
                representation=selected.representation,
                fit_families=fit_families,
            ),
            "outer_source_centered_member_opened": False,
            "artifact_file": {
                "path": artifact_path.name,
                "size_bytes": artifact_path.stat().st_size,
                "sha256": artifact_sha,
            },
            "arrays": early._array_manifest(arrays),
            "fit_audit": model.scaler.fit_audit,
        }
    )
    manifest_path = destination / "final_per_scale_scaler_manifest.json"
    manifest_sha = early._atomic_json(manifest_path, value)
    return artifact_path, manifest_path, manifest_sha


def authenticate_final_scaler(
    artifact_path: Path,
    manifest_path: Path,
    *,
    plan: Plan,
    selected: CandidateSpec,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
    expected_manifest_sha256: str,
) -> VerifiedScaler:
    value, file_sha = _read_manifest(manifest_path, expected_manifest_sha256)
    _require(
        value.get("schema") == SCALER_MANIFEST_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("config_sha256") == plan.sha256
        and value.get("git_commit") == git_commit
        and value.get("outer_family") == outer_family
        and value.get("fit_families") == list(fit_families)
        and value.get("selected_candidate") == _json_safe(_candidate_payload(selected))
        and value.get("source_centered_evidence")
        == _evidence_binding(
            plan,
            representation=selected.representation,
            fit_families=fit_families,
        )
        and value.get("outer_source_centered_member_opened") is False,
        "scaler manifest provenance drifted",
    )
    records = value.get("arrays")
    _require(isinstance(records, Mapping) and set(records) == set(SCALER_ARRAY_NAMES), "scaler array records drifted")
    arrays, artifact_sha = _verify_npz_arrays(
        artifact_path, file_record=value["artifact_file"], records=records
    )
    scaler = PerScaleNegativeScaler.from_arrays(arrays)
    _require(_json_safe(scaler.fit_audit) == value.get("fit_audit"), "rebuilt scaler audit drifted")
    return VerifiedScaler(
        manifest_path,
        file_sha,
        artifact_sha,
        early._deep_freeze(value),
        scaler,
        _AUTHENTICATION_SEAL,
    )


def write_final_calibration(
    destination: Path,
    model: PerScaleNegativeTailModel,
    scaler: VerifiedScaler,
    *,
    plan: Plan,
    selected: CandidateSpec,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
) -> tuple[Path, Path, str]:
    _require(scaler.seal is _AUTHENTICATION_SEAL, "calibration requires authenticated scaler")
    arrays = model.tail_calibrator.export_arrays()
    artifact_path = destination / "final_tail_calibration.npz"
    artifact_sha = early._atomic_npz(artifact_path, arrays)
    value = _manifest(
        {
            "schema": CALIBRATION_MANIFEST_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_sha256": plan.sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "fit_families": list(fit_families),
            "selected_candidate": _candidate_payload(selected),
            "source_centered_evidence": _evidence_binding(
                plan,
                representation=selected.representation,
                fit_families=fit_families,
            ),
            "scaler_manifest": {
                "path": scaler.manifest_path.name,
                "file_sha256": scaler.manifest_file_sha256,
                "content_sha256": scaler.manifest["content_sha256"],
                "artifact_file_sha256": scaler.artifact_file_sha256,
            },
            "outer_source_centered_member_opened": False,
            "artifact_file": {
                "path": artifact_path.name,
                "size_bytes": artifact_path.stat().st_size,
                "sha256": artifact_sha,
            },
            "arrays": early._array_manifest(arrays),
            "fit_audit": model.tail_calibrator.fit_audit,
        }
    )
    manifest_path = destination / "final_tail_calibration_manifest.json"
    manifest_sha = early._atomic_json(manifest_path, value)
    return artifact_path, manifest_path, manifest_sha


def authenticate_final_calibration(
    artifact_path: Path,
    manifest_path: Path,
    scaler: VerifiedScaler,
    *,
    plan: Plan,
    selected: CandidateSpec,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
    expected_manifest_sha256: str,
) -> VerifiedCalibration:
    _require(scaler.seal is _AUTHENTICATION_SEAL, "calibration authentication requires scaler")
    value, file_sha = _read_manifest(manifest_path, expected_manifest_sha256)
    expected_scaler = {
        "path": scaler.manifest_path.name,
        "file_sha256": scaler.manifest_file_sha256,
        "content_sha256": scaler.manifest["content_sha256"],
        "artifact_file_sha256": scaler.artifact_file_sha256,
    }
    _require(
        value.get("schema") == CALIBRATION_MANIFEST_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("config_sha256") == plan.sha256
        and value.get("git_commit") == git_commit
        and value.get("outer_family") == outer_family
        and value.get("fit_families") == list(fit_families)
        and value.get("selected_candidate") == _json_safe(_candidate_payload(selected))
        and value.get("source_centered_evidence")
        == _evidence_binding(
            plan,
            representation=selected.representation,
            fit_families=fit_families,
        )
        and value.get("scaler_manifest") == expected_scaler
        and value.get("outer_source_centered_member_opened") is False,
        "calibration manifest provenance drifted",
    )
    records = value.get("arrays")
    _require(isinstance(records, Mapping) and records, "calibration array records are missing")
    arrays, artifact_sha = _verify_npz_arrays(
        artifact_path, file_record=value["artifact_file"], records=records
    )
    model = PerScaleNegativeTailModel.from_artifacts(
        scaler.scaler.export_arrays(), arrays
    )
    _require(model.ks == (selected.k,), "rebuilt calibrator k drifted")
    _require(_json_safe(model.tail_calibrator.fit_audit) == value.get("fit_audit"), "rebuilt calibration audit drifted")
    return VerifiedCalibration(
        manifest_path,
        file_sha,
        artifact_sha,
        early._deep_freeze(value),
        model,
        _AUTHENTICATION_SEAL,
    )


def _inner_evidence_identity(
    path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    identity = _stable_file_identity(path, path.stat().st_size, expected_sha256)
    return {
        "path": path.name,
        "size_bytes": identity["size_bytes"],
        "sha256": identity["sha256"],
    }


def write_selected_candidate(
    destination: Path,
    *,
    plan: Plan,
    selected: CandidateSpec,
    selected_summary: Mapping[str, Any],
    scaler: VerifiedScaler,
    calibration: VerifiedCalibration,
    inner_paths: Mapping[str, tuple[Path, str]],
    outer_family: str,
    git_commit: str,
) -> tuple[Path, str]:
    _require(scaler.seal is calibration.seal is _AUTHENTICATION_SEAL, "selection requires authenticated model artifacts")
    fit_families = [family for family in plan.family_order if family != outer_family]
    value = _manifest(
        {
            "schema": SELECTED_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_sha256": plan.sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "candidate_count": FROZEN_CANDIDATE_COUNT,
            "candidate": _candidate_payload(selected),
            "inner_selection_summary": dict(selected_summary),
            "inner_evidence": {
                name: _inner_evidence_identity(path, digest)
                for name, (path, digest) in inner_paths.items()
            },
            "source_centered_evidence": _evidence_binding(
                plan,
                representation=selected.representation,
                fit_families=fit_families,
            ),
            "scaler_manifest": {
                "path": scaler.manifest_path.name,
                "file_sha256": scaler.manifest_file_sha256,
                "content_sha256": scaler.manifest["content_sha256"],
                "artifact_file_sha256": scaler.artifact_file_sha256,
            },
            "calibration_manifest": {
                "path": calibration.manifest_path.name,
                "file_sha256": calibration.manifest_file_sha256,
                "content_sha256": calibration.manifest["content_sha256"],
                "artifact_file_sha256": calibration.artifact_file_sha256,
            },
            "outer_source_centered_member_opened": False,
        }
    )
    path = destination / "selected_candidate.json"
    return path, early._atomic_json(path, value)


def authenticate_selected_candidate(
    path: Path,
    *,
    plan: Plan,
    selected: CandidateSpec,
    selected_summary: Mapping[str, Any],
    scaler: VerifiedScaler,
    calibration: VerifiedCalibration,
    inner_paths: Mapping[str, tuple[Path, str]],
    outer_family: str,
    git_commit: str,
    expected_file_sha256: str,
) -> VerifiedSelection:
    value, file_sha = _read_manifest(path, expected_file_sha256)
    fit_families = [family for family in plan.family_order if family != outer_family]
    _require(
        value.get("schema") == SELECTED_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("config_sha256") == plan.sha256
        and value.get("git_commit") == git_commit
        and value.get("outer_family") == outer_family
        and int(value.get("candidate_count", -1)) == FROZEN_CANDIDATE_COUNT
        and value.get("candidate") == _json_safe(_candidate_payload(selected))
        and value.get("inner_selection_summary") == _json_safe(dict(selected_summary))
        and value.get("source_centered_evidence")
        == _evidence_binding(
            plan,
            representation=selected.representation,
            fit_families=fit_families,
        )
        and value.get("outer_source_centered_member_opened") is False,
        "selected-candidate provenance drifted",
    )
    for name, (evidence_path, digest) in inner_paths.items():
        _require(value["inner_evidence"][name] == _inner_evidence_identity(evidence_path, digest), f"selected inner evidence drifted: {name}")
    _require(
        value["scaler_manifest"]["file_sha256"] == scaler.manifest_file_sha256
        and value["scaler_manifest"]["artifact_file_sha256"] == scaler.artifact_file_sha256
        and value["calibration_manifest"]["file_sha256"] == calibration.manifest_file_sha256
        and value["calibration_manifest"]["artifact_file_sha256"] == calibration.artifact_file_sha256,
        "selected model-artifact binding drifted",
    )
    return VerifiedSelection(
        path, file_sha, early._deep_freeze(value), _AUTHENTICATION_SEAL
    )


@dataclass(frozen=True)
class AuthenticatedParentControl:
    run_directory: Path
    outer_family: str
    arrays: Mapping[str, np.ndarray]
    evidence: Mapping[str, Any]


def authenticate_parent_control(
    plan: Plan,
    outer_family: str,
) -> AuthenticatedParentControl:
    """Authenticate the inherited Early prediction bytes as a control.

    The parent fold was already independently fresh-replayed by job 51070392.
    This child version does not reinterpret or select from its metrics: it
    reauthenticates the immutable completion/result/prediction hash chain and
    joins only the label-free score/prediction arrays by exact valid-row ID.
    """

    parent = plan.raw.get("direct_parent")
    _require(isinstance(parent, Mapping), "direct parent is missing")
    directories = parent.get("outer_run_directories")
    _require(isinstance(directories, Mapping) and set(directories) == set(plan.family_order), "parent outer-run directory map drifted")
    run_directory = Path(str(directories[outer_family])).resolve()
    _require(run_directory.is_dir(), f"parent run directory does not exist: {run_directory}")
    completion_path = run_directory / "RUN_COMPLETE.json"
    completion_sha = sha256_file(completion_path)
    completion = _read_self_hashed_json(
        completion_path, expected_file_sha256=completion_sha
    )
    _require(
        completion.get("schema") == early.COMPLETE_SCHEMA
        and completion.get("experiment") == PARENT_EXPERIMENT
        and completion.get("outer_family") == outer_family
        and completion.get("git_commit") == PARENT_NUMERICAL_COMMIT
        and completion.get("config_sha256") == PARENT_CONFIG_SHA256
        and completion.get("result_manifest_file") == "result_manifest.json",
        "parent completion provenance drifted",
    )
    result_path = run_directory / "result_manifest.json"
    result = _read_self_hashed_json(
        result_path,
        expected_file_sha256=str(completion["result_manifest_file_sha256"]),
    )
    _require(
        result.get("schema") == early.RESULT_SCHEMA
        and result.get("experiment") == PARENT_EXPERIMENT
        and result.get("status") == "completed"
        and result.get("outer_family") == outer_family
        and result.get("git_commit") == PARENT_NUMERICAL_COMMIT
        and result.get("config_sha256") == PARENT_CONFIG_SHA256
        and result.get("content_sha256")
        == completion.get("result_manifest_content_sha256"),
        "parent result provenance drifted",
    )
    artifacts = result.get("artifacts")
    _require(isinstance(artifacts, Mapping), "parent result artifact map is missing")
    prediction_path = run_directory / "outer_predictions.npz"
    prediction_manifest_path = run_directory / "outer_prediction_manifest.json"
    for name, direct_field in (
        ("outer_predictions.npz", "prediction_file_sha256"),
        ("outer_prediction_manifest.json", "prediction_manifest_file_sha256"),
    ):
        record = artifacts.get(name)
        _require(isinstance(record, Mapping), f"parent artifact record is missing: {name}")
        _stable_file_identity(
            run_directory / name, int(record["size_bytes"]), str(record["sha256"])
        )
        _require(record["sha256"] == result[direct_field], f"parent direct artifact hash drifted: {name}")
    prediction_manifest = _read_self_hashed_json(
        prediction_manifest_path,
        expected_file_sha256=str(result["prediction_manifest_file_sha256"]),
    )
    _require(
        prediction_manifest.get("schema") == early.PREDICTION_MANIFEST_SCHEMA
        and prediction_manifest.get("prediction_schema") == early.PREDICTION_SCHEMA
        and prediction_manifest.get("experiment") == PARENT_EXPERIMENT
        and prediction_manifest.get("outer_family") == outer_family
        and prediction_manifest.get("git_commit") == PARENT_NUMERICAL_COMMIT
        and prediction_manifest.get("config_sha256") == PARENT_CONFIG_SHA256
        and prediction_manifest.get("valid_labels_opened") is False
        and prediction_manifest.get("metadata_json_opened") is False,
        "parent prediction manifest provenance drifted",
    )
    file_record = prediction_manifest.get("prediction_file")
    records = prediction_manifest.get("arrays")
    _require(
        isinstance(file_record, Mapping)
        and isinstance(records, Mapping)
        and set(records) == set(early.PREDICTION_ARRAY_DTYPES),
        "parent prediction array manifest drifted",
    )
    arrays, prediction_sha = _verify_npz_arrays(
        prediction_path, file_record=file_record, records=records
    )
    row_count = len(arrays["prediction"])
    for name, dtype in early.PREDICTION_ARRAY_DTYPES.items():
        _require(arrays[name].dtype == dtype and arrays[name].shape == (row_count,), f"parent prediction dtype/shape drifted: {name}")
    _require(
        prediction_sha == result["prediction_file_sha256"]
        and np.array_equal(
            arrays["assigned_row_index"],
            arrays["scale_block_index"].astype(np.int64) * DEFAULT_CENTER_COUNT
            + arrays["center_seed_index"],
        ),
        "parent prediction identity drifted",
    )
    expected_datasets = set(plan.families[outer_family])
    _require(set(arrays["dataset"].tolist()) == expected_datasets, "parent prediction dataset scope drifted")
    evidence = {
        "status": "authenticated_control_only",
        "run_directory": str(run_directory),
        "outer_family": outer_family,
        "parent_git_commit": PARENT_NUMERICAL_COMMIT,
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "completion_file_sha256": completion_sha,
        "completion_content_sha256": completion["content_sha256"],
        "result_manifest_file_sha256": completion["result_manifest_file_sha256"],
        "result_manifest_content_sha256": result["content_sha256"],
        "prediction_manifest_file_sha256": result["prediction_manifest_file_sha256"],
        "prediction_manifest_content_sha256": prediction_manifest["content_sha256"],
        "prediction_file_sha256": prediction_sha,
        "row_count": row_count,
        "fresh_numerical_replay_in_child": False,
        "upstream_complete_five_fresh_replay_job": 51070392,
    }
    return AuthenticatedParentControl(
        run_directory,
        outer_family,
        early._deep_freeze(arrays),
        early._deep_freeze(evidence),
    )


UNIQUE_PREDICTION_DTYPES: Mapping[str, np.dtype[Any]] = MappingProxyType(
    {
        "unique_dataset": np.dtype("<U64"),
        "unique_source_ordinal": np.dtype(np.int16),
        "unique_source_index": np.dtype(np.int64),
        "unique_center_seed_index": np.dtype(np.int64),
        "legacy_score": np.dtype(np.float64),
        "expanded_score": np.dtype(np.float64),
        "paired_score": np.dtype(np.float64),
        "legacy_valid": np.dtype(np.bool_),
        "expanded_valid": np.dtype(np.bool_),
        "legacy_eligible": np.dtype(np.bool_),
        "expanded_eligible": np.dtype(np.bool_),
        "combined_eligible": np.dtype(np.bool_),
        "paired_prediction": np.dtype(np.bool_),
        "legacy_prediction": np.dtype(np.bool_),
        "expanded_prediction": np.dtype(np.bool_),
        "direct_min_dx_score": np.dtype(np.float64),
        "direct_min_dx_prediction": np.dtype(np.bool_),
        "direct_dx_rank_mean_score": np.dtype(np.float64),
        "direct_dx_rank_mean_prediction": np.dtype(np.bool_),
    }
)
VALID_PREDICTION_DTYPES: Mapping[str, np.dtype[Any]] = MappingProxyType(
    {
        "valid_dataset": np.dtype("<U64"),
        "valid_source_ordinal": np.dtype(np.int16),
        "valid_source_index": np.dtype(np.int64),
        "valid_scale_id": np.dtype(np.int32),
        "valid_center_seed_index": np.dtype(np.int64),
        "valid_scale_block_index": np.dtype(np.int8),
        "valid_assigned_row_index": np.dtype(np.int64),
        "valid_raw_negative_distance": np.dtype(np.float32),
        "valid_tail_probability": np.dtype(np.float64),
        "valid_tail_anomaly": np.dtype(np.float64),
        "valid_spatial_score": np.dtype(np.float64),
        "valid_spatial_denominator": np.dtype(np.float64),
        "valid_retrieval_supported": np.dtype(np.bool_),
        "valid_calibration_supported": np.dtype(np.bool_),
        "valid_spatial_imputed": np.dtype(np.bool_),
        "valid_spatial_unimputable": np.dtype(np.bool_),
        "valid_calibration_mode": np.dtype(np.int8),
        "valid_scaler_mode": np.dtype(np.int8),
        "valid_paired_score": np.dtype(np.float64),
        "valid_paired_prediction": np.dtype(np.bool_),
        "valid_parent_score": np.dtype(np.float64),
        "valid_parent_prediction": np.dtype(np.bool_),
    }
)
PREDICTION_DTYPES: Mapping[str, np.dtype[Any]] = MappingProxyType(
    {**UNIQUE_PREDICTION_DTYPES, **VALID_PREDICTION_DTYPES}
)


def _parent_rows_for_cache(
    control: AuthenticatedParentControl,
    cache: SourceCenteredCacheProjection,
) -> np.ndarray:
    arrays = control.arrays
    selected = (
        (arrays["dataset"] == cache.row.dataset)
        & (arrays["source_ordinal"] == cache.row.source_ordinal)
        & (arrays["source_index"] == cache.row.source_index)
    )
    positions = np.flatnonzero(selected)
    _require(len(positions) == cache.count, "parent/child valid-row count differs")
    by_assigned = np.full(2 * DEFAULT_CENTER_COUNT, -1, dtype=np.int64)
    assigned = np.asarray(arrays["assigned_row_index"])[positions]
    _require(len(np.unique(assigned)) == len(assigned), "parent control has duplicate valid assigned row")
    by_assigned[assigned] = positions
    ordered = by_assigned[cache.assigned_row_indices]
    _require(np.all(ordered >= 0), "parent control lacks a child valid-row identity")
    _require(
        np.array_equal(arrays["scale_id"][ordered], cache.scale_ids)
        and np.array_equal(arrays["center_seed_index"][ordered], cache.center_indices)
        and np.array_equal(arrays["scale_block_index"][ordered], cache.block_indices),
        "parent/child valid-row identity join drifted",
    )
    return ordered


def _outer_source_binding_rows(
    plan: Plan,
    caches: Sequence[SourceCenteredCacheProjection],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cache in caches:
        population_row = _population_row(plan, cache.row)
        group_mean = np.asarray(cache.sidecar_group_mean_curl_xyz)
        _require(
            group_mean.dtype == np.dtype(np.float64)
            and group_mean.shape == (20, 3)
            and np.isfinite(group_mean).all(),
            "outer source mean contract drifted",
        )
        group_mean_sha = canonical_array_sha256(group_mean)
        _require(
            group_mean_sha == cache.sidecar_group_mean_curl_xyz_sha256,
            "outer source mean canonical hash drifted",
        )
        rows.append(
            {
                "dataset": cache.row.dataset,
                "dataset_index": int(population_row["dataset_index"]),
                "physical_family": cache.row.family,
                "source_ordinal": cache.row.source_ordinal,
                "source_index": cache.row.source_index,
                "sidecar_relative_path": population_row["sidecar_relative_path"],
                "sidecar_size_bytes": int(population_row["sidecar_size_bytes"]),
                "sidecar_file_sha256": cache.sidecar_file_sha256,
                "sidecar_combined_array_sha256": (
                    cache.sidecar_combined_array_sha256
                ),
                "group_mean_curl_xyz_dtype": group_mean.dtype.str,
                "group_mean_curl_xyz_shape": list(group_mean.shape),
                "group_mean_curl_xyz_sha256": group_mean_sha,
                "group_mean_curl_xyz": group_mean.tolist(),
            }
        )
    rows.sort(
        key=lambda value: (
            int(value["dataset_index"]),
            int(value["source_ordinal"]),
        )
    )
    _require(
        len(rows) == len(caches)
        and len(
            {
                (str(row["dataset"]), int(row["source_ordinal"]))
                for row in rows
            }
        )
        == len(rows),
        "outer source binding identities are incomplete or duplicated",
    )
    return rows


def write_outer_source_centered_binding(
    destination: Path,
    caches: Sequence[SourceCenteredCacheProjection],
    *,
    plan: Plan,
    selected: CandidateSpec,
    selection: VerifiedSelection,
    scaler: VerifiedScaler,
    calibration: VerifiedCalibration,
    outer_family: str,
    git_commit: str,
) -> tuple[Path, str]:
    _require(
        selection.seal is scaler.seal is calibration.seal is _AUTHENTICATION_SEAL,
        "outer source binding requires authenticated closed fit artifacts",
    )
    rows = _outer_source_binding_rows(plan, caches)
    value = _manifest(
        {
            "schema": OUTER_SOURCE_BINDING_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_sha256": plan.sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "selected_candidate": _candidate_payload(selected),
            "selected_candidate_file_sha256": selection.file_sha256,
            "scaler_manifest_file_sha256": scaler.manifest_file_sha256,
            "scaler_artifact_file_sha256": scaler.artifact_file_sha256,
            "calibration_manifest_file_sha256": calibration.manifest_file_sha256,
            "calibration_artifact_file_sha256": calibration.artifact_file_sha256,
            "source_centered_population_manifest_file_sha256": (
                plan.sidecar_population_file_sha256
            ),
            "source_count": len(rows),
            "sources": rows,
            "label_or_ivd_member_opened": False,
            "write_phase": "after_final_fit_and_selection_before_prediction",
        }
    )
    path = destination / "outer_source_centered_binding.json"
    return path, early._atomic_json(path, value)


def authenticate_outer_source_centered_binding(
    path: Path,
    caches: Sequence[SourceCenteredCacheProjection],
    *,
    plan: Plan,
    selected: CandidateSpec,
    selection: VerifiedSelection,
    scaler: VerifiedScaler,
    calibration: VerifiedCalibration,
    outer_family: str,
    git_commit: str,
    expected_file_sha256: str,
) -> VerifiedOuterSourceBinding:
    _require(
        selection.seal is scaler.seal is calibration.seal is _AUTHENTICATION_SEAL,
        "outer source binding authentication requires closed fit artifacts",
    )
    value, file_sha = _read_manifest(path, expected_file_sha256)
    expected_rows = _outer_source_binding_rows(plan, caches)
    _require(
        value.get("schema") == OUTER_SOURCE_BINDING_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("config_sha256") == plan.sha256
        and value.get("git_commit") == git_commit
        and value.get("outer_family") == outer_family
        and value.get("selected_candidate")
        == _json_safe(_candidate_payload(selected))
        and value.get("selected_candidate_file_sha256") == selection.file_sha256
        and value.get("scaler_manifest_file_sha256")
        == scaler.manifest_file_sha256
        and value.get("scaler_artifact_file_sha256")
        == scaler.artifact_file_sha256
        and value.get("calibration_manifest_file_sha256")
        == calibration.manifest_file_sha256
        and value.get("calibration_artifact_file_sha256")
        == calibration.artifact_file_sha256
        and value.get("source_centered_population_manifest_file_sha256")
        == plan.sidecar_population_file_sha256
        and int(value.get("source_count", -1)) == len(expected_rows)
        and value.get("sources") == _json_safe(expected_rows)
        and value.get("label_or_ivd_member_opened") is False
        and value.get("write_phase")
        == "after_final_fit_and_selection_before_prediction",
        "outer source-centered binding provenance or source mean drifted",
    )
    return VerifiedOuterSourceBinding(
        path,
        file_sha,
        early._deep_freeze(value),
        _AUTHENTICATION_SEAL,
    )


def build_outer_prediction_arrays(
    caches: Sequence[SourceCenteredCacheProjection],
    model: PerScaleNegativeTailModel,
    selected: CandidateSpec,
    plan: Plan,
    parent_control: AuthenticatedParentControl,
    source_binding: VerifiedOuterSourceBinding,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    _require(
        source_binding.seal is _AUTHENTICATION_SEAL,
        "outer prediction requires an authenticated source-mean binding",
    )
    _require(caches and all(cache.labels is None and cache.center_labels is None and not cache.metadata for cache in caches), "outer caches must remain label-free")
    query = _query_cache_batch(
        model,
        caches,
        selected.representation,
        plan,
        device=device,
        ks=(selected.k,),
    )[selected.k]
    parts: dict[str, list[np.ndarray]] = {name: [] for name in PREDICTION_DTYPES}
    audits: list[dict[str, Any]] = []
    for cache_index, cache in enumerate(caches):
        values = query[cache_index]
        spatial = _row_spatial_scores(
            cache, values, sigma=selected.sigma, plan=plan
        )
        fusion = fuse_paired_scale_centers(
            cache.center_indices,
            cache.block_indices,
            spatial["spatial_score"],
            spatial["score_supported"],
            weight=selected.weight,
            center_count=DEFAULT_CENTER_COUNT,
            top_fraction=selected.top_fraction,
        )
        legacy_prediction, expanded_prediction = separate_block_center_predictions(
            fusion
        )
        direct = direct_source_centered_diagnostics(
            cache.assigned_center_indices,
            cache.assigned_block_indices,
            cache.assigned_scale_ids,
            cache.assigned_physical_dx,
            cache.assigned_source_centered_seed4,
            center_count=DEFAULT_CENTER_COUNT,
            top_fraction=DEFAULT_TOP_FRACTION,
        )
        parent_rows = _parent_rows_for_cache(parent_control, cache)
        unique_count = DEFAULT_CENTER_COUNT
        valid_count = cache.count
        unique_values: dict[str, np.ndarray] = {
            "unique_dataset": np.full(unique_count, cache.row.dataset, dtype="<U64"),
            "unique_source_ordinal": np.full(unique_count, cache.row.source_ordinal, dtype=np.int16),
            "unique_source_index": np.full(unique_count, cache.row.source_index, dtype=np.int64),
            "unique_center_seed_index": fusion.center_seed_index,
            "legacy_score": fusion.legacy_score,
            "expanded_score": fusion.expanded_score,
            "paired_score": fusion.paired_score,
            "legacy_valid": fusion.legacy_valid,
            "expanded_valid": fusion.expanded_valid,
            "legacy_eligible": fusion.legacy_eligible,
            "expanded_eligible": fusion.expanded_eligible,
            "combined_eligible": fusion.combined_eligible,
            "paired_prediction": fusion.prediction,
            "legacy_prediction": legacy_prediction,
            "expanded_prediction": expanded_prediction,
            "direct_min_dx_score": direct.min_dx_centered_curl_score,
            "direct_min_dx_prediction": direct.min_dx_prediction,
            "direct_dx_rank_mean_score": direct.dx_rank_mean_score,
            "direct_dx_rank_mean_prediction": direct.dx_rank_mean_prediction,
        }
        valid_values: dict[str, np.ndarray] = {
            "valid_dataset": np.full(valid_count, cache.row.dataset, dtype="<U64"),
            "valid_source_ordinal": np.full(valid_count, cache.row.source_ordinal, dtype=np.int16),
            "valid_source_index": np.full(valid_count, cache.row.source_index, dtype=np.int64),
            "valid_scale_id": cache.scale_ids,
            "valid_center_seed_index": cache.center_indices,
            "valid_scale_block_index": cache.block_indices,
            "valid_assigned_row_index": cache.assigned_row_indices,
            "valid_raw_negative_distance": values["raw_distance"],
            "valid_tail_probability": values["tail_probability"],
            "valid_tail_anomaly": values["tail_anomaly"],
            "valid_spatial_score": spatial["spatial_score"],
            "valid_spatial_denominator": spatial["spatial_denominator"],
            "valid_retrieval_supported": values["retrieval_supported"],
            "valid_calibration_supported": values["calibration_supported"],
            "valid_spatial_imputed": spatial["spatial_imputed"],
            "valid_spatial_unimputable": spatial["spatial_unimputable"],
            "valid_calibration_mode": values["calibration_mode"],
            "valid_scaler_mode": values["scaler_mode"],
            "valid_paired_score": fusion.paired_score[cache.center_indices],
            "valid_paired_prediction": fusion.valid_row_prediction,
            "valid_parent_score": np.asarray(parent_control.arrays["spatial_score"])[parent_rows],
            "valid_parent_prediction": np.asarray(parent_control.arrays["prediction"])[parent_rows],
        }
        for name, values_array in {**unique_values, **valid_values}.items():
            parts[name].append(np.ascontiguousarray(values_array, dtype=PREDICTION_DTYPES[name]))
        audits.append(
            {
                "dataset": cache.row.dataset,
                "source_ordinal": cache.row.source_ordinal,
                "source_index": cache.row.source_index,
                "valid_row_count": valid_count,
                "unique_center_count": unique_count,
                "combined_valid_count": int((fusion.legacy_valid | fusion.expanded_valid).sum()),
                "both_valid_count": int(fusion.both_valid.sum()),
                "legacy_only_count": int(fusion.legacy_only.sum()),
                "expanded_only_count": int(fusion.expanded_only.sum()),
                "neither_valid_count": int(fusion.neither_valid.sum()),
                "combined_coverage": fusion.combined_coverage,
                "paired_prediction_count": int(fusion.prediction.sum()),
                "valid_projected_prediction_count": int(fusion.valid_row_prediction.sum()),
                "legacy_prediction_count": int(legacy_prediction.sum()),
                "expanded_prediction_count": int(expanded_prediction.sum()),
                "direct_min_dx_prediction_count": int(direct.min_dx_prediction.sum()),
                "direct_dx_rank_mean_prediction_count": int(direct.dx_rank_mean_prediction.sum()),
                "parent_prediction_count": int(valid_values["valid_parent_prediction"].sum()),
                "sidecar_file_sha256": cache.sidecar_file_sha256,
                "sidecar_combined_array_sha256": cache.sidecar_combined_array_sha256,
                "sidecar_group_mean_curl_xyz_sha256": (
                    cache.sidecar_group_mean_curl_xyz_sha256
                ),
            }
        )
    arrays = {
        name: np.ascontiguousarray(np.concatenate(parts[name]), dtype=dtype)
        for name, dtype in PREDICTION_DTYPES.items()
    }
    return arrays, audits


def write_outer_prediction(
    destination: Path,
    arrays: Mapping[str, np.ndarray],
    audits: Sequence[Mapping[str, Any]],
    *,
    plan: Plan,
    selected: CandidateSpec,
    selection: VerifiedSelection,
    scaler: VerifiedScaler,
    calibration: VerifiedCalibration,
    source_binding: VerifiedOuterSourceBinding,
    parent_control: AuthenticatedParentControl,
    outer_family: str,
    git_commit: str,
) -> tuple[Path, Path, str]:
    _require(
        selection.seal
        is scaler.seal
        is calibration.seal
        is source_binding.seal
        is _AUTHENTICATION_SEAL,
        "prediction requires authenticated closed artifacts",
    )
    _require(set(arrays) == set(PREDICTION_DTYPES), "prediction member set drifted")
    artifact_path = destination / "outer_predictions.npz"
    artifact_sha = early._atomic_npz(artifact_path, arrays)
    fit_families = [family for family in plan.family_order if family != outer_family]
    value = _manifest(
        {
            "schema": PREDICTION_MANIFEST_SCHEMA,
            "prediction_schema": PREDICTION_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_sha256": plan.sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "selected_candidate": _candidate_payload(selected),
            "source_centered_evidence": _evidence_binding(
                plan,
                representation=selected.representation,
                fit_families=fit_families,
            ),
            "selected_candidate_artifact": {
                "path": selection.path.name,
                "file_sha256": selection.file_sha256,
                "content_sha256": selection.manifest["content_sha256"],
            },
            "scaler_manifest_file_sha256": scaler.manifest_file_sha256,
            "scaler_artifact_file_sha256": scaler.artifact_file_sha256,
            "calibration_manifest_file_sha256": calibration.manifest_file_sha256,
            "calibration_artifact_file_sha256": calibration.artifact_file_sha256,
            "outer_source_centered_binding": {
                "path": source_binding.path.name,
                "file_sha256": source_binding.file_sha256,
                "content_sha256": source_binding.manifest["content_sha256"],
            },
            "parent_control": dict(parent_control.evidence),
            "valid_labels_opened": False,
            "reference_labels_all_opened": False,
            "prediction_file": {
                "path": artifact_path.name,
                "size_bytes": artifact_path.stat().st_size,
                "sha256": artifact_sha,
            },
            "unique_row_count": len(arrays["unique_center_seed_index"]),
            "valid_row_count": len(arrays["valid_assigned_row_index"]),
            "arrays": early._array_manifest(arrays),
            "group_audits": list(audits),
            "direct_diagnostic_evidence_boundary": {
                "direct_min_dx": "not_template_matching_success",
                "direct_dx_rank_mean": "not_template_matching_success",
            },
        }
    )
    manifest_path = destination / "outer_prediction_manifest.json"
    manifest_sha = early._atomic_json(manifest_path, value)
    return artifact_path, manifest_path, manifest_sha


def _require_float64_ulp(
    stored: np.ndarray,
    replayed: np.ndarray,
    *,
    name: str,
    bound: int,
) -> None:
    left = np.asarray(stored)
    right = np.asarray(replayed)
    _require(
        left.dtype == right.dtype == np.dtype(np.float64)
        and left.shape == right.shape
        and np.isfinite(left).all()
        and np.isfinite(right).all(),
        f"{name} replay contract drifted",
    )
    _require(np.array_equal(left == 0.0, right == 0.0), f"{name} replay zero mask drifted")
    left_bits = np.ascontiguousarray(left).view(np.uint64)
    right_bits = np.ascontiguousarray(right).view(np.uint64)
    greater = left_bits >= right_bits
    distance = np.empty_like(left_bits)
    np.subtract(left_bits, right_bits, out=distance, where=greater)
    np.subtract(right_bits, left_bits, out=distance, where=~greater)
    maximum = int(distance.max(initial=np.uint64(0)))
    _require(maximum <= bound, f"{name} replay exceeds {bound}-ULP bound: {maximum}")


_PORTABLE_FLOAT_FIELDS = frozenset(
    {
        "legacy_score",
        "expanded_score",
        "paired_score",
        "valid_spatial_score",
        "valid_spatial_denominator",
        "valid_paired_score",
    }
)


def authenticate_outer_prediction(
    artifact_path: Path,
    manifest_path: Path,
    *,
    plan: Plan,
    selected: CandidateSpec,
    selection: VerifiedSelection,
    scaler: VerifiedScaler,
    calibration: VerifiedCalibration,
    source_binding: VerifiedOuterSourceBinding,
    parent_control: AuthenticatedParentControl,
    outer_caches: Sequence[SourceCenteredCacheProjection],
    outer_family: str,
    git_commit: str,
    device: str,
    expected_manifest_sha256: str,
) -> VerifiedPrediction:
    _require(
        selection.seal
        is scaler.seal
        is calibration.seal
        is source_binding.seal
        is _AUTHENTICATION_SEAL,
        "prediction authentication requires closed artifacts",
    )
    value, file_sha = _read_manifest(manifest_path, expected_manifest_sha256)
    fit_families = [family for family in plan.family_order if family != outer_family]
    _require(
        value.get("schema") == PREDICTION_MANIFEST_SCHEMA
        and value.get("prediction_schema") == PREDICTION_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("config_sha256") == plan.sha256
        and value.get("git_commit") == git_commit
        and value.get("outer_family") == outer_family
        and value.get("selected_candidate") == _json_safe(_candidate_payload(selected))
        and value.get("source_centered_evidence")
        == _evidence_binding(
            plan,
            representation=selected.representation,
            fit_families=fit_families,
        )
        and value.get("parent_control") == _json_safe(dict(parent_control.evidence))
        and value.get("valid_labels_opened") is False
        and value.get("reference_labels_all_opened") is False,
        "prediction manifest provenance drifted",
    )
    _require(
        value.get("selected_candidate_artifact", {}).get("file_sha256")
        == selection.file_sha256
        and value.get("scaler_manifest_file_sha256") == scaler.manifest_file_sha256
        and value.get("scaler_artifact_file_sha256") == scaler.artifact_file_sha256
        and value.get("calibration_manifest_file_sha256")
        == calibration.manifest_file_sha256
        and value.get("calibration_artifact_file_sha256")
        == calibration.artifact_file_sha256,
        "prediction artifact chain drifted",
    )
    _require(
        value.get("outer_source_centered_binding")
        == {
            "path": source_binding.path.name,
            "file_sha256": source_binding.file_sha256,
            "content_sha256": source_binding.manifest["content_sha256"],
        },
        "prediction source-mean binding drifted",
    )
    records = value.get("arrays")
    _require(isinstance(records, Mapping) and set(records) == set(PREDICTION_DTYPES), "prediction array records drifted")
    arrays, artifact_sha = _verify_npz_arrays(
        artifact_path, file_record=value["prediction_file"], records=records
    )
    unique_count = int(value.get("unique_row_count", -1))
    valid_count = int(value.get("valid_row_count", -1))
    _require(
        unique_count == len(outer_caches) * DEFAULT_CENTER_COUNT
        and valid_count == sum(cache.count for cache in outer_caches),
        "prediction population count drifted",
    )
    for name, dtype in UNIQUE_PREDICTION_DTYPES.items():
        _require(arrays[name].dtype == dtype and arrays[name].shape == (unique_count,), f"unique prediction contract drifted: {name}")
    for name, dtype in VALID_PREDICTION_DTYPES.items():
        _require(arrays[name].dtype == dtype and arrays[name].shape == (valid_count,), f"valid prediction contract drifted: {name}")
    _require(
        np.array_equal(
            arrays["valid_assigned_row_index"],
            arrays["valid_scale_block_index"].astype(np.int64) * DEFAULT_CENTER_COUNT
            + arrays["valid_center_seed_index"],
        ),
        "valid prediction assigned identity drifted",
    )
    expected_arrays, expected_audits = build_outer_prediction_arrays(
        outer_caches,
        calibration.model,
        selected,
        plan,
        parent_control,
        source_binding,
        device=device,
    )
    for name in sorted(arrays):
        if name in _PORTABLE_FLOAT_FIELDS:
            _require_float64_ulp(
                arrays[name], expected_arrays[name], name=name, bound=16
            )
        else:
            _require(
                canonical_array_sha256(arrays[name])
                == canonical_array_sha256(expected_arrays[name]),
                f"prediction fresh replay drifted: {name}",
            )
    _require(_json_safe(value.get("group_audits")) == _json_safe(expected_audits), "prediction group audit fresh replay drifted")
    return VerifiedPrediction(
        manifest_path,
        file_sha,
        artifact_sha,
        early._deep_freeze(value),
        early._deep_freeze(arrays),
        _AUTHENTICATION_SEAL,
    )


OUTER_METRIC_FIELDS = (
    "outer_family",
    "dataset",
    "source_ordinal",
    "source_index",
    "arm",
    "population",
    "template_success_eligible",
    "candidate_id",
    "representation",
    "k",
    "sigma",
    "weight",
    "decision_rule",
    "decision_value",
    "sample_count",
    "positive_count",
    "negative_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "average_precision",
    "auroc",
    "unique_center_combined_coverage",
    "both_valid_count",
    "legacy_only_count",
    "expanded_only_count",
    "neither_valid_count",
)
CLASSIFICATION_FIELDS = (
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
)
CLASSIFICATION_COUNT_FIELDS = (
    "sample_count",
    "positive_count",
    "negative_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
)


def _outer_metric_row(
    *,
    outer_family: str,
    cache: SourceCenteredCacheProjection,
    arm: str,
    population: str,
    template_success_eligible: bool,
    selected: CandidateSpec,
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "outer_family": outer_family,
        "dataset": cache.row.dataset,
        "source_ordinal": cache.row.source_ordinal,
        "source_index": cache.row.source_index,
        "arm": arm,
        "population": population,
        "template_success_eligible": template_success_eligible,
        **_candidate_payload(selected),
        **_classification_metric_values(labels, scores, predictions),
        **coverage,
    }


def load_outer_references_after_prediction(
    plan: Plan,
    outer_family: str,
    verified: VerifiedPrediction,
) -> tuple[
    dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, Mapping[str, Any]]],
    list[dict[str, Any]],
]:
    _require(verified.seal is _AUTHENTICATION_SEAL, "reference access requires authenticated prediction")
    cache_rows, _identity = load_cache_rows(plan.parent_plan)
    outer_rows = [row for row in cache_rows if row.family == outer_family]
    expected_keys = {
        (dataset, ordinal)
        for dataset in plan.families[outer_family]
        for ordinal in range(4)
    }
    _require({(row.dataset, row.source_ordinal) for row in outer_rows} == expected_keys, "outer reference scope is incomplete")
    references: dict[
        tuple[str, int, int], tuple[np.ndarray, np.ndarray, Mapping[str, Any]]
    ] = {}
    audits: list[dict[str, Any]] = []
    arrays = verified.arrays
    for row in outer_rows:
        parent, metadata = _load_parent_projection(plan, row, include_references=True)
        selected_rows = (
            (arrays["valid_dataset"] == row.dataset)
            & (arrays["valid_source_ordinal"] == row.source_ordinal)
            & (arrays["valid_source_index"] == row.source_index)
        )
        _require(int(selected_rows.sum()) == len(parent["valid_labels"]), "outer reference/prediction count differs")
        _require(
            np.array_equal(arrays["valid_assigned_row_index"][selected_rows], parent["valid_assigned_row_index"])
            and np.array_equal(arrays["valid_scale_id"][selected_rows], parent["valid_scale_id"]),
            "outer reference/prediction identity differs",
        )
        key = (row.dataset, row.source_ordinal, row.source_index)
        _require(key not in references, "duplicate outer reference key")
        valid_labels = np.ascontiguousarray(parent["valid_labels"], dtype=np.bool_)
        center_labels = np.ascontiguousarray(
            parent["reference_labels_all"][:DEFAULT_CENTER_COUNT], dtype=np.bool_
        )
        valid_labels.setflags(write=False)
        center_labels.setflags(write=False)
        references[key] = (valid_labels, center_labels, early._deep_freeze(metadata))
        audits.append(
            {
                "dataset": row.dataset,
                "source_ordinal": row.source_ordinal,
                "source_index": row.source_index,
                "cache_path": str(row.path),
                "cache_file_sha256": row.sha256,
                "members_opened": ["valid_labels", "reference_labels_all", "metadata_json"],
                "opened_after_prediction_fresh_authentication": True,
                "prediction_manifest_file_sha256": verified.manifest_file_sha256,
                "prediction_file_sha256": verified.artifact_file_sha256,
            }
        )
    return references, audits


def evaluate_outer_prediction(
    plan: Plan,
    selected: CandidateSpec,
    verified: VerifiedPrediction,
    *,
    outer_family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references, access_audits = load_outer_references_after_prediction(
        plan, outer_family, verified
    )
    arrays = verified.arrays
    rows: list[dict[str, Any]] = []
    for key in sorted(references):
        dataset, source_ordinal, source_index = key
        valid_labels, center_labels, _metadata = references[key]
        valid = (
            (arrays["valid_dataset"] == dataset)
            & (arrays["valid_source_ordinal"] == source_ordinal)
            & (arrays["valid_source_index"] == source_index)
        )
        unique = (
            (arrays["unique_dataset"] == dataset)
            & (arrays["unique_source_ordinal"] == source_ordinal)
            & (arrays["unique_source_index"] == source_index)
        )
        _require(int(unique.sum()) == DEFAULT_CENTER_COUNT and int(valid.sum()) == len(valid_labels), "outer metric group population drifted")
        cache = SourceCenteredCacheProjection(
            row=next(
                row
                for row in load_cache_rows(plan.parent_plan)[0]
                if (row.dataset, row.source_ordinal, row.source_index) == key
            ),
            fmt_features=np.empty((0, 161), dtype=np.float32),
            source_centered_seed4=np.empty((0, 4), dtype=np.float32),
            scale_ids=np.empty(0, dtype=np.int32),
            center_indices=np.empty(0, dtype=np.int64),
            block_indices=np.empty(0, dtype=np.int8),
            assigned_row_indices=np.empty(0, dtype=np.int64),
            labels=None,
            center_labels=None,
            metadata={},
            assigned_center_indices=np.empty(0, dtype=np.int64),
            assigned_block_indices=np.empty(0, dtype=np.int8),
            assigned_scale_ids=np.empty(0, dtype=np.int32),
            assigned_physical_dx=np.empty(0, dtype=np.float64),
            assigned_source_centered_seed4=np.empty((0, 4), dtype=np.float32),
            sidecar_file_sha256="",
            sidecar_combined_array_sha256="",
            sidecar_group_mean_curl_xyz=np.empty((0, 3), dtype=np.float64),
            sidecar_group_mean_curl_xyz_sha256="",
        )
        legacy_valid = np.asarray(arrays["legacy_valid"][unique], dtype=np.bool_)
        expanded_valid = np.asarray(arrays["expanded_valid"][unique], dtype=np.bool_)
        combined = legacy_valid | expanded_valid
        both = legacy_valid & expanded_valid
        legacy_only = legacy_valid & ~expanded_valid
        expanded_only = ~legacy_valid & expanded_valid
        neither = ~legacy_valid & ~expanded_valid
        coverage = {
            "unique_center_combined_coverage": float(combined.mean()),
            "both_valid_count": int(both.sum()),
            "legacy_only_count": int(legacy_only.sum()),
            "expanded_only_count": int(expanded_only.sum()),
            "neither_valid_count": int(neither.sum()),
        }
        rows.append(
            _outer_metric_row(
                outer_family=outer_family,
                cache=cache,
                arm="source_centered_paired_centers",
                population="all_parent_valid_rows",
                template_success_eligible=True,
                selected=selected,
                labels=valid_labels,
                scores=arrays["valid_paired_score"][valid],
                predictions=arrays["valid_paired_prediction"][valid],
                coverage=coverage,
            )
        )
        rows.append(
            _outer_metric_row(
                outer_family=outer_family,
                cache=cache,
                arm="parent_current_replay",
                population="all_parent_valid_rows",
                template_success_eligible=False,
                selected=selected,
                labels=valid_labels,
                scores=arrays["valid_parent_score"][valid],
                predictions=arrays["valid_parent_prediction"][valid],
                coverage=coverage,
            )
        )
        # The unique-center metric excludes centers with neither block valid;
        # those centers contribute only to the mandatory coverage gate.
        rows.append(
            _outer_metric_row(
                outer_family=outer_family,
                cache=cache,
                arm="source_centered_paired_centers",
                population="combined_valid_unique_centers",
                template_success_eligible=False,
                selected=selected,
                labels=center_labels[combined],
                scores=arrays["paired_score"][unique][combined],
                predictions=arrays["paired_prediction"][unique][combined],
                coverage=coverage,
            )
        )
        diagnostic_arms = (
            (
                "source_centered_separate_legacy_block",
                arrays["legacy_score"][unique],
                arrays["legacy_prediction"][unique],
            ),
            (
                "source_centered_separate_expanded_block",
                arrays["expanded_score"][unique],
                arrays["expanded_prediction"][unique],
            ),
            (
                "direct_source_centered_min_dx_top5",
                arrays["direct_min_dx_score"][unique],
                arrays["direct_min_dx_prediction"][unique],
            ),
            (
                "direct_source_centered_dx_rank_mean_top5",
                arrays["direct_dx_rank_mean_score"][unique],
                arrays["direct_dx_rank_mean_prediction"][unique],
            ),
        )
        for arm, scores, predictions in diagnostic_arms:
            rows.append(
                _outer_metric_row(
                    outer_family=outer_family,
                    cache=cache,
                    arm=arm,
                    population="all_unique_centers",
                    template_success_eligible=False,
                    selected=selected,
                    labels=center_labels,
                    scores=scores,
                    predictions=predictions,
                    coverage=coverage,
                )
            )
        for population, mask in (
            ("both_valid_unique_centers", both),
            ("legacy_only_unique_centers", legacy_only),
            ("expanded_only_unique_centers", expanded_only),
            ("neither_valid_unique_centers", neither),
        ):
            if mask.any():
                rows.append(
                    _outer_metric_row(
                        outer_family=outer_family,
                        cache=cache,
                        arm="source_centered_paired_centers",
                        population=population,
                        template_success_eligible=False,
                        selected=selected,
                        labels=center_labels[mask],
                        scores=arrays["paired_score"][unique][mask],
                        predictions=arrays["paired_prediction"][unique][mask],
                        coverage=coverage,
                    )
                )
    _require(rows and tuple(rows[0]) == OUTER_METRIC_FIELDS, "outer metric field contract drifted")
    return rows, access_audits


def _mean_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(rows, "metric summary has no groups")
    result: dict[str, Any] = {"group_count": len(rows)}
    for field in CLASSIFICATION_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[field] = float(np.mean(finite)) if len(finite) else float("nan")
    for field in CLASSIFICATION_COUNT_FIELDS:
        result[field] = int(sum(int(row[field]) for row in rows))
    result["unique_center_combined_coverage"] = float(
        np.mean([float(row["unique_center_combined_coverage"]) for row in rows])
    )
    for field in (
        "both_valid_count",
        "legacy_only_count",
        "expanded_only_count",
        "neither_valid_count",
    ):
        result[field] = int(sum(int(row[field]) for row in rows))
    return result


def outer_summary(
    rows: Sequence[Mapping[str, Any]],
    outer_family: str,
) -> dict[str, Any]:
    primary = [
        row
        for row in rows
        if row["arm"] == "source_centered_paired_centers"
        and row["population"] == "all_parent_valid_rows"
    ]
    parent = [
        row
        for row in rows
        if row["arm"] == "parent_current_replay"
        and row["population"] == "all_parent_valid_rows"
    ]
    _require(primary and len(primary) == len(parent), "primary/parent source groups differ")
    diagnostics: dict[str, Any] = {}
    diagnostic_keys = sorted(
        {
            (str(row["arm"]), str(row["population"]))
            for row in rows
            if not bool(row["template_success_eligible"])
            and row["arm"] != "parent_current_replay"
        }
    )
    for arm, population in diagnostic_keys:
        selected = [
            row
            for row in rows
            if row["arm"] == arm and row["population"] == population
        ]
        diagnostics[f"{arm}|{population}"] = _mean_metric_rows(selected)
    return {
        "schema": OUTER_SUMMARY_SCHEMA,
        "experiment": EXPERIMENT,
        "outer_family": outer_family,
        "classification_metric_population": "all_parent_valid_rows_after_exact_center_projection",
        "primary": _mean_metric_rows(primary),
        "parent_control": _mean_metric_rows(parent),
        "diagnostics_not_template_success": diagnostics,
        "source_group_count": len(primary),
    }


_INNER_STRING_FIELDS = frozenset(
    {
        "outer_family",
        "inner_family",
        "dataset",
        "candidate_id",
        "representation",
        "decision_rule",
    }
)
_INNER_INTEGER_FIELDS = frozenset(
    {
        "source_ordinal",
        "k",
        "sample_count",
        "positive_count",
        "negative_count",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "both_valid_count",
        "legacy_only_count",
        "expanded_only_count",
        "neither_valid_count",
    }
)


def _parse_inner_metric_csv(
    path: Path,
    *,
    expected_sha256: str,
    plan: Plan,
    outer_family: str,
) -> list[dict[str, Any]]:
    content = early._read_authenticated_bytes(path, expected_sha256=expected_sha256)
    with io.StringIO(content.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == INNER_METRIC_FIELDS, "inner metric CSV fields drifted")
        raw_rows = list(reader)
    rows: list[dict[str, Any]] = []
    candidates = {value.candidate_id: value for value in candidate_specs(plan)}
    for raw in raw_rows:
        parsed: dict[str, Any] = {}
        for field_name in INNER_METRIC_FIELDS:
            text = raw[field_name]
            if field_name in _INNER_STRING_FIELDS:
                parsed[field_name] = text
            elif field_name in _INNER_INTEGER_FIELDS:
                parsed[field_name] = int(text)
            else:
                parsed[field_name] = float(text)
        candidate = candidates.get(str(parsed["candidate_id"]))
        _require(candidate is not None, "inner CSV has unknown candidate")
        canonical = _candidate_payload(candidate)
        for name in (
            "candidate_id",
            "representation",
            "k",
            "sigma",
            "weight",
            "decision_rule",
            "decision_value",
        ):
            _require(parsed[name] == canonical[name], f"inner CSV candidate identity drifted: {name}")
        _require(parsed["outer_family"] == outer_family, "inner CSV outer family drifted")
        _require(
            parsed["inner_family"] in plan.family_order
            and parsed["inner_family"] != outer_family
            and parsed["dataset"] in plan.families[parsed["inner_family"]],
            "inner CSV family/dataset scope drifted",
        )
        rows.append(parsed)
    expected_group_count = sum(
        4 * len(plan.families[family])
        for family in plan.family_order
        if family != outer_family
    )
    _require(
        len(rows) == FROZEN_CANDIDATE_COUNT * expected_group_count,
        "inner CSV candidate/group population drifted",
    )
    return rows


def _authenticate_summary_csv(
    path: Path,
    *,
    expected_sha256: str,
    expected: Sequence[Mapping[str, Any]],
) -> None:
    content = early._read_authenticated_bytes(path, expected_sha256=expected_sha256)
    with io.StringIO(content.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == INNER_SUMMARY_FIELDS, "inner summary fields drifted")
        rows = list(reader)
    _require(len(rows) == len(expected) == FROZEN_CANDIDATE_COUNT, "inner summary count drifted")
    for index, (raw, value) in enumerate(zip(rows, expected, strict=True)):
        for field_name in INNER_SUMMARY_FIELDS:
            _require(
                raw[field_name] == early._csv_value(value[field_name]).__str__(),
                f"inner summary persisted value drifted: row={index}/{field_name}",
            )


def _authenticate_inner_fit_audits(
    path: Path,
    *,
    expected_sha256: str,
    plan: Plan,
    outer_family: str,
) -> None:
    value, _file_sha = _read_manifest(path, expected_sha256)
    fits = value.get("fits")
    _require(
        value.get("schema") == INNER_AUDIT_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("outer_family") == outer_family
        and isinstance(fits, list)
        and int(value.get("fit_count", -1)) == len(fits) == 12,
        "inner fit-audit provenance drifted",
    )
    expected_pairs = {
        (inner, representation)
        for inner in plan.family_order
        if inner != outer_family
        for representation in plan.representations
    }
    actual_pairs = {
        (str(item["inner_family"]), str(item["representation"]))
        for item in fits
    }
    _require(actual_pairs == expected_pairs, "inner fit-audit coverage drifted")
    _require(
        all(
            item.get("source_centered_population_manifest_file_sha256")
            == plan.sidecar_population_file_sha256
            for item in fits
        ),
        "inner fit-audit sidecar binding drifted",
    )


def persist_and_authenticate_inner_selection(
    destination: Path,
    rows: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    *,
    plan: Plan,
    outer_family: str,
) -> tuple[
    CandidateSpec,
    dict[str, Any],
    Mapping[str, tuple[Path, str]],
]:
    _require(rows and tuple(rows[0]) == INNER_METRIC_FIELDS, "inner metric row fields drifted")
    metrics_path = destination / "inner_group_metrics.csv"
    metrics_sha = early._atomic_csv(metrics_path, INNER_METRIC_FIELDS, rows)
    persisted_rows = _parse_inner_metric_csv(
        metrics_path,
        expected_sha256=metrics_sha,
        plan=plan,
        outer_family=outer_family,
    )
    summaries, selected, selected_summary = aggregate_and_select_inner(
        plan, persisted_rows
    )
    _require(tuple(summaries[0]) == INNER_SUMMARY_FIELDS, "inner summary fields drifted")
    summary_path = destination / "inner_candidate_summary.csv"
    summary_sha = early._atomic_csv(
        summary_path, INNER_SUMMARY_FIELDS, summaries
    )
    _authenticate_summary_csv(
        summary_path, expected_sha256=summary_sha, expected=summaries
    )
    audit_path = destination / "inner_fit_audits.json"
    audit_sha = early._atomic_json(
        audit_path,
        _manifest(
            {
                "schema": INNER_AUDIT_SCHEMA,
                "experiment": EXPERIMENT,
                "outer_family": outer_family,
                "fit_count": len(audits),
                "fits": list(audits),
            }
        ),
    )
    _authenticate_inner_fit_audits(
        audit_path,
        expected_sha256=audit_sha,
        plan=plan,
        outer_family=outer_family,
    )
    return (
        selected,
        selected_summary,
        MappingProxyType(
            {
                "inner_group_metrics": (metrics_path, metrics_sha),
                "inner_candidate_summary": (summary_path, summary_sha),
                "inner_fit_audits": (audit_path, audit_sha),
            }
        ),
    )


def _fresh_replay_before_reference(
    plan: Plan,
    destination: Path,
    *,
    selected: CandidateSpec,
    selected_summary: Mapping[str, Any],
    inner_paths: Mapping[str, tuple[Path, str]],
    outer_family: str,
    git_commit: str,
    device: str,
    scaler_manifest_sha256: str,
    calibration_manifest_sha256: str,
    selection_file_sha256: str,
    source_binding_file_sha256: str,
    prediction_manifest_sha256: str,
) -> VerifiedPrediction:
    fit_families = [family for family in plan.family_order if family != outer_family]
    scaler = authenticate_final_scaler(
        destination / "final_per_scale_scaler.npz",
        destination / "final_per_scale_scaler_manifest.json",
        plan=plan,
        selected=selected,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_sha256=scaler_manifest_sha256,
    )
    calibration = authenticate_final_calibration(
        destination / "final_tail_calibration.npz",
        destination / "final_tail_calibration_manifest.json",
        scaler,
        plan=plan,
        selected=selected,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_sha256=calibration_manifest_sha256,
    )
    selection = authenticate_selected_candidate(
        destination / "selected_candidate.json",
        plan=plan,
        selected=selected,
        selected_summary=selected_summary,
        scaler=scaler,
        calibration=calibration,
        inner_paths=inner_paths,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=selection_file_sha256,
    )
    cache_rows, _identity = load_cache_rows(plan.parent_plan)
    outer_rows = [row for row in cache_rows if row.family == outer_family]
    outer_caches = [
        load_source_centered_cache_projection(
            plan, row, include_references=False
        )
        for row in outer_rows
    ]
    source_binding = authenticate_outer_source_centered_binding(
        destination / "outer_source_centered_binding.json",
        outer_caches,
        plan=plan,
        selected=selected,
        selection=selection,
        scaler=scaler,
        calibration=calibration,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=source_binding_file_sha256,
    )
    parent_control = authenticate_parent_control(plan, outer_family)
    return authenticate_outer_prediction(
        destination / "outer_predictions.npz",
        destination / "outer_prediction_manifest.json",
        plan=plan,
        selected=selected,
        selection=selection,
        scaler=scaler,
        calibration=calibration,
        source_binding=source_binding,
        parent_control=parent_control,
        outer_caches=outer_caches,
        outer_family=outer_family,
        git_commit=git_commit,
        device=device,
        expected_manifest_sha256=prediction_manifest_sha256,
    )


def run(
    config_path: str | Path,
    outer_family: str,
    output_dir: str | Path,
    *,
    device: str,
    sidecar_input_manifest_path: str | Path,
    sidecar_input_manifest_file_sha256: str,
    sidecar_root: str | Path,
    sidecar_population_manifest_path: str | Path,
    sidecar_population_manifest_file_sha256: str,
    expected_config_sha256: str | None = EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    plan = load_plan(config_path)
    _require(outer_family in plan.family_order, f"unknown outer family: {outer_family}")
    if expected_config_sha256 is not None:
        _require(plan.sha256 == expected_config_sha256, "expected config SHA-256 drifted")
    git_commit, dirty = _git_identity()
    _require(not dirty, "Ibex numerical run requires a clean committed worktree")
    plan = bind_source_centered_evidence(
        plan,
        input_manifest_path=sidecar_input_manifest_path,
        input_manifest_file_sha256=sidecar_input_manifest_file_sha256,
        sidecar_root=sidecar_root,
        population_manifest_path=sidecar_population_manifest_path,
        population_manifest_file_sha256=sidecar_population_manifest_file_sha256,
    )
    _configure_execution(device)
    destination = Path(output_dir).resolve()
    _require(not destination.exists(), f"immutable output directory exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    print(
        f"[{_utc_now()}] {EXPERIMENT} outer={outer_family} commit={git_commit}",
        flush=True,
    )

    cache_rows, input_manifest_identity = load_cache_rows(plan.parent_plan)
    nonouter_rows = [row for row in cache_rows if row.family != outer_family]
    outer_rows = [row for row in cache_rows if row.family == outer_family]
    _require(nonouter_rows and outer_rows, "outer split produced an empty side")
    # The sidecar population envelope was authenticated above, but no outer
    # sidecar NPZ member has been opened at this point.
    nonouter_caches = [
        load_source_centered_cache_projection(
            plan, row, include_references=True
        )
        for row in nonouter_rows
    ]
    inner_rows, inner_audits = _inner_metric_rows(
        plan, nonouter_caches, outer_family, device=device
    )
    selected, selected_summary, inner_paths = persist_and_authenticate_inner_selection(
        destination,
        inner_rows,
        inner_audits,
        plan=plan,
        outer_family=outer_family,
    )

    fit_families = [family for family in plan.family_order if family != outer_family]
    final_model = _fit_tail_model(
        nonouter_caches,
        selected.representation,
        plan,
        device=device,
        ks=(selected.k,),
    )
    scaler_path, scaler_manifest_path, scaler_manifest_sha = write_final_scaler(
        destination,
        final_model,
        plan=plan,
        selected=selected,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
    )
    scaler = authenticate_final_scaler(
        scaler_path,
        scaler_manifest_path,
        plan=plan,
        selected=selected,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_sha256=scaler_manifest_sha,
    )
    calibration_path, calibration_manifest_path, calibration_manifest_sha = write_final_calibration(
        destination,
        final_model,
        scaler,
        plan=plan,
        selected=selected,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
    )
    calibration = authenticate_final_calibration(
        calibration_path,
        calibration_manifest_path,
        scaler,
        plan=plan,
        selected=selected,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_sha256=calibration_manifest_sha,
    )
    del final_model, nonouter_caches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    selected_path, selected_sha = write_selected_candidate(
        destination,
        plan=plan,
        selected=selected,
        selected_summary=selected_summary,
        scaler=scaler,
        calibration=calibration,
        inner_paths=inner_paths,
        outer_family=outer_family,
        git_commit=git_commit,
    )
    selection = authenticate_selected_candidate(
        selected_path,
        plan=plan,
        selected=selected,
        selected_summary=selected_summary,
        scaler=scaler,
        calibration=calibration,
        inner_paths=inner_paths,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=selected_sha,
    )

    # Only now may the held-out family's sealed source-centered NPZ members be
    # opened.  The inherited parent prediction is label-free control evidence.
    parent_control = authenticate_parent_control(plan, outer_family)
    outer_caches = [
        load_source_centered_cache_projection(
            plan, row, include_references=False
        )
        for row in outer_rows
    ]
    source_binding_path, source_binding_sha = write_outer_source_centered_binding(
        destination,
        outer_caches,
        plan=plan,
        selected=selected,
        selection=selection,
        scaler=scaler,
        calibration=calibration,
        outer_family=outer_family,
        git_commit=git_commit,
    )
    source_binding = authenticate_outer_source_centered_binding(
        source_binding_path,
        outer_caches,
        plan=plan,
        selected=selected,
        selection=selection,
        scaler=scaler,
        calibration=calibration,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=source_binding_sha,
    )
    arrays, prediction_audits = build_outer_prediction_arrays(
        outer_caches,
        calibration.model,
        selected,
        plan,
        parent_control,
        source_binding,
        device=device,
    )
    prediction_path, prediction_manifest_path, prediction_manifest_sha = write_outer_prediction(
        destination,
        arrays,
        prediction_audits,
        plan=plan,
        selected=selected,
        selection=selection,
        scaler=scaler,
        calibration=calibration,
        source_binding=source_binding,
        parent_control=parent_control,
        outer_family=outer_family,
        git_commit=git_commit,
    )
    del arrays, outer_caches
    gc.collect()

    verified = _fresh_replay_before_reference(
        plan,
        destination,
        selected=selected,
        selected_summary=selected_summary,
        inner_paths=inner_paths,
        outer_family=outer_family,
        git_commit=git_commit,
        device=device,
        scaler_manifest_sha256=scaler_manifest_sha,
        calibration_manifest_sha256=calibration_manifest_sha,
        selection_file_sha256=selected_sha,
        source_binding_file_sha256=source_binding_sha,
        prediction_manifest_sha256=prediction_manifest_sha,
    )
    outer_rows_metrics, reference_audits = evaluate_outer_prediction(
        plan, selected, verified, outer_family=outer_family
    )
    outer_metrics_path = destination / "outer_group_metrics.csv"
    outer_metrics_sha = early._atomic_csv(
        outer_metrics_path, OUTER_METRIC_FIELDS, outer_rows_metrics
    )
    summary_value = _manifest(outer_summary(outer_rows_metrics, outer_family))
    outer_summary_path = destination / "outer_summary.json"
    outer_summary_sha = early._atomic_json(outer_summary_path, summary_value)
    reference_value = _manifest(
        {
            "schema": REFERENCE_AUDIT_SCHEMA,
            "experiment": EXPERIMENT,
            "outer_family": outer_family,
            "first_open_phase": "after_outer_prediction_file_manifest_and_fresh_replay_authentication",
            "prediction_manifest_file_sha256": verified.manifest_file_sha256,
            "prediction_file_sha256": verified.artifact_file_sha256,
            "row_count": len(reference_audits),
            "rows": reference_audits,
        }
    )
    reference_path = destination / "outer_reference_access_audit.json"
    reference_sha = early._atomic_json(reference_path, reference_value)

    artifact_names = tuple(
        name
        for name in plan.required_fold_files
        if name not in {"result_manifest.json", "RUN_COMPLETE.json"}
    )
    result = _manifest(
        {
            "schema": RESULT_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "completed",
            "completed_utc": _utc_now(),
            "git_commit": git_commit,
            "config_path": str(plan.path),
            "config_sha256": plan.sha256,
            "input_manifest": input_manifest_identity,
            "source_centered_evidence": _evidence_binding(
                plan,
                representation=selected.representation,
                fit_families=fit_families,
            ),
            "parent_control": dict(parent_control.evidence),
            "outer_family": outer_family,
            "selected_candidate": _candidate_payload(selected),
            "selected_candidate_file_sha256": selected_sha,
            "outer_source_centered_binding_file_sha256": source_binding_sha,
            "final_scaler_manifest_file_sha256": scaler_manifest_sha,
            "final_scaler_file_sha256": scaler.artifact_file_sha256,
            "final_calibration_manifest_file_sha256": calibration_manifest_sha,
            "final_calibration_file_sha256": calibration.artifact_file_sha256,
            "prediction_manifest_file_sha256": prediction_manifest_sha,
            "prediction_file_sha256": verified.artifact_file_sha256,
            "inner_group_metrics_file_sha256": inner_paths["inner_group_metrics"][1],
            "inner_candidate_summary_file_sha256": inner_paths["inner_candidate_summary"][1],
            "inner_fit_audits_file_sha256": inner_paths["inner_fit_audits"][1],
            "outer_group_metrics_file_sha256": outer_metrics_sha,
            "outer_summary_file_sha256": outer_summary_sha,
            "outer_reference_access_audit_file_sha256": reference_sha,
            "environment": early._environment_audit(device),
            "artifacts": {
                name: {
                    "size_bytes": (destination / name).stat().st_size,
                    "sha256": sha256_file(destination / name),
                }
                for name in artifact_names
            },
        }
    )
    result_path = destination / "result_manifest.json"
    result_sha = early._atomic_json(result_path, result)
    completion = _manifest(
        {
            "schema": COMPLETE_SCHEMA,
            "experiment": EXPERIMENT,
            "outer_family": outer_family,
            "git_commit": git_commit,
            "config_sha256": plan.sha256,
            "result_manifest_file": result_path.name,
            "result_manifest_file_sha256": result_sha,
            "result_manifest_content_sha256": result["content_sha256"],
            "completed_utc": _utc_now(),
        }
    )
    early._atomic_json(destination / "RUN_COMPLETE.json", completion)
    _require(
        {path.name for path in destination.iterdir()} == set(plan.required_fold_files),
        "completed fold file set drifted",
    )
    primary_f1 = float(summary_value["primary"]["f1"])
    print(
        f"[{_utc_now()}] completed outer={outer_family} primary_valid_row_F1={primary_f1:.6f}",
        flush=True,
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "config" / "Verify_SourceCenteredPairedScaleTemplate_1.1.yaml"
        ),
    )
    parser.add_argument("--outer-family", required=True, choices=FAMILY_ORDER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sidecar-input-manifest", required=True)
    parser.add_argument("--sidecar-input-manifest-sha256", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--sidecar-population-manifest", required=True)
    parser.add_argument("--sidecar-population-manifest-sha256", required=True)
    parser.add_argument("--expected-config-sha256", default=EXPECTED_CONFIG_SHA256)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    run(
        arguments.config,
        arguments.outer_family,
        arguments.output_dir,
        device=arguments.device,
        sidecar_input_manifest_path=arguments.sidecar_input_manifest,
        sidecar_input_manifest_file_sha256=arguments.sidecar_input_manifest_sha256,
        sidecar_root=arguments.sidecar_root,
        sidecar_population_manifest_path=arguments.sidecar_population_manifest,
        sidecar_population_manifest_file_sha256=(
            arguments.sidecar_population_manifest_sha256
        ),
        expected_config_sha256=arguments.expected_config_sha256,
    )


if __name__ == "__main__":
    main()

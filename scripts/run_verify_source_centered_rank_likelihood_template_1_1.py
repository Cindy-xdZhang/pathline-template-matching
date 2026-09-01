#!/usr/bin/env python3
"""Frozen nested-family runner for SourceCenteredRankLikelihoodTemplate 1.1.

The inherited source-centered sidecars are immutable producer artifacts.  This
runner authenticates their historical Git/config identity, but never rewrites
them and never pretends that they were produced by this experiment.  Only the
assigned centered-curl coordinate and the parent valid-row identities are
opened.  FMT features, raw features, and outer labels are not opened while the
rank-likelihood library, candidate selection, or outer predictions are built.

For every source, centered-curl values are converted to empirical midranks in
each scale-block/dx group.  A candidate first fuses the two assigned block
ranks at each center.  Combined-valid fit centers form the positive/negative
template library.  The primary score is the calibrated dual-class histogram
likelihood score; a negative-only empirical-CDF arm and the direct rank arm are
reported as ineligible controls.  Outer valid labels are opened only after the
complete prediction NPZ and manifest have been freshly replayed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.paired_scale_center_fusion import (  # noqa: E402
    DEFAULT_CENTER_COUNT,
    DEFAULT_TOP_FRACTION,
    direct_source_centered_diagnostics,
    fixed_top_fraction_over_centers,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.source_centered_sidecar import (  # noqa: E402
    ASSIGNED_ROW_COUNT,
)
from pathline_template_matching.source_centered_rank_likelihood import (  # noqa: E402
    FamilyBalancedRankLikelihoodModel,
    FamilySourceRankBatch,
    PairedCenterRanks,
    assigned_block_dx_midranks,
    conservative_strict_ecdf,
    pair_assigned_center_ranks,
    strict_absolute_threshold,
)
from scripts import (  # noqa: E402
    run_verify_source_centered_paired_scale_template_1_1 as source_runner,
)
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
    load_cache_rows,
)


EXPERIMENT = "Verify_SourceCenteredRankLikelihoodTemplate_1.1"
# Replaced only after the protocol/config bytes are frozen.  load_plan rejects
# this sentinel, so an unfrozen checkout cannot start a numerical run.
EXPECTED_CONFIG_SHA256 = (
    "41d6e7be70b898715c6df6f92cfb17176d2f1bb6153fa37b09dd4da9a6059ffa"
)

SOURCE_EXPERIMENT = source_runner.EXPERIMENT
SOURCE_CONFIG_SHA256 = source_runner.EXPECTED_CONFIG_SHA256
SOURCE_NUMERICAL_COMMIT = "a85c007ef961ce53bb40946ca3f38f033bf7a646"
SOURCE_INPUT_MANIFEST_SHA256 = (
    "5f7e567a2f989d18b51389814938a5d18025c4ed5247730d07df30b13458fec9"
)
SOURCE_POPULATION_MANIFEST_SHA256 = (
    "50d9d53f7dc9255d5153f0101c922975e006303b550bfb43317074080a0a97e2"
)
SOURCE_AGGREGATE_JOB_ID = 51160422

FAMILY_ORDER = source_runner.FAMILY_ORDER
BLOCK_NAMES = source_runner.BLOCK_NAMES
GRID_SHAPE = source_runner.GRID_SHAPE
WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
BIN_COUNTS = (64, 128, 256)
BETAS = (0.5, 2.0)
SIGMAS = (0.0, 0.5, 1.0)
THRESHOLDS = (0.90, 0.925, 0.95, 0.975, 0.99, 0.995)
PRIMARY_CANDIDATE_COUNT = 540
NEGATIVE_CONTROL_CANDIDATE_COUNT = 90
DIRECT_WEIGHT = 0.5
DIRECT_TOP_FRACTION = 0.05

PARENT_BINDING_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_parent_binding.v1"
)
BINDING_COMPLETE_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_binding_complete.v1"
)
MODEL_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_model.v1"
)
MODEL_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_model_manifest.v1"
)
CALIBRATION_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_calibration.v1"
)
CALIBRATION_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_calibration_manifest.v1"
)
CONTROL_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_negative_control.v1"
)
CONTROL_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_negative_control_manifest.v1"
)
SELECTED_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_selected.v1"
)
OUTER_FEATURE_BINDING_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_outer_feature_binding.v1"
)
PREDICTION_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_prediction.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_prediction_manifest.v1"
)
INNER_AUDIT_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_inner_audit.v1"
)
OUTER_SUMMARY_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_outer_summary.v1"
)
REFERENCE_AUDIT_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_reference_audit.v1"
)
RESULT_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_result.v1"
)
COMPLETE_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_complete.v1"
)

REQUIRED_FOLD_FILES = (
    "inner_group_metrics.csv",
    "inner_candidate_summary.csv",
    "inner_fit_audits.json",
    "final_rank_likelihood_model.npz",
    "final_rank_likelihood_model_manifest.json",
    "final_rank_likelihood_calibration.npz",
    "final_rank_likelihood_calibration_manifest.json",
    "final_negative_ecdf_control.npz",
    "final_negative_ecdf_control_manifest.json",
    "selected_candidate.json",
    "outer_rank_binding.json",
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
    source_plan: source_runner.Plan
    family_order: tuple[str, ...]
    families: Mapping[str, tuple[str, ...]]
    dataset_to_family: Mapping[str, str]
    weights: tuple[float, ...]
    bin_counts: tuple[int, ...]
    betas: tuple[float, ...]
    sigmas: tuple[float, ...]
    thresholds: tuple[float, ...]
    grid_shape: tuple[int, int, int]
    gaussian_truncate: float
    output_root: Path
    parent_run_directories: Mapping[str, Path]
    source_input_manifest_path: Path
    source_sidecar_root: Path
    source_population_manifest_path: Path
    required_fold_files: tuple[str, ...] = REQUIRED_FOLD_FILES
    source_evidence: Mapping[str, Any] | None = None
    parent_binding_path: Path | None = None
    parent_binding_file_sha256: str | None = None
    parent_binding_content_sha256: str | None = None
    binding_completion_path: Path | None = None
    binding_completion_file_sha256: str | None = None


@dataclass
class RankCacheProjection:
    row: CacheRow
    scale_ids: np.ndarray
    center_indices: np.ndarray
    block_indices: np.ndarray
    assigned_row_indices: np.ndarray
    labels: np.ndarray | None
    assigned_center_indices: np.ndarray
    assigned_block_indices: np.ndarray
    assigned_scale_ids: np.ndarray
    assigned_source_centered_seed4: np.ndarray
    sidecar_file_sha256: str
    sidecar_combined_array_sha256: str
    parent_members_opened: tuple[str, ...]
    sidecar_members_opened: tuple[str, ...]
    assigned_rank_cache: np.ndarray | None = field(default=None, repr=False)

    @property
    def count(self) -> int:
        return len(self.scale_ids)


@dataclass(frozen=True)
class CandidateSpec:
    weight: float
    bin_count: int
    beta: float
    sigma: float
    threshold: float

    @property
    def candidate_id(self) -> str:
        return (
            "arm=dual_histogram_llr"
            f"|legacy_weight={self.weight:.2f}|bins={self.bin_count}"
            f"|beta={self.beta:.1f}|sigma={self.sigma:.1f}"
            f"|strict_threshold={self.threshold:.3f}"
        )


@dataclass(frozen=True)
class ControlSpec:
    weight: float
    sigma: float
    threshold: float

    @property
    def candidate_id(self) -> str:
        return (
            "arm=negative_ecdf"
            f"|legacy_weight={self.weight:.2f}|sigma={self.sigma:.1f}"
            f"|strict_threshold={self.threshold:.3f}"
        )


@dataclass(frozen=True)
class VerifiedLibrary:
    artifact_paths: Mapping[str, Path]
    manifest_paths: Mapping[str, Path]
    artifact_file_sha256: Mapping[str, str]
    manifest_file_sha256: Mapping[str, str]
    manifests: Mapping[str, Mapping[str, Any]]
    arrays: Mapping[str, np.ndarray]
    model: FamilyBalancedRankLikelihoodModel = field(repr=False, compare=False)
    seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedSelection:
    path: Path
    file_sha256: str
    manifest: Mapping[str, Any]
    seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedFeatureBinding:
    path: Path
    file_sha256: str
    manifest: Mapping[str, Any]
    seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedPrediction:
    artifact_path: Path
    manifest_path: Path
    artifact_file_sha256: str
    manifest_file_sha256: str
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class AuthenticatedParentControl:
    run_directory: Path
    outer_family: str
    arrays: Mapping[str, np.ndarray]
    evidence: Mapping[str, Any]


_AUTHENTICATION_SEAL = object()


def _manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return source_runner._manifest(payload)


def _read_manifest(path: Path, expected_file_sha256: str) -> tuple[dict[str, Any], str]:
    return source_runner._read_manifest(path, expected_file_sha256)


def _candidate_payload(candidate: CandidateSpec) -> dict[str, Any]:
    return {
        "arm": "dual_histogram_llr",
        "candidate_id": candidate.candidate_id,
        "weight": candidate.weight,
        "bin_count": candidate.bin_count,
        "beta": candidate.beta,
        "sigma": candidate.sigma,
        "decision_rule": "strict_absolute_threshold",
        "decision_value": candidate.threshold,
    }


def _control_payload(candidate: ControlSpec) -> dict[str, Any]:
    return {
        "arm": "negative_ecdf",
        "candidate_id": candidate.candidate_id,
        "weight": candidate.weight,
        "bin_count": None,
        "beta": None,
        "sigma": candidate.sigma,
        "decision_rule": "strict_absolute_threshold",
        "decision_value": candidate.threshold,
    }


def candidate_specs(plan: Plan) -> tuple[CandidateSpec, ...]:
    result = tuple(
        CandidateSpec(weight, bins, beta, sigma, threshold)
        for weight in plan.weights
        for bins in plan.bin_counts
        for beta in plan.betas
        for sigma in plan.sigmas
        for threshold in plan.thresholds
    )
    _require(len(result) == PRIMARY_CANDIDATE_COUNT, "primary candidate count drifted")
    _require(len({item.candidate_id for item in result}) == len(result), "duplicate primary candidate")
    return result


def control_specs(plan: Plan) -> tuple[ControlSpec, ...]:
    result = tuple(
        ControlSpec(weight, sigma, threshold)
        for weight in plan.weights
        for sigma in plan.sigmas
        for threshold in plan.thresholds
    )
    _require(
        len(result) == NEGATIVE_CONTROL_CANDIDATE_COUNT,
        "negative-control candidate count drifted",
    )
    _require(len({item.candidate_id for item in result}) == len(result), "duplicate control candidate")
    return result


def _mapping(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    _require(isinstance(value, Mapping), f"config section is missing: {name}")
    return value


def load_plan(config_path: str | Path) -> Plan:
    path = Path(config_path).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    _require(
        digest == EXPECTED_CONFIG_SHA256, "frozen config SHA-256 drifted"
    )
    raw_value = yaml.safe_load(payload.decode("utf-8"))
    _require(isinstance(raw_value, Mapping), "config root must be a mapping")
    raw = dict(raw_value)
    _require(raw.get("experiment") == EXPERIMENT, "experiment identity drifted")
    _require(
        raw.get("status")
        == "frozen_after_authenticated_parent_diagnostics_before_any_new_method_prediction_or_metric",
        "freeze status drifted",
    )

    source = _mapping(raw, "direct_parent")
    input_record = _mapping(source, "source_centered_input_manifest")
    population_record = _mapping(source, "source_centered_sidecar_population")
    _require(
        source.get("experiment") == SOURCE_EXPERIMENT
        and source.get("config_sha256") == SOURCE_CONFIG_SHA256
        and source.get("numerical_commit") == SOURCE_NUMERICAL_COMMIT
        and input_record.get("file_sha256") == SOURCE_INPUT_MANIFEST_SHA256
        and population_record.get("file_sha256")
        == SOURCE_POPULATION_MANIFEST_SHA256,
        "inherited source-centered producer identity drifted",
    )
    source_config_path = (ROOT / str(source["config_path"])).resolve()
    source_plan = source_runner.load_plan(source_config_path)

    families_raw = _mapping(raw, "families")
    split = _mapping(raw, "nested_split")
    family_order = tuple(str(value) for value in split.get("outer_order", ()))
    _require(
        family_order == FAMILY_ORDER
        and tuple(split.get("inner_order", ())) == FAMILY_ORDER,
        "nested family order drifted",
    )
    _require(split.get("outer_labels_available_to_selection") is False, "outer label gate drifted")
    families = {
        str(name): tuple(str(value) for value in values)
        for name, values in families_raw.items()
    }
    _require(tuple(families) == FAMILY_ORDER, "family mapping order drifted")
    _require(families == dict(source_plan.families), "source-centered family population drifted")

    rank = _mapping(raw, "rank_representation")
    primary = _mapping(raw, "primary_rank_likelihood_template")
    control = _mapping(raw, "negative_ecdf_control")
    spatial = _mapping(raw, "spatial_transform")
    candidates = _mapping(raw, "candidates")
    decision = _mapping(raw, "decision_rule")
    weights = tuple(float(value) for value in rank.get("weights", ()))
    bin_counts = tuple(int(value) for value in primary.get("bin_counts", ()))
    betas = tuple(float(value) for value in primary.get("additive_beta_values", ()))
    sigmas = tuple(float(value) for value in spatial.get("gaussian_sigmas_grid_indices", ()))
    thresholds = tuple(float(value) for value in decision.get("tau_values", ()))
    _require(weights == WEIGHTS, "weight grid drifted")
    _require(bin_counts == BIN_COUNTS, "histogram-bin grid drifted")
    _require(betas == BETAS, "beta grid drifted")
    _require(sigmas == SIGMAS, "spatial sigma grid drifted")
    _require(thresholds == THRESHOLDS, "strict-threshold grid drifted")
    _require(
        decision.get("comparison") == "score_strictly_greater_than_tau"
        and primary.get("library_unit") == "one_combined_valid_center_once"
        and control.get("uses_positive_templates") is False,
        "rank-likelihood score semantics drifted",
    )
    primary_candidates = _mapping(candidates, "primary")
    control_candidates = _mapping(candidates, "negative_ecdf_control")
    _require(
        int(primary_candidates.get("exact_count", -1)) == PRIMARY_CANDIDATE_COUNT
        and int(control_candidates.get("exact_count", -1))
        == NEGATIVE_CONTROL_CANDIDATE_COUNT
        and candidates.get("outer_result_selection") == "forbidden",
        "candidate population drifted",
    )
    output = _mapping(raw, "output")
    _require(output.get("overwrite") == "forbidden", "output overwrite contract drifted")
    parent = source
    directories = source.get("outer_run_directories")
    aggregate_record = _mapping(source, "authenticated_aggregate")
    _require(
        parent.get("experiment") == SOURCE_EXPERIMENT
        and parent.get("config_sha256") == SOURCE_CONFIG_SHA256
        and parent.get("numerical_commit") == SOURCE_NUMERICAL_COMMIT
        and int(aggregate_record.get("job_id", -1))
        == SOURCE_AGGREGATE_JOB_ID
        and isinstance(directories, Mapping)
        and set(directories) == set(FAMILY_ORDER),
        "direct-parent control identity drifted",
    )
    plan = Plan(
        path=path,
        sha256=digest,
        raw=MappingProxyType(raw),
        source_plan=source_plan,
        family_order=family_order,
        families=MappingProxyType(families),
        dataset_to_family=MappingProxyType(
            {dataset: family for family in family_order for dataset in families[family]}
        ),
        weights=weights,
        bin_counts=bin_counts,
        betas=betas,
        sigmas=sigmas,
        thresholds=thresholds,
        grid_shape=tuple(int(value) for value in spatial["grid_shape_zyx"]),
        gaussian_truncate=float(spatial["gaussian_truncate"]),
        output_root=Path(str(output["root"])),
        parent_run_directories=MappingProxyType(
            {family: Path(str(directories[family])).resolve() for family in family_order}
        ),
        source_input_manifest_path=Path(str(input_record["path"])).resolve(),
        source_sidecar_root=(
            Path(str(population_record["path"])).resolve().parent
        ),
        source_population_manifest_path=Path(
            str(population_record["path"])
        ).resolve(),
    )
    candidate_specs(plan)
    control_specs(plan)
    return plan


def _git_blob_sha256(commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(
        result.returncode == 0,
        f"historical producer source is unavailable: {commit}:{path}",
    )
    return hashlib.sha256(result.stdout).hexdigest()


def authenticate_historical_sidecar_population(plan: Plan) -> Plan:
    """Authenticate parent envelopes/files without opening any sidecar NPZ.

    The old producer authenticator intentionally requires the running checkout
    to be the producer commit.  Reusing immutable data at a later commit needs
    a historical adapter instead: every recorded producer source hash is
    checked against the exact Git blob at ``SOURCE_NUMERICAL_COMMIT`` and every
    manifest/completion/sidecar byte identity is replayed.  No NPZ member is
    opened in this stage.
    """

    input_path = plan.source_input_manifest_path
    population_path = plan.source_population_manifest_path
    sidecar_root = plan.source_sidecar_root
    _require(
        population_path == sidecar_root / "SIDECAR_POPULATION.json",
        "historical population path differs from its sidecar root",
    )
    input_value = source_runner._read_self_hashed_json(
        input_path, expected_file_sha256=SOURCE_INPUT_MANIFEST_SHA256
    )
    _require(
        input_value.get("experiment") == SOURCE_EXPERIMENT
        and input_value.get("git_commit") == SOURCE_NUMERICAL_COMMIT
        and input_value.get("verify_config_sha256") == SOURCE_CONFIG_SHA256
        and input_value.get("status") == "frozen",
        "historical input manifest provenance drifted",
    )
    input_rows = input_value.get("rows")
    _require(
        isinstance(input_rows, list)
        and len(input_rows) == 32
        and input_value.get("rows_content_sha256")
        == canonical_json_sha256(input_rows),
        "historical input row population drifted",
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
        "historical input rows are missing, duplicated, extra, or reordered",
    )

    population = source_runner._read_self_hashed_json(
        population_path, expected_file_sha256=SOURCE_POPULATION_MANIFEST_SHA256
    )
    _require(
        set(population) == source_runner._POPULATION_FIELDS,
        "historical population field set drifted",
    )
    _require(
        population.get("experiment") == SOURCE_EXPERIMENT
        and population.get("git_commit") == SOURCE_NUMERICAL_COMMIT
        and population.get("verify_config_sha256") == SOURCE_CONFIG_SHA256
        and population.get("input_manifest_path") == str(input_path)
        and population.get("input_manifest_file_sha256")
        == SOURCE_INPUT_MANIFEST_SHA256
        and int(population.get("sidecar_count", -1)) == 32
        and int(population.get("assigned_row_count_total", -1))
        == 32 * ASSIGNED_ROW_COUNT
        and population.get("forbidden_parent_members_opened") == []
        and population.get("forbidden_dataset_access") is False,
        "historical population provenance drifted",
    )
    recorded_sources = population.get("source_file_sha256")
    _require(isinstance(recorded_sources, Mapping), "producer source hashes are missing")
    for source_path, expected_sha in sorted(recorded_sources.items()):
        _require(
            _git_blob_sha256(SOURCE_NUMERICAL_COMMIT, str(source_path))
            == str(expected_sha),
            f"producer Git blob hash drifted: {source_path}",
        )
    _require(
        population.get("source_file_sha256_content_sha256")
        == canonical_json_sha256(dict(recorded_sources)),
        "producer source hash map content drifted",
    )

    rows = population.get("rows")
    _require(
        isinstance(rows, list)
        and len(rows) == 32
        and population.get("rows_content_sha256") == canonical_json_sha256(rows),
        "historical sidecar rows drifted",
    )
    expected_files = {population_path}
    valid_total = 0
    for input_row, row in zip(input_rows, rows, strict=True):
        _require(
            isinstance(row, Mapping)
            and set(row) == source_runner._POPULATION_ROW_FIELDS,
            "historical sidecar row fields drifted",
        )
        for identity_name in (
            "dataset",
            "dataset_index",
            "physical_family",
            "source_ordinal",
            "source_index",
        ):
            _require(
                row.get(identity_name) == input_row.get(identity_name),
                f"historical input/population identity drifted: {identity_name}",
            )
        completion = (sidecar_root / str(row["completion_relative_path"])).resolve()
        sidecar = (sidecar_root / str(row["sidecar_relative_path"])).resolve()
        _require(
            completion.is_relative_to(sidecar_root)
            and sidecar.is_relative_to(sidecar_root),
            "historical sidecar child escapes its root",
        )
        expected_files.update((completion, sidecar))
        _stable_file_identity(
            completion,
            int(row["completion_size_bytes"]),
            str(row["completion_file_sha256"]),
        )
        _stable_file_identity(
            sidecar,
            int(row["sidecar_size_bytes"]),
            str(row["sidecar_file_sha256"]),
        )
        completion_value = source_runner._read_self_hashed_json(
            completion,
            expected_file_sha256=str(row["completion_file_sha256"]),
        )
        _require(
            completion_value.get("experiment") == SOURCE_EXPERIMENT
            and completion_value.get("dataset") == row["dataset"]
            and completion_value.get("source_ordinal") == row["source_ordinal"]
            and completion_value.get("sidecar_file_sha256")
            == row["sidecar_file_sha256"]
            and completion_value.get("sidecar_combined_array_sha256")
            == row["sidecar_combined_array_sha256"]
            and completion_value.get("valid_projection_sha256")
            == row["valid_projection_sha256"],
            "historical completion/sidecar binding drifted",
        )
        _require(
            int(row["assigned_row_count"]) == ASSIGNED_ROW_COUNT,
            "historical assigned row count drifted",
        )
        valid_total += int(row["valid_projection_row_count"])
    actual_files = {
        child.resolve() for child in sidecar_root.rglob("*") if child.is_file()
    }
    _require(actual_files == expected_files, "historical sidecar file set drifted")
    _require(
        int(population["valid_projection_row_count_total"]) == valid_total,
        "historical valid projection total drifted",
    )
    bound_source_plan = replace(
        plan.source_plan,
        sidecar_input_manifest_path=input_path,
        sidecar_input_manifest_file_sha256=SOURCE_INPUT_MANIFEST_SHA256,
        sidecar_input_manifest_content_sha256=str(input_value["content_sha256"]),
        sidecar_root=sidecar_root,
        sidecar_population_path=population_path,
        sidecar_population_file_sha256=SOURCE_POPULATION_MANIFEST_SHA256,
        sidecar_population_content_sha256=str(population["content_sha256"]),
        sidecar_population=source_runner.early._deep_freeze(population),
    )
    evidence = {
        "producer_experiment": SOURCE_EXPERIMENT,
        "producer_git_commit": SOURCE_NUMERICAL_COMMIT,
        "producer_config_sha256": SOURCE_CONFIG_SHA256,
        "input_manifest": {
            "path": str(input_path),
            "file_sha256": SOURCE_INPUT_MANIFEST_SHA256,
            "content_sha256": input_value["content_sha256"],
        },
        "sidecar_population": {
            "root": str(sidecar_root),
            "manifest_path": str(population_path),
            "manifest_file_sha256": SOURCE_POPULATION_MANIFEST_SHA256,
            "manifest_content_sha256": population["content_sha256"],
            "row_count": 32,
            "rows_content_sha256": population["rows_content_sha256"],
            "assigned_row_count_total": 32 * ASSIGNED_ROW_COUNT,
            "valid_projection_row_count_total": valid_total,
        },
        "historical_source_file_sha256": dict(recorded_sources),
        "historical_source_file_sha256_content_sha256": population[
            "source_file_sha256_content_sha256"
        ],
        "sidecar_npz_members_opened": [],
        "labels_or_references_opened": [],
        "authentication_mode": "historical_git_blob_and_complete_file_hash_replay",
    }
    return replace(
        plan,
        source_plan=bound_source_plan,
        source_evidence=source_runner.early._deep_freeze(evidence),
    )


def bind_parent_sidecar_release(
    plan: Plan,
    *,
    parent_binding_path: str | Path,
    parent_binding_file_sha256: str,
    binding_completion_path: str | Path,
    binding_completion_file_sha256: str,
) -> Plan:
    """Freshly replay a two-file binding, then replay all historical files."""

    binding_path = Path(parent_binding_path).resolve()
    completion_path = Path(binding_completion_path).resolve()
    _require(
        binding_path.parent == completion_path.parent
        and binding_path.name == "parent_sidecar_binding.json"
        and completion_path.name == "BINDING_COMPLETE.json",
        "parent binding file layout drifted",
    )
    _require(
        {child.name for child in binding_path.parent.iterdir()}
        == {binding_path.name, completion_path.name},
        "parent binding directory contains unexpected files",
    )
    binding = source_runner._read_self_hashed_json(
        binding_path, expected_file_sha256=parent_binding_file_sha256
    )
    completion = source_runner._read_self_hashed_json(
        completion_path, expected_file_sha256=binding_completion_file_sha256
    )
    git_commit, dirty = _git_identity()
    _require(not dirty, "rank-likelihood run requires a clean worktree")
    _require(
        binding.get("schema") == PARENT_BINDING_SCHEMA
        and binding.get("experiment") == EXPERIMENT
        and binding.get("git_commit") == git_commit
        and binding.get("config_sha256") == plan.sha256
        and completion.get("schema") == BINDING_COMPLETE_SCHEMA
        and completion.get("experiment") == EXPERIMENT
        and completion.get("git_commit") == git_commit
        and completion.get("config_sha256") == plan.sha256
        and completion.get("parent_binding_file") == binding_path.name
        and completion.get("parent_binding_file_sha256")
        == parent_binding_file_sha256
        and completion.get("parent_binding_content_sha256")
        == binding.get("content_sha256"),
        "parent binding provenance drifted",
    )
    replayed = authenticate_historical_sidecar_population(plan)
    _require(
        binding.get("historical_source_centered_evidence")
        == _json_safe(replayed.source_evidence),
        "parent binding differs from fresh historical replay",
    )
    return replace(
        replayed,
        parent_binding_path=binding_path,
        parent_binding_file_sha256=parent_binding_file_sha256,
        parent_binding_content_sha256=str(binding["content_sha256"]),
        binding_completion_path=completion_path,
        binding_completion_file_sha256=binding_completion_file_sha256,
    )


def _require_bound(plan: Plan) -> None:
    _require(plan.source_evidence is not None, "historical sidecar evidence is unbound")
    _require(plan.parent_binding_path is not None, "parent binding release is unbound")
    source_runner._require_evidence_bound(plan.source_plan)


def _population_row(plan: Plan, row: CacheRow) -> Mapping[str, Any]:
    _require_bound(plan)
    return source_runner._population_row(plan.source_plan, row)


def _load_parent_minimal(
    row: CacheRow,
    *,
    include_labels: bool,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Open only identity fields, plus valid_labels in an authorized phase."""

    names = [
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
    ]
    if include_labels:
        names.append("valid_labels")
    with source_runner.early._authenticated_open_file(
        row.path, expected_size=row.size_bytes, expected_sha256=row.sha256
    ) as opened:
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(
                all(name in archive.files for name in names),
                f"{row.path}: minimal parent projection is incomplete",
            )
            arrays = {
                name: np.array(archive[name], copy=True, order="C") for name in names
            }
    scales = arrays["valid_scale_id"]
    centers = arrays["valid_center_seed_index"]
    blocks = arrays["valid_scale_block_index"]
    assigned = arrays["valid_assigned_row_index"]
    count = len(scales)
    expected = {
        "valid_scale_id": np.dtype(np.int32),
        "valid_center_seed_index": np.dtype(np.int64),
        "valid_scale_block_index": np.dtype(np.int8),
        "valid_assigned_row_index": np.dtype(np.int64),
    }
    if include_labels:
        expected["valid_labels"] = np.dtype(np.bool_)
    for name, dtype in expected.items():
        _require(
            arrays[name].dtype == dtype and arrays[name].shape == (count,),
            f"{row.path}: minimal parent member contract drifted: {name}",
        )
    _require(np.all((scales >= 0) & (scales < 2000)), "parent scale outside 0..1999")
    _require(np.all((centers >= 0) & (centers < DEFAULT_CENTER_COUNT)), "parent center outside grid")
    _require(np.all((blocks >= 0) & (blocks < 2)), "parent block outside 0..1")
    _require(np.array_equal(blocks, (scales >= 1000).astype(np.int8)), "parent scale/block mismatch")
    _require(
        np.array_equal(
            assigned,
            blocks.astype(np.int64) * DEFAULT_CENTER_COUNT + centers,
        ),
        "parent assigned-row identity mismatch",
    )
    for block_index in (0, 1):
        selected = centers[blocks == block_index]
        _require(len(np.unique(selected)) == len(selected), "duplicate parent block/center")
    return arrays, tuple(names)


def load_rank_cache_projection(
    plan: Plan,
    row: CacheRow,
    *,
    include_labels: bool,
) -> RankCacheProjection:
    parent, opened_members = _load_parent_minimal(row, include_labels=include_labels)
    population_row = _population_row(plan, row)
    assert plan.source_plan.sidecar_root is not None
    sidecar_path = (
        plan.source_plan.sidecar_root / str(population_row["sidecar_relative_path"])
    ).resolve()
    _require(sidecar_path.is_relative_to(plan.source_plan.sidecar_root), "sidecar path escapes root")
    sidecar_names = (
        "assigned_row_index",
        "center_seed_index",
        "scale_block_index",
        "scale_id",
        "source_centered_seed4",
        "valid_assigned_row_index",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_scale_id",
    )
    with source_runner.early._authenticated_open_file(
        sidecar_path,
        expected_size=int(population_row["sidecar_size_bytes"]),
        expected_sha256=str(population_row["sidecar_file_sha256"]),
    ) as opened:
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(
                all(name in archive.files for name in sidecar_names),
                "minimal sidecar projection is incomplete",
            )
            sidecar_arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in sidecar_names
            }
    expected_sidecar = {
        "assigned_row_index": (np.dtype(np.int64), (ASSIGNED_ROW_COUNT,)),
        "center_seed_index": (np.dtype(np.int64), (ASSIGNED_ROW_COUNT,)),
        "scale_block_index": (np.dtype(np.int8), (ASSIGNED_ROW_COUNT,)),
        "scale_id": (np.dtype(np.int32), (ASSIGNED_ROW_COUNT,)),
        "source_centered_seed4": (
            np.dtype(np.float32),
            (ASSIGNED_ROW_COUNT, 4),
        ),
    }
    valid_count = len(parent["valid_assigned_row_index"])
    expected_sidecar.update(
        {
            "valid_assigned_row_index": (np.dtype(np.int64), (valid_count,)),
            "valid_center_seed_index": (np.dtype(np.int64), (valid_count,)),
            "valid_scale_block_index": (np.dtype(np.int8), (valid_count,)),
            "valid_scale_id": (np.dtype(np.int32), (valid_count,)),
        }
    )
    for name, (dtype, shape) in expected_sidecar.items():
        values = sidecar_arrays[name]
        _require(
            values.dtype == dtype and values.shape == shape,
            f"minimal sidecar member contract drifted: {name}",
        )
    assigned_centers = sidecar_arrays["center_seed_index"]
    assigned_blocks = sidecar_arrays["scale_block_index"]
    assigned_scales = sidecar_arrays["scale_id"]
    _require(
        np.array_equal(
            sidecar_arrays["assigned_row_index"],
            assigned_blocks.astype(np.int64) * DEFAULT_CENTER_COUNT
            + assigned_centers,
        )
        and np.array_equal(
            assigned_blocks, (assigned_scales // 1000).astype(np.int8)
        ),
        "minimal sidecar assigned identity drifted",
    )
    seed4 = sidecar_arrays["source_centered_seed4"]
    _require(
        np.isfinite(seed4).all() and np.all(seed4[:, 0] >= 0.0),
        "minimal sidecar source-centered rank coordinate is invalid",
    )
    for left, right, name in (
        (sidecar_arrays["valid_assigned_row_index"], parent["valid_assigned_row_index"], "assigned row"),
        (sidecar_arrays["valid_center_seed_index"], parent["valid_center_seed_index"], "center"),
        (sidecar_arrays["valid_scale_block_index"], parent["valid_scale_block_index"], "block"),
        (sidecar_arrays["valid_scale_id"], parent["valid_scale_id"], "scale"),
    ):
        _require(np.array_equal(left, right), f"sidecar/parent {name} join drifted")
    return RankCacheProjection(
        row=row,
        scale_ids=np.ascontiguousarray(parent["valid_scale_id"]),
        center_indices=np.ascontiguousarray(parent["valid_center_seed_index"]),
        block_indices=np.ascontiguousarray(parent["valid_scale_block_index"]),
        assigned_row_indices=np.ascontiguousarray(parent["valid_assigned_row_index"]),
        labels=(
            np.ascontiguousarray(parent["valid_labels"], dtype=np.bool_)
            if include_labels
            else None
        ),
        assigned_center_indices=np.ascontiguousarray(assigned_centers),
        assigned_block_indices=np.ascontiguousarray(assigned_blocks),
        assigned_scale_ids=np.ascontiguousarray(assigned_scales),
        assigned_source_centered_seed4=np.ascontiguousarray(seed4),
        sidecar_file_sha256=str(population_row["sidecar_file_sha256"]),
        sidecar_combined_array_sha256=str(
            population_row["sidecar_combined_array_sha256"]
        ),
        parent_members_opened=opened_members,
        sidecar_members_opened=sidecar_names,
    )


def _assigned_ranks(cache: RankCacheProjection) -> np.ndarray:
    if cache.assigned_rank_cache is None:
        cache.assigned_rank_cache = assigned_block_dx_midranks(
            cache.assigned_center_indices,
            cache.assigned_block_indices,
            cache.assigned_scale_ids,
            np.ascontiguousarray(cache.assigned_source_centered_seed4[:, 0]),
            center_count=DEFAULT_CENTER_COUNT,
        )
    return cache.assigned_rank_cache


def paired_center_ranks(cache: RankCacheProjection, weight: float) -> PairedCenterRanks:
    return pair_assigned_center_ranks(
        cache.assigned_center_indices,
        cache.assigned_block_indices,
        _assigned_ranks(cache),
        cache.center_indices,
        cache.block_indices,
        weight=weight,
        center_count=DEFAULT_CENTER_COUNT,
    )


def _center_labels(cache: RankCacheProjection, paired: PairedCenterRanks) -> np.ndarray:
    _require(cache.labels is not None, "fit labels are unavailable")
    labels = np.zeros(DEFAULT_CENTER_COUNT, dtype=np.bool_)
    seen = np.zeros(DEFAULT_CENTER_COUNT, dtype=np.bool_)
    for center, value in zip(cache.center_indices, cache.labels, strict=True):
        index = int(center)
        if seen[index]:
            _require(labels[index] == bool(value), "both valid blocks disagree on center label")
        else:
            labels[index] = bool(value)
            seen[index] = True
    _require(np.array_equal(seen, paired.combined_valid), "center label population differs from combined valid")
    return labels


def family_rank_batches(
    caches: Sequence[RankCacheProjection],
    weight: float,
) -> tuple[dict[str, FamilySourceRankBatch], list[dict[str, Any]]]:
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    audits: list[dict[str, Any]] = []
    ordered_caches = sorted(
        caches,
        key=lambda cache: (
            cache.row.family,
            cache.row.dataset,
            cache.row.source_ordinal,
            cache.row.source_index,
        ),
    )
    source_keys = [
        (
            cache.row.dataset,
            cache.row.source_ordinal,
            cache.row.source_index,
        )
        for cache in ordered_caches
    ]
    _require(len(source_keys) == len(set(source_keys)), "duplicate fit source identity")
    source_id_by_key = {
        key: index for index, key in enumerate(source_keys)
    }
    for cache in ordered_caches:
        paired = paired_center_ranks(cache, weight)
        labels = _center_labels(cache, paired)
        selected = paired.combined_valid
        ranks = np.ascontiguousarray(paired.paired_rank[selected], dtype=np.float64)
        selected_labels = np.ascontiguousarray(labels[selected], dtype=np.bool_)
        source_key = (
            cache.row.dataset,
            cache.row.source_ordinal,
            cache.row.source_index,
        )
        stable_source_id = source_id_by_key[source_key]
        source_ids = np.full(len(ranks), stable_source_id, dtype=np.int64)
        grouped.setdefault(cache.row.family, []).append((ranks, selected_labels, source_ids))
        audits.append(
            {
                "dataset": cache.row.dataset,
                "physical_family": cache.row.family,
                "source_ordinal": cache.row.source_ordinal,
                "source_index": cache.row.source_index,
                "stable_loo_source_id": stable_source_id,
                "stable_loo_source_identity": {
                    "dataset": cache.row.dataset,
                    "source_ordinal": cache.row.source_ordinal,
                    "source_index": cache.row.source_index,
                },
                "combined_valid_center_count": int(selected.sum()),
                "positive_center_count": int(selected_labels.sum()),
                "negative_center_count": int((~selected_labels).sum()),
                "paired_rank_sha256": canonical_array_sha256(ranks),
                "label_sha256": canonical_array_sha256(selected_labels),
                "source_centered_sidecar_file_sha256": cache.sidecar_file_sha256,
                "parent_members_opened": list(cache.parent_members_opened),
                "sidecar_members_opened": list(cache.sidecar_members_opened),
                "fmt_features_opened": False,
                "raw_features_opened": False,
            }
        )
    batches: dict[str, FamilySourceRankBatch] = {}
    for family in sorted(grouped):
        parts = grouped[family]
        batches[family] = FamilySourceRankBatch(
            np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]),
            np.concatenate([part[2] for part in parts]),
        )
    return batches, audits


def fit_rank_model(
    caches: Sequence[RankCacheProjection],
    *,
    weight: float,
    bin_count: int,
    beta: float,
) -> tuple[FamilyBalancedRankLikelihoodModel, list[dict[str, Any]]]:
    batches, audits = family_rank_batches(caches, weight)
    return (
        FamilyBalancedRankLikelihoodModel(
            batches, bin_count=bin_count, beta=beta
        ),
        audits,
    )


def fit_negative_control(
    caches: Sequence[RankCacheProjection],
    *,
    weight: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Fit only per-family natural-negative rank references.

    Labels are read solely to exclude positive centers.  No positive
    histogram, likelihood, or leave-one-source-out dual reference is built.
    """

    grouped: dict[str, list[np.ndarray]] = {}
    audits: list[dict[str, Any]] = []
    ordered_caches = sorted(
        caches,
        key=lambda cache: (
            cache.row.family,
            cache.row.dataset,
            cache.row.source_ordinal,
            cache.row.source_index,
        ),
    )
    source_keys = [
        (
            cache.row.dataset,
            cache.row.source_ordinal,
            cache.row.source_index,
        )
        for cache in ordered_caches
    ]
    _require(len(source_keys) == len(set(source_keys)), "duplicate control source identity")
    source_id_by_key = {key: index for index, key in enumerate(source_keys)}
    for cache in ordered_caches:
        paired = paired_center_ranks(cache, weight)
        labels = _center_labels(cache, paired)
        combined_labels = labels[paired.combined_valid]
        combined_ranks = paired.paired_rank[paired.combined_valid]
        negative_ranks = np.ascontiguousarray(
            combined_ranks[~combined_labels], dtype=np.float64
        )
        grouped.setdefault(cache.row.family, []).append(negative_ranks)
        source_key = (
            cache.row.dataset,
            cache.row.source_ordinal,
            cache.row.source_index,
        )
        audits.append(
            {
                "dataset": cache.row.dataset,
                "physical_family": cache.row.family,
                "source_ordinal": cache.row.source_ordinal,
                "source_index": cache.row.source_index,
                "stable_source_id": source_id_by_key[source_key],
                "stable_source_identity": {
                    "dataset": cache.row.dataset,
                    "source_ordinal": cache.row.source_ordinal,
                    "source_index": cache.row.source_index,
                },
                "combined_valid_center_count": int(paired.combined_valid.sum()),
                "positive_center_count_observed_but_not_templated": int(
                    combined_labels.sum()
                ),
                "negative_reference_count": int(len(negative_ranks)),
                "negative_rank_sha256": canonical_array_sha256(negative_ranks),
                "source_centered_sidecar_file_sha256": cache.sidecar_file_sha256,
                "parent_members_opened": list(cache.parent_members_opened),
                "sidecar_members_opened": list(cache.sidecar_members_opened),
                "uses_positive_templates": False,
                "fmt_features_opened": False,
                "raw_features_opened": False,
            }
        )
    family_order = tuple(sorted(grouped))
    references = [
        np.sort(np.concatenate(grouped[family]), kind="mergesort")
        for family in family_order
    ]
    _require(
        len(family_order) >= 2 and all(len(values) > 0 for values in references),
        "every fit family must contain natural-negative control references",
    )
    offsets = np.zeros(len(references) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        np.asarray([len(values) for values in references], dtype=np.int64)
    )
    width = max(len(family) for family in family_order)
    arrays = {
        "control_family_order_unicode": np.asarray(
            family_order, dtype=f"<U{width}"
        ),
        "control_negative_rank_reference_values": np.ascontiguousarray(
            np.concatenate(references), dtype=np.float64
        ),
        "control_negative_rank_reference_offsets": offsets,
        "control_weight_float64": np.asarray(weight, dtype=np.float64),
    }
    return arrays, audits


def query_negative_control_arrays(
    arrays: Mapping[str, np.ndarray], ranks: np.ndarray
) -> np.ndarray:
    values = np.asarray(ranks, dtype=np.float64)
    family_order = np.asarray(arrays["control_family_order_unicode"])
    references = np.asarray(
        arrays["control_negative_rank_reference_values"], dtype=np.float64
    )
    offsets = np.asarray(
        arrays["control_negative_rank_reference_offsets"], dtype=np.int64
    )
    _require(
        family_order.ndim == 1
        and len(family_order) >= 2
        and offsets.shape == (len(family_order) + 1,)
        and offsets[0] == 0
        and offsets[-1] == len(references)
        and np.all(np.diff(offsets) > 0),
        "negative-control reference packing drifted",
    )
    score = np.zeros(len(values), dtype=np.float64)
    for index in range(len(family_order)):
        score += conservative_strict_ecdf(
            references[int(offsets[index]) : int(offsets[index + 1])], values
        )
    score /= float(len(family_order))
    return np.ascontiguousarray(score, dtype=np.float64)


def _spatial_center_score(
    paired: PairedCenterRanks,
    combined_valid_score: np.ndarray,
    *,
    sigma: float,
    plan: Plan,
) -> tuple[np.ndarray, np.ndarray]:
    selected_centers = paired.center_seed_index[paired.combined_valid]
    values = np.asarray(combined_valid_score, dtype=np.float64)
    _require(
        values.shape == selected_centers.shape,
        "combined-valid score population drifted",
    )
    spatial = source_runner.early.spatial_calibrated_tail_scores(
        values,
        np.ones(len(values), dtype=np.bool_),
        selected_centers,
        sigma=sigma,
        grid_shape=plan.grid_shape,
        truncate=plan.gaussian_truncate,
    )
    full_score = np.zeros(DEFAULT_CENTER_COUNT, dtype=np.float64)
    denominator = np.zeros(DEFAULT_CENTER_COUNT, dtype=np.float64)
    full_score[selected_centers] = spatial.scores
    denominator[selected_centers] = spatial.denominator
    return full_score, denominator


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
    _require(
        reference.ndim == values.ndim == predicted.ndim == 1
        and reference.shape == values.shape == predicted.shape
        and len(reference) > 0,
        "classification arrays are empty or misaligned",
    )
    _require(np.isfinite(values).all(), "classification score is nonfinite")
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
    "source_index",
    "arm",
    "candidate_id",
    "weight",
    "bin_count",
    "beta",
    "sigma",
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

INNER_SUMMARY_FIELDS = (
    "arm",
    "candidate_id",
    "weight",
    "bin_count",
    "beta",
    "sigma",
    "decision_rule",
    "decision_value",
    "inner_family_count",
    "group_count",
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
    "unique_center_combined_coverage",
)


def _coverage(paired: PairedCenterRanks) -> dict[str, Any]:
    both = paired.legacy_valid & paired.expanded_valid
    legacy_only = paired.legacy_valid & ~paired.expanded_valid
    expanded_only = ~paired.legacy_valid & paired.expanded_valid
    neither = ~paired.legacy_valid & ~paired.expanded_valid
    return {
        "unique_center_combined_coverage": float(paired.combined_valid.mean()),
        "both_valid_count": int(both.sum()),
        "legacy_only_count": int(legacy_only.sum()),
        "expanded_only_count": int(expanded_only.sum()),
        "neither_valid_count": int(neither.sum()),
    }


def _inner_row(
    *,
    outer_family: str,
    inner_family: str,
    cache: RankCacheProjection,
    paired: PairedCenterRanks,
    payload: Mapping[str, Any],
    center_scores: np.ndarray,
    center_predictions: np.ndarray,
    ranking_metrics: tuple[float, float] | None = None,
) -> dict[str, Any]:
    _require(cache.labels is not None, "inner labels are unavailable")
    scores = center_scores[cache.center_indices]
    predictions = center_predictions[cache.center_indices]
    return {
        "outer_family": outer_family,
        "inner_family": inner_family,
        "dataset": cache.row.dataset,
        "source_ordinal": cache.row.source_ordinal,
        "source_index": cache.row.source_index,
        **dict(payload),
        **_classification_metric_values(
            cache.labels,
            scores,
            predictions,
            ranking_metrics=ranking_metrics,
        ),
        **_coverage(paired),
    }


def _inner_ranking_metrics(
    cache: RankCacheProjection, center_scores: np.ndarray
) -> tuple[float, float]:
    _require(cache.labels is not None, "inner labels are unavailable")
    labels = cache.labels
    scores = center_scores[cache.center_indices]
    _require(
        labels.shape == scores.shape and np.isfinite(scores).all(),
        "inner ranking arrays are invalid",
    )
    if labels.any() and (~labels).any():
        return _ranking_metrics_one_sort(labels, scores)
    return float("nan"), float("nan")


def _inner_metric_rows(
    plan: Plan,
    nonouter_caches: Sequence[RankCacheProjection],
    outer_family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for inner_family in plan.family_order:
        if inner_family == outer_family:
            continue
        fit_caches = [
            cache
            for cache in nonouter_caches
            if cache.row.family != inner_family
        ]
        query_caches = [
            cache
            for cache in nonouter_caches
            if cache.row.family == inner_family
        ]
        _require(fit_caches and query_caches, "inner split produced an empty side")
        for weight in plan.weights:
            query_paired = [
                paired_center_ranks(cache, weight) for cache in query_caches
            ]
            control_arrays, control_fit_audits = fit_negative_control(
                fit_caches, weight=weight
            )
            audits.append(
                {
                    "outer_family": outer_family,
                    "inner_family": inner_family,
                    "fit_families": sorted(
                        {cache.row.family for cache in fit_caches}
                    ),
                    "arm": "negative_ecdf",
                    "weight": weight,
                    "uses_positive_templates": False,
                    "negative_reference_count_by_family": np.diff(
                        control_arrays[
                            "control_negative_rank_reference_offsets"
                        ]
                    ).tolist(),
                    "control_arrays_combined_sha256": canonical_json_sha256(
                        {
                            name: canonical_array_sha256(value)
                            for name, value in sorted(control_arrays.items())
                        }
                    ),
                    "fit_sources": control_fit_audits,
                }
            )
            control_raw_results = [
                query_negative_control_arrays(
                    control_arrays,
                    paired.paired_rank[paired.combined_valid],
                )
                for paired in query_paired
            ]
            for sigma in plan.sigmas:
                control_spatial = [
                    _spatial_center_score(
                        paired,
                        score,
                        sigma=sigma,
                        plan=plan,
                    )[0]
                    for paired, score in zip(
                        query_paired, control_raw_results, strict=True
                    )
                ]
                control_ranking = [
                    _inner_ranking_metrics(cache, score)
                    for cache, score in zip(
                        query_caches, control_spatial, strict=True
                    )
                ]
                for threshold in plan.thresholds:
                    selected_control = ControlSpec(weight, sigma, threshold)
                    payload = _control_payload(selected_control)
                    for cache, paired, score, ranking_metrics in zip(
                        query_caches,
                        query_paired,
                        control_spatial,
                        control_ranking,
                        strict=True,
                    ):
                        prediction = strict_absolute_threshold(
                            score,
                            paired.combined_valid,
                            threshold=threshold,
                        )
                        rows.append(
                            _inner_row(
                                outer_family=outer_family,
                                inner_family=inner_family,
                                cache=cache,
                                paired=paired,
                                payload=payload,
                                center_scores=score,
                                center_predictions=prediction,
                                ranking_metrics=ranking_metrics,
                            )
                        )
            for bin_count in plan.bin_counts:
                for beta in plan.betas:
                    model, fit_audits = fit_rank_model(
                        fit_caches,
                        weight=weight,
                        bin_count=bin_count,
                        beta=beta,
                    )
                    export = model.export_arrays()
                    audits.append(
                        {
                            "outer_family": outer_family,
                            "inner_family": inner_family,
                            "fit_families": sorted(
                                {cache.row.family for cache in fit_caches}
                            ),
                            "weight": weight,
                            "bin_count": bin_count,
                            "beta": beta,
                            "family_order": list(model.family_order),
                            "family_class_totals": model.family_class_totals.tolist(),
                            "dual_negative_reference_count_by_family": np.diff(
                                model.dual_negative_reference_offsets
                            ).tolist(),
                            "negative_rank_reference_count_by_family": np.diff(
                                model.negative_rank_reference_offsets
                            ).tolist(),
                            "model_arrays_combined_sha256": canonical_json_sha256(
                                {
                                    name: canonical_array_sha256(value)
                                    for name, value in sorted(export.items())
                                }
                            ),
                            "fit_sources": fit_audits,
                        }
                    )
                    query_results = [
                        model.query(paired.paired_rank[paired.combined_valid])
                        for paired in query_paired
                    ]
                    for sigma in plan.sigmas:
                        primary_spatial = [
                            _spatial_center_score(
                                paired,
                                result.dual_template_score,
                                sigma=sigma,
                                plan=plan,
                            )[0]
                            for paired, result in zip(
                                query_paired, query_results, strict=True
                            )
                        ]
                        primary_ranking = [
                            _inner_ranking_metrics(cache, score)
                            for cache, score in zip(
                                query_caches, primary_spatial, strict=True
                            )
                        ]
                        for threshold in plan.thresholds:
                            candidate = CandidateSpec(
                                weight, bin_count, beta, sigma, threshold
                            )
                            payload = _candidate_payload(candidate)
                            for cache, paired, score, ranking_metrics in zip(
                                query_caches,
                                query_paired,
                                primary_spatial,
                                primary_ranking,
                                strict=True,
                            ):
                                prediction = strict_absolute_threshold(
                                    score,
                                    paired.combined_valid,
                                    threshold=threshold,
                                )
                                rows.append(
                                    _inner_row(
                                        outer_family=outer_family,
                                        inner_family=inner_family,
                                        cache=cache,
                                        paired=paired,
                                        payload=payload,
                                        center_scores=score,
                                        center_predictions=prediction,
                                        ranking_metrics=ranking_metrics,
                                    )
                                )
                    del model, query_results
                    gc.collect()
    _require(rows and tuple(rows[0]) == INNER_METRIC_FIELDS, "inner metric fields drifted")
    expected_primary = {candidate.candidate_id for candidate in candidate_specs(plan)}
    expected_control = {candidate.candidate_id for candidate in control_specs(plan)}
    _require(
        {str(row["candidate_id"]) for row in rows if row["arm"] == "dual_histogram_llr"}
        == expected_primary,
        "inner primary candidate population drifted",
    )
    _require(
        {str(row["candidate_id"]) for row in rows if row["arm"] == "negative_ecdf"}
        == expected_control,
        "inner control candidate population drifted",
    )
    return rows, audits


def _hierarchical_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    family_values = []
    for family in sorted({str(row["inner_family"]) for row in rows}):
        values = np.asarray(
            [float(row[field]) for row in rows if row["inner_family"] == family],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        family_values.append(float(np.mean(finite)) if len(finite) else float("nan"))
    selected = np.asarray(family_values, dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    return float(np.mean(selected)) if len(selected) else float("nan")


def _summarize_candidates(
    candidates: Mapping[str, CandidateSpec | ControlSpec],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], CandidateSpec | ControlSpec, dict[str, Any]]:
    grouped = {candidate_id: [] for candidate_id in candidates}
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    expected_keys: set[tuple[str, str, int]] | None = None
    summaries: list[dict[str, Any]] = []
    for candidate_id in sorted(candidates):
        candidate_rows = grouped[candidate_id]
        keys = {
            (str(row["inner_family"]), str(row["dataset"]), int(row["source_ordinal"]))
            for row in candidate_rows
        }
        _require(len(keys) == len(candidate_rows), f"{candidate_id}: duplicate inner group")
        if expected_keys is None:
            expected_keys = keys
        _require(
            keys == expected_keys and len({key[0] for key in keys}) == 4,
            f"{candidate_id}: incomplete inner group set",
        )
        candidate = candidates[candidate_id]
        summary = (
            _candidate_payload(candidate)
            if isinstance(candidate, CandidateSpec)
            else _control_payload(candidate)
        )
        summary["inner_family_count"] = 4
        summary["group_count"] = len(candidate_rows)
        for field in (
            "accuracy",
            "average_precision",
            "f1",
            "balanced_accuracy",
            "auroc",
            "precision",
            "recall",
            "unique_center_combined_coverage",
        ):
            summary[field] = _hierarchical_mean(candidate_rows, field)
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


def aggregate_and_select_inner(
    plan: Plan,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    CandidateSpec,
    dict[str, Any],
    ControlSpec,
    dict[str, Any],
]:
    primary_map = {candidate.candidate_id: candidate for candidate in candidate_specs(plan)}
    control_map = {candidate.candidate_id: candidate for candidate in control_specs(plan)}
    primary_summaries, primary, primary_summary = _summarize_candidates(
        primary_map,
        [row for row in rows if row["arm"] == "dual_histogram_llr"],
    )
    control_summaries, control, control_summary = _summarize_candidates(
        control_map,
        [row for row in rows if row["arm"] == "negative_ecdf"],
    )
    _require(isinstance(primary, CandidateSpec), "primary selection type drifted")
    _require(isinstance(control, ControlSpec), "control selection type drifted")
    summaries = primary_summaries + control_summaries
    _require(tuple(summaries[0]) == INNER_SUMMARY_FIELDS, "inner summary fields drifted")
    return summaries, primary, primary_summary, control, control_summary


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
    ControlSpec,
    dict[str, Any],
    Mapping[str, tuple[Path, str]],
]:
    metrics_path = destination / "inner_group_metrics.csv"
    metrics_sha = source_runner.early._atomic_csv(
        metrics_path, INNER_METRIC_FIELDS, rows
    )
    parsed_rows = _parse_inner_metric_csv(
        metrics_path,
        expected_sha256=metrics_sha,
        plan=plan,
        outer_family=outer_family,
    )
    summaries, primary, primary_summary, control, control_summary = (
        aggregate_and_select_inner(plan, parsed_rows)
    )
    summary_path = destination / "inner_candidate_summary.csv"
    summary_sha = source_runner.early._atomic_csv(
        summary_path, INNER_SUMMARY_FIELDS, summaries
    )
    audit_path = destination / "inner_fit_audits.json"
    audit_sha = source_runner.early._atomic_json(
        audit_path,
        _manifest(
            {
                "schema": INNER_AUDIT_SCHEMA,
                "experiment": EXPERIMENT,
                "outer_family": outer_family,
                "fit_count": len(audits),
                "fits": list(audits),
                "fmt_features_opened": False,
                "raw_features_opened": False,
            }
        ),
    )
    # Read every persisted byte back before it may select a final library.
    replay = aggregate_and_select_inner(plan, parsed_rows)
    _require(
        replay[1:] == (primary, primary_summary, control, control_summary),
        "persisted inner selection differs from memory selection",
    )
    _authenticate_summary_csv(
        summary_path, expected_sha256=summary_sha, expected=summaries
    )
    audit_value = source_runner._read_self_hashed_json(
        audit_path, expected_file_sha256=audit_sha
    )
    _require(
        audit_value.get("schema") == INNER_AUDIT_SCHEMA
        and audit_value.get("outer_family") == outer_family
        and int(audit_value.get("fit_count", -1)) == len(audits),
        "inner audit authentication drifted",
    )
    return (
        primary,
        primary_summary,
        control,
        control_summary,
        MappingProxyType(
            {
                "inner_group_metrics": (metrics_path, metrics_sha),
                "inner_candidate_summary": (summary_path, summary_sha),
                "inner_fit_audits": (audit_path, audit_sha),
            }
        ),
    )


_INNER_STRING_FIELDS = frozenset(
    {
        "outer_family",
        "inner_family",
        "dataset",
        "arm",
        "candidate_id",
        "decision_rule",
    }
)
_INNER_INTEGER_FIELDS = frozenset(
    {
        "source_ordinal",
        "source_index",
        "bin_count",
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
    payload = source_runner.early._read_authenticated_bytes(
        path, expected_sha256=expected_sha256
    )
    with io.StringIO(payload.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == INNER_METRIC_FIELDS, "inner CSV fields drifted")
        raw_rows = list(reader)
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = {}
        for name in INNER_METRIC_FIELDS:
            text_value = raw[name]
            if name in _INNER_STRING_FIELDS:
                row[name] = text_value
            elif name in _INNER_INTEGER_FIELDS:
                row[name] = None if text_value == "" else int(text_value)
            else:
                row[name] = float("nan") if text_value == "" else float(text_value)
        rows.append(row)
    _require(
        rows and all(row["outer_family"] == outer_family for row in rows),
        "inner CSV outer family drifted",
    )
    expected_ids = {
        candidate.candidate_id for candidate in candidate_specs(plan)
    } | {candidate.candidate_id for candidate in control_specs(plan)}
    _require({str(row["candidate_id"]) for row in rows} == expected_ids, "inner CSV candidate scope drifted")
    return rows


def _authenticate_summary_csv(
    path: Path,
    *,
    expected_sha256: str,
    expected: Sequence[Mapping[str, Any]],
) -> None:
    payload = source_runner.early._read_authenticated_bytes(
        path, expected_sha256=expected_sha256
    )
    with io.StringIO(payload.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == INNER_SUMMARY_FIELDS, "inner summary fields drifted")
        rows = list(reader)
    _require(len(rows) == len(expected), "inner summary row count drifted")
    for index, (raw, value) in enumerate(zip(rows, expected, strict=True)):
        for field_name in INNER_SUMMARY_FIELDS:
            expected_text = str(source_runner.early._csv_value(value[field_name]))
            _require(
                raw[field_name] == expected_text,
                f"inner summary value drifted: row={index}/{field_name}",
            )


MODEL_ARRAY_NAMES = frozenset(
    {
        "serialization_version_int16",
        "bin_count_int64",
        "beta_float64",
        "family_order_unicode",
        "family_class_histogram_counts",
        "family_class_totals",
        "family_class_density",
        "full_class_density",
    }
)
CALIBRATION_ARRAY_NAMES = frozenset(
    {
        "dual_negative_reference_values",
        "dual_negative_reference_offsets",
        "negative_rank_reference_values",
        "negative_rank_reference_offsets",
    }
)
CONTROL_ARRAY_NAMES = frozenset(
    {
        "control_family_order_unicode",
        "control_negative_rank_reference_values",
        "control_negative_rank_reference_offsets",
        "control_weight_float64",
    }
)
FINAL_ARTIFACT_NAMES = MappingProxyType(
    {
        "model": (
            "final_rank_likelihood_model.npz",
            "final_rank_likelihood_model_manifest.json",
            MODEL_SCHEMA,
            MODEL_MANIFEST_SCHEMA,
            MODEL_ARRAY_NAMES,
        ),
        "calibration": (
            "final_rank_likelihood_calibration.npz",
            "final_rank_likelihood_calibration_manifest.json",
            CALIBRATION_SCHEMA,
            CALIBRATION_MANIFEST_SCHEMA,
            CALIBRATION_ARRAY_NAMES,
        ),
        "control": (
            "final_negative_ecdf_control.npz",
            "final_negative_ecdf_control_manifest.json",
            CONTROL_SCHEMA,
            CONTROL_MANIFEST_SCHEMA,
            CONTROL_ARRAY_NAMES,
        ),
    }
)


def _evidence_binding(plan: Plan, *, fit_families: Sequence[str]) -> dict[str, Any]:
    _require_bound(plan)
    return {
        "parent_binding": {
            "path": str(plan.parent_binding_path),
            "file_sha256": plan.parent_binding_file_sha256,
            "content_sha256": plan.parent_binding_content_sha256,
        },
        "binding_completion": {
            "path": str(plan.binding_completion_path),
            "file_sha256": plan.binding_completion_file_sha256,
        },
        "historical_source_centered_evidence": _json_safe(plan.source_evidence),
        "fit_families": list(fit_families),
        "config_sha256": plan.sha256,
        "fmt_features_opened": False,
        "raw_features_opened": False,
        "reference_labels_all_opened": False,
    }


def write_final_artifacts(
    destination: Path,
    model: FamilyBalancedRankLikelihoodModel,
    control_arrays: Mapping[str, np.ndarray],
    fit_audits: Sequence[Mapping[str, Any]],
    control_fit_audits: Sequence[Mapping[str, Any]],
    *,
    plan: Plan,
    primary: CandidateSpec,
    control: ControlSpec,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
) -> Mapping[str, tuple[Path, Path, str]]:
    exported = model.export_arrays()
    _require(
        set(exported) == MODEL_ARRAY_NAMES | CALIBRATION_ARRAY_NAMES,
        "core model serialization array set drifted",
    )
    _require(set(control_arrays) == CONTROL_ARRAY_NAMES, "control array set drifted")
    exported.update(control_arrays)
    result: dict[str, tuple[Path, Path, str]] = {}
    prior_manifest_hashes: dict[str, str] = {}
    for key in ("model", "calibration", "control"):
        artifact_name, manifest_name, artifact_schema, manifest_schema, names = (
            FINAL_ARTIFACT_NAMES[key]
        )
        arrays = {
            name: np.array(exported[name], copy=True, order="C")
            for name in sorted(names)
        }
        artifact_path = destination / artifact_name
        artifact_sha = source_runner.early._atomic_npz(artifact_path, arrays)
        manifest = _manifest(
            {
                "schema": manifest_schema,
                "artifact_schema": artifact_schema,
                "experiment": EXPERIMENT,
                "created_utc": _utc_now(),
                "git_commit": git_commit,
                "config_sha256": plan.sha256,
                "outer_family": outer_family,
                "primary_candidate": _candidate_payload(primary),
                "selected_control": _control_payload(control),
                "source_centered_evidence": _evidence_binding(
                    plan, fit_families=fit_families
                ),
                "fit_source_audits": (
                    list(control_fit_audits) if key == "control" else list(fit_audits)
                ),
                "artifact_file": {
                    "path": artifact_path.name,
                    "size_bytes": artifact_path.stat().st_size,
                    "sha256": artifact_sha,
                },
                "arrays": source_runner.early._array_manifest(arrays),
                "predecessor_manifest_file_sha256": dict(prior_manifest_hashes),
                "fit_labels_opened": True,
                "outer_sidecar_members_opened": False,
                "outer_labels_opened": False,
                "fmt_features_opened": False,
                "raw_features_opened": False,
                "reference_labels_all_opened": False,
            }
        )
        manifest_path = destination / manifest_name
        manifest_sha = source_runner.early._atomic_json(manifest_path, manifest)
        prior_manifest_hashes[key] = manifest_sha
        result[key] = (artifact_path, manifest_path, manifest_sha)
    return MappingProxyType(result)


def authenticate_final_artifacts(
    destination: Path,
    *,
    plan: Plan,
    primary: CandidateSpec,
    control: ControlSpec,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
    expected_manifest_sha256: Mapping[str, str],
) -> VerifiedLibrary:
    combined: dict[str, np.ndarray] = {}
    artifact_paths: dict[str, Path] = {}
    manifest_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    manifests_hashes: dict[str, str] = {}
    manifests: dict[str, Mapping[str, Any]] = {}
    prior_hashes: dict[str, str] = {}
    for key in ("model", "calibration", "control"):
        artifact_name, manifest_name, artifact_schema, manifest_schema, names = (
            FINAL_ARTIFACT_NAMES[key]
        )
        artifact_path = destination / artifact_name
        manifest_path = destination / manifest_name
        manifest, manifest_sha = _read_manifest(
            manifest_path, str(expected_manifest_sha256[key])
        )
        _require(
            manifest.get("schema") == manifest_schema
            and manifest.get("artifact_schema") == artifact_schema
            and manifest.get("experiment") == EXPERIMENT
            and manifest.get("git_commit") == git_commit
            and manifest.get("config_sha256") == plan.sha256
            and manifest.get("outer_family") == outer_family
            and manifest.get("primary_candidate")
            == _json_safe(_candidate_payload(primary))
            and manifest.get("selected_control")
            == _json_safe(_control_payload(control))
            and manifest.get("source_centered_evidence")
            == _json_safe(_evidence_binding(plan, fit_families=fit_families))
            and manifest.get("predecessor_manifest_file_sha256") == prior_hashes
            and manifest.get("outer_sidecar_members_opened") is False
            and manifest.get("outer_labels_opened") is False
            and manifest.get("fmt_features_opened") is False
            and manifest.get("raw_features_opened") is False
            and manifest.get("reference_labels_all_opened") is False,
            f"final {key} manifest provenance drifted",
        )
        records = manifest.get("arrays")
        _require(
            isinstance(records, Mapping) and set(records) == set(names),
            f"final {key} array manifest drifted",
        )
        arrays, artifact_sha = source_runner._verify_npz_arrays(
            artifact_path,
            file_record=manifest["artifact_file"],
            records=records,
        )
        _require(not (set(combined) & set(arrays)), "duplicate final model array")
        combined.update(arrays)
        artifact_paths[key] = artifact_path
        manifest_paths[key] = manifest_path
        artifact_hashes[key] = artifact_sha
        manifests_hashes[key] = manifest_sha
        manifests[key] = source_runner.early._deep_freeze(manifest)
        prior_hashes[key] = manifest_sha
    primary_arrays = {
        name: combined[name]
        for name in MODEL_ARRAY_NAMES | CALIBRATION_ARRAY_NAMES
    }
    model = FamilyBalancedRankLikelihoodModel.from_arrays(primary_arrays)
    _require(
        model.bin_count == primary.bin_count
        and model.beta == primary.beta
        and tuple(model.family_order) == tuple(sorted(fit_families)),
        "reconstructed final model identity drifted",
    )
    control_families = tuple(
        str(value)
        for value in np.asarray(combined["control_family_order_unicode"]).tolist()
    )
    control_values = np.asarray(
        combined["control_negative_rank_reference_values"], dtype=np.float64
    )
    control_offsets = np.asarray(
        combined["control_negative_rank_reference_offsets"], dtype=np.int64
    )
    control_weight = np.asarray(combined["control_weight_float64"])
    _require(
        control_families == tuple(sorted(fit_families))
        and control_weight.shape == ()
        and control_weight.dtype == np.dtype(np.float64)
        and float(control_weight) == control.weight
        and control_values.ndim == 1
        and len(control_values) > 0
        and np.isfinite(control_values).all()
        and np.all((control_values > 0.0) & (control_values < 1.0))
        and control_offsets.dtype == np.dtype(np.int64)
        and control_offsets.shape == (len(control_families) + 1,)
        and control_offsets[0] == 0
        and control_offsets[-1] == len(control_values)
        and np.all(np.diff(control_offsets) > 0),
        "negative-control family/weight/reference contract drifted",
    )
    for start, stop in zip(control_offsets[:-1], control_offsets[1:], strict=True):
        part = control_values[int(start) : int(stop)]
        _require(
            np.all(part[1:] >= part[:-1]),
            "negative-control family references are not sorted",
        )
    return VerifiedLibrary(
        MappingProxyType(artifact_paths),
        MappingProxyType(manifest_paths),
        MappingProxyType(artifact_hashes),
        MappingProxyType(manifests_hashes),
        MappingProxyType(manifests),
        source_runner.early._deep_freeze(combined),
        model,
        _AUTHENTICATION_SEAL,
    )


def query_negative_control(
    library: VerifiedLibrary,
    ranks: np.ndarray,
) -> np.ndarray:
    return query_negative_control_arrays(library.arrays, ranks)


def write_selected_candidate(
    destination: Path,
    *,
    plan: Plan,
    primary: CandidateSpec,
    primary_summary: Mapping[str, Any],
    control: ControlSpec,
    control_summary: Mapping[str, Any],
    library: VerifiedLibrary,
    inner_paths: Mapping[str, tuple[Path, str]],
    outer_family: str,
    git_commit: str,
) -> tuple[Path, str]:
    _require(library.seal is _AUTHENTICATION_SEAL, "selection requires authenticated final artifacts")
    value = _manifest(
        {
            "schema": SELECTED_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "git_commit": git_commit,
            "config_sha256": plan.sha256,
            "outer_family": outer_family,
            "primary_candidate": _candidate_payload(primary),
            "primary_inner_selection_summary": dict(primary_summary),
            "selected_control": _control_payload(control),
            "control_inner_selection_summary": dict(control_summary),
            "final_artifacts": {
                key: {
                    "artifact_path": library.artifact_paths[key].name,
                    "artifact_file_sha256": library.artifact_file_sha256[key],
                    "manifest_path": library.manifest_paths[key].name,
                    "manifest_file_sha256": library.manifest_file_sha256[key],
                    "manifest_content_sha256": library.manifests[key][
                        "content_sha256"
                    ],
                }
                for key in ("model", "calibration", "control")
            },
            "inner_evidence": {
                key: {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                }
                for key, (path, digest) in inner_paths.items()
            },
            "outer_results_visible_to_selection": False,
            "outer_sidecar_members_opened": False,
            "outer_labels_opened": False,
        }
    )
    path = destination / "selected_candidate.json"
    return path, source_runner.early._atomic_json(path, value)


def authenticate_selected_candidate(
    path: Path,
    *,
    plan: Plan,
    primary: CandidateSpec,
    primary_summary: Mapping[str, Any],
    control: ControlSpec,
    control_summary: Mapping[str, Any],
    library: VerifiedLibrary,
    inner_paths: Mapping[str, tuple[Path, str]],
    outer_family: str,
    git_commit: str,
    expected_file_sha256: str,
) -> VerifiedSelection:
    value, digest = _read_manifest(path, expected_file_sha256)
    _require(
        value.get("schema") == SELECTED_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("git_commit") == git_commit
        and value.get("config_sha256") == plan.sha256
        and value.get("outer_family") == outer_family
        and value.get("primary_candidate")
        == _json_safe(_candidate_payload(primary))
        and value.get("primary_inner_selection_summary")
        == _json_safe(primary_summary)
        and value.get("selected_control") == _json_safe(_control_payload(control))
        and value.get("control_inner_selection_summary")
        == _json_safe(control_summary)
        and value.get("outer_results_visible_to_selection") is False
        and value.get("outer_sidecar_members_opened") is False
        and value.get("outer_labels_opened") is False,
        "selected-candidate provenance drifted",
    )
    expected_artifacts = {
        key: {
            "artifact_path": library.artifact_paths[key].name,
            "artifact_file_sha256": library.artifact_file_sha256[key],
            "manifest_path": library.manifest_paths[key].name,
            "manifest_file_sha256": library.manifest_file_sha256[key],
            "manifest_content_sha256": library.manifests[key]["content_sha256"],
        }
        for key in ("model", "calibration", "control")
    }
    expected_inner = {
        key: {
            "path": item_path.name,
            "size_bytes": item_path.stat().st_size,
            "sha256": item_sha,
        }
        for key, (item_path, item_sha) in inner_paths.items()
    }
    _require(
        value.get("final_artifacts") == expected_artifacts
        and value.get("inner_evidence") == expected_inner,
        "selected-candidate artifact chain drifted",
    )
    return VerifiedSelection(
        path,
        digest,
        source_runner.early._deep_freeze(value),
        _AUTHENTICATION_SEAL,
    )


def _outer_rank_binding_sources(
    caches: Sequence[RankCacheProjection],
    *,
    weights: Sequence[float],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for cache in caches:
        assigned_rank = _assigned_ranks(cache)
        paired_hashes = {}
        combined_hash: str | None = None
        for weight in sorted(set(float(value) for value in weights)):
            paired = pair_assigned_center_ranks(
                cache.assigned_center_indices,
                cache.assigned_block_indices,
                assigned_rank,
                cache.center_indices,
                cache.block_indices,
                weight=weight,
                center_count=DEFAULT_CENTER_COUNT,
            )
            paired_hashes[format(weight, ".2f")] = canonical_array_sha256(
                paired.paired_rank
            )
            observed_combined = canonical_array_sha256(paired.combined_valid)
            if combined_hash is None:
                combined_hash = observed_combined
            _require(
                combined_hash == observed_combined,
                "combined-valid mask changed with rank weight",
            )
        sources.append(
            {
                "dataset": cache.row.dataset,
                "physical_family": cache.row.family,
                "source_ordinal": cache.row.source_ordinal,
                "source_index": cache.row.source_index,
                "parent_cache_path": str(cache.row.path),
                "parent_cache_file_sha256": cache.row.sha256,
                "parent_members_opened": list(cache.parent_members_opened),
                "sidecar_file_sha256": cache.sidecar_file_sha256,
                "sidecar_combined_array_sha256": cache.sidecar_combined_array_sha256,
                "sidecar_members_opened": list(cache.sidecar_members_opened),
                "assigned_rank_sha256": canonical_array_sha256(assigned_rank),
                "paired_rank_sha256_by_weight": paired_hashes,
                "combined_valid_sha256": combined_hash,
                "valid_projection_row_count": cache.count,
                "fmt_features_opened": False,
                "raw_features_opened": False,
                "valid_labels_opened": False,
                "reference_labels_all_opened": False,
            }
        )
    return sources


def write_outer_rank_binding(
    destination: Path,
    caches: Sequence[RankCacheProjection],
    *,
    plan: Plan,
    primary: CandidateSpec,
    control: ControlSpec,
    library: VerifiedLibrary,
    selection: VerifiedSelection,
    outer_family: str,
    git_commit: str,
) -> tuple[Path, str]:
    _require(
        library.seal is selection.seal is _AUTHENTICATION_SEAL,
        "outer rank binding requires closed selection and final artifacts",
    )
    sources = _outer_rank_binding_sources(
        caches,
        weights=(primary.weight, control.weight, DIRECT_WEIGHT),
    )
    value = _manifest(
        {
            "schema": OUTER_FEATURE_BINDING_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "git_commit": git_commit,
            "config_sha256": plan.sha256,
            "outer_family": outer_family,
            "primary_candidate": _candidate_payload(primary),
            "selected_control": _control_payload(control),
            "selected_candidate_file_sha256": selection.file_sha256,
            "final_artifact_manifest_file_sha256": dict(
                library.manifest_file_sha256
            ),
            "source_count": len(sources),
            "sources": sources,
            "phase": "after_final_fit_and_both_selections_before_prediction",
            "outer_labels_opened": False,
            "fmt_features_opened": False,
            "raw_features_opened": False,
            "reference_labels_all_opened": False,
        }
    )
    path = destination / "outer_rank_binding.json"
    return path, source_runner.early._atomic_json(path, value)


def authenticate_outer_rank_binding(
    path: Path,
    caches: Sequence[RankCacheProjection],
    *,
    plan: Plan,
    primary: CandidateSpec,
    control: ControlSpec,
    library: VerifiedLibrary,
    selection: VerifiedSelection,
    outer_family: str,
    git_commit: str,
    expected_file_sha256: str,
) -> VerifiedFeatureBinding:
    value, digest = _read_manifest(path, expected_file_sha256)
    expected_sources = _outer_rank_binding_sources(
        caches,
        weights=(primary.weight, control.weight, DIRECT_WEIGHT),
    )
    _require(
        value.get("schema") == OUTER_FEATURE_BINDING_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("git_commit") == git_commit
        and value.get("config_sha256") == plan.sha256
        and value.get("outer_family") == outer_family
        and value.get("primary_candidate")
        == _json_safe(_candidate_payload(primary))
        and value.get("selected_control") == _json_safe(_control_payload(control))
        and value.get("selected_candidate_file_sha256") == selection.file_sha256
        and value.get("final_artifact_manifest_file_sha256")
        == dict(library.manifest_file_sha256)
        and int(value.get("source_count", -1)) == len(expected_sources)
        and value.get("sources") == _json_safe(expected_sources)
        and value.get("outer_labels_opened") is False
        and value.get("fmt_features_opened") is False
        and value.get("raw_features_opened") is False
        and value.get("reference_labels_all_opened") is False,
        "outer rank binding fresh replay drifted",
    )
    return VerifiedFeatureBinding(
        path,
        digest,
        source_runner.early._deep_freeze(value),
        _AUTHENTICATION_SEAL,
    )


UNIQUE_PREDICTION_DTYPES: Mapping[str, np.dtype[Any]] = MappingProxyType(
    {
        "unique_dataset": np.dtype("<U64"),
        "unique_source_ordinal": np.dtype(np.int16),
        "unique_source_index": np.dtype(np.int64),
        "unique_center_seed_index": np.dtype(np.int64),
        "unique_legacy_valid": np.dtype(np.bool_),
        "unique_expanded_valid": np.dtype(np.bool_),
        "unique_combined_valid": np.dtype(np.bool_),
        "unique_primary_paired_rank": np.dtype(np.float64),
        "unique_primary_log_likelihood_ratio": np.dtype(np.float64),
        "unique_primary_raw_score": np.dtype(np.float64),
        "unique_primary_spatial_score": np.dtype(np.float64),
        "unique_primary_spatial_denominator": np.dtype(np.float64),
        "unique_primary_prediction": np.dtype(np.bool_),
        "unique_control_paired_rank": np.dtype(np.float64),
        "unique_control_raw_score": np.dtype(np.float64),
        "unique_control_spatial_score": np.dtype(np.float64),
        "unique_control_spatial_denominator": np.dtype(np.float64),
        "unique_control_prediction": np.dtype(np.bool_),
        "unique_direct_rank_mean_score": np.dtype(np.float64),
        "unique_direct_rank_mean_prediction": np.dtype(np.bool_),
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
        "valid_primary_score": np.dtype(np.float64),
        "valid_primary_prediction": np.dtype(np.bool_),
        "valid_control_score": np.dtype(np.float64),
        "valid_control_prediction": np.dtype(np.bool_),
        "valid_direct_rank_mean_score": np.dtype(np.float64),
        "valid_direct_rank_mean_prediction": np.dtype(np.bool_),
    }
)
PREDICTION_DTYPES: Mapping[str, np.dtype[Any]] = MappingProxyType(
    {**UNIQUE_PREDICTION_DTYPES, **VALID_PREDICTION_DTYPES}
)


def build_outer_prediction_arrays(
    caches: Sequence[RankCacheProjection],
    library: VerifiedLibrary,
    primary: CandidateSpec,
    control: ControlSpec,
    plan: Plan,
    binding: VerifiedFeatureBinding,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    _require(
        library.seal is binding.seal is _AUTHENTICATION_SEAL,
        "outer prediction requires authenticated rank/library artifacts",
    )
    _require(
        caches
        and all(cache.labels is None for cache in caches)
        and all("valid_labels" not in cache.parent_members_opened for cache in caches),
        "outer feature caches must remain label-free",
    )
    parts: dict[str, list[np.ndarray]] = {name: [] for name in PREDICTION_DTYPES}
    audits: list[dict[str, Any]] = []
    for cache in caches:
        primary_paired = paired_center_ranks(cache, primary.weight)
        control_paired = paired_center_ranks(cache, control.weight)
        direct_paired = paired_center_ranks(cache, DIRECT_WEIGHT)
        _require(
            np.array_equal(
                primary_paired.combined_valid, control_paired.combined_valid
            )
            and np.array_equal(
                primary_paired.combined_valid, direct_paired.combined_valid
            ),
            "combined-valid population changed across arms",
        )
        selected = primary_paired.combined_valid
        query = library.model.query(primary_paired.paired_rank[selected])
        primary_score, primary_denominator = _spatial_center_score(
            primary_paired,
            query.dual_template_score,
            sigma=primary.sigma,
            plan=plan,
        )
        primary_prediction = strict_absolute_threshold(
            primary_score,
            selected,
            threshold=primary.threshold,
        )
        control_raw = query_negative_control(
            library, control_paired.paired_rank[selected]
        )
        control_score, control_denominator = _spatial_center_score(
            control_paired,
            control_raw,
            sigma=control.sigma,
            plan=plan,
        )
        control_prediction = strict_absolute_threshold(
            control_score,
            selected,
            threshold=control.threshold,
        )
        direct_score = np.ascontiguousarray(
            direct_paired.paired_rank, dtype=np.float64
        )
        direct_prediction = fixed_top_fraction_over_centers(
            direct_score,
            np.ones(DEFAULT_CENTER_COUNT, dtype=np.bool_),
            fraction=DIRECT_TOP_FRACTION,
            require_strictly_positive_score=False,
        )
        raw_llr = np.zeros(DEFAULT_CENTER_COUNT, dtype=np.float64)
        raw_dual = np.zeros(DEFAULT_CENTER_COUNT, dtype=np.float64)
        raw_control = np.zeros(DEFAULT_CENTER_COUNT, dtype=np.float64)
        raw_llr[selected] = query.log_likelihood_ratio
        raw_dual[selected] = query.dual_template_score
        raw_control[selected] = control_raw
        unique_count = DEFAULT_CENTER_COUNT
        valid_count = cache.count
        unique_values: dict[str, np.ndarray] = {
            "unique_dataset": np.full(unique_count, cache.row.dataset, dtype="<U64"),
            "unique_source_ordinal": np.full(unique_count, cache.row.source_ordinal, dtype=np.int16),
            "unique_source_index": np.full(unique_count, cache.row.source_index, dtype=np.int64),
            "unique_center_seed_index": primary_paired.center_seed_index,
            "unique_legacy_valid": primary_paired.legacy_valid,
            "unique_expanded_valid": primary_paired.expanded_valid,
            "unique_combined_valid": selected,
            "unique_primary_paired_rank": primary_paired.paired_rank,
            "unique_primary_log_likelihood_ratio": raw_llr,
            "unique_primary_raw_score": raw_dual,
            "unique_primary_spatial_score": primary_score,
            "unique_primary_spatial_denominator": primary_denominator,
            "unique_primary_prediction": primary_prediction,
            "unique_control_paired_rank": control_paired.paired_rank,
            "unique_control_raw_score": raw_control,
            "unique_control_spatial_score": control_score,
            "unique_control_spatial_denominator": control_denominator,
            "unique_control_prediction": control_prediction,
            "unique_direct_rank_mean_score": direct_score,
            "unique_direct_rank_mean_prediction": direct_prediction,
        }
        centers = cache.center_indices
        valid_values: dict[str, np.ndarray] = {
            "valid_dataset": np.full(valid_count, cache.row.dataset, dtype="<U64"),
            "valid_source_ordinal": np.full(valid_count, cache.row.source_ordinal, dtype=np.int16),
            "valid_source_index": np.full(valid_count, cache.row.source_index, dtype=np.int64),
            "valid_scale_id": cache.scale_ids,
            "valid_center_seed_index": centers,
            "valid_scale_block_index": cache.block_indices,
            "valid_assigned_row_index": cache.assigned_row_indices,
            "valid_primary_score": primary_score[centers],
            "valid_primary_prediction": primary_prediction[centers],
            "valid_control_score": control_score[centers],
            "valid_control_prediction": control_prediction[centers],
            "valid_direct_rank_mean_score": direct_score[centers],
            "valid_direct_rank_mean_prediction": direct_prediction[centers],
        }
        for name, value in {**unique_values, **valid_values}.items():
            parts[name].append(
                np.ascontiguousarray(value, dtype=PREDICTION_DTYPES[name])
            )
        coverage = _coverage(primary_paired)
        audits.append(
            {
                "dataset": cache.row.dataset,
                "source_ordinal": cache.row.source_ordinal,
                "source_index": cache.row.source_index,
                "valid_row_count": valid_count,
                "unique_center_count": unique_count,
                **coverage,
                "primary_prediction_count": int(primary_prediction.sum()),
                "control_prediction_count": int(control_prediction.sum()),
                "direct_prediction_count": int(direct_prediction.sum()),
                "sidecar_file_sha256": cache.sidecar_file_sha256,
                "sidecar_combined_array_sha256": cache.sidecar_combined_array_sha256,
                "parent_members_opened": list(cache.parent_members_opened),
                "sidecar_members_opened": list(cache.sidecar_members_opened),
                "valid_labels_opened": False,
                "fmt_features_opened": False,
                "raw_features_opened": False,
                "reference_labels_all_opened": False,
            }
        )
    arrays = {
        name: np.ascontiguousarray(np.concatenate(values), dtype=dtype)
        for name, dtype in PREDICTION_DTYPES.items()
        for values in (parts[name],)
    }
    return arrays, audits


def write_outer_prediction(
    destination: Path,
    arrays: Mapping[str, np.ndarray],
    audits: Sequence[Mapping[str, Any]],
    *,
    plan: Plan,
    primary: CandidateSpec,
    control: ControlSpec,
    library: VerifiedLibrary,
    selection: VerifiedSelection,
    binding: VerifiedFeatureBinding,
    outer_family: str,
    git_commit: str,
) -> tuple[Path, Path, str]:
    _require(set(arrays) == set(PREDICTION_DTYPES), "prediction member set drifted")
    artifact_path = destination / "outer_predictions.npz"
    artifact_sha = source_runner.early._atomic_npz(artifact_path, arrays)
    manifest = _manifest(
        {
            "schema": PREDICTION_MANIFEST_SCHEMA,
            "prediction_schema": PREDICTION_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "git_commit": git_commit,
            "config_sha256": plan.sha256,
            "outer_family": outer_family,
            "primary_candidate": _candidate_payload(primary),
            "selected_control": _control_payload(control),
            "selected_candidate_file_sha256": selection.file_sha256,
            "final_artifact_manifest_file_sha256": dict(
                library.manifest_file_sha256
            ),
            "outer_rank_binding_file_sha256": binding.file_sha256,
            "prediction_file": {
                "path": artifact_path.name,
                "size_bytes": artifact_path.stat().st_size,
                "sha256": artifact_sha,
            },
            "unique_row_count": len(arrays["unique_center_seed_index"]),
            "valid_row_count": len(arrays["valid_assigned_row_index"]),
            "arrays": source_runner.early._array_manifest(arrays),
            "group_audits": list(audits),
            "outer_labels_opened": False,
            "parent_control_prediction_opened": False,
            "fmt_features_opened": False,
            "raw_features_opened": False,
            "reference_labels_all_opened": False,
        }
    )
    manifest_path = destination / "outer_prediction_manifest.json"
    manifest_sha = source_runner.early._atomic_json(manifest_path, manifest)
    return artifact_path, manifest_path, manifest_sha


def authenticate_outer_prediction(
    artifact_path: Path,
    manifest_path: Path,
    *,
    plan: Plan,
    primary: CandidateSpec,
    control: ControlSpec,
    library: VerifiedLibrary,
    selection: VerifiedSelection,
    binding: VerifiedFeatureBinding,
    outer_caches: Sequence[RankCacheProjection],
    outer_family: str,
    git_commit: str,
    expected_manifest_sha256: str,
) -> VerifiedPrediction:
    value, manifest_sha = _read_manifest(
        manifest_path, expected_manifest_sha256
    )
    _require(
        value.get("schema") == PREDICTION_MANIFEST_SCHEMA
        and value.get("prediction_schema") == PREDICTION_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("git_commit") == git_commit
        and value.get("config_sha256") == plan.sha256
        and value.get("outer_family") == outer_family
        and value.get("primary_candidate")
        == _json_safe(_candidate_payload(primary))
        and value.get("selected_control") == _json_safe(_control_payload(control))
        and value.get("selected_candidate_file_sha256") == selection.file_sha256
        and value.get("final_artifact_manifest_file_sha256")
        == dict(library.manifest_file_sha256)
        and value.get("outer_rank_binding_file_sha256") == binding.file_sha256
        and value.get("outer_labels_opened") is False
        and value.get("parent_control_prediction_opened") is False
        and value.get("fmt_features_opened") is False
        and value.get("raw_features_opened") is False
        and value.get("reference_labels_all_opened") is False,
        "outer prediction provenance drifted",
    )
    records = value.get("arrays")
    _require(
        isinstance(records, Mapping) and set(records) == set(PREDICTION_DTYPES),
        "prediction array records drifted",
    )
    arrays, artifact_sha = source_runner._verify_npz_arrays(
        artifact_path,
        file_record=value["prediction_file"],
        records=records,
    )
    unique_count = len(outer_caches) * DEFAULT_CENTER_COUNT
    valid_count = sum(cache.count for cache in outer_caches)
    for name, dtype in UNIQUE_PREDICTION_DTYPES.items():
        _require(
            arrays[name].dtype == dtype and arrays[name].shape == (unique_count,),
            f"unique prediction array contract drifted: {name}",
        )
    for name, dtype in VALID_PREDICTION_DTYPES.items():
        _require(
            arrays[name].dtype == dtype and arrays[name].shape == (valid_count,),
            f"valid prediction array contract drifted: {name}",
        )
    expected_arrays, expected_audits = build_outer_prediction_arrays(
        outer_caches, library, primary, control, plan, binding
    )
    for name in sorted(arrays):
        _require(
            canonical_array_sha256(arrays[name])
            == canonical_array_sha256(expected_arrays[name]),
            f"prediction fresh replay drifted: {name}",
        )
    _require(
        value.get("group_audits") == _json_safe(expected_audits),
        "prediction group audit fresh replay drifted",
    )
    return VerifiedPrediction(
        artifact_path,
        manifest_path,
        artifact_sha,
        manifest_sha,
        source_runner.early._deep_freeze(value),
        source_runner.early._deep_freeze(arrays),
        _AUTHENTICATION_SEAL,
    )


def _authenticate_parent_aggregate(plan: Plan) -> Mapping[str, Any]:
    parent = _mapping(plan.raw, "direct_parent")
    aggregate_record = _mapping(parent, "authenticated_aggregate")
    directory = Path(str(aggregate_record["directory"])).resolve()
    completion_path = directory / "AGGREGATE_COMPLETE.json"
    completion = source_runner._read_self_hashed_json(
        completion_path,
        expected_file_sha256=str(aggregate_record["completion_file_sha256"]),
    )
    _require(
        completion.get("experiment") == SOURCE_EXPERIMENT
        and completion.get("config_sha256") == SOURCE_CONFIG_SHA256
        and completion.get("aggregator_git_commit") == SOURCE_NUMERICAL_COMMIT
        and completion.get("fold_git_commit") == SOURCE_NUMERICAL_COMMIT
        and completion.get("mode") == "complete_five_fold_aggregate",
        "direct-parent aggregate completion provenance drifted",
    )
    manifest = source_runner._read_self_hashed_json(
        directory / "aggregate_manifest.json",
        expected_file_sha256=str(aggregate_record["manifest_file_sha256"]),
    )
    report = source_runner._read_self_hashed_json(
        directory / "aggregate_summary.json",
        expected_file_sha256=str(aggregate_record["report_file_sha256"]),
    )
    _stable_file_identity(
        directory / "outer_family_summary.csv",
        (directory / "outer_family_summary.csv").stat().st_size,
        str(aggregate_record["table_file_sha256"]),
    )
    _require(
        manifest.get("experiment") == SOURCE_EXPERIMENT
        and manifest.get("config_sha256") == SOURCE_CONFIG_SHA256
        and manifest.get("report_file_sha256")
        == aggregate_record["report_file_sha256"]
        and manifest.get("outer_family_summary_file_sha256")
        == aggregate_record["table_file_sha256"]
        and report.get("experiment") == SOURCE_EXPERIMENT
        and report.get("config_sha256") == SOURCE_CONFIG_SHA256
        and report.get("outer_families") == list(plan.family_order)
        and len(manifest.get("source_folds", [])) == 5,
        "direct-parent aggregate manifest/report drifted",
    )
    return source_runner.early._deep_freeze(
        {
            "directory": str(directory),
            "completion_file_sha256": aggregate_record[
                "completion_file_sha256"
            ],
            "manifest_file_sha256": aggregate_record["manifest_file_sha256"],
            "report_file_sha256": aggregate_record["report_file_sha256"],
            "table_file_sha256": aggregate_record["table_file_sha256"],
            "source_folds": manifest["source_folds"],
        }
    )


def authenticate_parent_control(
    plan: Plan,
    outer_family: str,
) -> AuthenticatedParentControl:
    """Authenticate the stopped SourceCentered prediction without FMT reads."""

    aggregate = _authenticate_parent_aggregate(plan)
    run_directory = plan.parent_run_directories[outer_family]
    fold_records = [
        item
        for item in aggregate["source_folds"]
        if item.get("outer_family") == outer_family
    ]
    _require(len(fold_records) == 1, "direct-parent aggregate fold does not resolve uniquely")
    fold_record = fold_records[0]
    _require(
        Path(str(fold_record["run_directory"])).resolve() == run_directory,
        "direct-parent fold directory differs from frozen config",
    )
    completion_path = run_directory / "RUN_COMPLETE.json"
    completion = source_runner._read_self_hashed_json(
        completion_path,
        expected_file_sha256=str(fold_record["completion_file_sha256"]),
    )
    result = source_runner._read_self_hashed_json(
        run_directory / "result_manifest.json",
        expected_file_sha256=str(fold_record["result_manifest_file_sha256"]),
    )
    _require(
        completion.get("schema") == source_runner.COMPLETE_SCHEMA
        and completion.get("experiment") == SOURCE_EXPERIMENT
        and completion.get("outer_family") == outer_family
        and completion.get("git_commit") == SOURCE_NUMERICAL_COMMIT
        and completion.get("config_sha256") == SOURCE_CONFIG_SHA256
        and result.get("schema") == source_runner.RESULT_SCHEMA
        and result.get("experiment") == SOURCE_EXPERIMENT
        and result.get("outer_family") == outer_family
        and result.get("git_commit") == SOURCE_NUMERICAL_COMMIT
        and result.get("config_sha256") == SOURCE_CONFIG_SHA256
        and result.get("content_sha256")
        == completion.get("result_manifest_content_sha256"),
        "direct-parent fold provenance drifted",
    )
    artifacts = result.get("artifacts")
    _require(isinstance(artifacts, Mapping), "direct-parent artifact map is missing")
    prediction_path = run_directory / "outer_predictions.npz"
    prediction_manifest_path = run_directory / "outer_prediction_manifest.json"
    for name, direct_field in (
        ("outer_predictions.npz", "prediction_file_sha256"),
        ("outer_prediction_manifest.json", "prediction_manifest_file_sha256"),
    ):
        record = artifacts.get(name)
        _require(isinstance(record, Mapping), f"direct-parent artifact missing: {name}")
        _stable_file_identity(
            run_directory / name,
            int(record["size_bytes"]),
            str(record["sha256"]),
        )
        _require(
            record["sha256"] == result[direct_field],
            f"direct-parent direct hash drifted: {name}",
        )
    prediction_manifest = source_runner._read_self_hashed_json(
        prediction_manifest_path,
        expected_file_sha256=str(result["prediction_manifest_file_sha256"]),
    )
    _require(
        prediction_manifest.get("schema")
        == source_runner.PREDICTION_MANIFEST_SCHEMA
        and prediction_manifest.get("experiment") == SOURCE_EXPERIMENT
        and prediction_manifest.get("outer_family") == outer_family
        and prediction_manifest.get("git_commit") == SOURCE_NUMERICAL_COMMIT
        and prediction_manifest.get("config_sha256") == SOURCE_CONFIG_SHA256
        and prediction_manifest.get("valid_labels_opened") is False
        and prediction_manifest.get("reference_labels_all_opened") is False,
        "direct-parent prediction manifest provenance drifted",
    )
    records = prediction_manifest.get("arrays")
    _require(
        isinstance(records, Mapping)
        and set(records) == set(source_runner.PREDICTION_DTYPES),
        "direct-parent prediction array records drifted",
    )
    arrays, prediction_sha = source_runner._verify_npz_arrays(
        prediction_path,
        file_record=prediction_manifest["prediction_file"],
        records=records,
    )
    row_count = len(arrays["valid_assigned_row_index"])
    for name, dtype in source_runner.PREDICTION_DTYPES.items():
        expected_count = (
            len(arrays["unique_center_seed_index"])
            if name in source_runner.UNIQUE_PREDICTION_DTYPES
            else row_count
        )
        _require(
            arrays[name].dtype == dtype
            and arrays[name].shape == (expected_count,),
            f"direct-parent prediction shape/dtype drifted: {name}",
        )
    evidence = {
        "status": "authenticated_stopped_parent_control_only",
        "run_directory": str(run_directory),
        "outer_family": outer_family,
        "parent_git_commit": SOURCE_NUMERICAL_COMMIT,
        "parent_config_sha256": SOURCE_CONFIG_SHA256,
        "aggregate_job_id": SOURCE_AGGREGATE_JOB_ID,
        "aggregate_completion_file_sha256": aggregate[
            "completion_file_sha256"
        ],
        "completion_file_sha256": fold_record["completion_file_sha256"],
        "result_manifest_file_sha256": fold_record["result_manifest_file_sha256"],
        "prediction_manifest_file_sha256": result[
            "prediction_manifest_file_sha256"
        ],
        "prediction_file_sha256": prediction_sha,
        "valid_row_count": row_count,
        "fresh_numerical_replay_in_child": False,
        "parent_fmt_or_raw_members_opened_in_child": False,
    }
    return AuthenticatedParentControl(
        run_directory,
        outer_family,
        source_runner.early._deep_freeze(arrays),
        source_runner.early._deep_freeze(evidence),
    )


OUTER_METRIC_FIELDS = (
    "outer_family",
    "dataset",
    "source_ordinal",
    "source_index",
    "arm",
    "population",
    "template_success_eligible",
    "primary_candidate_id",
    "control_candidate_id",
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


def load_outer_references_after_prediction(
    plan: Plan,
    verified: VerifiedPrediction,
    *,
    outer_family: str,
) -> tuple[
    Mapping[tuple[str, int, int], np.ndarray],
    list[dict[str, Any]],
]:
    _require(
        verified.seal is _AUTHENTICATION_SEAL,
        "outer references require a freshly authenticated prediction",
    )
    cache_rows, _identity = load_cache_rows(plan.source_plan.parent_plan)
    outer_rows = [row for row in cache_rows if row.family == outer_family]
    references: dict[tuple[str, int, int], np.ndarray] = {}
    audits: list[dict[str, Any]] = []
    arrays = verified.arrays
    for row in outer_rows:
        parent, members = _load_parent_minimal(row, include_labels=True)
        selected = (
            (arrays["valid_dataset"] == row.dataset)
            & (arrays["valid_source_ordinal"] == row.source_ordinal)
            & (arrays["valid_source_index"] == row.source_index)
        )
        _require(
            int(selected.sum()) == len(parent["valid_labels"])
            and np.array_equal(
                arrays["valid_assigned_row_index"][selected],
                parent["valid_assigned_row_index"],
            )
            and np.array_equal(
                arrays["valid_scale_id"][selected], parent["valid_scale_id"]
            ),
            "outer reference/prediction identity drifted",
        )
        key = (row.dataset, row.source_ordinal, row.source_index)
        labels = np.ascontiguousarray(parent["valid_labels"], dtype=np.bool_)
        labels.setflags(write=False)
        references[key] = labels
        audits.append(
            {
                "dataset": row.dataset,
                "source_ordinal": row.source_ordinal,
                "source_index": row.source_index,
                "cache_path": str(row.path),
                "cache_file_sha256": row.sha256,
                "members_opened": list(members),
                "fmt_features_opened": False,
                "raw_features_opened": False,
                "reference_labels_all_opened": False,
                "opened_after_prediction_fresh_authentication": True,
                "prediction_manifest_file_sha256": verified.manifest_file_sha256,
                "prediction_file_sha256": verified.artifact_file_sha256,
            }
        )
    return MappingProxyType(references), audits


def _metric_row(
    *,
    outer_family: str,
    dataset: str,
    source_ordinal: int,
    source_index: int,
    arm: str,
    population: str,
    eligible: bool,
    primary: CandidateSpec,
    control: ControlSpec,
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "outer_family": outer_family,
        "dataset": dataset,
        "source_ordinal": source_ordinal,
        "source_index": source_index,
        "arm": arm,
        "population": population,
        "template_success_eligible": eligible,
        "primary_candidate_id": primary.candidate_id,
        "control_candidate_id": control.candidate_id,
        **_classification_metric_values(labels, scores, predictions),
        **dict(coverage),
    }


def evaluate_outer_prediction(
    plan: Plan,
    primary: CandidateSpec,
    control: ControlSpec,
    verified: VerifiedPrediction,
    parent_control: AuthenticatedParentControl,
    *,
    outer_family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references, audits = load_outer_references_after_prediction(
        plan, verified, outer_family=outer_family
    )
    arrays = verified.arrays
    parent = parent_control.arrays
    rows: list[dict[str, Any]] = []
    for key in sorted(references):
        dataset, source_ordinal, source_index = key
        labels = references[key]
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
        parent_valid = (
            (parent["valid_dataset"] == dataset)
            & (parent["valid_source_ordinal"] == source_ordinal)
            & (parent["valid_source_index"] == source_index)
        )
        _require(
            int(valid.sum()) == len(labels)
            and int(unique.sum()) == DEFAULT_CENTER_COUNT
            and int(parent_valid.sum()) == len(labels)
            and np.array_equal(
                arrays["valid_assigned_row_index"][valid],
                parent["valid_assigned_row_index"][parent_valid],
            )
            and np.array_equal(
                arrays["valid_center_seed_index"][valid],
                parent["valid_center_seed_index"][parent_valid],
            )
            and np.array_equal(
                arrays["valid_scale_block_index"][valid],
                parent["valid_scale_block_index"][parent_valid],
            )
            and np.array_equal(
                arrays["valid_scale_id"][valid],
                parent["valid_scale_id"][parent_valid],
            ),
            "new/parent exact valid-row identity join drifted",
        )
        legacy = np.asarray(arrays["unique_legacy_valid"][unique], dtype=np.bool_)
        expanded = np.asarray(arrays["unique_expanded_valid"][unique], dtype=np.bool_)
        combined = legacy | expanded
        coverage = {
            "unique_center_combined_coverage": float(combined.mean()),
            "both_valid_count": int((legacy & expanded).sum()),
            "legacy_only_count": int((legacy & ~expanded).sum()),
            "expanded_only_count": int((~legacy & expanded).sum()),
            "neither_valid_count": int((~legacy & ~expanded).sum()),
        }
        arms = (
            (
                "dual_histogram_llr",
                True,
                arrays["valid_primary_score"][valid],
                arrays["valid_primary_prediction"][valid],
            ),
            (
                "parent_source_centered_paired_scale",
                False,
                parent["valid_paired_score"][parent_valid],
                parent["valid_paired_prediction"][parent_valid],
            ),
            (
                "negative_ecdf",
                False,
                arrays["valid_control_score"][valid],
                arrays["valid_control_prediction"][valid],
            ),
            (
                "direct_rank_mean_top5",
                False,
                arrays["valid_direct_rank_mean_score"][valid],
                arrays["valid_direct_rank_mean_prediction"][valid],
            ),
        )
        for arm, success_eligible, score, prediction in arms:
            rows.append(
                _metric_row(
                    outer_family=outer_family,
                    dataset=dataset,
                    source_ordinal=source_ordinal,
                    source_index=source_index,
                    arm=arm,
                    population="all_parent_valid_rows",
                    eligible=success_eligible,
                    primary=primary,
                    control=control,
                    labels=labels,
                    scores=score,
                    predictions=prediction,
                    coverage=coverage,
                )
            )
        # Combined-valid center labels are reconstructed only now from valid
        # labels; reference_labels_all remains unopened.
        center_labels = np.zeros(DEFAULT_CENTER_COUNT, dtype=np.bool_)
        seen = np.zeros(DEFAULT_CENTER_COUNT, dtype=np.bool_)
        valid_centers = arrays["valid_center_seed_index"][valid]
        for center, label in zip(valid_centers, labels, strict=True):
            index = int(center)
            if seen[index]:
                _require(center_labels[index] == bool(label), "outer center labels disagree")
            else:
                center_labels[index] = bool(label)
                seen[index] = True
        _require(np.array_equal(seen, combined), "outer combined-valid label mask drifted")
        rows.append(
            _metric_row(
                outer_family=outer_family,
                dataset=dataset,
                source_ordinal=source_ordinal,
                source_index=source_index,
                arm="dual_histogram_llr",
                population="combined_valid_unique_centers",
                eligible=False,
                primary=primary,
                control=control,
                labels=center_labels[combined],
                scores=arrays["unique_primary_spatial_score"][unique][combined],
                predictions=arrays["unique_primary_prediction"][unique][combined],
                coverage=coverage,
            )
        )
    _require(rows and tuple(rows[0]) == OUTER_METRIC_FIELDS, "outer metric fields drifted")
    return rows, audits


def _mean_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(rows, "metric summary has no groups")
    result: dict[str, Any] = {"group_count": len(rows)}
    for field in CLASSIFICATION_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[field] = float(np.mean(finite)) if len(finite) else float("nan")
    for field in CLASSIFICATION_COUNT_FIELDS:
        result[field] = int(sum(int(row[field]) for row in rows))
    return result


def outer_summary(rows: Sequence[Mapping[str, Any]], outer_family: str) -> dict[str, Any]:
    def select(arm: str, population: str) -> list[Mapping[str, Any]]:
        return [
            row
            for row in rows
            if row["arm"] == arm and row["population"] == population
        ]

    primary = select("dual_histogram_llr", "all_parent_valid_rows")
    parent = select(
        "parent_source_centered_paired_scale", "all_parent_valid_rows"
    )
    control = select("negative_ecdf", "all_parent_valid_rows")
    direct = select("direct_rank_mean_top5", "all_parent_valid_rows")
    unique = select("dual_histogram_llr", "combined_valid_unique_centers")
    _require(primary and parent and control and direct and unique, "outer summary arm missing")
    coverage = float(
        np.mean(
            [float(row["unique_center_combined_coverage"]) for row in primary]
        )
    )
    return {
        "schema": OUTER_SUMMARY_SCHEMA,
        "experiment": EXPERIMENT,
        "outer_family": outer_family,
        "primary": {
            **_mean_metric_rows(primary),
            "unique_center_combined_coverage": coverage,
        },
        "parent_control": _mean_metric_rows(parent),
        "negative_ecdf_control_not_template_success": _mean_metric_rows(control),
        "direct_rank_mean_top5_not_template_success": _mean_metric_rows(direct),
        "primary_combined_valid_unique_centers_secondary": _mean_metric_rows(unique),
    }


def _fresh_inner_selection(
    plan: Plan,
    outer_family: str,
    inner_paths: Mapping[str, tuple[Path, str]],
    *,
    claimed_primary: CandidateSpec,
    claimed_primary_summary: Mapping[str, Any],
    claimed_control: ControlSpec,
    claimed_control_summary: Mapping[str, Any],
) -> tuple[CandidateSpec, dict[str, Any], ControlSpec, dict[str, Any]]:
    rows = _parse_inner_metric_csv(
        inner_paths["inner_group_metrics"][0],
        expected_sha256=inner_paths["inner_group_metrics"][1],
        plan=plan,
        outer_family=outer_family,
    )
    summaries, primary, primary_summary, control, control_summary = (
        aggregate_and_select_inner(plan, rows)
    )
    _authenticate_summary_csv(
        inner_paths["inner_candidate_summary"][0],
        expected_sha256=inner_paths["inner_candidate_summary"][1],
        expected=summaries,
    )
    audit = source_runner._read_self_hashed_json(
        inner_paths["inner_fit_audits"][0],
        expected_file_sha256=inner_paths["inner_fit_audits"][1],
    )
    _require(
        audit.get("schema") == INNER_AUDIT_SCHEMA
        and audit.get("outer_family") == outer_family
        and audit.get("fmt_features_opened") is False
        and audit.get("raw_features_opened") is False,
        "inner fit audit fresh replay drifted",
    )
    _require(
        primary == claimed_primary
        and primary_summary == _json_safe(claimed_primary_summary)
        and control == claimed_control
        and control_summary == _json_safe(claimed_control_summary),
        "selected primary/control differs from fresh inner replay",
    )
    return primary, primary_summary, control, control_summary


def _rebuild_and_compare_final_models(
    plan: Plan,
    nonouter_caches: Sequence[RankCacheProjection],
    library: VerifiedLibrary,
    primary: CandidateSpec,
    control: ControlSpec,
) -> None:
    primary_model, primary_audits = fit_rank_model(
        nonouter_caches,
        weight=primary.weight,
        bin_count=primary.bin_count,
        beta=primary.beta,
    )
    control_arrays, control_audits = fit_negative_control(
        nonouter_caches, weight=control.weight
    )
    exported = primary_model.export_arrays()
    for name in MODEL_ARRAY_NAMES | CALIBRATION_ARRAY_NAMES:
        _require(
            canonical_array_sha256(exported[name])
            == canonical_array_sha256(library.arrays[name]),
            f"final primary model fresh replay drifted: {name}",
        )
    for name, values in control_arrays.items():
        _require(
            canonical_array_sha256(values)
            == canonical_array_sha256(library.arrays[name]),
            f"final negative control fresh replay drifted: {name}",
        )
    _require(
        _json_safe(library.manifests["model"].get("fit_source_audits"))
        == _json_safe(primary_audits)
        and _json_safe(
            library.manifests["calibration"].get("fit_source_audits")
        )
        == _json_safe(primary_audits)
        and _json_safe(library.manifests["control"].get("fit_source_audits"))
        == _json_safe(control_audits),
        "final fit-source audit fresh replay drifted",
    )


def _fresh_replay_before_reference(
    plan: Plan,
    destination: Path,
    *,
    outer_family: str,
    primary: CandidateSpec,
    primary_summary: Mapping[str, Any],
    control: ControlSpec,
    control_summary: Mapping[str, Any],
    inner_paths: Mapping[str, tuple[Path, str]],
    git_commit: str,
    final_manifest_sha256: Mapping[str, str],
    selection_file_sha256: str,
    rank_binding_file_sha256: str,
    prediction_manifest_file_sha256: str,
) -> VerifiedPrediction:
    primary, primary_summary, control, control_summary = _fresh_inner_selection(
        plan,
        outer_family,
        inner_paths,
        claimed_primary=primary,
        claimed_primary_summary=primary_summary,
        claimed_control=control,
        claimed_control_summary=control_summary,
    )
    fit_families = [family for family in plan.family_order if family != outer_family]
    library = authenticate_final_artifacts(
        destination,
        plan=plan,
        primary=primary,
        control=control,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_sha256=final_manifest_sha256,
    )
    selection = authenticate_selected_candidate(
        destination / "selected_candidate.json",
        plan=plan,
        primary=primary,
        primary_summary=primary_summary,
        control=control,
        control_summary=control_summary,
        library=library,
        inner_paths=inner_paths,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=selection_file_sha256,
    )
    cache_rows, _identity = load_cache_rows(plan.source_plan.parent_plan)
    nonouter_caches = [
        load_rank_cache_projection(plan, row, include_labels=True)
        for row in cache_rows
        if row.family != outer_family
    ]
    _rebuild_and_compare_final_models(
        plan, nonouter_caches, library, primary, control
    )
    del nonouter_caches
    gc.collect()
    outer_caches = [
        load_rank_cache_projection(plan, row, include_labels=False)
        for row in cache_rows
        if row.family == outer_family
    ]
    binding = authenticate_outer_rank_binding(
        destination / "outer_rank_binding.json",
        outer_caches,
        plan=plan,
        primary=primary,
        control=control,
        library=library,
        selection=selection,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=rank_binding_file_sha256,
    )
    return authenticate_outer_prediction(
        destination / "outer_predictions.npz",
        destination / "outer_prediction_manifest.json",
        plan=plan,
        primary=primary,
        control=control,
        library=library,
        selection=selection,
        binding=binding,
        outer_caches=outer_caches,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_manifest_sha256=prediction_manifest_file_sha256,
    )


def run(
    config_path: str | Path,
    outer_family: str,
    output_dir: str | Path,
    *,
    device: str,
    parent_binding_path: str | Path,
    parent_binding_file_sha256: str,
    binding_completion_path: str | Path,
    binding_completion_file_sha256: str,
    expected_config_sha256: str | None = EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    plan = load_plan(config_path)
    _require(outer_family in plan.family_order, f"unknown outer family: {outer_family}")
    if expected_config_sha256 is not None:
        _require(plan.sha256 == expected_config_sha256, "expected config SHA-256 drifted")
    git_commit, dirty = _git_identity()
    _require(not dirty, "Ibex numerical run requires a clean committed worktree")
    plan = bind_parent_sidecar_release(
        plan,
        parent_binding_path=parent_binding_path,
        parent_binding_file_sha256=parent_binding_file_sha256,
        binding_completion_path=binding_completion_path,
        binding_completion_file_sha256=binding_completion_file_sha256,
    )
    _configure_execution(device)
    destination = Path(output_dir).resolve()
    _require(not destination.exists(), f"immutable output directory exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    print(
        f"[{_utc_now()}] {EXPERIMENT} outer={outer_family} commit={git_commit}",
        flush=True,
    )
    cache_rows, input_manifest_identity = load_cache_rows(
        plan.source_plan.parent_plan
    )
    nonouter_rows = [row for row in cache_rows if row.family != outer_family]
    outer_rows = [row for row in cache_rows if row.family == outer_family]
    _require(nonouter_rows and outer_rows, "outer split produced an empty side")
    # At this phase outer sidecars were authenticated only as whole files by
    # the global parent binding.  No outer NPZ member has been opened.
    nonouter_caches = [
        load_rank_cache_projection(plan, row, include_labels=True)
        for row in nonouter_rows
    ]
    inner_rows, inner_audits = _inner_metric_rows(
        plan, nonouter_caches, outer_family
    )
    (
        primary,
        primary_summary,
        control,
        control_summary,
        inner_paths,
    ) = persist_and_authenticate_inner_selection(
        destination,
        inner_rows,
        inner_audits,
        plan=plan,
        outer_family=outer_family,
    )
    fit_families = [family for family in plan.family_order if family != outer_family]
    primary_model, primary_fit_audits = fit_rank_model(
        nonouter_caches,
        weight=primary.weight,
        bin_count=primary.bin_count,
        beta=primary.beta,
    )
    control_arrays, control_fit_audits = fit_negative_control(
        nonouter_caches, weight=control.weight
    )
    final_paths = write_final_artifacts(
        destination,
        primary_model,
        control_arrays,
        primary_fit_audits,
        control_fit_audits,
        plan=plan,
        primary=primary,
        control=control,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
    )
    final_manifest_sha256 = {
        key: value[2] for key, value in final_paths.items()
    }
    library = authenticate_final_artifacts(
        destination,
        plan=plan,
        primary=primary,
        control=control,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_sha256=final_manifest_sha256,
    )
    _rebuild_and_compare_final_models(
        plan, nonouter_caches, library, primary, control
    )
    del primary_model, control_arrays, nonouter_caches
    gc.collect()
    selected_path, selected_sha = write_selected_candidate(
        destination,
        plan=plan,
        primary=primary,
        primary_summary=primary_summary,
        control=control,
        control_summary=control_summary,
        library=library,
        inner_paths=inner_paths,
        outer_family=outer_family,
        git_commit=git_commit,
    )
    selection = authenticate_selected_candidate(
        selected_path,
        plan=plan,
        primary=primary,
        primary_summary=primary_summary,
        control=control,
        control_summary=control_summary,
        library=library,
        inner_paths=inner_paths,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=selected_sha,
    )

    # The outer sidecar rank members are opened only after both independently
    # selected arms and every final model artifact are closed/authenticated.
    outer_caches = [
        load_rank_cache_projection(plan, row, include_labels=False)
        for row in outer_rows
    ]
    binding_path, binding_sha = write_outer_rank_binding(
        destination,
        outer_caches,
        plan=plan,
        primary=primary,
        control=control,
        library=library,
        selection=selection,
        outer_family=outer_family,
        git_commit=git_commit,
    )
    binding = authenticate_outer_rank_binding(
        binding_path,
        outer_caches,
        plan=plan,
        primary=primary,
        control=control,
        library=library,
        selection=selection,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=binding_sha,
    )
    prediction_arrays, prediction_audits = build_outer_prediction_arrays(
        outer_caches, library, primary, control, plan, binding
    )
    prediction_path, prediction_manifest_path, prediction_manifest_sha = (
        write_outer_prediction(
            destination,
            prediction_arrays,
            prediction_audits,
            plan=plan,
            primary=primary,
            control=control,
            library=library,
            selection=selection,
            binding=binding,
            outer_family=outer_family,
            git_commit=git_commit,
        )
    )
    del prediction_arrays, outer_caches
    gc.collect()
    verified = _fresh_replay_before_reference(
        plan,
        destination,
        outer_family=outer_family,
        primary=primary,
        primary_summary=primary_summary,
        control=control,
        control_summary=control_summary,
        inner_paths=inner_paths,
        git_commit=git_commit,
        final_manifest_sha256=final_manifest_sha256,
        selection_file_sha256=selected_sha,
        rank_binding_file_sha256=binding_sha,
        prediction_manifest_file_sha256=prediction_manifest_sha,
    )

    # Parent prediction bytes and new outer labels are both opened only after
    # the new prediction has passed its complete numerical fresh replay.
    parent_control = authenticate_parent_control(plan, outer_family)
    metric_rows, reference_audits = evaluate_outer_prediction(
        plan,
        primary,
        control,
        verified,
        parent_control,
        outer_family=outer_family,
    )
    outer_metrics_path = destination / "outer_group_metrics.csv"
    outer_metrics_sha = source_runner.early._atomic_csv(
        outer_metrics_path, OUTER_METRIC_FIELDS, metric_rows
    )
    summary_value = _manifest(outer_summary(metric_rows, outer_family))
    outer_summary_path = destination / "outer_summary.json"
    outer_summary_sha = source_runner.early._atomic_json(
        outer_summary_path, summary_value
    )
    reference_value = _manifest(
        {
            "schema": REFERENCE_AUDIT_SCHEMA,
            "experiment": EXPERIMENT,
            "outer_family": outer_family,
            "first_open_phase": "after_new_prediction_file_manifest_and_complete_fresh_replay",
            "prediction_manifest_file_sha256": verified.manifest_file_sha256,
            "prediction_file_sha256": verified.artifact_file_sha256,
            "parent_control": dict(parent_control.evidence),
            "row_count": len(reference_audits),
            "rows": reference_audits,
            "reference_labels_all_opened": False,
            "fmt_features_opened": False,
            "raw_features_opened": False,
        }
    )
    reference_path = destination / "outer_reference_access_audit.json"
    reference_sha = source_runner.early._atomic_json(
        reference_path, reference_value
    )
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
                plan, fit_families=fit_families
            ),
            "outer_family": outer_family,
            "primary_candidate": _candidate_payload(primary),
            "selected_control": _control_payload(control),
            "selected_candidate_file_sha256": selected_sha,
            "final_artifact_manifest_file_sha256": final_manifest_sha256,
            "outer_rank_binding_file_sha256": binding_sha,
            "prediction_manifest_file_sha256": prediction_manifest_sha,
            "prediction_file_sha256": verified.artifact_file_sha256,
            "parent_control": dict(parent_control.evidence),
            "inner_group_metrics_file_sha256": inner_paths[
                "inner_group_metrics"
            ][1],
            "inner_candidate_summary_file_sha256": inner_paths[
                "inner_candidate_summary"
            ][1],
            "inner_fit_audits_file_sha256": inner_paths["inner_fit_audits"][1],
            "outer_group_metrics_file_sha256": outer_metrics_sha,
            "outer_summary_file_sha256": outer_summary_sha,
            "outer_reference_access_audit_file_sha256": reference_sha,
            "environment": source_runner.early._environment_audit(device),
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
    result_sha = source_runner.early._atomic_json(result_path, result)
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
    source_runner.early._atomic_json(destination / "RUN_COMPLETE.json", completion)
    _require(
        {path.name for path in destination.iterdir()}
        == set(plan.required_fold_files),
        "completed fold file set drifted",
    )
    print(
        f"[{_utc_now()}] completed outer={outer_family} "
        f"primary_valid_row_F1={float(summary_value['primary']['f1']):.6f}",
        flush=True,
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "config"
            / "Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml"
        ),
    )
    parser.add_argument("--outer-family", required=True, choices=FAMILY_ORDER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parent-binding", required=True)
    parser.add_argument("--parent-binding-sha256", required=True)
    parser.add_argument("--binding-completion", required=True)
    parser.add_argument("--binding-completion-sha256", required=True)
    parser.add_argument("--expected-config-sha256", default=EXPECTED_CONFIG_SHA256)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    run(
        arguments.config,
        arguments.outer_family,
        arguments.output_dir,
        device=arguments.device,
        parent_binding_path=arguments.parent_binding,
        parent_binding_file_sha256=arguments.parent_binding_sha256,
        binding_completion_path=arguments.binding_completion,
        binding_completion_file_sha256=arguments.binding_completion_sha256,
        expected_config_sha256=arguments.expected_config_sha256,
    )


if __name__ == "__main__":
    main()
